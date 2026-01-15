"""Streamlit frontend for BBO GIB Bidding Playground.

All heavy work is delegated to the FastAPI service defined in
`bbo_bidding_queries_api.py`.

Run the server first (takes about 8-10 minutes to complete):

    python bbo_bidding_queries_api.py

Then run this app (may wait for the server to finish loading data):

    streamlit run bbo_bidding_queries_streamlit.py
"""

from __future__ import annotations


import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode
from st_aggrid.shared import GridUpdateMode, DataReturnMode
import polars as pl
import duckdb  # type: ignore[import-not-found]
import pandas as pd
import time
import requests
import base64
import numpy as np
from datetime import datetime, timezone
import importlib.metadata as importlib_metadata
import pathlib
import os
import sys
import re
from typing import Any, Dict, List, Set

from bbo_bidding_queries_lib import (
    normalize_auction_input,
    normalize_auction_user_text,
    format_elapsed,
)

# Import criteria evaluation helpers from handlers
from plugins.bbo_handlers_common import (
    evaluate_sl_criterion,
    annotate_criterion_with_value,
)


API_BASE = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Par Contract formatting (copied from plugins.bbo_handlers_common to avoid
# import path issues with mlBridge dependencies)
# ---------------------------------------------------------------------------

def _to_python_list(x: Any) -> list:
    """Convert Polars list/Series or Python list to Python list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # Handle Polars Series or list types
    if hasattr(x, "to_list"):
        try:
            return x.to_list()
        except Exception:
            pass
    # Fallback: try to iterate
    try:
        return list(x)
    except Exception:
        return []


def _par_contract_signature(c: Any) -> str:
    """Stable signature for a par-contract dict (used for de-duping)."""
    # Handle Polars struct which may not be a dict
    if hasattr(c, "get"):
        get_fn = c.get
    elif isinstance(c, dict):
        get_fn = c.get
    else:
        # Try to convert to dict
        try:
            c = dict(c) if hasattr(c, "__iter__") else {"_raw": c}
            get_fn = c.get
        except Exception:
            return str(c)
    
    level = get_fn("Level", "")
    strain = get_fn("Strain", "")
    dbl = get_fn("Doubled", "")
    if dbl == "":
        dbl = get_fn("Double", "")
    pair_dir = get_fn("Pair_Direction", "")
    result = get_fn("Result", "")
    return f"{level}|{strain}|{dbl}|{pair_dir}|{result}"


def _dedup_par_contracts(par_contracts: Any) -> list[dict]:
    """Return de-duplicated par contracts (preserving first-seen order)."""
    # Convert to Python list first
    contracts_list = _to_python_list(par_contracts)
    if not contracts_list:
        return []
    
    seen: set[str] = set()
    out: list[dict] = []
    for c in contracts_list:
        # Convert struct to dict if needed
        if not isinstance(c, dict) and hasattr(c, "__iter__"):
            try:
                c = dict(c)
            except Exception:
                continue
        if not isinstance(c, dict):
            continue
        sig = _par_contract_signature(c)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def _format_par_contracts(par_contracts: Any) -> str | None:
    """Format ParContracts into a readable string, de-duped and with correct 'Doubled' key."""
    if par_contracts is None:
        return None
    
    # If already a string, return as-is
    if isinstance(par_contracts, str):
        return par_contracts
    
    # Convert and dedup
    contracts = _dedup_par_contracts(par_contracts)
    if not contracts:
        # Fallback to string representation if we couldn't parse
        return str(par_contracts) if par_contracts else None
    
    formatted: list[str] = []
    for c in contracts:
        level = c.get("Level", "")
        strain = c.get("Strain", "")
        dbl = c.get("Doubled", "")
        if dbl == "":
            dbl = c.get("Double", "")
        pair_dir = c.get("Pair_Direction", "")
        result = c.get("Result", "")
        contract_str = f"{level}{strain}{dbl}"
        if pair_dir:
            contract_str += f" {pair_dir}"
        if result is not None and result != "":
            if isinstance(result, int):
                if result == 0:
                    contract_str += " ="
                elif result > 0:
                    contract_str += f" +{result}"
                else:
                    contract_str += f" {result}"
            else:
                contract_str += f" {result}"
        formatted.append(contract_str.strip())
    return ", ".join(formatted)

# Shared constants
SEAT_ROLES = {1: "Opener/Dealer", 2: "LHO", 3: "Partner", 4: "RHO"}
MATCH_ALL_DEALERS_HELP = (
    "When checked, your pattern matches the auction regardless of which seat deals first (expands to 4× matches). "
    "For example, '1N-p-3N' finds this auction whether North, East, South, or West is dealer."
)


def prepend_all_seats_prefix(pattern: str) -> str:
    """Prepend (p-)* to a pattern for matching all 4 dealer positions."""
    if pattern.startswith("^"):
        return "^(p-)*" + pattern[1:]
    return "(p-)*" + pattern


def calc_grid_height(n_rows: int, max_height: int = 400, row_height: int = 35, header_height: int = 50) -> int:
    """Calculate appropriate grid height based on row count."""
    return min(max_height, header_height + n_rows * row_height)


def order_columns(df: pl.DataFrame, priority_cols: list[str], drop_cols: list[str] | None = None) -> pl.DataFrame:
    """Reorder DataFrame columns: priority columns first, then remaining alphabetically.
    
    Args:
        df: The DataFrame to reorder
        priority_cols: Columns to put first (in this order). Missing columns are skipped.
        drop_cols: Columns to drop from output. Optional.
    
    Returns:
        DataFrame with reordered columns
    """
    available = [c for c in priority_cols if c in df.columns]
    other = sorted([c for c in df.columns if c not in priority_cols])
    df = df.select(available + other)
    if drop_cols:
        to_drop = [c for c in drop_cols if c in df.columns]
        if to_drop:
            df = df.drop(to_drop)
    return df


def _to_list_utf8_cell(x: Any) -> list[str] | None:
    """Normalize a cell value into list[str] (or None) for strongly-typed List(Utf8) Series.

    We see Agg_Expr_Seat_* cells as lists, None, and occasionally other list-like wrappers.
    """
    if x is None:
        return None
    if isinstance(x, pl.Series):
        x = x.to_list()
    elif hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        try:
            x = x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        out = ["" if v is None else str(v) for v in x]
        out = [s for s in out if s != ""]
        return out or None
    if isinstance(x, (str, bytes)):
        s = x.decode() if isinstance(x, bytes) else x
        s = s.strip()
        return [s] if s else None
    s = str(x).strip()
    return [s] if s else None


def _expr_to_criteria_list(expr: Any) -> list[str]:
    """Best-effort split of a BT `Expr` cell into atomic criterion strings.

    Notes:
    - BT rows may have criteria in `Agg_Expr_Seat_*` (preferred) OR only in `Expr`.
    - This splitter is intentionally conservative; unknown tokens will be passed through and
      the server will classify them as failed/untracked as appropriate.
    """
    if expr is None:
        return []

    # If Expr is already a list (common), handle it directly.
    # Important: an empty list should mean "no criteria", not the literal string "[]".
    if isinstance(expr, (list, tuple)):
        if len(expr) == 0:
            return []

    # Most commonly Expr is already a list[str] (bt_seat1 schema uses List(String)).
    direct = _to_list_utf8_cell(expr)
    if direct is not None and not (len(direct) == 1 and ("&" in direct[0] or " and " in direct[0].lower())):
        return direct

    # Normalize common boolean separators to '&', then split.
    # Examples seen: "A & B", "A and B", "A && B"
    s = str(expr).strip()
    if not s:
        return []
    if s in ("[]", "[ ]"):
        return []
    s = re.sub(r"(?i)\s+and\s+", " & ", s)
    s = s.replace("&&", "&")
    parts = re.split(r"\s*&\s*|\s*;\s*|\s*,\s*|\n+", s)
    out = []
    for p in parts:
        p2 = str(p).strip()
        if not p2:
            continue
        # Drop outer parentheses to better match bitmap column names.
        while p2.startswith("(") and p2.endswith(")") and len(p2) >= 2:
            p2 = p2[1:-1].strip()
        if p2:
            out.append(p2)
    return out


def _to_scalar_auction_text(x: Any) -> str | None:
    """Best-effort: turn a grid cell into a single auction string (or None).

    In Bidding Arena we sometimes display a list of matching Rules auctions in the Auction column.
    Drilldowns must not treat that list as a regex pattern.
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        # Common case: stringified Python list from AgGrid selection like "['1S-p-p-p', '1S-4N-p-p-p']"
        if s.startswith("[") and s.endswith("]") and ("'" in s or "\"" in s):
            return None
        return s or None
    if isinstance(x, (list, tuple)):
        # Ambiguous: caller should use bt_index list or Auction_Rules_Selected instead.
        return None
    try:
        s = str(x).strip()
        return s or None
    except Exception:
        return None


def parse_distribution_pattern(pattern: str) -> dict | None:
    """
    Parse a suit distribution pattern into filter criteria.
    
    Supports multiple notations (S-H-D-C order):
        - Dash-separated: '4-3-3-3', '[2-4]-[2-3]-5-4', '4-3-.*'
        - Compact numeric: '4333', '5332' (exact lengths only, ordered S-H-D-C)
        - X for any: '4-3-x-x' or '4-3-X-X'
        - Plus for "or more": '5+-3-3-2'
        - Minus for "or fewer": '3--4-4-2'
        - Range with colon: '4:6-3-3-1:3' (4-6 spades, 1-3 clubs)
    
    Returns:
        dict with keys 'S', 'H', 'D', 'C', each containing (min, max) tuple or None
        Returns None if pattern is empty or invalid.
    
    Examples:
        '4-3-3-3' → {'S': (4,4), 'H': (3,3), 'D': (3,3), 'C': (3,3)}
        '5+-3-3-2' → {'S': (5,13), 'H': (3,3), 'D': (3,3), 'C': (2,2)}
        '4:6-3-x-x' → {'S': (4,6), 'H': (3,3), 'D': None, 'C': None}
        '4333' → {'S': (4,4), 'H': (3,3), 'D': (3,3), 'C': (3,3)}
    """
    import re
    
    if not pattern or not pattern.strip():
        return None
    
    pattern = pattern.strip()
    suits = ['S', 'H', 'D', 'C']
    result: dict[str, tuple[int, int] | None] = {s: None for s in suits}
    
    # Try compact numeric format first (e.g., '4333', '5332')
    if re.match(r'^[0-9]{4}$', pattern):
        for i, suit in enumerate(suits):
            val = int(pattern[i])
            result[suit] = (val, val)
        return result
    
    # Split by dash for other formats
    parts = pattern.split('-')
    if len(parts) != 4:
        return None  # Invalid format
    
    for i, (part, suit) in enumerate(zip(parts, suits)):
        part = part.strip()
        
        # Wildcard: x, X, .*, .+, or empty
        if part.lower() == 'x' or part in ('.*', '.+', '*', ''):
            result[suit] = None  # No constraint
            continue
        
        # Range bracket notation: [2-4] or [2:4]
        bracket_match = re.match(r'^\[(\d+)[-:](\d+)\]$', part)
        if bracket_match:
            result[suit] = (int(bracket_match.group(1)), int(bracket_match.group(2)))
            continue
        
        # Range colon notation: 2:4
        colon_match = re.match(r'^(\d+):(\d+)$', part)
        if colon_match:
            result[suit] = (int(colon_match.group(1)), int(colon_match.group(2)))
            continue
        
        # Plus notation: 5+ (5 or more)
        plus_match = re.match(r'^(\d+)\+$', part)
        if plus_match:
            result[suit] = (int(plus_match.group(1)), 13)
            continue
        
        # Minus notation: 3- (3 or fewer)
        minus_match = re.match(r'^(\d+)-$', part)
        if minus_match:
            result[suit] = (0, int(minus_match.group(1)))
            continue
        
        # Exact number: 4
        if re.match(r'^\d+$', part):
            val = int(part)
            result[suit] = (val, val)
            continue
        
        # Unknown format for this part - treat as no constraint
        result[suit] = None
    
    return result


def parse_sorted_shape(pattern: str) -> list[int] | None:
    """
    Parse a sorted shape pattern (e.g., '5431', '4432', '5332').
    
    Sorted shapes match any suit arrangement. E.g., '5431' matches:
    - 5S-4H-3D-1C, 5S-4D-3H-1C, 5H-4S-3D-1C, etc.
    
    Returns:
        List of 4 integers sorted descending, or None if invalid.
    
    Examples:
        '5431' → [5, 4, 3, 1]
        '4-4-3-2' → [4, 4, 3, 2]
    """
    import re
    
    if not pattern or not pattern.strip():
        return None
    
    pattern = pattern.strip()
    
    # Try compact format: '5431'
    if re.match(r'^[0-9]{4}$', pattern):
        lengths = [int(c) for c in pattern]
        if sum(lengths) == 13:  # Valid hand
            return sorted(lengths, reverse=True)
        return None
    
    # Try dash-separated: '5-4-3-1'
    parts = pattern.split('-')
    if len(parts) == 4:
        try:
            lengths = [int(p.strip()) for p in parts]
            if sum(lengths) == 13:
                return sorted(lengths, reverse=True)
        except ValueError:
            pass
    
    return None


def build_distribution_sql(
    dist_pattern: str | None,
    sorted_shape: str | None,
    seat: int,
    available_columns: list[str]
) -> tuple[str, str]:
    """
    Build SQL WHERE clause for distribution filtering.
    
    Args:
        dist_pattern: Ordered distribution pattern (S-H-D-C order)
        sorted_shape: Sorted shape pattern (any suit order)
        seat: Seat number (1-4)
        available_columns: List of available column names in the table
    
    Returns:
        Tuple of (where_clause, description)
        where_clause is empty string if no filter, otherwise "WHERE ..."
    """
    conditions = []
    descriptions = []
    suits = ['S', 'H', 'D', 'C']
    
    # Handle ordered distribution pattern
    if dist_pattern:
        parsed = parse_distribution_pattern(dist_pattern)
        if parsed:
            for suit, constraint in parsed.items():
                if constraint is None:
                    continue
                min_val, max_val = constraint
                min_col = f"SL_{suit}_min_S{seat}"
                max_col = f"SL_{suit}_max_S{seat}"
                
                if min_col in available_columns and max_col in available_columns:
                    # Overlap condition: row_min <= max_val AND row_max >= min_val
                    conditions.append(f'"{min_col}" <= {max_val}')
                    conditions.append(f'"{max_col}" >= {min_val}')
                    if min_val == max_val:
                        descriptions.append(f"{suit}={min_val}")
                    else:
                        descriptions.append(f"{suit}∈[{min_val},{max_val}]")
    
    # Handle sorted shape pattern
    if sorted_shape:
        shape = parse_sorted_shape(sorted_shape)
        if shape:
            # For sorted shape, we need to check if the sorted suit lengths match
            # This requires comparing sorted values of the 4 suit length columns
            # Using a subquery or CASE expressions to sort the values
            
            # Build list of suit length column references (using min or max based on what's available)
            suit_cols = []
            for suit in suits:
                # Prefer using min column for filtering (more restrictive)
                min_col = f"SL_{suit}_min_S{seat}"
                max_col = f"SL_{suit}_max_S{seat}"
                if min_col in available_columns:
                    suit_cols.append(f'"{min_col}"')
                elif max_col in available_columns:
                    suit_cols.append(f'"{max_col}"')
            
            if len(suit_cols) == 4:
                # Create a condition that checks if sorted lengths match the shape
                # We use GREATEST/LEAST combinations to check each position
                # sorted_shape is [largest, 2nd, 3rd, smallest]
                
                # Generate all permutations to check if ANY arrangement matches
                from itertools import permutations
                perm_conditions = []
                
                # Get unique permutations of the shape
                unique_perms = set(permutations(shape))
                
                for perm in unique_perms:
                    # perm is (S_len, H_len, D_len, C_len)
                    perm_parts = []
                    for i, (suit, expected_len) in enumerate(zip(suits, perm)):
                        min_col = f"SL_{suit}_min_S{seat}"
                        max_col = f"SL_{suit}_max_S{seat}"
                        if min_col in available_columns and max_col in available_columns:
                            # Range overlap: expected_len falls within [min, max]
                            perm_parts.append(
                                f'("{min_col}" <= {expected_len} AND "{max_col}" >= {expected_len})'
                            )
                    if len(perm_parts) == 4:
                        perm_conditions.append(f"({' AND '.join(perm_parts)})")
                
                if perm_conditions:
                    sorted_condition = f"({' OR '.join(perm_conditions)})"
                    conditions.append(sorted_condition)
                    descriptions.append(f"shape={''.join(map(str, shape))}")
    
    if not conditions:
        return "", ""
    
    where_clause = "WHERE " + " AND ".join(conditions)
    description = ", ".join(descriptions)
    return where_clause, description


def filter_by_distribution_duckdb(
    df: "pl.DataFrame",
    dist_pattern: str | None,
    sorted_shape: str | None,
    seat: int
) -> tuple["pl.DataFrame", str]:
    """
    Filter a DataFrame by distribution pattern using DuckDB.
    
    Args:
        df: DataFrame with SL_S_min_S{seat}, SL_S_max_S{seat}, etc. columns
        dist_pattern: Ordered distribution pattern (S-H-D-C)
        sorted_shape: Sorted shape pattern (any suit order)
        seat: Seat number (1-4)
    
    Returns:
        Tuple of (filtered_df, sql_query_used)
    """
    if df.is_empty():
        return df, ""
    
    # Build the WHERE clause
    where_clause, _ = build_distribution_sql(
        dist_pattern, sorted_shape, seat, df.columns
    )
    
    if not where_clause:
        return df, "SELECT * FROM df"
    
    # Build the full query
    sql_query = f"SELECT * FROM df {where_clause}"
    
    # Execute with DuckDB
    try:
        result = duckdb.sql(sql_query).pl()
        # duckdb's `.pl()` can be typed as DataFrame | LazyFrame; force eager for downstream display.
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        return result, sql_query
    except Exception as e:
        # Return original df if query fails
        return df, f"-- Error: {e}\n{sql_query}"


def format_distribution_help() -> str:
    """Return help text for distribution pattern input."""
    return """**Ordered Distribution** (S-H-D-C order)

**Notations:**
- `4-3-3-3` — exact: 4S, 3H, 3D, 3C
- `4333` — compact: same as above
- `5+-3-3-2` — 5+ spades
- `3--4-4-2` — 3 or fewer spades  
- `[2-4]-3-5-3` — 2-4 spades (range)
- `2:4-3-5-3` — 2-4 spades (range)
- `5-4-x-x` — any D/C

**Sorted Shape** (any suit order)
- `5431` — matches 5-4-3-1 in ANY suits
- `4432` — matches 4-4-3-2 in ANY suits
- `5332` — balanced with 5-card suit
"""


def parse_hand_suit_lengths(hand_str: str) -> dict[str, int] | None:
    """
    Parse a PBN hand string and return suit lengths.
    
    PBN format: "AKQ2.J54.T98.765" (Spades.Hearts.Diamonds.Clubs)
    
    Returns dict like {'S': 4, 'H': 3, 'D': 3, 'C': 3} or None if invalid.
    """
    if not hand_str or not isinstance(hand_str, str):
        return None
    
    parts = hand_str.split('.')
    if len(parts) != 4:
        return None
    
    suits = ['S', 'H', 'D', 'C']
    return {suit: len(part) for suit, part in zip(suits, parts)}


def add_suit_length_columns(df: "pl.DataFrame", direction: str) -> "pl.DataFrame":
    """
    Add suit length columns for a specific direction's hand.
    
    Args:
        df: DataFrame with Hand_{direction} column
        direction: 'N', 'E', 'S', or 'W'
    
    Returns:
        DataFrame with added SL_S_{dir}, SL_H_{dir}, SL_D_{dir}, SL_C_{dir} columns
    """
    hand_col = f"Hand_{direction}"
    if hand_col not in df.columns:
        return df
    
    suits = ['S', 'H', 'D', 'C']
    
    # Compute suit lengths for each row
    for suit_idx, suit in enumerate(suits):
        col_name = f"SL_{suit}_{direction}"
        
        # Extract suit length using string split
        # Hand format: "AKQ2.J54.T98.765" -> split by '.' gives [spades, hearts, diamonds, clubs]
        df = df.with_columns(
            pl.col(hand_col).str.split('.').list.get(suit_idx).str.len_chars().alias(col_name)
        )
    
    return df


def build_deal_distribution_sql(
    dist_pattern: str | None,
    sorted_shape: str | None,
    direction: str
) -> str:
    """
    Build SQL WHERE clause for filtering deals by hand distribution.
    
    Args:
        dist_pattern: Ordered distribution pattern (S-H-D-C)
        sorted_shape: Sorted shape pattern (any suit order)
        direction: 'N', 'E', 'S', or 'W'
    
    Returns:
        WHERE clause string (empty if no filter)
    """
    from itertools import permutations
    
    conditions = []
    suits = ['S', 'H', 'D', 'C']
    
    # Handle ordered distribution pattern
    if dist_pattern:
        parsed = parse_distribution_pattern(dist_pattern)
        if parsed:
            for suit, constraint in parsed.items():
                if constraint is None:
                    continue
                min_val, max_val = constraint
                col = f"SL_{suit}_{direction}"
                
                if min_val == max_val:
                    conditions.append(f'"{col}" = {min_val}')
                else:
                    conditions.append(f'"{col}" >= {min_val} AND "{col}" <= {max_val}')
    
    # Handle sorted shape pattern
    if sorted_shape:
        shape = parse_sorted_shape(sorted_shape)
        if shape:
            perm_conditions = []
            unique_perms = set(permutations(shape))
            
            for perm in unique_perms:
                perm_parts = []
                for suit, expected_len in zip(suits, perm):
                    col = f"SL_{suit}_{direction}"
                    perm_parts.append(f'"{col}" = {expected_len}')
                perm_conditions.append(f"({' AND '.join(perm_parts)})")
            
            if perm_conditions:
                conditions.append(f"({' OR '.join(perm_conditions)})")
    
    if not conditions:
        return ""
    
    return " AND ".join(conditions)


def filter_deals_by_distribution_duckdb(
    df: "pl.DataFrame",
    dist_pattern: str | None,
    sorted_shape: str | None,
    direction: str
) -> tuple["pl.DataFrame", str]:
    """
    Filter a deals DataFrame by hand distribution using DuckDB.
    
    Args:
        df: DataFrame with Hand_{direction} column
        dist_pattern: Ordered distribution pattern (S-H-D-C)
        sorted_shape: Sorted shape pattern (any suit order)
        direction: 'N', 'E', 'S', or 'W'
    
    Returns:
        Tuple of (filtered_df, sql_query_used)
    """
    if df.is_empty():
        return df, ""
    
    # Add suit length columns if not present
    hand_col = f"Hand_{direction}"
    if hand_col not in df.columns:
        return df, f"-- No {hand_col} column"
    
    df_with_sl = add_suit_length_columns(df, direction)
    
    # Build WHERE clause
    where_clause = build_deal_distribution_sql(dist_pattern, sorted_shape, direction)
    
    if not where_clause:
        return df, "SELECT * FROM df"
    
    sql_query = f"SELECT * FROM df_with_sl WHERE {where_clause}"
    
    try:
        result = duckdb.sql(sql_query).pl()
        # duckdb's `.pl()` can be typed as DataFrame | LazyFrame; force eager for downstream display.
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        # Remove the added SL columns for cleaner display
        sl_cols = [f"SL_{s}_{direction}" for s in ['S', 'H', 'D', 'C']]
        result = result.drop([c for c in sl_cols if c in result.columns])
        return result, sql_query
    except Exception as e:
        return df, f"-- Error: {e}\n{sql_query}"


DEFAULT_API_TIMEOUT = 60  # seconds; prevents Streamlit from hanging indefinitely


def api_get(path: str, timeout: int | None = None) -> Dict[str, Any]:
    """GET from API with a default timeout to prevent hanging."""
    t0 = time.perf_counter()
    resp = requests.get(f"{API_BASE}{path}", timeout=timeout or DEFAULT_API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    try:
        if isinstance(data, dict):
            data["_client_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        pass
    return data


def api_post(path: str, payload: Dict[str, Any], timeout: int | None = None) -> Dict[str, Any]:
    """POST to API with a default timeout to prevent hanging."""
    t0 = time.perf_counter()
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout or DEFAULT_API_TIMEOUT)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Surface FastAPI error details in the UI
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"{e}\nServer detail: {detail}", response=resp) from e
    data = resp.json()
    try:
        if isinstance(data, dict):
            data["_client_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        pass
    return data


def _st_info_elapsed(label: str, data: Dict[str, Any] | None) -> None:
    """Show elapsed time (server elapsed_ms preferred, client fallback)."""
    if not isinstance(data, dict):
        return
    elapsed_ms = data.get("elapsed_ms")
    if elapsed_ms is None:
        elapsed_ms = data.get("_client_elapsed_ms")
    if elapsed_ms is None:
        return
    try:
        st.info(f"⏱️ {label} completed in {format_elapsed(float(elapsed_ms))}")
    except Exception:
        pass


def generate_passthrough_sql(df: pl.DataFrame, table_name: str) -> str:
    """Generate a pass-through SQL SELECT statement for the DataFrame columns.
    
    This creates a simple SELECT statement that would retrieve the same columns
    from a DuckDB table. Useful for showing users what SQL equivalent would look like.
    
    Args:
        df: DataFrame whose columns to include in the SELECT
        table_name: Name of the table to select from
        
    Returns:
        SQL SELECT statement string
    """
    if df.is_empty() or not df.columns:
        return f"SELECT * FROM {table_name}"
    
    # Quote column names that might have special characters
    quoted_cols = [f'"{col}"' for col in df.columns]
    cols_str = ",\n    ".join(quoted_cols)
    return f"SELECT\n    {cols_str}\nFROM {table_name}"


def render_aggrid(
    records: Any, 
    key: str, 
    height: int | None = None, 
    table_name: str | None = None,
    update_mode: GridUpdateMode = GridUpdateMode.NO_UPDATE,
    sort_model: list[dict[str, Any]] | None = None,
    hide_cols: list[str] | None = None,
    show_copy_panel: bool = False,
    copy_panel_default_col: str | None = None,
    fit_columns_to_view: bool = True,
    show_sql_expander: bool = True,
) -> list[dict[str, Any]]:
    """Render a list-of-dicts or DataFrame using AgGrid.
    
    Args:
        records: DataFrame or list-of-dicts to display
        key: Unique key for the AgGrid component
        height: Optional height in pixels
        table_name: Optional table name for SQL display (shows SQL expander if provided)
        update_mode: GridUpdateMode (default: NO_UPDATE for performance)
    
    Returns:
        List of selected rows (dicts)
    """
    if records is None:
        st.info("No data.")
        return []
    if isinstance(records, pl.DataFrame):
        df = records
    else:
        try:
            df = pl.DataFrame(records)
        except Exception:
            st.json(records)
            return []
    if df.is_empty():
        st.info("No rows to display.")
        return []

    # ---------------------------------------------------------------------
    # SQL expander (always on by default)
    # ---------------------------------------------------------------------
    # Many tables in this app show a "demo SQL" query. We can make the default
    # pass-through query fully functional by registering the DataFrame into
    # local DuckDB and executing it to drive rendering.
    sql_table = table_name or "df"
    # Ensure it's a safe identifier for DuckDB (avoid spaces, punctuation).
    sql_table = re.sub(r"[^A-Za-z0-9_]", "_", str(sql_table))
    if not sql_table:
        sql_table = "df"
    sql_query = generate_passthrough_sql(df, sql_table)

    if show_sql_expander:
        with st.expander("📝 SQL Query", expanded=False):
            st.code(sql_query, language="sql")

    # Best-effort: execute the SQL locally to make it real and drive the rendered df.
    # If this fails (e.g., unsupported list/struct columns), fall back to the original df.
    try:
        # duckdb is imported at module scope in this file
        local_con = duckdb.connect(":memory:")
        try:
            # Register Pandas to DuckDB, then produce a Polars DataFrame back.
            local_con.register(sql_table, df.to_pandas())
            result = local_con.execute(sql_query).pl()
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            if isinstance(result, pl.DataFrame) and result.height == df.height and result.width == df.width:
                df = result
            elif isinstance(result, pl.DataFrame) and result.height > 0:
                # If the SQL changes shape, still show it (it is user-visible SQL).
                df = result
        finally:
            local_con.close()
    except Exception:
        # Keep df as-is; SQL remains a best-effort representation.
        pass

    # Round float columns to 2 decimal places for cleaner display
    float_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    if float_cols:
        df = df.with_columns([pl.col(c).round(2) for c in float_cols])
    
    def _stringify_ev_list(x: Any) -> str:
        # Handle Polars Series / NumPy arrays / Python lists uniformly to avoid "shape: ..." strings.
        if x is None:
            return ""
        if isinstance(x, pl.Series):
            x = x.to_list()
        elif hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
            # numpy.ndarray, etc.
            try:
                x = x.tolist()
            except Exception:
                pass
        if isinstance(x, (list, tuple)):
            return ", ".join("" if v is None else str(v) for v in x)
        return str(x)

    # AgGrid renders list/array values with commas but no spaces (e.g., "1,2,3").
    # For EV_ParContracts (a list), display as "1, 2, 3" for readability.
    if "EV_ParContracts" in df.columns:
        try:
            df = df.with_columns(
                pl.col("EV_ParContracts").map_elements(_stringify_ev_list, return_dtype=pl.Utf8).alias("EV_ParContracts")
            )
        except Exception:
            # Best-effort: leave as-is if conversion fails
            pass
    
    # For ParContracts (list of structs), format using the same logic as API handlers
    if "ParContracts" in df.columns:
        # Only process if it's not already a string column
        if df["ParContracts"].dtype not in (pl.Utf8, pl.String):
            try:
                df = df.with_columns(
                    pl.col("ParContracts").map_elements(_format_par_contracts, return_dtype=pl.Utf8).alias("ParContracts")
                )
            except Exception:
                # Best-effort: leave as-is if conversion fails
                pass

    # Generic stringification for any remaining list columns (prevents [object Object] in AgGrid).
    # This catches "Failed Exprs", "Expr", "Agg_Expr_Seat", and others.
    for c in df.columns:
        dtype = df[c].dtype
        if isinstance(dtype, pl.List):
            try:
                # Use a robust way to check for string inner type
                is_string_list = False
                try:
                    # In some Polars versions, List.inner is accessible
                    inner = getattr(dtype, "inner", None)
                    if inner in (pl.Utf8, pl.String):
                        is_string_list = True
                except Exception:
                    pass
                
                if is_string_list:
                    df = df.with_columns(pl.col(c).list.join("; ").alias(c))
                else:
                    # Fallback for other list types (int, etc.): stringify
                    df = df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))
            except Exception:
                pass

    # Dynamic height based on explicit row/header heights set below.
    # rowHeight=28, headerHeight=32, plus border/scrollbar buffer.
    if height is None:
        n_rows = len(df)
        ROW_HEIGHT = 28
        HEADER_HEIGHT = 32
        BUFFER = 10  # borders, scrollbar track, etc.
        # Cap at ~10 rows before scrolling kicks in.
        height = min(10 * ROW_HEIGHT + HEADER_HEIGHT + BUFFER,
                     n_rows * ROW_HEIGHT + HEADER_HEIGHT + BUFFER)

    # Build the grid options
    pandas_df = df.to_pandas()
    gb = GridOptionsBuilder.from_dataframe(pandas_df)
    # Disable pagination entirely to allow scrolling within the fixed height
    gb.configure_pagination(enabled=False)
    # Make all columns read-only (not editable), resizable, filterable, sortable.
    gb.configure_default_column(resizable=True, filter=True, sortable=True, editable=False)
    
    # Custom formatting for columns that should display as percentages
    pct_cols = [c for c in df.columns if 
                "Makes %" in c or "make_pct" in c or "Makes_Pct" in c or
                "Rate" in c or "rate" in c or "Percentage" in c or "percentage" in c or "pct" in c or
                "Frequency" in c or "%" in c]
    for col in pct_cols:
        gb.configure_column(col, valueFormatter="x !== null ? x.toFixed(1) + '%' : ''")

    # Enforce 1 decimal place for EV columns (display only; keeps numeric sorting)
    for col in [c for c in df.columns if c in ("EV at Bid", "EV Std")]:
        gb.configure_column(col, valueFormatter="x !== null ? x.toFixed(1) : ''")

    # Optionally hide helper columns (e.g., sort keys)
    if hide_cols:
        for c in hide_cols:
            if c in df.columns:
                gb.configure_column(c, hide=True)
        
    # Enable row selection - clicking anywhere on a row highlights the entire row
    gb.configure_selection(selection_mode="single", use_checkbox=False, suppressRowClickSelection=False)
    # Explicitly set row/header heights to ensure consistent sizing
    # suppressCellFocus prevents cell-level focus (keeps row selection clean) but can interfere with copy UX.
    # Tighten specific columns that tend to be too wide (must be before build())
    # Width is max of target width and column name length (approx 8px per char)
    tight_cols = {
        # Matching Deals columns
        "Score": 40, "ParScore": 40, "Result": 30, "Contract": 30,  # Contract reduced by 25%
        "Dealer": 20, "Vul": 30, "index": 70, "Auction_Actual": 140,  # Auction_Actual reduced by 30%
        "Hand_N": 135, "Hand_E": 135, "Hand_S": 135, "Hand_W": 135,
        # Auction Summary columns
        "Bid Num": 60, "Seat": 55, "Bid": 50, "BT Index": 60, "Deals": 70,
        "Criteria Count": 25, "Complete": 25,
        # Rankings-style stats columns (NV/V split)
        "Matches_NV": 90, "Matches_V": 85,
        "Avg Par_NV": 95, "Avg Par_V": 90,
        "EV_NV": 70, "EV_V": 65,
        "EV Std_NV": 85, "EV Std_V": 80,
        # Wrong Bid Analysis columns
        "criterion": 150, "failure_count": 95,
        "check_count": 85, "affected_auctions": 115, "fail_rate": 85,
        "wrong_bid_rate": 105, "wrong_bid_rate_%": 115,
        "Wrong Bids": 95, "Wrong Bid Rate": 120,
    }
    # Ensure headers never get truncated: set minWidth based on header text length.
    # AgGrid's auto-size can shrink columns based on cell contents; minWidth prevents
    # it from shrinking below what's needed for the column name.
    CHAR_WIDTH = 12  # Approximate pixels per character for header text
    for col_name, target_width in tight_cols.items():
        if col_name in df.columns:
            # Extra padding accounts for sort icon + (optional) filter/menu affordances.
            header_width = len(col_name) * CHAR_WIDTH + 48
            width = max(target_width, header_width)
            # Prevent auto-size from shrinking below the header width.
            gb.configure_column(col_name, width=width, minWidth=width)

    # ---------------------------------------------------------------------
    # Autosize columns: max(content width, header width)
    # ---------------------------------------------------------------------
    # Streamlit-AgGrid auto-size can under-size headers; we compute a best-effort
    # width from sampled content (first N rows) and the column name.
    #
    # This is intentionally approximate (fast) and avoids scanning huge tables.
    SAMPLE_ROWS = 200
    MAX_CHARS = 80  # cap overly wide string columns
    MIN_WIDTH = 60
    MAX_WIDTH = 650
    CELL_PAD = 48  # sort icon + filter/menu affordances + padding

    try:
        sample_df = pandas_df.head(SAMPLE_ROWS)
        for col_name in df.columns:
            if hide_cols and col_name in hide_cols:
                continue
            # Respect explicit configs above
            if col_name in tight_cols:
                continue
            if col_name in ("Agg_Expr",):
                # handled separately (flex)
                continue
            if col_name == "Auction":
                # handled separately (maxWidth)
                continue

            header_chars = len(str(col_name))
            # Compute max content length on a sample (NaN/None -> empty)
            try:
                series = sample_df[col_name]
                # Convert to string safely; avoid "nan"
                vals = series.astype("object").where(series.notna(), "")
                max_cell_chars = int(vals.map(lambda x: len(str(x))).max()) if len(vals) else 0
            except Exception:
                max_cell_chars = 0

            max_chars = min(MAX_CHARS, max(header_chars, max_cell_chars))
            width = max(MIN_WIDTH, min(MAX_WIDTH, max_chars * CHAR_WIDTH + CELL_PAD))
            gb.configure_column(col_name, width=width, minWidth=width)
    except Exception:
        # Best-effort only; grid still renders fine without these hints.
        pass
    
    # Agg_Expr: Use fixed width instead of flex to prevent resizing on grid rerender
    if "Agg_Expr" in df.columns:
        tooltip_field = "Agg_Expr_full" if "Agg_Expr_full" in df.columns else "Agg_Expr"
        gb.configure_column(
            "Agg_Expr", 
            width=400,
            minWidth=200,
            wrapText=True,
            autoHeight=True,
            suppressSizeToFit=True,  # Prevent width changes on rerender
            tooltipField=tooltip_field,  # Show full content on hover
        )

    # Categories column: Use fixed width instead of flex to prevent resizing on grid rerender
    if "Categories" in df.columns:
        tooltip_field = "Categories_full" if "Categories_full" in df.columns else "Categories"
        gb.configure_column(
            "Categories",
            width=300,
            minWidth=200,
            wrapText=True,
            autoHeight=True,
            suppressSizeToFit=True,  # Prevent width changes on rerender
            tooltipField=tooltip_field,
        )
    
    # Auction column can be wide - constrain it
    if "Auction" in df.columns:
        gb.configure_column("Auction", maxWidth=200)
    
    gb.configure_grid_options(
        rowHeight=28,
        headerHeight=32,
        suppressCellFocus=True,
        tooltipShowDelay=300,  # Show tooltips after 300ms hover
        suppressColumnVirtualisation=True,  # Prevent virtualization from affecting widths on rerender
    )
    grid_options = gb.build()

    # If provided, set an initial sort. (AgGrid can persist user sorts per-key; this
    # plus changing the key is the most reliable way to enforce the default.)
    if sort_model is not None and "sortModel" not in grid_options:
        grid_options["sortModel"] = sort_model
    
    # Make columns feel tighter (AgGrid defaults can be quite wide)
    # - Smaller minWidth prevents excess whitespace in narrow columns
    # - Slightly reduced padding tightens the visual layout
    default_col_def = grid_options.get("defaultColDef") or {}
    default_col_def.setdefault("minWidth", 60)
    default_col_def.setdefault("cellStyle", {"padding": "2px 6px"})
    grid_options["defaultColDef"] = default_col_def

    # Choose auto-size mode: FIT_ALL_COLUMNS_TO_VIEW for compact tables, FIT_CONTENTS for wide tables
    auto_size_mode = ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW if fit_columns_to_view else ColumnsAutoSizeMode.FIT_CONTENTS

    response = AgGrid(
        df.to_pandas(),
        gridOptions=grid_options,
        height=height,
        theme="balham",
        key=key,
        columns_auto_size_mode=auto_size_mode,
        # Critical UX: clicking a row should only highlight it locally and NOT
        # emit selection/model updates back to Streamlit (which would cause a rerun)
        # unless specifically requested via update_mode.
        update_mode=update_mode,
        data_return_mode=DataReturnMode.AS_INPUT,
        # For clickable grids (selection triggers actions), add a blue hover highlight + pointer.
        # For read-only grids, add a very subtle gray hover to help eye tracking.
        custom_css=(
            {
                ".ag-row": {"cursor": "pointer", "transition": "background-color 0.1s"},
                ".ag-row-hover": {
                    "background-color": "#E8F4FF !important",
                    "border-left": "3px solid #007BFF !important"
                },
                ".ag-row-selected": {"background-color": "#D1E9FF !important"},
            }
            if update_mode != GridUpdateMode.NO_UPDATE
            else {
                ".ag-row": {"cursor": "default"},
                ".ag-row-hover": {"background-color": "#F8F9FA !important"},
            }
        ),
    )
    
    selected_rows: Any = response.get("selected_rows", [])
    # AgGrid returns a list of dicts or a list of dataframes depending on version
    if selected_rows is not None and hasattr(selected_rows, "to_dict"):
        selected_records = selected_rows.to_dict("records")
    else:
        selected_records = list(selected_rows) if selected_rows is not None else []

    # Option B: Copy panel (best-effort, works regardless of AgGrid clipboard support)
    if show_copy_panel and selected_records:
        row0 = selected_records[0] or {}
        cols = list(df.columns)
        default_col = copy_panel_default_col if (copy_panel_default_col in cols) else (cols[0] if cols else "")
        if default_col:
            # If the widget keys don't change when the *selected row* changes, Streamlit will
            # preserve the previous text_area state and the displayed value won't update.
            # Compute a best-effort fingerprint for the selected row to include in widget keys.
            def _row_fingerprint(r: dict[str, Any]) -> str:
                try:
                    for k in ("_row_idx", "index", "bt_index", "Auction"):
                        if k in r and r.get(k) is not None:
                            return str(r.get(k))
                    # Fall back to a stable hash of the row dict content.
                    import json
                    payload = json.dumps(r, sort_keys=True, default=str)
                    return str(abs(hash(payload)))
                except Exception:
                    return "row"

            row_fp = _row_fingerprint(row0)
            sel_col = st.selectbox(
                "Copy field from selected row",
                cols,
                index=cols.index(default_col),
                key=f"{key}__copy_col",
                help="Select a column, then Ctrl+A / Ctrl+C the value below.",
            )
            val = row0.get(sel_col)
            st.text_area(
                "Value (Ctrl+A, Ctrl+C)",
                value="" if val is None else str(val),
                height=80,
                # Key must vary with selected column *and selected row*, otherwise Streamlit
                # will preserve stale state and the displayed value won't update.
                key=f"{key}__copy_val__{sel_col}__{row_fp}",
            )

    return selected_records

if st.session_state.get("first_run", True):
    st.session_state.app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    st.session_state.first_run = False

def app_info() -> None:
    """Display app information"""
    def pkg_version(dist_name: str) -> str:
        try:
            return importlib_metadata.version(dist_name)
        except Exception:
            return "n/a"

    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(
        " ".join(
            [
                f"Python:{'.'.join(map(str, sys.version_info[:3]))}",
                f"duckdb:{pkg_version('duckdb')}",
                f"FastAPI:{pkg_version('fastapi')}",
                f"Uvicorn:{pkg_version('uvicorn')}",
                f"pandas:{pd.__version__}",
                f"polars:{pl.__version__}",
                f"endplay:{pkg_version('endplay')}",
            ]
        )
    )
    return

st.set_page_config(layout="wide")
st.title("BBO GIB Bidding Playground (Proof of Concept)")
app_info()

# ---------------------------------------------------------------------------
# server initialization / maintenance gate
# ---------------------------------------------------------------------------

try:
    status = api_get("/status")
except Exception as exc:  # pragma: no cover - connectivity issues
    st.error(f"Cannot reach server at {API_BASE}: {exc}")
    st.stop()

if not status["initialized"] or status.get("warming", False):
    st.header("Maintenance in progress")
    if status.get("warming", False):
        st.write("Server is warming up endpoints; please wait...")
    else:
        st.write("Server is loading data; please wait...")
    
    # Show current loading step
    loading_step = status.get("loading_step")
    if loading_step:
        st.info(f"📍 **Current step:** {loading_step}")
    
    # Show loaded files with row counts
    loaded_files = status.get("loaded_files")
    if loaded_files:
        st.subheader("Files loaded:")
        for file_name, row_count in loaded_files.items():
            # Handle both int and string row counts (e.g., "100 of 15,994,827")
            if isinstance(row_count, int):
                st.write(f"✅ **{file_name}**: {row_count:,}")
            else:
                st.write(f"✅ **{file_name}**: {row_count}")
    
    # Show raw status in expander
    with st.expander("Raw status JSON"):
        st.json(status)

    # Auto-poll: wait then rerun to check if ready
    with st.spinner("Waiting for server to finish loading..."):
        time.sleep(30)
    st.rerun()

if status.get("error"):
    st.warning(f"Server reported an initialization error: {status['error']}")

# Display dataset info
bt_df_rows = status.get("bt_df_rows")
deal_df_rows = status.get("deal_df_rows")
if bt_df_rows is not None and deal_df_rows is not None:
    st.info(f"📊 Loaded data: **{deal_df_rows:,}** deals, **{bt_df_rows:,}** bidding table entries")

# ---------------------------------------------------------------------------
# Auto-reload custom criteria overlay on every page refresh
# ---------------------------------------------------------------------------
# This ensures CSV edits are picked up immediately without manual button clicks.
# The reload is fast (just re-reads the CSV) so it's safe to do on every refresh.

def _auto_reload_criteria_overlay():
    """Silently reload custom criteria overlay from CSV."""
    try:
        resp = requests.post(f"{API_BASE}/custom-criteria-reload", timeout=10)
        if resp.ok:
            data = resp.json()
            rules_count = data.get("stats", {}).get("rules_applied", 0)
            # Store in session state for display in sidebar if needed
            st.session_state["_criteria_overlay_rules"] = rules_count
    except Exception:
        pass  # Silently ignore - criteria will use last loaded version

_auto_reload_criteria_overlay()

# ---------------------------------------------------------------------------
# Page Render Functions - one per selectbox option
# ---------------------------------------------------------------------------

def render_opening_bids_by_deal():
    """Browse deals by index and see which opening bids match."""
    sample_size = st.sidebar.number_input("Sample Deals", value=6, min_value=1, max_value=100)

    all_seats = st.sidebar.checkbox("All Seats", value=True)
    seats = None if all_seats else st.sidebar.multiselect("Seats", [1, 2, 3, 4], default=[1, 2, 3, 4])

    DIRECTIONS = ["N", "E", "S", "W"]
    all_dirs = st.sidebar.checkbox("All Dealer Directions", value=True)
    directions = None if all_dirs else st.sidebar.multiselect(
        "Dealer Directions", DIRECTIONS, default=DIRECTIONS
    )

    all_openers = st.sidebar.checkbox("All Openers", value=True)
    opening_directions = None if all_openers else st.sidebar.multiselect(
        "Opening Directions", DIRECTIONS, default=["S"]
    )

    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_opening"))

    payload = {
        "sample_size": int(sample_size),
        "seats": seats,
        "directions": directions,
        "opening_directions": opening_directions,
        "seed": seed,
    }

    with st.spinner("Fetching Openings by Deal Index from server. Takes 20 to 60 seconds."):
        data = api_post("/openings-by-deal-index", payload)

    deals = data.get("deals", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    _st_info_elapsed("Opening Bids by Deal", data)
    if not deals:
        st.info("No deals matched the specified filters.")
    else:
        st.info(f"Showing {len(deals)} deal(s) in {format_elapsed(elapsed_ms)}")
        for d in deals:
            st.subheader(f"Dealer {d['dealer']} – Deal Index {d['index']}")
            st.write(f"Opening seat: {d.get('opening_seat')}")
            st.write(f"Opening bid indices: {d.get('opening_bid_indices', [])}")

            opening_bids_df = d.get("opening_bids_df", [])
            
            # Extract invariant columns (same for all rows) to show in Deal Info instead
            invariant_cols = ["Dealer", "Actual_Auction", "Actual_Contract", "ParScore", "ParContract", "EV_ParContracts"]
            invariant_values: dict = {}
            
            if opening_bids_df:
                st.write("Opening Bids:")
                # Rank bids best→worst using reverse sort order of Auction (per request)
                try:
                    bids_df = pl.DataFrame(opening_bids_df)
                    
                    # Extract invariant values from first row before removing columns
                    if bids_df.height > 0:
                        first_row = bids_df.row(0, named=True)
                        for col in invariant_cols:
                            if col in first_row:
                                invariant_values[col] = first_row[col]
                    
                    if "Auction" in bids_df.columns:
                        # Sort by Seat ascending, then Auction descending
                        if "seat" in bids_df.columns:
                            bids_df = bids_df.sort(["seat", "Auction"], descending=[False, True])
                        else:
                            bids_df = bids_df.sort("Auction", descending=True)
                        bids_df = bids_df.with_row_index("Best_Bid").with_columns(
                            (pl.col("Best_Bid") + 1).alias("Best_Bid")
                        )
                        
                        bids_df = order_columns(bids_df, priority_cols=[
                            "Best_Bid", "index", "Auction", "seat",
                            "Rules_Contract", "DD_Score_Declarer", "EV_Score_Declarer", "Expr",
                        ], drop_cols=invariant_cols)
                    
                    render_aggrid(bids_df, key=f"bids_{d['dealer']}_{d['index']}", table_name="opening_bids")
                except Exception:
                    # Fallback: render raw payload
                    render_aggrid(opening_bids_df, key=f"bids_{d['dealer']}_{d['index']}", table_name="opening_bids")
            else:
                st.info("No opening bids found.")

            # Show Deal Info: hands + invariant columns from opening bids
            if d.get("hands") or invariant_values:
                st.write("Deal Info:")
                hands_dict = d.get("hands", {})
                deal_info = {
                    "Hand_N": hands_dict.get("Hand_N"),
                    "Hand_E": hands_dict.get("Hand_E"),
                    "Hand_S": hands_dict.get("Hand_S"),
                    "Hand_W": hands_dict.get("Hand_W"),
                }
                # Add invariant values from opening bids
                deal_info.update(invariant_values)
                df_deal_info = pl.DataFrame([deal_info])
                df_deal_info = order_columns(df_deal_info, priority_cols=[
                    "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                    "Dealer", "Actual_Auction", "Actual_Contract", "ParScore", "ParContract", "EV_ParContracts",
                ])
                render_aggrid(df_deal_info, key=f"deal_info_{d['dealer']}_{d['index']}", table_name="deal_info")
            st.divider()


def render_random_auction_samples():
    """View random completed auction sequences from the bidding table."""
    n_samples = st.sidebar.number_input("Number of Samples", value=10, min_value=1)
    
    auction_type = st.sidebar.radio(
        "Auction Type",
        options=["Completed Only", "Partial Only", "50-50 Mix"],
        index=0,
        help="Completed: auctions ending in final contract. Partial: intermediate sequences. 50-50: mix of both.",
    )
    
    # Map radio selection to API parameters
    completed_only = auction_type == "Completed Only"
    partial_only = auction_type == "Partial Only"

    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_random"))

    payload = {"n_samples": int(n_samples), "seed": seed, "completed_only": completed_only, "partial_only": partial_only}
    with st.spinner("Fetching bidding sequences from server. Takes about 10 seconds."):
        data = api_post("/random-auction-sequences", payload)

    samples = data.get("samples", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    _st_info_elapsed("Random Auction Samples", data)
    if not samples:
        st.info("No auctions found.")
    else:
        st.info(f"Showing {len(samples)} matching auction(s) in {format_elapsed(elapsed_ms)}")
        
        # Build combined DataFrame from all samples for comparison
        all_rows = []
        for s in samples:
            if isinstance(s.get("sequence"), list):
                for row in s["sequence"]:
                    all_rows.append(row)
        
        if all_rows:
            combined_df = pl.DataFrame(all_rows)
            combined_df = order_columns(combined_df, priority_cols=[
                "index", "Auction", "Expr",
                "Agg_Expr_S1_Base", "Agg_Expr_S1_Learned", "Agg_Expr_S1_Full",
                "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
            ])
            if "index" in combined_df.columns:
                combined_df = combined_df.sort("index")
        else:
            combined_df = pl.DataFrame()
        
        # Display individual samples
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: {s['auction']}")
            seq_df = pl.DataFrame(s["sequence"]) if isinstance(s["sequence"], list) else s["sequence"]
            seq_df = order_columns(seq_df, priority_cols=[
                "index", "Auction", "Expr",
                "Agg_Expr_S1_Base", "Agg_Expr_S1_Learned", "Agg_Expr_S1_Full",
                "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
            ])
            if "index" in seq_df.columns:
                seq_df = seq_df.sort("index")
            render_aggrid(seq_df, key=f"seq_random_{i}", table_name="auction_sequences")
            st.divider()


def render_find_auction_sequences(pattern: str | None, indices: list[int] | None = None):
    """Search for auction sequences by regex pattern OR by bt_index list (mutually exclusive)."""
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)
    allow_initial_passes = st.sidebar.checkbox(
        "Match all 4 dealer positions",
        value=False,
        help=MATCH_ALL_DEALERS_HELP,
        key="allow_initial_passes_find",
    )

    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_find"))

    if indices:
        if pattern:
            st.error("Please use either Auction Regex or bt_index list (not both).")
            return
        payload = {"indices": [int(x) for x in indices], "allow_initial_passes": bool(allow_initial_passes)}
        with st.spinner("Fetching auctions by bt_index from server. Takes about 10-60 seconds."):
            data = api_post("/auction-sequences-by-index", payload)
        _st_info_elapsed("Find Auction Sequences", data)
        missing = data.get("missing_indices") or []
        if missing:
            st.warning(f"{len(missing)} bt_index values were not found (or not completed auctions).")
    else:
        # Ensure we always send a string to the API (avoid 422 if pattern is None).
        pattern = pattern or ""
        if not pattern:
            st.info("Enter an Auction Regex or provide a bt_index list in the sidebar.")
            return
        payload = {"pattern": pattern, "allow_initial_passes": bool(allow_initial_passes), "n_samples": int(n_samples), "seed": seed}
        with st.spinner("Fetching auctions from server. Takes about 10-60 seconds."):
            data = api_post("/auction-sequences-matching", payload)
        _st_info_elapsed("Find Auction Sequences", data)

    # Show the effective pattern including (p-)* prefix when matching all seats
    effective_pattern = data.get('pattern', pattern)
    if indices:
        st.caption(f"Using bt_index list ({len(indices)} provided)")
    elif allow_initial_passes and effective_pattern:
        display_pattern = prepend_all_seats_prefix(effective_pattern)
        st.caption(f"Effective pattern: {display_pattern} (4 seat variants)")
    else:
        st.caption(f"Effective pattern: {effective_pattern}")
    samples = data.get("samples", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    if not samples:
        st.info("No auctions matched the pattern.")
    else:
        st.info(f"Showing {len(samples)} matching auction(s) in {format_elapsed(elapsed_ms)}")
        
        def _to_int_or_default(x: Any, default: int = 1) -> int:
            try:
                return default if x is None else int(x)
            except Exception:
                return default

        def _add_seat_columns(df: pl.DataFrame, opener_seat: int) -> pl.DataFrame:
            """Add Seat column (cycling 1-4 per row) and Agg_Expr_Seat from the appropriate column."""
            n_rows = df.height
            # Compute seat for each row: opener_seat, opener_seat+1, ... cycling 1-4
            seats = [((opener_seat - 1 + i) % 4) + 1 for i in range(n_rows)]
            df = df.with_columns(pl.Series("Seat", seats))
            
            # Build Agg_Expr_Seat by picking the right column value per row
            agg_expr_vals = []
            for i, seat in enumerate(seats):
                agg_col = f"Agg_Expr_Seat_{seat}"
                if agg_col in df.columns:
                    agg_expr_vals.append(_to_list_utf8_cell(df[agg_col][i]))
                else:
                    agg_expr_vals.append(None)
            # Force a strongly-typed list-of-strings column (prevents Polars panic on all-None).
            df = df.with_columns(pl.Series("Agg_Expr_Seat", agg_expr_vals, dtype=pl.List(pl.Utf8)))
            
            # Drop individual Agg_Expr_Seat_[1-4] columns
            drop_cols = [c for c in df.columns if c.startswith("Agg_Expr_Seat_") and c[-1].isdigit()]
            if drop_cols:
                df = df.drop(drop_cols)
            return df

        # Build combined DataFrame from all samples for comparison
        all_rows = []
        for s in samples:
            opener_seat = _to_int_or_default(s.get("opener_seat"), 1)
            if isinstance(s.get("sequence"), list):
                for row_idx, row in enumerate(s["sequence"]):
                    r = dict(row)
                    # Seat cycles: opener_seat, opener_seat+1, ... (1-4)
                    seat = ((opener_seat - 1 + row_idx) % 4) + 1
                    r["Seat"] = seat
                    # Pick the right Agg_Expr_Seat column value
                    r["Agg_Expr_Seat"] = r.get(f"Agg_Expr_Seat_{seat}")
                    all_rows.append(r)
        
        if all_rows:
            combined_df = pl.DataFrame(all_rows)
            # Drop individual Agg_Expr_Seat_[1-4] columns
            drop_cols = [c for c in combined_df.columns if c.startswith("Agg_Expr_Seat_") and c[-1].isdigit()]
            if drop_cols:
                combined_df = combined_df.drop(drop_cols)
            combined_df = order_columns(combined_df, priority_cols=[
                "index", "Auction", "Seat", "Expr", "Agg_Expr_Seat",
            ])
            if "index" in combined_df.columns:
                combined_df = combined_df.sort("index")
        else:
            combined_df = pl.DataFrame()
        
        # Display individual samples
        for i, s in enumerate(samples, start=1):
            opener_seat = _to_int_or_default(s.get("opener_seat"), 1)
            st.subheader(f"Sample {i}: {s['auction']} (Opener Seat {opener_seat})")
            seq_df = pl.DataFrame(s["sequence"]) if isinstance(s["sequence"], list) else s["sequence"]
            seq_df = _add_seat_columns(seq_df, opener_seat)
            seq_df = order_columns(seq_df, priority_cols=[
                "index", "Auction", "Seat", "Expr", "Agg_Expr_Seat",
            ])
            if "index" in seq_df.columns:
                seq_df = seq_df.sort("index")
            render_aggrid(seq_df, key=f"seq_pattern_{i}", table_name="auction_sequences")
            st.divider()
    
    criteria_rejected = data.get("criteria_rejected", [])
    if criteria_rejected:
        st.markdown(f"🚫 **Rejected auctions due to custom criteria** ({len(criteria_rejected)} shown)")
        st.caption("Rows filtered out by `bbo_custom_auction_criteria.csv` rules (hot-reloadable overlay).")
        try:
            rejected_df = pl.DataFrame(criteria_rejected)
            render_aggrid(rejected_df, key="rejected_find_auction", height=calc_grid_height(len(criteria_rejected), max_height=300), table_name="rejected_auctions")
        except Exception as e:
            st.warning(f"Could not render as table: {e}")
            st.json(criteria_rejected)


def render_deals_by_auction_pattern(pattern: str | None):
    """Find deals matching an auction pattern's criteria."""
    n_auction_samples = st.sidebar.number_input("Auction Samples", value=2, min_value=1, max_value=10)
    n_deal_samples = st.sidebar.number_input("Deal Samples per Auction", value=100, min_value=1, max_value=10_000_000)
    allow_initial_passes = st.sidebar.checkbox(
        "Match all 4 dealer positions",
        value=False,
        help=MATCH_ALL_DEALERS_HELP,
        key="allow_initial_passes_deals",
    )

    # Distribution filter for deals
    st.sidebar.divider()
    st.sidebar.subheader("Distribution Filter")
    
    deal_dist_direction = st.sidebar.selectbox("Filter Hand", ["N", "E", "S", "W"], index=2,
        help="Which hand's distribution to filter")
    
    deal_dist_pattern = st.sidebar.text_input("Ordered Distribution (S-H-D-C)", value="",
        placeholder="e.g., 5-4-3-1, 5+-4-x-x",
        help="Filter deals by exact suit order. Leave empty for no filter.",
        key="deal_dist_pattern")
    
    if deal_dist_pattern:
        parsed_dist = parse_distribution_pattern(deal_dist_pattern)
        if parsed_dist:
            dist_display = []
            for suit in ['S', 'H', 'D', 'C']:
                constraint = parsed_dist[suit]
                if constraint is None:
                    dist_display.append(f"{suit}:any")
                elif constraint[0] == constraint[1]:
                    dist_display.append(f"{suit}:{constraint[0]}")
                else:
                    dist_display.append(f"{suit}:{constraint[0]}-{constraint[1]}")
            st.sidebar.caption(f"→ {', '.join(dist_display)}")
        else:
            st.sidebar.warning("Invalid distribution pattern")
    
    deal_sorted_shape = st.sidebar.text_input("Sorted Shape (any suit order)", value="",
        placeholder="e.g., 5431, 4432, 5332",
        help="Filter deals by shape regardless of suit.",
        key="deal_sorted_shape")
    
    if deal_sorted_shape:
        parsed_shape = parse_sorted_shape(deal_sorted_shape)
        if parsed_shape:
            st.sidebar.caption(f"→ shape {''.join(map(str, parsed_shape))} (any suits)")
        else:
            st.sidebar.warning("Invalid sorted shape (must be 4 digits summing to 13)")
    
    with st.sidebar.expander("Distribution notation help"):
        st.markdown(format_distribution_help())

    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_deals"))

    # Ensure we always send a string to the API (avoid 422 if pattern is None).
    pattern = pattern or ""
    if not pattern:
        st.info("Enter an Auction Regex in the sidebar.")
        return

    # Build payload with distribution filter params (server-side filtering)
    payload = {
        "pattern": pattern,
        "allow_initial_passes": bool(allow_initial_passes),
        "n_auction_samples": int(n_auction_samples),
        "n_deal_samples": int(n_deal_samples),
        "seed": seed,
        "dist_pattern": deal_dist_pattern if deal_dist_pattern else None,
        "sorted_shape": deal_sorted_shape if deal_sorted_shape else None,
        "dist_direction": deal_dist_direction,
    }

    with st.spinner("Fetching Deals Matching Auction from server. Takes about 15-30s seconds."):
        data = api_post("/deals-matching-auction", payload)
    _st_info_elapsed("Deals by Auction Pattern", data)

    elapsed_ms = data.get("elapsed_ms", 0)
    # Show the effective pattern including (p-)* prefix when matching all seats
    effective_pattern = data.get('pattern', pattern)
    if allow_initial_passes and effective_pattern:
        if effective_pattern.startswith("^"):
            display_pattern = "^(p-)*" + effective_pattern[1:]
        else:
            display_pattern = "(p-)*" + effective_pattern
        st.caption(f"Effective pattern: {display_pattern} (4 seat variants)")
    else:
        st.caption(f"Effective pattern: {effective_pattern}")
    
    # Show distribution filter info if applied
    dist_filter = data.get("dist_filter")
    if dist_filter:
        st.caption(f"Distribution filter: {dist_filter.get('dist_pattern') or ''} {dist_filter.get('sorted_shape') or ''} (Hand_{dist_filter.get('direction', 'N')})")
    
    auctions = data.get("auctions", [])
    if not auctions:
        st.info("No auctions matched the pattern.")
    else:
        st.info(f"Showing {len(auctions)} matching auction(s) in {format_elapsed(elapsed_ms)}")
        for i, a in enumerate(auctions, start=1):
            bt_idx = a.get("bt_index")
            bt_suffix = f" (bt_index={bt_idx})" if bt_idx is not None else ""
            st.subheader(f"Auction {i}: {a['auction']}{bt_suffix}")
            
            # Show sample count vs total matching deals
            total_matching = a.get("total_matching_deals", 0)
            st.caption(f"Showing {n_deal_samples} of {total_matching:,} matching deals")
            
            # Show criteria debug info (Applied + Missing = all criteria for auction)
            criteria_debug = a.get("criteria_debug", {})
            row_seat = criteria_debug.get("row_seat", "?")
            actual_final_seat = criteria_debug.get("actual_final_seat", "?")
            missing = criteria_debug.get("missing", {})
            found = criteria_debug.get("found", {})
            
            # Explain seat positions
            st.caption(f"ℹ️ Row seat={row_seat}, Actual final seat={actual_final_seat}. "
                      f"Seat 1=Dealer, Seat 2=LHO, Seat 3=Partner, Seat 4=RHO")
            
            if missing:
                with st.expander(f"⚠️ Missing Criteria ({sum(len(v) for v in missing.values())} total)", expanded=True):
                    st.warning("These criteria could not be matched to pre-computed bitmaps - filtering may be incomplete!")
                    for key, criteria_list in missing.items():
                        seat_num = int(key.split('_')[1]) if '_' in key else 0
                        role = SEAT_ROLES.get(seat_num, "")
                        st.write(f"**{key}** ({role}): {', '.join(criteria_list)}")
            if found:
                with st.expander(f"✅ Applied Criteria ({sum(len(v) for v in found.values())} total)", expanded=False):
                    for key in sorted(found.keys()):
                        criteria_list = found[key]
                        seat_num = int(key.split('_')[1]) if '_' in key else 0
                        role = SEAT_ROLES.get(seat_num, "")
                        if criteria_list:
                            st.write(f"**{key}** ({role}): {', '.join(criteria_list)}")
                        else:
                            st.write(f"**{key}** ({role}): *(no criteria)*")
            
            # Show distribution SQL if returned
            dist_sql = a.get("dist_sql_query")
            if dist_sql:
                with st.expander("🔍 Distribution SQL Query", expanded=False):
                    st.code(dist_sql, language="sql")
            
            # AI vs Actual Comparison table (show first as summary)
            total_imp = a.get("total_imp_rules", 0)
            total_deals = a.get("total_deals", 0)
            imp_ai = a.get("imp_rules_advantage", 0)
            imp_actual = a.get("imp_actual_advantage", 0)
            ai_makes = a.get("rules_makes_count", 0)
            contract_makes = a.get("contract_makes_count", 0)
            ai_par = a.get("rules_par_count", 0)
            contract_par = a.get("contract_par_count", 0)
            avg_ev_contract = a.get("avg_ev_contract")
            avg_ev_ai = a.get("avg_ev_rules")
            avg_ev_diff = a.get("avg_ev_diff")
            
            if total_deals > 0:
                # Calculate percentages
                ai_makes_pct = (ai_makes / total_deals * 100) if total_deals > 0 else 0
                contract_makes_pct = (contract_makes / total_deals * 100) if total_deals > 0 else 0
                ai_par_pct = (ai_par / total_deals * 100) if total_deals > 0 else 0
                contract_par_pct = (contract_par / total_deals * 100) if total_deals > 0 else 0
                
                # Build comparison table
                st.write("**Overalls: AI vs Actual Comparison**")
                
                stats_rows = []
                
                # IMPs row
                imp_diff_str = f"+{total_imp}" if total_imp >= 0 else str(total_imp)
                stats_rows.append({"Metric": "IMPs", "AI": imp_ai, "Actual": imp_actual, "Diff": imp_diff_str})
                
                # DD Makes row (only if data exists)
                if ai_makes > 0 or contract_makes > 0:
                    makes_diff = ai_makes_pct - contract_makes_pct
                    stats_rows.append({"Metric": "DD Makes %", "AI": round(ai_makes_pct, 1), "Actual": round(contract_makes_pct, 1), "Diff": round(makes_diff, 1)})
                
                # Par Achieved row (only if data exists)
                if ai_par > 0 or contract_par > 0:
                    par_diff = ai_par_pct - contract_par_pct
                    stats_rows.append({"Metric": "Par Achieved %", "AI": round(ai_par_pct, 1), "Actual": round(contract_par_pct, 1), "Diff": round(par_diff, 1)})
                
                # EV row (only if data exists)
                if avg_ev_ai is not None and avg_ev_contract is not None:
                    ev_diff_str = f"+{avg_ev_diff:.2f}" if avg_ev_diff >= 0 else f"{avg_ev_diff:.2f}"
                    stats_rows.append({"Metric": "Avg EV", "AI": f"{avg_ev_ai:.2f}", "Actual": f"{avg_ev_contract:.2f}", "Diff": ev_diff_str})
                
                if stats_rows:
                    stats_df = pl.DataFrame(stats_rows)
                    stats_df = order_columns(stats_df, priority_cols=[
                        "Metric", "Value", "Notes",
                    ])
                    render_aggrid(stats_df, key=f"stats_{i}", height=150, table_name="auction_stats")
            
            # Contract summary table (detailed breakdown by contract)
            contract_summary = a.get("contract_summary", [])
            if contract_summary:
                st.write("**Contract Summary:**")
                summary_df = pl.DataFrame(contract_summary)
                summary_df = order_columns(summary_df, priority_cols=[
                    "Contract", "Count", "Avg_IMP_Rules", "Contract_Made%", "Rules_Made%",
                    "Contract_Par%", "Rules_Par%",
                ])
                render_aggrid(summary_df, key=f"contract_summary_{i}", height=200, table_name="contract_summary")
            
            deals = a.get("deals", [])
            if deals:
                deals_df = pl.DataFrame(deals)
                deals_df = order_columns(deals_df, priority_cols=[
                    "index", "Dealer", "Vul", "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                    "Contract", "Result", "Tricks", "Score", "DD_Score_Declarer", "ParScore",
                    "HCP_N", "HCP_E", "HCP_S", "HCP_W",
                ])
                if "index" in deals_df.columns:
                    deals_df = deals_df.sort("index")
                st.write(f"**Matching Deals:** (showing {len(deals_df)})")
                render_aggrid(deals_df, key=f"deals_{i}", table_name="deals")
            else:
                st.info("No matching deals (criteria may be too restrictive or distribution filter removed all).")
            st.divider()
    
    # Show criteria-rejected rows for debugging (from criteria.csv)
    criteria_rejected = data.get("criteria_rejected", [])
    if criteria_rejected:
        st.markdown(f"🚫 **Rejected auctions due to custom criteria** ({len(criteria_rejected)} shown)")
        st.caption("Rows filtered out by `bbo_custom_auction_criteria.csv` rules (hot-reloadable overlay).")
        try:
            rejected_df = pl.DataFrame(criteria_rejected)
            render_aggrid(rejected_df, key="rejected_deals_by_auction", height=calc_grid_height(len(criteria_rejected), max_height=300), table_name="rejected_auctions")
        except Exception as e:
            st.warning(f"Could not render as table: {e}")
            st.json(criteria_rejected)


def render_bidding_table_explorer():
    """Browse bidding table entries with aggregate statistics."""
    st.header("Bidding Table Statistics Viewer")
    st.caption("View bidding table entries with aggregate statistics (mean, std, min, max) for matching deals.")
    
    st.sidebar.subheader("Statistics Filters")
    
    raw_auction_pattern = st.sidebar.text_input("Auction Regex", value="^1N-p-3N$",
        help="Trailing '-p-p-p' is assumed if not present (e.g., '1N-p-3N' → '1N-p-3N-p-p-p')",
        key="bt_explorer_auction_pattern")
    allow_initial_passes = st.sidebar.checkbox(
        "Match all 4 dealer positions",
        value=False,
        help=MATCH_ALL_DEALERS_HELP,
        key="allow_initial_passes_bt",
    )
    auction_pattern = normalize_auction_user_text(raw_auction_pattern)
    # Show the effective pattern including (p-)* prefix when matching all seats
    if allow_initial_passes:
        display_pattern = prepend_all_seats_prefix(auction_pattern)
        st.sidebar.caption(f"→ {display_pattern} (4 seat variants)")
    elif auction_pattern != raw_auction_pattern:
        st.sidebar.caption(f"→ {auction_pattern}")
    
    sample_size = st.sidebar.number_input("Sample Size", value=25, min_value=1, max_value=10000, key="bt_explorer_sample")
    min_matches = st.sidebar.number_input("Min Matching Deals (0=all)", value=0, min_value=0, max_value=100000)

    st.sidebar.divider()
    show_categories = st.sidebar.checkbox(
        "Show bid category flags (Phase 4)",
        value=False,
        help="Includes ~100 boolean is_* columns from bbo_bt_categories.parquet (can be wide).",
        key="bt_explorer_show_categories",
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("Distribution Filter")
    
    dist_seat = st.sidebar.selectbox("Filter Seat", [1, 2, 3, 4], index=0,
        help="Which seat's distribution to filter (S1=opener in most auctions)",
        key="bt_explorer_dist_seat")
    
    dist_pattern = st.sidebar.text_input("Ordered Distribution (S-H-D-C)", value="",
        placeholder="e.g., 5-4-3-1, 5+-4-x-x",
        help="Filter by exact suit order. Leave empty for no filter.",
        key="bt_explorer_dist_pattern")
    
    if dist_pattern:
        parsed_dist = parse_distribution_pattern(dist_pattern)
        if parsed_dist:
            dist_display = []
            for suit in ['S', 'H', 'D', 'C']:
                constraint = parsed_dist[suit]
                if constraint is None:
                    dist_display.append(f"{suit}:any")
                elif constraint[0] == constraint[1]:
                    dist_display.append(f"{suit}:{constraint[0]}")
                else:
                    dist_display.append(f"{suit}:{constraint[0]}-{constraint[1]}")
            st.sidebar.caption(f"→ {', '.join(dist_display)}")
        else:
            st.sidebar.warning("Invalid distribution pattern")
    
    sorted_shape = st.sidebar.text_input("Sorted Shape (any suit order)", value="",
        placeholder="e.g., 5431, 4432, 5332",
        help="Filter by shape regardless of suit.",
        key="bt_explorer_sorted_shape")
    
    if sorted_shape:
        parsed_shape = parse_sorted_shape(sorted_shape)
        if parsed_shape:
            st.sidebar.caption(f"→ shape {''.join(map(str, parsed_shape))} (any suits)")
        else:
            st.sidebar.warning("Invalid sorted shape (must be 4 digits summing to 13)")
    
    with st.sidebar.expander("Distribution notation help"):
        st.markdown(format_distribution_help())
    
    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_explorer"))

    payload = {
        "auction_pattern": auction_pattern,
        "allow_initial_passes": bool(allow_initial_passes),
        "sample_size": int(sample_size),
        "min_matches": int(min_matches),
        "seed": seed,
        "dist_pattern": dist_pattern if dist_pattern else None,
        "sorted_shape": sorted_shape if sorted_shape else None,
        "dist_seat": dist_seat,
        "include_categories": bool(show_categories),
    }
    
    with st.spinner("Fetching Bidding Table Statistics from server..."):
        data = api_post("/bidding-table-statistics", payload)
    _st_info_elapsed("Bidding Table Explorer", data)
    
    has_criteria = data.get("has_criteria", False)
    has_aggregates = data.get("has_aggregates", False)
    if not has_aggregates:
        st.warning("Aggregate statistics not available. Run bbo_bt_aggregate.py to generate them.")
    if not has_criteria:
        st.warning("Criteria columns not available. Run bt_criteria_extractor.py to generate them.")
    
    total_matches = data.get("total_matches", 0)
    rows = data.get("rows", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    # Use the pattern returned by API (after server-side normalization) for accurate display
    actual_pattern = data.get("pattern", auction_pattern)
    
    if total_matches == 0:
        st.warning(f"No auctions match pattern: `{actual_pattern}`")
        if data.get("message"):
            st.info(data["message"])
    else:
        st.info(f"Showing {total_matches:,} auctions matching pattern: `{actual_pattern}` in {format_elapsed(elapsed_ms)}")
        
        if rows:
            display_df = pl.DataFrame(rows)

            # If category flags are present, add a compact summary column.
            if show_categories:
                cat_cols = [c for c in display_df.columns if isinstance(c, str) and c.startswith("is_")]
                if cat_cols:
                    try:
                        # For small samples, Python per-row is simplest + fast enough.
                        dcts = display_df.select(["row_idx"] + cat_cols).to_dicts()
                        summaries: list[str] = []
                        for r in dcts:
                            trues = [c for c in cat_cols if bool(r.get(c)) is True]
                            summaries.append(", ".join(trues[:25]) + ("..." if len(trues) > 25 else ""))
                        display_df = display_df.with_columns(pl.Series("Categories_True", summaries))
                    except Exception:
                        pass
            
            dist_sql_query = data.get("dist_sql_query")
            if dist_sql_query:
                st.info(f"📐 Distribution filter applied (Seat {dist_seat})")
                with st.expander("🔍 Distribution SQL Query", expanded=False):
                    st.code(dist_sql_query, language="sql")
            
            key_cols = ["row_idx", "original_idx", "Auction"]
            if "matching_deal_count" in display_df.columns:
                key_cols.append("matching_deal_count")
            
            other_cols = [c for c in display_df.columns if c not in key_cols]
            display_df = display_df.select([c for c in key_cols if c in display_df.columns] + sorted(other_cols))
            
            st.subheader("Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows Displayed", len(display_df))
            with col2:
                st.metric("Total Matches", f"{total_matches:,}")
            with col3:
                if "matching_deal_count" in display_df.columns:
                    avg_matches = display_df["matching_deal_count"].mean()
                    st.metric("Avg Matching Deals", f"{avg_matches:,.0f}" if avg_matches else "N/A")
                else:
                    st.metric("Aggregates", "N/A")
            with col4:
                st.metric("Columns", len(display_df.columns))
            
            st.subheader("Bidding Table with Statistics")
            display_df = order_columns(display_df, priority_cols=[
                "row_idx", "original_idx", "Auction", "matching_deal_count",
                "Categories_True",
                "Expr", "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
                "HCP_min_S1", "HCP_max_S1", "HCP_min_S2", "HCP_max_S2",
                "HCP_min_S3", "HCP_max_S3", "HCP_min_S4", "HCP_max_S4",
            ])
            # Let render_aggrid pick a dynamic height based on row count instead of forcing 500px.
            render_aggrid(display_df, key="bt_stats_grid", table_name="bidding_table_stats")
            
            with st.expander("Column Descriptions"):
                st.markdown("""
                **Key Columns**:
                - `row_idx`: Display row index
                - `original_idx`: Original row index in source files
                - `Auction`: The auction sequence
                - `matching_deal_count`: Number of deals matching the criteria
                
                **Criteria Columns** (from bt_criteria.parquet):
                - `{col}_min_S{seat}`: Minimum value constraint
                - `{col}_max_S{seat}`: Maximum value constraint
                
                **Aggregate Statistics** (from bbo_bt_aggregate.parquet):
                - `{col}_mean_S{seat}`: Mean value across matching deals
                - `{col}_std_S{seat}`: Standard deviation
                - `{col}_min_S{seat}`, `{col}_max_S{seat}`: Min/Max observed values
                
                **Columns**: HCP, SL_C, SL_D, SL_H, SL_S, Total_Points (for each of 4 seats)
                """)


def render_analyze_deal():
    """Analyze a PBN/LIN deal and find matching auctions."""
    st.header("Auction AI - Analyze Deal and Find Matching Auctions")
    st.caption("Enter a PBN/LIN deal string, file path, or URL to analyze and find matching auctions.")
    
    PBN_EXAMPLES = {
        "Custom": "",
        "PBN String (1NT opener)": "N:AK65.KQ2.A54.K32 Q82.JT95.K62.A87 JT97.A843.QJ3.54 43.76.T987.QJT96",
        "PBN String (5-card major)": "S:AKJ97.K82.Q5.A43 Q84.AJ5.KT92.K87 T62.QT943.A76.52 53.76.J843.QJT96",
        "Local LIN file": r"C:\sw\bridge\ML-Contract-Bridge\src\Calculate_PBN_Results\3457345193-1681319161-bsalita.lin",
        "LIN String from BBO": "pn|bsalita,~~M42455,~~M42453,~~M42454|st||md|2S56TKAH28D3TAC3JK,S7JH579JD267KC67T,S24QH36TQAD458C9Q,|rh||ah|Board 8|sv|o|mb|p|mb|p|mb|p|mb|1S|an|Major suit opening -- 5+ !S; 11-21 HCP; 12-22 total points|mb|p|mb|2C!|an|Drury -- 3+ !S; 11- HCP; 10-12 total points |mb|p|mb|2D|an|Invite to game -- 5+ !S; 13-14 total points|mb|p|mb|2S|an|3+ !S; 10-11 total points |mb|p|mb|3C|an|3+ !C; 5+ !S; Q+ in !C; 14 total points; forcing to 3S|mb|p|mb|3H|an|3+ !H; 3+ !S; Q+ in !H; 10 total points; forcing |mb|p|mb|3S|an|3+ !C; 5+ !S; Q+ in !C; 14 total points|mb|p|mb|p|mb|p|pg||pc|H5|pc|HT|pc|HK|pc|H2|pg||pc|CA|pc|C3|pc|C6|pc|C9|pg||pc|DQ|pc|DA|pc|D7|pc|D4|pg||pc|S5|pc|S7|pc|SQ|pc|S9|pg||pc|S2|pc|S8|pc|SK|pc|SJ|pg||pc|SA|pc|D6|pc|S4|pc|S3|pg||pc|CK|pc|C7|pc|CQ|pc|C8|pg||pc|CJ|pc|CT|pc|D5|pc|C4|pg||pc|H8|pc|HJ|pc|HA|pc|H4|pg||pc|HQ|pc|D9|pc|D3|pc|H9|pg||pc|H6|pc|C5|pc|DT|pc|H7|pg||pc|DK|pc|D8|pc|DJ|pc|S6|pg||pc|ST|pc|D2|pc|H3|pc|C2|pg||",
        "GitHub PBN raw URL 1N": "https://raw.githubusercontent.com/ADavidBailey/Practice-Bidding-Scenarios/refs/heads/main/pbn/1N.pbn",
        "GitHub PBN raw URL GIB 1N": "https://raw.githubusercontent.com/ADavidBailey/Practice-Bidding-Scenarios/refs/heads/main/pbn/GIB_1N.pbn",
    }
    
    selected_example = st.sidebar.selectbox(
        "Select Example",
        options=list(PBN_EXAMPLES.keys()),
        index=0,
        help="Choose a pre-defined example or 'Custom' to enter your own"
    )
    
    if selected_example == "Custom":
        pbn_input = st.sidebar.text_area(
            "PBN/LIN Input",
            value="",
            height=100,
            placeholder="Enter PBN/LIN string, file path, or URL",
            help="Auto-detects: PBN/LIN deal string, local file path (.pbn/.lin), or URL"
        )
    else:
        pbn_input = st.sidebar.text_area(
            "PBN/LIN Input",
            value=PBN_EXAMPLES[selected_example],
            height=100,
            help="Auto-detects: PBN/LIN deal string, local file path (.pbn/.lin), or URL"
        )
    
    match_seat = st.sidebar.selectbox("Match Auction Seat", [1, 2, 3, 4], index=0,
        help="Which seat's criteria to match against the deal")
    
    max_auctions = st.sidebar.number_input("Max Auctions to Show", value=20, min_value=1, max_value=500)
    
    include_par = st.sidebar.checkbox("Calculate Par Score", value=True,
        help="Calculate par score using double-dummy analysis (server-side)")
    
    vul_option = st.sidebar.selectbox("Vulnerability", ["None", "Both", "NS", "EW"], index=0,
        help="Vulnerability for par score calculation")
    
    if not pbn_input:
        st.info("Select an example or enter a PBN/LIN string, file path, or URL to analyze.")
        return
    
    with st.spinner("Processing PBN/LIN deal(s)..."):
        try:
            pbn_payload = {
                "pbn": pbn_input,
                "include_par": include_par,
                "vul": vul_option,
            }
            pbn_data = api_post("/process-pbn", pbn_payload)
            _st_info_elapsed("Analyze Deal (parse PBN/LIN)", pbn_data)
        except Exception as e:
            st.error(f"Failed to process PBN/LIN: {e}")
            return
    
    deals = pbn_data.get("deals", [])
    if not deals:
        st.warning("No valid deals found.")
        return
    
    input_type = pbn_data.get("input_type", "unknown")
    input_source = pbn_data.get("input_source", "")
    type_emoji = {"LIN string": "📝", "PBN string": "📝", "LIN file": "📁", "PBN file": "📁", "LIN URL": "🌐", "PBN URL": "🌐"}.get(input_type, "❓")
    st.info(f"{type_emoji} Detected **{input_type}** — Showing {len(deals)} parsed deal(s) in {format_elapsed(pbn_data.get('elapsed_ms', 0))}")
    if input_source and len(input_source) < 200:
        st.caption(f"Source: `{input_source}`")
    
    if len(deals) > 1:
        progress_bar = st.progress(0, text="Finding matching auctions for each deal...")
    else:
        progress_bar = None
    
    for deal_idx, deal in enumerate(deals):
        pbn = deal.get("pbn")
        par_score = deal.get("Par_Score")
        par_contracts = deal.get("Par_Contract", "")
        
        # Extract hands from flat keys (Hand_N, Hand_E, etc.)
        hands = {d: deal.get(f"Hand_{d}", "") for d in "NESW"}
        
        # Use subheader instead of expander to avoid AgGrid rendering issues
        label = f"Deal {deal_idx + 1}: {pbn[:50]}..." if len(pbn) > 50 else f"Deal {deal_idx + 1}: {pbn}"
        st.subheader(label)
        
        if par_score is not None:
            st.info(f"🎯 **Par Score**: {par_score} ({par_contracts}) | Vul: {vul_option}")
        
        st.write("**Hands:**")
        hands_row = {f"Hand_{d}": hands.get(d, "") for d in "NESW"}
        render_aggrid(pl.DataFrame([hands_row]), key=f"hands_deal_{deal_idx}", height=80, table_name="hands")
        
        # Extract hand stats from flat keys (HCP_N, SL_S_N, etc.)
        stats_rows = []
        for d in "NESW":
            hcp = deal.get(f"HCP_{d}")
            if hcp is not None:
                row = {
                    "Dir": d,
                    "HCP": hcp,
                    "SL_S": deal.get(f"SL_S_{d}", 0),
                    "SL_H": deal.get(f"SL_H_{d}", 0),
                    "SL_D": deal.get(f"SL_D_{d}", 0),
                    "SL_C": deal.get(f"SL_C_{d}", 0),
                    "Total_Points": deal.get(f"Total_Points_{d}", hcp),
                }
                stats_rows.append(row)
        if stats_rows:
            st.write("**Hand Statistics:**")
            render_aggrid(pl.DataFrame(stats_rows), key=f"stats_deal_{deal_idx}", height=150, table_name="hand_statistics")
        
        with st.spinner(f"Finding matching auctions for deal {deal_idx + 1}..."):
            # Map seat (1-4) to direction based on dealer
            # Seat 1 = Dealer, Seat 2 = LHO, Seat 3 = Partner, Seat 4 = RHO
            dealer = deal.get("Dealer", "N")
            direction_order = ["N", "E", "S", "W"]
            dealer_idx = direction_order.index(dealer) if dealer in direction_order else 0
            seat_direction = direction_order[(dealer_idx + match_seat - 1) % 4]
            
            # Extract hand statistics for the selected seat's direction
            hcp = deal.get(f"HCP_{seat_direction}", 0)
            sl_s = deal.get(f"SL_S_{seat_direction}", 0)
            sl_h = deal.get(f"SL_H_{seat_direction}", 0)
            sl_d = deal.get(f"SL_D_{seat_direction}", 0)
            sl_c = deal.get(f"SL_C_{seat_direction}", 0)
            total_points = deal.get(f"Total_Points_{seat_direction}", hcp)
            
            match_payload = {
                "hcp": int(hcp) if hcp else 0,
                "sl_s": int(sl_s) if sl_s else 0,
                "sl_h": int(sl_h) if sl_h else 0,
                "sl_d": int(sl_d) if sl_d else 0,
                "sl_c": int(sl_c) if sl_c else 0,
                "total_points": int(total_points) if total_points else 0,
                "seat": match_seat,
                "max_results": int(max_auctions),
            }
            match_data = api_post("/find-matching-auctions", match_payload)
            _st_info_elapsed(f"Find Matching Auctions (deal {deal_idx + 1})", match_data)
        
        matches = match_data.get("matches", [])
        elapsed_ms = match_data.get("elapsed_ms", 0)
        
        if not matches:
            st.warning(f"No matching auctions found for seat {match_seat}.")
        else:
            st.info(f"Showing {len(matches)} matching auction(s) for seat {match_seat} in {format_elapsed(elapsed_ms)}")
        matches_df = pl.DataFrame(matches)
        matches_df = order_columns(matches_df, priority_cols=[
            "Auction", "Rules_Auction", "Expr",
            "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
        ])
        render_aggrid(matches_df, key=f"matches_deal_{deal_idx}", height=calc_grid_height(len(matches)), table_name="auction_matches")
        
        st.divider()
        
        if progress_bar:
            progress_bar.progress((deal_idx + 1) / len(deals), text=f"Processed {deal_idx + 1}/{len(deals)} deals")
    
    if progress_bar:
        progress_bar.empty()


def render_pbn_database_lookup():
    """Check if a PBN deal exists in the database."""
    st.header("PBN Lookup - Find Deal in Database")
    st.caption("Look up a PBN deal string to check if it exists in bbo_mldf_augmented.parquet")
    
    @st.cache_data(ttl=3600)
    def get_sample_pbn():
        try:
            data = api_get("/pbn-sample")
            return data.get("pbn", "")
        except Exception:
            return ""
    
    sample_pbn = get_sample_pbn()
    
    pbn_input = st.sidebar.text_area(
        "PBN Deal String",
        value=sample_pbn,
        height=100,
        placeholder="e.g., N:AK65.KQ2.A54.K32 Q82.JT95.K62.A87 JT97.A843.QJ3.54 43.76.T987.QJT96",
        help="Enter a PBN deal string in format: Dealer:Hand_N Hand_E Hand_S Hand_W"
    )
    
    if not pbn_input:
        st.info("Enter a PBN deal string to look up.")
        return
    
    with st.spinner("Looking up PBN in database..."):
        payload = {"pbn": pbn_input.strip()}
        data = api_post("/pbn-lookup", payload)
    _st_info_elapsed("PBN Database Lookup", data)
    
    # Canonical API schema: {matches, count, total_in_df, pbn_searched, elapsed_ms}
    elapsed = data["elapsed_ms"]
    total = data["total_in_df"]
    matches = data["matches"]
    count = data["count"]
    
    # Parse PBN to extract hands for SQL query generation
    def parse_pbn_for_sql(pbn_str: str) -> dict:
        """Parse PBN string to extract hands for SQL WHERE clause."""
        try:
            parts = pbn_str.strip().split()
            if len(parts) < 4:
                return {}
            first_part = parts[0]
            if ':' in first_part:
                start_dir, first_hand = first_part.split(':', 1)
                hands = [first_hand] + parts[1:4]
            else:
                start_dir = 'N'
                hands = parts[:4]
            
            dirs = ['N', 'E', 'S', 'W']
            start_idx = dirs.index(start_dir.upper()) if start_dir.upper() in dirs else 0
            result = {}
            for i, hand in enumerate(hands):
                dir_idx = (start_idx + i) % 4
                result[f'Hand_{dirs[dir_idx]}'] = hand
            return result
        except Exception:
            return {}
    
    parsed_hands = parse_pbn_for_sql(pbn_input)
    
    if count > 0:
        st.success(f"✅ PBN found in database! (searched {total:,} rows)")
        
        # Use first returned match as the primary deal info
        deal_info = matches[0] if matches and isinstance(matches[0], dict) else {}
        if deal_info:
            st.write("**Deal Information:**")
            
            key_cols = ["PBN", "Vul", "Dealer", "Actual_Auction", "Contract", "Result", "Tricks", "Score", "DD_Score_Declarer", "ParScore"]
            key_info = {k: v for k, v in deal_info.items() if k in key_cols and v is not None}
            if key_info:
                render_aggrid(pl.DataFrame([key_info]), key="pbn_lookup_key", height=80, table_name="deal_key_info")
            
            st.markdown("**Full Deal Data:**")
            all_info_df = pl.DataFrame([deal_info])
            all_info_df = order_columns(all_info_df, priority_cols=[
                "PBN", "Dealer", "Vul", "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                "Actual_Auction", "Contract", "Result", "Tricks", "Score", "ParScore",
                "HCP_N", "HCP_E", "HCP_S", "HCP_W",
            ])
            # Let AgGrid choose a dynamic height for the full-deal info instead of forcing 200px.
            render_aggrid(all_info_df, key="pbn_lookup_full", table_name="deal_full_info")

        # If multiple matches exist, show a small sample list for clarity
        if count > 1:
            st.markdown(f"**All matches ({count:,}):**")
            matches_df = pl.DataFrame(matches)
            render_aggrid(matches_df, key="pbn_lookup_matches", height=250, table_name="pbn_matches")
    else:
        st.warning(f"❌ PBN not found in database (searched {total:,} rows)")
        st.write("**Searched PBN:**")
        st.code(pbn_input)


def render_analyze_actual_auctions():
    """Group deals by their actual auction and analyze outcomes."""
    st.header("Group by Bid - Analyze Deals by Actual Auction")
    st.caption("Group deals from bbo_mldf_augmented by their actual auction sequence (bid column) and show deal characteristics.")
    
    raw_auction_regex = st.sidebar.text_input(
        "Auction Regex",
        value="^1N-p-3N$",
        help="Regex pattern to filter auctions. Trailing '-p-p-p' appended if not present.",
        key="group_by_bid_regex"
    )
    match_all_dealers = st.sidebar.checkbox(
        "Match all 4 dealer positions",
        value=False,
        help=MATCH_ALL_DEALERS_HELP,
        key="match_all_dealers_group",
    )
    auction_regex = normalize_auction_user_text(raw_auction_regex)
    # Prepend (p-)* when matching all dealers
    if match_all_dealers:
        auction_regex = prepend_all_seats_prefix(auction_regex)
        st.sidebar.caption(f"→ {auction_regex} (4 seat variants)")
    elif auction_regex != raw_auction_regex:
        st.sidebar.caption(f"→ {auction_regex}")
    
    max_groups = st.sidebar.number_input("Max Auction Groups", value=10, min_value=1, max_value=100)
    deals_per_group = st.sidebar.number_input("Deals per Group", value=100, min_value=1, max_value=1000)
    
    show_blocking_criteria = st.sidebar.checkbox(
        "🔍 Show Blocking Criteria",
        value=False,
        help="For each deal, evaluate which criteria from the BT row are blocking. Adds columns showing failed/untracked criteria per deal."
    )
    
    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_group"))

    payload = {
        "auction_pattern": auction_regex,
        "n_auction_groups": int(max_groups),
        "n_deals_per_group": int(deals_per_group),
        "seed": seed,
    }
    
    with st.spinner("Grouping deals by bid..."):
        try:
            data = api_post("/group-by-bid", payload)
            _st_info_elapsed("Analyze Actual Auctions", data)
        except requests.HTTPError as e:
            # Handle invalid/malformed regex with a friendly message
            if "Invalid regex pattern" in str(e) or "regular expression" in str(e):
                st.error("Malformed Regex")
            else:
                st.error(f"API error: {e}")
            return
        except Exception as e:
            st.error(f"Error loading auctions: {e}")
            return
    
    # Canonical API schema: {pattern, auction_groups, total_matching_deals, unique_auctions, elapsed_ms}
    if not isinstance(data, dict):
        st.error("Unexpected response from server")
        return

    detail = data.get("detail") or data.get("error")
    if detail and ("regex" in str(detail).lower() or "regular expression" in str(detail).lower()):
        st.error("Malformed Regex")
        return

    groups = data.get("auction_groups")
    elapsed_ms = data.get("elapsed_ms", 0)
    total_auctions = data.get("unique_auctions", 0)
    total_matching_deals = data.get("total_matching_deals", 0)
    effective_pattern = data.get("pattern", auction_regex)

    if groups is None:
        st.error("Unexpected response from server")
        return
    
    if not groups:
        st.warning(f"No auctions matched pattern: `{effective_pattern}`")
        return
    
    deals_msg = f", {total_matching_deals:,} deals" if isinstance(total_matching_deals, int) and total_matching_deals >= 0 else ""
    st.info(f"Showing {total_auctions:,} matching auctions{deals_msg}, showing {len(groups)} groups in {format_elapsed(elapsed_ms)}")
    
    for i, group in enumerate(groups):
        bid_auction = group.get("auction")
        bt_auction = group.get("bt_auction")
        deal_count = group.get("deal_count", 0)
        sample_count = group.get("sample_count", 0)
        bt_info = group.get("bt_info")
        stats = group.get("stats", {})
        deals = group.get("deals", [])
        
        # Use subheader instead of expander to avoid AgGrid rendering issues
        label = f"{bid_auction}"
        if bt_auction and bt_auction != bid_auction:
            label += f" → {bt_auction}"
        label += f" ({deal_count:,} deals, {sample_count} shown)"
        st.subheader(label)
        
        if bt_info:
            total_criteria = 0
            for s in range(1, 5):
                agg_col = f"Agg_Expr_Seat_{s}"
                if agg_col in bt_info and bt_info[agg_col]:
                    total_criteria += len(bt_info[agg_col])
            
            # Keep this as expander since it's just text, not AgGrid
            with st.expander(f"📋 Bidding Table Criteria ({total_criteria} total)", expanded=False):
                for s in range(1, 5):
                    agg_col = f"Agg_Expr_Seat_{s}"
                    role = SEAT_ROLES.get(s, "")
                    if agg_col in bt_info and bt_info[agg_col]:
                        criteria_str = ", ".join(str(x) for x in bt_info[agg_col])
                        st.write(f"**Seat_{s}** ({role}): {criteria_str}")
                    else:
                        st.write(f"**Seat_{s}** ({role}): *(no criteria)*")
                
                expr = bt_info.get("Expr")
                if expr:
                    if isinstance(expr, list):
                        expr_str = ", ".join(str(x) for x in expr if x)
                    else:
                        expr_str = str(expr)
                    if expr_str:
                        st.write(f"**Expr:** {expr_str}")
            
        if stats:
            st.markdown("**Statistics**")

            # Direction summary table (easy to scan across N/E/S/W)
            dir_rows: list[dict[str, Any]] = []
            for d in "NESW":
                hcp_avg = stats.get(f"HCP_{d}_avg")
                hcp_min = stats.get(f"HCP_{d}_min")
                hcp_max = stats.get(f"HCP_{d}_max")
                total_points_avg = stats.get(f"TP_{d}_avg")

                if all(v is None for v in [hcp_avg, hcp_min, hcp_max, total_points_avg]):
                    continue

                hcp_range = None
                if hcp_min is not None and hcp_max is not None:
                    hcp_range = f"{hcp_min}-{hcp_max}"

                dir_rows.append(
                    {
                        "Dir": d,
                        "HCP_avg": hcp_avg,
                        "HCP_range": hcp_range,
                        "Total_Points_avg": total_points_avg,
                    }
                )

            if dir_rows:
                dir_df = pl.DataFrame(dir_rows)
                # Stable ordering + nicer formatting
                if "Dir" in dir_df.columns:
                    dir_df = dir_df.with_columns(
                        pl.col("Dir").cast(pl.Utf8)
                    )
                if "HCP_avg" in dir_df.columns:
                    dir_df = dir_df.with_columns(pl.col("HCP_avg").cast(pl.Float32).round(1))
                if "Total_Points_avg" in dir_df.columns:
                    dir_df = dir_df.with_columns(pl.col("Total_Points_avg").cast(pl.Float32).round(1))
                render_aggrid(dir_df, key=f"group_by_bid_stats_{i}", height=160, table_name="bid_direction_stats")

            # Score summary (separate row, avoids mixing with per-direction stats)
            score_delta_avg = stats.get("Score_Delta_Avg")
            score_delta_std = stats.get("Score_Delta_StdDev")
            score_mp_avg = stats.get("Score_MP_Avg")
            score_mp_pct_avg = stats.get("Score_MP_Pct_Avg")

            score_rows: list[dict[str, Any]] = []
            if score_delta_avg is not None:
                # Server definition: Score_Delta = Score - ParScore
                avg = float(score_delta_avg)
                sd = float(score_delta_std) if score_delta_std is not None else None
                score_rows.append(
                    {
                        "Metric": "Score − ParScore (avg ± sd)",
                        "Avg": avg,
                        "StdDev": sd,
                        "Notes": "Computed per deal as Score - ParScore (positive = above par; negative = below par).",
                    }
                )
            if score_mp_avg is not None:
                score_rows.append(
                    {
                        "Metric": "MP (avg)",
                        "Avg": float(score_mp_avg),
                        "StdDev": None,
                        "Notes": None,
                    }
                )
            if score_mp_pct_avg is not None:
                score_rows.append(
                    {
                        "Metric": "MP% (avg)",
                        "Avg": float(score_mp_pct_avg),
                        "StdDev": None,
                        "Notes": None,
                    }
                )

            if score_rows:
                score_df = pl.DataFrame(score_rows)
                render_aggrid(score_df, key=f"group_by_bid_score_stats_{i}", height=140, table_name="score_stats")
            
        if deals:
            st.write(f"**Sample Deals** ({len(deals)} shown):")
            deals_df = pl.DataFrame(deals)
            
            # Add blocking criteria columns if enabled
            if show_blocking_criteria and bt_info:
                # Build checks from bt_info criteria
                checks = []
                for seat in [1, 2, 3, 4]:
                    crits = bt_info.get(f"Agg_Expr_Seat_{seat}", [])
                    if crits:
                        checks.append({"seat": seat, "criteria": crits})
                
                if checks:
                    blocking_data = []
                    with st.spinner("Evaluating criteria..."):
                        t_eval0 = time.perf_counter()
                        for deal in deals:
                            deal_row_idx = deal.get("_row_idx")
                            deal_index = deal.get("index")
                            dealer = deal.get("Dealer", "N")
                            
                            try:
                                eval_data = api_post(
                                    "/deal-criteria-eval-batch",
                                    {
                                        "deal_row_idx": int(deal_row_idx if deal_row_idx is not None else 0),
                                        "deal_index": (int(deal_index) if deal_row_idx is None and deal_index is not None else None),
                                        "dealer": str(dealer),
                                        "checks": checks
                                    },
                                    timeout=30,
                                )
                                
                                failed_list = []
                                untracked_list = []
                                for result in eval_data.get("results", []):
                                    seat = result.get("seat")
                                    for f in result.get("failed", []):
                                        failed_list.append(f"S{seat}:{f}")
                                    for u in result.get("untracked", []):
                                        untracked_list.append(f"S{seat}:{u}")
                                
                                blocking_data.append({
                                    "_idx": deal.get("index", deal_row_idx),
                                    "Failed_Criteria": ", ".join(failed_list[:5]) + ("..." if len(failed_list) > 5 else "") if failed_list else "",
                                    "Untracked_Criteria": ", ".join(untracked_list[:3]) + ("..." if len(untracked_list) > 3 else "") if untracked_list else "",
                                    "Fail_Count": len(failed_list),
                                    "Untracked_Count": len(untracked_list),
                                })
                            except Exception:
                                blocking_data.append({
                                    "_idx": deal.get("index", deal_row_idx),
                                    "Failed_Criteria": "(error)",
                                    "Untracked_Criteria": "",
                                    "Fail_Count": -1,
                                    "Untracked_Count": 0,
                                })
                        elapsed_eval_s = time.perf_counter() - t_eval0
                        if elapsed_eval_s >= 0.5:
                            st.info(f"⏱️ Blocking criteria evaluation completed in {elapsed_eval_s:.2f}s")
                    
                    # Merge blocking data into deals_df
                    if blocking_data:
                        blocking_df = pl.DataFrame(blocking_data)
                        if "_idx" in blocking_df.columns and "index" in deals_df.columns:
                            deals_df = deals_df.join(
                                blocking_df.rename({"_idx": "index"}),
                                on="index",
                                how="left"
                            )
            
            deals_df = order_columns(deals_df, priority_cols=[
                "PBN",
                "index", "Dealer", "Vul", "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                "Failed_Criteria", "Untracked_Criteria", "Fail_Count", "Untracked_Count",
                "Contract", "Result", "Score", "Score_MP", "Score_MP_Pct",
                "DD_Score_Declarer", "ParScore",
                "HCP_N", "HCP_E", "HCP_S", "HCP_W",
                "Total_Points_N", "Total_Points_E", "Total_Points_S", "Total_Points_W",
            ], drop_cols=["Auction", "Board_ID"])
            render_aggrid(deals_df, key=f"group_by_bid_deals_{i}", height=calc_grid_height(len(deals)), table_name="bid_group_deals")
        
        st.divider()


def render_bt_seat_stats_tool():
    """Compute on-the-fly seat stats for a single bt_seat1 row using criteria bitmaps."""
    st.header("BT Seat Stats (On-the-fly)")
    st.caption(
        "Compute HCP / suit-length / total-points ranges per seat directly from deals, "
        "using the bidding table's criteria bitmaps. "
        "Enter a `bt_index` from bt_seat1 (shown in the Bidding Table Explorer), or use a random sample row."
    )

    bt_index = st.sidebar.number_input(
        "bt_index (from bt_seat1)",
        value=0,
        min_value=0,
        step=1,
        help="Copy bt_index from the Bidding Table Explorer grid.",
    )

    seat_option = st.sidebar.selectbox(
        "Seat",
        ["All (1-4)", "1", "2", "3", "4"],
        index=0,
        help="Seat 1=Dealer, 2=LHO, 3=Partner, 4=RHO",
    )
    max_deals = st.sidebar.number_input(
        "Max Deals (0 = all)",
        value=0,
        min_value=0,
        help="Optional cap on the number of deals to aggregate (currently exact stats; cap reserved for future use).",
    )

    # Random seed (used when bt_index == 0 to pick a random bt row).
    seed_bt_stats = int(
        st.sidebar.number_input(
            "Random Seed (0=random)",
            value=0,
            min_value=0,
            key="bt_seat_stats_seed",
        )
    )

    # Always show basic instructions.
    st.info(
        "Use the sidebar to choose a `bt_index` (from the Bidding Table Explorer) and seat(s). "
        "Seat 1=Dealer, Seat 2=LHO, Seat 3=Partner, Seat 4=RHO. "
        "If `bt_index` is 0, a random completed auction row will be sampled (seed-controlled)."
    )

    if bt_index < 0:
        st.error("bt_index must be non-negative.")
        return

    seat = 0 if seat_option.startswith("All") else int(seat_option)

    # If bt_index is 0, fetch a random completed-auction bt_index from the API.
    if bt_index == 0:
        try:
            with st.spinner("Sampling a random bt_index from the bidding table..."):
                sample_payload = {
                    "auction_pattern": ".*",
                    "allow_initial_passes": False,
                    "sample_size": 1,
                    "min_matches": 0,
                    "seed": seed_bt_stats,  # 0 => non-deterministic each call
                    "dist_pattern": None,
                    "sorted_shape": None,
                    "dist_seat": 1,
                }
                sample_data = api_post("/bidding-table-statistics", sample_payload)
                _st_info_elapsed("BT Seat Stats (sample bt_index)", sample_data)
                rows = sample_data.get("rows", [])
                if not rows:
                    st.warning(
                        "Could not find a sample bt_index automatically. "
                        "Please enter a bt_index from the Bidding Table Explorer in the sidebar."
                    )
                    return
                row0 = rows[0]
                # Use bt_index column (REQUIRED)
                effective_bt_index = int(row0["bt_index"])
        except Exception as e:
            st.error(f"Failed to sample a bt_index from bidding table: {e}")
            return
    else:
        effective_bt_index = int(bt_index)
    payload = {
        "bt_index": effective_bt_index,
        "seat": int(seat),
        "max_deals": int(max_deals),
    }

    with st.spinner("Computing on-the-fly stats from deals..."):
        try:
            data = api_post("/bt-seat-stats", payload)
            _st_info_elapsed("BT Seat Stats (On-the-fly)", data)
        except Exception as e:
            st.error(f"Failed to compute stats: {e}")
            return

    seats = data.get("seats", {})
    if not seats:
        st.info("No matching deals found for this bt_index / seat combination.")
        return

    # Pivot: one row per seat, with metrics expanded into columns such as
    # HCP_min, HCP_max, HCP_mean, HCP_std, SL_S_min, ..., Total_Points_std.
    pivot_rows: list[dict[str, object]] = []
    for seat_key, seat_res in seats.items():
        seat_num = int(seat_key)
        mcount = int(seat_res.get("matching_deal_count", 0))
        stats = seat_res.get("stats") or {}
        expr_list = seat_res.get("expr") or []
        if expr_list:
            expr_str = ", ".join(str(e) for e in expr_list)
        else:
            expr_str = "(empty criteria list)"

        row: dict[str, object] = {
            "Auction": data.get("auction"),
            "Seat": seat_num,
            "Expr": expr_str,
            "Deals": mcount,
        }

        # Flatten each metric's stats into columns.
        for metric_name, metric_vals in stats.items():
            prefix = str(metric_name)
            row[f"{prefix}_min"] = metric_vals.get("min")
            row[f"{prefix}_max"] = metric_vals.get("max")
            row[f"{prefix}_mean"] = metric_vals.get("mean")
            row[f"{prefix}_std"] = metric_vals.get("std")

        pivot_rows.append(row)

    if not pivot_rows:
        st.info("No metrics available for the selected bt_index / seat.")
        return

    df_stats = pl.DataFrame(pivot_rows)
    df_stats = order_columns(
        df_stats,
        priority_cols=["Auction", "Seat", "Expr", "Deals"],
    )

    # Show the auction context above the stats table (slightly emphasized).
    auction_str = data.get("auction") or "(unknown auction)"
    bt_idx_val = data.get("bt_index")
    if bt_idx_val is not None:
        st.markdown(f"**Auction: {auction_str} (bt_index={bt_idx_val})**")
    else:
        st.markdown(f"**Auction: {auction_str}**")

    st.subheader("Seat Stats")
    render_aggrid(
        df_stats,
        key="bt_seat_stats_grid",
        height=calc_grid_height(len(df_stats)),
        table_name="bt_seat_stats",
    )


# ---------------------------------------------------------------------------
# Auction Criteria Debugger – Why is a specific auction being rejected?
# ---------------------------------------------------------------------------

def render_auction_criteria_debugger():
    """Debug why a specific auction is being rejected as a Rules candidate.
    
    Shows deals matching an auction pattern, the Rules auction, and which criteria
    from the target auction's BT row are blocking it from being selected.
    """
    st.header("🔍 Auction Criteria Debugger")
    st.markdown("""
    Enter an auction pattern (e.g., `1S-p-p-p`) to see why that auction isn't being selected 
    by the Rules model for deals where it's the actual auction.
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        auction_pattern = st.text_input(
            "Target Auction Pattern",
            value="1S-p-p-p",
            help="The auction you want to debug (e.g., 1S-p-p-p, 2H-p-p-p, 5N-p-p-p)",
        )
    with col2:
        sample_size = st.number_input(
            "Sample Size",
            min_value=1,
            max_value=10000,
            value=25,
            help="Number of deals to analyze",
        )

    seed = st.number_input("Random Seed", min_value=0, value=42, help="For reproducibility")
    
    if not auction_pattern:
        st.info("Enter an auction pattern to begin.")
        return
    
    with st.spinner(f"Analyzing why '{auction_pattern}' is rejected..."):
        try:
                # Step 1: Get the BT row for this auction pattern
                st.subheader(f"1️⃣ BT Row for '{auction_pattern}'")
                
                # Normalize pattern for exact match
                pattern_exact = f"^{auction_pattern.upper()}$"
                bt_data = api_post("/auction-sequences-matching", {"pattern": pattern_exact, "n_samples": 1, "seed": int(seed)}, timeout=30)
                _st_info_elapsed("Auction Criteria Debugger: BT lookup", bt_data)
                
                bt_samples = bt_data.get("samples", [])
                if not bt_samples:
                    st.warning(f"No BT row found for auction '{auction_pattern}'. Check the pattern.")
                    return
                
                bt_sample = bt_samples[0]
                sequence = bt_sample.get("sequence", [])
                
                # Find the target row (the final auction row)
                target_row = None
                for row in sequence:
                    if row.get("Auction", "").upper() == auction_pattern.upper():
                        target_row = row
                        break
                if not target_row and sequence:
                    target_row = sequence[0]  # Use first row with aggregated criteria
                
                if not target_row:
                    st.warning("Could not find target BT row in sequence.")
                    return
                
                bt_index = target_row.get("index")
                st.info(f"Showing BT row: **bt_index={bt_index}**, Auction=**{target_row.get('Auction')}** in {format_elapsed(bt_data.get('elapsed_ms', 0))}")
                
                # Show criteria per seat as a DataFrame (BT row: raw/overlay)
                criteria_by_seat = {}
                seat_rows = []
                for seat in [1, 2, 3, 4]:
                    criteria = target_row.get(f"Agg_Expr_Seat_{seat}", [])
                    criteria_by_seat[seat] = criteria if criteria else []
                    criteria_str = "; ".join(criteria[:10]) + ("..." if len(criteria) > 10 else "") if criteria else "(none)"
                    seat_rows.append({
                        "Seat": seat,
                        "Count": len(criteria) if criteria else 0,
                        "Criteria": criteria_str,
                    })
                
                # If present, add Rules model's seat-1 criteria (merged+overlay) for this bt_index.
                rules_s1 = target_row.get("Agg_Expr_Seat_1_Rules")
                if rules_s1 is not None:
                    rules_str = "; ".join(rules_s1[:10]) + ("..." if len(rules_s1) > 10 else "") if rules_s1 else "(none)"
                    seat_rows[0]["Rules_S1_Criteria"] = rules_str
                    seat_rows[0]["Rules_S1_Count"] = len(rules_s1) if rules_s1 else 0
                
                seat_df = pl.DataFrame(seat_rows)
                render_aggrid(seat_df, key="auction_debugger_seats", height=calc_grid_height(len(seat_df)), table_name="auction_debugger_seats")
                
                if not any(criteria_by_seat.values()):
                    st.info("No criteria on any seat for this BT row.")
                
                st.divider()
                
                # Step 2: Get deals where actual auction matches this pattern
                st.subheader(f"2️⃣ Deals with Actual Auction Matching '{auction_pattern}'")
                
                arena_data = api_post(
                    "/bidding-arena",
                    {
                        "model_a": "Actual",
                        "model_b": "Rules",
                        "sample_size": int(sample_size),
                        "seed": int(seed),
                        "auction_pattern": f"^{auction_pattern.upper()}",
                    },
                    timeout=120,
                )
                _st_info_elapsed("Auction Criteria Debugger: sample deals", arena_data)
                
                sample_deals = arena_data.get("sample_deals", [])
                if not sample_deals:
                    st.warning(f"No deals found with actual auction matching '{auction_pattern}'")
                    return
                
                st.info(f"Showing **{len(sample_deals)}** sample deals in {format_elapsed(arena_data.get('elapsed_ms', 0))}")
                
                st.divider()
                
                # Step 3: Evaluate criteria for each deal
                st.subheader("3️⃣ Criteria Failures Analysis")
                
                results_data = []
                checks = [{"seat": seat, "criteria": crits} for seat, crits in criteria_by_seat.items()]
                
                progress_bar = st.progress(0)
                t_eval0 = time.perf_counter()
                for i, deal in enumerate(sample_deals):
                    progress_bar.progress((i + 1) / len(sample_deals))
                    
                    deal_row_idx = deal.get("_row_idx")
                    deal_index = deal.get("index")
                    dealer = deal.get("Dealer", "N")
                    actual_auction = deal.get("Auction_Actual", "")
                    rules_auction = deal.get("Auction_Rules", "")
                    
                    # Evaluate criteria
                    eval_data = api_post(
                        "/deal-criteria-eval-batch",
                        {
                            "deal_row_idx": int(deal_row_idx if deal_row_idx is not None else 0),
                            "deal_index": (int(deal_index) if deal_row_idx is None and deal_index is not None else None),
                            "dealer": str(dealer),
                            "checks": checks
                        },
                        timeout=30,
                    )
                    
                    failed_criteria = []
                    untracked_criteria = []
                    
                    for result in eval_data.get("results", []):
                        seat = result.get("seat")
                        seat_dir = result.get("seat_dir", "?")
                        failed = result.get("failed", [])
                        untracked = result.get("untracked", [])
                        if failed:
                            failed_criteria.extend([f"S{seat}({seat_dir}):{c}" for c in failed])
                        if untracked:
                            untracked_criteria.extend([f"S{seat}({seat_dir}):{c}" for c in untracked])
                    
                    # Combine failures
                    all_blocking = failed_criteria + [f"{c} (CSV)" for c in untracked_criteria]
                    
                    results_data.append({
                        "Dealer": dealer,
                        "Vul": deal.get("Vul", ""),
                        "Hand_N": deal.get("Hand_N", ""),
                        "Hand_E": deal.get("Hand_E", ""),
                        "Hand_S": deal.get("Hand_S", ""),
                        "Hand_W": deal.get("Hand_W", ""),
                        "Actual_Auction": actual_auction,
                        # If no blocking criteria and rules_auction is empty, rules agrees with actual
                        "Rules_Auction": rules_auction if rules_auction else (actual_auction if not all_blocking else "no match"),
                        "Blocking_Criteria": ", ".join(all_blocking[:5]) + ("..." if len(all_blocking) > 5 else "") if all_blocking else "",
                        "Failed_Count": len(failed_criteria),
                        "Untracked_Count": len(untracked_criteria),
                        "Result": deal.get("Result", ""),
                        "Score": deal.get("Score", ""),
                        "ParScore": deal.get("ParScore", ""),
                    })
                
                progress_bar.empty()
                elapsed_eval = time.perf_counter() - t_eval0
                if elapsed_eval > 0.5:
                    st.info(f"⏱️ Criteria evaluation completed in {elapsed_eval:.2f}s")
                
                # Display results table
                if results_data:
                    results_df = pl.DataFrame(results_data)
                    
                    # Show SQL query equivalent
                    criteria_list = []
                    for seat, crits in criteria_by_seat.items():
                        for c in crits[:3]:
                            criteria_list.append(f"S{seat}:{c}")
                    criteria_preview = ", ".join(criteria_list[:5]) + ("..." if len(criteria_list) > 5 else "")
                    
                    sql_query = f"""-- Conceptual SQL for Auction Criteria Debugger
-- Target auction: {auction_pattern}
-- Criteria checked: {criteria_preview}

SELECT 
    d.Dealer,
    d.Vul,
    d.Hand_N, d.Hand_E, d.Hand_S, d.Hand_W,
    d.bid AS Actual_Auction,
    rules.matched_auction AS Rules_Auction,
    CASE 
        WHEN NOT check_criteria(d, bt.criteria) THEN get_failed_criteria(d, bt.criteria)
        ELSE NULL 
    END AS Blocking_Criteria
FROM deals d
CROSS JOIN bt_rows bt
LEFT JOIN rules_matches rules ON d.id = rules.deal_id
WHERE d.bid = '{auction_pattern}'
  AND bt.auction = '{auction_pattern}'
ORDER BY d.id
LIMIT {sample_size};"""
                    
                    with st.expander("📝 SQL Query (Conceptual)", expanded=False):
                        st.code(sql_query, language="sql")
                    
                    st.markdown(f"""
                    **Legend:**
                    - `S1:`, `S2:`, etc. = Seat number where criterion failed
                    - `(CSV)` = Criterion from CSV overlay (untracked in bitmap)
                    - `no match` = Rules model found no matching auction
                    - Empty = All criteria passed (target auction should match)
                    """)
                    
                    render_aggrid(results_df, key="criteria_debug_results", height=calc_grid_height(len(results_df)))
                    
                    # Summary statistics
                    st.subheader("📊 Summary")
                    total = len(results_data)
                    rules_none = sum(1 for r in results_data if r["Rules_Auction"] == "no match")
                    rules_match = sum(1 for r in results_data if r["Rules_Auction"].upper() == auction_pattern.upper())
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Deals", total)
                    col2.metric("Rules = None", rules_none, help="Deals where Rules model found no match")
                    col3.metric(f"Rules = {auction_pattern}", rules_match, help="Deals where Rules returned the target auction")
                    
                    # Most common blocking criteria
                    from collections import Counter
                    all_failed = []
                    for r in results_data:
                        blocking = r["Blocking_Criteria"]
                        if blocking:
                            for c in blocking.replace("...", "").split(", "):
                                if c.strip():
                                    all_failed.append(c.strip())
                    
                    if all_failed:
                        st.subheader("🚫 Most Common Blocking Criteria")
                        counter = Counter(all_failed)
                        common_df = pl.DataFrame([
                            {"Criterion": k, "Count": v, "Pct": f"{v/total*100:.1f}%"} 
                            for k, v in counter.most_common(10)
                        ])
                        render_aggrid(common_df, key="common_blocking", height=calc_grid_height(len(common_df), max_height=300))
                
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Bidding Arena – Head-to-head model comparison
# ---------------------------------------------------------------------------

def render_bidding_arena():
    """Render the Bidding Arena: head-to-head model comparison between bidding models."""
    st.header("🏟️ Bidding Arena")
    st.markdown("Compare bidding models head-to-head with DD scores, EV, and IMP differentials.")
    
    # Get available models
    try:
        models_data = api_get("/bidding-models", timeout=30)
        _st_info_elapsed("Load bidding models", models_data)
        model_names = [m["name"] for m in models_data.get("models", [])]
    except Exception as e:
        st.error(f"Failed to get models: {e}")
        model_names = ["Rules_Base", "Actual"]

    # Ensure baseline models are always present (even if /bidding-models omits them)
    for base_model in ("Actual", "Rules_Base"):
        if base_model not in model_names:
            model_names.append(base_model)

    # Add Rules variations (learned criteria) only if the server reports merged_rules loaded
    try:
        status_resp = api_get("/status", timeout=10)
        _st_info_elapsed("Load server status", status_resp)
        loaded_files = status_resp.get("loaded_files") or {}
        merged_ok = ("merged_rules" in loaded_files) and (loaded_files.get("merged_rules") not in (None, 0, "0"))
        if merged_ok:
            # Add all rules pipeline stages
            for rules_model in ("Rules", "Rules_Learned", "Rules_Base"):
                if rules_model not in model_names:
                    model_names.append(rules_model)
    except Exception:
        # If status can't be fetched, don't advertise Rules (learned).
        pass

    # Stable ordering for UI - show pipeline stages in order
    preferred_order = ["Actual", "Rules", "Rules_Learned", "Rules_Base"]
    model_names = [m for m in preferred_order if m in model_names] + sorted([m for m in model_names if m not in preferred_order])
    
    # Model selection - default to Actual vs Rules
    col1, col2 = st.columns(2)
    with col1:
        default_a = model_names.index("Actual") if "Actual" in model_names else 0
        model_a = st.selectbox("Model A", model_names, index=default_a)
    with col2:
        default_b = model_names.index("Rules") if "Rules" in model_names else (1 if len(model_names) > 1 else 0)
        model_b = st.selectbox("Model B", model_names, index=min(default_b, len(model_names) - 1))
    
    # Options
    col3, col4, col5 = st.columns(3)
    with col3:
        sample_size = st.number_input("Sample Size", min_value=10, max_value=10000, value=25, step=100)
    with col4:
        seed = st.number_input("Random Seed", min_value=0, value=42)
    with col5:
        auction_pattern = st.text_input("Auction Pattern (optional)", value="", 
            help="Regex pattern to filter auctions (e.g., '^1N-p-3N')")

    pinned_indexes_raw = st.sidebar.text_area(
        "Pinned deal indexes (optional)",
        value="",
        help="Comma/space/newline-separated deal 'index' values to force-include in Sample Deal Comparisons.",
    )
    pinned_indexes: list[int] = []
    if pinned_indexes_raw.strip():
        tokens = re.split(r"[,\s]+", pinned_indexes_raw.strip())
        for t in tokens:
            if not t:
                continue
            try:
                pinned_indexes.append(int(t))
            except Exception:
                continue

    search_all_bt_rows = st.sidebar.checkbox(
        "Search all completed BT rows (very slow)",
        value=False,
        help=(
            "If enabled, the Rules model will scan *all* completed BT auctions when trying to find a matching rule for each deal. "
            "This can take a very long time; prefer pinned-only mode or a small sample size."
        ),
    )

    if st.sidebar.button(
        "Clear BT drilldown caches",
        help="Clears cached /auction-sequences-* and /deal-criteria-eval-batch results used by the drilldowns.",
    ):
        for k in ("_bt_seq_cache", "_bt_by_index_cache", "_deal_bt_eval_cache", "_deal_fail_cache"):
            if k in st.session_state:
                del st.session_state[k]
        st.sidebar.success("Cleared. Re-run will fetch fresh sequence/eval results.")
        st.rerun()

    if st.sidebar.button(
        "Clear Arena cache",
        help="Clears the cached /bidding-arena response (use this after changing sidebar options).",
    ):
        for k in ("_arena_cache_key", "_arena_cache_data"):
            if k in st.session_state:
                del st.session_state[k]
        st.sidebar.success("Cleared. Re-run will fetch a fresh /bidding-arena response.")
        st.rerun()

    debug_bt_index_raw = st.sidebar.text_input(
        "Debug BT index (optional)",
        value="",
        help="If provided (and you select a deal row below), we'll evaluate this specific bt_index's criteria against the selected deal and show which criteria pass/fail per seat.",
    )
    debug_bt_index: int | None = None
    if debug_bt_index_raw.strip():
        try:
            debug_bt_index = int(debug_bt_index_raw.strip())
        except Exception:
            debug_bt_index = None
    if debug_bt_index is not None:
        st.sidebar.caption(f"Debug BT index is set to {debug_bt_index}. Clear it to enable auto-debug-from-Actual.")

    def _render_debug_bt_index(
        bt_index: int,
        deal_row_idx: int,
        dealer_dir: str,
        deal_index_label: Any,
        key_suffix: str,
    ) -> None:
        """Evaluate a specific bt_index's criteria against a selected deal row."""
        try:
            st.markdown("**Debug BT index vs selected deal**")
            st.caption(f"Deal index: {deal_index_label} (dealer={dealer_dir})")

            bt_payload = {"indices": [int(bt_index)], "allow_initial_passes": False}
            bt_cache = st.session_state.setdefault("_bt_by_index_cache", {})
            bt_key = ("auction-sequences-by-index", int(bt_index))
            if bt_key in bt_cache:
                bt_data = bt_cache[bt_key]
            else:
                bt_data = api_post("/auction-sequences-by-index", bt_payload)
                bt_cache[bt_key] = bt_data
            _st_info_elapsed("Arena Debug: BT sequence lookup", bt_data)

            bt_samples = bt_data.get("samples") or []
            if not bt_samples:
                st.info(f"bt_index {bt_index} not found (or not a completed auction).")
                return

            seq = bt_samples[0].get("sequence") or []
            if not isinstance(seq, list) or not seq:
                st.info(f"bt_index {bt_index}: empty sequence")
                return

            # /auction-sequences-by-index sorts by bt_index, so select the exact row for bt_index if present.
            target_row: dict[str, Any] | None = None
            for r in seq:
                try:
                    if int(r.get("index") or -1) == int(bt_index):
                        target_row = r
                        break
                except Exception:
                    continue
            if target_row is None:
                for r in seq:
                    if r.get("is_match_row") is True:
                        target_row = r
                        break
            if target_row is None:
                target_row = seq[-1]

            # For type-checkers: target_row is always bound at this point.
            assert target_row is not None

            auction_val = target_row.get("Auction")
            st.caption(f"[{key_suffix}] bt_index={bt_index} auction={auction_val}")

            checks = []
            for seat in (1, 2, 3, 4):
                crits = target_row.get(f"Agg_Expr_Seat_{seat}") or []
                crits = _to_list_utf8_cell(crits) or []
                checks.append({"seat": seat, "criteria": crits})

            eval_payload = {
                "deal_row_idx": int(deal_row_idx),
                "dealer": str(dealer_dir or "N").upper(),
                "checks": checks,
            }
            eval_cache = st.session_state.setdefault("_deal_bt_eval_cache", {})
            eval_key = ("deal-criteria-eval-batch", int(deal_row_idx), str(dealer_dir or "N").upper(), int(bt_index))
            if eval_key in eval_cache:
                eval_data = eval_cache[eval_key]
            else:
                eval_data = api_post("/deal-criteria-eval-batch", eval_payload)
                eval_cache[eval_key] = eval_data
            _st_info_elapsed("Arena Debug: deal-criteria-eval-batch", eval_data)

            res = eval_data.get("results") or []
            rows = []
            for rr in res:
                rows.append(
                    {
                        "seat": rr.get("seat"),
                        "passed_n": len(rr.get("passed") or []),
                        "failed": rr.get("failed") or [],
                        "untracked": rr.get("untracked") or [],
                    }
                )
            dbg_df = pl.DataFrame(rows) if rows else pl.DataFrame()
            if not dbg_df.is_empty():
                render_aggrid(
                    dbg_df,
                    key=f"arena_debug_bt_{bt_index}_{deal_index_label}_{key_suffix}",
                    height=calc_grid_height(len(dbg_df), max_height=220),
                )
        except Exception as e:
            st.warning(f"Debug BT index failed: {e}")

    effective_sample_size = int(sample_size)
    if pinned_indexes:
        # If the user pins specific deal indexes, only show those deals (no padding with random samples).
        # The API handler includes pinned deals and then pads up to sample_size; setting sample_size=len(pins)
        # makes it return exactly the pinned set.
        effective_sample_size = max(1, len(pinned_indexes))
        st.sidebar.caption(f"Using pinned-only mode: {len(pinned_indexes)} deal(s).")
    
    # Optional: Custom deals URI
    deals_uri = st.text_input(
        "Custom Deals URI (optional)",
        value="",
        help="Path to a parquet/CSV file or URL with custom deals. Leave blank to use default dataset.",
    )

    # Auto-run arena comparison when this page is active (no button).
    if len(model_names) < 2:
        st.warning("At least two models are required to run the Bidding Arena.")
        return

    if model_a == model_b:
        st.error("Please choose two different models for comparison (Model A and Model B must differ).")
        return

    with st.spinner("Running arena comparison..."):
        try:
            payload = {
                "model_a": model_a,
                "model_b": model_b,
                "sample_size": effective_sample_size,
                "seed": seed,
                "auction_pattern": auction_pattern if auction_pattern else None,
                "deals_uri": deals_uri if deals_uri else None,
                "deal_indices": pinned_indexes if pinned_indexes else None,
                "search_all_bt_rows": bool(search_all_bt_rows),
            }
            # Cache the expensive arena response across Streamlit reruns (e.g., AgGrid selection changes).
            cache_key = ("bidding-arena", tuple(sorted(payload.items())))
            if st.session_state.get("_arena_cache_key") == cache_key and st.session_state.get("_arena_cache_data") is not None:
                data = st.session_state["_arena_cache_data"]
            else:
                timeout_s = 600 if search_all_bt_rows else 300
                if search_all_bt_rows:
                    st.warning("Searching all completed BT rows is enabled; this request may take minutes.")
                data = api_post("/bidding-arena", payload, timeout=timeout_s)
                st.session_state["_arena_cache_key"] = cache_key
                st.session_state["_arena_cache_data"] = data
            
            elapsed_ms = data.get("elapsed_ms", 0)
            st.info(f"🏟️ Bidding Arena startup completed in {format_elapsed(elapsed_ms)}")
            rules_search = data.get("rules_search") or {}
            if rules_search:
                st.caption(
                    f"Rules search: mode={rules_search.get('mode')}, "
                    f"search_all_bt_rows={rules_search.get('search_all_bt_rows')}, "
                    f"candidate_count={rules_search.get('candidate_count')}, "
                    f"fallback_candidate_count={rules_search.get('fallback_candidate_count')}"
                )

            # Display summary
            st.subheader("📊 Summary")
            summary = data.get("summary", {})
            
            # Extract key counts for clarity
            total_deals = data.get("total_deals", 0)  # After pattern filter
            analyzed_deals = data.get("analyzed_deals", 0)  # After sampling
            deals_compared = data.get("deals_compared", summary.get("total_deals", 0))  # Successfully processed
            
            # Show diagnostic info if there's a problem
            if analyzed_deals == 0:
                st.warning(f"No deals to analyze (total matching: {total_deals})")
            
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Deals Compared", deals_compared, delta=f"of {analyzed_deals} sampled")
            with summary_cols[1]:
                st.metric(f"{model_a} Avg Score", f"{summary.get(f'avg_dd_score_{model_a.lower()}', 0):.1f}")
            with summary_cols[2]:
                st.metric(f"{model_b} Avg Score", f"{summary.get(f'avg_dd_score_{model_b.lower()}', 0):.1f}")
            with summary_cols[3]:
                imp_diff = summary.get(f"avg_imp_{model_a.lower()}_vs_{model_b.lower()}", 0)
                st.metric(f"Avg IMP ({model_a} vs {model_b})", f"{imp_diff:+.2f}")

            # Head-to-head stats
            st.subheader("🥊 Head-to-Head")
            h2h = data.get("head_to_head", {})
            h2h_cols = st.columns(3)
            with h2h_cols[0]:
                st.metric(f"{model_a} Wins", h2h.get(f"{model_a.lower()}_wins", 0))
            with h2h_cols[1]:
                st.metric("Ties", h2h.get("ties", 0))
            with h2h_cols[2]:
                st.metric(f"{model_b} Wins", h2h.get(f"{model_b.lower()}_wins", 0))

            # Contract quality metrics
            if "contract_quality" in data:
                st.subheader("📈 Contract Quality")
                cq = data["contract_quality"]
                cq_df = pl.DataFrame(
                    [
                        {"Metric": k, model_a: v.get(model_a.lower(), 0), model_b: v.get(model_b.lower(), 0)}
                        for k, v in cq.items()
                    ]
                )
                render_aggrid(cq_df, key="arena_contract_quality", height=calc_grid_height(len(cq_df)))

            # Segmentation
            if "segmentation" in data:
                st.subheader("📊 Segmentation Analysis")
                for seg_idx, (seg_name, seg_data) in enumerate(data["segmentation"].items()):
                    # seg_name is like "by_vulnerability" -> "Vulnerability"
                    display_name = seg_name.replace("by_", "").replace("_", " ").title()
                    st.markdown(f"**By {display_name}**")
                    if seg_data:
                        seg_df = pl.DataFrame(seg_data)
                        render_aggrid(seg_df, key=f"arena_seg_{seg_idx}", height=calc_grid_height(len(seg_df)))
                    else:
                        st.info("No segmentation data available.")

            # Sample deals - split into two views:
            # 1. "Match Actual Auction" - diagnostic view of criteria matching
            # 2. "Deal Comparison" - side-by-side comparison where Rules found a match
            if "sample_deals" in data and data["sample_deals"]:
                sample_df = pl.DataFrame(data["sample_deals"])
                # Avoid mixing list-typed "Rules_Matches_Auctions" into Auction_* columns (breaks Polars casting).
                # Instead, render an optional string column for debugging/visibility.
                if "Rules_Matches_Auctions_Str" in sample_df.columns:
                    pass
                elif "Rules_Matches_Auctions" in sample_df.columns:
                    try:
                        sample_df = sample_df.with_columns(
                            pl.when(pl.col("Rules_Matches_Auctions").is_null())
                            .then(pl.lit(""))
                            .otherwise(
                                pl.col("Rules_Matches_Auctions")
                                .list.eval(pl.element().fill_null("").cast(pl.Utf8))
                                .list.join(", ")
                            )
                            .alias("Rules_Matches_Auctions_Str")
                        )
                    except Exception:
                        # If list casting fails (e.g., List[Null] everywhere), just skip this derived column.
                        pass

                # ─────────────────────────────────────────────────────────────
                # Section 1: "Deal Comparison" - only deals with matched Rules auction
                # ─────────────────────────────────────────────────────────────
                rules_col = f"Auction_{model_a}" if model_a == "Rules" else f"Auction_{model_b}" if model_b == "Rules" else None
                actual_col = "Auction_Actual"
                comparison_df = None
                if rules_col and rules_col in sample_df.columns and actual_col in sample_df.columns:
                    # Filter to deals where Rules found a match (non-null, non-empty)
                    comparison_df = sample_df.filter(
                        pl.col(rules_col).is_not_null() & (pl.col(rules_col) != "")
                    )
                    if comparison_df.height > 0:
                        st.subheader("⚔️ Deal Comparison")
                        st.caption(f"Deals where Rules found a matching BT auction ({comparison_df.height}/{sample_df.height} deals)")
                        
                        # Add Match column: does Rules auction match Actual?
                        comparison_df = comparison_df.with_columns(
                            (pl.col(actual_col).str.to_lowercase() == pl.col(rules_col).str.to_lowercase())
                            .alias("Auction_Match")
                        )
                        
                        # Focused columns for comparison
                        compare_cols = [
                            "index",
                            "Dealer",
                            "Vul",
                            actual_col,
                            rules_col,
                            "Auction_Match",
                            "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                            f"DD_Score_{model_a}" if f"DD_Score_{model_a}" in sample_df.columns else None,
                            f"DD_Score_{model_b}" if f"DD_Score_{model_b}" in sample_df.columns else None,
                            "IMP_Diff",
                        ]
                        compare_cols = [c for c in compare_cols if c and c in comparison_df.columns]
                        compare_remaining = [c for c in comparison_df.columns if c not in compare_cols]
                        comparison_df = comparison_df.select(compare_cols + compare_remaining)
                        
                        # Show match rate
                        if "Auction_Match" in comparison_df.columns:
                            match_count = comparison_df.filter(pl.col("Auction_Match")).height
                            match_pct = 100 * match_count / comparison_df.height if comparison_df.height > 0 else 0
                            st.metric("Auction Match Rate", f"{match_pct:.1f}%", f"{match_count}/{comparison_df.height} deals")
                        
                        render_aggrid(
                            comparison_df,
                            key="arena_deal_comparison",
                            height=calc_grid_height(len(comparison_df)),
                            table_name="deal_comparison",
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            show_copy_panel=True,
                            copy_panel_default_col="index",
                            hide_cols=["_row_idx", "Rules_Matches", "Rules_Matches_Auctions", "Auction_Rules_Selected"],
                        )
                    else:
                        st.info("No deals with matched Rules auction for comparison.")
                
                # ─────────────────────────────────────────────────────────────
                # Section 2: "No BT Match" - deals requiring investigation
                # ─────────────────────────────────────────────────────────────
                if rules_col and rules_col in sample_df.columns:
                    # Filter to deals where Rules did NOT find a match (null or empty)
                    no_match_df = sample_df.filter(
                        pl.col(rules_col).is_null() | (pl.col(rules_col) == "")
                    )
                    if no_match_df.height > 0:
                        st.subheader("🔎 No BT Match (Investigation Needed)")
                        st.caption(f"Deals where no BT auction matched criteria ({no_match_df.height}/{sample_df.height} deals)")
                        
                        # Focused columns for investigation
                        investigate_cols = [
                            "index",
                            "Dealer",
                            "Vul",
                            "Auction_Actual",
                            "Rules_Actual_BT_Lookup",
                            "Rules_Actual_Criteria_OK",
                            "Rules_Actual_First_Failure",
                            "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                            "Rules_Actual_BT_Index",
                            "Rules_Actual_Lead_Passes",
                            "Rules_Actual_Opener_Seat",
                            "Rules_Actual_Seat1_Criteria",
                            "Rules_Actual_BT_Base_Criteria_By_Seat",
                            "Rules_Actual_BT_MergedOnly_Criteria_By_Seat",
                            "Rules_Actual_BT_Merged_Criteria_By_Seat",
                        ]
                        investigate_existing = [c for c in investigate_cols if c in no_match_df.columns]
                        investigate_remaining = [c for c in no_match_df.columns if c not in investigate_existing]
                        no_match_df = no_match_df.select(investigate_existing + investigate_remaining)
                        
                        render_aggrid(
                            no_match_df,
                            key="arena_no_bt_match",
                            height=calc_grid_height(len(no_match_df)),
                            table_name="no_bt_match",
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            show_copy_panel=True,
                            copy_panel_default_col="index",
                            hide_cols=["_row_idx", "Rules_Matches", "Rules_Matches_Auctions", "Auction_Rules_Selected"],
                        )
                    else:
                        st.success("All deals have a matching BT auction!")
                
                # ─────────────────────────────────────────────────────────────
                # Section 3: "Match Actual Auction" - full diagnostic view
                # ─────────────────────────────────────────────────────────────
                st.subheader("🔍 Match Actual Auction")
                st.caption("Diagnostic view: checks if each deal's actual auction matches BT criteria")
                
                # Order columns sensibly - Hand_[NESW] in NESW order
                priority_cols = [
                    "index",
                    "Dealer",
                    "Vul",
                    "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                    f"Auction_{model_a}",
                    f"Auction_{model_b}",
                    "Rules_Actual_BT_Lookup",
                    "Rules_Actual_BT_Index",
                    "Rules_Actual_Lead_Passes",
                    "Rules_Actual_Opener_Seat",
                    "Rules_Actual_Criteria_OK",
                    "Rules_Actual_First_Failure",
                    "Rules_Actual_Seat1_Criteria",
                    "Rules_Actual_BT_Base_Criteria_By_Seat",
                    "Rules_Actual_BT_MergedOnly_Criteria_By_Seat",
                    "Rules_Actual_BT_Merged_Criteria_By_Seat",
                    "Rules_Matches_Count",
                    "Rules_Matches_Truncated",
                    "Rules_Matches_Auctions_Str",
                    f"DD_Score_{model_a}",
                    f"DD_Score_{model_b}",
                    "Rejection_Reason",
                    "IMP_Diff",
                ]
                existing = [c for c in priority_cols if c in sample_df.columns]
                remaining = [c for c in sample_df.columns if c not in existing]
                sample_df = sample_df.select(existing + remaining)
                selected_rows = render_aggrid(
                    sample_df,
                    key="arena_sample_deals",
                    height=calc_grid_height(len(sample_df)),
                    table_name="arena_samples",
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    show_copy_panel=True,
                    copy_panel_default_col="index",
                    hide_cols=["_row_idx", "Rules_Matches", "Rules_Matches_Auctions", "Auction_Rules_Selected"],
                )

                # Convenience: if we're pinned-only (single row) and Debug BT index is set,
                # auto-run the debug without requiring a grid click.
                if debug_bt_index is not None and pinned_indexes and len(pinned_indexes) == 1 and len(sample_df) == 1 and not selected_rows:
                    try:
                        only = sample_df.to_dicts()[0] if sample_df.height == 1 else {}
                        only_row_idx = only.get("_row_idx")
                        only_dealer = str(only.get("Dealer") or "N").upper()
                        if only_row_idx is not None:
                            st.subheader("🔎 Selected Deal: BT Auction Sequence (Step-by-step)")
                            st.caption(f"Selected deal index: {only.get('index')}")
                            _render_debug_bt_index(
                                int(debug_bt_index),
                                int(only_row_idx),
                                only_dealer,
                                only.get("index"),
                                "auto",
                            )
                    except Exception as e:
                        st.warning(f"Auto debug failed: {e}")

                # Copy-friendly index list
                if "index" in sample_df.columns and len(sample_df) > 0:
                    try:
                        idx_vals = [int(x) for x in sample_df.get_column("index").drop_nulls().to_list()]
                    except Exception:
                        idx_vals = [str(x) for x in sample_df.get_column("index").drop_nulls().to_list()]
                    idx_csv = ", ".join(str(x) for x in idx_vals)
                    st.text_area(
                        "Copy indexes (comma-separated)",
                        value=idx_csv,
                        height=80,
                    )

                # Click-to-drill into BT auction sequence (previous_bid_indices chain)
                if selected_rows:
                    sel = selected_rows[0] or {}
                    sel_idx = sel.get("index")
                    sel_row_idx = sel.get("_row_idx")
                    sel_dealer = str(sel.get("Dealer") or "").upper()
                    dealer_to_seat = {"N": 1, "E": 2, "S": 3, "W": 4}
                    dealer_seat = dealer_to_seat.get(sel_dealer, 1)
                    rules_like_models = {"Rules", "Rules_Learned", "Rules_Base"}
                    rules_missing_for_row = (
                        (model_a in rules_like_models and not sel.get(f"Auction_{model_a}"))
                        or (model_b in rules_like_models and not sel.get(f"Auction_{model_b}"))
                    )

                    st.subheader("🔎 Selected Deal: BT Auction Sequence (Step-by-step)")
                    if sel_idx is not None:
                        st.caption(f"Selected deal index: {sel_idx}")
                    if sel_row_idx is None:
                        st.warning("Missing internal deal row index (_row_idx); cannot compute 'Failed Exprs' via bitmap lookups.")
                    rejection_reason = str(sel.get("Rejection_Reason") or "").strip()
                    rejected_model: str | None = None
                    for m in ("Rules", "Rules_Learned", "Rules_Base"):
                        if rejection_reason.startswith(f"{m}: no matching auction"):
                            rejected_model = m
                            break
                    if rejected_model is not None:
                        st.caption(
                            f"Note: this rejection reason is about the **{rejected_model} model** failing to find any BT completed auction whose criteria match this deal. "
                            "The 'Failed Exprs' column below evaluates the criteria for the specific BT auction sequence shown (if any)."
                        )

                    # Show all matching Rules BT rows for this deal (if present)
                    rules_matches = sel.get("Rules_Matches") or []
                    selected_rules_bt_index: int | None = None
                    if rules_matches:
                        st.markdown("**Rules matches for this deal (all matching BT rows)**")
                        try:
                            rm_df = pl.DataFrame(rules_matches)
                            if "bt_index" in rm_df.columns:
                                rm_df = rm_df.sort("bt_index")
                            st.caption("Tip: click a row to show that BT sequence immediately below.")
                            selected_rules_rows = render_aggrid(
                                rm_df,
                                key=f"arena_rules_matches_{sel_idx}",
                                height=calc_grid_height(len(rm_df), max_height=260),
                                table_name="rules_matches",
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                show_copy_panel=True,
                                copy_panel_default_col="bt_index" if "bt_index" in rm_df.columns else None,
                            )
                            if selected_rules_rows:
                                try:
                                    v = (selected_rules_rows[0] or {}).get("bt_index")
                                    selected_rules_bt_index = int(v) if v is not None else None
                                except Exception:
                                    selected_rules_bt_index = None
                        except Exception as e:
                            st.warning(f"Could not render Rules matches table: {e}")
                            st.json(rules_matches)
                            selected_rules_bt_index = None

                    # Optional: debug a specific bt_index against this selected deal.
                    if debug_bt_index is not None and sel_row_idx is not None:
                        _render_debug_bt_index(int(debug_bt_index), int(sel_row_idx), sel_dealer or "N", sel_idx, "selected")

                    # If there are many Rules matches, show a few of their BT sequences below for visibility.
                    if rules_matches and sel_row_idx is not None:
                        try:
                            max_show = st.sidebar.number_input(
                                "Max Rules matches to expand in Selected Deal",
                                min_value=0,
                                max_value=25,
                                value=5,
                                step=1,
                                help="How many matching Rules BT rows to render as step-by-step sequences below.",
                                key="arena_rules_matches_max_show",
                            )
                        except Exception:
                            max_show = 5

                        try:
                            bt_idxs = []
                            for m in rules_matches:
                                try:
                                    bt_idxs.append(int(m.get("bt_index")))
                                except Exception:
                                    continue
                            bt_idxs = [x for x in bt_idxs if x is not None]
                            if max_show and bt_idxs:
                                # If the user clicked a Rules match row, "jump" to it by rendering it first.
                                bt_idxs_to_show: list[int] = []
                                if selected_rules_bt_index is not None:
                                    bt_idxs_to_show.append(int(selected_rules_bt_index))
                                for x in bt_idxs:
                                    if len(bt_idxs_to_show) >= int(max_show):
                                        break
                                    if selected_rules_bt_index is not None and int(x) == int(selected_rules_bt_index):
                                        continue
                                    bt_idxs_to_show.append(int(x))

                                st.markdown(f"**Rules match sequences (showing {len(bt_idxs_to_show)} of {len(bt_idxs)})**")
                                seq_payload = {"indices": bt_idxs_to_show, "allow_initial_passes": False}
                                bt_cache = st.session_state.setdefault("_bt_by_index_cache", {})
                                bt_key = ("auction-sequences-by-index", tuple(bt_idxs_to_show))
                                if bt_key in bt_cache:
                                    bt_data = bt_cache[bt_key]
                                else:
                                    bt_data = api_post("/auction-sequences-by-index", seq_payload)
                                    bt_cache[bt_key] = bt_data
                                for s in (bt_data.get("samples") or []):
                                    seq_rows = s.get("sequence") or []
                                    if not isinstance(seq_rows, list) or not seq_rows:
                                        continue
                                    # Take the matched row itself for a quick header
                                    bt_idx = None
                                    auc = None
                                    try:
                                        for rr in seq_rows:
                                            if int(rr.get("index") or -1) in bt_idxs_to_show and rr.get("is_match_row") is True:
                                                bt_idx = rr.get("index")
                                                auc = rr.get("Auction")
                                                break
                                    except Exception:
                                        bt_idx = None
                                        auc = None
                                    if bt_idx is None:
                                        last = seq_rows[-1]
                                        bt_idx = last.get("index")
                                        auc = last.get("Auction")

                                    st.markdown(f"**Rules BT {bt_idx}: {auc}**")
                                    # Reuse the existing renderer by pattern (cheap) would requery;
                                    # instead, just show the raw sequence table.
                                    try:
                                        seq_df2 = pl.DataFrame(seq_rows)
                                        if "index" in seq_df2.columns:
                                            seq_df2 = seq_df2.sort("index")
                                        render_aggrid(
                                            seq_df2,
                                            key=f"arena_rules_seq_{sel_idx}_{bt_idx}",
                                            height=calc_grid_height(len(seq_df2), max_height=260),
                                            table_name="auction_sequences",
                                            show_copy_panel=True,
                                            copy_panel_default_col="index",
                                        )
                                    except Exception as e:
                                        st.warning(f"Failed to render rules sequence for bt_index={bt_idx}: {e}")
                        except Exception as e:
                            st.warning(f"Failed to render Rules match sequences: {e}")

                    def _render_bt_sequence_for_auction(auction_text: str, label: str) -> bool:
                        # Guard: some grid cells may contain lists of auctions (e.g. all Rules matches).
                        if not isinstance(auction_text, str):
                            return False
                        if not auction_text or auction_text.strip() == "":
                            return False
                        try:
                            # Determine how many leading passes the *selected auction text* has.
                            # This is necessary to map seat-1 canonical BT seats back to deal-relative seats
                            # (seat 1 = dealer, 2 = LHO, 3 = partner, 4 = RHO).
                            raw_for_passes = str(auction_text).strip()
                            raw_for_passes = raw_for_passes.lstrip("^").rstrip("$").strip()
                            m_pass = re.match(r"(?i)^((p-)+)", raw_for_passes)
                            leading_passes = (m_pass.group(1).upper().count("P-")) if m_pass else 0

                            # For drilldown we want the BT seat-1 canonical auction (no leading p- padding),
                            # so the sequence we show matches the actual bt_seat1 row (e.g. "1s-p-p-p"),
                            # not a seat-expanded display variant (e.g. "p-p-1s-p-p-p").
                            normalized = normalize_auction_user_text(str(auction_text))
                            base_auction = re.sub(r"(?i)^(p-)+", "", normalized)

                            # Try exact match first, then fall back to a plain (substring/regex) match.
                            patterns_to_try: list[str] = []
                            exact_pattern = base_auction
                            if not (exact_pattern.startswith("^") or exact_pattern.endswith("$")):
                                exact_pattern = f"^{exact_pattern}$"
                            patterns_to_try.append(exact_pattern)
                            if base_auction != exact_pattern:
                                patterns_to_try.append(base_auction)

                            samples: list[dict[str, Any]] = []
                            for patt in patterns_to_try:
                                payload = {
                                    "pattern": patt,
                                    # We already canonicalized to seat-1, so don't expand into dealer/seat variants.
                                    "allow_initial_passes": False,
                                    "n_samples": 1,
                                    "seed": 0,
                                }
                                # Cache drilldown lookups per (pattern, allow_initial_passes) to avoid repeated API calls on reruns.
                                seq_cache = st.session_state.setdefault("_bt_seq_cache", {})
                                seq_cache_key = ("auction-sequences-matching", payload["pattern"], payload["allow_initial_passes"])
                                if seq_cache_key in seq_cache:
                                    seq_data = seq_cache[seq_cache_key]
                                else:
                                    seq_data = api_post("/auction-sequences-matching", payload)
                                    seq_cache[seq_cache_key] = seq_data
                                samples = seq_data.get("samples", []) or []
                                if samples:
                                    break
                            chosen = samples[0] if samples else None
                            if not chosen:
                                st.info(f"{label}: no BT auction sequence found for {base_auction}")
                                return False

                            opener_seat = 1  # seat-1 canonical sequence
                            seq_rows = chosen.get("sequence") or []
                            seq_df = pl.DataFrame(seq_rows) if isinstance(seq_rows, list) else pl.DataFrame()
                            if seq_df.is_empty():
                                st.info(f"{label}: empty sequence")
                                return False

                            # Add Seat columns and Agg_Expr_Seat.
                            # Seat mapping:
                            # - BT_Seat: seat-1 canonical (opener = 1, next = 2, ...)
                            # - Seat: deal-relative (dealer = 1, LHO = 2, partner = 3, RHO = 4)
                            # We map BT_Seat -> Seat by rotating by the number of leading passes in the selected auction text.
                            n_rows = seq_df.height
                            bt_seats: list[int] = []
                            deal_seats: list[int] = []
                            agg_expr_vals: list[list[str] | None] = []

                            for i in range(n_rows):
                                inferred_bt_seat: int | None = None
                                if "Expr" in seq_df.columns:
                                    expr_list = _to_list_utf8_cell(seq_df["Expr"][i]) or []
                                    if expr_list:
                                        for s in (1, 2, 3, 4):
                                            col = f"Agg_Expr_Seat_{s}"
                                            if col not in seq_df.columns:
                                                continue
                                            cand = _to_list_utf8_cell(seq_df[col][i]) or []
                                            if cand and all(e in cand for e in expr_list):
                                                inferred_bt_seat = s
                                                break

                                bt_seat = inferred_bt_seat or (((opener_seat - 1 + i) % 4) + 1)
                                bt_seats.append(bt_seat)

                                # Rotate canonical BT seats into deal-relative seats using leading passes.
                                deal_seat = ((bt_seat - 1 + (leading_passes % 4)) % 4) + 1
                                deal_seats.append(deal_seat)

                                agg_col = f"Agg_Expr_Seat_{bt_seat}"
                                if agg_col in seq_df.columns:
                                    agg_expr_vals.append(_to_list_utf8_cell(seq_df[agg_col][i]))
                                else:
                                    agg_expr_vals.append(None)

                            seq_df = seq_df.with_columns(
                                pl.Series("BT_Seat", bt_seats),
                                pl.Series("Seat", deal_seats),
                            )
                            # Force strongly-typed list-of-strings column (prevents Polars panic on all-None).
                            seq_df = seq_df.with_columns(pl.Series("Agg_Expr_Seat", agg_expr_vals, dtype=pl.List(pl.Utf8)))
                            drop_cols = [c for c in seq_df.columns if c.startswith("Agg_Expr_Seat_") and c[-1].isdigit()]
                            if drop_cols:
                                seq_df = seq_df.drop(drop_cols)

                            # Compute Failed Exprs for each step (server-side bitmap evaluation)
                            if sel_row_idx is not None and "Agg_Expr_Seat" in seq_df.columns and "Seat" in seq_df.columns:
                                try:
                                    checks = []
                                    needed_cols = ["Seat", "Agg_Expr_Seat"]
                                    if "Expr" in seq_df.columns:
                                        needed_cols.append("Expr")
                                    for rr in seq_df.select(needed_cols).to_dicts():
                                        crits = rr.get("Agg_Expr_Seat") or []
                                        # Fallback: if Agg_Expr_Seat is empty, derive criteria from Expr.
                                        # This matches the original intent of "Failed Exprs" being based on Expr evaluation.
                                        if not crits:
                                            crits = _expr_to_criteria_list(rr.get("Expr"))
                                        checks.append(
                                            {
                                                "seat": int(rr.get("Seat") or 0),
                                                "criteria": crits,
                                            }
                                        )
                                    payload_fail = {
                                        "deal_row_idx": int(sel_row_idx),
                                        "dealer": sel_dealer or "N",
                                        "checks": checks,
                                    }
                                    fail_cache = st.session_state.setdefault("_deal_fail_cache", {})
                                    fail_key = (
                                        "deal-criteria-eval-batch",
                                        payload_fail["deal_row_idx"],
                                        payload_fail["dealer"],
                                        tuple((c["seat"], tuple(c["criteria"])) for c in checks),
                                    )
                                    if fail_key in fail_cache:
                                        fail_data = fail_cache[fail_key]
                                    else:
                                        fail_data = api_post("/deal-criteria-eval-batch", payload_fail)
                                        fail_cache[fail_key] = fail_data
                                    res_list = fail_data.get("results") or []
                                    failed_col = []
                                    for r in res_list:
                                        f = r.get("failed") or []
                                        u = r.get("untracked") or []
                                        parts = []
                                        if f:
                                            parts.extend([str(x) for x in f])
                                        if u:
                                            parts.extend([f"UNTRACKED: {x}" for x in u])
                                        failed_col.append(parts)
                                    if len(failed_col) == seq_df.height:
                                        seq_df = seq_df.with_columns(
                                            pl.Series("Failed Exprs", failed_col, dtype=pl.List(pl.Utf8))
                                        )
                                except Exception as e:
                                    st.warning(f"Failed to compute Failed Exprs: {e}")

                            seq_df = order_columns(seq_df, priority_cols=["index", "Auction", "Seat", "BT_Seat", "is_match_row", "Failed Exprs", "Expr", "Agg_Expr_Seat"])
                            if "index" in seq_df.columns:
                                seq_df = seq_df.sort("index")

                            # Show what BT row actually matched (from server), not just the requested text.
                            matched_bt_idx = None
                            matched_bt_auc = None
                            try:
                                if "is_match_row" in seq_df.columns and "index" in seq_df.columns and "Auction" in seq_df.columns:
                                    mdf = seq_df.filter(pl.col("is_match_row") == True).select(["index", "Auction"])
                                    if mdf.height == 1:
                                        matched_bt_idx = mdf["index"][0]
                                        matched_bt_auc = mdf["Auction"][0]
                            except Exception:
                                matched_bt_idx = None
                                matched_bt_auc = None

                            title_auction = matched_bt_auc or base_auction
                            st.markdown(f"**{label} (BT seat-1): {title_auction}**")
                            if matched_bt_idx is not None and matched_bt_auc is not None:
                                st.caption(f"Matched BT row: bt_index={matched_bt_idx}, auction={matched_bt_auc}")
                            render_aggrid(
                                seq_df,
                                key=f"arena_bt_seq_{label}_{sel_idx}",
                                table_name="auction_sequences",
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                show_copy_panel=True,
                                copy_panel_default_col="index",
                            )

                            # If Rules had no match for this deal, auto-debug the BT row corresponding to the
                            # *Actual* auction sequence shown here. This answers: "does this deal satisfy the
                            # criteria of the BT row that shares its auction string?"
                            if (
                                debug_bt_index is None
                                and rules_missing_for_row
                                and sel_row_idx is not None
                                and "Actual" in label
                                and "index" in seq_df.columns
                            ):
                                try:
                                    bt_idx_val = None
                                    if "is_match_row" in seq_df.columns:
                                        mdf = seq_df.filter(pl.col("is_match_row") == True).select("index")
                                        if mdf.height == 1:
                                            bt_idx_val = mdf.item()
                                    if bt_idx_val is None and seq_df.height > 0:
                                        bt_idx_val = seq_df.select("index").tail(1).item()
                                    if bt_idx_val is not None:
                                        _render_debug_bt_index(
                                            int(bt_idx_val),
                                            int(sel_row_idx),
                                            sel_dealer or "N",
                                            sel_idx,
                                            "auto_from_actual",
                                        )
                                except Exception as e:
                                    st.warning(f"Auto debug (from Actual BT row) failed: {e}")
                            return True
                        except Exception as e:
                            st.warning(f"{label}: failed to fetch/render BT sequence: {e}")
                            return False

                    # Show sequences for both model auctions (if present)
                    auct_a = sel.get(f"Auction_{model_a}")
                    auct_b = sel.get(f"Auction_{model_b}")

                    # If a model is Rules, the displayed Auction cell may be a list (all matches).
                    # For drilldown-by-auction-string, use the single selected Rules auction instead.
                    if model_a == "Rules":
                        auct_a = sel.get("Auction_Rules_Selected") or auct_a
                    if model_b == "Rules":
                        auct_b = sel.get("Auction_Rules_Selected") or auct_b

                    auct_a_s = _to_scalar_auction_text(auct_a)
                    auct_b_s = _to_scalar_auction_text(auct_b)
                    st.caption(
                        f"Selected row auctions: {model_a}={auct_a if auct_a else '(none)'}; {model_b}={auct_b if auct_b else '(none)'}"
                    )
                    if auct_a_s:
                        _render_bt_sequence_for_auction(auct_a_s, f"Auction_{model_a}")
                    elif auct_a:
                        if model_a == "Rules":
                            st.info(
                                f"Auction_{model_a}: multiple auctions shown in grid; "
                                "use Rules matches table below to drill into specific bt_index rows."
                            )
                        else:
                            st.info(f"Auction_{model_a}: value is not a single auction string; cannot drill down by auction.")
                    if auct_b_s and auct_b_s != auct_a_s:
                        _render_bt_sequence_for_auction(auct_b_s, f"Auction_{model_b}")
                    elif auct_b and auct_b != auct_a:
                        if model_b == "Rules":
                            st.info(
                                f"Auction_{model_b}: multiple auctions shown in grid; "
                                "use Rules matches table below to drill into specific bt_index rows."
                            )
                        else:
                            st.info(f"Auction_{model_b}: value is not a single auction string; cannot drill down by auction.")
                    if not auct_a and not auct_b:
                        st.info("This row has no auction values (likely rejected / missing Rules match), so no BT sequence can be shown.")

        except Exception as e:
            st.error(f"Arena comparison failed: {e}")


# ---------------------------------------------------------------------------
# Custom Criteria Editor – Manage bbo_custom_auction_criteria.csv
# ---------------------------------------------------------------------------

def render_custom_criteria_editor():
    """Render the Custom Criteria Editor for managing auction criteria rules."""
    st.header("📝 Custom Criteria Editor")
    st.markdown("Manage custom auction criteria applied as a **hot-reloadable overlay** (no server restart).")
    
    # Action buttons row
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("↻ Reload from Server", help="Hot-reload criteria overlay from CSV (no server restart)"):
            with st.spinner("Reloading..."):
                try:
                    resp = requests.post(f"{API_BASE}/custom-criteria-reload", timeout=30)
                    if resp.ok:
                        data = resp.json()
                        if data.get("success"):
                            st.success(f"Reloaded! {data.get('stats', {}).get('rules_applied', 0)} rules applied.")
                            st.rerun()
                        else:
                            st.error("Reload failed")
                    else:
                        st.error(f"Reload failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Load current rules (single API call, reused for file path display)
    try:
        rules_resp = requests.get(f"{API_BASE}/custom-criteria-rules", timeout=10)
        rules_resp.raise_for_status()
        rules_data = rules_resp.json()
        current_rules = rules_data.get("rules", [])
        file_path = rules_data.get("file_path", "N/A")
    except Exception as e:
        st.error(f"Failed to load rules: {e}")
        current_rules = []
        file_path = "N/A"
    
    with col2:
        st.caption(f"📁 `{file_path}`")
    
    # Initialize session state for rules if not present (only on first load)
    if "criteria_rules" not in st.session_state:
        st.session_state.criteria_rules = current_rules
    
    st.divider()
    
    # Current rules display with edit/delete
    st.subheader(f"📋 Current Rules ({len(st.session_state.criteria_rules)})")
    
    if not st.session_state.criteria_rules:
        st.info("No rules defined. Add one below.")
    else:
        # Display rules in a table with actions
        rules_to_delete = []
        
        for idx, rule in enumerate(st.session_state.criteria_rules):
            with st.container():
                cols = st.columns([2, 4, 1, 1, 1])
                
                with cols[0]:
                    new_partial = st.text_input(
                        "Partial",
                        value=rule["partial_auction"],
                        key=f"partial_{idx}",
                        label_visibility="collapsed",
                    )
                    # Normalize to canonical uppercase
                    new_partial_normalized = new_partial.strip().upper() if new_partial else ""
                    if new_partial_normalized != rule["partial_auction"]:
                        st.session_state.criteria_rules[idx]["partial_auction"] = new_partial_normalized
                
                with cols[1]:
                    criteria_str = ", ".join(rule["criteria"])
                    new_criteria_str = st.text_input(
                        "Criteria",
                        value=criteria_str,
                        key=f"criteria_{idx}",
                        label_visibility="collapsed",
                        help="Comma-separated: HCP >= 12, SL_C >= 3",
                    )
                    if new_criteria_str != criteria_str:
                        new_criteria = [c.strip() for c in new_criteria_str.split(",") if c.strip()]
                        st.session_state.criteria_rules[idx]["criteria"] = new_criteria
                
                with cols[2]:
                    # Preview button
                    if st.button("👁", key=f"preview_{idx}", help="Preview impact"):
                        try:
                            preview_resp = requests.post(
                                f"{API_BASE}/custom-criteria-preview",
                                json={"partial_auction": rule["partial_auction"]},
                                timeout=10,
                            )
                            if preview_resp.ok:
                                preview = preview_resp.json()
                                st.info(f"Seat {preview['seat_affected']}: {preview['auctions_affected']:,} auctions")
                        except Exception:
                            pass
                
                with cols[3]:
                    # Validate button
                    if st.button("✓", key=f"validate_{idx}", help="Validate syntax"):
                        all_valid = True
                        for criterion in rule["criteria"]:
                            try:
                                val_resp = requests.post(
                                    f"{API_BASE}/custom-criteria-validate",
                                    json={"expression": criterion},
                                    timeout=5,
                                )
                                if val_resp.ok:
                                    val_data = val_resp.json()
                                    if not val_data.get("valid"):
                                        st.warning(f"'{criterion}': {val_data.get('error')}")
                                        all_valid = False
                            except Exception:
                                pass
                        if all_valid:
                            st.success("All valid!")
                
                with cols[4]:
                    if st.button("🗑", key=f"delete_{idx}", help="Delete rule"):
                        rules_to_delete.append(idx)
        
        # Process deletions
        if rules_to_delete:
            for idx in sorted(rules_to_delete, reverse=True):
                st.session_state.criteria_rules.pop(idx)
            st.rerun()
    
    st.divider()
    
    # Add new rule section
    st.subheader("➕ Add New Rule")
    
    col_add1, col_add2 = st.columns([1, 3])
    
    with col_add1:
        new_partial = st.text_input(
            "Partial Auction",
            value="",
            placeholder="e.g., 1c, 1n-p-3n",
            key="new_partial",
        )
    
    with col_add2:
        new_criteria_input = st.text_input(
            "Criteria (comma-separated)",
            value="",
            placeholder="e.g., HCP >= 12, SL_C >= 3",
            key="new_criteria",
        )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("Preview Impact"):
            if new_partial:
                try:
                    preview_resp = requests.post(
                        f"{API_BASE}/custom-criteria-preview",
                        json={"partial_auction": new_partial},
                        timeout=10,
                    )
                    if preview_resp.ok:
                        preview = preview_resp.json()
                        st.info(
                            f"Would affect **{preview['auctions_affected']:,}** auctions "
                            f"(Seat {preview['seat_affected']})"
                        )
                        if preview.get("sample_auctions"):
                            st.caption(f"Samples: {', '.join(preview['sample_auctions'][:5])}")
                except Exception as e:
                    st.error(f"Preview failed: {e}")
            else:
                st.warning("Enter a partial auction first")
    
    with col_btn2:
        if st.button("Validate Syntax"):
            if new_criteria_input:
                criteria_list = [c.strip() for c in new_criteria_input.split(",") if c.strip()]
                all_valid = True
                for criterion in criteria_list:
                    try:
                        val_resp = requests.post(
                            f"{API_BASE}/custom-criteria-validate",
                            json={"expression": criterion},
                            timeout=5,
                        )
                        if val_resp.ok:
                            val_data = val_resp.json()
                            if val_data.get("valid"):
                                st.success(f"✓ `{criterion}`")
                            else:
                                st.warning(f"⚠ `{criterion}`: {val_data.get('error')}")
                                all_valid = False
                    except Exception as e:
                        st.error(f"Validation error: {e}")
                        all_valid = False
            else:
                st.warning("Enter criteria first")
    
    with col_btn3:
        if st.button("➕ Add Rule", type="primary"):
            if new_partial and new_criteria_input:
                new_criteria = [c.strip() for c in new_criteria_input.split(",") if c.strip()]
                st.session_state.criteria_rules.append({
                    "partial_auction": new_partial.strip().upper(),  # Canonical uppercase
                    "criteria": new_criteria,
                })
                st.success(f"Added rule for '{new_partial}'")
                # Clear inputs explicitly (these widgets use session_state keys)
                st.session_state["new_partial"] = ""
                st.session_state["new_criteria"] = ""
                st.rerun()
            else:
                st.warning("Both partial auction and criteria are required")
    
    st.divider()
    
    # Save section
    st.subheader("💾 Save Changes")
    
    # Show if there are unsaved changes
    if st.session_state.criteria_rules != current_rules:
        st.warning("⚠️ You have unsaved changes!")
    
    col_save1, col_save2 = st.columns([1, 3])
    
    with col_save1:
        if st.button("💾 Save to File", type="primary"):
            try:
                # Convert to API format
                rules_to_save = [
                    {"partial_auction": r["partial_auction"], "criteria": r["criteria"]}
                    for r in st.session_state.criteria_rules
                ]
                
                save_resp = requests.post(
                    f"{API_BASE}/custom-criteria-rules",
                    json={"rules": rules_to_save},
                    timeout=10,
                )
                
                if save_resp.ok:
                    save_data = save_resp.json()
                    if save_data.get("success"):
                        st.success(f"Saved {save_data.get('rules_saved', 0)} rules!")
                        st.session_state._needs_reload = True
                    else:
                        st.error("Save failed")
                else:
                    st.error(f"Save failed: {save_resp.text}")
            except Exception as e:
                st.error(f"Error saving: {e}")
    
    with col_save2:
        if st.button("🔄 Discard Changes"):
            st.session_state.criteria_rules = current_rules
            st.rerun()
    
    # Hot reload button (shown after save)
    if st.session_state.get("_needs_reload"):
        st.info("💡 Rules saved to file. Click below to apply changes to the running server.")
        if st.button("↻ Apply Changes (Hot Reload)", type="secondary"):
            try:
                reload_resp = requests.post(f"{API_BASE}/custom-criteria-reload", timeout=60)
                if reload_resp.ok:
                    reload_data = reload_resp.json()
                    if reload_data.get("success"):
                        st.success("Changes applied to server!")
                        st.session_state._needs_reload = False
                        st.rerun()
                else:
                    st.error(f"Reload failed: {reload_resp.text}")
            except Exception as e:
                st.error(f"Reload error: {e}")
    
    # Refresh from server button
    if st.button("🔃 Refresh from Server"):
        if "criteria_rules" in st.session_state:
            del st.session_state["criteria_rules"]
        st.rerun()
    
    # Analytics section
    st.divider()
    with st.expander("📊 Analytics"):
        # Show current merged stats
        try:
            info_resp = requests.get(f"{API_BASE}/custom-criteria-info", timeout=10)
            if info_resp.ok:
                info_data = info_resp.json()
                stats = info_data.get("stats", {})
                
                if stats.get("rules_applied"):
                    st.metric("Rules Applied", stats.get("rules_applied", 0))
                    st.metric("Auctions Modified", f"{stats.get('auctions_modified', 0):,}")
                    
                    # Distribution by seat
                    if stats.get("rules"):
                        seat_counts = {}
                        for rule in stats["rules"]:
                            seat = rule.get("seat", 0)
                            seat_counts[seat] = seat_counts.get(seat, 0) + 1
                        
                        st.markdown("**Rules by Seat:**")
                        for seat in sorted(seat_counts.keys()):
                            st.write(f"- Seat {seat}: {seat_counts[seat]} rules")
                else:
                    st.info("No rules currently applied on server. Save and reload to apply.")
        except Exception:
            st.info("Could not fetch analytics.")


# ---------------------------------------------------------------------------
# Wrong Bid Analysis – Statistics, failed criteria, and leaderboard
# ---------------------------------------------------------------------------

def render_wrong_bid_analysis():
    """Render wrong bid analysis tools: stats, failed criteria summary, and leaderboard."""
    st.header("🚫 Wrong Bid Analysis")
    st.markdown("Analyze auctions where deals do not conform to the expected bidding criteria.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Overall Stats", "❌ Failed Criteria Summary", "🏆 Leaderboard"])
    
    with tab1:
        st.subheader("Wrong Bid Statistics")
        # Auto-load overall stats when this tab is active.
        with st.spinner("Loading statistics..."):
            try:
                data = api_post("/wrong-bid-stats", {}, timeout=60)
                _st_info_elapsed("Wrong Bid Statistics", data)

                # Overall stats
                stats = data.get("overall_stats", {})
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Total Deals Analyzed", stats.get("total_deals", 0))
                with cols[1]:
                    st.metric("Deals with Wrong Bids", stats.get("deals_with_wrong_bids", 0))
                with cols[2]:
                    rate = stats.get("wrong_bid_rate", 0) * 100
                    st.metric("Wrong Bid Rate", f"{rate:.2f}%")
                with cols[3]:
                    st.metric("Unique Auctions", stats.get("unique_auctions_with_wrong_bids", 0))

                # Per-seat breakdown
                if "per_seat" in data:
                    st.subheader("Per-Seat Breakdown")
                    seat_data = data["per_seat"]
                    seat_df = pl.DataFrame(
                        [
                            {
                                "Seat": f"Seat {i}",
                                "Wrong Bids": seat_data.get(f"seat_{i}_wrong_bids", 0),
                                "Wrong Bid Rate": seat_data.get(f"seat_{i}_rate", 0) * 100,
                            }
                            for i in range(1, 5)
                        ]
                    )
                    render_aggrid(
                        seat_df, 
                        key="wrong_bid_per_seat", 
                        height=calc_grid_height(len(seat_df))
                    )

            except Exception as e:
                st.error(f"Failed to load stats: {e}")
    
    with tab2:
        st.subheader("Failed Criteria Summary")
        top_n = st.number_input("Top N Criteria", min_value=5, max_value=100, value=20, key="failed_criteria_top_n")

        # Auto-load failed criteria summary when this tab is active.
        with st.spinner("Analyzing failed criteria..."):
            try:
                data = api_post("/failed-criteria-summary", {"top_n": int(top_n)}, timeout=60)
                _st_info_elapsed("Failed Criteria Summary", data)

                if "criteria" in data and data["criteria"]:
                    criteria_df = pl.DataFrame(data["criteria"])
                    render_aggrid(
                        criteria_df, 
                        key="failed_criteria_grid", 
                        height=calc_grid_height(len(criteria_df))
                    )

                    # Visualization
                    if len(data["criteria"]) > 0:
                        st.subheader("📊 Top Failed Criteria")
                        import altair as alt

                        chart_data = criteria_df.head(min(10, len(criteria_df))).to_pandas()
                        chart = (
                            alt.Chart(chart_data)
                            .mark_bar()
                            .encode(
                                x=alt.X("failure_count:Q", title="Failure Count"),
                                y=alt.Y("criterion:N", sort="-x", title="Criterion"),
                                tooltip=["criterion", "failure_count", "affected_auctions"],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(chart, width="stretch")
                else:
                    st.info("No failed criteria found.")

            except Exception as e:
                st.error(f"Failed to load criteria: {e}")
    
    with tab3:
        st.subheader("Wrong Bid Leaderboard")
        st.markdown("Auctions with the highest wrong bid rates.")

        top_n_lb = st.number_input("Top N Auctions", min_value=5, max_value=100, value=20, key="leaderboard_top_n")
        min_deals = st.number_input(
            "Minimum Deals",
            min_value=1,
            max_value=1000,
            value=10,
            help="Only show auctions with at least this many deals",
        )

        # Auto-load leaderboard when this page is active (no button).
        with st.spinner("Loading leaderboard..."):
            try:
                data = api_post("/wrong-bid-leaderboard", {"top_n": int(top_n_lb), "min_deals": int(min_deals)}, timeout=60)
                _st_info_elapsed("Wrong Bid Leaderboard", data)

                if "leaderboard" in data and data["leaderboard"]:
                    lb_df = pl.DataFrame(data["leaderboard"])

                    # Format rate as percentage
                    if "wrong_bid_rate" in lb_df.columns:
                        lb_df = lb_df.with_columns(
                            (pl.col("wrong_bid_rate") * 100).round(2).alias("wrong_bid_rate_%")
                        )

                    render_aggrid(
                        lb_df, 
                        key="wrong_bid_leaderboard", 
                        height=calc_grid_height(len(lb_df)), 
                        table_name="leaderboard"
                    )
                else:
                    st.info("No auctions found matching criteria.")

            except Exception as e:
                st.error(f"Failed to load leaderboard: {e}")


# ---------------------------------------------------------------------------
# Rank Next Bids by EV – Rank next bids after an auction by Expected Value
# ---------------------------------------------------------------------------

def render_rank_by_ev():
    """Render the Rank Next Bids by EV tool.
    
    Given an auction prefix (or empty for opening bids), ranks all possible next bids
    by Expected Value (average Par score for matching deals).
    Also shows DD analysis (contract recommendations, par breakdown) for the auction.
    """
    st.header("🎯 Rank Next Bids by EV")
    st.markdown("""
    Rank possible next bids by Expected Value (EV). 
    """)
    
    # Sidebar inputs
    auction_input_raw = st.sidebar.text_input(
        "Auction Prefix",
        value="",
        help="Enter an auction prefix (e.g., '1N' for responses to 1NT), or leave empty for opening bids"
    )
    auction_input = normalize_auction_input(auction_input_raw)
    if auction_input_raw and auction_input != auction_input_raw:
        st.sidebar.caption(f"Normalized: `{auction_input}`")

    st.sidebar.markdown(
        """
        - **Empty input**: show all opening bids ranked
        - **Auction prefix**: show all next bids after that prefix (e.g. `1N`, `1H-p-3H-p`)
        """
    )
    st.sidebar.divider()
    
    max_deals = st.sidebar.number_input(
        "Max Deals",
        value=500,
        min_value=1,
        max_value=10000,
        help="Maximum deals to sample for analysis"
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    
    vul_filter = st.sidebar.selectbox(
        "Vulnerability",
        ["all", "None", "Both", "NS", "EW"],
        index=0,
        help="Filter deals by vulnerability"
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("Output Options")
    
    include_hands = st.sidebar.checkbox("Include Hands", value=True, help="Include Hand_N/E/S/W columns")
    include_scores = st.sidebar.checkbox("Include DD Scores", value=True, help="Include DD_Score columns in Deal Data table")
    
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_rank_ev"))
    
    # =========================================================================
    # Call API (single endpoint provides both bid rankings and DD analysis)
    # =========================================================================
    # Cache version: increment when API response format changes
    CACHE_VERSION = 25  # v25: force 'Which candidate bids perform best' nulls-last sort
    
    # Always fetch all columns from API; filter display based on checkboxes
    # This prevents cache busting when output options change
    payload = {
        "auction": auction_input,
        "max_deals": int(max_deals),
        "seed": seed,
        "vul_filter": vul_filter if vul_filter != "all" else None,
        "include_hands": True,  # Always fetch hands; filter display later
        "include_scores": True,  # Always fetch scores; filter display later
        "_cache_version": CACHE_VERSION,  # Busts cache when API response format changes
    }
    
    # Cache the API response to make row selection nearly instantaneous
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_rank_data(p: Dict[str, Any]) -> Dict[str, Any]:
        return api_post("/rank-bids-by-ev", p)

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_contract_ev_deals(p: Dict[str, Any]) -> Dict[str, Any]:
        return api_post("/contract-ev-deals", p)
    
    desc = f"responses to '{auction_input}'" if auction_input else "opening bids"
    
    # Use st.status for better loading UX with expandable details
    with st.status(f"🔍 Analyzing {desc}...", expanded=True) as status:
        status.write("⏳ Matching deals to bid criteria (30 seconds for opening bids)...")
        try:
            data = fetch_rank_data(payload)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"API call failed: {e}")
            return

    # Use auction_normalized (falling back to "Opening Bids") instead of auction_input
    # for all aggregated section headers.
    auction_normalized = data.get("auction_normalized", "")
    auction_display = auction_normalized if auction_normalized else "Opening Bids"
    
    elapsed_ms = data.get("elapsed_ms", 0)
    _st_info_elapsed("Rank Next Bids by EV", data)
    total_bids = data.get("total_next_bids", 0)
    total_matches = data.get("total_matches", 0)
    returned_count = data.get("returned_count", 0)
    
    # Show error or message
    if "error" in data:
        st.error(f"⚠️ {data['error']}")
        return
    
    if "message" in data:
        st.info(data["message"])
    
    opening_seat = data.get("opening_seat", "Dealer (Seat 1)")
    st.info(f"✅ Showing {total_bids} bids, {total_matches:,} matched deals — Opener: {opening_seat} in {format_elapsed(elapsed_ms)}")
    
    # -------------------------------------------------------------------------
    # Bid Rankings Table (with row selection)
    # -------------------------------------------------------------------------
    bid_rankings = data.get("bid_rankings", [])
    selected_bid = None
    
    if bid_rankings:
        st.subheader(f"🏆 Which candidate bids perform best? ({len(bid_rankings)} bids)")
        st.markdown("*Evaluate the potential success of each candidate bid. 'EV at Bid' shows the average score achieved when that bid becomes the final contract, based on all historical deals that followed this auction sequence. Click a row to see matching deals below.*")
        
        rankings_df = pl.DataFrame(bid_rankings)
        
        # (EV_Score_* columns moved out of bid_rankings to ev_all_combos_by_bid)
        
        # Select and rename columns for display
        display_cols = []
        col_map = [
            ("bid", "Bid"),
            ("auction", "Full Auction"),
            ("next_seat", "Seat"),
            ("vul", "Vul"),
            ("match_count", "Matches"),
            ("avg_par", "Avg Par Contract"),
            ("ev_score", "EV at Bid"),
            ("ev_std", "EV Std"),
        ]
        # Full Auction placed immediately after Bid above
        
        for col, alias in col_map:
            if col in rankings_df.columns:
                display_cols.append(pl.col(col).alias(alias))
        
        if display_cols:
            rankings_df = rankings_df.select(display_cols)

        # Ensure initial display is sorted by EV at Bid (AgGrid can persist old sort state)
        if "EV at Bid" in rankings_df.columns:
            rankings_df = rankings_df.sort("EV at Bid", descending=True, nulls_last=True)
        
        selected_bid_rows = render_aggrid(
            rankings_df, 
            key=f"rank_bids_rankings_ev_{CACHE_VERSION}",
            height=calc_grid_height(len(rankings_df), max_height=400), 
            table_name="bid_rankings",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            sort_model=[{"colId": "EV at Bid", "sort": "desc"}],
        )
        
        # Capture the selected bid (default to first row if none selected)
        selected_bid_name = None
        if selected_bid_rows is not None and len(selected_bid_rows) > 0:
            selected_bid = selected_bid_rows[0]
            selected_bid_name = selected_bid.get("Bid")
        elif len(bid_rankings) > 0:
            # Fallback to first row
            selected_bid = bid_rankings[0]
            selected_bid_name = selected_bid.get("bid")
            
        # ---------------------------------------------------------------------
        # Contract EV Rankings (for selected next bid only)
        # ---------------------------------------------------------------------
        if selected_bid_name:
            # Seat number (1-4) from the clicked row; used to filter contract table to that seat only.
            selected_next_seat = None
            try:
                if selected_bid_rows is not None and len(selected_bid_rows) > 0 and selected_bid is not None:
                    seat_raw = selected_bid.get("Seat")
                    selected_next_seat = int(seat_raw) if seat_raw is not None else None
                if selected_next_seat is None:
                    # Fallback to raw API key if we fell back to the first row
                    seat_raw = selected_bid.get("next_seat") if selected_bid else None
                    selected_next_seat = int(seat_raw) if seat_raw is not None else None
            except Exception:
                selected_next_seat = None

            # Show which next-bid auction is being used for this contract table
            selected_bid_auction = None
            if selected_bid_rows is not None and len(selected_bid_rows) > 0:
                if selected_bid is not None:
                    selected_bid_auction = selected_bid.get("Full Auction")
            if not selected_bid_auction:
                original_row_for_title = next((r for r in bid_rankings if r.get("bid") == selected_bid_name), None)
                selected_bid_auction = original_row_for_title.get("auction") if original_row_for_title else None
            selected_bid_auction = selected_bid_auction or selected_bid_name

            # Add sample size in title when it is a fixed value (use bid-level match_count).
            original_row_for_title = next((r for r in bid_rankings if r.get("bid") == selected_bid_name), None)
            bid_match_n = original_row_for_title.get("match_total") if original_row_for_title else None
            n_val = f"{int(bid_match_n):,}" if isinstance(bid_match_n, (int, float)) else "?"

            seat_suffix = f" (Seat {selected_next_seat})" if isinstance(selected_next_seat, int) else ""
            st.subheader(f"🥇 Best scoring contracts for deals matching auction: {selected_bid_auction}{seat_suffix}")
            st.caption(f"Expected Value (EV) for all possible final contracts, computed from the {n_val} deals that matched this specific auction.")
            
            # Extract EV_Score_ and Makes_Pct_ columns from the separate per-bid blob.
            ev_blob = (data.get("ev_all_combos_by_bid") or {}).get(selected_bid_name)
            if ev_blob:
                # Get the two rows for this bid (NV and V) from bid_rankings to extract their Avg Par Contract
                rows_for_bid = [r for r in bid_rankings if r.get("bid") == selected_bid_name]
                avg_par_map = {r.get("vul"): r.get("avg_par") for r in rows_for_bid}

                ev_data = []
                strain_names = {'N': 'NT', 'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}
                # Seat numbers are relative to dealer (Seat 1 = Dealer)
                
                # First collect all available EV scores
                for k, v in ev_blob.items():
                    if k.startswith("EV_Score_") and v is not None:
                        # Format: EV_Score_{level}{strain}_{vul}_S{seat}
                        # e.g., EV_Score_3N_NV_S1
                        parts = k.split("_")
                        if len(parts) >= 5:
                            contract_part = parts[2] # "3N"
                            vul_part = parts[3]      # "NV" or "V"
                            seat_part = parts[4]     # "S1"
                            
                            level = contract_part[0]
                            strain = contract_part[1:]
                            seat_num = int(seat_part[1:])

                            # IMPORTANT: Filter this table to the seat of the clicked bid row.
                            if isinstance(selected_next_seat, int) and seat_num != selected_next_seat:
                                continue
                            
                            strain_display = strain_names.get(strain, strain)
                            # Look up corresponding Makes %
                            makes_key = f"Makes_Pct_{contract_part}_{vul_part}_{seat_part}"
                            makes_pct = ev_blob.get(makes_key)
                            
                            ev_data.append({
                                "Contract": f"{level}{strain_display}",
                                "Seat": seat_num,
                                "Vul": vul_part,
                                "Avg Par Contract": avg_par_map.get(vul_part),
                                "EV at Bid": round(float(v), 1),
                                "Makes %": round(float(makes_pct), 1) if makes_pct is not None else None,
                                "_level": int(level),
                                "_strain": strain,
                            })
                
                if ev_data:
                    ev_df = pl.DataFrame(ev_data)
                    
                    # Use Polars vectorized "when" logic for contract type classification
                    # (Patterned after mlBridgeAugmentLib.add_contract_type)
                    ev_df = ev_df.with_columns(
                        pl.when(pl.col('_level').eq(5) & pl.col('_strain').is_in(['C', 'D'])).then(pl.lit("Game"))
                        .when(pl.col('_level').is_in([4, 5]) & pl.col('_strain').is_in(['H', 'S'])).then(pl.lit("Game"))
                        .when(pl.col('_level').is_in([3, 4, 5]) & pl.col('_strain').eq('N')).then(pl.lit("Game"))
                        .when(pl.col('_level').eq(6)).then(pl.lit("SSlam"))
                        .when(pl.col('_level').eq(7)).then(pl.lit("GSlam"))
                        .otherwise(pl.lit("Partial"))
                        .alias('Type')
                    )
                    
                    # Sort and select columns for display
                    ev_df = ev_df.sort("EV at Bid", descending=True)
                    # Keep _level/_strain only for internal use (hidden in AgGrid)
                    display_cols = ["Contract", "Seat", "Vul", "Type", "Avg Par Contract", "EV at Bid", "Makes %", "_level", "_strain"]
                    ev_df = ev_df.select([c for c in display_cols if c in ev_df.columns])

                    selected_contract_ev_rows = render_aggrid(
                        ev_df,
                        key=f"contract_ev_rankings_{selected_bid_name}",
                        height=400,
                        table_name="top_ev",
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        hide_cols=["_level", "_strain"]
                    )
                    selected_contract_ev = selected_contract_ev_rows[0] if selected_contract_ev_rows else (ev_df.to_dicts()[0] if ev_df.height > 0 else None)

                    # -----------------------------------------------------------------
                    # Deal table for Contract EV Rankings (per selected next bid)
                    # -----------------------------------------------------------------
                    if selected_contract_ev:
                        inv_strains = {'NT': 'N', 'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}
                        contract_str = selected_contract_ev.get("Contract")
                        seat_num = selected_contract_ev.get("Seat")
                        vul_state = selected_contract_ev.get("Vul")

                        level = contract_str[0] if isinstance(contract_str, str) and contract_str else ""
                        strain_alias = contract_str[1:] if isinstance(contract_str, str) and len(contract_str) > 1 else ""
                        strain = inv_strains.get(strain_alias, strain_alias)
                        contract_api = f"{level}{strain}" if level and strain else ""

                        if contract_api and vul_state and selected_bid_name:
                            deals_payload = {
                                "auction": auction_input,
                                "next_bid": selected_bid_name,
                                "contract": contract_api,
                                # declarer is ignored when seat is provided (kept for backwards compatibility)
                                "declarer": "N",
                                "seat": int(seat_num) if isinstance(seat_num, (int, float)) else None,
                                "vul": vul_state,
                                "max_deals": int(max_deals),
                                "seed": seed,
                                "include_hands": bool(include_hands),
                                "_cache_version": CACHE_VERSION,  # Bust cache on version change
                            }
                            try:
                                deals_resp = fetch_contract_ev_deals(deals_payload)
                                deals = deals_resp.get("deals", [])
                                total_m = int(deals_resp.get("total_matches", 0) or 0)
                                shown = len(deals)

                                seat_label = f"Seat {int(seat_num)}" if isinstance(seat_num, (int, float)) else "Seat ?"
                                st.subheader(f"📄 Deals matching auction: {selected_bid_auction} ({seat_label}) (Showing {shown:,} of {total_m:,})")
                                if deals:
                                    ddf = pl.DataFrame(deals)
                                    if not include_hands:
                                        drop_hand_cols = [c for c in ddf.columns if c.startswith("Hand_")]
                                        if drop_hand_cols:
                                            ddf = ddf.drop(drop_hand_cols)

                                    # Drop legacy alias columns (seat-relative DD_*/EV_* are shown instead)
                                    drop_legacy = [c for c in ["DD_Score", "EV_Score"] if c in ddf.columns]
                                    if drop_legacy:
                                        ddf = ddf.drop(drop_legacy)

                                    # Ensure stable ordering: sort by global deal index ascending.
                                    # (AgGrid can persist user sorting per-key; we also bust the key below.)
                                    if "index" in ddf.columns:
                                        try:
                                            ddf = ddf.with_columns(pl.col("index").cast(pl.Int64, strict=False))
                                        except Exception:
                                            pass
                                        try:
                                            ddf = ddf.sort("index")
                                        except Exception:
                                            pass

                                    # Seat-aware /contract-ev-deals already returns seat-relative DD_<contract> / EV_<contract>
                                    # when "seat" is provided, so no additional renaming is required here.

                                    render_aggrid(
                                        ddf,
                                        key=f"contract_ev_deals_{selected_bid_name}_{contract_api}_{seat_label}_{CACHE_VERSION}",
                                        height=calc_grid_height(len(ddf), max_height=450),
                                        table_name="contract_ev_deals",
                                        sort_model=[{"colId": "index", "sort": "asc"}],
                                    )
                                else:
                                    st.info("No deals found for this selection.")
                            except Exception as e:
                                st.warning(f"Could not load deals for Contract EV Rankings: {e}")
                else:
                    st.info("No contract-level EV data available for this bid.")
        
    else:
        st.warning("No next bids found.")
        return
    
    # =========================================================================
    # DD Analysis (from aggregated deals across all next bids)
    # =========================================================================
    st.divider()
    
    selected_contract = None
    
    # -------------------------------------------------------------------------
    # Contract Recommendations (EV Rankings)
    # -------------------------------------------------------------------------
    contract_recs = data.get("contract_recommendations", [])
    if contract_recs:
        # Include sample size in title only when it's a single fixed value across rows.
        n_hint = ""
        try:
            ns = [int(r.get("sample_size") or 0) for r in contract_recs if r.get("sample_size") is not None]
            if ns and min(ns) == max(ns) and ns[0] > 0:
                n_hint = f" (N={ns[0]:,})"
        except Exception:
            n_hint = ""

        total_matches = data.get("total_matches", 0)

        st.subheader("🏆 Which final contracts score best?")
        st.caption(f"Compare the Expected Value (EV) of all possible outcomes across the entire pool of {total_matches:,} deals after grouping by contract, declarer, vulnerability. This is without regard to any auction sequence.")
        
        rec_df = pl.DataFrame(contract_recs)
        # Keep numeric values for numeric sorting
        if "make_pct" in rec_df.columns:
            rec_df = rec_df.with_columns(
                pl.col("make_pct").alias("Makes %")
            )
        
        # Use existing columns or aliases - separate rows for each vulnerability
        display_cols = []
        col_map = [
            ("contract", "Contract"),
            ("declarer", "Declarer"),
            ("vul", "Vul"),  # Vulnerability state (NV or V)
            ("Makes %", "Makes %"),
            ("ev", "EV"),  # Expected Value for this vulnerability
            ("sample_size", "Sample"),
        ]
        for c, alias in col_map:
            if c in rec_df.columns:
                display_cols.append(pl.col(c).alias(alias))
            elif alias in rec_df.columns:
                display_cols.append(pl.col(alias))
        
        if display_cols:
            rec_df = rec_df.select(display_cols)
            selected_contract_rows = render_aggrid(
                rec_df, 
                key="dd_analysis_recommendations", 
                height=calc_grid_height(len(rec_df)), 
                table_name="recommendations",
                update_mode=GridUpdateMode.SELECTION_CHANGED
            )
            
            # Capture the selected contract if any
            if selected_contract_rows is not None and len(selected_contract_rows) > 0:
                selected_contract = selected_contract_rows[0]
    
    # -------------------------------------------------------------------------
    # Deal Data Table - Shown only for Contract Rankings by EV (default first row)
    # -------------------------------------------------------------------------
    dd_deals = data.get("dd_data", [])
    dd_data_by_bid = data.get("dd_data_by_bid", {})  # bid -> list of deal dicts (up to max_deals each)
    
    # Decode matched_by_bid from base64 if present (optimized binary encoding)
    matched_by_bid_b64 = data.get("matched_by_bid_b64", {})
    matched_by_bid: Dict[str, Set[int]] = {}
    if matched_by_bid_b64:
        for bid, b64_str in matched_by_bid_b64.items():
            if b64_str:
                indices = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint32)
                matched_by_bid[bid] = set(indices.tolist())
            else:
                matched_by_bid[bid] = set()
    else:
        # Fallback to legacy field if present
        legacy_matched = data.get("matched_by_bid", {})
        matched_by_bid = {bid: set(indices) for bid, indices in legacy_matched.items()}
    
    # Filter dd_deals based on selection (Contract Rankings by EV only)
    filtered_deals = dd_deals
    selection_msg = ""
    has_selection = bool(selected_contract)
    
    # Placeholder values for linter
    sel_level = ""
    sel_strain = ""
    sel_declarer = ""
    sel_vul = ""

    # Default to first contract row if none selected
    if contract_recs and not selected_contract:
        try:
            selected_contract = {
                "Contract": contract_recs[0].get("contract"),
                "Declarer": contract_recs[0].get("declarer"),
                "Vul": contract_recs[0].get("vul"),
            }
            has_selection = True
        except Exception:
            selected_contract = None
            has_selection = False

    if selected_contract:
        # User clicked on a row in Contract Rankings by EV
        contract_str = selected_contract.get("Contract")
        declarer = selected_contract.get("Declarer")
        vul_state = selected_contract.get("Vul")
        
        # Convert contract_str (e.g. "3NT") back to level/strain for column lookup
        inv_strains = {'NT': 'N', 'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}
        level = contract_str[0] if contract_str else ""
        strain_alias = contract_str[1:] if contract_str and len(contract_str) > 1 else ""
        strain = inv_strains.get(strain_alias, strain_alias)
        
        # Store for display
        sel_level = level
        sel_strain = strain_alias
        sel_declarer = declarer
        sel_vul = vul_state
        
        score_col = f"DD_Score_{level}{strain}_{declarer}"
        
        # Convert contract to bid format for filtering
        # e.g., "1NT" -> "1N", "4S" -> "4S"
        expected_bid = f"{level}{strain}"
        
        # Get deal indices that matched this bid's criteria (O(1) lookup)
        valid_indices = set(matched_by_bid.get(expected_bid, []))
        
        # Define vulnerability subsets for THIS declarer
        if declarer in ["N", "S"]:
            nv_vuls = ["None", "E_W"]
            v_vuls = ["N_S", "Both"]
        else:
            nv_vuls = ["None", "N_S"]
            v_vuls = ["E_W", "Both"]
            
        target_vuls = nv_vuls if vul_state == "NV" else v_vuls
        
        # For opening bids (seat 1), the opener is the dealer, so declarer should match dealer.
        # For later bids, this relationship is more complex, but for now we filter by dealer.
        # The declarer in the DD analysis represents who would play the contract.
        # For a 1N opening by seat 1:
        #   - If Dealer=N, then N opens 1N → N would declare 1NT
        #   - If Dealer=S, then S opens 1N → S would declare 1NT
        
        # Filter deals to match the same population used for the aggregated
        # Contract Rankings by EV row:
        # - vulnerability subset for the declarer's side (NV vs V)
        # - deals must have a non-null DD_Score for the selected contract when available
        #
        # IMPORTANT: do NOT filter by Dealer here. For non-opening auctions the declarer
        # is not necessarily the dealer, and the backend aggregation does not filter by Dealer.
        new_filtered = []
        for d in dd_deals:
            vul_ok = d.get("Vul") in target_vuls
            # If the score column isn't present in the sampled dd_deals (e.g. include_scores=False),
            # we can't filter on it; otherwise require non-null.
            score_ok = (d.get(score_col) is not None) if score_col in d else True
            
            if vul_ok and score_ok:
                new_filtered.append(d)
        
        filtered_deals = new_filtered
        selection_msg = f" (Stats for {level}{strain_alias} by {declarer} ({vul_state}))"
    
    if filtered_deals and has_selection:
        st.subheader(f"📈 Deals matching {sel_level}{sel_strain} by {sel_declarer} ({sel_vul}) ({len(filtered_deals)} deals)")
        
        shown_count = len(filtered_deals)
        if total_matches > shown_count and not has_selection:
            st.info(f"Showing a random sample of {shown_count:,} out of {total_matches:,} total matches in {format_elapsed(elapsed_ms)}")
        
        dd_df = pl.DataFrame(filtered_deals)
        
        # Filter columns based on output options (data is always fetched; filter display here)
        if not include_hands:
            drop_hand_cols = [c for c in dd_df.columns if c.startswith("Hand_")]
            if drop_hand_cols:
                dd_df = dd_df.drop(drop_hand_cols)
        
        if not include_scores:
            drop_score_cols = [c for c in dd_df.columns if c.startswith("DD_")]
            if drop_score_cols:
                dd_df = dd_df.drop(drop_score_cols)
        
        # Order columns sensibly
        # DD_Score and EV_Score show the score/EV for the matching BT bid specifically
        priority_cols = ["index", "Dealer", "Vul"]
        hand_cols = ["Hand_N", "Hand_E", "Hand_S", "Hand_W"] if include_hands else []
        # Keep key score columns near the front for readability
        # - DD_Score_Declarer: deal's actual contract score (if present)
        # - ParScore: par result for the deal
        score_cols = ["DD_Score", "EV_Score", "DD_Score_Declarer", "ParScore", "ParContracts"]
        
        dd_cols = data.get("dd_columns", []) if include_scores else []
        ordered_dd = []
        
        # Check if we have raw trick columns or DD_Score columns
        has_raw_tricks = any(col.startswith("DD_") and not col.startswith("DD_Score") for col in dd_cols)
        
        if has_raw_tricks:
            # Order raw trick columns by direction then strain
            for direction in ['N', 'E', 'S', 'W']:
                for strain in ['C', 'D', 'H', 'S', 'N']:
                    col = f"DD_{direction}_{strain}"
                    if col in dd_cols:
                        ordered_dd.append(col)
        else:
            # Order DD_Score columns by level then strain then direction
            for level in range(1, 8):
                for strain in ['N', 'S', 'H', 'D', 'C']:
                    for direction in ['N', 'E', 'S', 'W']:
                        col = f"DD_Score_{level}{strain}_{direction}"
                        if col in dd_cols:
                            ordered_dd.append(col)
            # Add any remaining DD_Score columns not in the order
            for col in dd_cols:
                if col not in ordered_dd:
                    ordered_dd.append(col)
        
        # Build final column order (deduplicated while preserving priority)
        final_cols: list[str] = []
        seen: set[str] = set()

        for c in priority_cols + hand_cols + score_cols + ordered_dd:
            if c in dd_df.columns and c not in seen:
                final_cols.append(c)
                seen.add(c)

        # Add any remaining columns
        for c in dd_df.columns:
            if c not in seen:
                final_cols.append(c)
                seen.add(c)

        dd_df = dd_df.select(final_cols)

        # Ensure stable ordering: sort by global deal index ascending.
        if "index" in dd_df.columns:
            try:
                dd_df = dd_df.with_columns(pl.col("index").cast(pl.Int64, strict=False))
            except Exception:
                pass
            try:
                dd_df = dd_df.sort("index")
            except Exception:
                pass
        
        render_aggrid(
            dd_df,
            key=f"dd_analysis_results_{CACHE_VERSION}",
            height=400,
            table_name="dd_results",
            sort_model=[{"colId": "index", "sort": "asc"}],
        )
    elif has_selection:
        st.info(f"No matching deals found for the selection{selection_msg}.")
    else:
        st.info("Click a row in 'Contract Rankings by EV' to see matched deals (defaults to the first row).")

    # -------------------------------------------------------------------------
    # Par Contract Statistics (Sacrifices, Sets, etc.)  [LAST]
    # -------------------------------------------------------------------------
    par_contract_stats = data.get("par_contract_stats", [])
    if par_contract_stats:
        st.subheader("📊 Par Contract Breakdown")
        st.markdown("*Distribution of par results split by vulnerability (NV vs V)*")

        par_stats_df = pl.DataFrame(par_contract_stats)
        display_cols = ["category", "count", "pct", "count_nv", "avg_nv", "ev_nv", "count_v", "avg_v", "ev_v"]
        col_aliases = {
            "category": "Category",
            "count": "Total",
            "pct": "%",
            "count_nv": "# NV",
            "avg_nv": "Par NV",
            "ev_nv": "EV NV",
            "count_v": "# V",
            "avg_v": "Par V",
            "ev_v": "EV V",
        }
        par_stats_df = par_stats_df.select([c for c in display_cols if c in par_stats_df.columns])
        par_stats_df = par_stats_df.rename({c: col_aliases.get(c, c) for c in par_stats_df.columns if c in col_aliases})
        render_aggrid(par_stats_df, key="dd_analysis_par_stats", height=calc_grid_height(len(par_stats_df), max_height=350), table_name="par_stats")


# ---------------------------------------------------------------------------
# Auction Builder – Build an auction step-by-step with BT lookups
# ---------------------------------------------------------------------------

def render_new_rules_metrics():
    """View detailed metrics for newly discovered criteria."""
    st.header("📈 New Rules Metrics")
    st.markdown("""
    Explore detailed metrics for criteria discovered during the rule learning process.
    This data is loaded from `bbo_bt_new_rules.parquet`.
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        auction_pattern = st.text_input(
            "Auction Step (e.g., 1S-p-2C)",
            value="1S-p-2C",
            help="The specific auction step to inspect (prefix + next bid)"
        )
    with col2:
        bt_index = st.number_input("BT Index (optional)", min_value=0, value=0, help="Optional bt_index filter")
    
    if not auction_pattern:
        st.info("Enter an auction step to begin.")
        return
    
    payload: dict[str, Any] = {"auction": auction_pattern}
    if bt_index > 0:
        payload["bt_index"] = int(bt_index)
        
    with st.spinner(f"Loading metrics for '{auction_pattern}'..."):
        try:
            resp = requests.post(f"{API_BASE}/new-rules-lookup", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get("found"):
                st.warning(f"No metrics found for auction: `{auction_pattern}`")
                if bt_index > 0:
                    st.caption(f"Checked both auction and bt_index={bt_index}")
                return
            
            # Display Header Info
            st.info(f"Showing metrics for: **{data['auction']}** (BT Index: {data['bt_index']}) in {format_elapsed(data.get('elapsed_ms', 0))}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Seat", data["seat"])
            with c2:
                st.metric("Pos Count", f"{data['pos_count']:,}")
            with c3:
                st.metric("Neg Count", f"{data['neg_count']:,}")
            
            # Criteria Details - Single table with boolean columns for each set
            st.subheader("📊 Criteria Details")
            
            # Gather all criteria from all sets
            base_rules = set(str(r) for r in (data.get("base_rules") or []))
            accepted = set(str(r) for r in (data.get("accepted_criteria") or []))
            rejected = set(str(r) for r in (data.get("rejected_criteria") or []))
            merged = set(str(r) for r in (data.get("merged_rules") or []))
            merged_deduped = set(str(r) for r in (data.get("merged_rules_deduped") or []))
            
            # Build lookup for metrics from criteria_details
            details = data.get("criteria_details") or []
            metrics_by_crit: dict[str, dict[str, float | None]] = {}
            for d in details:
                crit_key = str(d.get("criterion", ""))
                metrics_by_crit[crit_key] = {
                    "lift": d.get("lift"),
                    "pos_rate": d.get("pos_rate"),
                    "neg_rate": d.get("neg_rate"),
                }
            
            # Collect all unique criteria
            all_criteria = sorted(base_rules | accepted | rejected | merged | merged_deduped)
            
            if all_criteria:
                rows = []
                for crit in all_criteria:
                    metrics = metrics_by_crit.get(crit, {})
                    rows.append({
                        "Criteria": crit,
                        "Base": crit in base_rules,
                        "Accepted": crit in accepted,
                        "Rejected": crit in rejected,
                        "Merged": crit in merged,
                        "Merged (Deduped)": crit in merged_deduped,
                        "lift": metrics.get("lift"),
                        "pos_rate": metrics.get("pos_rate"),
                        "neg_rate": metrics.get("neg_rate"),
                    })
                df_rules = pl.DataFrame(rows)
                
                # Show counts
                st.caption(
                    f"**{len(all_criteria)}** unique criteria | "
                    f"Base: {len(base_rules)} | Accepted: {len(accepted)} | "
                    f"Rejected: {len(rejected)} | Merged: {len(merged)} | "
                    f"Merged (Deduped): {len(merged_deduped)}"
                )
                
                render_aggrid(
                    df_rules,
                    key="new_rules_all_sets",
                    height=calc_grid_height(len(df_rules), max_height=400),
                    table_name="new_rules_all_sets",
                )
            else:
                st.info("No rules found.")
                
        except Exception as e:
            st.error(f"Error fetching new rules metrics: {e}")


def render_auction_builder():
    """Build an auction step-by-step by selecting bids from BT-derived options."""
    st.header("🔨 Auction Builder")
    st.markdown("""
    Build an auction step-by-step. At each step, select the next bid from available 
    BT continuations. See criteria (Agg_Expr) for each seat and find matching deals.
    
    **Tip:** Pin a specific deal to see if each step's criteria match that deal.
    """)
    
    # Initialize session state for auction path
    if "auction_builder_path" not in st.session_state:
        st.session_state.auction_builder_path = []  # List of {"bid": str, "bt_index": int, "agg_expr": list}
    if "auction_builder_options" not in st.session_state:
        st.session_state.auction_builder_options = {}  # Cache of available options per prefix
    if "auction_builder_pinned_deal" not in st.session_state:
        st.session_state.auction_builder_pinned_deal = None  # Cached pinned deal data
    
    # Sidebar controls
    st.sidebar.subheader("📌 Pinned Deal")
    pinned_input = st.sidebar.text_input(
        "Deal Index or PBN",
        value="",
        placeholder="e.g., 12345 or N:AKQ.xxx.xxx.xxxx ...",
        help="Enter a deal index number or PBN string to pin. Criteria will be evaluated against this deal.",
    )
    
    # Parse and fetch pinned deal
    pinned_deal: dict | None = None
    pinned_deal_error: str | None = None
    
    if pinned_input.strip():
        # Check if it's a deal index (numeric) or PBN
        input_stripped = pinned_input.strip()
        cache_key = f"pinned_deal_{input_stripped}"
        
        if cache_key in st.session_state and st.session_state[cache_key] is not None:
            pinned_deal = st.session_state[cache_key]
        else:
            try:
                if input_stripped.isdigit():
                    # Fetch deal by index (fast-path: monotonic index cache in API)
                    idx = int(input_stripped)
                    data = api_post("/deals-by-index", {"indices": [idx], "max_rows": 1}, timeout=10)
                    _st_info_elapsed("Auction Builder: pin deal (by index)", data)
                    rows = data.get("rows", [])
                    if rows:
                        deal_data = dict(rows[0])
                        deal_data["_source"] = f"index:{idx}"
                        pinned_deal = deal_data
                    else:
                        pinned_deal_error = f"Deal index {idx} not found"
                else:
                    # Process as PBN
                    data = api_post("/process-pbn", {"pbn": input_stripped, "include_par": True, "vul": "None"}, timeout=10)
                    _st_info_elapsed("Auction Builder: pin deal (parse PBN/LIN)", data)
                    deals = data.get("deals", [])
                    if deals:
                        deal_data = dict(deals[0])
                        deal_data["_source"] = "pbn"
                        pinned_deal = deal_data
                    else:
                        pinned_deal_error = "Invalid PBN format"
                
                if pinned_deal:
                    st.session_state[cache_key] = pinned_deal
            except Exception as e:
                pinned_deal_error = str(e)
    
    # Show pinned deal info in sidebar
    if pinned_deal:
        st.sidebar.success("✅ Deal pinned")
        # Show hands compactly
        hands_str = []
        for d in ["N", "E", "S", "W"]:
            hand = pinned_deal.get(f"Hand_{d}", "")
            if hand:
                hands_str.append(f"**{d}:** {hand}")
        if hands_str:
            st.sidebar.markdown(" | ".join(hands_str[:2]))  # N, E
            st.sidebar.markdown(" | ".join(hands_str[2:]))  # S, W
        
        # Show dealer/vul if available
        dealer = pinned_deal.get("Dealer", "?")
        vul = pinned_deal.get("Vul", pinned_deal.get("Vulnerability", "?"))
        par = pinned_deal.get("ParScore", pinned_deal.get("Par_Score", "?"))
        st.sidebar.caption(f"Dealer: {dealer} | Vul: {vul} | Par: {par}")
        
        if st.sidebar.button("❌ Clear Pinned Deal"):
            # Clear all pinned deal caches
            keys_to_clear = [k for k in st.session_state if k.startswith("pinned_deal_")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()
    elif pinned_deal_error:
        st.sidebar.error(pinned_deal_error)
    
    # Checkbox to show/hide failed criteria coloring (only when deal is pinned)
    show_failed_criteria = False
    if pinned_deal:
        show_failed_criteria = st.sidebar.checkbox(
            "Color failed bidding candidates",
            value=True,
            help="Highlight bids that don't match the pinned deal's criteria"
        )
    
    st.sidebar.divider()
    
    max_matching_deals = st.sidebar.number_input(
        "Max Matching Deals",
        value=25,
        min_value=1,
        max_value=500,
        help="Maximum deals to show in matching deals list"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        value=42,
        min_value=0,
        help="Seed for reproducible deal sampling"
    )
    
    # Build current auction string from path
    current_path = st.session_state.auction_builder_path
    current_auction = "-".join([step["bid"] for step in current_path]) if current_path else ""
    current_seat = len(current_path) + 1  # Next seat to bid (1-indexed)
    
    # Check if auction is complete
    is_complete = False
    if current_path:
        last_step = current_path[-1]
        is_complete = last_step.get("is_complete", False)
    
    # Display current auction state with editable input
    st.subheader("Current Auction")
    
    # Track last applied auction to prevent render loops
    if "auction_builder_last_applied" not in st.session_state:
        st.session_state.auction_builder_last_applied = ""
    
    # Keep the buttons visually tight: put them in a compact right column with 3 sub-columns.
    # Auction textbox is narrow (2), buttons compact (2), rest is spacer (3)
    col_auction, col_btns, col_spacer = st.columns([2, 2, 3])
    with col_auction:
        # Use dynamic key based on path length to force update when bids are added
        edit_key = f"auction_builder_edit_input_{len(current_path)}"
        edited_auction = st.text_input(
            "Edit auction directly",
            value=current_auction,
            placeholder="e.g., 1C-P-1H-P (Enter to apply)",
            label_visibility="collapsed",
            key=edit_key,
        )

    with col_btns:
        # Tight button row - each button fills its narrow column
        btn_cols = st.columns([2, 2, 2, 1], gap="small")
        with btn_cols[0]:
            apply_edit = st.button("Apply", key="auction_builder_apply_edit", use_container_width=True)
        with btn_cols[1]:
            append_ppp = st.button("Pass Out", key="auction_builder_append_ppp", help="Append '-P-P-P' and apply", use_container_width=True)
        with btn_cols[2]:
            if current_path and st.button("Undo", key="auction_builder_undo", use_container_width=True):
                st.session_state.auction_builder_path.pop()
                st.session_state.auction_builder_last_applied = ""
                st.rerun()
        # btn_cols[3] is minimal spacer
    
    # Handle manual auction edit - trigger on Enter (value change) OR button click
    # But skip if we just applied this same value (prevents render loop)
    edited_normalized = normalize_auction_input(edited_auction).upper() if edited_auction else ""

    # Convenience: append trailing passes and apply immediately.
    if append_ppp:
        # Prefer current text box value; fall back to current_auction.
        base = edited_normalized or (current_auction.upper() if current_auction else "")
        if base and not base.endswith("-P-P-P"):
            base = base + "-P-P-P"
        # Update the input widget and trigger apply logic below.
        try:
            st.session_state[edit_key] = base
        except Exception:
            pass
        edited_auction = base
        edited_normalized = base
        # Ensure we don't get blocked by the "already applied" guard.
        st.session_state.auction_builder_last_applied = ""
        apply_edit = True
    already_applied = edited_normalized == st.session_state.auction_builder_last_applied
    value_changed = edited_normalized != current_auction.upper()
    
    if (value_changed or apply_edit) and not already_applied:
        edited_normalized = normalize_auction_input(edited_auction).upper()
        if edited_normalized:
            # Single batched call to resolve the entire path
            with st.spinner(f"Resolving path: {edited_normalized}..."):
                try:
                    data = api_post("/resolve-auction-path", {"auction": edited_normalized}, timeout=30)
                    _st_info_elapsed("Auction Builder: resolve path", data)
                    new_path = data.get("path", [])
                    if not new_path:
                        # Fallback for manually entered path if API returns empty
                        bids = [b.strip() for b in edited_normalized.split("-") if b.strip()]
                        new_path = [{"bid": b.upper(), "bt_index": None, "agg_expr": [], "is_complete": False} for b in bids]
                except Exception as e:
                    st.error(f"Error resolving auction path: {e}")
                    bids = [b.strip() for b in edited_normalized.split("-") if b.strip()]
                    new_path = [{"bid": b.upper(), "bt_index": None, "agg_expr": [], "is_complete": False} for b in bids]
            
            st.session_state.auction_builder_path = new_path
            st.session_state.auction_builder_options = {}  # Clear cache
            st.session_state.auction_builder_last_applied = edited_normalized  # Prevent re-trigger
            # Mark as already rehydrated (we just resolved the full path)
            st.session_state[f"_rehydrated_{edited_normalized}"] = True
            st.rerun()
        else:
            # Clear auction
            st.session_state.auction_builder_path = []
            st.session_state.auction_builder_options = {}
            st.session_state.auction_builder_last_applied = ""  # Prevent re-trigger
            # Clear rehydration tracking keys, deals counts cache, and categories cache
            for key in list(st.session_state.keys()):
                if isinstance(key, str) and key.startswith("_rehydrated_"):
                    del st.session_state[key]
            if "_deals_counts_all" in st.session_state:
                del st.session_state["_deals_counts_all"]
            if "_deals_counts_elapsed_ms" in st.session_state:
                del st.session_state["_deals_counts_elapsed_ms"]
            if "_deals_counts_steps" in st.session_state:
                del st.session_state["_deals_counts_steps"]
            if "auction_builder_bt_categories_cache" in st.session_state:
                del st.session_state["auction_builder_bt_categories_cache"]
            st.rerun()
    
    if is_complete:
        st.success("✅ Auction Complete")
    
    # Fetch available next bids using fast /list-next-bids endpoint
    def get_next_bid_options(prefix: str, force_refresh: bool = False) -> list[dict]:
        """Get available next bids from BT using the optimized list-next-bids endpoint.
        
        NOTE: The Auction Summary stores the chosen path with its agg_expr at the time of selection.
        If the API logic changes (or earlier calls timed out), those stored agg_expr lists may be empty.
        This helper supports `force_refresh=True` so we can rehydrate the path from the server.
        """
        cache_key = (prefix or "__opening__", seed)  # Include seed in cache key
        if not force_refresh and cache_key in st.session_state.auction_builder_options:
            return st.session_state.auction_builder_options[cache_key]
        
        try:
            # Use fast /list-next-bids endpoint which uses next_bid_indices for O(1) lookup
            data = api_post("/list-next-bids", {"auction": prefix or ""}, timeout=DEFAULT_API_TIMEOUT)
            # Store elapsed for display outside spinner
            st.session_state["_auction_builder_bids_elapsed_ms"] = data.get("_client_elapsed_ms") or data.get("elapsed_ms")
            
            next_bids = data.get("next_bids", [])
            
            if not next_bids:
                error = data.get("error", "")
                if error:
                    st.caption(f"🔍 {error}")
                else:
                    st.caption(f"🔍 No next bids found for: `{prefix or '(opening)'}`")
            
            # Convert to expected format
            options = []
            for item in next_bids:
                options.append({
                    "bid": item.get("bid", ""),
                    "bt_index": item.get("bt_index"),
                    "agg_expr": item.get("agg_expr", []) or [],
                    "is_complete": item.get("is_completed_auction", False),
                })
            
            # Already sorted by the API
            st.session_state.auction_builder_options[cache_key] = options
            return options
            
        except requests.exceptions.RequestException as e:
            st.error(f"API error fetching bid options: {e}")
            return []
        except Exception as e:
            st.error(f"Error fetching bid options: {e}")
            return []
    
    # Display bid selection
    if not is_complete:
        # Seat cycles 1-4, Bid Num is always increasing
        seat_1_to_4 = ((current_seat - 1) % 4) + 1
        st.markdown(f"**Select Bid for Seat {seat_1_to_4}**")
        
        # Check cache first to avoid showing spinner when cached
        cache_key = (current_auction or "__opening__", seed)
        if cache_key in st.session_state.auction_builder_options:
            options = st.session_state.auction_builder_options[cache_key]
        else:
            with st.spinner("Loading available bids..."):
                options = get_next_bid_options(current_auction)
        
        # Show elapsed time for loading bids
        bids_elapsed_ms = st.session_state.get("_auction_builder_bids_elapsed_ms")
        if bids_elapsed_ms:
            st.info(f"⏱️ Loaded {len(options)} bids in {bids_elapsed_ms/1000:.2f}s")
        
        if not options:
            st.warning("No more bids available in BT for this auction prefix.")
        else:
            # Sort options: P first, then D, then R, then rest alphabetically
            def bid_sort_key(opt):
                bid = opt.get("bid", "").upper()
                if bid == "P":
                    return (0, bid)
                elif bid == "D":
                    return (1, bid)
                elif bid == "R":
                    return (2, bid)
                else:
                    return (3, bid)
            
            sorted_options = sorted(options, key=bid_sort_key)

            # -----------------------------------------------------------------
            # Bid Categories (Phase 4): attach category names per bt_index.
            # Batched lookup + session cache. No elapsed banner here (avoid duplicates).
            # -----------------------------------------------------------------
            bt_idx_to_categories_opt: dict[int, str] = {}
            try:
                bt_indices_opt: list[int] = []
                for opt in sorted_options:
                    bt_idx = opt.get("bt_index")
                    if bt_idx is None:
                        continue
                    try:
                        bt_indices_opt.append(int(bt_idx))
                    except Exception:
                        continue
                bt_indices_opt = sorted(set(bt_indices_opt))
                if bt_indices_opt:
                    cache = st.session_state.setdefault("auction_builder_bt_categories_cache", {})
                    cache_key = ("options", tuple(bt_indices_opt))
                    if cache_key in cache:
                        bt_idx_to_categories_opt = cache[cache_key]
                    else:
                        cat_data = api_post(
                            "/bt-categories-by-index",
                            {"indices": bt_indices_opt, "max_rows": 500},
                            timeout=10,
                        )
                        rows = cat_data.get("rows") or []
                        tmp_opt: dict[int, str] = {}
                        for r in rows:
                            try:
                                bti_raw = r.get("bt_index")
                                if bti_raw is None:
                                    continue
                                bti = int(bti_raw)
                            except Exception:
                                continue
                            cats = r.get("categories_true") or []
                            tmp_opt[bti] = ", ".join(str(x) for x in cats if x)
                        bt_idx_to_categories_opt = tmp_opt
                        # Only cache if we got results (avoid caching failures)
                        if bt_idx_to_categories_opt:
                            cache[cache_key] = bt_idx_to_categories_opt
            except Exception:
                bt_idx_to_categories_opt = {}
            
            # Helper to check if pinned deal matches criteria for a bid option
            def check_pinned_match_with_failures(criteria_list: list, seat: int) -> tuple[bool, list[str]]:
                """Check if pinned deal matches all criteria. Returns (matches, failed_list).
                
                Uses evaluate_sl_criterion from bbo_handlers_common which supports:
                - Simple SL: 'SL_S >= 5'
                - Relative SL: 'SL_S >= SL_H'
                - Complex: 'SL_D >= SL_C & ((SL_D > SL_H & SL_D > SL_S) | (SL_H <= 4 & SL_S <= 4))'
                
                Also handles simple HCP/Total_Points criteria directly.
                """
                if not pinned_deal or not criteria_list:
                    return True, []
                
                dealer = pinned_deal.get("Dealer", "N")
                directions = ["N", "E", "S", "W"]
                try:
                    dealer_idx = directions.index(dealer.upper())
                except ValueError:
                    dealer_idx = 0
                direction = directions[(dealer_idx + seat - 1) % 4]
                
                # Capture pinned_deal in a local variable for type checker
                deal_data = pinned_deal  # Already checked not None above
                
                def eval_simple_criterion(crit_str: str) -> bool | None:
                    """Evaluate simple HCP/Total_Points criteria like 'HCP >= 12'."""
                    import re as re_mod
                    m = re_mod.match(r"^(HCP|Total_Points)\s*(>=|<=|>|<|==|!=)\s*(\d+)$", crit_str.strip())
                    if not m:
                        return None  # Not a simple criterion
                    var_name, op, num_str = m.groups()
                    threshold = int(num_str)
                    
                    # Get actual value
                    if var_name == "HCP":
                        val = deal_data.get(f"HCP_{direction}")
                    else:  # Total_Points
                        val = deal_data.get(f"Total_Points_{direction}")
                    
                    if val is None:
                        return False  # Missing data = fail
                    
                    val = int(val)
                    if op == ">=":
                        return val >= threshold
                    elif op == "<=":
                        return val <= threshold
                    elif op == ">":
                        return val > threshold
                    elif op == "<":
                        return val < threshold
                    elif op == "==":
                        return val == threshold
                    elif op == "!=":
                        return val != threshold
                    return None
                
                failed = []
                for crit in criteria_list:
                    crit_str = str(crit).strip()
                    if not crit_str:
                        continue
                    try:
                        # Use the robust evaluator from bbo_handlers_common (handles SL and complex)
                        result = evaluate_sl_criterion(crit_str, dealer, seat, pinned_deal, fail_on_missing=True)
                        if result is False:
                            # Annotate the failed criterion with actual values
                            annotated = annotate_criterion_with_value(crit_str, dealer, seat, pinned_deal)
                            failed.append(annotated)
                        elif result is None:
                            # Not an SL/complex criterion - try simple HCP/Total_Points
                            simple_result = eval_simple_criterion(crit_str)
                            if simple_result is False:
                                annotated = annotate_criterion_with_value(crit_str, dealer, seat, pinned_deal)
                                failed.append(annotated)
                            # If simple_result is None or True, criterion passes (unrecognized = pass)
                    except Exception:
                        continue  # Skip unparseable criteria
                
                return len(failed) == 0, failed
            
            # Build DataFrame for bid selection
            bid_rows = []
            for i, opt in enumerate(sorted_options):
                criteria_list = opt.get("agg_expr", [])
                criteria_str = "; ".join(criteria_list)
                complete_marker = " ✅" if opt.get("is_complete") else ""
                
                # Check if pinned deal matches this bid's criteria
                matches_pinned = None
                failed_criteria_str = ""
                if pinned_deal and show_failed_criteria:
                    matches, failed_list = check_pinned_match_with_failures(criteria_list, seat_1_to_4)
                    matches_pinned = matches
                    failed_criteria_str = "; ".join(failed_list) if failed_list else ""
                
                row_data = {
                    "_idx": i,
                    "_matches": matches_pinned,  # Hidden column for styling
                    "Bid Num": current_seat,
                    "Seat": seat_1_to_4,
                    "Bid": f"{opt['bid']}{complete_marker}",
                    "Criteria": criteria_str if criteria_str else "(none)",
                    "Categories": "",
                }
                # Attach categories (true flags) for this candidate's bt_index (if available)
                try:
                    bt_idx = opt.get("bt_index")
                    if bt_idx is not None:
                        row_data["Categories"] = bt_idx_to_categories_opt.get(int(bt_idx), "")
                except Exception:
                    pass
                if show_failed_criteria:
                    row_data["Failed_Criteria"] = failed_criteria_str
                bid_rows.append(row_data)
            
            bids_df = pl.DataFrame(bid_rows)
            
            # Use AgGrid with row selection and conditional styling
            pandas_df = bids_df.to_pandas()
            gb = GridOptionsBuilder.from_dataframe(pandas_df)
            gb.configure_selection(selection_mode="single", use_checkbox=False)
            gb.configure_column("_idx", hide=True)
            gb.configure_column("_matches", hide=True)
            gb.configure_column("Bid Num", width=85, minWidth=85)
            gb.configure_column("Seat", width=60, minWidth=60)
            gb.configure_column("Bid", width=80, minWidth=80)
            gb.configure_column(
                "Criteria",
                flex=1,
                minWidth=100,
                wrapText=True,
                autoHeight=True,
                tooltipField="Criteria",  # Show full content on hover
            )
            gb.configure_column(
                "Categories",
                flex=1,
                minWidth=140,
                wrapText=True,
                autoHeight=True,
                tooltipField="Categories",  # Show full content on hover
            )
            if show_failed_criteria:
                gb.configure_column("Failed_Criteria", flex=1, minWidth=120, headerName="Failed Criteria", wrapText=True, autoHeight=True)
            
            # Add row styling based on pinned deal match
            if show_failed_criteria:
                gb.configure_grid_options(
                    getRowStyle=JsCode("""
                        function(params) {
                            if (params.data._matches === true) {
                                return {'backgroundColor': '#d4edda'};
                            } else if (params.data._matches === false) {
                                return {'backgroundColor': '#f8d7da'};
                            }
                            return null;
                        }
                    """)
                )
            gb.configure_grid_options(tooltipShowDelay=300)  # Tooltips after 300ms hover
            
            grid_options = gb.build()
            
            # Track last selected to detect new clicks
            if "auction_builder_last_selected" not in st.session_state:
                st.session_state.auction_builder_last_selected = None
            
            grid_response = AgGrid(
                pandas_df,
                gridOptions=grid_options,
                height=min(200, 35 + len(bid_rows) * 28),
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                key=f"auction_builder_bids_grid_{current_seat}",
                theme="balham",
                allow_unsafe_jscode=True,  # Required for JsCode row styling
                custom_css={
                    ".ag-row": {"cursor": "pointer", "transition": "background-color 0.1s"},
                    ".ag-row-hover": {
                        "background-color": "#E8F4FF !important",
                        "border-left": "3px solid #007BFF !important"
                    },
                    ".ag-row-selected": {"background-color": "#D1E9FF !important"},
                },
            )
            
            selected_rows = grid_response.get("selected_rows")
            if selected_rows is not None and len(selected_rows) > 0:
                selected_row = selected_rows[0] if isinstance(selected_rows, list) else selected_rows.iloc[0].to_dict()
                selected_idx = int(selected_row.get("_idx", 0))
                selected_bid = sorted_options[selected_idx]["bid"]
                
                # Check if this is a new selection (different from last)
                selection_key = f"{current_auction}-{selected_bid}"
                if selection_key != st.session_state.auction_builder_last_selected:
                    st.session_state.auction_builder_last_selected = selection_key
                    selected_opt = sorted_options[selected_idx]
                    st.session_state.auction_builder_path.append(selected_opt)
                    st.session_state.auction_builder_last_applied = ""
                    st.rerun()
    
    # Summary DataFrame
    if current_path:
        def _truncate_60(s: Any) -> str:
            """Return a string truncated to 60 chars (with ...)."""
            txt = "" if s is None else str(s)
            if len(txt) <= 60:
                return txt
            return txt[:57] + "..."

        def _join_tooltip(values: Any, max_chars: int = 2000, max_items: int = 500) -> str:
            """Join values for tooltips without creating megabyte strings.

            AgGrid still serializes hidden columns to the browser, so we must cap tooltip payload size.
            """
            if not values:
                return "(none)"
            try:
                items = list(values)
            except Exception:
                items = [values]
            out_parts: list[str] = []
            total = 0
            n = 0
            for v in items:
                if v is None:
                    continue
                s = str(v)
                if not s:
                    continue
                if n >= max_items:
                    out_parts.append("… (truncated)")
                    break
                add = (2 if out_parts else 0) + len(s)
                if total + add > max_chars:
                    out_parts.append("… (truncated)")
                    break
                if out_parts:
                    out_parts.append("; ")
                    total += 2
                out_parts.append(s)
                total += len(s)
                n += 1
            return "".join(out_parts) if out_parts else "(none)"

        # Rehydrate criteria for the stored path if any step has missing data.
        # This prevents "Criteria Count = 0" when earlier API calls used older logic.
        # Track rehydration attempts to avoid repeated slow API calls
        rehydrate_key = f"_rehydrated_{current_auction}"
        already_attempted = st.session_state.get(rehydrate_key, False)
        # Note: agg_expr=[] is valid (no criteria), only rehydrate if agg_expr is None or bt_index is None
        needs_rehydrate = any((step.get("agg_expr") is None or step.get("bt_index") is None) for step in current_path)
        rehydrate_elapsed_ms: float | None = None
        if needs_rehydrate and current_auction and not already_attempted:
            st.session_state[rehydrate_key] = True  # Mark as attempted
            with st.spinner("Loading auction criteria..."):
                try:
                    # Increased timeout: DuckDB fallback can take 5-7s per token
                    data = api_post("/resolve-auction-path", {"auction": current_auction}, timeout=90)
                    new_path = data.get("path", [])
                    if new_path and len(new_path) == len(current_path):
                        st.session_state.auction_builder_path = new_path
                        current_path = new_path
                    rehydrate_elapsed_ms = data.get("_client_elapsed_ms") or data.get("elapsed_ms")
                except Exception:
                    pass

        st.subheader("📋 Auction Summary")
        if rehydrate_elapsed_ms is not None:
            st.info(f"Loaded auction criteria in {rehydrate_elapsed_ms/1000:.2f}s")
        
        # Check for cached deal stats to include in summary
        deals_cache_key = f"auction_builder_deals_{current_auction}_{max_matching_deals}_{seed}"
        cached_stats = None
        if deals_cache_key in st.session_state and st.session_state[deals_cache_key] is not None:
            cached_stats = st.session_state[deals_cache_key].get("stats")
        
        # Helper to evaluate criteria against pinned deal
        def evaluate_criteria_for_pinned(criteria_list: list, seat: int, dealer: str, deal: dict) -> tuple[bool, list[str]]:
            """Evaluate criteria against a pinned deal. Returns (all_pass, failed_criteria)."""
            if not criteria_list or not deal:
                return True, []
            
            # Map seat to direction based on dealer
            directions = ["N", "E", "S", "W"]
            try:
                dealer_idx = directions.index(dealer.upper())
            except ValueError:
                dealer_idx = 0
            # Seat 1 = dealer, seat 2 = next, etc.
            direction = directions[(dealer_idx + seat - 1) % 4]
            
            failed = []
            for crit in criteria_list:
                crit_str = str(crit).strip()
                if not crit_str:
                    continue
                
                passed = True
                fail_info = ""
                try:
                    # Parse HCP criteria: HCP >= N, HCP <= N
                    if crit_str.startswith("HCP"):
                        col_name = f"HCP_{direction}"
                        hcp_val = deal.get(col_name)
                        if hcp_val is None:
                            failed.append(f"{crit_str} [NOT FOUND]")
                            continue
                        if ">=" in crit_str:
                            threshold = int(crit_str.split(">=")[1].strip())
                            passed = hcp_val >= threshold
                            if not passed:
                                fail_info = f"HCP({hcp_val}) >= {threshold}"
                        elif "<=" in crit_str:
                            threshold = int(crit_str.split("<=")[1].strip())
                            passed = hcp_val <= threshold
                            if not passed:
                                fail_info = f"HCP({hcp_val}) <= {threshold}"
                    
                    # Parse SL (suit length) criteria: SL_S >= N, SL_H <= N, etc.
                    elif crit_str.startswith("SL_"):
                        # Column names are SL_{direction}_{suit} e.g. SL_N_H, SL_E_S
                        suit_char = crit_str[3].upper()
                        if suit_char in ("S", "H", "D", "C"):
                            col_name = f"SL_{direction}_{suit_char}"
                            sl_val = deal.get(col_name)
                            if sl_val is None:
                                failed.append(f"{crit_str} [NOT FOUND]")
                                continue
                            if ">=" in crit_str:
                                threshold = int(crit_str.split(">=")[1].strip())
                                passed = sl_val >= threshold
                                if not passed:
                                    fail_info = f"SL_{suit_char}({sl_val}) >= {threshold}"
                            elif "<=" in crit_str:
                                threshold = int(crit_str.split("<=")[1].strip())
                                passed = sl_val <= threshold
                                if not passed:
                                    fail_info = f"SL_{suit_char}({sl_val}) <= {threshold}"
                    
                    # Parse Total_Points criteria
                    elif crit_str.startswith("Total_Points"):
                        col_name = f"Total_Points_{direction}"
                        tp_val = deal.get(col_name)
                        if tp_val is None:
                            failed.append(f"{crit_str} [NOT FOUND]")
                            continue
                        if ">=" in crit_str:
                            threshold = int(crit_str.split(">=")[1].strip())
                            passed = tp_val >= threshold
                            if not passed:
                                fail_info = f"Total_Points({tp_val}) >= {threshold}"
                        elif "<=" in crit_str:
                            threshold = int(crit_str.split("<=")[1].strip())
                            passed = tp_val <= threshold
                            if not passed:
                                fail_info = f"Total_Points({tp_val}) <= {threshold}"
                    
                    # For other criteria, we can't evaluate client-side - skip
                    else:
                        continue  # Unknown criteria - don't count as pass or fail
                        
                except (ValueError, TypeError, IndexError):
                    continue  # Parse error - skip this criterion
                
                if not passed and fail_info:
                    failed.append(fail_info)
            
            return len(failed) == 0, failed
        
        summary_rows = []
        pinned_all_pass = True  # Track if all steps pass for pinned deal

        # Pre-fetch deal counts for all steps (cached globally, fetch only missing)
        if "_deals_counts_all" not in st.session_state:
            st.session_state["_deals_counts_all"] = {}
        deals_counts: dict[str, int] = st.session_state["_deals_counts_all"]
        
        # Find steps that need fetching
        missing_partials: list[str] = []
        for i in range(len(current_path)):
            partial = "-".join([s["bid"] for s in current_path[: i + 1]])
            if partial not in deals_counts:
                missing_partials.append(partial)
        
        # Fetch missing counts
        if missing_partials:
            t0_counts = time.perf_counter()
            for partial in missing_partials:
                partial_upper = partial.upper()
                # Build regex pattern for deals starting with this partial auction
                if partial_upper.endswith("-P-P-P"):
                    pattern = f"^{partial_upper}$"
                else:
                    pattern = f"^{partial_upper}.*-P-P-P$"
                try:
                    resp = api_post(
                        "/sample-deals-by-auction-pattern",
                        {"pattern": pattern, "sample_size": 1, "seed": int(seed)},  # Only need count, not deals
                        timeout=10,
                    )
                    deals_counts[partial] = resp.get("total_count", 0)
                except Exception:
                    deals_counts[partial] = 0
            # Cache elapsed time for display on subsequent renders
            st.session_state["_deals_counts_elapsed_ms"] = (time.perf_counter() - t0_counts) * 1000
            st.session_state["_deals_counts_steps"] = len(missing_partials)
        
        # Show cached elapsed time (persists across reruns)
        if "_deals_counts_elapsed_ms" in st.session_state:
            elapsed_ms = st.session_state["_deals_counts_elapsed_ms"]
            steps = st.session_state.get("_deals_counts_steps", len(current_path))
            st.info(f"⏱️ Loaded deal counts for {steps} steps in {elapsed_ms/1000:.2f}s")

        for i, step in enumerate(current_path):
            bid_num = i + 1
            seat_1_to_4 = ((bid_num - 1) % 4) + 1
            # Store tooltip text but cap it (otherwise Auction Summary can take 20s+ to render/serialize).
            criteria_str = _join_tooltip(step.get("agg_expr", []))
            # Build partial auction up to this step for deals count lookup
            partial_auction = "-".join([s["bid"] for s in current_path[:bid_num]])
            # Use pre-fetched deals count
            deals_count = deals_counts.get(partial_auction, 0)
            row = {
                "Bid Num": bid_num,
                "Seat": seat_1_to_4,
                "Bid": step["bid"],
                "BT Index": step.get("bt_index"),
                "Deals": deals_count,
                "Criteria Count": len(step.get("agg_expr", [])),
                # Display columns are truncated; *_full columns back tooltips.
                "Agg_Expr_full": criteria_str,
                "Agg_Expr": _truncate_60(criteria_str),
                "Categories_full": "",
                "Categories": "",
                "Complete": "✅" if step.get("is_complete") else "",
            }

            # Use categories from the step (populated by resolve-auction-path)
            cats = step.get("categories", [])
            if cats:
                full = ", ".join(str(x) for x in cats if x)
                row["Categories_full"] = full[:2000]
                row["Categories"] = _truncate_60(full)
            
            # Evaluate criteria against pinned deal if available
            if pinned_deal:
                dealer = pinned_deal.get("Dealer", "N")
                criteria_list = step.get("agg_expr", [])
                passes, failed = evaluate_criteria_for_pinned(criteria_list, seat_1_to_4, dealer, pinned_deal)
                if passes:
                    row["Pinned"] = "✅"
                else:
                    row["Pinned"] = "❌"
                    row["Failed_Criteria"] = "; ".join(failed)  # Show all failures
                    pinned_all_pass = False
            
            # Add stats columns only on the last row (they apply to the full auction)
            # Use Rankings-style naming: Matches, Avg Par, EV at Bid, EV Std (with NV/V suffix)
            if cached_stats and i == len(current_path) - 1:
                row["Matches_NV"] = cached_stats.get("matches_nv")
                row["Matches_V"] = cached_stats.get("matches_v")
                row["Avg Par_NV"] = cached_stats.get("avg_par_nv")
                row["Avg Par_V"] = cached_stats.get("avg_par_v")
                row["EV_NV"] = cached_stats.get("ev_nv")
                row["EV_V"] = cached_stats.get("ev_v")
                row["EV Std_NV"] = cached_stats.get("ev_std_nv")
                row["EV Std_V"] = cached_stats.get("ev_std_v")
            elif cached_stats:
                row["Matches_NV"] = None
                row["Matches_V"] = None
                row["Avg Par_NV"] = None
                row["Avg Par_V"] = None
                row["EV_NV"] = None
                row["EV_V"] = None
                row["EV Std_NV"] = None
                row["EV Std_V"] = None
            summary_rows.append(row)
        
        # Show pinned deal match summary
        if pinned_deal and current_path:
            if pinned_all_pass:
                st.success("📌 Pinned deal matches ALL criteria in this auction path")
            else:
                st.warning("📌 Pinned deal FAILS some criteria - see 'Pinned' and 'Failed_Criteria' columns")
        
        summary_df = pl.DataFrame(summary_rows)
        summary_df = order_columns(
            summary_df,
            priority_cols=[
                "Bid Num",
                "Seat",
                "Bid",
                "BT Index",
                "Deals",
                "Criteria Count",
                "Agg_Expr",
                "Categories",
                "Complete",
            ],
        )
        # Make summary grid selectable to show deals for a specific auction step
        # Use SELECTION_CHANGED to trigger rerun when row is selected
        # Use fit_columns_to_view=False to prevent column width changes on rerender
        from st_aggrid import GridUpdateMode as _GridUpdateMode
        summary_selection = render_aggrid(
            summary_df,
            key="auction_builder_summary",
            height=calc_grid_height(4),
            table_name="auction_builder_summary",
            hide_cols=["Agg_Expr_full", "Categories_full"],
            update_mode=_GridUpdateMode.SELECTION_CHANGED,
            fit_columns_to_view=False,
        )
        
        # Handle row selection - show matching deals for selected step
        # render_aggrid returns a list of selected rows directly
        selected_step_auction: str | None = None
        if summary_selection and len(summary_selection) > 0:
            sel_bid_num = summary_selection[0].get("Bid Num")
            if sel_bid_num and sel_bid_num <= len(current_path):
                # Build auction up to selected step
                selected_step_auction = "-".join([s["bid"] for s in current_path[:sel_bid_num]])
        
        # Matching Deals section - only shown when a row in Auction Summary is clicked
        if selected_step_auction is not None:
            display_auction = selected_step_auction
            st.subheader("🎯 Matching Deals")
            st.caption(f"Deals where the actual auction starts with: **{display_auction}**")
            
            # Use display_auction for cache key (changes when row is selected)
            display_deals_cache_key = f"auction_builder_deals_{display_auction}_{max_matching_deals}_{seed}"
            
            # Auto-load deals when row is selected (different from current cache)
            if display_deals_cache_key not in st.session_state:
                st.session_state[display_deals_cache_key] = None  # Trigger load
            
            # Check if we have cached deals or need to load
            if display_deals_cache_key in st.session_state and st.session_state[display_deals_cache_key] is not None:
                # Show cached deals
                cached_data = st.session_state[display_deals_cache_key]
                sample_deals = cached_data.get("sample_deals", [])
                if sample_deals:
                    elapsed = cached_data.get("elapsed_sec", 0)
                    total_count = cached_data.get("total_count", len(sample_deals))
                    st.info(f"Showing **{len(sample_deals)}** of **{total_count}** matching deals in {elapsed:.2f}s")
                    deals_df = pl.DataFrame(cached_data.get("display_rows", []))
                    render_aggrid(
                        deals_df,
                        key="auction_builder_deals",
                        height=calc_grid_height(len(deals_df), max_height=400),
                        table_name="auction_builder_deals",
                    )
                else:
                    st.info("No deals found matching this auction pattern.")
            elif display_deals_cache_key in st.session_state:
                # Row was selected, load deals
                with st.spinner("Fetching matching deals..."):
                    try:
                        # Lightweight: sample deals by actual-auction regex (no Rules/BT).
                        auction_upper = display_auction.upper()
                        if is_complete or auction_upper.endswith("-P-P-P"):
                            arena_pattern = f"^{auction_upper}$"
                        else:
                            arena_pattern = f"^{auction_upper}.*-P-P-P$"
                        deals_data = api_post(
                            "/sample-deals-by-auction-pattern",
                            {"pattern": arena_pattern, "sample_size": int(max_matching_deals), "seed": int(seed)},
                            timeout=30,
                        )
                        sample_deals = deals_data.get("deals", []) or []
                        
                        # Compute stats from matching deals, split by vulnerability
                        # Use Rankings-style naming: Matches, Avg Par, EV (at Bid), EV Std
                        scores_nv, scores_v = [], []
                        par_scores_nv, par_scores_v = [], []
                        
                        for deal in sample_deals:
                            vul = str(deal.get("Vul", "")).upper()
                            is_vul = vul not in ("NONE", "-", "", "O")  # O = None in some formats
                            
                            score = deal.get("Score")
                            par = deal.get("ParScore")
                            
                            if score is not None:
                                try:
                                    s = float(score)
                                    if is_vul:
                                        scores_v.append(s)
                                    else:
                                        scores_nv.append(s)
                                except (ValueError, TypeError):
                                    pass
                            if par is not None:
                                try:
                                    p = float(par)
                                    if is_vul:
                                        par_scores_v.append(p)
                                    else:
                                        par_scores_nv.append(p)
                                except (ValueError, TypeError):
                                    pass
                        
                        # Compute EV (average) and EV Std (standard deviation) for each vulnerability
                        def calc_std(values: list) -> float | None:
                            if len(values) < 2:
                                return None
                            mean = sum(values) / len(values)
                            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                            return round(variance ** 0.5, 1)
                        
                        stats = {
                            "matches_nv": len(scores_nv) if scores_nv else None,
                            "matches_v": len(scores_v) if scores_v else None,
                            "avg_par_nv": round(sum(par_scores_nv) / len(par_scores_nv), 1) if par_scores_nv else None,
                            "avg_par_v": round(sum(par_scores_v) / len(par_scores_v), 1) if par_scores_v else None,
                            "ev_nv": round(sum(scores_nv) / len(scores_nv), 1) if scores_nv else None,
                            "ev_v": round(sum(scores_v) / len(scores_v), 1) if scores_v else None,
                            "ev_std_nv": calc_std(scores_nv),
                            "ev_std_v": calc_std(scores_v),
                        }
                        
                        # Build display DataFrame
                        display_rows = []
                        for deal in sample_deals:
                            display_rows.append({
                                "index": deal.get("index"),
                                "Dealer": deal.get("Dealer"),
                                "Vul": deal.get("Vul"),
                                "Hand_N": deal.get("Hand_N", ""),
                                "Hand_E": deal.get("Hand_E", ""),
                                "Hand_S": deal.get("Hand_S", ""),
                                "Hand_W": deal.get("Hand_W", ""),
                                "Auction_Actual": deal.get("Auction_Actual", ""),
                                "Contract": deal.get("Contract", ""),
                                "Result": deal.get("Result", ""),
                                "Score": deal.get("Score", ""),
                                "ParScore": deal.get("ParScore", ""),
                            })
                        
                        # Cache the results including stats, timing, and total count
                        st.session_state[display_deals_cache_key] = {
                            "sample_deals": sample_deals,
                            "display_rows": display_rows,
                            "stats": stats,
                            "total_count": deals_data.get("total_count", len(sample_deals)),
                            "elapsed_sec": round((deals_data.get("_client_elapsed_ms") or deals_data.get("elapsed_ms") or 0) / 1000, 2),
                        }
                        
                        if sample_deals:
                            # Rerun to update Auction Summary with the new stats
                            st.rerun()
                        else:
                            st.info("No deals found matching this auction pattern.")
                            
                    except requests.exceptions.RequestException as e:
                        st.error(f"API error: {e}")
                    except Exception as e:
                        st.error(f"Error fetching deals: {e}")


# ---------------------------------------------------------------------------
# Main UI – function selector and controls
# ---------------------------------------------------------------------------

st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
st.sidebar.header("Settings")
func_choice = st.sidebar.selectbox(
    "Function",
    [
        "Deals by Auction Pattern",      # Primary: find deals matching auction criteria
        "Analyze Actual Auctions",       # Group deals by bid column, analyze outcomes
        "Bidding Arena",                 # Head-to-head model comparison
        "Auction Builder",               # Build auction step-by-step with BT lookups
        "Auction Criteria Debugger",     # Debug why an auction is rejected
        "New Rules Metrics",             # View detailed rule discovery metrics
        "Wrong Bid Analysis",            # Wrong bid statistics and leaderboard
        "Custom Criteria Editor",        # Manage bbo_custom_auction_criteria.csv
        "Rank Next Bids by EV",          # Rank next bids after an auction by EV
        "Analyze Deal (PBN/LIN)",        # Input a deal, find matching auctions
        "Bidding Table Explorer",        # Browse bt_df with statistics
        "Find Auction Sequences",        # Regex search bt_df
        "PBN Database Lookup",           # Check if PBN exists in deal_df
        "Random Auction Samples",        # Random completed auctions
        "Opening Bids by Deal",          # Browse deals, see opening bid matches
        "BT Seat Stats (On-the-fly)",    # Compute stats per seat from deals using bt criteria
    ],
)

# Global settings
st.sidebar.divider()

# Function descriptions (WIP)
FUNC_DESCRIPTIONS = {
    "Deals by Auction Pattern": "Find deals matching an auction pattern's criteria. Compare Rules contracts vs actual using DD scores and EV.",
    "Analyze Actual Auctions": "Group deals by their actual auction (bid column). Analyze criteria compliance, score deltas, and outcomes.",
    "Bidding Arena": "Head-to-head model comparison. Compare bidding models (Rules, Actual, NN, etc.) with DD scores, EV, and IMP differentials.",
    "Auction Builder": "Build an auction step-by-step by selecting bids from BT-derived options. See criteria per seat and find matching deals.",
    "Auction Criteria Debugger": "Debug why a specific auction is being rejected as a Rules candidate. Shows deals matching an auction pattern and which criteria are blocking.",
    "New Rules Metrics": "View detailed metrics for newly discovered criteria (lift, pos/neg rates) from bbo_bt_new_rules.parquet.",
    "Wrong Bid Analysis": "Analyze wrong bids: statistics, failed criteria summary, and leaderboard of auctions with highest wrong bid rates.",
    "Custom Criteria Editor": "Manage custom auction criteria applied as a hot-reloadable overlay. Add/edit/delete rules and apply without restarting the server.",
    "Rank Next Bids by EV": "Rank all possible next bids after an auction by EV. Empty input shows opening bids.",
    "Analyze Deal (PBN/LIN)": "Input a PBN/LIN deal and find which bidding table auctions match the hand characteristics.",
    "Bidding Table Explorer": "Browse bidding table entries with aggregate statistics (min/max ranges) for hand criteria per auction.",
    "Find Auction Sequences": "Search for auction sequences matching a regex pattern. Shows criteria per seat.",
    "PBN Database Lookup": "Check if a specific PBN deal exists in the database. Returns game results if found.",
    "Random Auction Samples": "View random completed auction sequences from the bidding table.",
    "Opening Bids by Deal": "Randomly sample deals (with optional filters) and show which opening bids match based on pre-computed criteria.",
    "BT Seat Stats (On-the-fly)": "Compute HCP / suit-length / total-points stats per seat directly from deals, using the bidding table's criteria bitmaps.",
}

# Display function description
st.info(f"**{func_choice}:** {FUNC_DESCRIPTIONS.get(func_choice, 'No description available.')}")

# Show custom criteria info by default
try:
    criteria_resp = requests.get(f"{API_BASE}/custom-criteria-info", timeout=10)
    if criteria_resp.ok:
        criteria_data = criteria_resp.json()
        stats = criteria_data.get("stats", {})
        if stats.get("criteria_file_exists"):
            with st.expander("📋 Custom Auction Criteria (from CSV)", expanded=False):
                st.caption(f"File: `{criteria_data.get('criteria_file', 'N/A')}`")
                rules = stats.get("rules", [])
                if rules:
                    st.markdown(f"**{len(rules)} rules applied** affecting {stats.get('auctions_modified', 0):,} auctions")
                    rules_df = pl.DataFrame(rules)
                    render_aggrid(rules_df, key="custom_criteria_rules", height=calc_grid_height(len(rules_df)), table_name="custom_criteria_rules")
                else:
                    st.info("No rules defined in the CSV file.")
except Exception:
    pass  # Silently ignore errors fetching criteria info

# Auction inputs for pattern-based functions
pattern = None
auction_sequence_indices: list[int] | None = None
if func_choice in ["Find Auction Sequences", "Deals by Auction Pattern"]:
    def _has_regex_metacharacters(s: str) -> bool:
        """Return True if the string contains common regex metacharacters."""
        # Note: '-' is not treated as a metacharacter here because auction strings are dash-separated.
        return any(ch in s for ch in r".^$*+?{}[]\|()")

    def _auto_anchor_if_plain(s: str) -> str:
        """If user input looks like a plain auction string, auto-anchor it as ^...$."""
        s = (s or "").strip()
        if not s:
            return s
        # If user already provided anchors, respect them.
        if s.startswith("^") or s.endswith("$"):
            return s
        # If it contains regex metacharacters, assume the user intends regex.
        if _has_regex_metacharacters(s):
            return s
        return f"^{s}$"

    if func_choice == "Find Auction Sequences":
        seq_input_mode = st.sidebar.radio(
            "Find Auction Sequences Input",
            ["Auction Regex", "bt_index list"],
            index=0,
            help="Choose ONE input method. Regex and bt_index list are mutually exclusive.",
        )
        if seq_input_mode == "Auction Regex":
            raw_pattern = st.sidebar.text_input(
                "Auction Regex",
                value="^1N-p-3N$",
                help="Trailing '-p-p-p' is assumed if not present (e.g., '1N-p-3N' → '1N-p-3N-p-p-p')",
                key="find_seq_regex",
            )
            normalized = normalize_auction_user_text(raw_pattern)
            pattern = _auto_anchor_if_plain(normalized)
            if pattern != raw_pattern:
                st.sidebar.caption(f"→ {pattern}")
        else:
            raw_idxs = st.sidebar.text_area(
                "bt_index list",
                value="",
                help="Comma/space/newline-separated bt_index values. Example: 123, 456 789",
                key="find_seq_bt_indices",
            )
            toks = re.split(r"[,\s]+", (raw_idxs or "").strip())
            parsed: list[int] = []
            for t in toks:
                if not t:
                    continue
                try:
                    parsed.append(int(t))
                except Exception:
                    continue
            auction_sequence_indices = parsed if parsed else []
            if auction_sequence_indices:
                st.sidebar.caption(f"{len(auction_sequence_indices)} index(es) provided")
    else:
        # Deals by Auction Pattern always uses regex.
        raw_pattern = st.sidebar.text_input(
            "Auction Regex",
            value="^1N-p-3N$",
            help="Trailing '-p-p-p' is assumed if not present (e.g., '1N-p-3N' → '1N-p-3N-p-p-p')",
            key="deals_by_auction_regex",
        )
        normalized = normalize_auction_user_text(raw_pattern)
        pattern = _auto_anchor_if_plain(normalized)
        if pattern != raw_pattern:
            st.sidebar.caption(f"→ {pattern}")

    st.sidebar.divider()
else:
    st.sidebar.divider()

# ---------------------------------------------------------------------------
# Dispatch to appropriate render function based on selection
# ---------------------------------------------------------------------------

match func_choice:
    case "Deals by Auction Pattern":
        render_deals_by_auction_pattern(pattern)
    case "Analyze Actual Auctions":
        render_analyze_actual_auctions()
    case "Bidding Arena":
        render_bidding_arena()
    case "Auction Builder":
        render_auction_builder()
    case "Auction Criteria Debugger":
        render_auction_criteria_debugger()
    case "New Rules Metrics":
        render_new_rules_metrics()
    case "Wrong Bid Analysis":
        render_wrong_bid_analysis()
    case "Custom Criteria Editor":
        render_custom_criteria_editor()
    case "Rank Next Bids by EV":
        render_rank_by_ev()
    case "Analyze Deal (PBN/LIN)":
        render_analyze_deal()
    case "Bidding Table Explorer":
        render_bidding_table_explorer()
    case "Find Auction Sequences":
        render_find_auction_sequences(pattern, auction_sequence_indices)
    case "BT Seat Stats (On-the-fly)":
        render_bt_seat_stats_tool()
    case "PBN Database Lookup":
        render_pbn_database_lookup()
    case "Random Auction Samples":
        render_random_auction_samples()
    case "Opening Bids by Deal":
        render_opening_bids_by_deal()
    case _:
        st.error(f"Unknown function: {func_choice}")

