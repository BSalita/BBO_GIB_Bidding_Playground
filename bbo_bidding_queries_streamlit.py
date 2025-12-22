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
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
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

# Add mlBridgeLib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlBridgeLib"))

from bbo_bidding_queries_lib import normalize_auction_pattern, normalize_auction_pattern_to_seat1


API_BASE = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Par Contract formatting (copied from plugins.bbo_handlers_common to avoid
# import path issues with mlBridgeLib dependencies)
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
    - 5♠-4♥-3♦-1♣, 5♠-4♦-3♥-1♣, 5♥-4♠-3♦-1♣, etc.
    
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
        return result, sql_query
    except Exception as e:
        # Return original df if query fails
        return df, f"-- Error: {e}\n{sql_query}"


def format_distribution_help() -> str:
    """Return help text for distribution pattern input."""
    return """**Ordered Distribution** (S-H-D-C order)

**Notations:**
- `4-3-3-3` — exact: 4♠, 3♥, 3♦, 3♣
- `4333` — compact: same as above
- `5+-3-3-2` — 5+ spades
- `3--4-4-2` — 3 or fewer spades  
- `[2-4]-3-5-3` — 2-4 spades (range)
- `2:4-3-5-3` — 2-4 spades (range)
- `5-4-x-x` — any ♦/♣

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
        # Remove the added SL columns for cleaner display
        sl_cols = [f"SL_{s}_{direction}" for s in ['S', 'H', 'D', 'C']]
        result = result.drop([c for c in sl_cols if c in result.columns])
        return result, sql_query
    except Exception as e:
        return df, f"-- Error: {e}\n{sql_query}"


def api_get(path: str) -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE}{path}")
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE}{path}", json=payload)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Surface FastAPI error details in the UI
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"{e}\nServer detail: {detail}", response=resp) from e
    return resp.json()


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
    update_mode: GridUpdateMode = GridUpdateMode.NO_UPDATE
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

    # Show SQL expander if table_name is provided
    if table_name:
        sql_query = generate_passthrough_sql(df, table_name)
        with st.expander("📝 SQL Query", expanded=False):
            st.code(sql_query, language="sql")

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
    gb = GridOptionsBuilder.from_dataframe(df.to_pandas())
    # Disable pagination entirely to allow scrolling within the fixed height
    gb.configure_pagination(enabled=False)
    # Make all columns read-only (not editable), resizable, filterable, sortable
    gb.configure_default_column(resizable=True, filter=True, sortable=True, editable=False)
    
    # Custom formatting for columns that should display as percentages
    pct_cols = [c for c in df.columns if 
                "Makes %" in c or "make_pct" in c or "Makes_Pct" in c or
                "Rate" in c or "rate" in c or "Percentage" in c or "percentage" in c or "pct" in c or
                "Frequency" in c or "%" in c]
    for col in pct_cols:
        gb.configure_column(col, valueFormatter="x !== null ? x.toFixed(1) + '%' : ''")
        
    # Enable row selection - clicking anywhere on a row highlights the entire row
    gb.configure_selection(selection_mode="single", use_checkbox=False, suppressRowClickSelection=False)
    # Explicitly set row/header heights to ensure consistent sizing
    # suppressCellFocus prevents cell-level focus (keeps row selection clean)
    gb.configure_grid_options(rowHeight=28, headerHeight=32, suppressCellFocus=True)
    grid_options = gb.build()
    
    # Make columns feel tighter (AgGrid defaults can be quite wide)
    # - Smaller minWidth prevents excess whitespace in narrow columns
    # - Slightly reduced padding tightens the visual layout
    default_col_def = grid_options.get("defaultColDef") or {}
    default_col_def.setdefault("minWidth", 60)
    default_col_def.setdefault("cellStyle", {"padding": "2px 6px"})
    grid_options["defaultColDef"] = default_col_def

    response = AgGrid(
        df.to_pandas(),
        gridOptions=grid_options,
        height=height,
        theme="balham",
        key=key,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        # Critical UX: clicking a row should only highlight it locally and NOT
        # emit selection/model updates back to Streamlit (which would cause a rerun)
        # unless specifically requested via update_mode.
        update_mode=update_mode,
        data_return_mode=DataReturnMode.AS_INPUT,
    )
    
    selected_rows: Any = response.get("selected_rows", [])
    # AgGrid returns a list of dicts or a list of dataframes depending on version
    if selected_rows is not None and hasattr(selected_rows, "to_dict"):
        return selected_rows.to_dict("records")
    return list(selected_rows) if selected_rows is not None else []

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
                st.write(f"✅ **{file_name}**: {row_count:,} rows")
            else:
                st.write(f"✅ **{file_name}**: {row_count} rows")
    
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
    if not deals:
        st.info(f"No deals matched the specified filters. ({elapsed_ms/1000:.1f}s)")
    else:
        st.success(f"Found {len(deals)} deal(s) in {elapsed_ms/1000:.1f}s")
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
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)

    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_random"))

    payload = {"n_samples": int(n_samples), "seed": seed}
    with st.spinner("Fetching bidding sequences from server. Takes about 10 seconds."):
        data = api_post("/random-auction-sequences", payload)

    samples = data.get("samples", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    if not samples:
        st.info(f"No completed auctions found. ({elapsed_ms/1000:.1f}s)")
    else:
        st.success(f"Found {len(samples)} matching auction(s) in {elapsed_ms/1000:.1f}s")
        
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
                "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
            ])
        else:
            combined_df = pl.DataFrame()
        
        # Display individual samples
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: {s['auction']}")
            seq_df = pl.DataFrame(s["sequence"]) if isinstance(s["sequence"], list) else s["sequence"]
            seq_df = order_columns(seq_df, priority_cols=[
                "index", "Auction", "Expr",
                "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
            ])
            render_aggrid(seq_df, key=f"seq_random_{i}", table_name="auction_sequences")
            st.divider()


def render_find_auction_sequences(pattern: str | None):
    """Search for auction sequences matching a regex pattern."""
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)
    allow_initial_passes = st.sidebar.checkbox(
        "Match all 4 dealer positions",
        value=True,
        help=MATCH_ALL_DEALERS_HELP,
        key="allow_initial_passes_find",
    )

    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_find"))

    payload = {"pattern": pattern, "allow_initial_passes": bool(allow_initial_passes), "n_samples": int(n_samples), "seed": seed}
    with st.spinner("Fetching auctions from server. Takes about 10-60 seconds."):
        data = api_post("/auction-sequences-matching", payload)

    # Show the effective pattern including (p-)* prefix when matching all seats
    effective_pattern = data.get('pattern', pattern)
    if allow_initial_passes and effective_pattern:
        display_pattern = prepend_all_seats_prefix(effective_pattern)
        st.caption(f"Effective pattern: {display_pattern} (4 seat variants)")
    else:
        st.caption(f"Effective pattern: {effective_pattern}")
    samples = data.get("samples", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    if not samples:
        st.info(f"No auctions matched the pattern. ({elapsed_ms/1000:.1f}s)")
    else:
        st.success(f"Found {len(samples)} matching auction(s) in {elapsed_ms/1000:.1f}s")
        
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
                "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
            ])
        else:
            combined_df = pl.DataFrame()
        
        # Display individual samples
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: {s['auction']}")
            seq_df = pl.DataFrame(s["sequence"]) if isinstance(s["sequence"], list) else s["sequence"]
            seq_df = order_columns(seq_df, priority_cols=[
                "index", "Auction", "Expr",
                "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
            ])
            render_aggrid(seq_df, key=f"seq_pattern_{i}", table_name="auction_sequences")
            st.divider()
    
    criteria_rejected = data.get("criteria_rejected", [])
    if criteria_rejected:
        st.markdown(f"🚫 **Rejected auctions due to custom criteria** ({len(criteria_rejected)} shown)")
        st.caption("Rows filtered out by bbo_custom_auction_criteria.csv rules.")
        try:
            rejected_df = pl.DataFrame(criteria_rejected)
            render_aggrid(rejected_df, key="rejected_find_auction", height=calc_grid_height(len(criteria_rejected), max_height=300), table_name="rejected_auctions")
        except Exception as e:
            st.warning(f"Could not render as table: {e}")
            st.json(criteria_rejected)


def render_deals_by_auction_pattern(pattern: str | None):
    """Find deals matching an auction pattern's criteria."""
    n_auction_samples = st.sidebar.number_input("Auction Samples", value=2, min_value=1, max_value=10)
    n_deal_samples = st.sidebar.number_input("Deal Samples per Auction", value=100, min_value=1, max_value=10000)
    allow_initial_passes = st.sidebar.checkbox(
        "Match all 4 dealer positions",
        value=True,
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

    with st.spinner("Fetching Deals Matching Auction from server. Takes about 10-180 seconds."):
        data = api_post("/deals-matching-auction", payload)

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
        st.info(f"No auctions matched the pattern. ({elapsed_ms/1000:.1f}s)")
    else:
        st.success(f"Found {len(auctions)} matching auction(s) in {elapsed_ms/1000:.1f}s")
        for i, a in enumerate(auctions, start=1):
            st.subheader(f"Auction {i}: {a['auction']}")
            
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
            total_imp = a.get("total_imp_ai", 0)
            total_deals = a.get("total_deals", 0)
            imp_ai = a.get("imp_ai_advantage", 0)
            imp_actual = a.get("imp_actual_advantage", 0)
            ai_makes = a.get("ai_makes_count", 0)
            contract_makes = a.get("contract_makes_count", 0)
            ai_par = a.get("ai_par_count", 0)
            contract_par = a.get("contract_par_count", 0)
            avg_ev_contract = a.get("avg_ev_contract")
            avg_ev_ai = a.get("avg_ev_ai")
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
                    "Contract", "Result", "Tricks", "Score", "ParScore", "DD_Score_Declarer",
                    "HCP_N", "HCP_E", "HCP_S", "HCP_W",
                ])
                st.write(f"**Matching Deals:** (showing {len(deals_df)})")
                render_aggrid(deals_df, key=f"deals_{i}", table_name="deals")
            else:
                st.info("No matching deals (criteria may be too restrictive or distribution filter removed all).")
            st.divider()
    
    # Show criteria-rejected rows for debugging (from criteria.csv)
    criteria_rejected = data.get("criteria_rejected", [])
    if criteria_rejected:
        st.markdown(f"🚫 **Rejected auctions due to custom criteria** ({len(criteria_rejected)} shown)")
        st.caption("Rows filtered out by bbo_custom_auction_criteria.csv rules.")
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
        value=True,
        help=MATCH_ALL_DEALERS_HELP,
        key="allow_initial_passes_bt",
    )
    auction_pattern = normalize_auction_pattern(raw_auction_pattern)
    # Show the effective pattern including (p-)* prefix when matching all seats
    if allow_initial_passes:
        display_pattern = prepend_all_seats_prefix(auction_pattern)
        st.sidebar.caption(f"→ {display_pattern} (4 seat variants)")
    elif auction_pattern != raw_auction_pattern:
        st.sidebar.caption(f"→ {auction_pattern}")
    
    sample_size = st.sidebar.number_input("Sample Size", value=100, min_value=1, max_value=10000, key="bt_explorer_sample")
    min_matches = st.sidebar.number_input("Min Matching Deals (0=all)", value=0, min_value=0, max_value=100000)
    
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
    }
    
    with st.spinner("Fetching Bidding Table Statistics from server..."):
        data = api_post("/bidding-table-statistics", payload)
    
    has_criteria = data.get("has_criteria", False)
    has_aggregates = data.get("has_aggregates", False)
    if not has_aggregates:
        st.warning("Aggregate statistics not available. Run bbo_bt_aggregate.py to generate them.")
    if not has_criteria:
        st.warning("Criteria columns not available. Run bt_criteria_extractor.py to generate them.")
    
    total_matches = data.get("total_matches", 0)
    rows = data.get("rows", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    
    if total_matches == 0:
        st.warning(f"No auctions match pattern: `{auction_pattern}` ({elapsed_ms/1000:.1f}s)")
        if data.get("message"):
            st.info(data["message"])
    else:
        st.success(f"Found {total_matches:,} auctions matching pattern: `{auction_pattern}` ({elapsed_ms/1000:.1f}s)")
        
        if rows:
            display_df = pl.DataFrame(rows)
            
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
    
    max_auctions = st.sidebar.number_input("Max Auctions to Show", value=50, min_value=1, max_value=500)
    
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
    st.success(f"{type_emoji} Detected **{input_type}** — Parsed {len(deals)} deal(s) in {pbn_data.get('elapsed_ms', 0)/1000:.1f}s")
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
        
        matches = match_data.get("matches", [])
        elapsed_ms = match_data.get("elapsed_ms", 0)
        
        if not matches:
            st.warning(f"No matching auctions found for seat {match_seat}. ({elapsed_ms/1000:.1f}s)")
        else:
            st.success(f"Found {len(matches)} matching auction(s) for seat {match_seat} in {elapsed_ms/1000:.1f}s")
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
        st.success(f"✅ PBN found in database! ({elapsed/1000:.1f}s, searched {total:,} rows)")
        
        # Use first returned match as the primary deal info
        deal_info = matches[0] if matches and isinstance(matches[0], dict) else {}
        if deal_info:
            st.write("**Deal Information:**")
            
            key_cols = ["PBN", "Vul", "Dealer", "Actual_Auction", "Contract", "Result", "Tricks", "Score", "ParScore", "DD_Score_Declarer"]
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
        st.warning(f"❌ PBN not found in database ({elapsed/1000:.1f}s, searched {total:,} rows)")
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
        value=True,
        help=MATCH_ALL_DEALERS_HELP,
        key="match_all_dealers_group",
    )
    auction_regex = normalize_auction_pattern(raw_auction_regex)
    # Prepend (p-)* when matching all dealers
    if match_all_dealers:
        auction_regex = prepend_all_seats_prefix(auction_regex)
        st.sidebar.caption(f"→ {auction_regex} (4 seat variants)")
    elif auction_regex != raw_auction_regex:
        st.sidebar.caption(f"→ {auction_regex}")
    
    max_groups = st.sidebar.number_input("Max Auction Groups", value=10, min_value=1, max_value=100)
    deals_per_group = st.sidebar.number_input("Deals per Group", value=100, min_value=1, max_value=1000)
    
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
        data = api_post("/group-by-bid", payload)
    
    # Canonical API schema: {pattern, auction_groups, total_matching_deals, unique_auctions, elapsed_ms}
    groups = data["auction_groups"]
    elapsed_ms = data["elapsed_ms"]
    total_auctions = data["unique_auctions"]
    total_matching_deals = data["total_matching_deals"]
    effective_pattern = data["pattern"]
    
    if not groups:
        st.warning(f"No auctions matched pattern: `{effective_pattern}` ({elapsed_ms/1000:.1f}s)")
        return
    
    deals_msg = f", {total_matching_deals:,} deals" if isinstance(total_matching_deals, int) and total_matching_deals >= 0 else ""
    st.success(f"Found {total_auctions:,} matching auctions{deals_msg}, showing {len(groups)} groups ({elapsed_ms/1000:.1f}s)")
    
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
            deals_df = order_columns(deals_df, priority_cols=[
                "PBN",
                "index", "Dealer", "Vul", "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                "Contract", "Result", "Score", "Score_MP", "Score_MP_Pct",
                "ParScore", "DD_Score_Declarer",
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
# Bidding Arena – Head-to-head model comparison
# ---------------------------------------------------------------------------

def render_bidding_arena():
    """Render the Bidding Arena: head-to-head model comparison between bidding models."""
    st.header("🏟️ Bidding Arena")
    st.markdown("Compare bidding models head-to-head with DD scores, EV, and IMP differentials.")
    
    # Get available models
    try:
        models_response = requests.get(f"{API_BASE}/bidding-models", timeout=30)
        models_response.raise_for_status()
        models_data = models_response.json()
        model_names = [m["name"] for m in models_data.get("models", [])]
    except Exception as e:
        st.error(f"Failed to get models: {e}")
        model_names = ["Rules", "Actual"]
    
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
        sample_size = st.number_input("Sample Size", min_value=10, max_value=10000, value=100, step=100)
    with col4:
        seed = st.number_input("Random Seed", min_value=0, value=42)
    with col5:
        auction_pattern = st.text_input("Auction Pattern (optional)", value="", 
            help="Regex pattern to filter auctions (e.g., '^1N-p-3N')")
    
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
                "sample_size": sample_size,
                "seed": seed,
                "auction_pattern": auction_pattern if auction_pattern else None,
                "deals_uri": deals_uri if deals_uri else None,
            }
            response = requests.post(f"{API_BASE}/bidding-arena", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Display summary
            st.subheader("📊 Summary")
            summary = data.get("summary", {})
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Deals Compared", summary.get("total_deals", 0))
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
                    with st.expander(f"By {seg_name.replace('_', ' ').title()}"):
                        if seg_data:
                            seg_df = pl.DataFrame(seg_data)
                            render_aggrid(seg_df, key=f"arena_seg_{seg_idx}", height=calc_grid_height(len(seg_df)))
                        else:
                            st.info("No segmentation data available.")

            # Sample deals
            if "sample_deals" in data and data["sample_deals"]:
                st.subheader("🎯 Sample Deal Comparisons")
                sample_df = pl.DataFrame(data["sample_deals"])
                # Order columns sensibly - Hand_[NESW] in NESW order
                priority_cols = [
                    "Dealer",
                    "Vul",
                    "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                    f"Auction_{model_a}",
                    f"Auction_{model_b}",
                    f"DD_Score_{model_a}",
                    f"DD_Score_{model_b}",
                    "IMP_Diff",
                ]
                existing = [c for c in priority_cols if c in sample_df.columns]
                remaining = [c for c in sample_df.columns if c not in existing]
                sample_df = sample_df.select(existing + remaining)
                render_aggrid(sample_df, key="arena_sample_deals", height=calc_grid_height(len(sample_df)), table_name="arena_samples")

        except Exception as e:
            st.error(f"Arena comparison failed: {e}")


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
                response = requests.post(f"{API_BASE}/wrong-bid-stats", json={}, timeout=60)
                response.raise_for_status()
                data = response.json()

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
                    render_aggrid(seat_df, key="wrong_bid_per_seat", height=calc_grid_height(len(seat_df)))

            except Exception as e:
                st.error(f"Failed to load stats: {e}")
    
    with tab2:
        st.subheader("Failed Criteria Summary")
        top_n = st.number_input("Top N Criteria", min_value=5, max_value=100, value=20, key="failed_criteria_top_n")

        # Auto-load failed criteria summary when this tab is active.
        with st.spinner("Analyzing failed criteria..."):
            try:
                response = requests.post(
                    f"{API_BASE}/failed-criteria-summary",
                    json={"top_n": int(top_n)},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                if "criteria" in data and data["criteria"]:
                    criteria_df = pl.DataFrame(data["criteria"])
                    render_aggrid(criteria_df, key="failed_criteria_grid", height=calc_grid_height(len(criteria_df)))

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
                response = requests.post(
                    f"{API_BASE}/wrong-bid-leaderboard",
                    json={"top_n": int(top_n_lb), "min_deals": int(min_deals)},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                if "leaderboard" in data and data["leaderboard"]:
                    lb_df = pl.DataFrame(data["leaderboard"])

                    # Format rate as percentage
                    if "wrong_bid_rate" in lb_df.columns:
                        lb_df = lb_df.with_columns(
                            (pl.col("wrong_bid_rate") * 100).round(2).alias("wrong_bid_rate_%")
                        )

                    render_aggrid(lb_df, key="wrong_bid_leaderboard", height=calc_grid_height(len(lb_df)), table_name="leaderboard")
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
    - **Empty input**: Show all opening bids ranked
    - **Auction prefix**: Show all responses/rebids ranked
    """)
    
    # Sidebar inputs
    auction_input = st.sidebar.text_input(
        "Auction Prefix",
        value="",
        help="Enter an auction prefix (e.g., '1N' for responses to 1NT), or leave empty for opening bids"
    )
    
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
    CACHE_VERSION = 17  # v17: broadened AgGrid percentage detection and fixed numeric sorting
    
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
    
    desc = f"responses to '{auction_input}'" if auction_input else "opening bids"
    
    # Use st.status for better loading UX with expandable details
    with st.status(f"🔍 Analyzing {desc}...", expanded=True) as status:
        status.write("⏳ Matching deals to bid criteria (this may take 1-3 minutes for opening bids)...")
        try:
            data = fetch_rank_data(payload)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"API call failed: {e}")
            return
    
    elapsed_ms = data.get("elapsed_ms", 0)
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
    st.success(f"✅ Analyzed {total_bids} bids, {total_matches:,} matched deals ({elapsed_ms/1000:.1f}s) — Opener: {opening_seat}")
    
    # Show parent BT row if present
    parent_bt = data.get("parent_bt_row")
    if parent_bt:
        st.subheader("🎴 Parent Auction")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Auction:** `{parent_bt.get('Auction', '')}`")
        with col2:
            st.write(f"**Last Bid:** `{parent_bt.get('candidate_bid', '')}`")
    elif auction_input:
        st.write(f"**Input:** `{auction_input}`")
    else:
        st.write("**Analyzing:** Opening bids (first bid of the auction)")
    
    # -------------------------------------------------------------------------
    # Bid Rankings Table (with row selection)
    # -------------------------------------------------------------------------
    bid_rankings = data.get("bid_rankings", [])
    selected_bid = None
    
    if bid_rankings:
        st.subheader(f"🏆 Next Bid Rankings ({len(bid_rankings)} bids)")
        st.markdown("*Bids ranked by average Par score (higher = better for NS). Click a row to see matching deals.*")
        
        rankings_df = pl.DataFrame(bid_rankings)
        
        # Cast all EV_Score columns to Float32 for efficiency
        ev_cols = [c for c in rankings_df.columns if c.startswith("EV_Score_")]
        if ev_cols:
            rankings_df = rankings_df.with_columns([
                pl.col(c).cast(pl.Float32) for c in ev_cols
            ])
        
        # Select and rename columns for display
        display_cols = []
        col_map = [
            ("bid", "Bid"),
            ("match_count", "Matches"),
            ("nv_count", "NV Deals"),
            ("v_count", "V Deals"),
            ("avg_par_nv", "Avg Par NV"),
            ("avg_par_v", "Avg Par V"),
            ("ev_score_nv", "EV NV"),
            ("ev_std_nv", "EV Std NV"),
            ("ev_score_v", "EV V"),
            ("ev_std_v", "EV Std V"),
        ]
        col_map.append(("auction", "Full Auction"))
        
        for col, alias in col_map:
            if col in rankings_df.columns:
                display_cols.append(pl.col(col).alias(alias))
        
        if display_cols:
            rankings_df = rankings_df.select(display_cols)
        
        selected_bid_rows = render_aggrid(
            rankings_df, 
            key="rank_bids_rankings", 
            height=calc_grid_height(len(rankings_df), max_height=400), 
            table_name="bid_rankings",
            update_mode=GridUpdateMode.SELECTION_CHANGED
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
        # RANKINGS OF CONTRACTS BY HIGHEST EV
        # ---------------------------------------------------------------------
        if selected_bid_name:
            st.subheader(f"🥇 Rankings of Contracts by Highest EV")
            
            # Extract EV_Score_ and Makes_Pct_ columns from the original bid_rankings dict
            # (since rankings_df was filtered to display only core columns)
            original_row = next((r for r in bid_rankings if r.get("bid") == selected_bid_name), None)
            
            if original_row:
                ev_data = []
                strain_names = {'N': 'NT', 'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
                seat_names = {1: 'N', 2: 'E', 3: 'S', 4: 'W'}
                
                # First collect all available EV scores
                for k, v in original_row.items():
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
                            
                            strain_display = strain_names.get(strain, strain)
                            seat_display = seat_names.get(seat_num, seat_part)
                            
                            # Look up corresponding Makes %
                            makes_key = f"Makes_Pct_{contract_part}_{vul_part}_{seat_part}"
                            makes_pct = original_row.get(makes_key)
                            
                            ev_data.append({
                                "Contract": f"{level}{strain_display}",
                                "Declarer": seat_display,
                                "Vul": vul_part,
                                "EV": round(float(v), 1),
                                "Makes %": round(float(makes_pct), 1) if makes_pct is not None else None,
                                "_level": int(level),
                                "_strain": strain
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
                    ev_df = ev_df.sort("EV", descending=True)
                    display_cols = ["Contract", "Declarer", "Vul", "Type", "EV", "Makes %"]
                    ev_df = ev_df.select([c for c in display_cols if c in ev_df.columns])
                    
                    render_aggrid(ev_df, key=f"top_ev_contracts_{selected_bid_name}", height=400, table_name="top_ev")
                else:
                    st.info("No contract-level EV data available for this bid.")
        
        # Add download button for rankings
        try:
            csv_str = rankings_df.write_csv(file=None)
            csv_bytes = ("\ufeff" + csv_str).encode("utf-8") if csv_str else b""
            safe_auction = re.sub(r'[^A-Za-z0-9_.-]+', "_", auction_input).strip("._-") if auction_input else "opening_bids"
            st.download_button(
                label="📥 Download Rankings as CSV",
                data=csv_bytes,
                file_name=f"bid_rankings_{safe_auction}.csv",
                mime="text/csv; charset=utf-8",
                key="download_rankings",
            )
        except Exception:
            pass
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
        st.subheader("🏆 Contract Rankings by EV")
        st.markdown("*Contracts ranked by EV.*")
        
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
    # Par Contract Statistics (Sacrifices, Sets, etc.)
    # -------------------------------------------------------------------------
    par_contract_stats = data.get("par_contract_stats", [])
    if par_contract_stats:
        st.subheader("📊 Par Contract Breakdown")
        st.markdown("*Distribution of par results split by vulnerability (NV vs V)*")
        
        par_stats_df = pl.DataFrame(par_contract_stats)
        # Reorder columns with NV/V split for both Avg Score and EV
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
    
    # -------------------------------------------------------------------------
    # Deal Data Table - Shown when a bid or contract is selected
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
    
    # Filter dd_deals based on selection (bid from Next Bid Rankings OR contract from Contract Rankings)
    filtered_deals = dd_deals
    selection_msg = ""
    has_selection = selected_bid or selected_contract
    
    if selected_bid:
        # User clicked on a row in Next Bid Rankings
        # Use dd_data_by_bid which has up to max_deals per bid
        clicked_bid = selected_bid.get("Bid", "")
        full_auction = selected_bid.get("Full Auction", clicked_bid)
        
        # Get the pre-sampled deals for this specific bid (up to max_deals)
        filtered_deals = dd_data_by_bid.get(clicked_bid, [])
        
        # Show how many deals matched vs how many are shown
        total_matched = len(matched_by_bid.get(clicked_bid, []))
        shown_count = len(filtered_deals)
        if total_matched > shown_count:
            selection_msg = f" (Showing {shown_count} of {total_matched} deals for auction: {full_auction})"
        else:
            selection_msg = f" (Deals matching auction: {full_auction})"
    
    elif selected_contract:
        # User clicked on a row in Contract Rankings by EV
        contract_str = selected_contract.get("Contract")
        declarer = selected_contract.get("Declarer")
        vul_state = selected_contract.get("Vul")
        
        # Convert contract_str (e.g. "3NT") back to level/strain for column lookup
        inv_strains = {'NT': 'N', '♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C'}
        level = contract_str[0] if contract_str else ""
        strain_alias = contract_str[1:] if contract_str and len(contract_str) > 1 else ""
        strain = inv_strains.get(strain_alias, strain_alias)
        
        score_col = f"DD_Score_{level}{strain}_{declarer}"
        
        # Convert contract to bid format for filtering
        # e.g., "1NT" -> "1N", "4♠" -> "4S"
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
        
        # Filter deals: 
        # 1. Deal index must be in valid_indices (matched this bid's criteria)
        # 2. Dealer must match the clicked declarer (for opening bids)
        # 3. Vulnerability must match
        # 4. DD_Score column must be non-null
        new_filtered = []
        for d in dd_deals:
            idx = d.get("index")
            bid_ok = idx in valid_indices if valid_indices else True  # Fallback if no matched_by_bid
            dealer_ok = d.get("Dealer") == declarer  # Opener (dealer) must match declarer
            vul_ok = d.get("Vul") in target_vuls
            score_ok = d.get(score_col) is not None
            
            if bid_ok and dealer_ok and vul_ok and score_ok:
                new_filtered.append(d)
        
        filtered_deals = new_filtered
        bid_filter_msg = f", bid={expected_bid}, dealer={declarer}" if valid_indices else ""
        selection_msg = f" (Stats for {level}{strain_alias} by {declarer} ({vul_state}){bid_filter_msg})"
    
    if filtered_deals and has_selection:
        st.subheader(f"📈 Deal Data ({len(filtered_deals)} deals){selection_msg}")
        
        shown_count = len(filtered_deals)
        if total_matches > shown_count and not has_selection:
            st.warning(f"⚠️ Showing a random sample of {shown_count:,} out of {total_matches:,} total matches.")
        
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
        score_cols = ["DD_Score", "EV_Score", "ParScore", "ParContracts"]
        
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
        
        # Build final column order
        final_cols = []
        for c in priority_cols + hand_cols + score_cols:
            if c in dd_df.columns:
                final_cols.append(c)
        final_cols.extend(ordered_dd)
        
        # Add any remaining columns (like DD_Score columns)
        for c in dd_df.columns:
            if c not in final_cols:
                final_cols.append(c)
        
        dd_df = dd_df.select([c for c in final_cols if c in dd_df.columns])
        
        # Add download button for deal data
        # Convert list/nested columns to strings for CSV export (CSV doesn't support nested data)
        csv_df = dd_df.clone()
        for col_name in csv_df.columns:
            col_dtype = csv_df[col_name].dtype
            dtype_str = str(col_dtype)
            # Check for nested types: List, Array, Struct - cast all to string
            if any(x in dtype_str for x in ["List", "Array", "Struct"]):
                # For list of strings, try join; otherwise cast to string repr
                try:
                    if "List(String)" in dtype_str or "List(Utf8)" in dtype_str:
                        csv_df = csv_df.with_columns(pl.col(col_name).list.join(", ").alias(col_name))
                    else:
                        # Cast entire column to string representation
                        csv_df = csv_df.with_columns(pl.col(col_name).cast(pl.Utf8).alias(col_name))
                except Exception:
                    # If casting fails, convert via Python repr
                    csv_df = csv_df.with_columns(
                        pl.col(col_name).map_elements(lambda x: str(x) if x is not None else "", return_dtype=pl.Utf8).alias(col_name)
                    )
        try:
            csv_str = csv_df.write_csv(file=None)
        except Exception:
            # Fallback: drop any remaining nested columns
            simple_cols = [c for c in csv_df.columns if not any(x in str(csv_df[c].dtype) for x in ["List", "Struct", "Array"])]
            csv_str = csv_df.select(simple_cols).write_csv(file=None) if simple_cols else ""
        
        # Ensure UTF-8 encoding with BOM for Excel compatibility
        csv_bytes = ("\ufeff" + csv_str).encode("utf-8") if csv_str else b""
        safe_auction = re.sub(r'[^A-Za-z0-9_.-]+', "_", str(data.get("auction_normalized", "auction"))).strip("._-") or "auction"
        st.download_button(
            label="📥 Download Deal Data as CSV",
            data=csv_bytes,
            file_name=f"dd_analysis_{safe_auction}.csv",
            mime="text/csv; charset=utf-8",
            key="download_deals",
        )
        
        render_aggrid(dd_df, key="dd_analysis_results", height=400, table_name="dd_results")
    elif has_selection:
        st.info(f"No matching deals found for the selection{selection_msg}.")
    else:
        st.info("Click a row in 'Next Bid Rankings' or 'Contract Rankings by EV' to see matched deals.")


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
        "Bidding Arena",                 # NEW: Head-to-head model comparison
        "Wrong Bid Analysis",            # NEW: Wrong bid statistics and leaderboard
        "Rank Next Bids by EV",               # Rank next bids after an auction by EV
        "Analyze Deal (PBN/LIN)",        # Input a deal, find matching auctions
        "Bidding Table Explorer",        # Browse bt_df with statistics
        "Find Auction Sequences",        # Regex search bt_df
        "PBN Database Lookup",           # Check if PBN exists in deal_df
        "Random Auction Samples",        # Random completed auctions
        "Opening Bids by Deal",          # Browse deals, see opening bid matches
        "BT Seat Stats (On-the-fly)",    # Compute stats per seat from deals using bt criteria
    ],
)

# Function descriptions (WIP)
FUNC_DESCRIPTIONS = {
    "Deals by Auction Pattern": "Find deals matching an auction pattern's criteria. Compare Rules contracts vs actual using DD scores and EV.",
    "Analyze Actual Auctions": "Group deals by their actual auction (bid column). Analyze criteria compliance, score deltas, and outcomes.",
    "Bidding Arena": "Head-to-head model comparison. Compare bidding models (Rules, Actual, NN, etc.) with DD scores, EV, and IMP differentials.",
    "Wrong Bid Analysis": "Analyze wrong bids: statistics, failed criteria summary, and leaderboard of auctions with highest wrong bid rates.",
    "Rank Next Bids by EV": "Rank all possible next bids after an auction by EV. Empty input shows opening bids.",
    "Analyze Deal (PBN/LIN)": "Input a PBN/LIN deal and find which bidding table auctions match the hand characteristics.",
    "Bidding Table Explorer": "Browse bidding table entries with aggregate statistics (min/max ranges) for hand criteria per auction.",
    "Find Auction Sequences": "Search for auction sequences matching a regex pattern. Shows criteria per seat.",
    "PBN Database Lookup": "Check if a specific PBN deal exists in the database. Returns game results if found.",
    "Random Auction Samples": "View random completed auction sequences from the bidding table.",
    "Opening Bids by Deal": "Browse deals by index and see which opening bids match based on pre-computed criteria.",
    "BT Seat Stats (On-the-fly)": "Compute HCP / suit-length / total-points stats per seat directly from deals, using the bidding table's criteria bitmaps.",
}

# Display function description
st.info(f"**{func_choice}:** {FUNC_DESCRIPTIONS.get(func_choice, 'No description available.')}")

# Auction Regex input - shown for pattern-based functions
pattern = None
if func_choice in ["Find Auction Sequences", "Deals by Auction Pattern"]:
    raw_pattern = st.sidebar.text_input("Auction Regex", value="^1N-p-3N$",
        help="Trailing '-p-p-p' is assumed if not present (e.g., '1N-p-3N' → '1N-p-3N-p-p-p')")
    pattern = normalize_auction_pattern(raw_pattern)
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
    case "Wrong Bid Analysis":
        render_wrong_bid_analysis()
    case "Rank Next Bids by EV":
        render_rank_by_ev()
    case "Analyze Deal (PBN/LIN)":
        render_analyze_deal()
    case "Bidding Table Explorer":
        render_bidding_table_explorer()
    case "Find Auction Sequences":
        render_find_auction_sequences(pattern)
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

