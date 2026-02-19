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
import streamlit.components.v1 as st_components
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode
from st_aggrid.shared import DataReturnMode
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
from typing import Any, Dict, List, Literal, Set

import mlBridge.mlBridgeLib as mlBridgeLib

from bbo_custom_questions_lib import (
    load_custom_questions_csv,
    sync_questions_into_docs,
    CustomQuestion,
    file_sha1,
)

from bbo_bidding_queries_lib import (
    is_regex_pattern,
    normalize_auction_input,
    normalize_auction_user_text,
    format_elapsed,
    parse_contract_from_auction,
    get_declarer_for_auction,
    get_ai_contract,
    get_dd_score_for_auction,
    get_dd_tricks_for_auction,
    get_ev_for_auction,
    get_ev_for_auction_pre,
    is_auction_valid,
    validate_auction,
    calculate_imp,
)

# Import helpers from handlers
from plugins.bbo_handlers_common import (
    annotate_criterion_with_value,
    format_par_contracts as _format_par_contracts,
)

from bbo_explanation_lib import (
    compute_partner_major_game_commit_adjustment,
    compute_pass_signoff_bonus,
    compute_post_game_slam_gate_adjustment,
    compute_rebiddable_major_game_bonus,
    compute_non_rebiddable_suit_rebid_penalty,
    compute_eeo_from_bid_details,
    compute_guardrail_penalty,
    expected_from_hist,
    hand_controls,
)
from bbo_hand_eval_lib import (
    estimate_partnership_tricks,
    estimate_quick_losers,
)


API_BASE = "http://127.0.0.1:8000"


# Shared constants
SEAT_ROLES = {1: "Opener/Dealer", 2: "LHO", 3: "Partner", 4: "RHO"}
MATCH_ALL_DEALERS_HELP = (
    "When checked, your pattern matches the auction regardless of which seat deals first (expands to 4× matches). "
    "For example, '1N-p-3N' finds this auction whether North, East, South, or West is dealer."
)

# Auction Builder Notes (CSV-backed, local)
AUCTION_BUILDER_NOTES_MAX_CHARS = 200
AUCTION_BUILDER_NOTES_CSV = pathlib.Path(__file__).resolve().parent / "data" / "auction_builder_notes.csv"
AUCTION_BUILDER_LAST_PIN_TXT = pathlib.Path(__file__).resolve().parent / "data" / "auction_builder_last_pinned_deal_index.txt"
CUSTOM_QUESTIONS_CSV = pathlib.Path(__file__).resolve().parent / "data" / "bbo_custom_questions.csv"


def _auction_builder_load_last_pinned_deal_index() -> int:
    """Load last pinned deal index from disk (0 if missing/invalid)."""
    try:
        if not AUCTION_BUILDER_LAST_PIN_TXT.exists():
            return 0
        raw = AUCTION_BUILDER_LAST_PIN_TXT.read_text(encoding="utf-8").strip()
        v = int(raw) if raw else 0
        return v if v > 0 else 0
    except Exception:
        return 0


def _auction_builder_save_last_pinned_deal_index(idx: int) -> None:
    """Persist last pinned deal index to disk (UTF-8)."""
    try:
        v = int(idx)
    except Exception:
        return
    if v <= 0:
        return
    AUCTION_BUILDER_LAST_PIN_TXT.parent.mkdir(parents=True, exist_ok=True)
    AUCTION_BUILDER_LAST_PIN_TXT.write_text(str(v), encoding="utf-8")


def _auction_builder_notes_empty_df() -> pl.DataFrame:
    return pl.DataFrame(
        {"deal_index": [], "note": [], "updated_at": []},
        schema={"deal_index": pl.Int64, "note": pl.Utf8, "updated_at": pl.Utf8},
    )


def _auction_builder_load_notes_df(path: pathlib.Path) -> pl.DataFrame:
    """Load notes CSV. Missing file => empty DataFrame."""
    if not path.exists():
        return _auction_builder_notes_empty_df()
    df = pl.read_csv(path, infer_schema_length=1000)
    # Defensive schema normalization (handles older/hand-edited files)
    if "deal_index" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("deal_index"))
    if "note" not in df.columns:
        df = df.with_columns(pl.lit("").cast(pl.Utf8).alias("note"))
    if "updated_at" not in df.columns:
        df = df.with_columns(pl.lit("").cast(pl.Utf8).alias("updated_at"))
    return (
        df.select(["deal_index", "note", "updated_at"])
        .with_columns(pl.col("deal_index").cast(pl.Int64), pl.col("note").cast(pl.Utf8), pl.col("updated_at").cast(pl.Utf8))
        .filter(pl.col("deal_index").is_not_null())
    )


def _auction_builder_notes_map(df: pl.DataFrame) -> dict[int, str]:
    out: dict[int, str] = {}
    for r in df.to_dicts():
        try:
            di = int(r.get("deal_index"))  # type: ignore[arg-type]
        except Exception:
            continue
        out[di] = str(r.get("note") or "")
    return out


def _auction_builder_atomic_write_csv(df: pl.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Ensure UTF-8 encoding even with non-ASCII notes.
    with tmp.open("w", encoding="utf-8", newline="") as f:
        df.write_csv(f)
    tmp.replace(path)


def auction_builder_upsert_or_delete_note(*, deal_index: int, note: str) -> None:
    """Upsert a note for deal_index. Empty note deletes the row."""
    note_clean = str(note or "").strip()
    if len(note_clean) > AUCTION_BUILDER_NOTES_MAX_CHARS:
        raise ValueError(f"Note must be <= {AUCTION_BUILDER_NOTES_MAX_CHARS} characters.")

    df = _auction_builder_load_notes_df(AUCTION_BUILDER_NOTES_CSV)
    df = df.filter(pl.col("deal_index") != int(deal_index))
    if note_clean:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        df = pl.concat(
            [
                df,
                pl.DataFrame([{"deal_index": int(deal_index), "note": note_clean, "updated_at": now}]),
            ],
            how="vertical",
        )
    df = df.sort("deal_index")
    _auction_builder_atomic_write_csv(df, AUCTION_BUILDER_NOTES_CSV)


def prepend_all_seats_prefix(pattern: str) -> str:
    """Prepend (p-)* to a pattern for matching all 4 dealer positions."""
    if pattern.startswith("^"):
        return "^(p-)*" + pattern[1:]
    return "(p-)*" + pattern


def calc_grid_height(n_rows: int, max_height: int = 400, row_height: int = 35, header_height: int = 50) -> int:
    """Calculate appropriate grid height based on row count."""
    return min(max_height, header_height + n_rows * row_height)


def render_deal_diagram(
    deal: dict[str, Any],
    title: str | None = None,
    show_header: bool = True,
    width_ratio: tuple[int, int] = (3, 7),
) -> None:
    """Render a bridge deal in traditional cross/diagram layout.
    
    Args:
        deal: Dict with keys Hand_N, Hand_E, Hand_S, Hand_W, and optionally
              Dealer, Vul/Vulnerability, ParScore/Par_Score
        title: Optional section title (e.g., "📋 Current Deal"). If None, no header shown.
        show_header: Whether to show the Dealer/Vul/Par caption
        width_ratio: Column ratio [deal_width, spacer_width] for layout (default 30%/70%)
    """
    def _format_hand_suits(hand_str: str) -> str:
        """Format a hand string (e.g., 'AKQ.JT9.876.5432') into suit lines with symbols."""
        if not hand_str:
            return ""
        suits = str(hand_str).split(".")
        symbols = ["♠", "♥", "♦", "♣"]
        lines = []
        for i, sym in enumerate(symbols):
            cards = suits[i] if i < len(suits) else ""
            cards = cards if cards and cards != "-" else "—"
            lines.append(f"{sym} {cards}")
        return "\n".join(lines)
    
    if title:
        st.subheader(title)
    
    hand_n = deal.get("Hand_N", "")
    hand_e = deal.get("Hand_E", "")
    hand_s = deal.get("Hand_S", "")
    hand_w = deal.get("Hand_W", "")
    
    # Wrap the cross layout in a narrow bordered container
    deal_col, _ = st.columns(list(width_ratio))
    with deal_col:
        with st.container(border=True):
            if show_header:
                dealer = deal.get("Dealer", "?")
                vul = deal.get("Vul", deal.get("Vulnerability", "?"))
                par = deal.get("ParScore", deal.get("Par_Score", "?"))
                score = deal.get("Score", deal.get("score", "—"))
                st.caption(f"Dealer: {dealer} | Vul: {vul} | Par: {par} | Score: {score}")
            
            # Cross layout: North at top center
            col_left, col_north, col_right = st.columns([1, 2, 1])
            with col_north:
                st.markdown("**North**")
                st.text(_format_hand_suits(hand_n))
            
            # West on left, East on right
            col_west, col_east = st.columns([1, 1])
            with col_west:
                st.markdown("**West**")
                st.text(_format_hand_suits(hand_w))
            with col_east:
                st.markdown("**East**")
                st.text(_format_hand_suits(hand_e))
            
            # South at bottom center
            col_left2, col_south, col_right2 = st.columns([1, 2, 1])
            with col_south:
                st.markdown("**South**")
                st.text(_format_hand_suits(hand_s))


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


def _is_underspecified_criteria(criteria_list: list[str]) -> bool:
    """Check if criteria list is underspecified (missing both HCP and Total_Points).
    
    Returns True if neither 'HCP' nor 'Total_Points' appears in any criteria expression.
    This filters out BT entries that only have shape-based criteria (like 'Biddable_H')
    without point-count requirements, which often produce ridiculous bid recommendations.
    """
    if not criteria_list:
        return False  # Empty criteria is handled separately by has_empty_criteria
    for crit in criteria_list:
        crit_upper = str(crit).upper()
        if "HCP" in crit_upper or "TOTAL_POINTS" in crit_upper:
            return False
    return True


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
    
    for i, (part, suit) in enumerate(list(zip(parts, suits))):
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
                    for i, (suit, expected_len) in enumerate(list(zip(suits, perm))):
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
                    descriptions.append(f"shape={''.join([str(x) for x in shape])}")
    
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
        # Cast to DataFrame to satisfy type checker (duckdb.pl() may return InProcessQuery)
        result_df: pl.DataFrame = result if isinstance(result, pl.DataFrame) else pl.DataFrame(result)
        return result_df, sql_query
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


def _fetch_auction_pattern_counts(
    auctions: list[str],
    cache: dict[str, int],
    *,
    pattern_style: Literal["bid_option", "completed_only"] = "completed_only",
    timeout: int = 15,
    max_patterns: int = 1000,
) -> None:
    """Fetch pattern-based deal counts for the given auctions and update cache in place.

    pattern_style:
      - "bid_option": pattern is auction with '-' -> '-*' and trailing '*' (match any completion).
      - "completed_only": pattern is ^auction$ if auction ends with -P-P-P, else ^auction.*-P-P-P$
    """
    missing = [a for a in auctions if a not in cache]
    if not missing:
        return
    # Guardrail: avoid very large regex batches that can stall the Auction Builder UI.
    # We still return deterministic values for skipped entries.
    if len(missing) > int(max_patterns):
        for a in missing[int(max_patterns):]:
            cache[a] = 0
        missing = missing[: int(max_patterns)]
    auction_to_pattern: dict[str, str] = {}
    for a in missing:
        a_u = (a or "").strip().upper()
        if pattern_style == "bid_option":
            pat = a_u.replace("-", "-*") + "*" if a_u else "*"
        else:
            if a_u.endswith("-P-P-P"):
                pat = f"^{a_u}$"
            else:
                pat = f"^{a_u}.*-P-P-P$"
        auction_to_pattern[a] = pat
    patterns = list(dict.fromkeys(auction_to_pattern.values()))
    try:
        resp = api_post("/auction-pattern-counts", {"patterns": patterns}, timeout=timeout)
        counts: dict[str, int] = resp.get("counts", {}) or {}
        for auc, pat in auction_to_pattern.items():
            cache[auc] = int(counts.get(pat, 0) or 0)
    except Exception:
        for a in missing:
            cache[a] = 0


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


def to_aggrid_safe_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas without Arrow-backed types.
    
    streamlit-aggrid's JS frontend uses an older Arrow library that can't
    deserialize newer Arrow types like LargeUtf8. This function ensures
    all columns use numpy/object dtypes that serialize correctly.
    """
    # Disable Arrow extension arrays to get regular numpy dtypes
    try:
        pandas_df = df.to_pandas(use_pyarrow_extension_array=False)
    except TypeError:
        # Older Polars versions don't have this parameter
        pandas_df = df.to_pandas()
    
    # Additional fallback: convert any remaining Arrow-backed or string columns
    for col in pandas_df.columns:
        dtype_name = str(pandas_df[col].dtype).lower()
        if "pyarrow" in dtype_name or "arrow" in dtype_name:
            pandas_df[col] = pandas_df[col].astype(object)
        elif pandas_df[col].dtype.name in ("string", "String"):
            pandas_df[col] = pandas_df[col].astype(object)
    
    return pandas_df


def render_aggrid(
    records: Any,
    key: str,
    height: int | None = None,
    table_name: str | None = None,
    update_on: list[str] | None = None,
    sort_model: list[dict[str, Any]] | None = None,
    hide_cols: list[str] | None = None,
    show_copy_panel: bool = False,
    copy_panel_default_col: str | None = None,
    # IMPORTANT: default is FIT_CONTENTS (no squishing to force-fit page width).
    # Set True only for small/compact tables where forcing all columns into view is desirable.
    fit_columns_to_view: bool = False,
    show_sql_expander: bool = True,
    # Optional row styling based on a boolean column (e.g., "_passes")
    # If set, rows will be colored green (True) or red (False)
    row_pass_fail_col: str | None = None,
    # Optional row styling based on a bucket column (e.g., "Source" or "Bucket")
    # If set, rows will be colored: Valid=green, Rejected=yellow, Invalid=red
    row_bucket_col: str | None = None,
    # Optional list of columns to show cell value as tooltip (for long text that gets truncated)
    tooltip_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Render a list-of-dicts or DataFrame using AgGrid.
    
    Args:
        records: DataFrame or list-of-dicts to display
        key: Unique key for the AgGrid component
        height: Optional height in pixels
        table_name: Optional table name for SQL display (shows SQL expander if provided)
        update_on: List of events to trigger updates (e.g., ["selectionChanged"]). Default: [] (no updates)
    
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

    # Show SQL expander only if both the parameter is True AND the global setting is True
    if show_sql_expander and st.session_state.get("show_sql_queries", True):
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
    # rowHeight=25, headerHeight=25, plus border/scrollbar buffer.
    if height is None:
        n_rows = len(df)
        ROW_HEIGHT = 25
        HEADER_HEIGHT = 25
        BUFFER = 10  # borders, scrollbar track, etc.
        # Cap at ~10 rows before scrolling kicks in.
        height = min(10 * ROW_HEIGHT + HEADER_HEIGHT + BUFFER,
                     n_rows * ROW_HEIGHT + HEADER_HEIGHT + BUFFER)

    # Build the grid options
    # Use helper to avoid Arrow LargeUtf8 incompatibility with streamlit-aggrid's JS frontend
    pandas_df = to_aggrid_safe_pandas(df)
    
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
    
    # Optionally enable tooltips for columns with long text
    if tooltip_cols:
        for c in tooltip_cols:
            if c in df.columns:
                gb.configure_column(c, tooltipField=c)
        
    # Enable row selection - clicking anywhere on a row highlights the entire row
    gb.configure_selection(selection_mode="single", use_checkbox=False, suppressRowClickSelection=False)
    # Explicitly set row/header heights to ensure consistent sizing
    # suppressCellFocus prevents cell-level focus (keeps row selection clean) but can interfere with copy UX.
    # Tighten specific columns that tend to be too wide (must be before build())
    # Width formula: ~8px/char + ~55px for filter icon and padding
    tight_cols = {
        # Matching Deals columns (content-based widths, header check below)
        "Score": 80, "ParScore": 120, "Result": 85, "Contract": 105,
        "Dealer": 85, "Vul": 75, "index": 95, "Auction_Actual": 165,
        "Hand_N": 135, "Hand_E": 135, "Hand_S": 135, "Hand_W": 135,
        # Pinned deal invariants
        "ParContracts": 155, "EV": 75,
        # Auction Summary columns
        "Bid Num": 115, "Direction": 125, "Seat": 90, "Bid": 80, "BT Index": 120,
        "Matches": 115, "Deals": 100,
        "Avg_EV": 105, "EV_NV": 100, "EV_V": 90,
        "Avg_Par": 110, "Par_NV": 100, "Par_V": 90,
        "Criteria Count": 150, "Complete": 110,
        # Rankings-style stats columns (NV/V split)
        "Matches_NV": 130, "Matches_V": 125,
        "Avg Par_NV": 130, "Avg Par_V": 125,
        "EV Std_NV": 120, "EV Std_V": 115,
        # Wrong Bid Analysis columns
        "criterion": 175, "failure_count": 145,
        "check_count": 135, "affected_auctions": 170, "fail_rate": 115,
        "wrong_bid_rate": 150, "wrong_bid_rate_%": 165,
        "Wrong Bids": 130, "Wrong Bid Rate": 155,
    }
    # Ensure headers never get truncated: set minWidth based on header text length.
    # AgGrid's auto-size can shrink columns based on cell contents; minWidth prevents
    # it from shrinking below what's needed for the column name.
    CHAR_WIDTH = 8  # Approximate pixels per character for header text
    ICON_PAD = 55   # Filter icon + sort icon + padding
    for col_name, target_width in tight_cols.items():
        if col_name in df.columns:
            # Extra padding accounts for sort icon + filter/menu affordances.
            header_width = len(col_name) * CHAR_WIDTH + ICON_PAD
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
    # Reuse CHAR_WIDTH and ICON_PAD from above for consistency

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
            width = max(MIN_WIDTH, min(MAX_WIDTH, max_chars * CHAR_WIDTH + ICON_PAD))
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
    
    # Add tooltips for columns with potentially long values
    # These columns show full content on hover to handle truncation
    tooltip_columns = [
        "Exprs", "Expr", "Failed Exprs", "Failed_Exprs",
        "Criteria", "Failed_Criteria", "Failed Criteria",
        "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
        "Invalid_Criteria_S1", "Invalid_Criteria_S2", "Invalid_Criteria_S3", "Invalid_Criteria_S4",
        "Wrong_Bid_S1", "Wrong_Bid_S2", "Wrong_Bid_S3", "Wrong_Bid_S4",
        "Hand_N", "Hand_E", "Hand_S", "Hand_W",
        "ParContracts", "EV_ParContracts", "Contract",
        "Description", "Notes", "Comment",
    ]
    for col in tooltip_columns:
        if col in df.columns:
            # Check for a _full version first
            full_col = f"{col}_full"
            tooltip_field = full_col if full_col in df.columns else col
            # Only configure if not already configured above
            if col not in ("Agg_Expr", "Categories"):
                gb.configure_column(col, tooltipField=tooltip_field)
    
    # Auction column can be wide - constrain it
    if "Auction" in df.columns:
        gb.configure_column("Auction", width=250, minWidth=250, maxWidth=350)
    
    gb.configure_grid_options(
        rowHeight=25,
        headerHeight=25,
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

    # Normalize update_on: None or empty list means no updates (read-only grid)
    effective_update_on = update_on if update_on else []
    is_interactive = bool(effective_update_on)
    
    # Row pass/fail styling: add getRowClass and hide the marker column
    if row_pass_fail_col and row_pass_fail_col in df.columns:
        # Hide the marker column
        for col_def in grid_options.get("columnDefs", []):
            if col_def.get("field") == row_pass_fail_col:
                col_def["hide"] = True
                break
        # Add getRowClass for conditional styling
        grid_options["getRowClass"] = JsCode(f"""
            function(params) {{
                if (params.data && params.data['{row_pass_fail_col}'] === true) {{
                    return 'row-pass';
                }} else if (params.data && params.data['{row_pass_fail_col}'] === false) {{
                    return 'row-fail';
                }}
                return '';
            }}
        """)
    
    # Row bucket styling: add getRowClass and hide the marker column
    if row_bucket_col and row_bucket_col in df.columns:
        # Hide the marker column
        for col_def in grid_options.get("columnDefs", []):
            if col_def.get("field") == row_bucket_col:
                col_def["hide"] = True
                break
        # Add getRowClass for conditional styling
        grid_options["getRowClass"] = JsCode(f"""
            function(params) {{
                if (params.data && params.data['{row_bucket_col}'] === 'Valid') {{
                    return 'row-suggest-valid';
                }}
                if (params.data && params.data['{row_bucket_col}'] === 'Rejected') {{
                    return 'row-suggest-rejected';
                }}
                if (params.data && params.data['{row_bucket_col}'] === 'Invalid') {{
                    return 'row-suggest-invalid';
                }}
                return null;
            }}
        """)
    
    # Build custom CSS
    if row_bucket_col and row_bucket_col in df.columns:
        # Bucket-based row styling (same colors as valid/rejected/invalid bid grids)
        custom_css = {
            ".ag-row": {"cursor": "pointer" if is_interactive else "default"},
            ".row-suggest-valid": {"background-color": "#d4edda !important"},
            ".row-suggest-rejected": {"background-color": "#fff3cd !important"},
            ".row-suggest-invalid": {"background-color": "#f8d7da !important"},
            ".row-suggest-valid.ag-row-hover": {"background-color": "#b8dfc4 !important", "border-left": "3px solid #28a745 !important"},
            ".row-suggest-rejected.ag-row-hover": {"background-color": "#ffe69c !important", "border-left": "3px solid #ffc107 !important"},
            ".row-suggest-invalid.ag-row-hover": {"background-color": "#f1b0b7 !important", "border-left": "3px solid #dc3545 !important"},
            ".row-suggest-valid.ag-row-selected": {"background-color": "#a3d4af !important"},
            ".row-suggest-rejected.ag-row-selected": {"background-color": "#ffe082 !important"},
            ".row-suggest-invalid.ag-row-selected": {"background-color": "#eb959f !important"},
        }
    elif row_pass_fail_col and row_pass_fail_col in df.columns:
        # Pass/fail row styling (same colors as valid/invalid bid grids)
        custom_css = {
            ".row-pass": {"background-color": "#d4edda"},  # Green
            ".row-fail": {"background-color": "#f8d7da"},  # Red
            ".row-pass.ag-row-hover": {"background-color": "#b8dfc4 !important", "border-left": "3px solid #28a745 !important"},
            ".row-fail.ag-row-hover": {"background-color": "#f1b0b7 !important", "border-left": "3px solid #dc3545 !important"},
            ".row-pass.ag-row-selected": {"background-color": "#a3d4af !important"},
            ".row-fail.ag-row-selected": {"background-color": "#eb959f !important"},
            ".ag-row": {"cursor": "pointer" if is_interactive else "default"},
        }
    elif is_interactive:
        custom_css = {
            ".ag-row": {"cursor": "pointer", "transition": "background-color 0.1s"},
            ".ag-row-hover": {
                "background-color": "#E8F4FF !important",
                "border-left": "3px solid #007BFF !important"
            },
            ".ag-row-selected": {"background-color": "#D1E9FF !important"},
        }
    else:
        custom_css = {
            ".ag-row": {"cursor": "default"},
            ".ag-row-hover": {"background-color": "#F8F9FA !important"},
        }
    
    response = AgGrid(
        pandas_df,
        gridOptions=grid_options,
        height=height,
        theme="balham",
        key=key,
        columns_auto_size_mode=auto_size_mode,
        # Critical UX: clicking a row should only highlight it locally and NOT
        # emit selection/model updates back to Streamlit (which would cause a rerun)
        # unless specifically requested via update_on.
        update_on=effective_update_on,
        data_return_mode=DataReturnMode.AS_INPUT,
        custom_css=custom_css,
        allow_unsafe_jscode=True if (row_pass_fail_col or row_bucket_col) else False,
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
        prewarm = status.get("prewarm_progress") or {}
        try:
            i = int(prewarm.get("i") or 0)
            n = int(prewarm.get("n") or 0)
            name = str(prewarm.get("name") or "").strip()
            if i > 0 and n > 0:
                st.info(f"🔥 Warm-up: **{i}/{n}** {name}")
        except Exception:
            pass
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
    with st.sidebar.expander("Settings", expanded=False):
        n_auction_samples = st.number_input("Auction Samples", value=2, min_value=1, max_value=10)
        n_deal_samples = st.number_input("Deal Samples per Auction", value=100, min_value=1, max_value=10_000_000)
        allow_initial_passes = st.checkbox(
            "Match all 4 dealer positions",
            value=False,
            help=MATCH_ALL_DEALERS_HELP,
            key="allow_initial_passes_deals",
        )

        # Distribution filter for deals
        with st.expander("Distribution Filter", expanded=False):
            deal_dist_direction = st.selectbox("Filter Hand", ["N", "E", "S", "W"], index=2,
                help="Which hand's distribution to filter")
            
            deal_dist_pattern = st.text_input("Ordered Distribution (S-H-D-C)", value="",
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
                    st.caption(f"→ {', '.join(dist_display)}")
                else:
                    st.warning("Invalid distribution pattern")
            
            deal_sorted_shape = st.text_input("Sorted Shape (any suit order)", value="",
                placeholder="e.g., 5431, 4432, 5332",
                help="Filter deals by shape regardless of suit.",
                key="deal_sorted_shape")
            
            if deal_sorted_shape:
                parsed_shape = parse_sorted_shape(deal_sorted_shape)
                if parsed_shape:
                    st.caption(f"→ shape {''.join(map(str, parsed_shape))} (any suits)")
                else:
                    st.warning("Invalid sorted shape (must be 4 digits summing to 13)")
            
            with st.expander("Distribution notation help"):
                st.markdown(format_distribution_help())

        # Random seed at bottom of sidebar
        seed = int(st.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_deals"))

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

    show_categories = st.sidebar.checkbox(
        "Show bid category flags (Phase 4)",
        value=False,
        help="Includes ~100 boolean is_* columns from bbo_bt_categories.parquet (can be wide).",
        key="bt_explorer_show_categories",
    )
    
    with st.sidebar.expander("Distribution Filter", expanded=False):
        dist_seat = st.selectbox("Filter Seat", [1, 2, 3, 4], index=0,
            help="Which seat's distribution to filter (S1=opener in most auctions)",
            key="bt_explorer_dist_seat")
        
        dist_pattern = st.text_input("Ordered Distribution (S-H-D-C)", value="",
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
                st.caption(f"→ {', '.join(dist_display)}")
            else:
                st.warning("Invalid distribution pattern")
        
        sorted_shape = st.text_input("Sorted Shape (any suit order)", value="",
            placeholder="e.g., 5431, 4432, 5332",
            help="Filter by shape regardless of suit.",
            key="bt_explorer_sorted_shape")
        
        if sorted_shape:
            parsed_shape = parse_sorted_shape(sorted_shape)
            if parsed_shape:
                st.caption(f"→ shape {''.join(map(str, parsed_shape))} (any suits)")
            else:
                st.warning("Invalid sorted shape (must be 4 digits summing to 13)")
        
        with st.expander("Distribution notation help"):
            st.markdown(format_distribution_help())
    
    # Random seed at bottom of sidebar
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
    
    with st.sidebar.expander("Settings", expanded=False):
        max_groups = st.number_input("Max Auction Groups", value=10, min_value=1, max_value=100)
        deals_per_group = st.number_input("Deals per Group", value=100, min_value=1, max_value=1000)
        
        show_blocking_criteria = st.checkbox(
            "🔍 Show Blocking Criteria",
            value=False,
            help="For each deal, evaluate which criteria from the BT row are blocking. Adds columns showing failed/untracked criteria per deal."
        )
        
        # Random seed at bottom of sidebar
        seed = int(st.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_group"))

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

    use_model_rules = st.sidebar.checkbox(
        "Use Model Rules (greedy) instead of Auction Rules",
        value=False,
        help=(
            "If enabled, Rules-like models will be evaluated via a step-by-step greedy path (similar to 'Model's Predicted Path') "
            "instead of selecting a best matching completed BT auction. This is more faithful but slower."
        ),
    )

    include_model_auction = st.sidebar.checkbox(
        "Also compute Model auction (greedy) for comparison",
        value=False,
        help=(
            "If enabled, the Arena will add extra columns (Auction_Model / Model_Contract / DD_Score_Model) "
            "to the Sample Deal tables so you can compare Actual vs Rules vs Model in one view. "
            "This does not change which two models are scored in the headline comparison."
        ),
    )

    debug_bt_index_raw = st.sidebar.text_input(
        "Pin BT index (optional).",
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
        st.sidebar.caption(f"Pin BT index is set to {debug_bt_index}. Clear it to enable auto-debug-from-Actual.")

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
                "use_model_rules": bool(use_model_rules),
                "include_model_auction": bool(include_model_auction),
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
                            "Auction_Model",
                            "Auction_Match",
                            "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                            f"DD_Score_{model_a}" if f"DD_Score_{model_a}" in sample_df.columns else None,
                            f"DD_Score_{model_b}" if f"DD_Score_{model_b}" in sample_df.columns else None,
                            "DD_Score_Model" if "DD_Score_Model" in sample_df.columns else None,
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
                            update_on=["selectionChanged"],
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
                            update_on=["selectionChanged"],
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
                    "Auction_Model",
                    "Model_Contract",
                    "DD_Score_Model",
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
                    update_on=["selectionChanged"],
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
                                update_on=["selectionChanged"],
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
                                update_on=["selectionChanged"],
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
# AI Model Batch Arena – Batch compare AI Model vs Actual auctions
# ---------------------------------------------------------------------------

def render_ai_model_batch_arena():
    """Batch-process deals: compare AI Model auctions against Actual auctions with DD scores, IMPs."""
    st.header("🤖 AI Model Batch Arena")
    st.markdown("Batch compare AI Model auctions against Actual auctions with DD scores, IMP differentials, and running totals.")

    # ---- Column list for deal rows (same broad set as Auction Builder) ----
    _BATCH_DEAL_COLS: list[str] = [
        "index", "Dealer", "Vul",
        "Hand_N", "Hand_E", "Hand_S", "Hand_W",
        "HCP_N", "HCP_E", "HCP_S", "HCP_W",
        "Total_Points_N", "Total_Points_E", "Total_Points_S", "Total_Points_W",
        "ParContracts", "ParScore", "bid",
    ]
    # DD score columns: DD_Score_{level}{strain}_{direction} (140 cols)
    for _lvl in range(1, 8):
        for _s in ("C", "D", "H", "S", "N"):
            for _d in ("N", "E", "S", "W"):
                _BATCH_DEAL_COLS.append(f"DD_Score_{_lvl}{_s}_{_d}")
    # Raw DD trick columns: DD_{direction}_{strain} (20 cols)
    for _d in ("N", "E", "S", "W"):
        for _s in ("C", "D", "H", "S", "N"):
            _BATCH_DEAL_COLS.append(f"DD_{_d}_{_s}")
    # EV columns: EV_{pair}_{declarer}_{strain}_{level} (140 cols)
    for _pair, _decls in [("NS", ("N", "S")), ("EW", ("E", "W"))]:
        for _decl in _decls:
            for _s in ("C", "D", "H", "S", "N"):
                for _lvl in range(1, 8):
                    _BATCH_DEAL_COLS.append(f"EV_{_pair}_{_decl}_{_s}_{_lvl}")

    # ---- Input controls ----
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        start_index = st.number_input("Start Deal Index", min_value=1, value=1, step=1, key="_batch_start")
    with col_b:
        deal_count = st.number_input("Number of Deals", min_value=1, max_value=500, value=25, step=25, key="_batch_count")
    with col_c:
        seed = st.number_input("Random Seed", min_value=0, value=0, key="_batch_seed")

    # ---- Run / Cancel buttons ----
    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        run_clicked = st.button("▶ Run Batch", key="_batch_run")
    with btn_col2:
        cancel_clicked = st.button("⏹ Cancel", key="_batch_cancel")

    if cancel_clicked:
        st.session_state["_arena_batch_cancel"] = True

    # ---- Helper: build contract string from auction ----
    def _contract_str(auction: str | None, dealer: str) -> str:
        if not auction:
            return ""
        toks = [t.strip().upper() for t in auction.split("-") if t.strip()]
        if all(t == "P" for t in toks):
            return "Pass"
        c = get_ai_contract(auction, dealer)
        return str(c) if c else ""

    # ---- Helper: compute DD tricks for an auction ----
    def _dd_tricks(auction: str | None, dealer: str, deal_row: dict) -> int | None:
        if not auction:
            return None
        toks = [t.strip().upper() for t in auction.split("-") if t.strip()]
        if all(t == "P" for t in toks):
            return None
        return get_dd_tricks_for_auction(auction, dealer, deal_row)

    # ---- Helper: compute result (over/under tricks) ----
    def _result(auction: str | None, dealer: str, deal_row: dict) -> int | None:
        tricks = _dd_tricks(auction, dealer, deal_row)
        if tricks is None:
            return None
        c = parse_contract_from_auction(auction or "")
        if not c:
            return None
        level_i, _, _ = c
        return int(tricks) - (int(level_i) + 6)

    # ---- Helper: NS-relative DD score ----
    def _dd_score_ns(auction: str | None, dealer: str, deal_row: dict) -> int | None:
        if not auction:
            return None
        toks = [t.strip().upper() for t in auction.split("-") if t.strip()]
        if all(t == "P" for t in toks):
            return 0
        raw = get_dd_score_for_auction(auction, dealer, deal_row)
        if raw is None:
            return None
        decl = get_declarer_for_auction(auction, dealer)
        if decl and str(decl).upper() in ("E", "W"):
            return -int(raw)
        return int(raw)

    # ---- Helper: signed IMP diff (positive = AI better for NS) ----
    def _imp_diff(ai_score_ns: int | None, actual_score_ns: int | None) -> int | None:
        if ai_score_ns is None or actual_score_ns is None:
            return None
        diff = int(ai_score_ns) - int(actual_score_ns)
        sign = 1 if diff >= 0 else -1
        return sign * calculate_imp(abs(diff))

    # ---- Helper: NS-relative EV (precomputed, ignores X/XX) ----
    def _ev_ns(auction: str | None, dealer: str, deal_row: dict) -> float | None:
        if not auction:
            return None
        toks = [t.strip().upper() for t in auction.split("-") if t.strip()]
        if all(t == "P" for t in toks):
            return 0.0
        raw = get_ev_for_auction_pre(auction, dealer, deal_row)
        if raw is None:
            return None
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None
        decl = get_declarer_for_auction(auction, dealer)
        if decl and str(decl).upper() in ("E", "W"):
            return round(-val, 1)
        return round(val, 1)

    # ---- Batch processing loop ----
    if run_clicked:
        st.session_state["_arena_batch_cancel"] = False
        st.session_state["_arena_batch_results"] = []

        progress_bar = st.progress(0, text="Starting batch...")
        status_text = st.empty()

        # Accumulate per-deal debug entries for data/batch_arena_debug.json
        _batch_debug_entries: list[dict] = []

        indices = list(range(int(start_index), int(start_index) + int(deal_count)))

        # Pre-fetch all deal rows in one batch call
        status_text.text("")
        try:
            deals_resp = api_post(
                "/deals-by-index",
                {"indices": indices, "max_rows": len(indices), "columns": _BATCH_DEAL_COLS},
                timeout=60,
            )
            deal_rows_raw = deals_resp.get("rows") or []
        except Exception as e:
            st.error(f"Failed to fetch deal rows: {e}")
            return

        # Build lookup: deal index -> deal row dict (with _row_idx)
        deal_row_map: dict[int, dict] = {}
        for dr in deal_rows_raw:
            try:
                deal_row_map[int(dr["index"])] = dr
            except Exception:
                continue

        if not deal_row_map:
            st.warning("No deal rows found for the requested index range.")
            return

        results: list[dict] = []
        imp_running = 0
        par_imp_running = 0
        t_batch_start = time.perf_counter()

        for i, deal_idx in enumerate(indices):
            # ---- Cancel check (Streamlit may interrupt via progress bar update) ----
            if st.session_state.get("_arena_batch_cancel"):
                status_text.text(f"Cancelled after {i} deal(s).")
                break

            deal_row = deal_row_map.get(deal_idx)
            if deal_row is None:
                results.append({"Deal": deal_idx, "Error": "not_found"})
                st.session_state["_arena_batch_results"] = list(results)
                progress_bar.progress((i + 1) / len(indices), text=f"Deal {i+1}/{len(indices)} (index {deal_idx}) — not found — IMP: {imp_running:+d}")
                continue

            dealer = str(deal_row.get("Dealer", "N")).upper()
            vul = deal_row.get("Vul", "")
            actual_auction = str(deal_row.get("bid", "") or "").strip()
            row_idx = deal_row.get("_row_idx")

            if row_idx is None:
                results.append({"Deal": deal_idx, "Error": "no_row_idx"})
                st.session_state["_arena_batch_results"] = list(results)
                progress_bar.progress((i + 1) / len(indices), text=f"Deal {i+1}/{len(indices)} — no _row_idx — IMP: {imp_running:+d}")
                continue

            # ---- Start AI Model async job ----
            progress_bar.progress(i / len(indices), text=f"Deal {i+1}/{len(indices)} (index {deal_idx}) — IMP: {imp_running:+d} — running AI Model...")
            try:
                start_resp = api_post(
                    "/ai-model-advanced-path/start",
                    {"deal_row_idx": int(row_idx), "seed": int(seed)},
                    timeout=15,
                )
                job_id = str(start_resp.get("job_id") or "").strip()
            except Exception as e:
                results.append({"Deal": deal_idx, "Error": f"start_failed: {e}"})
                st.session_state["_arena_batch_results"] = list(results)
                progress_bar.progress((i + 1) / len(indices), text=f"Deal {i+1}/{len(indices)} — start failed — IMP: {imp_running:+d}")
                continue

            if not job_id:
                results.append({"Deal": deal_idx, "Error": "no_job_id"})
                st.session_state["_arena_batch_results"] = list(results)
                continue

            # ---- Poll until done ----
            ai_auction = ""
            ai_steps_detail: list[dict] = []
            poll_ok = False
            for _poll in range(300):  # up to ~150 seconds
                time.sleep(0.5)
                try:
                    job = api_get(f"/ai-model-advanced-path/status/{job_id}", timeout=10)
                except Exception:
                    continue
                status_val = str(job.get("status") or "")
                if status_val == "completed":
                    res = job.get("result") or {}
                    ai_auction = str(res.get("auction") or "").strip()
                    ai_steps_detail = res.get("steps_detail") or []
                    poll_ok = True
                    break
                elif status_val == "failed":
                    results.append({"Deal": deal_idx, "Error": f"job_failed: {job.get('error')}"})
                    break
            else:
                results.append({"Deal": deal_idx, "Error": "timeout"})

            if not poll_ok:
                st.session_state["_arena_batch_results"] = list(results)
                progress_bar.progress((i + 1) / len(indices), text=f"Deal {i+1}/{len(indices)} — failed/timeout — IMP: {imp_running:+d}")
                continue

            # ---- Compute scores ----
            actual_contract = _contract_str(actual_auction, dealer)
            ai_contract = _contract_str(ai_auction, dealer)

            actual_dd_tricks = _dd_tricks(actual_auction, dealer, deal_row)
            ai_dd_tricks = _dd_tricks(ai_auction, dealer, deal_row)

            actual_result = _result(actual_auction, dealer, deal_row)
            ai_result = _result(ai_auction, dealer, deal_row)

            actual_score_ns = _dd_score_ns(actual_auction, dealer, deal_row)
            ai_score_ns = _dd_score_ns(ai_auction, dealer, deal_row)

            actual_ev = _ev_ns(actual_auction, dealer, deal_row)
            ai_ev = _ev_ns(ai_auction, dealer, deal_row)

            imp = _imp_diff(ai_score_ns, actual_score_ns)
            if imp is not None:
                imp_running += imp

            par_score = deal_row.get("ParScore")
            par_contracts_raw = deal_row.get("ParContracts")
            par_contracts = _format_par_contracts(par_contracts_raw) or ""

            # Par IMP: difference in IMPs between par score and AI DD score
            par_imp: int | None = None
            try:
                if par_score is not None and ai_score_ns is not None:
                    par_diff = int(ai_score_ns) - int(par_score)
                    par_sign = 1 if par_diff >= 0 else -1
                    par_imp = par_sign * calculate_imp(abs(par_diff))
            except (ValueError, TypeError):
                par_imp = None
            if par_imp is not None:
                par_imp_running += par_imp

            results.append({
                "Deal": deal_idx,
                "Dealer": dealer,
                "Vul": vul,
                "Actual_Auction": actual_auction,
                "AI_Auction": ai_auction,
                "Actual_Contract": actual_contract,
                "AI_Contract": ai_contract,
                "Actual_Tricks": actual_dd_tricks,
                "AI_Tricks": ai_dd_tricks,
                "Actual_Result": actual_result,
                "AI_Result": ai_result,
                "DD_Score_Actual": actual_score_ns,
                "DD_Score_AI": ai_score_ns,
                "EV_Actual": actual_ev,
                "EV_AI": ai_ev,
                "Par": par_score,
                "ParContracts": par_contracts,
                "IMP_Diff": imp,
                "IMP_Running": imp_running,
                "Par_IMP": par_imp,
                "Par_IMP_Total": par_imp_running,
            })
            st.session_state["_arena_batch_results"] = list(results)

            # ---- Accumulate debug entry for this deal ----
            try:
                _dbg_deal_info: dict[str, Any] = {}
                for _dk in [
                    "_row_idx", "index", "Dealer", "Vul",
                    "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                    "ParContracts", "ParScore", "bid",
                ]:
                    _dv = deal_row.get(_dk)
                    if _dv is not None:
                        _dbg_deal_info[_dk] = _dv
                # Add HCP/TP per direction (computed from hands if not in row)
                for _dir in ("N", "E", "S", "W"):
                    for _stat in ("HCP", "Total_Points"):
                        _sk = f"{_stat}_{_dir}"
                        _sv = deal_row.get(_sk)
                        if _sv is not None:
                            _dbg_deal_info[_sk] = _sv

                # ---- Compute divergence point ----
                # Find the first step where the AI bid differs from the actual
                # auction.  This is the most useful diagnostic: which bid went
                # wrong, what were the alternatives, and why?
                _actual_toks = [
                    t.strip().upper()
                    for t in (actual_auction or "").split("-")
                    if t.strip()
                ]
                _ai_toks = [
                    t.strip().upper()
                    for t in (ai_auction or "").split("-")
                    if t.strip()
                ]
                _divergence: dict[str, Any] | None = None
                for _si in range(max(len(_actual_toks), len(_ai_toks))):
                    _a_bid = _actual_toks[_si] if _si < len(_actual_toks) else None
                    _ai_bid = _ai_toks[_si] if _si < len(_ai_toks) else None
                    if _a_bid != _ai_bid:
                        # Find the matching step in ai_steps_detail
                        _step_data: dict[str, Any] | None = None
                        for _sd in ai_steps_detail:
                            if isinstance(_sd, dict) and _sd.get("step") == _si + 1:
                                _step_data = _sd
                                break
                        _divergence = {
                            "step": _si + 1,
                            "actual_bid": _a_bid,
                            "ai_bid": _ai_bid,
                            "auction_so_far": "-".join(_ai_toks[:_si]) if _si > 0 else "(opening)",
                        }
                        # Include compact scoring summary at divergence point
                        if _step_data:
                            _divergence["seat"] = _step_data.get("seat")
                            _divergence["seat_bt"] = _step_data.get("seat_bt")
                            _divergence["scored_n"] = _step_data.get("scored_n")
                            _divergence["bid_scores"] = _step_data.get("bid_scores")
                        break

                _batch_debug_entries.append({
                    "deal_index": deal_idx,
                    "deal": _dbg_deal_info,
                    "dealer": dealer,
                    "vul": vul,
                    "actual_auction": actual_auction,
                    "ai_auction": ai_auction,
                    "actual_contract": actual_contract,
                    "ai_contract": ai_contract,
                    "dd_score_actual_ns": actual_score_ns,
                    "dd_score_ai_ns": ai_score_ns,
                    "ev_actual_ns": actual_ev,
                    "ev_ai_ns": ai_ev,
                    "par": par_score,
                    "par_contracts": par_contracts,
                    "imp_diff": imp,
                    "par_imp": par_imp,
                    "divergence": _divergence,
                    "ai_model_steps": ai_steps_detail,
                })
            except Exception:
                pass  # Debug accumulation is best-effort

            # ---- Live update ----
            elapsed_s = time.perf_counter() - t_batch_start
            avg_s = elapsed_s / (i + 1)
            eta_s = avg_s * (len(indices) - i - 1)
            progress_bar.progress(
                (i + 1) / len(indices),
                text=f"Deal {i+1}/{len(indices)} (index {deal_idx}) — IMP running: {imp_running:+d} — ETA {eta_s:.0f}s",
            )
            # Live table intentionally omitted — AgGrid renders after st.rerun().

        else:
            # Loop completed without cancel
            elapsed_total = time.perf_counter() - t_batch_start
            status_text.text(f"Batch complete: {len(results)} deals in {elapsed_total:.1f}s. IMP running total: {imp_running:+d}")

        # Store results and rerun — the interactive table is rendered outside
        # this block (always, on every rerun) so there is never a duplicate.
        st.session_state["_arena_batch_results"] = list(results)
        st.session_state["_arena_deal_row_map"] = deal_row_map
        st.session_state["_batch_selected_result"] = None  # clear any stale selection
        st.rerun()

        # ── Write batch debug log for agent/diagnostic access ──
        try:
            import json as _ba_json
            from datetime import datetime as _ba_dt, timezone as _ba_tz
            _ba_debug_path = pathlib.Path("data/batch_arena_debug.json")
            _ba_debug_path.parent.mkdir(parents=True, exist_ok=True)

            # ---- Build worst-deals summary (top 10 by most negative IMP_Diff) ----
            # This goes at the top of the file so an agent can immediately see the
            # worst outcomes without scanning the full deals list.
            _worst_n = 10
            def _imp_sort_key(d: dict) -> float:
                v = d.get("imp_diff")
                return float(v) if v is not None else 0.0
            _sorted_entries = sorted(_batch_debug_entries, key=_imp_sort_key)
            _worst_deals_summary: list[dict[str, Any]] = []
            for _we in _sorted_entries[:_worst_n]:
                _imp = _we.get("imp_diff")
                if _imp is None or _imp >= 0:
                    continue  # only include negative IMP deals
                _div = _we.get("divergence") or {}
                _ws: dict[str, Any] = {
                    "deal_index": _we.get("deal_index"),
                    "imp_diff": _imp,
                    "par_imp": _we.get("par_imp"),
                    "actual_auction": _we.get("actual_auction"),
                    "ai_auction": _we.get("ai_auction"),
                    "actual_contract": _we.get("actual_contract"),
                    "ai_contract": _we.get("ai_contract"),
                    "dd_score_actual_ns": _we.get("dd_score_actual_ns"),
                    "dd_score_ai_ns": _we.get("dd_score_ai_ns"),
                    "par": _we.get("par"),
                    "par_contracts": _we.get("par_contracts"),
                }
                # Compact divergence synopsis
                if _div:
                    _ws["divergence_step"] = _div.get("step")
                    _ws["divergence_auction_so_far"] = _div.get("auction_so_far")
                    _ws["actual_bid_at_divergence"] = _div.get("actual_bid")
                    _ws["ai_bid_at_divergence"] = _div.get("ai_bid")
                    # Include what the AI chose vs top alternative scores
                    _bs = _div.get("bid_scores") or []
                    if _bs:
                        _ws["bid_scores_at_divergence"] = [
                            {
                                "bid": b.get("bid"),
                                "score": b.get("score"),
                                "guard_penalty": b.get("guard_penalty"),
                                "pass_source": b.get("pass_source"),
                            }
                            for b in _bs[:5]  # top 5 candidates
                        ]
                # Include hands for quick reference
                _deal = _we.get("deal") or {}
                _ws["hands"] = {
                    d: _deal.get(f"Hand_{d}") for d in ("N", "E", "S", "W")
                }
                _ws["hcp"] = {
                    d: _deal.get(f"HCP_{d}") for d in ("N", "E", "S", "W")
                }
                _ws["dealer"] = _we.get("dealer")
                _ws["vul"] = _we.get("vul")
                _worst_deals_summary.append(_ws)

            _ba_payload = {
                "timestamp": _ba_dt.now(_ba_tz.utc).isoformat(),
                "start_index": int(start_index),
                "deal_count": int(deal_count),
                "seed": int(seed),
                "deals_processed": len(results),
                "imp_running_total": imp_running,
                "par_imp_running_total": par_imp_running,
                "worst_deals": _worst_deals_summary,
                "deals": _batch_debug_entries,
            }
            _ba_debug_path.write_text(
                _ba_json.dumps(_ba_payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass  # Debug dump is best-effort; never break the UI

    # ---- Results table (always rendered; single source of truth) ----
    # This block runs on every Streamlit execution that is not mid-batch.
    # It shows the interactive AgGrid table, the export button, and — when a
    # row has been clicked — the deal diagram followed by the selected-row detail.
    all_results: list[dict] = st.session_state.get("_arena_batch_results") or []
    if all_results:
        _render_batch_results_df(all_results, st, interactive=True, aggrid_key="_batch_results_main")

        # Export button
        results_df = pd.DataFrame(all_results)
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "📥 Export Results (CSV)",
            data=csv_bytes,
            file_name=f"ai_model_batch_{int(start_index)}_{int(deal_count)}_{ts}.csv",
            mime="text/csv",
            key="_batch_export",
        )

        # ---- Row-click: deal diagram then single-row detail ----
        _sel_result: dict | None = st.session_state.get("_batch_selected_result")
        _deal_row_map: dict[int, dict] = st.session_state.get("_arena_deal_row_map") or {}
        if _sel_result:
            _sel_deal_idx = _sel_result.get("Deal")
            st.divider()

            # 1. Deal diagram
            _deal_row = _deal_row_map.get(int(_sel_deal_idx)) if _sel_deal_idx is not None else None
            if _deal_row:
                _diag_deal = dict(_deal_row)
                _diag_deal.setdefault("ParScore", _sel_result.get("Par"))
                _diag_deal.setdefault("Score", _sel_result.get("DD_Score_AI"))
                render_deal_diagram(
                    _diag_deal,
                    title=f"📋 Deal {_sel_deal_idx}",
                    show_header=True,
                    width_ratio=(3, 7),
                )
            else:
                st.info("Deal diagram unavailable — re-run the batch to load hand data.")

            # 2. Single-row detail — same colour as the clicked row
            _display_result = {k: v for k, v in _sel_result.items() if not k.startswith("_")}
            _imp_val = _display_result.get("IMP_Diff")
            if _imp_val is not None and not (isinstance(_imp_val, float) and pd.isna(_imp_val)):
                _imp_int = int(_imp_val)
                _imp_color = "Valid" if _imp_int > 0 else ("Invalid" if _imp_int < 0 else "Neutral")
            else:
                _imp_color = "Neutral"  # no class → white
            _display_result["_IMP_Color"] = _imp_color
            render_aggrid(
                [_display_result],
                key=f"_batch_selected_detail_{_sel_deal_idx}",
                table_name=None,
                show_sql_expander=False,
                row_bucket_col="_IMP_Color",
                hide_cols=["_IMP_Color"],
                height=calc_grid_height(1),
            )

            # Scroll to bottom after all content is rendered.
            st_components.html(
                "<script>window.parent.document.querySelector('.main').scrollTo("
                "{top: window.parent.document.querySelector('.main').scrollHeight, behavior: 'smooth'});"
                "</script>",
                height=0,
            )


def _render_batch_results_df(
    results: list[dict],
    container,
    *,
    interactive: bool = False,
    aggrid_key: str = "_batch_results_grid",
    height: int | None = None,
) -> None:
    """Render the batch results DataFrame with color-coded IMP_Diff rows.

    Args:
        results: List of result dicts produced by the batch loop.
        container: Streamlit container (st, st.empty(), etc.) to render into.
        interactive: When True uses AgGrid with selectionChanged so clicking a
            row triggers a Streamlit rerun and exposes the deal diagram.
            Set False during live in-loop updates to avoid key conflicts.
        aggrid_key: Unique AgGrid key (must differ between final and previous
            result renders to avoid widget-state collision).
    """
    if not results:
        return

    if not interactive:
        # Plain styled pandas table — used during the live batch-update loop.
        df = pd.DataFrame(results)
        if "IMP_Diff" in df.columns:
            def _color_row(row):
                imp = row.get("IMP_Diff")
                if imp is None or pd.isna(imp):
                    return [""] * len(row)
                if imp < 0:
                    return ["background-color: #ffcccc"] * len(row)
                elif imp > 0:
                    return ["background-color: #ccffcc"] * len(row)
                return [""] * len(row)
            styled = df.style.apply(_color_row, axis=1).format(
                {col: "{:.1f}" for col in ("EV_Actual", "EV_AI") if col in df.columns}
            )
            container.dataframe(styled, width="stretch", hide_index=True)
        else:
            container.dataframe(df, width="stretch", hide_index=True)
        return

    # ---- Interactive AgGrid with row-click support ----
    # Map IMP_Diff to a bucket column for row colouring:
    #   Valid (green) = AI did better, Invalid (red) = AI did worse, Rejected (yellow) = tied/unknown
    enriched: list[dict] = []
    for r in results:
        row = dict(r)
        imp = row.get("IMP_Diff")
        if imp is not None and not (isinstance(imp, float) and pd.isna(imp)):
            imp_int = int(imp)
            if imp_int > 0:
                row["_IMP_Color"] = "Valid"
            elif imp_int < 0:
                row["_IMP_Color"] = "Invalid"
            else:
                row["_IMP_Color"] = "Neutral"  # no class → white
        else:
            row["_IMP_Color"] = "Neutral"  # no class → white
        enriched.append(row)

    selected_records = render_aggrid(
        enriched,
        key=aggrid_key,
        table_name=None,
        show_sql_expander=False,
        row_bucket_col="_IMP_Color",
        hide_cols=["_IMP_Color"],
        update_on=["selectionChanged"],
        height=height if height is not None else calc_grid_height(len(enriched), max_height=600),
    )

    if selected_records:
        st.session_state["_batch_selected_result"] = selected_records[0]


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
                    # Normalize to canonical dash-separated form for literal auctions.
                    # (Regex patterns are kept as uppercase strings.)
                    new_partial_normalized = ""
                    if new_partial:
                        try:
                            if is_regex_pattern(new_partial):
                                new_partial_normalized = new_partial.strip().upper()
                            else:
                                new_partial_normalized = normalize_auction_input(new_partial).strip().upper()
                                new_partial_normalized = re.sub(r"(?i)^(p-)+", "", new_partial_normalized)
                        except Exception:
                            new_partial_normalized = new_partial.strip().upper()
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
    
    max_deals = st.sidebar.number_input(
        "Max Deals",
        value=500,
        min_value=1,
        max_value=10000,
        help="Maximum deals to sample for analysis"
    )
    
    st.sidebar.subheader("Filters")
    
    vul_filter = st.sidebar.selectbox(
        "Vulnerability",
        ["all", "None", "Both", "NS", "EW"],
        index=0,
        help="Filter deals by vulnerability"
    )
    
    st.sidebar.subheader("Output Options")
    
    include_hands = st.sidebar.checkbox("Include Hands", value=True, help="Include Hand_N/E/S/W columns")
    include_scores = st.sidebar.checkbox("Include DD Scores", value=True, help="Include DD_Score columns in Deal Data table")
    
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_rank_ev"))
    
    # =========================================================================
    # Call API (single endpoint provides both bid rankings and DD analysis)
    # =========================================================================
    # Cache version: increment when API response format changes
    CACHE_VERSION = 27  # v27: bust cache for contract EV blob EV_Score population
    
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

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_bid_details(p: Dict[str, Any]) -> Dict[str, Any]:
        return api_post("/bid-details", p)

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_explain_bid(p: Dict[str, Any]) -> Dict[str, Any]:
        return api_post("/explain-bid", p)
    
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
            # Note: avg_ev_precomputed/avg_par_precomputed exist but are not displayed
            # because they're not split by vulnerability (aggregate across NV/V is not useful)
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
            update_on=["selectionChanged"],
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
            # -----------------------------------------------------------------
            # MVP-2/3: Stable selected-bid details + EEO + templated explanations
            # -----------------------------------------------------------------
            # Make the expander self-describing: this is about the SELECTED BID row above.
            # (These are *par-contract* outcomes across matched deals, not the actual final contract reached in play.)
            exp_bid = str(selected_bid_name or "").strip().upper()
            exp_auc = None
            try:
                if selected_bid_rows is not None and len(selected_bid_rows) > 0 and selected_bid is not None:
                    exp_auc = selected_bid.get("Full Auction")
            except Exception:
                exp_auc = None
            exp_title = f"🧾 Selected bid: {exp_bid} — par posterior + explanation"
            if exp_auc:
                exp_title = f"🧾 Selected bid: {exp_bid} (auction: {exp_auc}) — par posterior + explanation"

            with st.expander(exp_title, expanded=False):
                try:
                    st.caption(
                        "This section summarizes the *par-contract posterior* across deals that match the criteria for the selected bid. "
                        "It does not mean the auction will reach these contracts in real play."
                    )
                    details_payload = {
                        "auction": auction_input,
                        "bid": selected_bid_name,
                        "max_deals": int(max_deals),
                        "seed": seed,
                        "vul_filter": vul_filter if vul_filter != "all" else None,
                        "topk": 10,
                        "include_phase2a": True,
                        "_cache_version": CACHE_VERSION,
                    }
                    details = fetch_bid_details(details_payload)
                    _st_info_elapsed("Bid details", details)

                    ps = details.get("par_score") or {}
                    pc = details.get("par_contracts") or {}
                    degraded_mode = details.get("degraded_mode")
                    degraded_reasons = details.get("degraded_reasons") or []
                    if degraded_mode:
                        st.warning(f"Degraded mode: `{degraded_mode}` ({'; '.join(str(x) for x in degraded_reasons if x)})")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Matched deals (total)", f"{int(details.get('matched_deals_total') or 0):,}")
                    c2.metric("Matched deals (sampled)", f"{int(details.get('matched_deals_sampled') or 0):,}")
                    c3.metric("Mean ParScore", f"{float(ps.get('mean') or 0.0):.1f}" if ps.get("mean") is not None else "NA")
                    c4.metric("Entropy (nats)", f"{float(pc.get('contract_entropy_full_nats') or 0.0):.3f}" if pc.get("contract_entropy_full_nats") is not None else "NA")

                    topk = pc.get("topk") or []
                    if topk:
                        topk_df = pl.DataFrame(topk)
                        # Keep compact columns
                        keep = [c for c in ["contract", "pair", "declarer", "prob", "avg_par_score"] if c in topk_df.columns]
                        if keep:
                            topk_df = topk_df.select(keep)
                        # Rename for clarity (these are par contracts, not "the contract we bid").
                        try:
                            rename_map = {
                                "contract": "ParContract",
                                "pair": "Pair",
                                "declarer": "Declarer",
                                "prob": "Prob",
                                "avg_par_score": "Avg_ParScore",
                            }
                            cols_present = {c: rename_map[c] for c in topk_df.columns if c in rename_map}
                            if cols_present:
                                topk_df = topk_df.rename(cols_present)
                            if "Prob" in topk_df.columns:
                                topk_df = topk_df.with_columns((pl.col("Prob") * 100).round(1).alias("Prob_%")).drop("Prob")
                        except Exception:
                            pass
                        # Render as AgGrid (consistent UI).
                        render_aggrid(
                            topk_df,
                            key=f"bid_details_topk_{exp_bid}_{CACHE_VERSION}",
                            height=calc_grid_height(len(topk_df), max_height=250),
                            table_name="bid_details_topk",
                            fit_columns_to_view=True,
                            show_sql_expander=False,
                        )
                    else:
                        st.info("No par-contract distribution available for this selection.")

                    # Counterfactual selection
                    all_bids = sorted({str(r.get("bid") or "").strip().upper() for r in bid_rankings if r.get("bid")})
                    why_not = st.selectbox(
                        "Why not… (optional)",
                        options=["(none)"] + all_bids,
                        index=0,
                        key=f"why_not_{CACHE_VERSION}",
                    )
                    explain_payload = {
                        "auction": auction_input,
                        "bid": selected_bid_name,
                        "why_not_bid": None if why_not == "(none)" else why_not,
                        "max_deals": int(max_deals),
                        "seed": seed,
                        "vul_filter": vul_filter if vul_filter != "all" else None,
                        "topk": 10,
                        "include_phase2a": True,
                        "_cache_version": CACHE_VERSION,
                    }
                    expl = fetch_explain_bid(explain_payload)
                    ex = expl.get("explanation") or {}
                    st.write(ex.get("text") or "No explanation returned.")
                    cf = expl.get("counterfactual") or {}
                    if cf.get("text"):
                        st.write(cf.get("text"))

                    # Phase 2a: auction-conditioned posteriors
                    phase2a = details.get("phase2a") or {}
                    roles = (phase2a.get("roles") or {}) if isinstance(phase2a, dict) else {}
                    threats = (phase2a.get("threats") or {}) if isinstance(phase2a, dict) else {}
                    keycards = (phase2a.get("keycards") or {}) if isinstance(phase2a, dict) else {}
                    onside = (phase2a.get("onside") or {}) if isinstance(phase2a, dict) else {}
                    rp = details.get("range_percentiles") or {}

                    if roles:
                        st.markdown("**Phase 2a: Auction-conditioned posteriors (SELF/PARTNER/LHO/RHO)**")
                        # Show compact threat + onside summaries (tables are too big for now)
                        if threats:
                            tdf = pl.DataFrame(
                                [{"Suit": su, **(threats.get(su) or {})} for su in ["S", "H", "D", "C"] if su in threats]
                            )
                            render_aggrid(
                                tdf,
                                key=f"phase2a_threats_{exp_bid}_{CACHE_VERSION}",
                                height=calc_grid_height(len(tdf), max_height=220),
                                table_name="phase2a_threats",
                                fit_columns_to_view=True,
                                show_sql_expander=False,
                            )
                        if isinstance(rp, dict) and rp.get("deal_index") is not None:
                            st.caption(
                                f"Pinned-deal percentiles are computed vs the SELF posterior (after pinned exclusion). "
                                f"deal_index={rp.get('deal_index')}, role={rp.get('role')}"
                            )
                        if keycards:
                            st.caption("Keycards are computed per-suit (treating each suit as potential trump).")
                        if onside:
                            st.caption("Onside convention: p_onside = P(honor with LHO), p_offside = P(honor with RHO).")
                except Exception as e:
                    st.warning(f"Could not load selected-bid details/explanation: {e}")

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
                        update_on=["selectionChanged"],
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
                update_on=["selectionChanged"],
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
        # - add the selected contract's DD_Score as a dedicated column and sort by it
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
                # Attach selected-contract DD score for stable sorting/display.
                rr = dict(d)
                rr["DD_Score_Selected"] = rr.get(score_col)
                new_filtered.append(rr)
        
        # Sort by selected DD score desc (best for this contract first), then by index asc.
        try:
            new_filtered.sort(
                key=lambda r: (
                    float(r.get("DD_Score_Selected")) if isinstance(r.get("DD_Score_Selected"), (int, float)) else float("-inf"),
                    -float(r.get("index")) if isinstance(r.get("index"), (int, float)) else 0.0,
                ),
                reverse=True,
            )
        except Exception:
            pass

        filtered_deals = new_filtered
        selection_msg = f" (Deals for {level}{strain_alias} by {declarer} ({vul_state}))"
    
    if filtered_deals and has_selection:
        st.subheader(f"📈 Deals for {sel_level}{sel_strain} by {sel_declarer} ({sel_vul}) ({len(filtered_deals)} deals)")
        
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

        # Keep the selected-contract DD score visible even if DD_* columns are hidden.
        if "DD_Score_Selected" in dd_df.columns:
            try:
                dd_df = dd_df.with_columns(pl.col("DD_Score_Selected").cast(pl.Float64, strict=False))
            except Exception:
                pass
        
        # Order columns sensibly
        # DD_Score and EV_Score show the score/EV for the matching BT bid specifically
        priority_cols = ["index", "Dealer", "Vul", "DD_Score_Selected"]
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


def render_auction_builder():  # pyright: ignore[reportGeneralTypeIssues]
    """Build an auction step-by-step by selecting bids from BT-derived options."""
    st.header("🔨 Auction Builder")
    st.markdown("""
    Build an auction step-by-step. At each step, select the next bid from available 
    BT continuations. See criteria (Agg_Expr) for each seat and find matching deals.
    
    **Tip:** Pin a specific deal to see if each step's criteria match that deal.
    """)

    # Reserve space for Notes immediately under the intro.
    # We render into this placeholder later (after pinned deal resolution).
    auction_note_slot = st.empty()
    
    # Initialize session state for auction path
    if "auction_builder_path" not in st.session_state:
        st.session_state.auction_builder_path = []  # List of {"bid": str, "bt_index": int, "agg_expr": list}
    if "auction_builder_options" not in st.session_state:
        st.session_state.auction_builder_options = {}  # Cache of available options per prefix
    if "auction_builder_pinned_deal" not in st.session_state:
        st.session_state.auction_builder_pinned_deal = None  # Cached pinned deal data

    # On entry to Auction Builder, auto-restore last pinned deal index (if nothing currently pinned).
    entered = bool(st.session_state.pop("_entered_auction_builder", False))
    if entered:
        # Sync custom questions into docs (fail-fast if CSV is missing/invalid).
        # Best practice: gate on CSV content hash so we don't rewrite docs if unchanged.
        try:
            repo_root = pathlib.Path(__file__).resolve().parent
            sha1_now = file_sha1(CUSTOM_QUESTIONS_CSV)
            sha1_prev = str(st.session_state.get("_custom_questions_csv_sha1") or "")
            if sha1_now != sha1_prev:
                meta = sync_questions_into_docs(
                    questions_csv=CUSTOM_QUESTIONS_CSV,
                    players_md=repo_root / "docs" / "BIDDING_AI_QA_FOR_PLAYERS.md",
                    coders_md=repo_root / "docs" / "BIDDING_AI_QA_FOR_CODERS.md",
                )
                st.session_state["_custom_questions_csv_sha1"] = meta.get("csv_sha1") or sha1_now
            else:
                # Already synced this exact CSV content in this Streamlit session.
                st.session_state["_custom_questions_csv_sha1"] = sha1_now
        except Exception as e:
            st.error(f"Custom questions sync failed: {e}")
            st.stop()

        try:
            curr_idx = int(st.session_state.get("auction_builder_pinned_index", 0) or 0)
        except Exception:
            curr_idx = 0
        curr_pbn = str(st.session_state.get("auction_builder_pinned_pbn", "") or "").strip()
        if curr_idx <= 0 and not curr_pbn:
            try:
                last_idx = int(st.session_state.get("auction_builder_last_saved_deal_index", 0) or 0)
            except Exception:
                last_idx = 0
            if last_idx <= 0:
                last_idx = _auction_builder_load_last_pinned_deal_index()
            if last_idx > 0:
                st.session_state["auction_builder_pinned_index"] = int(last_idx)
                st.session_state["auction_builder_pinned_pbn"] = ""

    def _clear_current_auction_state() -> None:
        """Clear the current auction path + related caches (does not clear pinned deal)."""
        st.session_state.auction_builder_path = []
        st.session_state.auction_builder_options = {}
        st.session_state.auction_builder_last_applied = ""
        st.session_state.auction_builder_last_selected = ""
        # Clear rehydration tracking keys, deals counts cache, categories cache, and best auctions caches
        keys_to_clear = []
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and (
                key.startswith("_rehydrated_")
                or key.startswith("_best_auctions_")
                or key.startswith("show_best_auctions_")
            ):
                keys_to_clear.append(key)
        for key in keys_to_clear:
            del st.session_state[key]
        if "_deals_counts_all" in st.session_state:
            del st.session_state["_deals_counts_all"]
        if "_deals_counts_elapsed_ms" in st.session_state:
            del st.session_state["_deals_counts_elapsed_ms"]
        if "_deals_counts_steps" in st.session_state:
            del st.session_state["_deals_counts_steps"]
        if "auction_builder_bt_categories_cache" in st.session_state:
            del st.session_state["auction_builder_bt_categories_cache"]
    
    # Sidebar controls
    
    # CSS to remove border and move X button left
    st.sidebar.markdown("""
        <style>
        /* Target the clear button specifically */
        div.stButton > button[kind="secondary"] {
            /* We need to be careful not to affect ALL buttons, but st.button uses kind="secondary" by default */
        }
        /* Use a more specific selector for our clear button */
        .st-key-clear_pin_btn button {
            border: none !important;
            background: transparent !important;
            margin-left: -48px !important;
            box-shadow: none !important;
            color: #888 !important;
        }
        .st-key-clear_pin_btn button:hover {
            color: #f00 !important;
            background: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Use columns to place a clear button next to the input
    pin_col_in, pin_col_cl = st.sidebar.columns([0.88, 0.12])
    with pin_col_in:
        def _on_change_pinned_index() -> None:
            # Make numeric pin authoritative: clear PBN/LIN when index changes to >0
            try:
                idx_v = int(st.session_state.get("auction_builder_pinned_index", 0) or 0)
            except Exception:
                idx_v = 0
            if idx_v > 0:
                st.session_state["auction_builder_pinned_pbn"] = ""
                # Remember last pinned index immediately (so it can be restored on next entry).
                st.session_state["auction_builder_last_saved_deal_index"] = idx_v
                _auction_builder_save_last_pinned_deal_index(idx_v)

        def _on_change_pinned_pbn() -> None:
            # Make PBN/LIN pin authoritative: clear index when PBN/LIN is non-empty
            p = str(st.session_state.get("auction_builder_pinned_pbn", "") or "").strip()
            if p:
                st.session_state["auction_builder_pinned_index"] = 0

        pinned_idx = st.number_input(
            "Deal Index",
            min_value=0,
            value=int(st.session_state.get("auction_builder_pinned_index", 0) or 0),
            step=1,
            help="Enter a deal index number to pin. Criteria will be evaluated against this deal.",
            key="auction_builder_pinned_index",
            on_change=_on_change_pinned_index,
        )
        pinned_pbn = st.text_input(
            "PBN/LIN (optional)",
            value=str(st.session_state.get("auction_builder_pinned_pbn", "") or ""),
            placeholder="e.g., N:AKQ.xxx.xxx.xxxx ...",
            help="Optional: paste a PBN/LIN deal string instead of a deal index.",
            key="auction_builder_pinned_pbn",
            on_change=_on_change_pinned_pbn,
        )
    with pin_col_cl:
        # Vertical alignment hack for the button
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        
        def _clear_pin():
            # Clear all pinned deal caches
            keys_to_clear = [k for k in st.session_state if k.startswith("pinned_deal_")]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.auction_builder_pinned_index = 0
            st.session_state.auction_builder_pinned_pbn = ""

        st.button("❌", key="clear_pin_btn", help="Clear pinned deal", on_click=_clear_pin)

    # Compute the actual pinned input (PBN wins if provided; else numeric index if > 0)
    pinned_pbn_str = str(pinned_pbn or "").strip()
    pinned_input = pinned_pbn_str if pinned_pbn_str else (str(int(pinned_idx)) if int(pinned_idx) > 0 else "")

    # If the pinned-deal textbox is changed (to a different value or cleared), clear the current auction.
    # This avoids stale "Current Auction" state being interpreted under a different pin context.
    last_pin_key = "_auction_builder_last_pinned_input"
    prev_pin = str(st.session_state.get(last_pin_key, "") or "").strip()
    curr_pin = str(pinned_input or "").strip()
    if prev_pin != curr_pin:
        _clear_current_auction_state()
        st.session_state[last_pin_key] = curr_pin
        st.rerun()
    st.session_state[last_pin_key] = curr_pin
    
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
                    # Request EV column if available (shown in pinned invariants table as EV)
                    # Build columns list including DD score columns for completion banner
                    # Note: Raw trick columns (DD_{dir}_{strain}) don't exist in dataset.
                    pinned_cols = [
                        "index",
                        "Dealer",
                        "Vul",
                        "Hand_N",
                        "Hand_E",
                        "Hand_S",
                        "Hand_W",
                        "HCP_N",
                        "HCP_E",
                        "HCP_S",
                        "HCP_W",
                        "Total_Points_N",
                        "Total_Points_E",
                        "Total_Points_S",
                        "Total_Points_W",
                        # Suit lengths per direction (for Actual Stats display)
                        "SL_S_N", "SL_S_E", "SL_S_S", "SL_S_W",
                        "SL_H_N", "SL_H_E", "SL_H_S", "SL_H_W",
                        "SL_D_N", "SL_D_E", "SL_D_S", "SL_D_W",
                        "SL_C_N", "SL_C_E", "SL_C_S", "SL_C_W",
                        "ParContracts",
                        "ParScore",
                        "EV_Score_Declarer",
                        "Contract",
                        "Score",
                        "Result",
                        "bid",
                    ]
                    # DD score columns: DD_Score_{level}{strain}_{direction} (140 columns)
                    for lvl in range(1, 8):
                        for s in ["C", "D", "H", "S", "N"]:
                            for d in ["N", "E", "S", "W"]:
                                pinned_cols.append(f"DD_Score_{lvl}{s}_{d}")
                    # Raw DD trick columns: DD_{direction}_{strain} (20 columns)
                    for d in ["N", "E", "S", "W"]:
                        for s in ["C", "D", "H", "S", "N"]:
                            pinned_cols.append(f"DD_{d}_{s}")
                    # EV columns: EV_{pair}_{declarer}_{strain}_{level} (140 columns)
                    # Note: EV columns do NOT have vulnerability suffix
                    for pair, decl_dirs in [("NS", ["N", "S"]), ("EW", ["E", "W"])]:
                        for decl in decl_dirs:
                            for strain in ["C", "D", "H", "S", "N"]:
                                for lvl in range(1, 8):
                                    pinned_cols.append(f"EV_{pair}_{decl}_{strain}_{lvl}")
                    # NOTE: We intentionally do NOT request the single-dummy probability matrix (Probs_*)
                    # here. It is very wide (560+ columns) and can cause timeouts when pinning
                    # deals quickly via the Deal Index spinner. Undoubled EV is still available
                    # via the precomputed EV_* columns; doubled/redoubled EV will be None.
                    data = api_post(
                        "/deals-by-index",
                        {
                            "indices": [idx],
                            "max_rows": 1,
                            "columns": pinned_cols,
                        },
                        timeout=20,
                    )
                    _st_info_elapsed("Auction Builder: pin deal (by index)", data)
                    rows = data.get("rows", [])
                    if rows:
                        deal_data = dict(rows[0])
                        deal_data["_source"] = f"index:{idx}"
                        pinned_deal = deal_data
                        # Remember last successfully pinned deal index.
                        st.session_state["auction_builder_last_saved_deal_index"] = idx
                        _auction_builder_save_last_pinned_deal_index(idx)
                    else:
                        pinned_deal_error = (
                            f"Deal index {idx} not found in the API server's loaded deals dataset. "
                            "This can happen if the server was started with a limited `--deal-rows` subset "
                            "or if that deal isn't present in the dataset (e.g., missing/partial auction data). "
                            "Try pinning via PBN instead."
                        )
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
    
    # Show pinned deal error in sidebar (success/hands moved to main area)
    if pinned_deal_error:
        st.sidebar.error(pinned_deal_error)
    
    # When a deal is pinned, always show failed criteria coloring and use 2 dataframes
    show_failed_criteria = bool(pinned_deal)
    # Also split Available vs Rejected even without a pinned deal, so rows are bucketed
    # consistently (dead end / no matches / no deals / missing criteria).
    use_two_dataframes = True

    # Render Notes box under the Auction Builder intro.
    with auction_note_slot.container(border=True):
        st.markdown("**📝 Auction note**")
        deal_idx_int: int | None = None
        if pinned_deal:
            try:
                deal_idx_int = int(pinned_deal.get("index")) if pinned_deal.get("index") is not None else None
            except Exception:
                deal_idx_int = None

        if deal_idx_int is None:
            st.caption("Pin a deal **index** to view/edit notes.")
        else:
            notes_df = _auction_builder_load_notes_df(AUCTION_BUILDER_NOTES_CSV)
            notes_m = _auction_builder_notes_map(notes_df)
            saved_note = notes_m.get(deal_idx_int, "")

            note_key = f"auction_builder_note_{deal_idx_int}"
            saved_key = f"auction_builder_note_saved_{deal_idx_int}"
            status_key = f"auction_builder_note_status_{deal_idx_int}"
            if saved_key not in st.session_state:
                st.session_state[saved_key] = saved_note
                st.session_state[note_key] = saved_note

            def _auction_note_save_cb(di: int, nk: str, sk: str, stk: str) -> None:
                """Callback (runs before widget instantiation on rerun)."""
                try:
                    auction_builder_upsert_or_delete_note(
                        deal_index=di,
                        note=str(st.session_state.get(nk, "") or ""),
                    )
                    refreshed = _auction_builder_notes_map(
                        _auction_builder_load_notes_df(AUCTION_BUILDER_NOTES_CSV)
                    ).get(di, "")
                    st.session_state[sk] = refreshed
                    st.session_state[nk] = refreshed
                    st.session_state[stk] = {"ok": True, "msg": "Saved."}
                except Exception as e:
                    st.session_state[stk] = {"ok": False, "msg": f"Could not save note: {e}"}

            def _auction_note_cancel_cb(nk: str, sk: str) -> None:
                """Callback (runs before widget instantiation on rerun)."""
                st.session_state[nk] = str(st.session_state.get(sk, "") or "")

            # Single-line notes: Enter (or blur) auto-saves via on_change callback.
            st.text_input(
                "Auction note (max 200 chars)",
                key=note_key,
                max_chars=AUCTION_BUILDER_NOTES_MAX_CHARS,
                label_visibility="collapsed",
                placeholder="Describe the issue with this auction…",
                on_change=_auction_note_save_cb,
                args=(deal_idx_int, note_key, saved_key, status_key),
            )

            btn_save, btn_cancel, _ = st.columns([1, 1, 6])
            with btn_save:
                st.button(
                    "Save",
                    key=f"auction_builder_note_save_{deal_idx_int}",
                    width="stretch",
                    on_click=_auction_note_save_cb,
                    args=(deal_idx_int, note_key, saved_key, status_key),
                )
            with btn_cancel:
                st.button(
                    "Cancel",
                    key=f"auction_builder_note_cancel_{deal_idx_int}",
                    width="stretch",
                    on_click=_auction_note_cancel_cb,
                    args=(note_key, saved_key),
                )

            # Render status message set by callbacks
            status = st.session_state.get(status_key)
            if isinstance(status, dict):
                if status.get("ok"):
                    # Use toast so "Saved." doesn't linger while editing.
                    st.toast(str(status.get("msg") or "Saved."))
                else:
                    st.error(str(status.get("msg") or "Could not save note."))
                try:
                    del st.session_state[status_key]
                except Exception:
                    pass
    
    
    with st.sidebar.expander("Settings", expanded=False):
        max_matching_deals = st.number_input(
            "Max Matching Deals",
            value=25,
            min_value=1,
            max_value=500,
            help="Maximum deals to show in matching deals list"
        )

        max_bt_criteria_deals = st.number_input(
            "Max BT Criteria Deals",
            value=2000,
            min_value=1,
            max_value=20000,
            help="Maximum deals to show for BT criteria-matching deal sets (backs the 'Deals' column for a next bid).",
        )

        max_best_auctions = st.number_input(
            "Max Best Auctions",
            value=25,
            min_value=1,
            max_value=1000,
            help="Maximum rows to show in 'Best Auctions by DD/EV' (samples from deal→BT matches).",
        )
        
        seed = st.number_input(
            "Random Seed",
            value=42,
            min_value=0,
            help="Seed for reproducible deal sampling"
        )
        
        always_valid_pass = st.checkbox(
            "Always treat Pass as valid bid",
            value=True,
            help="If checked, Pass bids are always considered valid even if their criteria fail. "
                 "This allows auctions with Pass to be included in Best Auctions results.",
            key="always_valid_pass"
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
    
    # Display pinned deal in main area (above Current Auction)
    if pinned_deal:
        deal_idx = pinned_deal.get("index")
        deal_title = f"📋 Current Deal (deal_index: {deal_idx})" if deal_idx is not None else "📋 Current Deal"
        render_deal_diagram(pinned_deal, title=deal_title)

        # Show pinned-deal invariant values (single-row table)
        # Use a column container to constrain width (prevents full-container expansion)
        inv = {
            "ParContracts": pinned_deal.get("ParContracts"),
            "ParScore": pinned_deal.get("ParScore", pinned_deal.get("Par_Score")),
            "EV": pinned_deal.get("EV_Score_Declarer", pinned_deal.get("EV_Score")),
        }
        inv_df = pl.DataFrame([inv])
        inv_col, _ = st.columns([1, 2])  # Use 1/3 of container width
        with inv_col:
            render_aggrid(
                inv_df,
                key=f"auction_builder_pinned_invariants_{deal_idx}",
                height=95,
                table_name="auction_builder_pinned_invariants",
                fit_columns_to_view=True,
            )
    
    # -----------------------------------------------------------------------
    # Cheater Model (derived from "Best Bids Ranked by Cheating Model")
    # -----------------------------------------------------------------------
    # We render the placeholder here (above "Bidding Sequence") but fill it later after
    # we compute suggested bids. This avoids reordering the UI logic.
    best_auction_placeholder = st.empty()
    ai_model_auction_placeholder = st.empty()
    ai_model_perf_placeholder = st.empty()
    ai_model_times_placeholder = st.empty()
    completed_summary_placeholder = st.empty()
    # Cache Best Auction per pinned deal so it remains invariant to the current built auction.
    best_auc_deal_key: str | None = None
    try:
        if pinned_deal:
            # Prefer stable row id when available; fall back to user-facing index.
            rk = pinned_deal.get("_row_idx")
            ik = pinned_deal.get("index")
            if rk is not None:
                best_auc_deal_key = f"row_{int(rk)}"
            elif ik is not None:
                best_auc_deal_key = f"idx_{int(ik)}"
    except Exception:
        best_auc_deal_key = None
    best_auc_ss_key = f"_auction_builder_best_auction__{best_auc_deal_key}" if best_auc_deal_key else "_auction_builder_best_auction"
    ai_model_auc_ss_key = (
        f"_auction_builder_ai_model_advanced_auc__{best_auc_deal_key}"
        if best_auc_deal_key
        else "_auction_builder_ai_model_advanced_auc"
    )
    ai_model_times_ss_key = (
        f"_auction_builder_ai_model_advanced_times__{best_auc_deal_key}"
        if best_auc_deal_key
        else "_auction_builder_ai_model_advanced_times"
    )
    ai_model_job_ss_key = (
        f"_auction_builder_ai_model_advanced_job__{best_auc_deal_key}"
        if best_auc_deal_key
        else "_auction_builder_ai_model_advanced_job"
    )
    ai_model_err_ss_key = (
        f"_auction_builder_ai_model_advanced_error__{best_auc_deal_key}"
        if best_auc_deal_key
        else "_auction_builder_ai_model_advanced_error"
    )
    ai_model_perf_ss_key = (
        f"_auction_builder_ai_model_advanced_perf__{best_auc_deal_key}"
        if best_auc_deal_key
        else "_auction_builder_ai_model_advanced_perf"
    )
    try:
        cached_best_auc = st.session_state.get(best_auc_ss_key)
        if isinstance(cached_best_auc, str) and cached_best_auc.strip():
            best_auction_placeholder.empty()
    except Exception:
        pass
    try:
        cached_ai_auc = st.session_state.get(ai_model_auc_ss_key)
        if isinstance(cached_ai_auc, str) and cached_ai_auc.strip():
            ai_model_auction_placeholder.empty()
        else:
            # Show computing/error state if a server-side job is in flight.
            job_id = st.session_state.get(ai_model_job_ss_key)
            if isinstance(job_id, str) and job_id.strip():
                ai_model_auction_placeholder.empty()
            err = st.session_state.get(ai_model_err_ss_key)
            if isinstance(err, str) and err.strip():
                ai_model_auction_placeholder.error(f"AI Model (Advanced) error: {err.strip()}")
        perf_s = st.session_state.get(ai_model_perf_ss_key)
        if isinstance(perf_s, str) and perf_s.strip():
            ai_model_perf_placeholder.caption(perf_s.strip())
        cached_ai_times = st.session_state.get(ai_model_times_ss_key)
        if isinstance(cached_ai_times, list) and cached_ai_times:
            lines = ["AI Model Step Details:"]
            for j, t in enumerate(cached_ai_times[:60], start=1):
                if not isinstance(t, dict):
                    continue
                b = str(t.get("bid", "") or "").strip().upper()
                ms = t.get("elapsed_ms")
                score = t.get("score")
                sign = t.get("acting_sign")
                try:
                    ms_f = float(ms) if ms is not None else None
                except Exception:
                    ms_f = None
                if b and ms_f is not None:
                    score_s = f"{score:.1f}" if score is not None else "?"
                    sign_s = f"{sign:+.0f}" if sign is not None else "?"
                    lines.append(f"- {j}. `{b}`: {ms_f:.0f} ms | Score: {score_s} (Sign: {sign_s})")
            if len(lines) > 1:
                ai_model_times_placeholder.markdown("\n".join(lines))
    except Exception:
        pass

    # Display current auction state with editable input
    # Include bt_index from last step if available (use 'is not None' since bt_index=0 is valid)
    current_bt_index = current_path[-1].get("bt_index") if current_path else None
    auction_header = f"Bidding Sequence (bt_index: {current_bt_index})" if current_bt_index is not None else "Bidding Sequence"
    st.subheader(auction_header)
    
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
        btn_cols = st.columns([2, 2, 2, 2], gap="small")
        with btn_cols[0]:
            apply_edit = st.button("Apply", key="auction_builder_apply_edit", width="stretch")
        with btn_cols[1]:
            append_ppp = st.button("Pass Out", key="auction_builder_append_ppp", help="Append '-P-P-P' and apply", width="stretch")
        with btn_cols[2]:
            if current_path and st.button("Undo", key="auction_builder_undo", width="stretch"):
                st.session_state.auction_builder_path.pop()
                st.session_state.auction_builder_last_applied = ""
                st.session_state.auction_builder_last_selected = ""  # Reset to allow re-selecting same bid
                st.rerun()
        with btn_cols[3]:
            if st.button("Clear", key="auction_builder_clear", help="Clear the current auction", width="stretch"):
                _clear_current_auction_state()
                st.rerun()

    # If another UI element wants to "move" an auction into the bidding sequence,
    # it can set this key and we will apply it via the existing resolve/apply logic.
    pending_set_key = "_auction_builder_pending_set_auction"
    pending_auction = st.session_state.get(pending_set_key)
    if isinstance(pending_auction, str) and pending_auction.strip():
        try:
            st.session_state[edit_key] = pending_auction.strip()
        except Exception:
            pass
        edited_auction = pending_auction.strip()
        st.session_state.auction_builder_last_applied = ""  # ensure it applies
        apply_edit = True
        try:
            del st.session_state[pending_set_key]
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # TODO(best-completions):
    # Re-enable "Best completions (DD/EV)" once PBN pinning supports on-the-fly
    # enrichment to a deal_row_idx (or equivalent server-side deal context),
    # so we can compute best completions for pinned PBN deals, not just indexed deals.
    #
    # The API endpoints remain available:
    # - POST /best-auctions-lookahead/start
    # - GET  /best-auctions-lookahead/status/{job_id}
    # -----------------------------------------------------------------------

    # Handle manual auction edit - trigger on Enter (value change) OR button click
    # But skip if we just applied this same value (prevents render loop)
    edited_normalized = normalize_auction_input(edited_auction).upper() if edited_auction else ""

    def _append_passes_to_complete(auction_text: str) -> str:
        """Append the minimum number of passes needed to complete the auction.

        - If there has been no non-pass bid yet: complete with 4 total passes (passed out).
        - Otherwise: complete with 3 trailing passes after the last non-pass call.
        """
        toks = [t.strip().upper() for t in str(auction_text or "").split("-") if t.strip()]
        trailing = 0
        for t in reversed(toks):
            if t == "P":
                trailing += 1
            else:
                break
        has_non_pass = any(t != "P" for t in toks)
        if not has_non_pass:
            needed = max(0, 4 - len(toks))
        else:
            needed = max(0, 3 - trailing)
        if needed <= 0:
            return "-".join(toks)
        return "-".join(toks + ["P"] * needed)

    # Convenience: append trailing passes and apply immediately.
    if append_ppp:
        # Prefer current text box value; fall back to current_auction.
        base_in = edited_normalized or (current_auction.upper() if current_auction else "")
        base = _append_passes_to_complete(base_in)
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
            # Validate auction before processing
            if not is_auction_valid(edited_normalized, require_complete=False):
                validation_result = validate_auction(edited_normalized, require_complete=False)
                error_msg = validation_result.get("error", "Invalid auction")
                st.error(f"❌ Invalid auction: {error_msg}")
                # Don't process invalid auctions
                st.session_state.auction_builder_last_applied = edited_normalized  # Prevent re-trigger on invalid input
                st.stop()
            
            # Build expected bids list for length comparison
            expected_bids = [b.strip().upper() for b in edited_normalized.split("-") if b.strip()]
            expected_len = len(expected_bids)
            
            # Single batched call to resolve the entire path
            with st.spinner(f"Resolving path: {edited_normalized}..."):
                try:
                    data = api_post("/resolve-auction-path", {"auction": edited_normalized}, timeout=30)
                    _st_info_elapsed("Auction Builder: resolve path", data)
                    new_path = data.get("path", [])
                    # If API returns a shorter path (BT doesn't have all bids), fall back to user input
                    if not new_path or len(new_path) < expected_len:
                        new_path = [{"bid": b, "bt_index": None, "agg_expr": [], "is_complete": False} for b in expected_bids]
                except Exception as e:
                    st.error(f"Error resolving auction path: {e}")
                    new_path = [{"bid": b, "bt_index": None, "agg_expr": [], "is_complete": False} for b in expected_bids]
            
            # Mark is_complete on final step if auction ends with 3 passes (or 4 passes for passed out)
            if new_path:
                all_bids = [step.get("bid", "").upper() for step in new_path]
                trailing_p = 0
                for b in reversed(all_bids):
                    if b == "P":
                        trailing_p += 1
                    else:
                        break
                has_non_pass = any(b != "P" for b in all_bids)
                is_complete = (trailing_p >= 3 and has_non_pass) or (not has_non_pass and len(all_bids) >= 4)
                new_path[-1]["is_complete"] = is_complete
            
            st.session_state.auction_builder_path = new_path
            st.session_state.auction_builder_options = {}  # Clear cache
            st.session_state.auction_builder_last_applied = edited_normalized  # Prevent re-trigger
            st.session_state.auction_builder_last_selected = ""  # Reset to allow selecting any bid
            # Mark as already rehydrated (we just resolved the full path)
            st.session_state[f"_rehydrated_{edited_normalized}"] = True
            st.rerun()
        else:
            # Clear auction
            st.session_state.auction_builder_path = []
            st.session_state.auction_builder_options = {}
            st.session_state.auction_builder_last_applied = ""  # Prevent re-trigger
            st.session_state.auction_builder_last_selected = ""  # Reset to allow selecting any bid
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
    
    # "Auction Complete" status is shown in the Auction Summary section (after criteria evaluation)
    # to ensure we know whether the pinned deal passes all criteria before displaying.
    
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
            payload: dict[str, Any] = {"auction": prefix or ""}
            # If a pinned deal exists, pass board context so the API can compute seat_to_act + seat_vul
            # and return convenience avg_par/avg_ev fields aligned with BIDDING_MODEL_IMPLEMENTATION.md.
            try:
                if pinned_deal:
                    payload["dealer"] = pinned_deal.get("Dealer")
                    payload["vulnerable"] = pinned_deal.get("Vul", pinned_deal.get("Vulnerability"))
            except Exception:
                pass
            data = api_post("/list-next-bids", payload, timeout=DEFAULT_API_TIMEOUT)
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
                    # BT raw per-step criteria (Expr column)
                    "expr": item.get("expr", []) or [],
                    "agg_expr": item.get("agg_expr", []) or [],
                    "can_complete": item.get("can_complete"),
                    "is_complete": item.get("is_completed_auction", False),
                    "is_dead_end": item.get("is_dead_end", False),
                    "matching_deal_count": item.get("matching_deal_count"),
                    # Precomputed EV split by vulnerability (optional, may be missing/None)
                    "avg_ev_nv": item.get("avg_ev_nv"),
                    "avg_ev_v": item.get("avg_ev_v"),
                    "avg_par_nv": item.get("avg_par_nv"),
                    "avg_par_v": item.get("avg_par_v"),
                    # Convenience (selected by board context if present)
                    "avg_ev": item.get("avg_ev"),
                    "avg_par": item.get("avg_par"),
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
    
    def _is_auction_complete_after_next_bid(current_auction_text: str, next_bid: str) -> bool:
        """Best-effort bridge completion rule for manual 'P' continuation.
        
        - Passed out: P-P-P-P
        - Otherwise: 3 consecutive passes after any non-pass bid
        """
        try:
            parts: list[str] = []
            if current_auction_text:
                parts = [p.strip().upper() for p in str(current_auction_text).split("-") if p.strip()]
            parts.append(str(next_bid).strip().upper())
            if not parts:
                return False
            # Passed out (4 passes)
            if len(parts) >= 4 and all(p == "P" for p in parts):
                return True
            # 3 trailing passes after any non-pass bid
            trailing = 0
            for p in reversed(parts):
                if p == "P":
                    trailing += 1
                else:
                    break
            if trailing >= 3 and any(p != "P" for p in parts):
                return True
        except Exception:
            return False
        return False

    def _next_display_seat_from_auction(auction_text: str) -> int:
        """Return the next bidder seat number (1-4) in display/dealer-relative order."""
        try:
            toks = [t.strip().upper() for t in str(auction_text or "").split("-") if t.strip()]
            return (len(toks) % 4) + 1
        except Exception:
            return 1

    def _greedy_best_auction_complete(
        start_auction: str,
        *,
        max_steps: int = 40,
    ) -> str | None:
        """Greedy completion: repeatedly apply the top-most Valid/green bid until complete.

        "Valid/green" semantics match the Auction Builder:
        - _matches == True (pinned deal passes criteria) and not rejected
        - Pass is optionally treated as always-valid (user setting)
        """
        if not pinned_deal:
            return None

        auc = normalize_auction_input(start_auction).upper() if start_auction else ""
        auc = auc.strip("-")

        for _ in range(max_steps):
            # Completion check (covers both passed-out and 3-pass endings).
            toks = [t.strip().upper() for t in str(auc or "").split("-") if t.strip()]
            if toks:
                if len(toks) >= 4 and all(t == "P" for t in toks):
                    return auc
                trailing = 0
                for t in reversed(toks):
                    if t == "P":
                        trailing += 1
                    else:
                        break
                if trailing >= 3 and any(t != "P" for t in toks):
                    return auc

            seat_1_to_4 = _next_display_seat_from_auction(auc)
            bt_prefix, leading_passes = _strip_leading_passes(auc)

            options = get_next_bid_options(bt_prefix)
            options = list(options or [])

            # Opening-pass support (mirror main UI logic).
            if bt_prefix == "":
                has_p = any(str(o.get("bid") or "").strip().upper() == "P" for o in options)
                if not has_p:
                    options.append(
                        {
                            "bid": "P",
                            "bt_index": None,
                            "agg_expr": [],
                            "is_complete": _is_auction_complete_after_next_bid(auc, "P"),
                        }
                    )

            # Permissive-pass support (mirror main UI logic).
            if st.session_state.get("always_valid_pass", True):
                has_p = any(str(o.get("bid") or "").strip().upper() == "P" for o in options)
                if not has_p:
                    options.append(
                        {
                            "bid": "P",
                            "bt_index": None,
                            "expr": [],
                            "agg_expr": [],
                            "can_complete": True,
                            "is_complete": _is_auction_complete_after_next_bid(auc, "P"),
                            "is_dead_end": False,
                            "matching_deal_count": 0,
                            "avg_ev_nv": None,
                            "avg_ev_v": None,
                        }
                    )

            if not options:
                return auc

            # Score and choose the best Valid option.
            dealer_actual = str(pinned_deal.get("Dealer", "N")).upper()
            vul = str(pinned_deal.get("Vul", pinned_deal.get("Vulnerability", ""))).upper()
            ns_vul = vul in ("N_S", "NS", "BOTH", "ALL")
            ew_vul = vul in ("E_W", "EW", "BOTH", "ALL")

            best_score: float = float("-inf")
            best_bid: str | None = None
            best_is_complete = False

            for opt in options:
                bid_str = str(opt.get("bid", "")).strip().upper()
                if not bid_str:
                    continue
                is_pass = bid_str in ("P", "PASS")
                pass_always_valid = bool(is_pass and st.session_state.get("always_valid_pass", True))
                criteria_list = opt.get("agg_expr", []) or []
                pass_no_criteria = bool(is_pass and (not criteria_list))

                # Determine if pinned deal matches criteria for this bid.
                matches_pinned = True
                if show_failed_criteria:
                    matches_pinned, _failed_list = check_pinned_match_with_failures(criteria_list, seat_1_to_4)
                if pass_always_valid:
                    matches_pinned = True

                # Rejection logic (mirror main UI).
                is_dead_end = bool(opt.get("is_dead_end", False))
                has_empty_criteria = (not criteria_list)
                can_complete = opt.get("can_complete")
                can_complete_b = bool(can_complete) if can_complete is not None else True

                # WORKAROUND: Invalidate bids with underspecified criteria (missing HCP and Total_Points)
                is_underspecified = _is_underspecified_criteria(criteria_list)

                is_rejected = is_dead_end or has_empty_criteria or is_underspecified
                if pass_no_criteria:
                    is_rejected = False
                if pass_always_valid:
                    is_rejected = False
                if show_failed_criteria and matches_pinned is True and (not can_complete_b):
                    if not (pass_no_criteria or pass_always_valid):
                        is_rejected = True

                if matches_pinned is not True or is_rejected:
                    continue

                # Seat direction for vulnerability/sign logic in score calculation.
                seat_direction = None
                try:
                    directions = ["N", "E", "S", "W"]
                    dealer_idx = directions.index(str(dealer_actual).upper()) if str(dealer_actual).upper() in directions else 0
                    seat_direction = directions[(dealer_idx + int(seat_1_to_4) - 1) % 4]
                except Exception:
                    seat_direction = None

                # Compute "model score" for this bid using the same logic as the grid.
                score_val: float = float("-inf")
                try:
                    next_auc = f"{auc}-{bid_str}" if auc else bid_str
                    next_auc = normalize_auction_input(next_auc).upper() if next_auc else ""

                    # Determine doubling state from auction calls after last contract.
                    bids = [b.strip().upper() for b in (next_auc or "").split("-") if b.strip()]
                    last_contract_idx = -1
                    for i in range(len(bids) - 1, -1, -1):
                        bb = bids[i]
                        if not bb or bb == "P":
                            continue
                        if bb in ("X", "D", "DBL", "DOUBLE"):
                            continue
                        if bb in ("XX", "R", "RDBL", "REDOUBLE"):
                            continue
                        if len(bb) >= 2 and bb[0].isdigit():
                            last_contract_idx = i
                            break

                    is_dbl_val = False
                    is_rdbl_val = False
                    dbl_val = ""
                    if last_contract_idx >= 0:
                        is_doubled = False
                        is_redoubled = False
                        for bb in bids[last_contract_idx + 1 :]:
                            if bb in ("X", "D", "DBL", "DOUBLE"):
                                is_doubled = True
                                is_redoubled = False
                            elif bb in ("XX", "R", "RDBL", "REDOUBLE"):
                                is_redoubled = True
                        is_rdbl_val = bool(is_redoubled)
                        is_dbl_val = bool(is_doubled and not is_redoubled)
                        dbl_val = "XX" if is_rdbl_val else ("X" if is_doubled else "")

                    c = parse_contract_from_auction(next_auc)
                    if not c:
                        # Passed out: treat score as 0
                        if bids and all(bb == "P" for bb in bids):
                            score_val = 0.0
                        else:
                            score_val = float("-inf")
                    else:
                        level_i, strain_i, _pos_i = c
                        declarer = get_declarer_for_auction(next_auc, dealer_actual)
                        if not declarer:
                            score_val = float("-inf")
                        else:
                            declarer_dir = str(declarer).upper()
                            is_ns = declarer_dir in ("N", "S")
                            is_vul_val = (is_ns and ns_vul) or ((not is_ns) and ew_vul)
                            tricks = get_dd_tricks_for_auction(next_auc, dealer_actual, pinned_deal)
                            if tricks is None:
                                score_val = float("-inf")
                            else:
                                result_val = int(tricks) - (int(level_i) + 6)
                                lvl0 = int(level_i) - 1
                                suit0 = mlBridgeLib.StrainSymToValue(str(strain_i).upper())
                                dbl0 = 2 if is_rdbl_val else (1 if is_dbl_val else 0)
                                decl0 = "NSEW".index(declarer_dir)
                                vul0 = 1 if is_vul_val else 0
                                s0 = mlBridgeLib.score(
                                    lvl0,
                                    suit0,
                                    dbl0,
                                    decl0,
                                    vul0,
                                    int(result_val),
                                    declarer_score=True,
                                )
                                # Normalize to "our side" (same as grid logic).
                                if seat_direction is not None:
                                    if (declarer_dir in ("N", "S")) ^ (seat_direction in ("N", "S")):
                                        s0 = -int(s0)
                                score_val = float(s0)
                except Exception:
                    score_val = float("-inf")

                if score_val > best_score:
                    best_score = score_val
                    best_bid = bid_str
                    best_is_complete = bool(opt.get("is_complete", False)) or _is_auction_complete_after_next_bid(auc, bid_str)

            if not best_bid:
                return auc

            auc = f"{auc}-{best_bid}" if auc else best_bid
            auc = normalize_auction_input(auc).upper() if auc else ""
            if best_is_complete:
                return auc

        # Safety cap reached: return best-effort prefix.
        return auc or None

    def _strip_leading_passes(auction_text: str) -> tuple[str, int]:
        """Return (bt_prefix, leading_passes) where bt_prefix omits leading passes before first non-pass.

        This lets Auction Builder support opening passes:
        - Display path can be "P-P-1C-..."
        - BT lookup prefix should be "1C-..." (seat-1 canonical)
        """
        try:
            if not auction_text:
                return "", 0
            toks = [t.strip().upper() for t in str(auction_text).split("-") if t.strip()]
            n = 0
            for t in toks:
                if t == "P":
                    n += 1
                    continue
                break
            # If all passes so far, canonical BT prefix is still opening ("").
            if n >= len(toks):
                return "", int(n)
            bt = "-".join(toks[n:])
            return bt, int(n)
        except Exception:
            return str(auction_text or ""), 0

    def _rotate_dealer_by(dealer: str, offset: int) -> str:
        """Rotate dealer direction by `offset` seats (N->E->S->W).

        Auction Builder supports leading opening passes by stripping them for BT lookup.
        BT paths are seat-1 canonical (opener acts as seat 1). To evaluate criteria against a
        real deal (dealer-relative), we rotate the dealer by the number of leading passes.
        """
        try:
            directions = ["N", "E", "S", "W"]
            d = str(dealer or "N").upper()
            i = directions.index(d) if d in directions else 0
            k = int(offset or 0) % 4
            return directions[(i + k) % 4]
        except Exception:
            return str(dealer or "N").upper()

    def _bt_seat_from_display_seat(seat_1_to_4: int, leading_passes: int) -> int:
        """Map dealer-relative seat (display) to BT-canonical seat when there are leading passes."""
        try:
            s = int(seat_1_to_4)
        except Exception:
            s = 1
        k = int(leading_passes or 0) % 4
        return ((s - 1 - k) % 4) + 1

    # Display bid selection
    if not is_complete:
        # Seat cycles 1-4, Bid Num is always increasing
        seat_1_to_4 = ((current_seat - 1) % 4) + 1
        st.markdown(f"**Select Bid for Seat {seat_1_to_4}**")
        
        # Allow opening passes: for BT lookups, strip leading passes (seat-1 canonical).
        bt_prefix, leading_passes = _strip_leading_passes(current_auction)

        # Check cache first to avoid showing spinner when cached
        cache_key = (bt_prefix or "__opening__", seed)
        if cache_key in st.session_state.auction_builder_options:
            options = st.session_state.auction_builder_options[cache_key]
        else:
            with st.spinner("Loading available bids..."):
                options = get_next_bid_options(bt_prefix)
        
        # Show elapsed time for loading bids
        bids_elapsed_ms = st.session_state.get("_auction_builder_bids_elapsed_ms")
        if bids_elapsed_ms:
            st.info(f"⏱️ Loaded {len(options)} bids in {bids_elapsed_ms/1000:.2f}s")

        if not options:
            st.warning("No more bids available in BT for this auction prefix.")

        # Opening-pass support:
        # If we are still before the opening bid (canonical BT prefix is empty), always offer 'P' as an option.
        # Selecting P advances the seat; BT prefix remains "" until the first non-pass bid is chosen.
        if bt_prefix == "":
            has_p = any(str(o.get("bid") or "").strip().upper() == "P" for o in (options or []))
            if not has_p:
                options = list(options or [])
                options.append(
                    {
                        "bid": "P",
                        "bt_index": None,
                        "agg_expr": [],
                        "is_complete": _is_auction_complete_after_next_bid(current_auction, "P"),
                    }
                )

        # Permissive-pass support:
        # If the user wants Pass to always be a valid bid, ensure 'P' is offered even when the BT
        # does not explicitly include it at this node.
        if always_valid_pass:
            has_p = any(str(o.get("bid") or "").strip().upper() == "P" for o in (options or []))
            if not has_p:
                options = list(options or [])
                options.append(
                    {
                        "bid": "P",
                        "bt_index": None,
                        "expr": [],
                        "agg_expr": [],
                        "can_complete": True,
                        "is_complete": _is_auction_complete_after_next_bid(current_auction, "P"),
                        "is_dead_end": False,
                        "matching_deal_count": 0,
                        "avg_ev_nv": None,
                        "avg_ev_v": None,
                    }
                )

        if options:
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
            # Player-style Q&A (Q1..Q11) for the current auction context
            # -----------------------------------------------------------------
            # This is intentionally anchored to the *current auction decision point*:
            # - We format prompts to include deal_index (if pinned) + auction text.
            # - For evidence-backed answers, we use existing stable endpoints:
            #   - /bid-details (phase2a posteriors, par posterior)
            #   - /explain-bid (templated explanation + counterfactual)
            #   - /list-next-bids (next-bid options + match counts)
            import math
            try:
                qa_deal_index: int | None = None
                if pinned_deal and pinned_deal.get("index") is not None:
                    qa_deal_index = int(pinned_deal.get("index"))
            except Exception:
                qa_deal_index = None

            def _qa_context_str() -> str:
                auc = str(current_auction or "").strip()
                di = str(qa_deal_index) if qa_deal_index is not None else "None"
                return f"Context: deal_index={di}, auction='{auc if auc else '(opening)'}'"

            def _qa_hist_stats(hist: dict[str, int] | None) -> dict[str, float | None]:
                """Return mean + (p05,p95) from an integer histogram {value: count}."""
                if not isinstance(hist, dict) or not hist:
                    return {"mean": None, "p05": None, "p95": None}
                try:
                    items: list[tuple[int, int]] = []
                    tot = 0
                    s = 0.0
                    for k, v in hist.items():
                        kk = int(k)
                        vv = int(v)
                        if vv <= 0:
                            continue
                        items.append((kk, vv))
                        tot += vv
                        s += float(kk) * float(vv)
                    if tot <= 0 or not items:
                        return {"mean": None, "p05": None, "p95": None}
                    items.sort(key=lambda x: x[0])
                    mean = float(s / float(tot))

                    def _q(p: float) -> float:
                        target = p * float(tot)
                        c = 0.0
                        for kk, vv in items:
                            c += float(vv)
                            if c >= target:
                                return float(kk)
                        return float(items[-1][0])

                    return {"mean": mean, "p05": _q(0.05), "p95": _q(0.95)}
                except Exception:
                    return {"mean": None, "p05": None, "p95": None}

            # Candidate bids (for evidence-backed questions)
            qa_bids: list[str] = []
            try:
                seen = set()
                for o in (sorted_options or []):
                    b = str(o.get("bid") or "").strip().upper()
                    if not b or b in seen:
                        continue
                    seen.add(b)
                    qa_bids.append(b)
            except Exception:
                qa_bids = []

            # Load custom questions from CSV (fail-fast).
            try:
                qa_questions_all: list[CustomQuestion] = load_custom_questions_csv(CUSTOM_QUESTIONS_CSV)
                qa_questions: list[CustomQuestion] = [q for q in qa_questions_all if q.enabled]
                if not qa_questions:
                    raise ValueError("No enabled questions in CSV")
            except Exception as e:
                st.error(f"Custom questions CSV invalid: {e}")
                st.stop()

            # Reset selection/answer when auction context changes
            qa_ctx_key = "_auction_builder_qa_ctx"
            qa_prev_ctx = str(st.session_state.get(qa_ctx_key, "") or "")
            qa_curr_ctx = f"{qa_deal_index}:{str(current_auction or '').strip().upper()}"
            if qa_prev_ctx != qa_curr_ctx:
                st.session_state[qa_ctx_key] = qa_curr_ctx
                for k in ["auction_builder_qa_qid", "auction_builder_qa_bid", "auction_builder_qa_why_not", "auction_builder_qa_answer"]:
                    if k in st.session_state:
                        del st.session_state[k]

            with st.expander("💬 Q&A (players) — ask about current auction", expanded=False):
                st.caption(_qa_context_str())

                qa_title_by_id = {q.id: q.title for q in qa_questions}
                qid = st.selectbox(
                    "Question",
                    options=[q.id for q in qa_questions],
                    format_func=lambda x: str(qa_title_by_id.get(x, x)),
                    key="auction_builder_qa_qid",
                )

                qobj = next((qq for qq in qa_questions if qq.id == qid), None)
                if qobj is None:
                    st.error(f"Question not found: {qid}")
                    st.stop()

                # Evidence controls: choose a candidate next bid only when required.
                sel_bid = ""
                if qobj.requires_candidate_bid:
                    sel_bid = st.selectbox(
                        "Candidate next bid (used as evidence context where applicable)",
                        options=(qa_bids if qa_bids else [""]),
                        index=(0 if qa_bids else 0),
                        key="auction_builder_qa_bid",
                        help="Some questions use /bid-details or /explain-bid, which are defined for (auction prefix + candidate next bid).",
                    )
                why_not_bid = None
                if qid == "Q1":
                    why_not_bid = st.selectbox(
                        "Why not… (counterfactual bid)",
                        options=["(none)"] + (qa_bids if qa_bids else []),
                        index=0,
                        key="auction_builder_qa_why_not",
                    )

                qa_max_deals = int(
                    st.number_input(
                        "Max deals (evidence sample size)",
                        value=500,
                        min_value=10,
                        max_value=10_000,
                        help="Controls the matched-deal sample size used by /bid-details and /explain-bid.",
                        key="auction_builder_qa_max_deals",
                    )
                )

                # Render the formatted prompt (always), using the template from CSV.
                auction_disp = str(current_auction or "").strip().upper()
                tmpl_vars = {
                    "deal_index": str(qa_deal_index) if qa_deal_index is not None else "None",
                    "auction": auction_disp if auction_disp else "(opening)",
                    "candidate_bid": str(sel_bid or "").strip().upper() if sel_bid else "",
                    "why_not_bid": (str(why_not_bid).strip().upper() if why_not_bid and why_not_bid != "(none)" else ""),
                }
                prompt_lines: list[str] = [_qa_context_str(), "", f'Prompt: "{qobj.prompt_player}"', qobj.prompt_template.format_map(tmpl_vars)]

                st.markdown("**Formatted question**")
                st.code("\n".join(prompt_lines))

                def _qa_fetch_bid_details(auction_text: str, bid_text: str) -> dict[str, Any]:
                    payload: dict[str, Any] = {
                        "auction": str(auction_text or ""),
                        "bid": str(bid_text or ""),
                        "max_deals": int(qa_max_deals),
                        "seed": int(seed),
                        "vul_filter": None,
                        "topk": 10,
                        "include_phase2a": True,
                        "_cache_version": 1,
                    }
                    if qa_deal_index is not None:
                        payload["deal_index"] = int(qa_deal_index)
                    return api_post("/bid-details", payload, timeout=60)

                def _qa_fetch_explain_bid(auction_text: str, bid_text: str, why_not_text: str | None) -> dict[str, Any]:
                    payload2: dict[str, Any] = {
                        "auction": str(auction_text or ""),
                        "bid": str(bid_text or ""),
                        "why_not_bid": None if not why_not_text or why_not_text == "(none)" else str(why_not_text),
                        "max_deals": int(qa_max_deals),
                        "seed": int(seed),
                        "vul_filter": None,
                        "topk": 10,
                        "include_phase2a": True,
                        "_cache_version": 1,
                    }
                    if qa_deal_index is not None:
                        payload2["deal_index"] = int(qa_deal_index)
                    return api_post("/explain-bid", payload2, timeout=90)

                def _qa_answer_q12_top_next_bids() -> pl.DataFrame:
                    rows = []
                    for o in (sorted_options or []):
                        b = str(o.get("bid") or "").strip().upper()
                        if not b:
                            continue
                        mc = o.get("matching_deal_count")
                        try:
                            mc_i = int(mc) if mc is not None else 0
                        except Exception:
                            mc_i = 0
                        rows.append({"Bid": b, "MatchingDeals": mc_i})
                    df = pl.DataFrame(rows) if rows else pl.DataFrame({"Bid": [], "MatchingDeals": []})
                    if not df.is_empty():
                        df = df.sort("MatchingDeals", descending=True)
                        tot = int(df["MatchingDeals"].sum()) if "MatchingDeals" in df.columns else 0
                        if tot > 0:
                            df = df.with_columns((pl.col("MatchingDeals") / float(tot) * 100.0).round(2).alias("Prob_%"))
                    return df.head(5)

                def _qa_answer_q10_expected_completed_auctions() -> pl.DataFrame:
                    """Approximate top completed auctions by chaining next-bid mass.

                    Uses /list-next-bids matching_deal_count as a probability proxy. We expand a small beam of
                    partial continuations and then "complete" each partial by appending trailing passes.
                    """
                    start = str(current_auction or "").strip().upper()
                    # If already complete, show itself.
                    try:
                        toks0 = [t.strip().upper() for t in start.split("-") if t.strip()]
                        if toks0:
                            if len(toks0) >= 4 and all(t == "P" for t in toks0):
                                return pl.DataFrame([{"Auction": start, "Prob_%": 100.0, "Depth": 0}])
                            trailing = 0
                            for t in reversed(toks0):
                                if t == "P":
                                    trailing += 1
                                else:
                                    break
                            if trailing >= 3 and any(t != "P" for t in toks0):
                                return pl.DataFrame([{"Auction": start, "Prob_%": 100.0, "Depth": 0}])
                    except Exception:
                        pass

                    # Beam search settings: conservative to avoid a storm of API calls.
                    max_depth = 4
                    beam_width = 8

                    # log-sum-exp accumulator for duplicate completed auctions
                    completed_logp: dict[str, float] = {}

                    def _lse(a: float | None, b: float) -> float:
                        if a is None:
                            return float(b)
                        m = a if a > b else b
                        return float(m + math.log(math.exp(a - m) + math.exp(b - m)))

                    beam: list[tuple[str, float, int]] = [(start, 0.0, 0)]  # (auction_text, logp, depth)
                    for _ in range(max_depth):
                        new_beam: list[tuple[str, float, int]] = []
                        for auc, logp, depth in beam:
                            bt_pref, _lp = _strip_leading_passes(auc)
                            opts = get_next_bid_options(bt_pref)
                            rows2: list[tuple[str, int]] = []
                            mass = 0
                            for o in (opts or []):
                                bid = str(o.get("bid") or "").strip().upper()
                                if not bid:
                                    continue
                                try:
                                    mc = int(o.get("matching_deal_count") or 0)
                                except Exception:
                                    mc = 0
                                if mc <= 0:
                                    continue
                                mass += mc
                                rows2.append((bid, mc))
                            if mass <= 0 or not rows2:
                                continue
                            rows2.sort(key=lambda x: x[1], reverse=True)
                            rows2 = rows2[:beam_width]
                            for bid, mc in rows2:
                                p = float(mc) / float(mass)
                                if not (p > 0.0):
                                    continue
                                next_auc = f"{auc}-{bid}" if auc else bid
                                lp2 = float(logp + math.log(p))
                                # "Complete" by appending trailing passes (default completion convention).
                                try:
                                    completed_auc = _append_passes_to_complete(next_auc)
                                except Exception:
                                    completed_auc = next_auc
                                prev = completed_logp.get(completed_auc)
                                completed_logp[completed_auc] = _lse(prev, lp2)
                                new_beam.append((next_auc, lp2, depth + 1))
                        if not new_beam:
                            break
                        new_beam.sort(key=lambda x: x[1], reverse=True)
                        beam = new_beam[:beam_width]

                    if not completed_logp:
                        return pl.DataFrame({"Auction": [], "Prob_%": [], "Depth": []})

                    # Normalize among the completed set for display.
                    max_logp = max(completed_logp.values())
                    weights = {k: math.exp(v - max_logp) for k, v in completed_logp.items()}
                    z = float(sum(weights.values())) if weights else 0.0
                    out = []
                    for auc, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:5]:
                        out.append({"Auction": auc, "Prob_%": round((float(w) / z) * 100.0, 2) if z > 0 else None})
                    return pl.DataFrame(out)

                def _qa_answer_q11_surprises() -> pl.DataFrame:
                    """Approximate 'surprise' using next-bid matching_deal_count mass at each step."""
                    calls = [str(s.get("bid") or "").strip().upper() for s in (current_path or []) if str(s.get("bid") or "").strip()]
                    if not calls:
                        return pl.DataFrame({"Step": [], "AuctionPrefix": [], "Call": [], "ApproxP_%": [], "Surprise": []})
                    out_rows: list[dict[str, Any]] = []
                    prefix_parts: list[str] = []
                    for i, call in enumerate(calls):
                        prefix = "-".join(prefix_parts)
                        bt_pref, _lp = _strip_leading_passes(prefix)
                        opts = get_next_bid_options(bt_pref)
                        mass = 0
                        call_mass = None
                        for o in (opts or []):
                            try:
                                mc = int(o.get("matching_deal_count") or 0)
                            except Exception:
                                mc = 0
                            mass += mc
                            if str(o.get("bid") or "").strip().upper() == call:
                                call_mass = mc
                        p = (float(call_mass) / float(mass)) if (mass and call_mass is not None) else float("nan")
                        try:
                            surprise = float("inf") if not (p > 0.0) else float(-math.log(p))
                        except Exception:
                            surprise = float("inf")
                        out_rows.append(
                            {
                                "Step": int(i + 1),
                                "AuctionPrefix": prefix if prefix else "(opening)",
                                "Call": call,
                                "ApproxP_%": None if not (p == p) else round(p * 100.0, 2),  # NaN check via p==p
                                "Surprise": None if surprise == float("inf") else round(surprise, 3),
                            }
                        )
                        prefix_parts.append(call)
                    df = pl.DataFrame(out_rows) if out_rows else pl.DataFrame()
                    if not df.is_empty() and "Surprise" in df.columns:
                        df = df.sort("Surprise", descending=True, nulls_last=True)
                    return df.head(3)

                # Reactive answering: compute only when relevant inputs change.
                q4_decl = None
                q4_strain = None
                if qid == "Q4":
                    q4_decl = st.selectbox("Declarer", options=["N", "E", "S", "W"], index=2, key="auction_builder_qa_q4_decl")
                    q4_strain = st.selectbox("Strain", options=["NT", "S", "H", "D", "C"], index=0, key="auction_builder_qa_q4_strain")

                qa_compute_key = (
                    "qa-v1",
                    qa_curr_ctx,
                    str(qid),
                    str(sel_bid or ""),
                    str(why_not_bid or ""),
                    int(qa_max_deals),
                    int(seed),
                    str(q4_decl or ""),
                    str(q4_strain or ""),
                )
                last_key = st.session_state.get("_auction_builder_qa_last_key")
                if last_key != qa_compute_key:
                    st.session_state["_auction_builder_qa_last_key"] = qa_compute_key

                try:
                    st.markdown("**Answer**")
                    if qid in {"Q1", "Q2", "Q3", "Q6", "Q9"} and not sel_bid:
                        st.warning("Pick a candidate next bid to compute evidence-backed posteriors/explanations.")
                    elif qid == "Q1":
                        with st.spinner("Computing explanation..."):
                            expl = _qa_fetch_explain_bid(current_auction, sel_bid, why_not_bid)
                        ex = (expl.get("explanation") or {}) if isinstance(expl, dict) else {}
                        cf = (expl.get("counterfactual") or {}) if isinstance(expl, dict) else {}
                        st.caption(
                            "Note: any 'Top par contracts' mentioned are *double-dummy par* outcomes over the evidence set; "
                            "they are not constrained to be at/above the contract reached in the current auction."
                        )
                        st.write(ex.get("text") or "No explanation returned.")
                        if cf.get("text"):
                            st.markdown("**Counterfactual**")
                            st.write(cf.get("text"))
                    elif qid == "Q2":
                        with st.spinner("Computing HCP posterior..."):
                            details = _qa_fetch_bid_details(current_auction, sel_bid)
                        hists = details.get("hcp_histograms") or {}
                        e_hist = hists.get("E") if isinstance(hists, dict) else None
                        stats = _qa_hist_stats(e_hist if isinstance(e_hist, dict) else None)
                        st.markdown("**Auction-conditioned; absolute direction E**")
                        if stats["mean"] is None:
                            st.write("No HCP posterior available for E in this evidence set.")
                        else:
                            st.write(
                                f"East HCP estimate: mean **{float(stats['mean']):.1f}**, 90% interval **[{int(stats['p05'])}, {int(stats['p95'])}]**."
                            )
                    elif qid == "Q3":
                        with st.spinner("Computing par posterior..."):
                            details = _qa_fetch_bid_details(current_auction, sel_bid)
                        ps = details.get("par_score") or {}
                        pc = details.get("par_contracts") or {}
                        st.caption("Evidence set is the matched deals for the selected candidate next bid.")
                        st.caption(
                            "Important: these are *par contracts* (double-dummy). They can be lower/higher than the contract the auction would reach."
                        )
                        mean = ps.get("mean")
                        p10 = ps.get("p10")
                        p90 = ps.get("p90")
                        if mean is not None:
                            st.write(
                                f"ParScore: mean **{float(mean):.1f}** (P10={float(p10):.1f} / P90={float(p90):.1f} if available)."
                            )
                        topk = pc.get("topk") or []
                        if topk:
                            df = pl.DataFrame(topk).head(3)
                            if "prob" in df.columns:
                                df = df.with_columns((pl.col("prob") * 100).round(1).alias("Prob_%")).drop("prob")
                            render_aggrid(
                                df,
                                key=f"auction_builder_qa_q3_topk_{qa_curr_ctx}",
                                height=calc_grid_height(len(df), max_height=180),
                                table_name="auction_builder_qa_q3_topk",
                                fit_columns_to_view=True,
                                show_sql_expander=False,
                            )
                        else:
                            st.write("No par-contract posterior available.")
                    elif qid == "Q4":
                        st.markdown("**Actual DD (double-dummy) tricks — pinned deal**")
                        if not pinned_deal:
                            st.warning("Pin a deal index to answer Q4 using double-dummy tricks.")
                        else:
                            decl = str(q4_decl or "S")
                            strain = str(q4_strain or "NT")
                            s = "N" if strain == "NT" else strain
                            col = f"DD_{decl}_{s}"
                            v = pinned_deal.get(col)
                            if v is None:
                                st.write(f"No `{col}` available on pinned deal.")
                            else:
                                st.write(f"Double-dummy tricks for {decl} in {strain}: **{int(v)}**.")
                    elif qid == "Q5":
                        st.markdown("**Actual DD (double-dummy) tricks grid — pinned deal**")
                        if not pinned_deal:
                            st.warning("Pin a deal index to show the actual DD tricks grid (4×5).")
                        else:
                            rows = []
                            for d in ["N", "E", "S", "W"]:
                                r = {"Declarer": d}
                                for su in ["C", "D", "H", "S", "N"]:
                                    key = f"DD_{d}_{su}"
                                    label = "NT" if su == "N" else su
                                    r[label] = pinned_deal.get(key)
                                rows.append(r)
                            df = pl.DataFrame(rows)
                            render_aggrid(
                                df,
                                key=f"auction_builder_qa_q5_dd_grid_{qa_curr_ctx}",
                                height=calc_grid_height(len(df), max_height=220),
                                table_name="auction_builder_qa_q5_dd_grid",
                                fit_columns_to_view=True,
                                show_sql_expander=False,
                            )
                    elif qid == "Q6":
                        with st.spinner("Computing suit-length posteriors..."):
                            details = _qa_fetch_bid_details(current_auction, sel_bid)
                        sl = details.get("suit_length_histograms") or {}
                        st.markdown("**Auction-conditioned; absolute directions**")
                        if not isinstance(sl, dict) or not sl:
                            st.write("No suit-length posteriors available in this evidence set.")
                        else:
                            out_rows = []
                            for d in ["N", "E", "S", "W"]:
                                d_obj = sl.get(d) or {}
                                if not isinstance(d_obj, dict):
                                    continue
                                for su in ["S", "H", "D", "C"]:
                                    st_hist = d_obj.get(su)
                                    stats = _qa_hist_stats(st_hist if isinstance(st_hist, dict) else None)
                                    out_rows.append(
                                        {
                                            "Dir": d,
                                            "Suit": su,
                                            "Mean": None if stats["mean"] is None else round(float(stats["mean"]), 2),
                                            "P05": stats["p05"],
                                            "P95": stats["p95"],
                                        }
                                    )
                            df = pl.DataFrame(out_rows) if out_rows else pl.DataFrame()
                            if not df.is_empty():
                                render_aggrid(
                                    df,
                                    key=f"auction_builder_qa_q6_sl_{qa_curr_ctx}",
                                    height=calc_grid_height(len(df), max_height=320),
                                    table_name="auction_builder_qa_q6_sl",
                                    fit_columns_to_view=True,
                                    show_sql_expander=False,
                                )
                    elif qid == "Q7":
                        st.info("Q7 is not wired yet (needs lead engine / lead data).")
                    elif qid == "Q8":
                        st.info("Q8 is not wired yet (needs lead engine + evaluation across matched deals).")
                    elif qid == "Q9":
                        toks = [t.strip().upper() for t in str(current_auction or "").split("-") if t.strip()]
                        if len(toks) < 1:
                            st.info("Need at least one call in the auction to compare before vs after.")
                        else:
                            last_call = toks[-1]
                            before = "-".join(toks[:-1])
                            with st.spinner("Computing before/after posteriors..."):
                                d_before = _qa_fetch_bid_details(before, last_call)
                                d_after = _qa_fetch_bid_details(current_auction, sel_bid)
                            hb = d_before.get("hcp_histograms") or {}
                            ha = d_after.get("hcp_histograms") or {}
                            rows = []
                            for d in ["N", "E", "S", "W"]:
                                sb = _qa_hist_stats((hb.get(d) if isinstance(hb, dict) else None))
                                sa = _qa_hist_stats((ha.get(d) if isinstance(ha, dict) else None))
                                if sb["mean"] is None or sa["mean"] is None:
                                    continue
                                rows.append(
                                    {
                                        "Dir": d,
                                        "HCP_mean_before": round(float(sb["mean"]), 2),
                                        "HCP_mean_after": round(float(sa["mean"]), 2),
                                        "Δ": round(float(sa["mean"]) - float(sb["mean"]), 2),
                                    }
                                )
                            df = pl.DataFrame(rows) if rows else pl.DataFrame()
                            st.markdown("**Approx: HCP belief shift**")
                            st.caption(f"Before: `{before if before else '(opening)'}` + call `{last_call}`; After: `{current_auction}`.")
                            if df.is_empty():
                                st.write("Could not compute a stable before/after HCP shift for this context.")
                            else:
                                render_aggrid(
                                    df,
                                    key=f"auction_builder_qa_q9_shift_{qa_curr_ctx}",
                                    height=calc_grid_height(len(df), max_height=220),
                                    table_name="auction_builder_qa_q9_shift",
                                    fit_columns_to_view=True,
                                    show_sql_expander=False,
                                )
                    elif qid == "Q10":
                        st.markdown("**Approx: top expected completed auctions**")
                        st.caption("Heuristic: chains the most-supported next bids (by matching-deal mass) and then appends trailing passes to complete the auction.")
                        df = _qa_answer_q10_expected_completed_auctions()
                        if df.is_empty():
                            st.write("Could not estimate completed auctions from next-bid mass in this context.")
                        else:
                            render_aggrid(
                                df,
                                key=f"auction_builder_qa_q10_completed_{qa_curr_ctx}",
                                height=calc_grid_height(len(df), max_height=240),
                                table_name="auction_builder_qa_q10_completed",
                                fit_columns_to_view=True,
                                show_sql_expander=False,
                            )
                    elif qid == "Q11":
                        st.markdown("**Approx: most surprising calls so far**")
                        df = _qa_answer_q11_surprises()
                        if df.is_empty():
                            st.write("No calls yet.")
                        else:
                            render_aggrid(
                                df,
                                key=f"auction_builder_qa_q11_surprise_{qa_curr_ctx}",
                                height=calc_grid_height(len(df), max_height=220),
                                table_name="auction_builder_qa_q11_surprise",
                                fit_columns_to_view=True,
                                show_sql_expander=False,
                            )
                    elif qid == "Q12":
                        st.markdown("**Approx: top expected next bids (at this decision point)**")
                        df = _qa_answer_q12_top_next_bids()
                        if df.is_empty():
                            st.write("No next-bid mass available.")
                        else:
                            render_aggrid(
                                df,
                                key=f"auction_builder_qa_q12_next_bids_{qa_curr_ctx}",
                                height=calc_grid_height(len(df), max_height=220),
                                table_name="auction_builder_qa_q12_next_bids",
                                fit_columns_to_view=True,
                                show_sql_expander=False,
                            )
                    elif qid == "Q13":
                        st.markdown("**Estimated partnership tricks grid (heuristic model)**")
                        if not pinned_deal:
                            st.warning("Pin a deal index to show the estimated tricks grid.")
                        else:
                            with st.spinner("Computing estimated tricks grid..."):
                                # Fetch bid-details to get Phase2a posteriors
                                details = _qa_fetch_bid_details(current_auction, sel_bid)
                                p2a = details.get("phase2a") or {}
                                hcp_hists = details.get("hcp_histograms") or {}
                                sl_hists = details.get("suit_length_histograms") or {}
                                fit_us = (p2a.get("fit") or {}).get("us")
                                fit_them = (p2a.get("fit") or {}).get("them")

                                # Determine acting direction for fit assignment
                                _directions = ["N", "E", "S", "W"]
                                _dealer_q13 = str(pinned_deal.get("Dealer", "N")).upper()
                                _dealer_idx_q13 = _directions.index(_dealer_q13) if _dealer_q13 in _directions else 0
                                _acting_dir_q13 = _directions[(_dealer_idx_q13 + int(seat_1_to_4) - 1) % 4]
                                _acting_is_ns = _acting_dir_q13 in ("N", "S")

                                _opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}

                                est_rows: list[dict[str, Any]] = []
                                for d in _directions:
                                    hand = str(pinned_deal.get(f"Hand_{d}", "") or "").strip()
                                    if not hand:
                                        est_rows.append({"Declarer": d, "C": None, "D": None, "H": None, "S": None, "NT": None})
                                        continue

                                    partner_d = _opposite[d]
                                    partner_hcp = hcp_hists.get(partner_d) if isinstance(hcp_hists, dict) else None
                                    partner_sl = sl_hists.get(partner_d) if isinstance(sl_hists, dict) else None

                                    # Same side as acting seat → fit.us; opposite side → fit.them
                                    d_is_ns = d in ("N", "S")
                                    fit = fit_us if (d_is_ns == _acting_is_ns) else fit_them

                                    r: dict[str, Any] = {"Declarer": d}
                                    for strain_label, strain_arg in [("C", "C"), ("D", "D"), ("H", "H"), ("S", "S"), ("NT", "NT")]:
                                        _trick_res = estimate_partnership_tricks(
                                            self_hand=hand,
                                            partner_hcp_hist=partner_hcp,
                                            partner_sl_hists=partner_sl,
                                            fit_us_hists=fit,
                                            strain=strain_arg,
                                        )
                                        r[strain_label] = _trick_res.get("est_tricks")
                                    est_rows.append(r)

                            df = pl.DataFrame(est_rows)
                            render_aggrid(
                                df,
                                key=f"auction_builder_qa_q13_est_grid_{qa_curr_ctx}",
                                height=calc_grid_height(len(df), max_height=220),
                                table_name="auction_builder_qa_q13_est_grid",
                                fit_columns_to_view=True,
                                show_sql_expander=False,
                            )
                            st.caption(
                                f"Acting seat: **{_acting_dir_q13}**. "
                                f"Partner posteriors from Phase2a (bid-details for `{sel_bid}`). "
                                f"Model: quick tricks + length tricks + fit bonus."
                            )
                    else:
                        st.info("Question not recognized.")
                except Exception as e:
                    st.error(f"Could not answer: {e}")

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
                
                Uses server-side bitmap evaluation via /deal-criteria-eval-batch when the pinned
                deal has _row_idx (fetched by index). PBN deals cannot be evaluated until
                enrichment via mlBridgeAugmentationLib is implemented.
                """
                if not pinned_deal or not criteria_list:
                    return True, []
                
                # IMPORTANT: When the displayed auction begins with leading passes ("P-P-..."),
                # we strip those passes for BT lookup (seat-1 canonical). Criteria returned for
                # those BT nodes are relative to that canonical seat ordering, not the real deal's
                # dealer-relative seats. To evaluate correctly, rotate the dealer and map the seat.
                dealer_actual = str(pinned_deal.get("Dealer", "N")).upper()
                dealer = _rotate_dealer_by(dealer_actual, leading_passes)
                seat = _bt_seat_from_display_seat(seat, leading_passes)
                row_idx = pinned_deal.get("_row_idx")
                
                # PBN deals don't have _row_idx - cannot evaluate criteria without enrichment
                if row_idx is None:
                    return True, ["(PBN deal - criteria evaluation not available)"]
                
                # Use server-side bitmap evaluation (accurate for all criteria)
                try:
                    # Use session cache to avoid repeated API calls
                    cache_key = ("deal_criteria_eval", int(row_idx), dealer, seat, tuple(criteria_list))
                    eval_cache = st.session_state.setdefault("_deal_criteria_eval_cache", {})
                    if cache_key in eval_cache:
                        cached = eval_cache[cache_key]
                        return cached["passes"], cached["failed"]
                    
                    payload = {
                        "deal_row_idx": int(row_idx),
                        "dealer": dealer,
                        "checks": [{"seat": seat, "criteria": list(criteria_list)}],
                    }
                    data = api_post("/deal-criteria-eval-batch", payload, timeout=10)
                    results = data.get("results", [])
                    if results:
                        r = results[0]
                        failed_list = r.get("failed", [])
                        untracked = r.get("untracked", [])
                        # Annotate failed criteria with actual values
                        annotated_failed = []
                        for f in failed_list:
                            annotated = annotate_criterion_with_value(str(f), dealer, seat, pinned_deal)
                            annotated_failed.append(annotated)
                        # Include untracked criteria as warnings
                        for u in untracked:
                            annotated_failed.append(f"UNTRACKED: {u}")
                        passes = len(failed_list) == 0 and len(untracked) == 0
                        eval_cache[cache_key] = {"passes": passes, "failed": annotated_failed}
                        return passes, annotated_failed
                except Exception as e:
                    return True, [f"(Server error: {e})"]
                
                return True, []
            
            # Fetch Matches (pattern-based count) for all bid options
            # Build cache key for matches lookup
            matches_cache_key = f"_bid_matches_{current_auction}_{seed}"
            if matches_cache_key not in st.session_state:
                st.session_state[matches_cache_key] = {}
            bid_matches: dict[str, int] = st.session_state[matches_cache_key]
            
            # Find bids that need fetching
            missing_bids = []
            for opt in sorted_options:
                bid_str = str(opt.get("bid", "")).upper()
                next_auction = f"{current_auction}-{bid_str}" if current_auction else bid_str
                if next_auction not in bid_matches:
                    missing_bids.append((bid_str, next_auction))
            
            # Fetch missing matches (pattern-based counts)
            if missing_bids:
                missing_auctions = [next_auction for _, next_auction in missing_bids]
                _fetch_auction_pattern_counts(
                    missing_auctions, bid_matches, pattern_style="bid_option", timeout=15
                )
            
            # Build DataFrame for bid selection
            bid_rows = []
            for i, opt in enumerate(sorted_options):
                criteria_list = opt.get("agg_expr", [])
                criteria_str = "; ".join(criteria_list)
                complete_marker = " ✅" if opt.get("is_complete") else ""
                
                # Get Matches count for this bid
                bid_str = str(opt.get("bid", "")).upper()
                next_auction = f"{current_auction}-{bid_str}" if current_auction else bid_str
                matches_count = bid_matches.get(next_auction, 0)
                is_pass = bid_str.strip().upper() in ("P", "PASS")
                pass_always_valid = bool(is_pass and st.session_state.get("always_valid_pass", True))
                
                # Check if pinned deal matches this bid's criteria
                matches_pinned = None
                failed_criteria_str = ""
                if pinned_deal and show_failed_criteria:
                    matches, failed_list = check_pinned_match_with_failures(criteria_list, seat_1_to_4)
                    matches_pinned = matches
                    failed_criteria_str = "; ".join(failed_list) if failed_list else ""
                # If enabled, Pass is always treated as matching the pinned deal (even if criteria fails).
                if pinned_deal and show_failed_criteria and pass_always_valid:
                    matches_pinned = True
                    failed_criteria_str = ""
                
                # Compute direction from dealer + seat when deal is pinned
                seat_direction: str | None = None
                if pinned_deal:
                    dealer = pinned_deal.get("Dealer", "N")
                    directions = ["N", "E", "S", "W"]
                    try:
                        dealer_idx = directions.index(str(dealer).upper())
                    except ValueError:
                        dealer_idx = 0
                    seat_direction = directions[(dealer_idx + seat_1_to_4 - 1) % 4]
                
                # Check if this bid should be rejected
                deal_count = opt.get("matching_deal_count")
                is_dead_end = opt.get("is_dead_end", False)
                # Rule: any Pass with empty criteria is ALWAYS routed to Valid (never Rejected).
                pass_no_criteria = bool(is_pass and (not criteria_list))
                has_empty_criteria = (not criteria_list)
                # Reject bids with underspecified criteria (missing HCP and Total_Points)
                has_underspecified_criteria = _is_underspecified_criteria(criteria_list)
                can_complete = opt.get("can_complete")
                can_complete_b = bool(can_complete) if can_complete is not None else True
                has_zero_deals = deal_count == 0
                has_zero_matches = matches_count == 0
                # IMPORTANT:
                # - "Matches" (pattern-based) and "Deals" (criteria-based) can be 0 even for a valid bid.
                #   Do NOT use those as rejection/invalid reasons.
                # - Keep rejection for structural/data-quality issues only.
                is_rejected = is_dead_end or has_empty_criteria or has_underspecified_criteria
                if pass_no_criteria:
                    is_rejected = False
                if pass_always_valid:
                    is_rejected = False
                
                # Build failure reason for display
                failure_reasons = []
                if is_dead_end:
                    failure_reasons.append("dead end")
                if has_empty_criteria:
                    failure_reasons.append("missing criteria")
                if has_underspecified_criteria:
                    failure_reasons.append("underspecified (no HCP/Total_Points)")
                if (show_failed_criteria and matches_pinned is True and not can_complete_b):
                    # Bid matches current criteria but cannot reach a completed auction anywhere downstream.
                    if not (pass_no_criteria or pass_always_valid):
                        is_rejected = True
                        failure_reasons.append("cannot complete")
                failure_str = "; ".join(failure_reasons) if failure_reasons else ""
                
                row_data: dict[str, Any] = {
                    "_idx": i,
                    "_matches": matches_pinned,  # Hidden column for styling
                    "_rejected": is_rejected,  # Hidden column for categorization
                    "BT Index": opt.get("bt_index"),
                    "Bid Num": current_seat,
                    "Seat": seat_1_to_4,
                }
                # Add Direction column only when deal is pinned
                if seat_direction:
                    row_data["Direction"] = seat_direction
                row_data["Bid"] = f"{opt['bid']}{complete_marker}"
                row_data["can_complete"] = can_complete
                # Add Matches and Deals counts
                row_data["Matches"] = matches_count
                row_data["Deals"] = deal_count if deal_count is not None else ""
                # Add Avg_EV with NV/V split from GPU pipeline
                avg_ev_nv = opt.get("avg_ev_nv")
                avg_ev_v = opt.get("avg_ev_v")
                avg_par_nv = opt.get("avg_par_nv")
                avg_par_v = opt.get("avg_par_v")
                # IMPORTANT: Normalize EV sign by partnership (NS positive).
                # BT seat-1 view: Seat 1/3 = NS, Seat 2/4 = EW. For EW seats, flip sign so values
                # are comparable across seats/steps.
                ev_sign = -1.0 if seat_1_to_4 in (2, 4) else 1.0
                row_data["EV_NV"] = round(ev_sign * float(avg_ev_nv), 1) if avg_ev_nv is not None else None
                row_data["EV_V"] = round(ev_sign * float(avg_ev_v), 1) if avg_ev_v is not None else None
                # Avg_EV: when pinned deal is present, pick the correct vul for this bidder seat
                # Only show EV when there are matching deals (otherwise the precomputed EV is misleading)
                row_data["Avg_EV"] = None
                if pinned_deal and seat_direction and matches_count and matches_count > 0:
                    try:
                        vul = str(pinned_deal.get("Vul", pinned_deal.get("Vulnerability", ""))).upper()
                        ns_vul = vul in ("N_S", "NS", "BOTH", "ALL")
                        ew_vul = vul in ("E_W", "EW", "BOTH", "ALL")
                        is_vul = (seat_direction in ("N", "S") and ns_vul) or (seat_direction in ("E", "W") and ew_vul)
                        chosen = avg_ev_v if is_vul else avg_ev_nv
                        # Keep the same NS-positive convention as EV_NV/EV_V
                        row_data["Avg_EV"] = round(ev_sign * float(chosen), 1) if chosen is not None else None
                    except Exception:
                        row_data["Avg_EV"] = None

                # -----------------------------------------------------------------
                # Par is first-class (mirror EV behavior)
                # -----------------------------------------------------------------
                row_data["Par_NV"] = round(ev_sign * float(avg_par_nv), 1) if avg_par_nv is not None else None
                row_data["Par_V"] = round(ev_sign * float(avg_par_v), 1) if avg_par_v is not None else None
                row_data["Avg_Par"] = None
                if pinned_deal and seat_direction and matches_count and matches_count > 0:
                    try:
                        vul = str(pinned_deal.get("Vul", pinned_deal.get("Vulnerability", ""))).upper()
                        ns_vul = vul in ("N_S", "NS", "BOTH", "ALL")
                        ew_vul = vul in ("E_W", "EW", "BOTH", "ALL")
                        is_vul = (seat_direction in ("N", "S") and ns_vul) or (seat_direction in ("E", "W") and ew_vul)
                        chosen_par = avg_par_v if is_vul else avg_par_nv
                        row_data["Avg_Par"] = round(ev_sign * float(chosen_par), 1) if chosen_par is not None else None
                    except Exception:
                        row_data["Avg_Par"] = None

                # -----------------------------------------------------------------
                # Contract + DD (based on last contract bid, not current call)
                # -----------------------------------------------------------------
                # Use the auction *after* making this candidate call to determine the current contract.
                # This ensures passes/doubles/redoubles still show DD for the standing contract.
                contract_str: str | None = None
                dd_col_name: str | None = None
                dd_val: int | None = None
                dd_score_val: int | None = None
                ev_contract: float | None = None
                ev_contract_pre: float | None = None
                declarer_dir_for_contract: str | None = None
                try:
                    candidate_auction_norm = normalize_auction_input(next_auction).upper() if next_auction else ""
                    # For per-deal EV lookup, force the auction to a completed form (append passes).
                    # This makes P vs 6S comparable at the same decision point.
                    candidate_auction_completed = _append_passes_to_complete(candidate_auction_norm) if candidate_auction_norm else ""
                    c = parse_contract_from_auction(candidate_auction_norm)
                    if candidate_auction_norm and c is None:
                        # Passed out (all passes)
                        toks = [t.strip().upper() for t in candidate_auction_norm.split("-") if t.strip()]
                        if toks and all(t == "P" for t in toks):
                            contract_str = "Passed out"
                    if c is not None:
                        level, strain, _ = c
                        s = "NT" if str(strain).upper() == "N" else str(strain).upper()
                        contract_str = f"{int(level)}{s}"
                        if pinned_deal:
                            dealer_for_contract = str(pinned_deal.get("Dealer", "N")).upper()
                            declarer = get_declarer_for_auction(candidate_auction_norm, dealer_for_contract)
                            if declarer:
                                declarer_dir_for_contract = str(declarer).upper()
                                dd_col_name = f"DD_{declarer}_{str(strain).upper()}"
                                raw = pinned_deal.get(dd_col_name)
                                dd_val = int(raw) if raw is not None else None
                                # DD score: for X/XX, score via endplay to apply doubling rules.
                                dd_score_raw = get_dd_score_for_auction(candidate_auction_norm, dealer_for_contract, pinned_deal)
                                dd_score_val = int(dd_score_raw) if dd_score_raw is not None else None
                    # EV_Contract: pinned-deal EV for the completed candidate auction's contract.
                    if pinned_deal and candidate_auction_completed:
                        dealer_for_ev_contract = str(pinned_deal.get("Dealer", "N")).upper()
                        # Precomputed EV (legacy; ignores X/XX)
                        ev_pre_val = get_ev_for_auction_pre(candidate_auction_completed, dealer_for_ev_contract, pinned_deal)
                        ev_contract_pre = round(float(ev_pre_val), 1) if ev_pre_val is not None else None
                        # New EV (X/XX-aware)
                        ev_val = get_ev_for_auction(candidate_auction_completed, dealer_for_ev_contract, pinned_deal)
                        ev_contract = round(float(ev_val), 1) if ev_val is not None else None
                except Exception:
                    contract_str = contract_str
                    dd_col_name = dd_col_name
                    dd_val = dd_val
                    dd_score_val = dd_score_val
                    ev_contract = ev_contract
                    ev_contract_pre = ev_contract_pre

                row_data["Contract"] = contract_str or ""
                row_data["DD_Tricks"] = dd_val
                row_data["_dd_col"] = dd_col_name or ""
                # DD/EV "contract" scores in the UI should be shown as "score for OUR side".
                # `get_dd_score_for_auction` / `get_ev_for_auction` are declarer-relative:
                # - Declarer makes => positive
                # - Declarer sets  => negative
                # So to display "our side" (the current bidder's partnership), negate when the
                # declarer is on the opponents' side.
                if pinned_deal and seat_direction and declarer_dir_for_contract:
                    bidder_side = "NS" if str(seat_direction).upper() in ("N", "S") else "EW"
                    declarer_side = "NS" if str(declarer_dir_for_contract).upper() in ("N", "S") else "EW"
                    side_sign = 1.0 if bidder_side == declarer_side else -1.0
                    if dd_score_val is not None:
                        dd_score_val = int(side_sign * float(dd_score_val))
                    if ev_contract is not None:
                        ev_contract = round(side_sign * float(ev_contract), 1)
                    if ev_contract_pre is not None:
                        ev_contract_pre = round(side_sign * float(ev_contract_pre), 1)
                row_data["DD_Score_Contract"] = dd_score_val
                row_data["EV_Contract"] = ev_contract
                row_data["EV_Contract_Pre"] = ev_contract_pre

                # NOTE: Avg_EV is only available for completed auctions (~975K rows).
                # Non-completed auction nodes (e.g. opening bids like 1C) legitimately
                # have Avg_EV=None and should NOT be rejected for it.

                # Enforce: Pass with empty criteria is never rejected.
                if pass_no_criteria:
                    is_rejected = False
                    row_data["_rejected"] = False
                    failure_str = ""
                # Enforce: Pass is never rejected when permissive-pass is enabled.
                if pass_always_valid:
                    is_rejected = False
                    row_data["_rejected"] = False
                    failure_str = ""

                row_data["Criteria_Count"] = len(criteria_list) if isinstance(criteria_list, list) else 0
                row_data["Criteria"] = criteria_str if criteria_str else "(none)"
                row_data["Categories"] = ""
                # Attach categories (true flags) for this candidate's bt_index (if available)
                try:
                    bt_idx = opt.get("bt_index")
                    if bt_idx is not None:
                        row_data["Categories"] = bt_idx_to_categories_opt.get(int(bt_idx), "")
                except Exception:
                    pass
                if show_failed_criteria:
                    row_data["Failed_Criteria"] = failed_criteria_str
                if is_rejected:
                    row_data["Failure"] = failure_str
                bid_rows.append(row_data)

            # Suggested bids ranked by EV Contract, Matches, Bid
            suggested_bids_rows: list[dict[str, Any]] = []
            # Build the base "top 5 valid" list exactly as before.
            valid_bid_rows: list[dict[str, Any]] = []
            other_bid_rows: list[dict[str, Any]] = []
            for r in bid_rows:
                if (r.get("_rejected") is True) or (r.get("_matches") is False):
                    other_bid_rows.append(r)
                elif r.get("_matches") is True:
                    valid_bid_rows.append(r)

            def _bucket_for_row(rr: dict[str, Any]) -> str:
                if rr.get("_rejected"):
                    return "Rejected"
                if rr.get("_matches") is True:
                    return "Valid"
                if rr.get("_matches") is False:
                    return "Invalid"
                return "Unknown"

            def _to_suggest_row(rr: dict[str, Any]) -> dict[str, Any]:
                bucket = _bucket_for_row(rr)

                criteria_count = int(r.get("Criteria_Count", 0) or 0)
                # EV for sorting (EV Contract)
                ev_contract_val = rr.get("EV_Contract")
                dd_score_contract_val = rr.get("DD_Score_Contract")

                # Any leading Pass (auction prefix is all P's) should have EV=0.0.
                # This covers opening sequences like "", "P", "P-P", "P-P-P", etc.
                try:
                    toks_now = [t.strip().upper() for t in str(current_auction or "").split("-") if t.strip()]
                    prefix_all_passes = (len(toks_now) == 0) or all(t == "P" for t in toks_now)
                except Exception:
                    prefix_all_passes = False
                bid0 = str(rr.get("Bid", "")).strip().split(" ")[0].upper()
                if prefix_all_passes and bid0 == "P":
                    ev_contract_val = 0.0
                    # For opening/lead-pass sequences, Pass is effectively "no-op" and should not
                    # get pushed below negative DD-score actions just because DD is the first sort key.
                    # Treat it as DD=0 for ranking (and display), matching the intent of EV=0 above.
                    dd_score_contract_val = 0

                # Display missing EV Contract as 0.0 for this table
                if ev_contract_val is None:
                    ev_contract_val = 0.0
                ev_contract_sort = float(ev_contract_val)
                # DD Score sort: missing sinks to bottom
                dd_sort = dd_score_contract_val
                try:
                    dd_sort_f = float(dd_sort) if dd_sort is not None else float("-inf")
                except Exception:
                    dd_sort_f = float("-inf")
                matches_raw = rr.get("Matches")
                try:
                    matches_i = int(matches_raw) if matches_raw is not None else 0
                except Exception:
                    matches_i = 0
                bid_key = str(rr.get("Bid", "")).strip().upper()

                return {
                    "_idx": rr.get("_idx"),
                    "Bid": rr.get("Bid"),
                    "Criteria": criteria_count,
                    "EV Contract": ev_contract_val,
                    "DD Score": dd_score_contract_val,
                    "Avg_EV": rr.get("Avg_EV"),
                    "Deals": rr.get("Deals"),
                    "Matches": rr.get("Matches"),
                    "Bucket": bucket,
                    # Sort: DD Score desc, EV Contract desc, Matches desc, Bid asc
                    "_sort": (-dd_sort_f, -ev_contract_sort, -float(matches_i), bid_key),
                }

            # Primary rows: top 5 valid (unchanged behavior).
            for r in valid_bid_rows:
                suggested_bids_rows.append(_to_suggest_row(r))

            suggested_bids_rows.sort(key=lambda x: x.get("_sort", (float("inf"), float("inf"), 0.0, "")))
            suggested_bids_rows = suggested_bids_rows[:5]

            # Extra rows requested:
            # Include any rejected/invalid bids where:
            #   - can_complete == True (treat None as True)
            #   - (DD Score >= DD Score of the current top row) OR (Avg_EV >= Avg_EV of the current top row)
            top_avg_ev: float | None = None
            top_dd: float | None = None
            if suggested_bids_rows:
                try:
                    v0 = suggested_bids_rows[0].get("Avg_EV")
                    top_avg_ev = float(v0) if v0 is not None else None
                except Exception:
                    top_avg_ev = None
                try:
                    d0 = suggested_bids_rows[0].get("DD Score")
                    top_dd = float(d0) if d0 is not None else None
                except Exception:
                    top_dd = None

            if top_avg_ev is not None or top_dd is not None:
                existing_idx = {row.get("_idx") for row in suggested_bids_rows if row.get("_idx") is not None}
                extras: list[dict[str, Any]] = []
                for r in other_bid_rows:
                    can_complete = r.get("can_complete")
                    can_complete_b = bool(can_complete) if can_complete is not None else True
                    if not can_complete_b:
                        continue
                    v = r.get("Avg_EV")
                    d = r.get("DD_Score_Contract")
                    try:
                        v_f = float(v) if v is not None else None
                    except Exception:
                        v_f = None
                    try:
                        d_f = float(d) if d is not None else None
                    except Exception:
                        d_f = None

                    meets = False
                    if top_avg_ev is not None and v_f is not None and v_f >= float(top_avg_ev):
                        meets = True
                    if top_dd is not None and d_f is not None and d_f >= float(top_dd):
                        meets = True
                    if not meets:
                        continue
                    rr = _to_suggest_row(r)
                    idx = rr.get("_idx")
                    if idx is not None and idx in existing_idx:
                        continue
                    extras.append(rr)
                    if idx is not None:
                        existing_idx.add(idx)
                # Keep extras stable and sensible: show higher Avg_EV first.
                try:
                    extras.sort(key=lambda x: float(x.get("Avg_EV") or float("-inf")), reverse=True)
                except Exception:
                    pass
                suggested_bids_rows.extend(extras)

            bids_df = pl.DataFrame(bid_rows)
            
            # Track last selected to detect new clicks
            if "auction_builder_last_selected" not in st.session_state:
                st.session_state.auction_builder_last_selected = None
            
            # Helper to build grid options for bid selection
            def build_bid_grid_options(pdf: "pd.DataFrame", apply_row_styling: bool = True) -> dict:
                gb = GridOptionsBuilder.from_dataframe(pdf)
                gb.configure_selection(selection_mode="single", use_checkbox=False)
                gb.configure_column("_idx", hide=True)
                gb.configure_column("_matches", hide=True)
                gb.configure_column("_rejected", hide=True)
                if "Criteria_Count" in pdf.columns:
                    gb.configure_column("Criteria_Count", hide=True)
                if "_dd_col" in pdf.columns:
                    gb.configure_column("_dd_col", hide=True)
                # Column widths: ~8px/char + ~55px for filter icon and padding
                gb.configure_column("BT Index", width=120, minWidth=110)   # 8 chars
                gb.configure_column("Bid Num", width=115, minWidth=105)    # 7 chars
                gb.configure_column("Seat", width=90, minWidth=80)         # 4 chars
                if pinned_deal:
                    gb.configure_column("Direction", width=125, minWidth=115)  # 9 chars
                gb.configure_column("Bid", width=80, minWidth=70)          # 3 chars
                if "Contract" in pdf.columns:
                    gb.configure_column("Contract", width=110, minWidth=95)
                if "can_complete" in pdf.columns:
                    gb.configure_column("can_complete", width=130, minWidth=120, headerName="Can Complete")
                if "DD_Tricks" in pdf.columns:
                    gb.configure_column("DD_Tricks", width=135, minWidth=120, headerName="DD_Tricks")
                if "DD_Score_Contract" in pdf.columns:
                    gb.configure_column("DD_Score_Contract", width=135, minWidth=120, headerName="DD Score")
                gb.configure_column("Matches", width=115, minWidth=105)    # 7 chars
                gb.configure_column("Deals", width=100, minWidth=90)       # 5 chars
                gb.configure_column("Avg_EV", width=105, minWidth=95, headerName="Avg EV")   # 6 chars
                if "Avg_Par" in pdf.columns:
                    gb.configure_column("Avg_Par", width=110, minWidth=100, headerName="Avg Par")
                if "EV_Contract" in pdf.columns:
                    gb.configure_column("EV_Contract", width=120, minWidth=110, headerName="EV Contract")
                if "EV_Contract_Pre" in pdf.columns:
                    gb.configure_column("EV_Contract_Pre", width=140, minWidth=125, headerName="EV Contract Pre")
                gb.configure_column("EV_NV", width=100, minWidth=90, headerName="EV NV")    # 5 chars
                gb.configure_column("EV_V", width=90, minWidth=80, headerName="EV V")       # 4 chars
                if "Par_NV" in pdf.columns:
                    gb.configure_column("Par_NV", width=100, minWidth=90, headerName="Par NV")
                if "Par_V" in pdf.columns:
                    gb.configure_column("Par_V", width=90, minWidth=80, headerName="Par V")
                gb.configure_column(
                    "Criteria",
                    flex=1,
                    minWidth=100,
                    tooltipField="Criteria",
                )
                gb.configure_column(
                    "Categories",
                    flex=1,
                    minWidth=140,
                    tooltipField="Categories",
                )
                if show_failed_criteria and "Failed_Criteria" in pdf.columns:
                    gb.configure_column("Failed_Criteria", flex=1, minWidth=120, headerName="Failed Criteria", tooltipField="Failed_Criteria")
                # Hide Failure in valid/invalid grids (shown only in rejected grid)
                if "Failure" in pdf.columns:
                    gb.configure_column("Failure", hide=True)

                # Display rules:
                # - Pinned deal: show Avg_EV/Avg_Par, hide EV_NV/EV_V and Par_NV/Par_V
                # - No pinned deal: hide Avg_EV/Avg_Par, show EV_NV/EV_V and Par_NV/Par_V
                if pinned_deal:
                    gb.configure_column("EV_NV", hide=True)
                    gb.configure_column("EV_V", hide=True)
                    if "Par_NV" in pdf.columns:
                        gb.configure_column("Par_NV", hide=True)
                    if "Par_V" in pdf.columns:
                        gb.configure_column("Par_V", hide=True)
                else:
                    gb.configure_column("Avg_EV", hide=True)
                    if "Avg_Par" in pdf.columns:
                        gb.configure_column("Avg_Par", hide=True)
                    if "EV_Contract" in pdf.columns:
                        gb.configure_column("EV_Contract", hide=True)
                    if "EV_Contract_Pre" in pdf.columns:
                        gb.configure_column("EV_Contract_Pre", hide=True)
                
                # Add row styling based on pinned deal match
                if apply_row_styling and show_failed_criteria:
                    gb.configure_grid_options(
                        getRowClass=JsCode("""
                            function(params) {
                                if (params.data._matches === true) {
                                    return 'row-match-pass';
                                } else if (params.data._matches === false) {
                                    return 'row-match-fail';
                                }
                                return null;
                            }
                        """)
                    )
                gb.configure_grid_options(tooltipShowDelay=300, rowHeight=25, headerHeight=25)
                return gb.build()
            
            # Custom CSS for row styling
            grid_custom_css = {
                ".ag-row": {"cursor": "pointer", "transition": "background-color 0.1s"},
                ".ag-row-hover": {
                    "background-color": "#E8F4FF !important",
                    "border-left": "3px solid #007BFF !important",
                },
                ".row-match-pass": {"background-color": "#d4edda !important"},
                ".row-match-pass.ag-row-hover": {
                    "background-color": "#b8dfc4 !important",
                    "border-left": "3px solid #28a745 !important",
                },
                ".row-match-pass.ag-row-selected": {"background-color": "#a3d4af !important"},
                ".row-match-fail": {"background-color": "#f8d7da !important"},
                ".row-match-fail.ag-row-hover": {
                    "background-color": "#f1b0b7 !important",
                    "border-left": "3px solid #dc3545 !important",
                },
                ".row-match-fail.ag-row-selected": {"background-color": "#eb959f !important"},
            }
            
            # Helper to handle bid selection from any grid.
            #
            # IMPORTANT: AgGrid returns `selected_rows` on *every* Streamlit rerun (even when the rerun
            # was triggered by unrelated widgets like checkboxes). If we auto-apply whenever
            # `selected_rows` is non-empty, toggling a checkbox can inadvertently "re-apply" a persisted
            # selection and change the auction state, making other sections appear/disappear.
            #
            # Fix: track the last selected `_idx` per grid, and only apply when it changes.
            def handle_bid_selection(grid_resp: Any, grid_name: str) -> None:
                sel_rows = grid_resp.get("selected_rows")
                sel_row = None
                if sel_rows is not None and len(sel_rows) > 0:
                    sel_row = sel_rows[0] if isinstance(sel_rows, list) else sel_rows.iloc[0].to_dict()
                try:
                    sel_idx = int(sel_row.get("_idx")) if sel_row is not None else None
                except Exception:
                    sel_idx = None

                init_key = f"_ab_sel_init_{grid_name}_{current_seat}"
                last_key = f"_ab_sel_last_{grid_name}_{current_seat}"
                if not st.session_state.get(init_key, False):
                    st.session_state[init_key] = True
                    st.session_state[last_key] = sel_idx
                    return

                if st.session_state.get(last_key) == sel_idx:
                    return
                st.session_state[last_key] = sel_idx

                if sel_idx is None:
                    return

                sel_opt = dict(sorted_options[sel_idx])
                sel_bid = sel_opt.get("bid")
                sel_bid_s = str(sel_bid or "").strip().upper()

                selection_key = f"{current_auction}-{sel_bid_s}"
                if selection_key != st.session_state.auction_builder_last_selected:
                    st.session_state.auction_builder_last_selected = selection_key
                    computed_complete = _is_auction_complete_after_next_bid(current_auction, sel_bid_s)
                    sel_opt["is_complete"] = bool(sel_opt.get("is_complete", False) or computed_complete)
                    sel_opt["bid"] = sel_bid_s
                    # Preserve client-side rule: Pass with empty criteria is valid.
                    sel_opt["_pass_no_criteria"] = bool(sel_bid_s == "P" and not (sel_opt.get("agg_expr") or []))
                    # Mark Pass selected from valid-bids grid as valid for Summary consistency.
                    # This handles cases where Pass is the only valid option but its criteria may
                    # fail re-evaluation with different dealer/seat context in the Summary.
                    if sel_bid_s == "P" and grid_name == "suggested":
                        sel_opt["_pass_valid_selection"] = True
                    # Attach categories from the lookup cache
                    bt_idx = sel_opt.get("bt_index")
                    if bt_idx is not None:
                        cats_str = bt_idx_to_categories_opt.get(int(bt_idx), "")
                        if cats_str:
                            sel_opt["categories"] = [c.strip() for c in cats_str.split(",") if c.strip()]
                    st.session_state.auction_builder_path.append(sel_opt)
                    # Normalize the entire stored path immediately using /resolve-auction-path.
                    # This prevents later "rehydration" from changing earlier rows (green->red)
                    # because the path mixes per-selection criteria with server-resolved criteria.
                    try:
                        new_auction = "-".join([step.get("bid", "") for step in st.session_state.auction_builder_path if step.get("bid")])
                        if new_auction:
                            data = api_post("/resolve-auction-path", {"auction": new_auction}, timeout=30)
                            new_path = data.get("path", []) or []
                            if new_path and len(new_path) == len(st.session_state.auction_builder_path):
                                # Preserve client-side flags that shouldn't be invalidated by server rehydrate.
                                try:
                                    old_path = st.session_state.auction_builder_path
                                    for j in range(min(len(old_path), len(new_path))):
                                        if old_path[j].get("_pass_no_criteria"):
                                            new_path[j]["_pass_no_criteria"] = True
                                        if old_path[j].get("_pass_valid_selection"):
                                            new_path[j]["_pass_valid_selection"] = True
                                except Exception:
                                    pass
                                # Preserve any client-side completion inference for trailing passes
                                try:
                                    if st.session_state.auction_builder_path and new_path:
                                        new_path[-1]["is_complete"] = bool(
                                            new_path[-1].get("is_complete", False)
                                            or st.session_state.auction_builder_path[-1].get("is_complete", False)
                                        )
                                except Exception:
                                    pass
                                st.session_state.auction_builder_path = new_path
                    except Exception:
                        # Best-effort only; keep the appended selection as-is.
                        pass
                    st.session_state.auction_builder_last_applied = ""
                    st.rerun()
            
            pandas_df = bids_df.to_pandas()
            
            # Count valid/invalid/rejected bids
            valid_count = sum(1 for r in bid_rows if r.get("_matches") is True and not r.get("_rejected"))
            invalid_count = sum(1 for r in bid_rows if r.get("_matches") is False and not r.get("_rejected"))
            rejected_count = sum(1 for r in bid_rows if r.get("_rejected"))
            
            if use_two_dataframes and show_failed_criteria:
                # Split into valid, rejected, and invalid DataFrames
                # Rejected bids are separated first (dead ends, 0 deals, empty criteria)
                valid_rows = [r for r in bid_rows if r.get("_matches") is True and not r.get("_rejected")]
                rejected_rows = [r for r in bid_rows if r.get("_rejected")]
                invalid_rows = [r for r in bid_rows if r.get("_matches") is False and not r.get("_rejected")]
                other_rows = [r for r in bid_rows if r.get("_matches") is None and not r.get("_rejected")]

                # Build Best Par Bids and Best EV Bids tables (5 columns each)
                # These show the top bids ranked by Par and EV for the pinned deal
                all_bids_for_ranking = valid_rows + rejected_rows + invalid_rows
                
                # Determine bucket for each bid
                def get_bucket(r: dict[str, Any]) -> str:
                    if r.get("_matches") is True and not r.get("_rejected"):
                        return "Valid"
                    elif r.get("_rejected"):
                        return "Rejected"
                    else:
                        return "Invalid"
                
                # Best Par Bids: sort by DD_Score_Contract (descending, best score first)
                best_par_bids: list[dict[str, Any]] = []
                par_sorted = sorted(
                    [r for r in all_bids_for_ranking if r.get("DD_Score_Contract") is not None],
                    key=lambda x: float(x.get("DD_Score_Contract") or -9999),
                    reverse=True
                )[:5]  # Top 5
                for r in par_sorted:
                    dd_val = r.get("DD_Score_Contract")
                    best_par_bids.append({
                        "_idx": r.get("_idx"),
                        "Bid": r.get("Bid", ""),
                        "Contract": r.get("Contract", ""),
                        "DD Score": int(dd_val) if dd_val is not None else None,
                        "Deals": r.get("Deals", ""),
                        "Matches": r.get("Matches", ""),
                        "Bucket": get_bucket(r),
                    })
                
                # Best EV Bids: sort by EV_Contract (descending, best EV first)
                best_ev_bids: list[dict[str, Any]] = []
                ev_sorted = sorted(
                    [r for r in all_bids_for_ranking if r.get("EV_Contract") is not None],
                    key=lambda x: float(x.get("EV_Contract") or -9999),
                    reverse=True
                )[:5]  # Top 5
                for r in ev_sorted:
                    best_ev_bids.append({
                        "_idx": r.get("_idx"),
                        "Bid": r.get("Bid", ""),
                        "Contract": r.get("Contract", ""),
                        "EV Score": r.get("EV_Contract"),
                        "Deals": r.get("Deals", ""),
                        "Matches": r.get("Matches", ""),
                        "Bucket": get_bucket(r),
                    })
                
                best_par_bids_df = pl.DataFrame(best_par_bids) if best_par_bids else pl.DataFrame()
                best_ev_bids_df = pl.DataFrame(best_ev_bids) if best_ev_bids else pl.DataFrame()

                # Shared CSS and row styling for all 3 tables
                bid_table_css = {
                    ".ag-row": {"cursor": "pointer"},
                    ".row-suggest-valid": {"background-color": "#d4edda !important"},
                    ".row-suggest-rejected": {"background-color": "#fff3cd !important"},
                    ".row-suggest-invalid": {"background-color": "#f8d7da !important"},
                }
                row_class_js = JsCode(
                    """
                    function(params) {
                        if (params.data.Bucket === 'Valid') {
                            return 'row-suggest-valid';
                        }
                        if (params.data.Bucket === 'Rejected') {
                            return 'row-suggest-rejected';
                        }
                        if (params.data.Bucket === 'Invalid') {
                            return 'row-suggest-invalid';
                        }
                        return null;
                    }
                    """
                )

                # -------------------------------------------------------------------
                # Matching BT Auctions (by Par)
                # -------------------------------------------------------------------
                par_contracts_raw = pinned_deal.get("ParContracts") if pinned_deal else None
                if par_contracts_raw:
                    # NOTE: Do not wrap AgGrid in an expander here (incompatible in this app).
                    st.subheader("🎯 Matching BT Auctions (by Par)")
                    prefix_display = f" starting with **{current_auction}**" if current_auction else ""
                    st.caption(f"Completed BT auctions{prefix_display} resulting in the pinned deal's ParContracts.")

                    deal_idx = pinned_deal.get("index", 0) if pinned_deal else 0
                    # Include current_auction in cache key so results update when auction changes
                    par_cache_key = f"auction_builder_par_matches_{deal_idx}_{seed}_{current_auction}"
                    par_elapsed_key = f"{par_cache_key}_elapsed"
                    if par_cache_key not in st.session_state:
                        with st.spinner("Searching BT..."):
                            try:
                                par_resp = api_post(
                                    "/find-bt-auctions-by-contracts",
                                    {
                                        "par_contracts": par_contracts_raw,
                                        "dealer": pinned_deal.get("Dealer", "N"),
                                        "auction_prefix": current_auction,
                                        # Fast-path for existing deals: row index in bbo_mldf_augmented.parquet
                                        "deal_row_idx": pinned_deal.get("_row_idx"),
                                    },
                                    timeout=30
                                )
                                st.session_state[par_cache_key] = par_resp.get("auctions", [])
                                st.session_state[par_elapsed_key] = par_resp.get("elapsed_ms", 0)
                            except Exception as e:
                                st.error(f"Error fetching par matches: {e}")
                                st.session_state[par_cache_key] = []
                                st.session_state[par_elapsed_key] = 0

                    par_matches = st.session_state[par_cache_key]
                    par_elapsed = st.session_state.get(par_elapsed_key, 0)
                    if par_matches:
                        # Enrich server-provided rows with deal-specific scores and UI-consistent stats.
                        # Use a cache key so enrichment only runs once per deal/auction combination.
                        enrich_cache_key = f"_par_matches_enriched_{deal_idx}_{seed}_{current_auction}"
                        if enrich_cache_key not in st.session_state:
                          with st.spinner("Enriching par-matching auctions..."):
                            dealer_actual = str(pinned_deal.get("Dealer", "N")).upper() if pinned_deal else "N"

                            # Precompute vul flags (used for Avg_EV from NV/V split)
                            vul = str(pinned_deal.get("Vul", pinned_deal.get("Vulnerability", ""))).upper() if pinned_deal else ""
                            ns_vul = vul in ("N_S", "NS", "BOTH", "ALL")
                            ew_vul = vul in ("E_W", "EW", "BOTH", "ALL")

                            enriched_rows: list[dict[str, Any]] = []
                            auc_list: list[str] = []
                            for r in par_matches:
                                rr = dict(r or {})
                                auc = str(rr.get("Auction") or "").strip()
                                if not auc:
                                    continue
                                auc_list.append(auc)

                                # Deal-specific scoring for the pinned deal
                                dd_score: int | None = None
                                ev_score: float | None = None
                                try:
                                    dd_v = get_dd_score_for_auction(auc, dealer_actual, pinned_deal)
                                    dd_score = int(dd_v) if dd_v is not None else None
                                except Exception:
                                    dd_score = None
                                try:
                                    ev_v = get_ev_for_auction(auc, dealer_actual, pinned_deal)
                                    ev_score = round(float(ev_v), 1) if ev_v is not None else None
                                except Exception:
                                    ev_score = None
                                rr["DD Score"] = dd_score
                                rr["EV Score"] = ev_score

                                # Avg_EV: use server-provided NV/V split (from bt_ev_stats_df) when available.
                                avg_ev_nv = rr.get("avg_ev_nv")
                                avg_ev_v = rr.get("avg_ev_v")
                                pair = str(rr.get("Pair") or "").upper()
                                rr["Avg_EV"] = None
                                if avg_ev_nv is not None or avg_ev_v is not None:
                                    try:
                                        is_vul = (pair == "NS" and ns_vul) or (pair == "EW" and ew_vul)
                                        chosen = avg_ev_v if is_vul else avg_ev_nv
                                        if chosen is not None:
                                            # Normalize sign to NS-positive convention.
                                            sign = 1.0 if pair == "NS" else (-1.0 if pair == "EW" else 1.0)
                                            rr["Avg_EV"] = round(sign * float(chosen), 1)
                                    except Exception:
                                        rr["Avg_EV"] = None

                                enriched_rows.append(rr)

                            # Matches: exact auction counts (batch), same endpoint used by bid options.
                            counts_cache_key = f"_par_matches_counts_{deal_idx}_{seed}_{current_auction}"
                            if counts_cache_key not in st.session_state:
                                st.session_state[counts_cache_key] = {}
                            par_counts: dict[str, int] = st.session_state[counts_cache_key]
                            _fetch_auction_pattern_counts(
                                auc_list, par_counts, pattern_style="completed_only", timeout=15, max_patterns=500
                            )

                            for rr in enriched_rows:
                                a = str(rr.get("Auction") or "").strip()
                                rr["Matches"] = int(par_counts.get(a, 0) or 0)

                            par_matches_df = pl.DataFrame(enriched_rows)
                            # UI request: color ALL rows green.
                            # Use render_aggrid's pass/fail row styling with a constant True marker.
                            par_matches_df = par_matches_df.with_columns(pl.lit(True).alias("_passes"))
                            st.session_state[enrich_cache_key] = par_matches_df

                        par_matches_df = st.session_state[enrich_cache_key]
                        st.info(f"Found {len(par_matches)} matching auctions in {format_elapsed(float(par_elapsed))}.")

                        selected_rows = render_aggrid(
                            par_matches_df,
                            key=f"auction_builder_par_matches_grid_{deal_idx}_{current_auction}",
                            height=calc_grid_height(len(par_matches_df), max_height=300),
                            table_name="auction_builder_par_matches",
                            update_on=["selectionChanged"],
                            row_pass_fail_col="_passes",
                        )

                        if selected_rows:
                            selected_auc = str((selected_rows[0] or {}).get("Auction", "")).strip()
                            if selected_auc:
                                # Clicking a row should immediately move the auction into the
                                # "Bidding Sequence" input and apply via existing resolve/apply logic.
                                last_sel_key = f"auction_builder_par_matches_last_selected_{deal_idx}"
                                if st.session_state.get(last_sel_key) != selected_auc:
                                    st.session_state[last_sel_key] = selected_auc
                                    st.session_state["_auction_builder_pending_set_auction"] = selected_auc
                                    st.rerun()
                    else:
                        no_match_msg = f"No matching auctions found starting with '{current_auction}'." if current_auction else "No matching auctions found in the BT for these ParContracts."
                        st.info(no_match_msg)

                # -------------------------------------------------------------------
                # Advanced Best Next Bids (Phase 1 + Phase 2a evidence)
                # -------------------------------------------------------------------
                st.subheader("🏅 Best Next Bids (Advanced)")
                st.caption(
                    "This table ranks candidate next bids using evidence from the selected bid-details schema "
                    "(Par posterior utility + ambiguity + tail risk) and, when a deal is pinned, Phase 2a "
                    "auction-conditioned posteriors (how typical your pinned hand is for that bid + opponent-fit threat). "
                    "Click a row to apply that bid to the auction."
                )

                try:
                    deal_idx_key = int(pinned_deal.get("index", 0) or 0) if pinned_deal else 0
                except Exception:
                    deal_idx_key = 0

                adv_controls = st.columns([1.0, 1.0, 1.2, 1.2], gap="small")
                with adv_controls[0]:
                    adv_top_n = st.slider(
                        "Analyze top N valid bids",
                        min_value=5,
                        max_value=30,
                        value=12,
                        step=1,
                        key=f"ab_adv_topn_{deal_idx_key}_{current_auction}",
                    )
                with adv_controls[1]:
                    adv_max_deals = st.selectbox(
                        "Sample deals per bid",
                        options=[200, 500, 1000],
                        index=1,
                        key=f"ab_adv_maxdeals_{deal_idx_key}_{current_auction}",
                        help="Controls runtime. Larger samples stabilize posteriors but cost more.",
                    )
                with adv_controls[2]:
                    w_desc = st.slider(
                        "Pinned typicality bonus (pts)",
                        min_value=0,
                        max_value=400,
                        value=150,
                        step=10,
                        key=f"ab_adv_wdesc_{deal_idx_key}_{current_auction}",
                        help="Adds points when the pinned hand is typical for that bid's Phase2a SELF posterior.",
                    )
                with adv_controls[3]:
                    w_threat = st.slider(
                        "Opponent-fit threat penalty (pts)",
                        min_value=0,
                        max_value=400,
                        value=120,
                        step=10,
                        key=f"ab_adv_wthreat_{deal_idx_key}_{current_auction}",
                        help="Subtracts points when Phase2a suggests opponents often have big fits (max over suits).",
                    )

                # Guardrail controls (two rows)
                with st.expander("Guardrail settings (overbid / underbid / sacrifice)", expanded=False):
                    guard_row1 = st.columns([1.0, 1.0, 1.0, 1.0, 1.0], gap="small")
                    with guard_row1[0]:
                        w_guard = st.slider(
                            "Master weight",
                            min_value=0.0,
                            max_value=3.0,
                            value=1.0,
                            step=0.1,
                            key=f"ab_adv_wguard_{deal_idx_key}_{current_auction}",
                            help="Multiplier for all guardrail penalties (0 = off, 1 = normal, >1 = stricter).",
                        )
                    with guard_row1[1]:
                        w_guard_overbid = st.slider(
                            "Overbid / level",
                            min_value=0,
                            max_value=200,
                            value=80,
                            step=10,
                            key=f"ab_adv_wguard_overbid_{deal_idx_key}_{current_auction}",
                            help="OVERBID_VS_PAR: points deducted per level the bid exceeds par.",
                        )
                    with guard_row1[2]:
                        w_guard_tp = st.slider(
                            "TP shortfall / pt",
                            min_value=0,
                            max_value=30,
                            value=15,
                            step=1,
                            key=f"ab_adv_wguard_tp_{deal_idx_key}_{current_auction}",
                            help="TP_SHORTFALL: points deducted per total-point shortfall below the standard minimum.",
                        )
                    with guard_row1[3]:
                        w_guard_neg = st.slider(
                            "Neg-par amplifier",
                            min_value=0.0,
                            max_value=2.0,
                            value=0.5,
                            step=0.1,
                            key=f"ab_adv_wguard_neg_{deal_idx_key}_{current_auction}",
                            help="NEG_PAR_HIGH_LEVEL: extra penalty multiplier for negative acting-par at 4+ level.",
                        )
                    with guard_row1[4]:
                        w_guard_strain = st.slider(
                            "Strain ineff.",
                            min_value=0,
                            max_value=200,
                            value=60,
                            step=10,
                            key=f"ab_adv_wguard_strain_{deal_idx_key}_{current_auction}",
                            help="STRAIN_INEFFICIENCY: flat penalty for bidding 5m when par says 3NT (2 extra tricks for same score).",
                        )
                    guard_row2 = st.columns([1.0, 1.0, 1.0, 1.0, 1.0], gap="small")
                    with guard_row2[0]:
                        w_guard_underbid = st.slider(
                            "Underbid / level",
                            min_value=0,
                            max_value=200,
                            value=40,
                            step=10,
                            key=f"ab_adv_wguard_underbid_{deal_idx_key}_{current_auction}",
                            help="UNDERBID_VS_PAR: points deducted per level the bid is below par *owned by our side*.",
                        )
                    with guard_row2[1]:
                        w_guard_tp_surplus = st.slider(
                            "TP surplus / pt",
                            min_value=0,
                            max_value=20,
                            value=8,
                            step=1,
                            key=f"ab_adv_wguard_tp_surplus_{deal_idx_key}_{current_auction}",
                            help="TP_SURPLUS_UNDERBID: points deducted per TP above the threshold for a higher achievable level.",
                        )
                    with guard_row2[2]:
                        w_guard_sacrifice = st.slider(
                            "Sacrifice discount %",
                            min_value=0,
                            max_value=100,
                            value=70,
                            step=5,
                            key=f"ab_adv_wguard_sacrifice_{deal_idx_key}_{current_auction}",
                            help="Fraction by which overbid penalties are reduced when par belongs to opponents "
                            "(sacrifice context). 70% = most of the overbid penalty is forgiven.",
                        )
                    with guard_row2[3]:
                        w_guard_tricks = st.slider(
                            "Tricks shortfall / trick",
                            min_value=0,
                            max_value=80,
                            value=30,
                            step=5,
                            key=f"ab_adv_wguard_tricks_{deal_idx_key}_{current_auction}",
                            help="TRICKS_SHORTFALL: points deducted per estimated trick below the number required for the contract level.",
                        )

                @st.cache_data(ttl=3600, show_spinner=False)
                def _ab_fetch_bid_details(
                    auction: str,
                    bid: str,
                    *,
                    max_deals: int,
                    seed: int,
                    vul_filter: str | None,
                    deal_index: int | None,
                    include_phase2a: bool,
                    topk: int,
                    cache_version: int,
                ) -> Dict[str, Any]:
                    return api_post(
                        "/bid-details",
                        {
                            "auction": auction,
                            "bid": bid,
                            "max_deals": int(max_deals),
                            "seed": int(seed),
                            "vul_filter": vul_filter,
                            "deal_index": deal_index,
                            "include_phase2a": bool(include_phase2a),
                            "topk": int(topk),
                            "_ab_adv_cache_version": int(cache_version),
                        },
                        timeout=30,
                    )

                # Seed candidates from the already-built bid rows (cheap).
                # Prefer "Valid" bids when pinned-match evaluation is available; otherwise fall back to
                # any non-rejected BT-backed bids.
                def _seed_key(r: dict[str, Any]) -> tuple[float, float, float]:
                    def _f(x: Any) -> float:
                        try:
                            if x is None or x == "":
                                return float("-inf")
                            return float(x)
                        except Exception:
                            return float("-inf")

                    return (_f(r.get("Avg_EV")), _f(r.get("Deals")), _f(r.get("Matches")))

                AB_ADV_CACHE_VERSION = 3  # bump to invalidate Streamlit-side cached bid-details

                # 1) Prefer bids that match the pinned deal ("Valid" bucket).
                candidate_rows = list(valid_rows or [])
                # 2) If none are "Valid" (common at opening / early seat before criteria-eval is ready),
                # fall back to any bid row that is not rejected and has a BT Index.
                if not candidate_rows:
                    candidate_rows = [
                        rr
                        for rr in (bid_rows or [])
                        if not bool(rr.get("_rejected"))
                    ]
                candidate_rows = sorted(candidate_rows, key=_seed_key, reverse=True)
                # Keep a permissive-pass candidate for fallback display (Pass is not BT-backed, so we
                # cannot call /bid-details for it, but it is still a legitimate next bid in Auction Builder).
                pass_rows = [
                    rr
                    for rr in (candidate_rows or [])
                    if str(rr.get("Bid", "") or "").strip().upper().split(" ")[0] in {"P", "PASS"}
                ]
                # Advanced ranking requires BT-backed evidence. Exclude Pass + any row without BT Index.
                candidate_rows = [
                    rr
                    for rr in candidate_rows
                    if str(rr.get("Bid", "") or "").strip().upper().split(" ")[0] not in {"P", "PASS"}
                    and (rr.get("BT Index") is not None)
                ][: int(adv_top_n)]

                pinned_deal_index_val: int | None = None
                if pinned_deal and pinned_deal.get("index") is not None:
                    try:
                        pinned_deal_index_val = int(pinned_deal.get("index"))
                    except Exception:
                        pinned_deal_index_val = None

                vul_filter_val: str | None = None
                if pinned_deal:
                    try:
                        v = str(pinned_deal.get("Vul", pinned_deal.get("Vulnerability", "")) or "").strip()
                        vul_filter_val = v if v else None
                    except Exception:
                        vul_filter_val = None

                advanced_rows: list[dict[str, Any]] = []
                include_phase2a_adv = bool(pinned_deal_index_val is not None)
                
                # Perspective/sign: bid-details ParScore is NS-positive. For bidding recommendations we want
                # scores from the perspective of the *acting seat* at this auction prefix.
                acting_dir_adv: str | None = None
                acting_sign_adv: float = 1.0  # +1 for NS, -1 for EW (default NS if unknown)
                try:
                    if pinned_deal:
                        dealer_actual = str(pinned_deal.get("Dealer", "N") or "N").strip().upper()
                    else:
                        dealer_actual = "N"
                    seat_1_to_4_adv = _next_display_seat_from_auction(current_auction)
                    directions = ["N", "E", "S", "W"]
                    dealer_idx = directions.index(dealer_actual) if dealer_actual in directions else 0
                    acting_dir_adv = directions[(dealer_idx + int(seat_1_to_4_adv) - 1) % 4]
                    acting_sign_adv = 1.0 if acting_dir_adv in ("N", "S") else -1.0
                except Exception:
                    acting_dir_adv = None
                    acting_sign_adv = 1.0
                
                # Helper to determine bucket from original row
                def get_bucket_from_row(r: dict[str, Any]) -> str:
                    if r.get("_matches") is True and not r.get("_rejected"):
                        return "Valid"
                    elif r.get("_rejected"):
                        return "Rejected"
                    else:
                        return "Invalid"

                with st.spinner("Computing advanced ranking (bid-details + Phase2a)..."):
                    for r in candidate_rows:
                        bid0 = str(r.get("Bid", "") or "").strip().upper().split(" ")[0]
                        if not bid0:
                            continue
                        try:
                            details = _ab_fetch_bid_details(
                                current_auction,
                                bid0,
                                max_deals=int(adv_max_deals),
                                seed=int(seed),
                                vul_filter=vul_filter_val,
                                deal_index=pinned_deal_index_val,
                                include_phase2a=include_phase2a_adv,
                                topk=10,
                                cache_version=AB_ADV_CACHE_VERSION,
                            )
                        except Exception as e:
                            advanced_rows.append(
                                {
                                    "Bid": bid0,
                                    "Error": str(e),
                                    "Score": None,
                                    "Guard": None,
                                    "GuardWhy": None,
                                    "U": None,
                                    "ParMean_NS": None,
                                    "ParMean_Act": None,
                                    "P>=0%": None,
                                    "Ent": None,
                                    "Tail": None,
                                    "Desc": None,
                                    "Threat6+%": None,
                                    "EstTricks": None,
                                    "QLosers": None,
                                    "SelfQT": None,
                                    "SelfLTC": None,
                                    "SelfTP": None,
                                    "PtnrTP": None,
                                    "TopPar": None,
                                    "TopPar%": None,
                                    "n": r.get("Deals", None) or r.get("Matches", None),
                                    "Degraded": "api_error",
                                    "Source": get_bucket_from_row(r),
                                    "Categories": r.get("Categories", ""),
                                }
                            )
                            continue

                        # /bid-details returns HTTP 200 for some "errors" (schema-stable), so handle that explicitly.
                        if details.get("error") or details.get("message"):
                            advanced_rows.append(
                                {
                                    "Bid": bid0,
                                    "Error": str(details.get("error") or details.get("message")),
                                    "BTI": details.get("child_bt_index"),
                                    "Flags": "",
                                    "Score": None,
                                    "Guard": None,
                                    "GuardWhy": None,
                                    "U": None,
                                    "ParMean_NS": None,
                                    "ParMean_Act": None,
                                    "P>=0%": None,
                                    "Ent": None,
                                    "Tail": None,
                                    "Desc": None,
                                    "Threat6+%": None,
                                    "EstTricks": None,
                                    "QLosers": None,
                                    "SelfQT": None,
                                    "SelfLTC": None,
                                    "SelfTP": None,
                                    "PtnrTP": None,
                                    "TopPar": None,
                                    "TopPar%": None,
                                    "n": details.get("matched_deals_total", r.get("Deals", None) or r.get("Matches", None)),
                                    "Degraded": str(details.get("degraded_mode") or "bid_details_error"),
                                    "Source": get_bucket_from_row(r),
                                    "Categories": r.get("Categories", ""),
                                }
                            )
                            continue

                        par_score = details.get("par_score") or {}
                        par_contracts = details.get("par_contracts") or {}

                        mean_par = par_score.get("mean")
                        p_ok = par_score.get("p_par_ge_0")
                        tail = par_score.get("tail_risk_mean_neg")
                        entropy = par_contracts.get("contract_entropy_full_nats")

                        # "Utility" for ranking: use the same MVP form as compute_eeo_from_bid_details,
                        # but apply the acting-seat sign to the MEAN term only (penalties always subtract).
                        utility = None
                        try:
                            lam = 0.5
                            mu = 0.25
                            m = float(mean_par) if mean_par is not None else None
                            t = float(tail) if isinstance(tail, (int, float)) else 0.0
                            e = float(entropy) if isinstance(entropy, (int, float)) else 0.0
                            if m is not None:
                                utility = float(acting_sign_adv) * float(m) - lam * float(t) - mu * float(e)
                        except Exception:
                            utility = None

                        # Surface "BT-level flags" (e.g. Forcing_*) even if they don't change the posterior.
                        flags_s = ""
                        try:
                            crit = details.get("bt_acting_criteria") or []
                            if isinstance(crit, list):
                                flags = [str(x) for x in crit if isinstance(x, str) and x.strip().startswith("Forcing_")]
                                if flags:
                                    flags_s = ", ".join(flags[:6])
                        except Exception:
                            flags_s = ""

                        # Phase2a threat summaries
                        opp_threat = None
                        fit_us_best = None
                        try:
                            threat = (details.get("phase2a") or {}).get("threat") or {}
                            vals = []
                            fit_vals = []
                            for su in ["S", "H", "D", "C"]:
                                t_su = threat.get(su) or {}
                                v = t_su.get("p_them_6plus")
                                if isinstance(v, (int, float)):
                                    vals.append(float(v))
                                fv = t_su.get("e_fit_us")
                                if isinstance(fv, (int, float)):
                                    fit_vals.append(float(fv))
                            if vals:
                                opp_threat = max(vals)
                            if fit_vals:
                                fit_us_best = max(fit_vals)
                        except Exception:
                            opp_threat = None
                            fit_us_best = None

                        # Pinned-hand typicality vs Phase2a SELF posterior percentiles (0..1, higher is better)
                        desc_score = None
                        hcp_pct = None
                        try:
                            rp = details.get("range_percentiles")
                            if isinstance(rp, dict) and rp.get("role") == "self":
                                pcts = []
                                h = (rp.get("hcp") or {}).get("percentile")
                                if isinstance(h, (int, float)):
                                    hcp_pct = float(h)
                                    pcts.append(float(h))
                                sl = rp.get("suit_lengths") or {}
                                for su in ["S", "H", "D", "C"]:
                                    p = ((sl.get(su) or {}).get("percentile"))
                                    if isinstance(p, (int, float)):
                                        pcts.append(float(p))
                                if pcts:
                                    dev = sum(abs(p - 50.0) for p in pcts) / len(pcts)
                                    desc_score = max(0.0, min(1.0, 1.0 - (dev / 50.0)))
                        except Exception:
                            desc_score = None
                            hcp_pct = None

                        # Top par contract for quick interpretation
                        top_contract = None
                        top_prob = None
                        try:
                            topk = (par_contracts.get("topk") or [])
                            if topk:
                                top_contract = (topk[0] or {}).get("contract")
                                top_prob = (topk[0] or {}).get("prob")
                        except Exception:
                            top_contract = None
                            top_prob = None

                        # -- Guardrail inputs --
                        # Self total points (from range_percentiles, available when pinned deal)
                        self_tp_val = None
                        try:
                            rp2 = details.get("range_percentiles")
                            if isinstance(rp2, dict):
                                self_tp_val = (rp2.get("total_points") or {}).get("value")
                                if self_tp_val is not None:
                                    self_tp_val = float(self_tp_val)
                        except Exception:
                            self_tp_val = None

                        # Partner expected TP (from phase2a roles histogram)
                        partner_tp_hist_val = None
                        try:
                            p2a = details.get("phase2a") or {}
                            partner_tp_hist_val = ((p2a.get("roles") or {}).get("partner") or {}).get("total_points_hist")
                        except Exception:
                            partner_tp_hist_val = None

                        # Par contracts topk for guardrail comparison
                        par_topk_for_guard = None
                        try:
                            par_topk_for_guard = (par_contracts.get("topk") or [])
                        except Exception:
                            par_topk_for_guard = None

                        # -- Trick / Loser estimation --
                        est_tricks_val: float | None = None
                        est_ql_val: float | None = None
                        est_self_qt_val: float | None = None
                        est_self_ltc_val: int | None = None
                        try:
                            # Extract bid strain
                            import re as _re_strain
                            _strain_m = _re_strain.match(r"^[1-7]\s*(NT|N|[CDHS])", bid0)
                            _bid_strain: str | None = None
                            if _strain_m:
                                _bs = _strain_m.group(1).upper()
                                _bid_strain = "NT" if _bs in ("N", "NT") else _bs

                            # Get self hand string (acting player)
                            _self_hand_str: str | None = None
                            if pinned_deal and acting_dir_adv:
                                _self_hand_str = str(pinned_deal.get(f"Hand_{acting_dir_adv}", "") or "").strip() or None

                            # Hand controls for INSUFFICIENT_FIRST_ROUND_CONTROLS guardrail
                            _self_aces_adv: int | None = None
                            _self_hvoids_adv: int | None = None
                            if _self_hand_str:
                                _self_aces_adv, _self_hvoids_adv = hand_controls(_self_hand_str)

                            if _bid_strain and _self_hand_str:
                                # Partner posteriors for trick estimation
                                _p2a = details.get("phase2a") or {}
                                _partner_hcp_h = ((_p2a.get("roles") or {}).get("partner") or {}).get("hcp_hist")
                                _partner_sl_h = ((_p2a.get("roles") or {}).get("partner") or {}).get("sl_hist")
                                _fit_us_h = (_p2a.get("fit") or {}).get("us")

                                _trick_res = estimate_partnership_tricks(
                                    self_hand=_self_hand_str,
                                    partner_hcp_hist=_partner_hcp_h,
                                    partner_sl_hists=_partner_sl_h,
                                    fit_us_hists=_fit_us_h,
                                    strain=_bid_strain,
                                )
                                est_tricks_val = _trick_res.get("est_tricks")
                                est_self_qt_val = _trick_res.get("self_qt")
                                est_self_ltc_val = _trick_res.get("self_ltc")

                                _ql_res = estimate_quick_losers(
                                    self_hand=_self_hand_str,
                                    partner_sl_hists=_partner_sl_h,
                                    fit_us_hists=_fit_us_h,
                                    strain=_bid_strain,
                                )
                                est_ql_val = _ql_res.get("quick_losers")
                        except Exception:
                            est_tricks_val = None
                            est_ql_val = None
                            est_self_qt_val = None
                            est_self_ltc_val = None

                        # Compute guardrail penalty
                        guard_penalty = 0.0
                        guard_reasons: list[str] = []
                        non_rebiddable_rebid_penalty = 0.0
                        rebiddable_major_game_bonus = 0.0
                        partner_major_game_bonus = 0.0
                        partner_major_detour_penalty = 0.0
                        post_game_slam_gate_penalty = 0.0
                        try:
                            if float(w_guard) > 0:
                                gp, gr = compute_guardrail_penalty(
                                    bid=bid0,
                                    par_contracts_topk=par_topk_for_guard,
                                    self_total_points=self_tp_val,
                                    partner_tp_hist=partner_tp_hist_val,
                                    par_score_mean=mean_par,
                                    acting_sign=float(acting_sign_adv),
                                    w_overbid_level=float(w_guard_overbid),
                                    w_tp_shortfall=float(w_guard_tp),
                                    w_neg_par_high=float(w_guard_neg),
                                    w_underbid_level=float(w_guard_underbid),
                                    w_tp_surplus=float(w_guard_tp_surplus),
                                    w_strain_ineff=float(w_guard_strain),
                                    sacrifice_discount=float(w_guard_sacrifice) / 100.0,
                                    est_tricks=est_tricks_val,
                                    w_tricks_shortfall=float(w_guard_tricks),
                                    self_aces=_self_aces_adv,
                                    self_helpful_voids=_self_hvoids_adv,
                                    debug_equivalence_bypass=True,
                                )
                                guard_penalty = float(gp) * float(w_guard)
                                guard_reasons = gr
                        except Exception:
                            guard_penalty = 0.0
                            guard_reasons = []

                        # Additional bridge-texture guardrail for suit rebids.
                        try:
                            if float(w_guard) > 0:
                                _nr_p, _nr_reason = compute_non_rebiddable_suit_rebid_penalty(
                                    bid_text=bid0,
                                    auction_tokens=[t.strip().upper() for t in str(current_auction or "").split("-") if t.strip()],
                                    acting_direction=acting_dir_adv,
                                    dealer_actual=dealer_actual if "dealer_actual" in locals() else "N",
                                    bt_acting_criteria=crit if isinstance(crit, list) else [],
                                )
                                non_rebiddable_rebid_penalty = float(_nr_p) * float(w_guard)
                                if _nr_reason:
                                    guard_reasons = list(guard_reasons or [])
                                    guard_reasons.append(_nr_reason)
                        except Exception:
                            non_rebiddable_rebid_penalty = 0.0

                        # Bonus for directly committing to game in a strongly
                        # rebiddable major already shown by acting player.
                        try:
                            _rb_bonus, _rb_reason = compute_rebiddable_major_game_bonus(
                                bid_text=bid0,
                                auction_tokens=[t.strip().upper() for t in str(current_auction or "").split("-") if t.strip()],
                                acting_direction=acting_dir_adv,
                                dealer_actual=dealer_actual if "dealer_actual" in locals() else "N",
                                bt_acting_criteria=crit if isinstance(crit, list) else [],
                            )
                            rebiddable_major_game_bonus = float(_rb_bonus)
                            if _rb_reason:
                                guard_reasons = list(guard_reasons or [])
                                guard_reasons.append(_rb_reason)
                        except Exception:
                            rebiddable_major_game_bonus = 0.0

                        # Partner-major context adjustment (shared with API scorer).
                        try:
                            _pm_bonus, _pm_pen, _pm_reason = compute_partner_major_game_commit_adjustment(
                                bid_text=bid0,
                                auction_tokens=[t.strip().upper() for t in str(current_auction or "").split("-") if t.strip()],
                                acting_direction=acting_dir_adv,
                                dealer_actual=dealer_actual if "dealer_actual" in locals() else "N",
                                bt_acting_criteria=crit if isinstance(crit, list) else [],
                            )
                            partner_major_game_bonus = float(_pm_bonus)
                            partner_major_detour_penalty = float(_pm_pen)
                            if _pm_reason:
                                guard_reasons = list(guard_reasons or [])
                                guard_reasons.append(_pm_reason)
                        except Exception:
                            partner_major_game_bonus = 0.0
                            partner_major_detour_penalty = 0.0

                        # Shared post-game slam gate (align with API scorer).
                        try:
                            _slam_adj = compute_post_game_slam_gate_adjustment(
                                bid_text=bid0,
                                auction_tokens=[t.strip().upper() for t in str(current_auction or "").split("-") if t.strip()],
                                acting_direction=acting_dir_adv,
                                dealer_actual=dealer_actual if "dealer_actual" in locals() else "N",
                                self_total_points=self_tp_val,
                                partner_tp_hist=partner_tp_hist_val,
                            )
                            post_game_slam_gate_penalty = float(_slam_adj.get("penalty") or 0.0)
                            _slam_reason = _slam_adj.get("reason")
                            if _slam_reason:
                                guard_reasons = list(guard_reasons or [])
                                guard_reasons.append(str(_slam_reason))
                        except Exception:
                            post_game_slam_gate_penalty = 0.0

                        # Composite Score (transparent & adjustable)
                        # MUST match AI Model server-side _score_one to be "one source of truth".
                        score = None
                        matched_n_val = None
                        try:
                            # Reconstruct utility from perspective of acting side.
                            # "utility" from eeo is NS-relative (Mean_NS - Penalties).
                            # We want Acting-relative score.
                            # Base: Acting_Sign * Mean_Par (NS).
                            base = float(acting_sign_adv) * float(mean_par)

                            # Bayesian Shrinkage (matches AI Model _score_one)
                            # N < 5 is penalized by shrinking score towards 0.
                            matched_n_val = float(details.get("matched_deals_total_excluding_pinned") or details.get("matched_deals_total") or 0.0)
                            shrinkage_k = 5.0
                            base_shrunk = (base * matched_n_val) / (matched_n_val + shrinkage_k)

                            # Add description/threat terms (always additive to quality)
                            desc_term = float(desc_score - 0.5) if desc_score is not None else 0.0
                            threat_term = float(opp_threat) if opp_threat is not None else 0.0
                            score = base_shrunk + float(w_desc) * desc_term - float(w_threat) * threat_term
                            # Subtract guardrail penalty
                            score -= guard_penalty
                            score -= non_rebiddable_rebid_penalty
                            score += rebiddable_major_game_bonus
                            score += partner_major_game_bonus
                            score -= partner_major_detour_penalty
                            score -= post_game_slam_gate_penalty
                        except Exception:
                            score = None

                        advanced_rows.append(
                            {
                                "Bid": bid0,
                                "Error": "",
                                "BTI": details.get("child_bt_index"),
                                "Flags": flags_s,
                                # Show the bid-details vulnerability filter in effect (explains mean shifts).
                                "VulFilter": vul_filter_val,
                                "Score": score,
                                "Guard": round(guard_penalty, 0) if guard_penalty > 0 else None,
                                "GuardWhy": "; ".join(guard_reasons) if guard_reasons else None,
                                "U": utility,
                                # Make ParScore mean explicit (users expect to see this).
                                # - ParMean_NS: raw par_score.mean from /bid-details (NS-positive).
                                # - ParMean_Act: acting-seat perspective (positive = good for bidder).
                                "ParMean_NS": float(mean_par) if isinstance(mean_par, (int, float)) else mean_par,
                                "ParMean_Act": (float(acting_sign_adv) * float(mean_par)) if isinstance(mean_par, (int, float)) else mean_par,
                                "P>=0%": (float(p_ok) * 100.0) if isinstance(p_ok, (int, float)) else None,
                                "Ent": entropy,
                                "Tail": tail,
                                "Desc": desc_score,
                                "HCP%ile": hcp_pct,
                                "Threat6+%": (float(opp_threat) * 100.0) if isinstance(opp_threat, (int, float)) else None,
                                "FitUs": fit_us_best,
                                "EstTricks": est_tricks_val,
                                "QLosers": est_ql_val,
                                "SelfQT": est_self_qt_val,
                                "SelfLTC": est_self_ltc_val,
                                "SelfTP": self_tp_val,
                                "PtnrTP": round(float(expected_from_hist(partner_tp_hist_val)), 1) if expected_from_hist(partner_tp_hist_val) is not None else None,
                                "TopPar": top_contract,
                                "TopPar%": (float(top_prob) * 100.0) if isinstance(top_prob, (int, float)) else None,
                                # Show sample sizes used by /bid-details.
                                "n": details.get("matched_deals_total_excluding_pinned", details.get("matched_deals_total")),
                                "Sampled": details.get("matched_deals_sampled"),
                                "Degraded": details.get("degraded_mode"),
                                "Source": get_bucket_from_row(r),
                                "Categories": r.get("Categories", ""),
                            }
                        )

                adv_df = pl.DataFrame(advanced_rows) if advanced_rows else pl.DataFrame()
                # Persist Advanced table rows for debug dump (read by ai_model_debug.json writer).
                try:
                    st.session_state["_ab_adv_rows_debug"] = list(advanced_rows)
                except Exception:
                    pass
                # Always offer a synthetic Pass row (selectable) when permissive-pass is enabled.
                # Pass is not BT-backed, so we cannot call /bid-details for it; we approximate a Score
                # using the best proxy available in the already-built bid row.
                try:
                    include_synth_pass = bool(st.session_state.get("always_valid_pass", True))
                except Exception:
                    include_synth_pass = True

                if include_synth_pass:
                    # Prefer the best Pass row (if present) by the same seed ordering used elsewhere.
                    pr = None
                    try:
                        if pass_rows:
                            pr = sorted(pass_rows, key=_seed_key, reverse=True)[0]
                    except Exception:
                        pr = pass_rows[0] if pass_rows else None

                    def _pick_pass_score_and_note(
                        row: dict[str, Any] | None,
                        a_sign: float,
                    ) -> tuple[float, str]:
                        """Score Pass using BT-aggregate avg_ev (matches AI Model _score_one).

                        IMPORTANT: Non-Pass bids use BT-aggregate mean_par from /bid-details.
                        Pass must use the same BT-aggregate scale (avg_ev from /list-next-bids)
                        for fair comparison.  Deal-specific EV is NOT used because it creates
                        an apples-vs-oranges comparison that favours stopping prematurely.
                        """
                        if isinstance(row, dict):
                            for k in ("Avg_EV", "Avg_Par"):
                                v = row.get(k)
                                try:
                                    if isinstance(v, (int, float)):
                                        return float(a_sign) * float(v), f"BT aggregate from {k}"
                                except Exception:
                                    pass
                        return 0.0, "default 0.0"

                    pass_score, pass_note = _pick_pass_score_and_note(
                        pr, acting_sign_adv
                    )
                    pass_bonus = 0.0
                    pass_bonus_reason = None
                    try:
                        pass_bonus, pass_bonus_reason = compute_pass_signoff_bonus(
                            auction_tokens=[t.strip().upper() for t in str(current_auction or "").split("-") if t.strip()],
                            acting_direction=acting_dir_adv,
                            dealer_actual=dealer_actual if "dealer_actual" in locals() else "N",
                        )
                    except Exception:
                        pass_bonus = 0.0
                        pass_bonus_reason = None
                    pass_score = float(pass_score) + float(pass_bonus)
                    if pass_bonus_reason:
                        pass_note = f"{pass_note}; {pass_bonus_reason}"
                    pass_row_dict = {
                        "Bid": "P",
                        "Error": f"Synthetic row: Pass is not BT-backed (no /bid-details evidence); {pass_note}.",
                        "VulFilter": vul_filter_val,
                        "Score": pass_score,
                        "Guard": None,
                        "GuardWhy": None,
                        "U": None,
                        "ParMean_NS": (
                            float(pr.get("Avg_Par"))
                            if isinstance(pr, dict) and isinstance(pr.get("Avg_Par"), (int, float))
                            else (pr.get("Avg_Par") if isinstance(pr, dict) else None)
                        ),
                        "ParMean_Act": (
                            (float(acting_sign_adv) * float(pr.get("Avg_Par")))
                            if isinstance(pr, dict) and isinstance(pr.get("Avg_Par"), (int, float))
                            else (pr.get("Avg_Par") if isinstance(pr, dict) else None)
                        ),
                        "P>=0%": None,
                        "Ent": None,
                        "Tail": None,
                        "Desc": None,
                        "HCP%ile": None,
                        "Threat6+%": None,
                        "FitUs": None,
                        "EstTricks": None,
                        "QLosers": None,
                        "SelfQT": None,
                        "SelfLTC": None,
                        "SelfTP": None,
                        "PtnrTP": None,
                        "TopPar": None,
                        "TopPar%": None,
                        "n": (pr.get("Deals", None) if isinstance(pr, dict) else None)
                        or (pr.get("Matches", None) if isinstance(pr, dict) else None),
                        "Sampled": None,
                        "Degraded": "synthetic_pass",
                        "Source": get_bucket_from_row(pr) if pr else "Valid",
                        "Categories": pr.get("Categories", "") if isinstance(pr, dict) else "",
                    }

                    try:
                        pass_df = pl.DataFrame([pass_row_dict])
                        if adv_df.is_empty():
                            adv_df = pass_df
                        else:
                            adv_df = pl.concat([adv_df, pass_df], how="diagonal_relaxed")
                        # Also include the synthetic pass row in debug data.
                        try:
                            _dbg_list = st.session_state.get("_ab_adv_rows_debug")
                            if isinstance(_dbg_list, list):
                                _dbg_list.append(pass_row_dict)
                        except Exception:
                            pass
                    except Exception:
                        # If concatenation fails for any reason, leave adv_df unchanged.
                        pass
                if not adv_df.is_empty():
                    # Rank column for readability
                    try:
                        # IMPORTANT: keep ordering consistent with automated AI Model.
                        # The top row must be the highest Score candidate.
                        adv_df = adv_df.sort("Score", descending=True, nulls_last=True)
                        adv_df = adv_df.with_row_index("Rank", offset=1)
                    except Exception:
                        # Fallback to simple Score sort
                        try:
                            adv_df = adv_df.sort("Score", descending=True, nulls_last=True)
                            adv_df = adv_df.with_row_index("Rank", offset=1)
                        except Exception:
                            pass

                    adv_selected = render_aggrid(
                        adv_df,
                        key=f"auction_builder_advanced_best_{deal_idx_key}_{current_auction}_{seed}_{adv_top_n}_{adv_max_deals}",
                        height=calc_grid_height(len(adv_df), max_height=340),
                        table_name="auction_builder_advanced_best_next_bids",
                        update_on=["selectionChanged"],
                        # IMPORTANT: don't force-fit ALL columns; it can squish them so much they look blank.
                        # We want a full-width table with horizontal scroll instead.
                        fit_columns_to_view=False,
                        show_sql_expander=False,
                        sort_model=[{"colId": "Score", "sort": "desc"}],
                        tooltip_cols=["Error", "Degraded", "Flags", "TopPar"],
                        row_bucket_col="Source",
                    )
                    if adv_selected:
                        sel_bid = str((adv_selected[0] or {}).get("Bid", "") or "").strip().upper()
                        if sel_bid:
                            next_auc = f"{current_auction}-{sel_bid}" if current_auction else sel_bid
                            next_auc = normalize_auction_input(next_auc).upper() if next_auc else ""
                            st.session_state["_auction_builder_pending_set_auction"] = next_auc
                            st.rerun()
                else:
                    st.info("No valid bids available to rank.")

                # Display 3-column layout: 5 Suggested Bids | Best Par Bids | Best EV Bids
                sug_col, par_col, ev_col = st.columns(3)
                
                with sug_col:
                    if suggested_bids_rows:
                        st.markdown("**Best Bids Ranked by Cheating Model**")
                        # Build enriched rows with Score, is_vul, Result, Dbl columns
                        enriched_sug_rows = []
                        dealer_actual = str(pinned_deal.get("Dealer", "N")).upper() if pinned_deal else "N"
                        vul = str(pinned_deal.get("Vul", pinned_deal.get("Vulnerability", ""))).upper() if pinned_deal else ""
                        ns_vul = vul in ("N_S", "NS", "BOTH", "ALL")
                        ew_vul = vul in ("E_W", "EW", "BOTH", "ALL")
                        
                        for r in suggested_bids_rows:
                            bid_str = str(r.get("Bid", "")).strip().upper().split(" ")[0]  # Remove complete marker if present
                            next_auction = f"{current_auction}-{bid_str}" if current_auction else bid_str
                            next_auction = normalize_auction_input(next_auction).upper() if next_auction else ""
                            
                            # Compute is_vul: True if declarer is vulnerable
                            is_vul_val = False
                            result_val: int | None = None
                            dbl_val = ""
                            is_dbl_val = False
                            is_rdbl_val = False
                            score_val: int | None = None
                            
                            if pinned_deal:
                                try:
                                    # Doubled status does NOT depend on declarer; it depends on the auction calls.
                                    # NOTE: parse_contract_from_auction() does NOT encode doubling state; its 3rd return
                                    # value is the contract bid position. We compute X/XX by scanning calls after the
                                    # last contract bid (same semantics as bbo_bidding_queries_lib.get_ai_contract()).
                                    bids = [b.strip().upper() for b in (next_auction or "").split("-") if b.strip()]
                                    last_contract_idx = -1
                                    for i in range(len(bids) - 1, -1, -1):
                                        bid = bids[i]
                                        if not bid or bid == "P":
                                            continue
                                        if bid in ("X", "D", "DBL", "DOUBLE"):
                                            continue
                                        if bid in ("XX", "R", "RDBL", "REDOUBLE"):
                                            continue
                                        if len(bid) >= 2 and bid[0].isdigit():
                                            last_contract_idx = i
                                            break

                                    if last_contract_idx >= 0:
                                        is_doubled = False
                                        is_redoubled = False
                                        for bid in bids[last_contract_idx + 1 :]:
                                            if bid in ("X", "D", "DBL", "DOUBLE"):
                                                is_doubled = True
                                                is_redoubled = False  # a new double resets redouble
                                            elif bid in ("XX", "R", "RDBL", "REDOUBLE"):
                                                is_redoubled = True

                                        is_rdbl_val = bool(is_redoubled)
                                        is_dbl_val = bool(is_doubled and not is_redoubled)
                                        dbl_val = "XX" if is_rdbl_val else ("X" if is_doubled else "")

                                    # Parse contract and get declarer
                                    c = parse_contract_from_auction(next_auction)
                                    if c:
                                        declarer = get_declarer_for_auction(next_auction, dealer_actual)
                                        if declarer:
                                            declarer_dir = str(declarer).upper()
                                            # Check if declarer's pair is vulnerable
                                            is_ns = declarer_dir in ("N", "S")
                                            is_vul_val = (is_ns and ns_vul) or (not is_ns and ew_vul)
                                            
                                            # Result: DD tricks (from get_dd_tricks_for_auction)
                                            try:
                                                tricks = get_dd_tricks_for_auction(next_auction, dealer_actual, pinned_deal)
                                                # "Result" convention: (tricks taken) - (contract level + 6)
                                                # e.g., 4H making exactly => 0, one overtrick => +1, down two => -2
                                                if tricks is None:
                                                    result_val = None
                                                else:
                                                    level, _strain, _pos = c
                                                    result_val = int(tricks) - (int(level) + 6)
                                            except Exception:
                                                result_val = None

                                            # Score: use mlBridgeLib.score
                                            # mlBridgeLib.score expects:
                                            # - level: 0..6 (contract level - 1)
                                            # - suit: 0..4 (CDHSN)
                                            # - double: 0..2 (''/X/XX)
                                            # - declarer: 0..3 (N/E/S/W index)
                                            # - vulnerability: 0..1 (is declarer vul)
                                            # - result: -13..6 (over/under tricks)
                                            if result_val is not None:
                                                level_i, strain_i, _pos_i = c
                                                lvl0 = int(level_i) - 1
                                                suit0 = mlBridgeLib.StrainSymToValue(str(strain_i).upper())
                                                dbl0 = 2 if is_rdbl_val else (1 if is_dbl_val else 0)
                                                decl0 = 'NSEW'.index(declarer_dir)
                                                vul0 = 1 if is_vul_val else 0
                                                score_val = mlBridgeLib.score(
                                                        lvl0,
                                                        suit0,
                                                        dbl0,
                                                        decl0,
                                                        vul0,
                                                        result_val,
                                                        declarer_score=True,
                                                    )
                                                # reverse sign of score if declarer and seat are on opposite sides
                                                if (declarer_dir in ("N", "S")) ^ (seat_direction in ("N", "S")):
                                                    score_val = -score_val
                                except Exception:
                                    pass
                            
                            enriched_sug_rows.append({
                                "_idx": r["_idx"],
                                "Bid": r["Bid"],
                                "EV Contract": r.get("EV Contract"),
                                "DD Score": r.get("DD Score"),
                                "Avg_EV": r.get("Avg_EV"),
                                "Score": score_val,
                                "is_vul": is_vul_val,
                                "is_dbl": is_dbl_val,
                                "is_rdbl": is_rdbl_val,
                                "Result": result_val,
                                "Dbl": dbl_val,
                                "Bucket": r["Bucket"],
                            })
                        
                        sug_df = pl.DataFrame(enriched_sug_rows)
                        # Ensure default ordering is by Score (AgGrid can persist old sort state).
                        if "Score" in sug_df.columns:
                            try:
                                sug_df = sug_df.sort("Score", descending=True, nulls_last=True)
                            except Exception:
                                pass

                        # Update Cheater Model display: full greedy completion.
                        try:
                            # IMPORTANT: Cheater Model should be invariant to the current built auction.
                            # Compute it once per pinned deal from the opening (empty prefix).
                            if best_auc_ss_key not in st.session_state:
                                best_complete = _greedy_best_auction_complete("", max_steps=40)
                                if isinstance(best_complete, str) and best_complete.strip():
                                    st.session_state[best_auc_ss_key] = best_complete.strip()
                                    best_auction_placeholder.empty()
                            else:
                                cached_best = st.session_state.get(best_auc_ss_key)
                                if isinstance(cached_best, str) and cached_best.strip():
                                    best_auction_placeholder.empty()
                        except Exception:
                            pass
                        sug_pdf = to_aggrid_safe_pandas(sug_df)
                        gb_sug = GridOptionsBuilder.from_dataframe(sug_pdf)
                        gb_sug.configure_selection(selection_mode="single", use_checkbox=False)
                        gb_sug.configure_column("_idx", hide=True)
                        gb_sug.configure_column("Bucket", hide=True)
                        if "EV Contract" in sug_pdf.columns:
                            gb_sug.configure_column("EV Contract", width=110)
                        if "DD Score" in sug_pdf.columns:
                            gb_sug.configure_column("DD Score", width=95)
                        if "Avg_EV" in sug_pdf.columns:
                            gb_sug.configure_column("Avg_EV", width=95)
                        if "Score" in sug_pdf.columns:
                            gb_sug.configure_column("Score", width=80)
                        if "is_vul" in sug_pdf.columns:
                            gb_sug.configure_column("is_vul", width=70)
                        if "is_dbl" in sug_pdf.columns:
                            gb_sug.configure_column("is_dbl", width=70)
                        if "is_rdbl" in sug_pdf.columns:
                            gb_sug.configure_column("is_rdbl", width=75)
                        if "Result" in sug_pdf.columns:
                            gb_sug.configure_column("Result", width=80)
                        if "Dbl" in sug_pdf.columns:
                            gb_sug.configure_column("Dbl", width=50)
                        gb_sug.configure_grid_options(
                            getRowClass=row_class_js,
                            rowHeight=25,
                            headerHeight=25,
                        )
                        sug_opts = gb_sug.build()
                        # Force initial sort by Score descending.
                        if "Score" in sug_pdf.columns and "sortModel" not in sug_opts:
                            sug_opts["sortModel"] = [{"colId": "Score", "sort": "desc"}]
                        sug_resp = AgGrid(
                            sug_pdf,
                            gridOptions=sug_opts,
                            height=175,
                            update_on=["selectionChanged"],
                            theme="balham",
                            allow_unsafe_jscode=True,
                            custom_css=bid_table_css,
                            key=f"auction_builder_suggested_bids_{current_auction}_{current_seat}",
                        )
                        handle_bid_selection(sug_resp, "suggested")
                    else:
                        st.markdown("**Best Bids Ranked by Cheating Model**")
                        st.caption("No suggested bids available")
                
                with par_col:
                    st.markdown("**Best Bids Ranked by DD Score**")
                    if best_par_bids:
                        par_pdf = to_aggrid_safe_pandas(best_par_bids_df)
                        gb_par = GridOptionsBuilder.from_dataframe(par_pdf)
                        gb_par.configure_selection(selection_mode="single", use_checkbox=False)
                        gb_par.configure_column("_idx", hide=True)
                        gb_par.configure_column("Bucket", hide=True)
                        gb_par.configure_column("Contract", width=90)  # 10% tighter
                        gb_par.configure_column("DD Score", width=95)
                        gb_par.configure_column("Matches", width=105)   # adjusted
                        gb_par.configure_grid_options(
                            getRowClass=row_class_js,
                            rowHeight=25,
                            headerHeight=25,
                        )
                        par_opts = gb_par.build()
                        par_resp = AgGrid(
                            par_pdf,
                            gridOptions=par_opts,
                            height=175,
                            update_on=["selectionChanged"],
                            theme="balham",
                            allow_unsafe_jscode=True,
                            custom_css=bid_table_css,
                            key=f"auction_builder_best_par_bids_{current_seat}",
                        )
                        handle_bid_selection(par_resp, "best_par")
                    else:
                        st.caption("No Par scores available")
                
                with ev_col:
                    st.markdown("**Best Bids Ranked by EV Score**")
                    if best_ev_bids:
                        ev_pdf = to_aggrid_safe_pandas(best_ev_bids_df)
                        gb_ev = GridOptionsBuilder.from_dataframe(ev_pdf)
                        gb_ev.configure_selection(selection_mode="single", use_checkbox=False)
                        gb_ev.configure_column("_idx", hide=True)
                        gb_ev.configure_column("Bucket", hide=True)
                        gb_ev.configure_column("Contract", width=90)  # 10% tighter
                        gb_ev.configure_column("EV Score", width=95)
                        gb_ev.configure_column("Matches", width=105)   # adjusted
                        gb_ev.configure_grid_options(
                            getRowClass=row_class_js,
                            rowHeight=25,
                            headerHeight=25,
                        )
                        ev_opts = gb_ev.build()
                        ev_resp = AgGrid(
                            ev_pdf,
                            gridOptions=ev_opts,
                            height=175,
                            update_on=["selectionChanged"],
                            theme="balham",
                            allow_unsafe_jscode=True,
                            custom_css=bid_table_css,
                            key=f"auction_builder_best_ev_bids_{current_seat}",
                        )
                        handle_bid_selection(ev_resp, "best_ev")
                    else:
                        st.caption("No EV scores available")

                # Best Auctions by DD: find top auctions via UI-valid bids (criteria-pass only)
                def find_best_reachable_auctions_from_valid(
                    start_valid_rows: list[dict[str, Any]],
                    start_auction: str,
                    max_results: int = 5,
                    max_depth: int = 20,
                    max_api_calls: int = 1000,
                ) -> list[dict[str, Any]]:
                    """
                    Best Auctions by DD.

                    Uses the same lookahead concept as the debug trace's UI_Valid_Bid_Scores:
                    for each UI-valid (criteria-pass) bid, explore deeper and compute the best
                    completed DD score reachable in that subtree. Collect completed auctions
                    and return all par-score auctions; if fewer than 5, fill with next-best DD.
                    """
                    if not pinned_deal or not start_valid_rows:
                        return []
                    
                    dealer_actual = str(pinned_deal.get("Dealer", "N")).upper()
                    row_idx = pinned_deal.get("_row_idx")
                    par_score = pinned_deal.get("ParScore", pinned_deal.get("Par_Score"))
                    par_score_i: int | None = None
                    try:
                        par_score_i = int(par_score) if par_score is not None else None
                    except Exception:
                        par_score_i = None
                    
                    # Collect completed auctions as: (dd_score, auction_str, contract_str, depth, path_bids)
                    results: list[tuple[int, str, str, int, list[str]]] = []
                    api_call_count = 0
                    # Performance caches (per search invocation)
                    ui_valid_cache: dict[str, list[dict[str, Any]]] = {}
                    criteria_pass_cache: dict[tuple[str, int, tuple[str, ...]], bool] = {}
                    
                    def get_contract_str(auc: str) -> str:
                        c = parse_contract_from_auction(auc)
                        if c:
                            l, s, _ = c
                            return f"{l}{'NT' if str(s).upper() == 'N' else str(s).upper()}"
                        return "Passed out" if all(t == "P" for t in auc.split("-") if t.strip()) else "?"

                    def is_auc_complete(auc: str) -> bool:
                        bids = [b.strip().upper() for b in auc.split("-") if b.strip()]
                        if len(bids) >= 4 and all(b == "P" for b in bids[:4]): return True
                        last_c = -1
                        for i, b in enumerate(bids):
                            if b not in ("P", "X", "XX") and b[0].isdigit(): last_c = i
                        return last_c >= 0 and len(bids) >= last_c + 4 and all(b == "P" for b in bids[-3:])

                    def _get_ui_valid_options(prefix_auc: str) -> list[dict[str, Any]]:
                        """UI-valid (criteria-pass) candidates, ignoring UI rejection reasons."""
                        nonlocal api_call_count
                        cached = ui_valid_cache.get(prefix_auc)
                        if cached is not None:
                            return cached
                        if api_call_count >= max_api_calls:
                            out: list[dict[str, Any]] = []
                            ui_valid_cache[prefix_auc] = out
                            return out
                        try:
                            bt_prefix, lp_here = _strip_leading_passes(prefix_auc)
                            api_call_count += 1
                            opts = get_next_bid_options(bt_prefix)
                        except Exception:
                            out = []
                            ui_valid_cache[prefix_auc] = out
                            return out

                        if not opts:
                            out = []
                            ui_valid_cache[prefix_auc] = out
                            return out

                        # Seat/dealer rotation must be computed from *this* prefix's leading passes.
                        toks = [b for b in prefix_auc.split("-") if b.strip()]
                        display_seat_here = (len(toks) % 4) + 1
                        bt_seat_here = _bt_seat_from_display_seat(display_seat_here, lp_here)
                        dealer_rot_here = _rotate_dealer_by(dealer_actual, lp_here)

                        cand = [o for o in opts if not o.get("is_dead_end")]
                        if row_idx is None:
                            # Cannot evaluate criteria -> permissive.
                            ui_valid_cache[prefix_auc] = cand
                            return cand

                        # Criteria-pass = failed/untracked are empty.
                        valid: list[dict[str, Any]] = []
                        no_crit = [o for o in cand if not (o.get("agg_expr") or [])]
                        valid.extend(no_crit)

                        with_crit = [o for o in cand if (o.get("agg_expr") or [])]
                        if not with_crit:
                            ui_valid_cache[prefix_auc] = valid
                            return valid

                        # Batch eval in chunks (avoid huge payloads), with per-criterion caching.
                        chunk_size = 200
                        i0 = 0
                        while i0 < len(with_crit) and api_call_count < max_api_calls:
                            chunk = with_crit[i0 : i0 + chunk_size]
                            i0 += chunk_size
                            try:
                                # Split into cached vs unknown checks
                                unknown: list[dict[str, Any]] = []
                                for o in chunk:
                                    crit = tuple(str(x) for x in (o.get("agg_expr") or []) if x is not None)
                                    ck = (dealer_rot_here, int(bt_seat_here), crit)
                                    cached_pass = criteria_pass_cache.get(ck)
                                    if cached_pass is True:
                                        valid.append(o)
                                    elif cached_pass is False:
                                        continue
                                    else:
                                        unknown.append(o)

                                if not unknown:
                                    continue

                                api_call_count += 1
                                res = api_post(
                                    "/deal-criteria-eval-batch",
                                    {
                                        "deal_row_idx": int(row_idx),
                                        "dealer": dealer_rot_here,
                                        "checks": [{"seat": int(bt_seat_here), "criteria": list(o.get("agg_expr", []))} for o in unknown],
                                    },
                                    timeout=10,
                                )
                                rr = res.get("results") or []
                                for j, r in enumerate(rr[: len(unknown)]):
                                    if not (r.get("failed") or []) and not (r.get("untracked") or []):
                                        valid.append(unknown[j])
                                        crit = tuple(str(x) for x in (unknown[j].get("agg_expr") or []) if x is not None)
                                        criteria_pass_cache[(dealer_rot_here, int(bt_seat_here), crit)] = True
                                    else:
                                        crit = tuple(str(x) for x in (unknown[j].get("agg_expr") or []) if x is not None)
                                        criteria_pass_cache[(dealer_rot_here, int(bt_seat_here), crit)] = False
                            except Exception:
                                # Be permissive on server error.
                                valid.extend(chunk)
                        ui_valid_cache[prefix_auc] = valid
                        return valid

                    # Core recursion: for each UI-valid bid at a node, explore deeper and compute
                    # the best DD reachable in that subtree (this is the same concept as UI_Valid_Bid_Scores).
                    best_dd_cache: dict[tuple[str, int], int] = {}

                    def explore(prefix_auc: str, path_bids: list[str], depth: int) -> int:
                        nonlocal api_call_count
                        if depth >= max_depth or api_call_count >= max_api_calls:
                            return -99999

                        cache_key = (prefix_auc, depth)
                        if cache_key in best_dd_cache:
                            return best_dd_cache[cache_key]

                        best_here = -99999
                        opts = _get_ui_valid_options(prefix_auc)
                        if not opts:
                            best_dd_cache[cache_key] = best_here
                            return best_here

                        # First compute per-candidate "best DD" by exploring further, then order candidates by that.
                        scored_children: list[tuple[int, dict[str, Any], str, list[str], bool]] = []
                        for o in opts:
                            bid = str(o.get("bid", "")).upper()
                            if not bid:
                                continue
                            child_auc = f"{prefix_auc}-{bid}" if prefix_auc else bid
                            child_path = path_bids + [bid]
                            is_complete_flag = bool(o.get("is_complete")) or is_auc_complete(child_auc)
                            if is_complete_flag:
                                ddv = None
                                try:
                                    ddv = get_dd_score_for_auction(child_auc, dealer_actual, pinned_deal)
                                except Exception:
                                    ddv = None
                                dd_i = int(ddv) if ddv is not None else -99999
                                best_here = max(best_here, dd_i)
                                if ddv is not None:
                                    results.append((dd_i, child_auc, get_contract_str(child_auc), depth + 1, child_path))
                                scored_children.append((dd_i, o, child_auc, child_path, True))
                            else:
                                child_best = explore(child_auc, child_path, depth + 1)
                                best_here = max(best_here, child_best)
                                scored_children.append((child_best, o, child_auc, child_path, False))

                        # If ParScore is known, optionally prune sub-branches that cannot reach Par.
                        if par_score_i is not None:
                            scored_children = [x for x in scored_children if x[0] >= int(par_score_i)]

                        # Order: highest "best reachable DD" first.
                        scored_children.sort(key=lambda x: x[0], reverse=True)

                        # If we pruned above, we already explored those subtrees when computing child_best.
                        # No further action needed here.
                        best_dd_cache[cache_key] = best_here
                        return best_here

                    # Kick off exploration from the current auction prefix.
                    # (We intentionally do NOT rely on the UI "Valid" grid rows here, because those exclude
                    # UI-rejected bids; per requirement we ignore UI rejection and only enforce criteria-pass.)
                    explore(start_auction, [], 0)
                    
                    # Dedup + sort results (best DD first, shorter depth tie-break)
                    results.sort(key=lambda x: (x[0], -x[3]), reverse=True)
                    uniq: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for dd_score, auction_str, contract_str, depth, path_bids in results:
                        if auction_str in seen:
                            continue
                        seen.add(auction_str)
                        achieves_par = par_score_i is not None and dd_score == int(par_score_i)
                        uniq.append(
                            {
                                "Auction": auction_str,
                                "Contract": contract_str,
                                "DD_Score": dd_score,
                                "Par": "✅" if achieves_par else "",
                                "Depth": depth,
                            }
                        )

                    # Output policy:
                    # - Include ALL Par-score auctions (if ParScore is known).
                    # - If fewer than 5 Par auctions, fill up to 5 rows with next-best DD auctions.
                    if par_score_i is not None:
                        par_rows = [r for r in uniq if r.get("Par") == "✅"]
                        non_par_rows = [r for r in uniq if r.get("Par") != "✅"]
                        out = list(par_rows)
                        if len(out) < 5:
                            out.extend(non_par_rows[: max(0, 5 - len(out))])
                        return out

                    # No ParScore available: return top-N best DD.
                    return uniq[:max_results]

                def find_best_auctions_by_ev_from_valid(
                    start_valid_rows: list[dict[str, Any]],
                    start_auction: str,
                    max_results: int = 5,
                    max_depth: int = 20,
                    max_api_calls: int = 1000,
                ) -> list[dict[str, Any]]:
                    """Best Auctions by EV (criteria-pass only; ignore UI rejection)."""
                    if not pinned_deal:
                        return []

                    dealer_actual = str(pinned_deal.get("Dealer", "N")).upper()
                    row_idx = pinned_deal.get("_row_idx")
                    par_score = pinned_deal.get("ParScore", pinned_deal.get("Par_Score"))
                    try:
                        par_score_i = int(par_score) if par_score is not None else None
                    except Exception:
                        par_score_i = None

                    api_call_count = 0
                    # Performance caches (per search invocation)
                    ui_valid_cache: dict[str, list[dict[str, Any]]] = {}
                    criteria_pass_cache: dict[tuple[str, int, tuple[str, ...]], bool] = {}
                    # Collect completed auctions as: (ev, dd, auction_str, contract_str, depth, path_bids)
                    ev_results: list[tuple[float, int, str, str, int, list[str]]] = []

                    def get_contract_str(auc: str) -> str:
                        c = parse_contract_from_auction(auc)
                        if c:
                            l, s, _ = c
                            return f"{l}{'NT' if str(s).upper() == 'N' else str(s).upper()}"
                        return "Passed out" if all(t == "P" for t in auc.split("-") if t.strip()) else "?"

                    def is_auc_complete(auc: str) -> bool:
                        bids = [b.strip().upper() for b in auc.split("-") if b.strip()]
                        if len(bids) >= 4 and all(b == "P" for b in bids[:4]):
                            return True
                        last_c = -1
                        for i, b in enumerate(bids):
                            if b not in ("P", "X", "XX") and b[0].isdigit():
                                last_c = i
                        return last_c >= 0 and len(bids) >= last_c + 4 and all(b == "P" for b in bids[-3:])

                    def _get_ui_valid_options(prefix_auc: str) -> list[dict[str, Any]]:
                        """UI-valid (criteria-pass) candidates, ignoring UI rejection reasons."""
                        nonlocal api_call_count
                        cached = ui_valid_cache.get(prefix_auc)
                        if cached is not None:
                            return cached
                        if api_call_count >= max_api_calls:
                            out: list[dict[str, Any]] = []
                            ui_valid_cache[prefix_auc] = out
                            return out
                        try:
                            bt_prefix, lp_here = _strip_leading_passes(prefix_auc)
                            api_call_count += 1
                            opts = get_next_bid_options(bt_prefix)
                        except Exception:
                            out = []
                            ui_valid_cache[prefix_auc] = out
                            return out

                        if not opts:
                            out = []
                            ui_valid_cache[prefix_auc] = out
                            return out

                        toks = [b for b in prefix_auc.split("-") if b.strip()]
                        display_seat_here = (len(toks) % 4) + 1
                        bt_seat_here = _bt_seat_from_display_seat(display_seat_here, lp_here)
                        dealer_rot_here = _rotate_dealer_by(dealer_actual, lp_here)

                        cand = [o for o in opts if not o.get("is_dead_end")]
                        if row_idx is None:
                            ui_valid_cache[prefix_auc] = cand
                            return cand

                        valid: list[dict[str, Any]] = []
                        no_crit = [o for o in cand if not (o.get("agg_expr") or [])]
                        valid.extend(no_crit)
                        with_crit = [o for o in cand if (o.get("agg_expr") or [])]
                        if not with_crit:
                            ui_valid_cache[prefix_auc] = valid
                            return valid

                        chunk_size = 200
                        i0 = 0
                        while i0 < len(with_crit) and api_call_count < max_api_calls:
                            chunk = with_crit[i0 : i0 + chunk_size]
                            i0 += chunk_size
                            try:
                                unknown: list[dict[str, Any]] = []
                                for o in chunk:
                                    crit = tuple(str(x) for x in (o.get("agg_expr") or []) if x is not None)
                                    ck = (dealer_rot_here, int(bt_seat_here), crit)
                                    cached_pass = criteria_pass_cache.get(ck)
                                    if cached_pass is True:
                                        valid.append(o)
                                    elif cached_pass is False:
                                        continue
                                    else:
                                        unknown.append(o)

                                if not unknown:
                                    continue

                                api_call_count += 1
                                res = api_post(
                                    "/deal-criteria-eval-batch",
                                    {
                                        "deal_row_idx": int(row_idx),
                                        "dealer": dealer_rot_here,
                                        "checks": [{"seat": int(bt_seat_here), "criteria": list(o.get("agg_expr", []))} for o in unknown],
                                    },
                                    timeout=10,
                                )
                                rr = res.get("results") or []
                                for j, r in enumerate(rr[: len(unknown)]):
                                    if not (r.get("failed") or []) and not (r.get("untracked") or []):
                                        valid.append(unknown[j])
                                        crit = tuple(str(x) for x in (unknown[j].get("agg_expr") or []) if x is not None)
                                        criteria_pass_cache[(dealer_rot_here, int(bt_seat_here), crit)] = True
                                    else:
                                        crit = tuple(str(x) for x in (unknown[j].get("agg_expr") or []) if x is not None)
                                        criteria_pass_cache[(dealer_rot_here, int(bt_seat_here), crit)] = False
                            except Exception:
                                valid.extend(chunk)
                        ui_valid_cache[prefix_auc] = valid
                        return valid

                    best_ev_cache: dict[tuple[str, int], float] = {}

                    def explore(prefix_auc: str, path_bids: list[str], depth: int) -> float:
                        nonlocal api_call_count
                        if depth >= max_depth or api_call_count >= max_api_calls:
                            return float("-inf")

                        cache_key = (prefix_auc, depth)
                        if cache_key in best_ev_cache:
                            return best_ev_cache[cache_key]

                        best_here = float("-inf")
                        opts = _get_ui_valid_options(prefix_auc)
                        if not opts:
                            best_ev_cache[cache_key] = best_here
                            return best_here

                        # Compute per-candidate best EV by exploring further
                        scored_children: list[tuple[float, dict[str, Any], str, list[str], bool]] = []
                        for o in opts:
                            bid = str(o.get("bid", "")).upper()
                            if not bid:
                                continue
                            child_auc = f"{prefix_auc}-{bid}" if prefix_auc else bid
                            child_path = path_bids + [bid]
                            is_complete_flag = bool(o.get("is_complete")) or is_auc_complete(child_auc)
                            if is_complete_flag:
                                evv = None
                                ddv = None
                                try:
                                    evv = get_ev_for_auction(child_auc, dealer_actual, pinned_deal)
                                except Exception:
                                    evv = None
                                try:
                                    ddv = get_dd_score_for_auction(child_auc, dealer_actual, pinned_deal)
                                except Exception:
                                    ddv = None
                                ev_f = float(evv) if evv is not None else float("-inf")
                                dd_i = int(ddv) if ddv is not None else -99999
                                best_here = max(best_here, ev_f)
                                if evv is not None:
                                    ev_results.append((ev_f, dd_i, child_auc, get_contract_str(child_auc), depth + 1, child_path))
                                scored_children.append((ev_f, o, child_auc, child_path, True))
                            else:
                                child_best = explore(child_auc, child_path, depth + 1)
                                best_here = max(best_here, child_best)
                                scored_children.append((child_best, o, child_auc, child_path, False))

                        scored_children.sort(key=lambda x: x[0], reverse=True)
                        best_ev_cache[cache_key] = best_here
                        return best_here

                    explore(start_auction, [], 0)

                    # Dedup + sort results (best EV first; tie-break by DD then shorter depth)
                    ev_results.sort(key=lambda x: (x[0], x[1], -x[4]), reverse=True)
                    uniq: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for ev_f, dd_i, auction_str, contract_str, depth, path_bids in ev_results:
                        if auction_str in seen:
                            continue
                        seen.add(auction_str)
                        achieves_par = par_score_i is not None and dd_i == int(par_score_i)
                        uniq.append(
                            {
                                "Auction": auction_str,
                                "Contract": contract_str,
                                "EV": round(float(ev_f), 1) if ev_f != float("-inf") else None,
                                "DD_Score": dd_i if dd_i > -90000 else None,
                                "Par": "✅" if achieves_par else "",
                                "Depth": depth,
                            }
                        )

                    return uniq[:max_results]

                # Best Auctions by DD/EV: sample matched BT rows for pinned deal (GPU-verified deal→BT index)
                if pinned_deal and valid_rows:
                    # Include deal identifier in cache key to ensure fresh data for each deal
                    # Use the pinned input string directly - guaranteed to change when user enters new deal
                    deal_cache_id = str(pinned_input or "").strip()
                    
                    show_best_auctions_key = f"show_best_auctions_{current_seat}_{deal_cache_id}"
                    if show_best_auctions_key not in st.session_state:
                        st.session_state[show_best_auctions_key] = False

                    cache_key_dd = f"_best_auctions_dd_cache_{current_auction}_{seed}_{int(max_best_auctions)}_{deal_cache_id}"
                    cache_key_dd_elapsed = f"{cache_key_dd}__elapsed_s"
                    cache_key_dd_attempted = f"{cache_key_dd}__attempted"
                    cache_key_ev = f"_best_auctions_ev_cache_{current_auction}_{seed}_{int(max_best_auctions)}_{deal_cache_id}"
                    cache_key_ev_attempted = f"{cache_key_ev}__attempted"
                    # Fallback: server-side lookahead (does not require verified deal→BT matches)
                    cache_key_dd_look = f"_best_auctions_dd_lookahead_cache_{current_auction}_{seed}_{int(max_best_auctions)}_{deal_cache_id}"
                    cache_key_dd_look_attempted = f"{cache_key_dd_look}__attempted"
                    cache_key_ev_look = f"_best_auctions_ev_lookahead_cache_{current_auction}_{seed}_{int(max_best_auctions)}_{deal_cache_id}"
                    cache_key_ev_look_attempted = f"{cache_key_ev_look}__attempted"

                    # If we've already tried loading and got no rows, hide the button and show a message.
                    attempted_dd = bool(st.session_state.get(cache_key_dd_attempted, False))
                    attempted_ev = bool(st.session_state.get(cache_key_ev_attempted, False))
                    cached_dd_rows = st.session_state.get(cache_key_dd, []) or []
                    cached_ev_rows = st.session_state.get(cache_key_ev, []) or []
                    if attempted_dd and attempted_ev and (not cached_dd_rows) and (not cached_ev_rows):
                        st.warning(
                            "No **pre-computed matched completed auctions** were found for this deal in the verified deal→BT index "
                            "(or they were all filtered out by overlay criteria). This does **not** mean there are no valid auctions.\n\n"
                            "A server-side lookahead search will be used below to find the best DD/EV auctions reachable from the current prefix."
                        )
                        st.session_state[show_best_auctions_key] = True
                        show_best_auctions = True
                    else:
                        # Green button styling
                        st.markdown(
                            """
                            <style>
                            div[data-testid="stButton"] > button[kind="primary"] {
                                background-color: #28a745;
                                border-color: #28a745;
                            }
                            div[data-testid="stButton"] > button[kind="primary"]:hover {
                                background-color: #218838;
                                border-color: #1e7e34;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            f"Show Best {int(max_best_auctions)} Pre-computed Auctions Ranked by DD/EV",
                            key=f"{show_best_auctions_key}__btn",
                            type="primary",
                        ):
                            st.session_state[show_best_auctions_key] = True
                            st.rerun()
                        show_best_auctions = bool(st.session_state.get(show_best_auctions_key, False))

                    if show_best_auctions:
                        st.markdown("**Best Auctions Ranked by DD**")
                        best_auctions_dd: list[dict[str, Any]] = st.session_state.get(cache_key_dd, [])
                        attempted = bool(st.session_state.get(cache_key_dd_attempted, False))

                        # Auto-refresh policy:
                        # - Load once per cache key (auction/seed/max_best_auctions).
                        # - If it failed/returned empty, do not keep retrying on every Streamlit rerun;
                        #   change inputs (or clear caches) to re-trigger.
                        if (not attempted) and (not best_auctions_dd):
                            with st.spinner("Loading (server-side)..."):
                                t0 = time.perf_counter()
                                row_idx = pinned_deal.get("_row_idx")
                                if row_idx is not None:
                                    try:
                                        resp = api_post(
                                            "/deal-matched-bt-sample",
                                            {
                                                "deal_row_idx": int(row_idx),
                                                "metric": "DD",
                                                "n_samples": int(max_best_auctions),
                                                "seed": seed,
                                                "permissive_pass": st.session_state.get("always_valid_pass", True),
                                            },
                                            timeout=30,
                                        )
                                        best_auctions_dd = resp.get("rows", []) or []
                                    except Exception as e:
                                        st.error(f"Server-side load failed: {e}")
                                        best_auctions_dd = []
                                else:
                                    best_auctions_dd = []
                                st.session_state[cache_key_dd] = best_auctions_dd
                                st.session_state[cache_key_dd_elapsed] = float(time.perf_counter() - t0)
                                st.session_state[cache_key_dd_attempted] = True

                        # Fallback: if precomputed sample returned empty, run bounded lookahead once.
                        if (not best_auctions_dd) and pinned_deal:
                            attempted_look = bool(st.session_state.get(cache_key_dd_look_attempted, False))
                            if not attempted_look:
                                with st.spinner("Searching (server-side lookahead)..."):
                                    row_idx = pinned_deal.get("_row_idx")
                                    if row_idx is not None:
                                        try:
                                            resp = api_post(
                                                "/best-auctions-lookahead",
                                                {
                                                    "deal_row_idx": int(row_idx),
                                                    "auction_prefix": current_auction or "",
                                                    "metric": "DD",
                                                    "max_depth": 20,
                                                    "max_results": int(max_best_auctions),
                                                },
                                                timeout=120,
                                            )
                                            aucs = resp.get("auctions") or []
                                            best_auctions_dd = [
                                                {
                                                    "Auction": a.get("auction"),
                                                    "Contract": a.get("contract"),
                                                    "DD_Score": a.get("dd_score"),
                                                    "EV": a.get("ev"),
                                                    "Par": "✅" if a.get("is_par") else "",
                                                }
                                                for a in aucs
                                                if a
                                            ]
                                        except Exception as e:
                                            st.error(f"Lookahead failed: {e}")
                                            best_auctions_dd = []
                                    st.session_state[cache_key_dd_look] = best_auctions_dd
                                    st.session_state[cache_key_dd_look_attempted] = True
                            else:
                                best_auctions_dd = st.session_state.get(cache_key_dd_look, []) or []

                        if best_auctions_dd:
                            par_score = pinned_deal.get("ParScore", pinned_deal.get("Par_Score"))
                            ach_count = sum(1 for a in best_auctions_dd if a.get("Par") == "✅")
                            if par_score is not None:
                                elapsed_s = st.session_state.get(cache_key_dd_elapsed)
                                elapsed_txt = f"{float(elapsed_s):.2f}s" if elapsed_s is not None else "?"
                                if ach_count > 0:
                                    st.success(f"Showing {len(best_auctions_dd)} row(s); {ach_count} achieve par {par_score} ({elapsed_txt})")
                                else:
                                    st.warning(f"Showing {len(best_auctions_dd)} row(s); no par auctions ({elapsed_txt})")

                            best_auctions_df = pl.DataFrame(best_auctions_dd)
                            # Drop API-only helper columns not meant for display.
                            drop_cols = [c for c in ["is_completed_auction", "Score"] if c in best_auctions_df.columns]
                            if drop_cols:
                                best_auctions_df = best_auctions_df.drop(drop_cols)
                            # Sort by DD_Score (desc), Matches (desc), EV (desc), bt_index (asc)
                            sort_cols = []
                            sort_desc = []
                            if "DD_Score" in best_auctions_df.columns:
                                sort_cols.append("DD_Score")
                                sort_desc.append(True)
                            if "Matches" in best_auctions_df.columns:
                                sort_cols.append("Matches")
                                sort_desc.append(True)
                            if "EV" in best_auctions_df.columns:
                                sort_cols.append("EV")
                                sort_desc.append(True)
                            if "bt_index" in best_auctions_df.columns:
                                sort_cols.append("bt_index")
                                sort_desc.append(False)
                            if sort_cols:
                                best_auctions_df = best_auctions_df.sort(sort_cols, descending=sort_desc, nulls_last=True)
                            # Add client-side row number 'index' (1..N) and force it to be first column.
                            if "index" not in best_auctions_df.columns:
                                best_auctions_df = best_auctions_df.with_row_index("index", offset=1)
                            cols = ["index"] + [c for c in best_auctions_df.columns if c != "index"]
                            best_auctions_df = best_auctions_df.select(cols)

                            best_auctions_pdf = to_aggrid_safe_pandas(best_auctions_df)
                            gb_best = GridOptionsBuilder.from_dataframe(best_auctions_pdf)
                            gb_best.configure_selection(selection_mode="single", use_checkbox=False)
                            if "index" in best_auctions_pdf.columns:
                                gb_best.configure_column("index", width=70, headerName="#")
                            gb_best.configure_column("Auction", flex=2, minWidth=200, tooltipField="Auction")
                            gb_best.configure_column("Contract", width=90)
                            if "DD_Score" in best_auctions_pdf.columns:
                                gb_best.configure_column("DD_Score", width=100, headerName="DD Score")
                            if "EV" in best_auctions_pdf.columns:
                                gb_best.configure_column("EV", width=90)
                            gb_best.configure_column("Par", width=60)
                            if "Matches" in best_auctions_pdf.columns:
                                gb_best.configure_column("Matches", width=105)
                            if "Deals" in best_auctions_pdf.columns:
                                gb_best.configure_column("Deals", width=100)
                            if "bt_index" in best_auctions_pdf.columns:
                                gb_best.configure_column("bt_index", width=110, headerName="BT Index")
                            gb_best.configure_grid_options(rowHeight=25, headerHeight=25)
                            best_opts = gb_best.build()

                            best_auctions_resp = AgGrid(
                                best_auctions_pdf,
                                gridOptions=best_opts,
                                height=min(260, 60 + len(best_auctions_dd) * 25),
                                update_on=["selectionChanged"],
                                theme="balham",
                                key=f"auction_builder_best_auctions_dd_{current_seat}_{deal_cache_id}",
                                custom_css={
                                    ".ag-row": {"cursor": "pointer"},
                                    ".ag-row-hover": {"background-color": "#E8F4FF !important", "border-left": "3px solid #007BFF !important"},
                                    ".ag-row-selected": {"background-color": "#CCE5FF !important"},
                                },
                            )

                            sel_rows = best_auctions_resp.get("selected_rows")
                            if sel_rows is not None and len(sel_rows) > 0:
                                sel_row_data = sel_rows[0] if isinstance(sel_rows, list) else sel_rows.iloc[0].to_dict()
                                selected_auction = str(sel_row_data.get("Auction", "")).strip()
                                if selected_auction and selected_auction != current_auction:
                                    try:
                                        data = api_post("/resolve-auction-path", {"auction": selected_auction}, timeout=30)
                                        new_path = data.get("path", [])
                                        if new_path:
                                            st.session_state.auction_builder_path = new_path
                                            st.session_state.auction_builder_last_applied = selected_auction.upper()
                                            st.session_state.auction_builder_last_selected = ""
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to apply auction: {e}")
                        elif cache_key_dd_attempted in st.session_state:
                            st.info("No matched BT rows returned for this deal. (Lookahead may also be empty.)")

                    cache_key_ev = f"_best_auctions_ev_cache_{current_auction}_{seed}_{int(max_best_auctions)}_{deal_cache_id}"
                    cache_key_ev_elapsed = f"{cache_key_ev}__elapsed_s"
                    cache_key_ev_attempted = f"{cache_key_ev}__attempted"

                    if show_best_auctions:
                        st.markdown("**Best Auctions Ranked by EV**")
                        best_auctions_ev: list[dict[str, Any]] = st.session_state.get(cache_key_ev, [])
                        attempted_ev = bool(st.session_state.get(cache_key_ev_attempted, False))

                        if (not attempted_ev) and (not best_auctions_ev):
                            with st.spinner("Loading (server-side)..."):
                                t0_ev = time.perf_counter()
                                row_idx = pinned_deal.get("_row_idx")
                                if row_idx is not None:
                                    try:
                                        resp = api_post(
                                            "/deal-matched-bt-sample",
                                            {
                                                "deal_row_idx": int(row_idx),
                                                "metric": "EV",
                                                "n_samples": int(max_best_auctions),
                                                "seed": seed,
                                                "permissive_pass": st.session_state.get("always_valid_pass", True),
                                            },
                                            timeout=30,
                                        )
                                        best_auctions_ev = resp.get("rows", []) or []
                                    except Exception as e:
                                        st.error(f"Server-side load failed: {e}")
                                        best_auctions_ev = []
                                else:
                                    best_auctions_ev = []
                                st.session_state[cache_key_ev] = best_auctions_ev
                                st.session_state[cache_key_ev_elapsed] = float(time.perf_counter() - t0_ev)
                                st.session_state[cache_key_ev_attempted] = True

                        # Fallback: if precomputed sample returned empty, run bounded lookahead once.
                        if (not best_auctions_ev) and pinned_deal:
                            attempted_look = bool(st.session_state.get(cache_key_ev_look_attempted, False))
                            if not attempted_look:
                                with st.spinner("Searching (server-side lookahead)..."):
                                    row_idx = pinned_deal.get("_row_idx")
                                    if row_idx is not None:
                                        try:
                                            resp = api_post(
                                                "/best-auctions-lookahead",
                                                {
                                                    "deal_row_idx": int(row_idx),
                                                    "auction_prefix": current_auction or "",
                                                    "metric": "EV",
                                                    "max_depth": 20,
                                                    "max_results": int(max_best_auctions),
                                                },
                                                timeout=120,
                                            )
                                            aucs = resp.get("auctions") or []
                                            best_auctions_ev = [
                                                {
                                                    "Auction": a.get("auction"),
                                                    "Contract": a.get("contract"),
                                                    "DD_Score": a.get("dd_score"),
                                                    "EV": a.get("ev"),
                                                    "Par": "✅" if a.get("is_par") else "",
                                                }
                                                for a in aucs
                                                if a
                                            ]
                                        except Exception as e:
                                            st.error(f"Lookahead failed: {e}")
                                            best_auctions_ev = []
                                    st.session_state[cache_key_ev_look] = best_auctions_ev
                                    st.session_state[cache_key_ev_look_attempted] = True
                            else:
                                best_auctions_ev = st.session_state.get(cache_key_ev_look, []) or []

                        if best_auctions_ev:
                            # Info summary (mirror DD expander style, but for EV).
                            elapsed_s = st.session_state.get(cache_key_ev_elapsed)
                            elapsed_txt = f"{float(elapsed_s):.2f}s" if elapsed_s is not None else "?"
                            best_ev = best_auctions_ev[0].get("EV") if best_auctions_ev else None
                            try:
                                best_ev_f = float(best_ev) if best_ev is not None else None
                            except Exception:
                                best_ev_f = None
                            if best_ev_f is None:
                                st.success(f"Showing {len(best_auctions_ev)} row(s); best EV — ({elapsed_txt})")
                            else:
                                achieves_best = 0
                                for r in best_auctions_ev:
                                    try:
                                        if float(r.get("EV")) == best_ev_f:
                                            achieves_best += 1
                                    except Exception:
                                        pass
                                st.success(
                                    f"Showing {len(best_auctions_ev)} row(s); "
                                    f"{achieves_best} achieve best EV of {best_ev_f} ({elapsed_txt})"
                                )

                            best_ev_df = pl.DataFrame(best_auctions_ev)
                            # Drop API-only helper columns not meant for display.
                            drop_cols = [c for c in ["is_completed_auction", "Score"] if c in best_ev_df.columns]
                            if drop_cols:
                                best_ev_df = best_ev_df.drop(drop_cols)
                            # Sort by EV (desc), Matches (desc), DD_Score (desc), bt_index (asc)
                            sort_cols_ev = []
                            sort_desc_ev = []
                            if "EV" in best_ev_df.columns:
                                sort_cols_ev.append("EV")
                                sort_desc_ev.append(True)
                            if "Matches" in best_ev_df.columns:
                                sort_cols_ev.append("Matches")
                                sort_desc_ev.append(True)
                            if "DD_Score" in best_ev_df.columns:
                                sort_cols_ev.append("DD_Score")
                                sort_desc_ev.append(True)
                            if "bt_index" in best_ev_df.columns:
                                sort_cols_ev.append("bt_index")
                                sort_desc_ev.append(False)
                            if sort_cols_ev:
                                best_ev_df = best_ev_df.sort(sort_cols_ev, descending=sort_desc_ev, nulls_last=True)
                            # Add client-side row number 'index' (1..N) and force it to be first column.
                            if "index" not in best_ev_df.columns:
                                best_ev_df = best_ev_df.with_row_index("index", offset=1)
                            cols = ["index"] + [c for c in best_ev_df.columns if c != "index"]
                            best_ev_df = best_ev_df.select(cols)

                            best_ev_pdf = to_aggrid_safe_pandas(best_ev_df)
                            gb_ev_auc = GridOptionsBuilder.from_dataframe(best_ev_pdf)
                            gb_ev_auc.configure_selection(selection_mode="single", use_checkbox=False)
                            if "index" in best_ev_pdf.columns:
                                gb_ev_auc.configure_column("index", width=70, headerName="#")
                            gb_ev_auc.configure_column("Auction", flex=2, minWidth=200, tooltipField="Auction")
                            gb_ev_auc.configure_column("Contract", width=90)
                            if "EV" in best_ev_pdf.columns:
                                gb_ev_auc.configure_column("EV", width=90)
                            if "DD_Score" in best_ev_pdf.columns:
                                gb_ev_auc.configure_column("DD_Score", width=110, headerName="DD Score")
                            if "Par" in best_ev_pdf.columns:
                                gb_ev_auc.configure_column("Par", width=60)
                            if "Matches" in best_ev_pdf.columns:
                                gb_ev_auc.configure_column("Matches", width=105)
                            if "Deals" in best_ev_pdf.columns:
                                gb_ev_auc.configure_column("Deals", width=100)
                            if "bt_index" in best_ev_pdf.columns:
                                gb_ev_auc.configure_column("bt_index", width=110, headerName="BT Index")
                            gb_ev_auc.configure_grid_options(rowHeight=25, headerHeight=25)
                            best_ev_opts = gb_ev_auc.build()

                            best_ev_resp = AgGrid(
                                best_ev_pdf,
                                gridOptions=best_ev_opts,
                                height=min(260, 60 + len(best_auctions_ev) * 25),
                                update_on=["selectionChanged"],
                                theme="balham",
                                key=f"auction_builder_best_auctions_ev_{current_seat}_{deal_cache_id}",
                                custom_css={
                                    ".ag-row": {"cursor": "pointer"},
                                    ".ag-row-hover": {"background-color": "#E8F4FF !important", "border-left": "3px solid #007BFF !important"},
                                    ".ag-row-selected": {"background-color": "#CCE5FF !important"},
                                },
                            )

                            sel_rows_ev = best_ev_resp.get("selected_rows")
                            if sel_rows_ev is not None and len(sel_rows_ev) > 0:
                                sel_row_ev = sel_rows_ev[0] if isinstance(sel_rows_ev, list) else sel_rows_ev.iloc[0].to_dict()
                                selected_auction_ev = str(sel_row_ev.get("Auction", "")).strip()
                                if selected_auction_ev and selected_auction_ev != current_auction:
                                    try:
                                        data = api_post("/resolve-auction-path", {"auction": selected_auction_ev}, timeout=30)
                                        new_path = data.get("path", [])
                                        if new_path:
                                            st.session_state.auction_builder_path = new_path
                                            st.session_state.auction_builder_last_applied = selected_auction_ev.upper()
                                            st.session_state.auction_builder_last_selected = ""
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to apply auction: {e}")
                        elif cache_key_ev_attempted in st.session_state:
                            st.info("No matched BT rows returned for this deal (or limit is 0).")

                    # Debug: Trace an auction path expander removed (no longer needed).

                if not valid_rows:
                    st.warning(
                        "No **pinned-deal-matching** next bids for this auction step. "
                        "This does not mean there are no **legal** bridge bids — it means the pinned deal fails "
                        "the criteria for all available BT continuations. "
                        "You can still select from **Rejected** / **Invalid** bids to continue the auction."
                    )
                
                # Valid bids grid
                if valid_rows:
                    st.markdown(f"**✅ Valid Bids ({len(valid_rows)})**")
                    valid_df = pl.DataFrame(valid_rows)
                    # Drop Failed Criteria from valid bids (not relevant since they all passed)
                    for col in ["Failed Criteria", "Failed_Criteria"]:
                        if col in valid_df.columns:
                            valid_df = valid_df.drop(col)
                    # can_complete is only meaningful for diagnosing rejected/invalid paths
                    if "can_complete" in valid_df.columns:
                        valid_df = valid_df.drop("can_complete")
                    valid_pdf = to_aggrid_safe_pandas(valid_df)
                    valid_opts = build_bid_grid_options(valid_pdf, apply_row_styling=False)
                    # Height: header (25) + rows (25 each) + buffer (35)
                    valid_resp = AgGrid(
                        valid_pdf,
                        gridOptions=valid_opts,
                        height=min(375, 60 + len(valid_rows) * 25),
                        update_on=["selectionChanged"],
                        key=f"auction_builder_valid_grid_{current_seat}",
                        theme="balham",
                        allow_unsafe_jscode=True,
                        custom_css={
                            ".ag-row": {"cursor": "pointer", "background-color": "#d4edda"},
                            ".ag-row-hover": {"background-color": "#b8dfc4 !important", "border-left": "3px solid #28a745 !important"},
                            ".ag-row-selected": {"background-color": "#a3d4af !important"},
                        },
                    )
                    handle_bid_selection(valid_resp, "valid")

                    # ----------------------------------------------------------
                    # Seat Stats: Actual (invariant) + Computed (per-bid means)
                    # ----------------------------------------------------------
                    _DIRECTIONS_STAT = ["N", "E", "S", "W"]
                    _DD_DECL = ["N", "E", "S", "W"]
                    _DD_STRAIN = ["C", "D", "H", "S", "N"]
                    _DD_FRIENDLY = {"C": "C", "D": "D", "H": "H", "S": "S", "N": "NT"}

                    # -- Actual Deal Stats (invariant, from pinned deal) --
                    try:
                        _actual_hand_df: pd.DataFrame | None = None
                        _actual_dd_df: pd.DataFrame | None = None
                        if pinned_deal:
                            _actual_hand_rows: list[dict] = []
                            _actual_dd_rows: list[dict] = []
                            # Hand_ strings use dot-notation: S.H.D.C
                            _HAND_SUIT_ORDER = ["S", "H", "D", "C"]
                            # Display order: C, D, H, S
                            _DISPLAY_SUIT_ORDER = ["C", "D", "H", "S"]
                            for _dir in _DIRECTIONS_STAT:
                                _hs_row: dict[str, Any] = {"Direction": _dir}
                                _hcp = pinned_deal.get(f"HCP_{_dir}")
                                _hs_row["HCP"] = round(float(_hcp), 1) if _hcp is not None else None
                                _hs_row["HCP_std"] = None  # single deal

                                # Compute suit lengths from Hand_ string (always available)
                                _hand_str = str(pinned_deal.get(f"Hand_{_dir}", "") or "")
                                _hand_parts = _hand_str.split(".") if _hand_str else []
                                _sl_by_suit: dict[str, int | None] = {}
                                if len(_hand_parts) == 4:
                                    for _si, _sk in enumerate(_HAND_SUIT_ORDER):
                                        _sl_by_suit[_sk] = len(_hand_parts[_si])
                                else:
                                    for _sk in _DISPLAY_SUIT_ORDER:
                                        _v = pinned_deal.get(f"SL_{_sk}_{_dir}")
                                        _sl_by_suit[_sk] = int(_v) if _v is not None else None
                                for _sk in _DISPLAY_SUIT_ORDER:
                                    _hs_row[f"SL_{_sk}"] = _sl_by_suit.get(_sk)

                                _tp = pinned_deal.get(f"Total_Points_{_dir}")
                                _hs_row["Total_Points"] = round(float(_tp), 1) if _tp is not None else None
                                _hs_row["Total_Points_std"] = None  # single deal
                                _actual_hand_rows.append(_hs_row)

                                pass  # DD rows built separately below

                            # DD tricks: 4 rows (declarer) × 5 cols (strain)
                            for _decl in _DD_DECL:
                                _dd_row: dict[str, Any] = {"Declarer": _decl}
                                for _strain in _DD_STRAIN:
                                    _v = pinned_deal.get(f"DD_{_decl}_{_strain}")
                                    _dd_row[_strain] = int(float(_v)) if _v is not None else None
                                _actual_dd_rows.append(_dd_row)

                            if any(v is not None for r in _actual_hand_rows for k, v in r.items() if k != "Direction"):
                                _actual_hand_df = pd.DataFrame(_actual_hand_rows).set_index("Direction")
                            if any(v is not None for r in _actual_dd_rows for k, v in r.items() if k != "Declarer"):
                                _actual_dd_df = pd.DataFrame(_actual_dd_rows).set_index("Declarer")

                            if _actual_hand_df is not None or _actual_dd_df is not None:
                                with st.expander("Actual Deal Stats (pinned deal)", expanded=False):
                                    if _actual_hand_df is not None:
                                        st.markdown("**Hand Stats**")
                                        st.dataframe(_actual_hand_df, width="stretch")
                                    if _actual_dd_df is not None:
                                        st.markdown("**DD Tricks**")
                                        st.dataframe(_actual_dd_df, width="stretch")
                    except Exception as _e_actual:
                        st.caption(f"⚠️ Actual deal stats error: {_e_actual}")

                    # -- Per-deal Stats per bid (from sampled deals matching bt_index) --
                    _load_per_bid_stats = st.toggle(
                        "Load sampled per-bid deal stats (slower)",
                        value=False,
                        key=f"ab_load_per_bid_stats_{current_seat}_{deal_cache_id}",
                        help="Fetches sampled deals for all valid BT rows. Can add significant latency.",
                    )
                    if _load_per_bid_stats:
                        try:
                            _bt_indices_for_stats = [
                                int(r["BT Index"])
                                for r in valid_rows
                                if r.get("BT Index") is not None
                            ]

                            _deals_by_bt: dict = {}
                            if _bt_indices_for_stats:
                                _sample_batch_cache = st.session_state.setdefault(
                                    "_auction_builder_sample_deals_batch_cache", {}
                                )
                                _batch_key = (
                                    tuple(sorted(_bt_indices_for_stats)),
                                    int(seed) if seed is not None else None,
                                    100,
                                )
                                _deals_by_bt = _sample_batch_cache.get(_batch_key)
                                if _deals_by_bt is None:
                                    _deals_resp = api_post(
                                        "/sample-deals-for-bt-indices",
                                        {
                                            "bt_indices": _bt_indices_for_stats,
                                            "sample_size": 100,
                                            "seed": int(seed) if seed is not None else None,
                                        },
                                        timeout=120,
                                    )
                                    _deals_by_bt = _deals_resp.get("deals_by_bt", {})
                                    _sample_batch_cache[_batch_key] = _deals_by_bt

                            _bt_to_bid = {
                                str(int(r["BT Index"])): str(r.get("Bid", "?"))
                                for r in valid_rows
                                if r.get("BT Index") is not None
                            }
                            # Determine active bidder's direction and shape from pinned deal
                            _active_dir: str | None = None
                            _active_shape: list[int] | None = None  # sorted suit lengths
                            _active_exact_sl: dict[str, int] | None = None  # exact suit lengths by suit
                            if pinned_deal:
                                try:
                                    _dirs = ["N", "E", "S", "W"]
                                    _dealer = str(pinned_deal.get("Dealer", "N")).upper()
                                    _d_idx = _dirs.index(_dealer) if _dealer in _dirs else 0
                                    _active_dir = _dirs[(_d_idx + seat_1_to_4 - 1) % 4]
                                    _hand_str = str(pinned_deal.get(f"Hand_{_active_dir}", "") or "")
                                    _parts = _hand_str.split(".") if _hand_str else []
                                    if len(_parts) == 4:
                                        _suits = ["S", "H", "D", "C"]
                                        _active_exact_sl = {s: len(p) for s, p in zip(_suits, _parts)}
                                        _active_shape = sorted((len(p) for p in _parts), reverse=True)
                                except Exception:
                                    pass

                            _DD_STRAINS = ["C", "D", "H", "S", "N"]

                            for _bt_key, _bt_deal_data in _deals_by_bt.items():
                                _bid_label = _bt_to_bid.get(_bt_key, _bt_key)
                                _deal_list = _bt_deal_data.get("deals", [])
                                _total_ct = _bt_deal_data.get("total_count", len(_deal_list))
                                if not _deal_list:
                                    continue

                                # Build Hand Stats DF: one row per deal, columns per direction
                                _hs_rows: list[dict] = []
                                _dd_rows: list[dict] = []
                                for _deal in _deal_list:
                                    _hs_r: dict[str, Any] = {"deal_idx": _deal.get("index")}
                                    _dd_r: dict[str, Any] = {"deal_idx": _deal.get("index")}
                                    for _d in ["N", "E", "S", "W"]:
                                        _hs_r[f"HCP_{_d}"] = _deal.get(f"HCP_{_d}")
                                        for _suit in ["S", "H", "D", "C"]:
                                            _hs_r[f"SL_{_suit}_{_d}"] = _deal.get(f"SL_{_suit}_{_d}")
                                        _hs_r[f"TP_{_d}"] = _deal.get(f"Total_Points_{_d}")
                                    for _decl in ["N", "E", "S", "W"]:
                                        for _strain in _DD_STRAINS:
                                            _dd_r[f"DD_{_decl}_{_strain}"] = _deal.get(f"DD_{_decl}_{_strain}")
                                    _hs_rows.append(_hs_r)
                                    _dd_rows.append(_dd_r)

                                _hs_df = pd.DataFrame(_hs_rows)
                                _dd_df = pd.DataFrame(_dd_rows)

                                # Build shape-filtered DFs
                                _hs_shape_df: pd.DataFrame | None = None
                                _dd_shape_df: pd.DataFrame | None = None
                                _shape_label = ""
                                if _active_dir and _active_shape and _active_exact_sl:
                                    _shape_label = "-".join(str(x) for x in _active_shape)
                                    _shape_mask = []
                                    for _deal in _deal_list:
                                        _hand = str(_deal.get(f"Hand_{_active_dir}", "") or "")
                                        _hp = _hand.split(".") if _hand else []
                                        if len(_hp) == 4:
                                            _dl = sorted((len(p) for p in _hp), reverse=True)
                                            _shape_mask.append(_dl == _active_shape)
                                        else:
                                            _shape_mask.append(False)
                                    if any(_shape_mask):
                                        _hs_shape_df = _hs_df[_shape_mask].reset_index(drop=True)
                                        _dd_shape_df = _dd_df[_shape_mask].reset_index(drop=True)

                                def _agg_hand_stats(df: pd.DataFrame) -> pd.DataFrame:
                                    """Aggregate hand stats: 4 rows (direction) × mean/std cols."""
                                    rows = []
                                    for _d in ["N", "E", "S", "W"]:
                                        r: dict[str, Any] = {"Direction": _d}
                                        for stat in ["HCP", "Total_Points"]:
                                            col = f"{stat}_{_d}" if stat == "HCP" else f"TP_{_d}"
                                            if col in df.columns:
                                                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                                                r[stat] = round(float(vals.mean()), 1) if len(vals) else None
                                                r[f"{stat}_std"] = round(float(vals.std()), 1) if len(vals) >= 2 else None
                                        for _suit in ["S", "H", "D", "C"]:
                                            c = f"SL_{_suit}_{_d}"
                                            if c in df.columns:
                                                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                                                r[f"SL_{_suit}"] = round(float(vals.mean()), 1) if len(vals) else None
                                        rows.append(r)
                                    return pd.DataFrame(rows).set_index("Direction")

                                def _agg_dd_tricks(df: pd.DataFrame) -> pd.DataFrame:
                                    """Aggregate DD tricks: 4 rows (declarer) × 5 strain cols (mean)."""
                                    rows = []
                                    for _decl in ["N", "E", "S", "W"]:
                                        r: dict[str, Any] = {"Declarer": _decl}
                                        for _s in _DD_STRAINS:
                                            c = f"DD_{_decl}_{_s}"
                                            if c in df.columns:
                                                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                                                r[_s] = round(float(vals.mean()), 1) if len(vals) else None
                                        rows.append(r)
                                    return pd.DataFrame(rows).set_index("Declarer")

                                with st.expander(
                                    f"Deal Stats: {_bid_label} (bt_index={_bt_key}, "
                                    f"{len(_deal_list)}/{_total_ct:,} deals)",
                                    expanded=False,
                                ):
                                    st.markdown(f"**Hand Stats ({len(_hs_df)} deals)**")
                                    st.dataframe(_hs_df, width="stretch", height=min(400, 35 + len(_hs_df) * 28))
                                    st.markdown("**Hand Stats (aggregated)**")
                                    st.dataframe(_agg_hand_stats(_hs_df), width="stretch")

                                    st.markdown(f"**DD Tricks ({len(_dd_df)} deals)**")
                                    st.dataframe(_dd_df, width="stretch", height=min(400, 35 + len(_dd_df) * 28))
                                    st.markdown("**DD Tricks (aggregated)**")
                                    st.dataframe(_agg_dd_tricks(_dd_df), width="stretch")

                                    if _hs_shape_df is not None and len(_hs_shape_df) > 0:
                                        st.markdown(
                                            f"**Hand Stats — {_active_dir} shape {_shape_label} "
                                            f"({len(_hs_shape_df)} deals)**"
                                        )
                                        st.dataframe(_hs_shape_df, width="stretch", height=min(400, 35 + len(_hs_shape_df) * 28))
                                        st.markdown(f"**Hand Stats — shape {_shape_label} (aggregated)**")
                                        st.dataframe(_agg_hand_stats(_hs_shape_df), width="stretch")
                                    if _dd_shape_df is not None and len(_dd_shape_df) > 0:
                                        st.markdown(
                                            f"**DD Tricks — {_active_dir} shape {_shape_label} "
                                            f"({len(_dd_shape_df)} deals)**"
                                        )
                                        st.dataframe(_dd_shape_df, width="stretch", height=min(400, 35 + len(_dd_shape_df) * 28))
                                        st.markdown(f"**DD Tricks — shape {_shape_label} (aggregated)**")
                                        st.dataframe(_agg_dd_tricks(_dd_shape_df), width="stretch")
                                    elif _active_dir and _active_shape:
                                        st.caption(f"No deals with {_active_dir} shape {_shape_label}")
                        except Exception as _e_computed:
                            st.caption(f"⚠️ Deal stats error: {_e_computed}")
                    else:
                        st.caption("Sampled per-bid deal stats not loaded (toggle on to fetch).")

                # Rejected bids grid (dead ends, 0 deals, empty criteria)
                if rejected_rows:
                    st.markdown(f"**⚠️ Rejected Bids ({len(rejected_rows)})**")
                    rejected_pdf = to_aggrid_safe_pandas(pl.DataFrame(rejected_rows))
                    # Configure Failure column
                    gb_rej = GridOptionsBuilder.from_dataframe(rejected_pdf)
                    gb_rej.configure_selection(selection_mode="single", use_checkbox=False)
                    gb_rej.configure_column("_idx", hide=True)
                    gb_rej.configure_column("_matches", hide=True)
                    gb_rej.configure_column("_rejected", hide=True)
                    if "Criteria_Count" in rejected_pdf.columns:
                        gb_rej.configure_column("Criteria_Count", hide=True)
                    if "_dd_col" in rejected_pdf.columns:
                        gb_rej.configure_column("_dd_col", hide=True)
                    if "Criteria_Count" in rejected_pdf.columns:
                        gb_rej.configure_column("Criteria_Count", hide=True)
                    # Column widths: ~8px/char + ~55px for filter icon and padding
                    gb_rej.configure_column("BT Index", width=120, minWidth=110)   # 8 chars
                    gb_rej.configure_column("Bid Num", width=115, minWidth=105)    # 7 chars
                    gb_rej.configure_column("Seat", width=90, minWidth=80)         # 4 chars
                    if pinned_deal:
                        gb_rej.configure_column("Direction", width=125, minWidth=115)  # 9 chars
                    gb_rej.configure_column("Bid", width=80, minWidth=70)          # 3 chars
                    if "Contract" in rejected_pdf.columns:
                        gb_rej.configure_column("Contract", width=110, minWidth=95)
                    if "can_complete" in rejected_pdf.columns:
                        gb_rej.configure_column("can_complete", width=130, minWidth=120, headerName="Can Complete")
                    gb_rej.configure_column("Matches", width=115, minWidth=105)    # 7 chars
                    gb_rej.configure_column("Deals", width=100, minWidth=90)       # 5 chars
                    gb_rej.configure_column("Avg_EV", width=105, minWidth=95, headerName="Avg EV")   # 6 chars
                    if "Avg_Par" in rejected_pdf.columns:
                        gb_rej.configure_column("Avg_Par", width=110, minWidth=100, headerName="Avg Par")
                    if "EV_Contract" in rejected_pdf.columns:
                        gb_rej.configure_column("EV_Contract", width=120, minWidth=110, headerName="EV Contract")
                    if "EV_Contract_Pre" in rejected_pdf.columns:
                        gb_rej.configure_column("EV_Contract_Pre", width=140, minWidth=125, headerName="EV Contract Pre")
                    gb_rej.configure_column("EV_NV", width=100, minWidth=90, headerName="EV NV")    # 5 chars
                    gb_rej.configure_column("EV_V", width=90, minWidth=80, headerName="EV V")       # 4 chars
                    if "Par_NV" in rejected_pdf.columns:
                        gb_rej.configure_column("Par_NV", width=100, minWidth=90, headerName="Par NV")
                    if "Par_V" in rejected_pdf.columns:
                        gb_rej.configure_column("Par_V", width=90, minWidth=80, headerName="Par V")
                    if "DD_Tricks" in rejected_pdf.columns:
                        gb_rej.configure_column("DD_Tricks", width=135, minWidth=120, headerName="DD_Tricks")
                    if "DD_Score_Contract" in rejected_pdf.columns:
                        gb_rej.configure_column("DD_Score_Contract", width=135, minWidth=120, headerName="DD Score")
                    gb_rej.configure_column("Failure", width=115, minWidth=105)    # 7 chars
                    gb_rej.configure_column("Criteria", flex=1, minWidth=120, tooltipField="Criteria")

                    # Display rules (match valid/invalid grids):
                    # - Pinned deal: show Avg_EV/Avg_Par, hide EV_NV/EV_V and Par_NV/Par_V
                    # - No pinned deal: hide Avg_EV/Avg_Par, show EV_NV/EV_V and Par_NV/Par_V
                    if pinned_deal:
                        gb_rej.configure_column("EV_NV", hide=True)
                        gb_rej.configure_column("EV_V", hide=True)
                        if "Par_NV" in rejected_pdf.columns:
                            gb_rej.configure_column("Par_NV", hide=True)
                        if "Par_V" in rejected_pdf.columns:
                            gb_rej.configure_column("Par_V", hide=True)
                    else:
                        gb_rej.configure_column("Avg_EV", hide=True)
                        if "Avg_Par" in rejected_pdf.columns:
                            gb_rej.configure_column("Avg_Par", hide=True)
                        if "EV_Contract" in rejected_pdf.columns:
                            gb_rej.configure_column("EV_Contract", hide=True)
                        if "EV_Contract_Pre" in rejected_pdf.columns:
                            gb_rej.configure_column("EV_Contract_Pre", hide=True)
                    gb_rej.configure_grid_options(rowHeight=25, headerHeight=25)
                    rejected_opts = gb_rej.build()
                    rejected_resp = AgGrid(
                        rejected_pdf,
                        gridOptions=rejected_opts,
                        height=min(275, 60 + len(rejected_rows) * 25),
                        update_on=["selectionChanged"],
                        key=f"auction_builder_rejected_grid_{current_seat}",
                        theme="balham",
                        allow_unsafe_jscode=True,
                        custom_css={
                            ".ag-row": {"cursor": "pointer", "background-color": "#fff3cd"},
                            ".ag-row-hover": {"background-color": "#ffe69c !important", "border-left": "3px solid #ffc107 !important"},
                            ".ag-row-selected": {"background-color": "#ffda6a !important"},
                        },
                    )
                    handle_bid_selection(rejected_resp, "rejected")
                
                # Invalid bids grid
                if invalid_rows:
                    st.markdown(f"**❌ Invalid Bids ({len(invalid_rows)})**")
                    invalid_pdf = to_aggrid_safe_pandas(pl.DataFrame(invalid_rows))
                    invalid_opts = build_bid_grid_options(invalid_pdf, apply_row_styling=False)
                    # Height: header (25) + rows (25 each) + buffer (35)
                    invalid_resp = AgGrid(
                        invalid_pdf,
                        gridOptions=invalid_opts,
                        height=min(375, 60 + len(invalid_rows) * 25),
                        update_on=["selectionChanged"],
                        key=f"auction_builder_invalid_grid_{current_seat}",
                        theme="balham",
                        allow_unsafe_jscode=True,
                        custom_css={
                            ".ag-row": {"cursor": "pointer", "background-color": "#f8d7da"},
                            ".ag-row-hover": {"background-color": "#f1b0b7 !important", "border-left": "3px solid #dc3545 !important"},
                            ".ag-row-selected": {"background-color": "#eb959f !important"},
                        },
                    )
                    handle_bid_selection(invalid_resp, "invalid")
                
                # Other rows (no match status - shouldn't happen when deal is pinned)
                if other_rows:
                    st.markdown(f"**Other Bids ({len(other_rows)})**")
                    other_pdf = to_aggrid_safe_pandas(pl.DataFrame(other_rows))
                    other_opts = build_bid_grid_options(other_pdf, apply_row_styling=False)
                    other_resp = AgGrid(
                        # Height: header (25) + rows (25 each) + buffer (10)
                        other_pdf,
                        gridOptions=other_opts,
                        height=min(300, 35 + len(other_rows) * 25),
                        update_on=["selectionChanged"],
                        key=f"auction_builder_other_grid_{current_seat}",
                        theme="balham",
                        allow_unsafe_jscode=True,
                        custom_css=grid_custom_css,
                    )
                    handle_bid_selection(other_resp, "other")
            elif use_two_dataframes and not show_failed_criteria:
                # No pinned deal: split into Available vs Rejected using the same rejection logic.
                available_rows = [r for r in bid_rows if not r.get("_rejected")]
                rejected_rows = [r for r in bid_rows if r.get("_rejected")]

                if available_rows:
                    st.markdown(f"**✅ Available Bids ({len(available_rows)})**")
                    avail_pdf = to_aggrid_safe_pandas(pl.DataFrame(available_rows))
                    avail_opts = build_bid_grid_options(avail_pdf, apply_row_styling=False)
                    avail_resp = AgGrid(
                        avail_pdf,
                        gridOptions=avail_opts,
                        height=min(350, 35 + len(available_rows) * 25),
                        update_on=["selectionChanged"],
                        key=f"auction_builder_available_grid_{current_seat}",
                        theme="balham",
                        allow_unsafe_jscode=True,
                        custom_css={
                            ".ag-row": {"cursor": "pointer", "background-color": "#d4edda"},
                            ".ag-row-hover": {"background-color": "#b8dfc4 !important", "border-left": "3px solid #28a745 !important"},
                            ".ag-row-selected": {"background-color": "#a3d4af !important"},
                        },
                    )
                    handle_bid_selection(avail_resp, "available")
                else:
                    st.warning("No available bids (all candidates are rejected by BT/matches/deals/criteria rules).")

                if rejected_rows:
                    st.markdown(f"**⚠️ Rejected Bids ({len(rejected_rows)})**")
                    rejected_pdf = to_aggrid_safe_pandas(pl.DataFrame(rejected_rows))

                    # Configure rejected grid with explicit Failure column visible
                    gb_rej = GridOptionsBuilder.from_dataframe(rejected_pdf)
                    gb_rej.configure_selection(selection_mode="single", use_checkbox=False)
                    gb_rej.configure_column("_idx", hide=True)
                    gb_rej.configure_column("_matches", hide=True)
                    gb_rej.configure_column("_rejected", hide=True)
                    gb_rej.configure_column("Bid Num", width=85, minWidth=85)
                    gb_rej.configure_column("Seat", width=60, minWidth=60)
                    gb_rej.configure_column("Bid", width=80, minWidth=80)
                    gb_rej.configure_column("Matches", width=80, minWidth=60)
                    gb_rej.configure_column("Deals", width=80, minWidth=60)
                    gb_rej.configure_column("Avg_EV", width=65, minWidth=55, headerName="Avg EV")
                    gb_rej.configure_column("EV_NV", width=65, minWidth=55, headerName="EV NV")
                    gb_rej.configure_column("EV_V", width=55, minWidth=50, headerName="EV V")
                    gb_rej.configure_column("Failure", width=160, minWidth=140)
                    gb_rej.configure_column("Criteria", flex=1, minWidth=100, tooltipField="Criteria")
                    gb_rej.configure_column("Categories", flex=1, minWidth=140, tooltipField="Categories")

                    # No pinned deal: hide Avg_EV, show EV_NV/EV_V (same as main grid rules)
                    gb_rej.configure_column("Avg_EV", hide=True)
                    gb_rej.configure_grid_options(rowHeight=25, headerHeight=25)
                    rejected_opts = gb_rej.build()

                    rejected_resp = AgGrid(
                        rejected_pdf,
                        gridOptions=rejected_opts,
                        height=min(325, 60 + len(rejected_rows) * 25),
                        update_on=["selectionChanged"],
                        key=f"auction_builder_rejected_grid_{current_seat}",
                        theme="balham",
                        allow_unsafe_jscode=True,
                        custom_css={
                            ".ag-row": {"cursor": "pointer", "background-color": "#fff3cd"},
                            ".ag-row-hover": {"background-color": "#ffe69c !important", "border-left": "3px solid #ffc107 !important"},
                            ".ag-row-selected": {"background-color": "#ffda6a !important"},
                        },
                    )
                    handle_bid_selection(rejected_resp, "rejected_unpinned")
            else:
                # Single DataFrame mode
                if show_failed_criteria and (valid_count > 0 or invalid_count > 0 or rejected_count > 0):
                    st.caption(f"✅ Valid: {valid_count} | ⚠️ Rejected: {rejected_count} | ❌ Invalid: {invalid_count}")
                
                grid_options = build_bid_grid_options(pandas_df)
                grid_response = AgGrid(
                    pandas_df,
                    gridOptions=grid_options,
                    height=min(400, 35 + len(bid_rows) * 25),
                    update_on=["selectionChanged"],
                    key=f"auction_builder_bids_grid_{current_seat}",
                    theme="balham",
                    allow_unsafe_jscode=True,
                    custom_css=grid_custom_css,
                )
                handle_bid_selection(grid_response, "generic")
    
    # Summary DataFrame
    if current_path or pinned_deal:
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
                        # Preserve client-side flags that shouldn't be invalidated by server rehydrate.
                        old_path = current_path
                        for j in range(min(len(old_path), len(new_path))):
                            if old_path[j].get("_pass_no_criteria"):
                                new_path[j]["_pass_no_criteria"] = True
                            if old_path[j].get("_pass_valid_selection"):
                                new_path[j]["_pass_valid_selection"] = True
                        st.session_state.auction_builder_path = new_path
                        current_path = new_path
                    rehydrate_elapsed_ms = data.get("_client_elapsed_ms") or data.get("elapsed_ms")
                except Exception:
                    pass

        # Include bt_index from last step in header (use 'is not None' since bt_index=0 is valid)
        summary_bt_index = current_path[-1].get("bt_index") if current_path else None
        summary_header = (
            f"📋 Completed Auction Summary (bt_index: {summary_bt_index})"
            if summary_bt_index is not None
            else "📋 Completed Auction Summary"
        )
        # Title is rendered later together with the completion DataFrame
        # inside completed_summary_placeholder (see "Completed Auction Summary" grid section below).
        _completed_summary_header = summary_header
        _completed_summary_rehydrate_ms = rehydrate_elapsed_ms

        # Per-step sampled-deal stats cache.
        # Populated when the user clicks a row (Matching Deals section computes sample stats).
        # Keep only the most recent selection to avoid accumulating lots of cached stats.
        step_stats_map_all: dict[str, dict[str, Any]] = st.session_state.setdefault("_auction_builder_step_stats", {})
        
        # Helper to evaluate criteria against pinned deal
        def evaluate_criteria_for_pinned(criteria_list: list, seat: int, dealer: str, deal: dict) -> tuple[bool, list[str]]:
            """Evaluate criteria against a pinned deal. Returns (all_pass, failed_criteria).
            
            Uses server-side bitmap evaluation when _row_idx is available for accurate
            evaluation of all criteria types. PBN deals cannot be evaluated.
            """
            if not deal:
                return True, []
            
            # WORKAROUND: Invalidate bids with underspecified criteria (missing HCP and Total_Points)
            if _is_underspecified_criteria(criteria_list):
                return False, ["underspecified (no HCP/Total_Points)"]

            if not criteria_list:
                return True, []
            
            dealer_s = str(dealer).upper()
            row_idx = deal.get("_row_idx")
            
            # PBN deals don't have _row_idx - cannot evaluate criteria without enrichment
            if row_idx is None:
                return True, ["(PBN deal - criteria evaluation not available)"]
            
            # Use server-side bitmap evaluation (accurate for all criteria)
            try:
                cache_key = ("deal_criteria_eval_summary", int(row_idx), dealer_s, seat, tuple(criteria_list))
                eval_cache = st.session_state.setdefault("_deal_criteria_eval_cache", {})
                if cache_key in eval_cache:
                    cached = eval_cache[cache_key]
                    return cached["passes"], cached["failed"]
                
                payload = {
                    "deal_row_idx": int(row_idx),
                    "dealer": dealer_s,
                    "checks": [{"seat": seat, "criteria": list(criteria_list)}],
                }
                data = api_post("/deal-criteria-eval-batch", payload, timeout=10)
                results = data.get("results", [])
                if results:
                    r = results[0]
                    failed_list = r.get("failed", [])
                    untracked = r.get("untracked", [])
                    annotated_failed = []
                    for f in failed_list:
                        annotated = annotate_criterion_with_value(str(f), dealer_s, seat, deal)
                        annotated_failed.append(annotated)
                    for u in untracked:
                        annotated_failed.append(f"UNTRACKED: {u}")
                    passes = len(failed_list) == 0 and len(untracked) == 0
                    eval_cache[cache_key] = {"passes": passes, "failed": annotated_failed}
                    return passes, annotated_failed
            except Exception as e:
                return True, [f"(Server error: {e})"]
            
            return True, []
        
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
            _fetch_auction_pattern_counts(
                missing_partials, deals_counts, pattern_style="completed_only", timeout=15
            )
            st.session_state["_deals_counts_elapsed_ms"] = (time.perf_counter() - t0_counts) * 1000
            st.session_state["_deals_counts_steps"] = len(missing_partials)

        # ------------------------------------------------------------------
        # Sampled-deal stats for ALL steps (on demand, cached).
        #
        # This uses /sample-deals-by-auction-pattern and computes NV/V splits
        # from the sampled deals. This is heavier than counts, but is now
        # intentionally enabled for every step in the Current Auction grid.
        # ------------------------------------------------------------------
        step_stats_map: dict[str, dict[str, Any]] = st.session_state.setdefault("_auction_builder_step_stats", {})

        # Keep cache small: only stats for the current path.
        current_partials = ["-".join([s["bid"] for s in current_path[: i + 1]]) for i in range(len(current_path))]
        for k in list(step_stats_map_all.keys()):
            if k not in current_partials:
                del step_stats_map_all[k]

        def _calc_std(values: list[float]) -> float | None:
            if len(values) < 2:
                return None
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return round(variance ** 0.5, 1)

        def _compute_sampled_stats_for_partial(partial: str) -> dict[str, Any]:
            """Compute sampled-deal stats for a partial auction prefix (or exact complete auction)."""
            auction_upper = str(partial or "").upper()
            if auction_upper.endswith("-P-P-P"):
                arena_pattern = f"^{auction_upper}$"
            else:
                arena_pattern = f"^{auction_upper}.*-P-P-P$"
            deals_data = api_post(
                "/sample-deals-by-auction-pattern",
                {"pattern": arena_pattern, "sample_size": int(max_matching_deals), "seed": int(seed)},
                timeout=30,
            )
            sample_deals = deals_data.get("deals", []) or []

            scores_nv: list[float] = []
            scores_v: list[float] = []
            par_scores_nv: list[float] = []
            par_scores_v: list[float] = []

            for deal in sample_deals:
                vul = str(deal.get("Vul", "")).upper()
                is_vul = vul not in ("NONE", "-", "", "O")  # O = None in some formats
                score = deal.get("Score")
                par = deal.get("ParScore")

                if score is not None:
                    try:
                        s = float(score)
                        (scores_v if is_vul else scores_nv).append(s)
                    except (ValueError, TypeError):
                        pass
                if par is not None:
                    try:
                        p = float(par)
                        (par_scores_v if is_vul else par_scores_nv).append(p)
                    except (ValueError, TypeError):
                        pass

            return {
                "_sample_size": int(max_matching_deals),
                "_seed": int(seed),
                "matches_nv": len(scores_nv) if scores_nv else None,
                "matches_v": len(scores_v) if scores_v else None,
                "avg_par_nv": round(sum(par_scores_nv) / len(par_scores_nv), 1) if par_scores_nv else None,
                "avg_par_v": round(sum(par_scores_v) / len(par_scores_v), 1) if par_scores_v else None,
                "ev_nv": round(sum(scores_nv) / len(scores_nv), 1) if scores_nv else None,
                "ev_v": round(sum(scores_v) / len(scores_v), 1) if scores_v else None,
                "ev_std_nv": _calc_std(scores_nv),
                "ev_std_v": _calc_std(scores_v),
            }

        # Compute any missing stats (or stale due to settings change)
        missing_stats = [
            p
            for p in current_partials
            if (
                p not in step_stats_map_all
                or step_stats_map_all[p].get("_sample_size") != int(max_matching_deals)
                or step_stats_map_all[p].get("_seed") != int(seed)
            )
        ]
        if missing_stats:
            with st.spinner(f"Computing sampled-deal stats for {len(missing_stats)} step(s)..."):
                prog = st.progress(0)
                try:
                    for j, p in enumerate(missing_stats):
                        step_stats_map_all[p] = _compute_sampled_stats_for_partial(p)
                        prog.progress(int(((j + 1) / max(1, len(missing_stats))) * 100))
                finally:
                    prog.empty()

        for i, step in enumerate(current_path):
            bid_num = i + 1
            seat_1_to_4 = ((bid_num - 1) % 4) + 1
            # Store tooltip text but cap it (otherwise Auction Summary can take 20s+ to render/serialize).
            criteria_str = _join_tooltip(step.get("agg_expr", []))
            # Build partial auction up to this step for deals count lookup
            partial_auction = "-".join([s["bid"] for s in current_path[:bid_num]])
            # Matches = pattern-based count (deals where actual auction matches)
            matches_count = deals_counts.get(partial_auction, 0)
            # Deals = criteria-based count (deals that qualify based on hand evaluation)
            deals_count = step.get("matching_deal_count")
            if deals_count is None:
                deals_count = matches_count  # Fall back to pattern-based if not available
            # Get NV/V split EV from the step (populated at selection time)
            avg_ev_nv = step.get("avg_ev_nv")
            avg_ev_v = step.get("avg_ev_v")
            # For Par, use sampled stats (computed from matched deals for this partial auction).
            step_stats = step_stats_map_all.get(partial_auction, {}) if isinstance(step_stats_map_all, dict) else {}
            avg_par_nv = step_stats.get("avg_par_nv")
            avg_par_v = step_stats.get("avg_par_v")
            deal_par = pinned_deal.get("ParScore", pinned_deal.get("Par_Score")) if pinned_deal else None
            deal_ev = pinned_deal.get("EV_Score_Declarer", pinned_deal.get("EV_Score")) if pinned_deal else None
            avg_ev: float | None = None
            avg_par: float | None = None
            ev_sign = -1.0 if seat_1_to_4 in (2, 4) else 1.0
            bid_dir: str | None = None
            if pinned_deal:
                try:
                    dealer = str(pinned_deal.get("Dealer", "N")).upper()
                    vul = str(pinned_deal.get("Vul", pinned_deal.get("Vulnerability", ""))).upper()
                    directions = ["N", "E", "S", "W"]
                    dealer_idx = directions.index(dealer) if dealer in directions else 0
                    bid_dir = directions[(dealer_idx + (seat_1_to_4 - 1)) % 4]
                    ns_vul = vul in ("N_S", "NS", "BOTH", "ALL")
                    ew_vul = vul in ("E_W", "EW", "BOTH", "ALL")
                    is_vul = (bid_dir in ("N", "S") and ns_vul) or (bid_dir in ("E", "W") and ew_vul)
                    chosen = avg_ev_v if is_vul else avg_ev_nv
                    avg_ev = round(ev_sign * float(chosen), 1) if chosen is not None else None
                    chosen_par = avg_par_v if is_vul else avg_par_nv
                    avg_par = round(ev_sign * float(chosen_par), 1) if chosen_par is not None else None
                except Exception:
                    avg_ev = None
                    avg_par = None
            row = {
                "Bid Num": bid_num,
                "Direction": bid_dir if pinned_deal else None,
                "Seat": seat_1_to_4,
                "Bid": step["bid"],
                # Keep the exact partial auction used for Matches/Deals computations.
                # This avoids relying on Bid Num reconstruction when the grid is sorted/filtered.
                "_partial_auction": partial_auction,
                "BT Index": step.get("bt_index"),
                "Matches": matches_count,
                "Deals": deals_count,
                "Avg_EV": avg_ev,
                "EV_NV": round(ev_sign * float(avg_ev_nv), 1) if avg_ev_nv is not None else None,
                "EV_V": round(ev_sign * float(avg_ev_v), 1) if avg_ev_v is not None else None,
                "Avg_Par": avg_par,
                "Par_NV": round(ev_sign * float(avg_par_nv), 1) if avg_par_nv is not None else None,
                "Par_V": round(ev_sign * float(avg_par_v), 1) if avg_par_v is not None else None,
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
                # Auction Summary uses the stored path. We normalize that path after each click
                # via /resolve-auction-path, which aligns criteria to dealer-relative seats already.
                # So evaluate using the actual dealer and the displayed seat.
                dealer = pinned_deal.get("Dealer", "N")
                # If this step was a client-side "Pass with no criteria" OR was selected from a valid-bids
                # grid (Best Bids Ranked by Cheating Model), treat it as passing to maintain consistency.
                # Also apply the "always_valid_pass" setting: if enabled and this is a Pass bid,
                # skip criteria evaluation to match the behavior when manually selecting bids.
                is_pass_bid = str(step.get("bid", "")).strip().upper() in ("P", "PASS")
                pass_always_valid = bool(is_pass_bid and st.session_state.get("always_valid_pass", True))
                skip_eval = step.get("_pass_no_criteria") or step.get("_pass_valid_selection") or pass_always_valid
                criteria_list = [] if skip_eval else step.get("agg_expr", [])
                passes, failed = evaluate_criteria_for_pinned(criteria_list, seat_1_to_4, dealer, pinned_deal)
                row["_passes"] = passes  # Hidden column for row styling
                if passes:
                    row["Pinned"] = "✅"
                else:
                    row["Pinned"] = "❌"
                    row["Failed_Criteria"] = "; ".join(failed)  # Show all failures
                    pinned_all_pass = False
            else:
                # When no pinned deal, color based on bt_index presence and underspecified status
                # Green = has bt_index (in BT) and not underspecified, Red = no bt_index or underspecified
                row["_passes"] = (step.get("bt_index") is not None) and not _is_underspecified_criteria(step.get("agg_expr", []))
            
            # Sampled-deal stats (computed and cached for all steps).
            row["Matches_NV"] = None
            row["Matches_V"] = None
            row["Avg Par_NV"] = None
            row["Avg Par_V"] = None
            row["EV at Bid_NV"] = None
            row["EV at Bid_V"] = None
            row["EV Std_NV"] = None
            row["EV Std_V"] = None
            stats_for_partial = step_stats_map_all.get(partial_auction)
            if isinstance(stats_for_partial, dict):
                row["Matches_NV"] = stats_for_partial.get("matches_nv")
                row["Matches_V"] = stats_for_partial.get("matches_v")
                row["Avg Par_NV"] = stats_for_partial.get("avg_par_nv")
                row["Avg Par_V"] = stats_for_partial.get("avg_par_v")
                row["EV at Bid_NV"] = stats_for_partial.get("ev_nv")
                row["EV at Bid_V"] = stats_for_partial.get("ev_v")
                row["EV Std_NV"] = stats_for_partial.get("ev_std_nv")
                row["EV Std_V"] = stats_for_partial.get("ev_std_v")
            summary_rows.append(row)
        
        # Show auction complete status and pinned deal match summary
        # NEW: Show comparison summary even for incomplete auctions if a pinned deal is present.
        if is_complete or pinned_deal:
            def _final_contract_from_calls(calls: list[str]) -> str:
                """Return final contract string like '4S', '4Sx', '4Sxx', or 'Passed out'."""
                last_contract: str | None = None
                dbl_state = 0  # 0=none, 1=x, 2=xx
                for raw in calls:
                    t = str(raw or "").strip().upper()
                    if not t:
                        continue
                    # Contract bids
                    if len(t) == 2 and t[0] in "1234567" and t[1] in "CDHSN":
                        last_contract = t
                        dbl_state = 0
                        continue
                    # Doubles / redoubles (this app uses D/R; accept X/XX too)
                    if t in ("D", "X"):
                        if last_contract and dbl_state == 0:
                            dbl_state = 1
                        continue
                    if t in ("R", "XX"):
                        if last_contract and dbl_state == 1:
                            dbl_state = 2
                        continue
                    # Ignore P and other tokens
                if not last_contract:
                    return "Passed out"
                if dbl_state == 1:
                    return f"{last_contract}x"
                if dbl_state == 2:
                    return f"{last_contract}xx"
                return last_contract

            def _last_contract_step_index(calls: list[str]) -> int | None:
                """Index of last non-pass action that defines the final contract (bid/double/redouble)."""
                # Find the last contract bid, then include subsequent D/R up to the end.
                last_bid_i: int | None = None
                for i, raw in enumerate(calls):
                    t = str(raw or "").strip().upper()
                    if len(t) == 2 and t[0] in "1234567" and t[1] in "CDHSN":
                        last_bid_i = i
                if last_bid_i is None:
                    return None
                # Walk forward until next contract bid (none, since we picked last) – so last defining action
                # is the last D/R after last_bid_i (if any).
                last_def = last_bid_i
                for j in range(last_bid_i + 1, len(calls)):
                    t = str(calls[j] or "").strip().upper()
                    if t in ("D", "X", "R", "XX"):
                        last_def = j
                return last_def

            def _seat_dir_from_dealer(dealer: str, seat_1_to_4: int) -> str:
                """Map seat 1..4 (relative to dealer) to absolute direction letter."""
                directions = ["N", "E", "S", "W"]
                d = str(dealer or "N").upper()
                try:
                    dealer_idx = directions.index(d)
                except Exception:
                    dealer_idx = 0
                s = int(seat_1_to_4)
                return directions[(dealer_idx + (s - 1)) % 4]

            def _declarer_seat_from_calls(calls: list[str]) -> int | None:
                """Compute declarer seat (1..4 relative to dealer) from auction calls.

                Declarer is the first player from the declaring partnership who bid the final strain.
                Returns None if no contract.
                """
                if not calls:
                    return None
                # Identify last contract bid (level+strain); doubles don't affect declarer.
                last_contract_i: int | None = None
                last_strain: str | None = None
                for i, raw in enumerate(calls):
                    t = str(raw or "").strip().upper()
                    if len(t) == 2 and t[0] in "1234567" and t[1] in "CDHSN":
                        last_contract_i = i
                        last_strain = t[1]
                if last_contract_i is None or not last_strain:
                    return None
                # Determine which partnership won (odd seats 1/3 vs even seats 2/4 relative to dealer)
                contract_seat = (last_contract_i % 4) + 1
                contract_parity = contract_seat % 2
                # Find first bid of that strain by that partnership
                declarer_seat: int | None = None
                for i, raw in enumerate(calls[: last_contract_i + 1]):
                    t = str(raw or "").strip().upper()
                    if len(t) == 2 and t[0] in "1234567" and t[1] == last_strain:
                        seat_i = (i % 4) + 1
                        if (seat_i % 2) == contract_parity:
                            declarer_seat = seat_i
                            break
                if declarer_seat is None:
                    declarer_seat = contract_seat
                return declarer_seat

            def _declarer_direction_from_calls(calls: list[str], dealer: str | None) -> str | None:
                """Compute declarer direction (N/E/S/W) from auction calls + dealer."""
                if not dealer:
                    return None
                seat = _declarer_seat_from_calls(calls)
                if seat is None:
                    return None
                return _seat_dir_from_dealer(str(dealer), int(seat))

            auction_calls = [str(s.get("bid", "")).strip().upper() for s in (current_path or []) if str(s.get("bid", "")).strip()]
            auction_text = "-".join(auction_calls) if auction_calls else ""
            final_contract = _final_contract_from_calls(auction_calls)
            deal_ev_val = pinned_deal.get("EV_Score_Declarer", pinned_deal.get("EV_Score")) if pinned_deal else None
            try:
                deal_ev_num = float(deal_ev_val) if deal_ev_val is not None else None
            except Exception:
                deal_ev_num = None
            declarer_dir: str | None = None
            declarer_seat: int | None = _declarer_seat_from_calls(auction_calls)
            declarer_pair: str | None = None
            if declarer_seat is not None:
                declarer_pair = "NS" if (int(declarer_seat) % 2) == 1 else "EW"
            if pinned_deal:
                declarer_dir = _declarer_direction_from_calls(auction_calls, str(pinned_deal.get("Dealer", "N")))
            # Avg_EV: use the step that sets the final contract (last bid/double/redouble), not the final pass.
            avg_ev_final: float | None = None
            avg_ev_final_nv: float | None = None
            avg_ev_final_v: float | None = None
            try:
                idx = _last_contract_step_index(auction_calls)
                if idx is not None and 0 <= idx < len(summary_rows):
                    v = summary_rows[idx].get("Avg_EV")
                    avg_ev_final = float(v) if v is not None else None
                    v_nv = summary_rows[idx].get("EV_NV")
                    v_v = summary_rows[idx].get("EV_V")
                    avg_ev_final_nv = float(v_nv) if v_nv is not None else None
                    avg_ev_final_v = float(v_v) if v_v is not None else None
            except Exception:
                avg_ev_final = None
                avg_ev_final_nv = None
                avg_ev_final_v = None

            def _fmt_num(x: float | None) -> str:
                return f"{x:.1f}" if isinstance(x, (int, float)) else "—"

            # Double-dummy score and tricks for the FINAL contract (when pinned deal is available)
            dd_score_final: int | None = None
            dd_tricks_final: int | None = None
            if pinned_deal and auction_text and final_contract != "Passed out":
                try:
                    dealer = str(pinned_deal.get("Dealer", "N")).upper()
                    # DD score: contract-level score columns are DD_Score_{level}{strain}_{dir}
                    dd_score_val = get_dd_score_for_auction(auction_text, dealer, pinned_deal)
                    dd_score_final = int(dd_score_val) if dd_score_val is not None else None
                    # DD tricks: raw trick columns are DD_{dir}_{strain}
                    dd_tricks_val = get_dd_tricks_for_auction(auction_text, dealer, pinned_deal)
                    dd_tricks_final = int(dd_tricks_val) if dd_tricks_val is not None else None
                except Exception:
                    dd_score_final = None
                    dd_tricks_final = None

            if pinned_deal:
                actual_auction = pinned_deal.get("bid") or pinned_deal.get("Actual_Auction")
                dealer = str(pinned_deal.get("Dealer", "N")).upper()
                
                # Compute actual bid, DD values, EV, and bt_index from the pinned deal's actual auction
                actual_bid: str | None = None
                actual_dd_tricks: int | None = None
                actual_dd_score: int | None = None
                actual_declarer: str | None = None
                actual_ev: float | None = None
                actual_bt_index: int | None = None
                if actual_auction:
                    try:
                        actual_calls = [c.strip() for c in str(actual_auction).split("-") if c.strip()]
                        actual_bid = _final_contract_from_calls(actual_calls)
                        actual_declarer = _declarer_direction_from_calls(actual_calls, dealer)
                        # DD values for actual contract
                        actual_dd_score_val = get_dd_score_for_auction(str(actual_auction), dealer, pinned_deal)
                        actual_dd_score = int(actual_dd_score_val) if actual_dd_score_val is not None else None
                        actual_dd_tricks_val = get_dd_tricks_for_auction(str(actual_auction), dealer, pinned_deal)
                        actual_dd_tricks = int(actual_dd_tricks_val) if actual_dd_tricks_val is not None else None
                        # EV for actual contract
                        actual_ev_val = get_ev_for_auction(str(actual_auction), dealer, pinned_deal)
                        if actual_ev_val is not None:
                            actual_ev = round(float(actual_ev_val), 0)
                        else:
                            # Fallback: use the deal-level EV metric if per-contract EV lookup is missing.
                            # This is typically declarer-relative; we will convert to EV_NS downstream
                            # using the computed actual_declarer direction.
                            deal_ev_fallback = pinned_deal.get("EV_Score_Declarer", pinned_deal.get("EV_Score"))
                            try:
                                actual_ev = round(float(deal_ev_fallback), 0) if deal_ev_fallback is not None else None
                            except Exception:
                                actual_ev = None
                    except Exception:
                        pass
                    # Get bt_index for actual auction via API
                    try:
                        actual_path_data = api_post("/resolve-auction-path", {"auction": str(actual_auction)}, timeout=5)
                        actual_path = actual_path_data.get("path", [])
                        if actual_path:
                            actual_bt_index = actual_path[-1].get("bt_index")
                    except Exception:
                        pass
                
                # Compute EV for built auction
                built_ev: float | None = None
                if auction_text:
                    try:
                        built_ev_val = get_ev_for_auction(str(auction_text), dealer, pinned_deal)
                        built_ev = round(float(built_ev_val), 0) if built_ev_val is not None else None
                    except Exception:
                        pass

                # Compute Cheater Model row (greedy completion) when available
                best_auction_text: str | None = None
                best_bt_index: int | None = None
                best_bid: str | None = None
                best_declarer: str | None = None
                best_dd_tricks: int | None = None
                best_dd_score: int | None = None
                best_ev: float | None = None

                # Ensure Cheater Model appears even if user never clicked the suggested-bids grid.
                # (That UI path normally computes/caches `best_auc_ss_key`.)
                try:
                    cached_best_auc = st.session_state.get(best_auc_ss_key)
                except Exception:
                    cached_best_auc = None
                if not (isinstance(cached_best_auc, str) and cached_best_auc.strip()):
                    # Prefer server-side implementation (more reliable; doesn't depend on UI grids being clicked).
                    with st.spinner("Computing Cheater Model (greedy completion)..."):
                        best_complete: str | None = None
                        try:
                            row_idx = pinned_deal.get("_row_idx")
                            payload = {
                                "auction_prefix": "",
                                "deal_row_idx": int(row_idx) if row_idx is not None else None,
                                "seed": int(seed) if isinstance(seed, (int, float)) else 42,
                                "max_depth": 40,
                                "permissive_pass": bool(st.session_state.get("always_valid_pass", True)),
                            }
                            resp = api_post("/greedy-model-path", payload, timeout=30)
                            gp = resp.get("greedy_path")
                            best_complete = str(gp).strip() if gp is not None else None
                            if not best_complete:
                                best_complete = None
                        except Exception:
                            best_complete = None

                        if isinstance(best_complete, str) and best_complete.strip():
                            try:
                                st.session_state[best_auc_ss_key] = best_complete.strip()
                            except Exception:
                                pass
                            # Also update the placeholder at the top of the page, if present.
                            try:
                                best_auction_placeholder.empty()
                            except Exception:
                                pass
                try:
                    best_auction_text = st.session_state.get(best_auc_ss_key)
                    if isinstance(best_auction_text, str):
                        best_auction_text = best_auction_text.strip()
                    if not best_auction_text:
                        best_auction_text = None
                except Exception:
                    best_auction_text = None
                if best_auction_text:
                    try:
                        best_calls = [c.strip() for c in str(best_auction_text).split("-") if c.strip()]
                        best_bid = _final_contract_from_calls(best_calls)
                        best_declarer = _declarer_direction_from_calls(best_calls, dealer)
                        best_dd_score_val = get_dd_score_for_auction(str(best_auction_text), dealer, pinned_deal)
                        best_dd_score = int(best_dd_score_val) if best_dd_score_val is not None else None
                        best_dd_tricks_val = get_dd_tricks_for_auction(str(best_auction_text), dealer, pinned_deal)
                        best_dd_tricks = int(best_dd_tricks_val) if best_dd_tricks_val is not None else None
                        best_ev_val = get_ev_for_auction(str(best_auction_text), dealer, pinned_deal)
                        best_ev = round(float(best_ev_val), 0) if best_ev_val is not None else None
                    except Exception:
                        pass
                    try:
                        best_path_data = api_post("/resolve-auction-path", {"auction": str(best_auction_text)}, timeout=5)
                        best_path = best_path_data.get("path", [])
                        if best_path:
                            best_bt_index = best_path[-1].get("bt_index")
                    except Exception:
                        pass

                # Two-row DataFrame: Actual vs Built
                # Get bt_index for built auction from last step of current_path
                built_bt_index = current_path[-1].get("bt_index") if current_path else None

                def _to_ns(val: int | float | None, declarer_dir: str | None) -> int | float | None:
                    """Convert declarer-relative score/EV to NS-relative (NS positive)."""
                    if val is None:
                        return None
                    d = str(declarer_dir or "").upper()
                    sign = -1 if d in ("E", "W") else 1
                    try:
                        # Preserve int-ness when possible
                        if isinstance(val, int):
                            return int(sign * val)
                        return float(sign * float(val))
                    except Exception:
                        return None

                def _score_ns_from_auction_and_dd_tricks(
                    auction: str | None,
                    dealer_dir: str,
                    dd_tricks: int | None,
                    vul_text: str | None,
                ) -> int | None:
                    """Compute Score_NS from auction + DD tricks (NS-positive convention)."""
                    if not auction:
                        return None
                    if dd_tricks is None:
                        return None
                    auc = str(auction).strip().upper()
                    if not auc:
                        return None
                    # Passed out => 0
                    toks = [t.strip().upper() for t in auc.split("-") if t.strip()]
                    if toks and all(t == "P" for t in toks):
                        return 0

                    c = parse_contract_from_auction(auc)
                    if not c:
                        return None
                    level_i, strain_i, pos_i = c
                    try:
                        # Determine doubling/redoubling state by scanning calls after last contract bid.
                        is_doubled = False
                        is_redoubled = False
                        for bid in toks[int(pos_i) + 1 :]:
                            if bid in ("X", "D", "DBL", "DOUBLE"):
                                is_doubled = True
                                is_redoubled = False
                            elif bid in ("XX", "R", "RDBL", "REDOUBLE"):
                                is_redoubled = True
                        dbl0 = 2 if is_redoubled else (1 if is_doubled else 0)
                    except Exception:
                        dbl0 = 0

                    declarer = get_declarer_for_auction(auc, dealer_dir)
                    if not declarer:
                        return None
                    declarer_dir = str(declarer).upper()

                    # Vulnerability: decide declarer's vulnerability from board vul.
                    vul = str(vul_text or "").strip().upper()
                    ns_vul = vul in ("N_S", "NS", "BOTH", "ALL")
                    ew_vul = vul in ("E_W", "EW", "BOTH", "ALL")
                    is_ns = declarer_dir in ("N", "S")
                    is_vul = (is_ns and ns_vul) or ((not is_ns) and ew_vul)

                    # Result: over/under tricks relative to contract.
                    try:
                        result_val = int(dd_tricks) - (int(level_i) + 6)
                    except Exception:
                        return None

                    # mlBridgeLib.score expects:
                    # - level: 0..6 (contract level - 1)
                    # - suit: 0..4 (CDHSN)
                    # - double: 0..2 (''/X/XX)
                    # - declarer: 0..3 (N/E/S/W index)
                    # - vulnerability: 0..1 (is declarer vul)
                    # - result: -13..6 (over/under tricks)
                    try:
                        lvl0 = int(level_i) - 1
                        suit0 = mlBridgeLib.StrainSymToValue(str(strain_i).upper())
                        decl0 = "NSEW".index(declarer_dir)
                        vul0 = 1 if is_vul else 0
                        declarer_score = int(
                            mlBridgeLib.score(
                                lvl0,
                                suit0,
                                int(dbl0),
                                int(decl0),
                                int(vul0),
                                int(result_val),
                                declarer_score=True,
                            )
                        )
                    except Exception:
                        return None

                    # Convert to NS-positive.
                    return (-declarer_score) if declarer_dir in ("E", "W") else declarer_score

                def _result_from_auction_and_dd_tricks(
                    auction: str | None,
                    dd_tricks: int | None,
                ) -> int | None:
                    """Return over/under tricks (e.g., +1, 0, -2) from auction + DD tricks."""
                    if not auction:
                        return None
                    if dd_tricks is None:
                        return None
                    auc = str(auction).strip().upper()
                    if not auc:
                        return None
                    toks = [t.strip().upper() for t in auc.split("-") if t.strip()]
                    # Passed out => 0 tricks delta
                    if toks and all(t == "P" for t in toks):
                        return 0
                    c = parse_contract_from_auction(auc)
                    if not c:
                        return None
                    level_i, _strain_i, _pos_i = c
                    try:
                        return int(dd_tricks) - (int(level_i) + 6)
                    except Exception:
                        return None

                dd_score_final_ns: int | None = None
                try:
                    v = _to_ns(dd_score_final, declarer_dir)
                    dd_score_final_ns = int(v) if v is not None else None
                except Exception:
                    dd_score_final_ns = None

                actual_dd_score_ns: int | None = None
                try:
                    v = _to_ns(actual_dd_score, actual_declarer)
                    actual_dd_score_ns = int(v) if v is not None else None
                except Exception:
                    actual_dd_score_ns = None

                built_ev_ns: float | None = None
                try:
                    v = _to_ns(built_ev, declarer_dir)
                    built_ev_ns = float(v) if v is not None else None
                except Exception:
                    built_ev_ns = None

                actual_ev_ns: float | None = None
                try:
                    v = _to_ns(actual_ev, actual_declarer)
                    actual_ev_ns = float(v) if v is not None else None
                except Exception:
                    actual_ev_ns = None

                def _score_diff_to_imps(diff: int | float | None) -> int | None:
                    """Convert score diff (NS points) to IMPs (NS-positive)."""
                    if diff is None:
                        return None
                    try:
                        d = int(round(float(diff)))
                    except Exception:
                        return None
                    sign = -1 if d < 0 else 1
                    ad = abs(d)
                    # Standard WBF IMP table thresholds (approx; gaps don't matter in practice).
                    # 0-19 => 0, 20-49 => 1, 50-89 => 2, ...
                    thresholds: list[tuple[int, int]] = [
                        (20, 1),
                        (50, 2),
                        (90, 3),
                        (130, 4),
                        (170, 5),
                        (220, 6),
                        (270, 7),
                        (320, 8),
                        (370, 9),
                        (430, 10),
                        (500, 11),
                        (600, 12),
                        (750, 13),
                        (900, 14),
                        (1100, 15),
                        (1300, 16),
                        (1500, 17),
                        (1750, 18),
                        (2000, 19),
                        (2250, 20),
                        (2500, 21),
                        (3000, 22),
                        (3500, 23),
                    ]
                    imp = 0
                    for lo, val in thresholds:
                        if ad >= lo:
                            imp = val
                        else:
                            break
                    if ad >= 4000:
                        imp = 24
                    return int(sign * imp)

                def _pinned_passes_criteria(criteria_list: list[Any], seat_1_to_4: int) -> bool:
                    """True iff pinned deal passes all criteria at this seat (dealer-relative)."""
                    if not pinned_deal:
                        return True
                    if not criteria_list:
                        return True
                    row_idx = pinned_deal.get("_row_idx")
                    if row_idx is None:
                        # PBN/LIN pinned deals don't support bitmap criteria eval yet.
                        return True
                    dealer_actual = str(pinned_deal.get("Dealer", "N")).upper()
                    cache_key = ("deal_criteria_eval_ai_model", int(row_idx), dealer_actual, int(seat_1_to_4), tuple(criteria_list))
                    eval_cache = st.session_state.setdefault("_deal_criteria_eval_cache", {})
                    if cache_key in eval_cache:
                        return bool(eval_cache[cache_key].get("passes", True))
                    try:
                        payload = {
                            "deal_row_idx": int(row_idx),
                            "dealer": dealer_actual,
                            "checks": [{"seat": int(seat_1_to_4), "criteria": list(criteria_list)}],
                        }
                        data = api_post("/deal-criteria-eval-batch", payload, timeout=10)
                        results = data.get("results", []) or []
                        passes = True
                        if results:
                            r0 = results[0] or {}
                            failed_list = r0.get("failed", []) or []
                            untracked = r0.get("untracked", []) or []
                            passes = (len(failed_list) == 0) and (len(untracked) == 0)
                        eval_cache[cache_key] = {"passes": bool(passes), "failed": []}
                        return bool(passes)
                    except Exception:
                        return True

                actual_row = {
                    "BT Index": actual_bt_index,
                    "Source": "Actual",
                    "Auction": actual_auction or None,
                    "Bid": actual_bid,
                    "Declarer": actual_declarer,
                    "Tricks": actual_dd_tricks,
                    "Result": _result_from_auction_and_dd_tricks(actual_auction, actual_dd_tricks),
                    "DD_Score_NS": actual_dd_score_ns,
                    "EV_NS": actual_ev_ns,
                    "Score_NS": _score_ns_from_auction_and_dd_tricks(
                        str(actual_auction) if actual_auction else None,
                        dealer,
                        actual_dd_tricks,
                        pinned_deal.get("Vul", pinned_deal.get("Vulnerability")),
                    ),
                }
                built_row = {
                    "BT Index": built_bt_index,
                    "Source": "Built",
                    "Auction": auction_text or None,
                    "Bid": final_contract,
                    "Declarer": declarer_dir or None,
                    "Tricks": dd_tricks_final,
                    "Result": _result_from_auction_and_dd_tricks(auction_text, dd_tricks_final),
                    "DD_Score_NS": dd_score_final_ns,
                    "EV_NS": built_ev_ns,
                    "Score_NS": _score_ns_from_auction_and_dd_tricks(
                        str(auction_text) if auction_text else None,
                        dealer,
                        dd_tricks_final,
                        pinned_deal.get("Vul", pinned_deal.get("Vulnerability")),
                    ),
                }
                # AI Model: complete auction by repeatedly taking the first row in
                # "Best Next Bids (Advanced)" starting from opening.
                ai_model_auction_text: str | None = None
                ai_model_steps: list[dict[str, Any]] = []
                ai_job_params_key = f"_ai_model_adv_job_params__{best_auc_deal_key}" if best_auc_deal_key else "_ai_model_adv_job_params"

                # Parameters: tie to Advanced controls when present.
                try:
                    ai_top_n = int(locals().get("adv_top_n", 12) or 12)
                    ai_max_deals = int(locals().get("adv_max_deals", 500) or 500)
                    ai_w_desc = float(locals().get("w_desc", 150) or 150)
                    ai_w_threat = float(locals().get("w_threat", 120) or 120)
                    ai_w_guard = float(locals().get("w_guard", 1.0) or 1.0)
                    ai_w_guard_overbid = float(locals().get("w_guard_overbid", 80) or 80)
                    ai_w_guard_tp = float(locals().get("w_guard_tp", 15) or 15)
                    ai_w_guard_neg = float(locals().get("w_guard_neg", 0.5) or 0.5)
                    ai_w_guard_underbid = float(locals().get("w_guard_underbid", 40) or 40)
                    ai_w_guard_tp_surplus = float(locals().get("w_guard_tp_surplus", 8) or 8)
                    ai_w_guard_strain = float(locals().get("w_guard_strain", 60) or 60)
                    ai_w_guard_sacrifice = float(locals().get("w_guard_sacrifice", 70) or 70) / 100.0
                    ai_w_guard_tricks = float(locals().get("w_guard_tricks", 30) or 30)
                except Exception:
                    ai_top_n, ai_max_deals, ai_w_desc, ai_w_threat = 12, 500, 150.0, 120.0
                    ai_w_guard, ai_w_guard_overbid, ai_w_guard_tp, ai_w_guard_neg = 1.0, 80.0, 15.0, 0.5
                    ai_w_guard_underbid, ai_w_guard_tp_surplus, ai_w_guard_strain, ai_w_guard_sacrifice = 40.0, 8.0, 60.0, 0.7
                    ai_w_guard_tricks = 30.0

                # If params changed, clear cached result + job id.
                ai_params = {
                    "deal_row_idx": int(pinned_deal.get("_row_idx")) if pinned_deal.get("_row_idx") is not None else None,
                    "seed": int(seed) if isinstance(seed, (int, float)) else 0,
                    "max_steps": 40,
                    "top_n": int(ai_top_n),
                    "max_deals": int(ai_max_deals),
                    "w_desc": float(ai_w_desc),
                    "w_threat": float(ai_w_threat),
                    "w_guard": float(ai_w_guard),
                    "w_guard_overbid": float(ai_w_guard_overbid),
                    "w_guard_tp": float(ai_w_guard_tp),
                    "w_guard_neg": float(ai_w_guard_neg),
                    "w_guard_underbid": float(ai_w_guard_underbid),
                    "w_guard_tp_surplus": float(ai_w_guard_tp_surplus),
                    "w_guard_strain": float(ai_w_guard_strain),
                    "w_guard_sacrifice": float(ai_w_guard_sacrifice),
                    "w_guard_tricks": float(ai_w_guard_tricks),
                    "permissive_pass": bool(st.session_state.get("always_valid_pass", True)),
                }
                try:
                    prev_params = st.session_state.get(ai_job_params_key)
                    if prev_params != ai_params:
                        st.session_state[ai_job_params_key] = dict(ai_params)
                        # Clear cached results for these display keys
                        if ai_model_auc_ss_key in st.session_state:
                            del st.session_state[ai_model_auc_ss_key]
                        if ai_model_times_ss_key in st.session_state:
                            del st.session_state[ai_model_times_ss_key]
                        if ai_model_job_ss_key in st.session_state:
                            del st.session_state[ai_model_job_ss_key]
                        if ai_model_err_ss_key in st.session_state:
                            del st.session_state[ai_model_err_ss_key]
                except Exception:
                    pass

                # Prefer cached completed auction
                try:
                    cached_ai_auc = st.session_state.get(ai_model_auc_ss_key)
                    cached_ai_times = st.session_state.get(ai_model_times_ss_key)
                    if isinstance(cached_ai_auc, str) and cached_ai_auc.strip():
                        ai_model_auction_text = cached_ai_auc.strip()
                    if isinstance(cached_ai_times, list):
                        ai_model_steps = list(cached_ai_times)
                except Exception:
                    pass

                # If not cached, start/poll async server job
                if ai_model_auction_text is None:
                    if ai_params.get("deal_row_idx") is not None:
                        job_id = st.session_state.get(ai_model_job_ss_key)
                        if not isinstance(job_id, str) or not job_id.strip():
                            try:
                                start_resp = api_post("/ai-model-advanced-path/start", ai_params, timeout=10)
                                job_id = str(start_resp.get("job_id") or "").strip()
                                if job_id:
                                    st.session_state[ai_model_job_ss_key] = job_id
                                    # Clear any prior error
                                    if ai_model_err_ss_key in st.session_state:
                                        del st.session_state[ai_model_err_ss_key]
                            except Exception as e:
                                try:
                                    st.session_state[ai_model_err_ss_key] = str(e)
                                except Exception:
                                    pass
                                job_id = ""
                        if isinstance(job_id, str) and job_id.strip():
                            ai_poll_key = f"{ai_model_job_ss_key}__last_poll_s"
                            try:
                                job = api_get(f"/ai-model-advanced-path/status/{job_id.strip()}", timeout=10)
                                # Record perf snapshot for UI (running/completed).
                                try:
                                    status = str(job.get("status") or "")
                                    created_at_s = job.get("created_at_s")
                                    wall_s = job.get("wall_elapsed_s")
                                    cpu_s = job.get("cpu_elapsed_s")
                                    res0 = job.get("result") or {}
                                    server_ms = res0.get("elapsed_ms")
                                    steps0 = res0.get("steps")
                                    now_s = float(time.time())
                                    run_for_s = None
                                    try:
                                        if created_at_s is not None:
                                            run_for_s = max(0.0, now_s - float(created_at_s))
                                    except Exception:
                                        run_for_s = None
                                    perf_parts: list[str] = []
                                    perf_parts.append(f"AI Model perf: status={status}")
                                    if run_for_s is not None and status == "running":
                                        perf_parts.append(f"running_for={run_for_s:.1f}s")
                                    if isinstance(wall_s, (int, float)):
                                        perf_parts.append(f"wall={float(wall_s):.2f}s")
                                    if isinstance(cpu_s, (int, float)):
                                        perf_parts.append(f"cpu={float(cpu_s):.2f}s")
                                    if isinstance(server_ms, (int, float)):
                                        perf_parts.append(f"handler={float(server_ms)/1000.0:.2f}s")
                                    if steps0 is not None:
                                        try:
                                            perf_parts.append(f"steps={int(steps0)}")
                                        except Exception:
                                            pass
                                    st.session_state[ai_model_perf_ss_key] = " | ".join(perf_parts)
                                except Exception:
                                    pass
                                if job.get("status") == "completed":
                                    res = job.get("result") or {}
                                    auc_s = str(res.get("auction") or "").strip()
                                    steps_detail = res.get("steps_detail") or []
                                    if auc_s:
                                        ai_model_auction_text = auc_s
                                        st.session_state[ai_model_auc_ss_key] = auc_s
                                        if isinstance(steps_detail, list):
                                            st.session_state[ai_model_times_ss_key] = steps_detail
                                            ai_model_steps = list(steps_detail)
                                        # Update placeholders near top
                                        try:
                                            ai_model_auction_placeholder.empty()
                                        except Exception:
                                            pass
                                        try:
                                            # Times placeholder will re-render from session state on next rerun.
                                            pass
                                        except Exception:
                                            pass
                                        # Clear job id (done)
                                        try:
                                            del st.session_state[ai_model_job_ss_key]
                                        except Exception:
                                            pass
                                        try:
                                            if ai_poll_key in st.session_state:
                                                del st.session_state[ai_poll_key]
                                        except Exception:
                                            pass
                                        # Clear any prior error
                                        try:
                                            if ai_model_err_ss_key in st.session_state:
                                                del st.session_state[ai_model_err_ss_key]
                                        except Exception:
                                            pass
                                        # Add slowest-step summary to perf line (top 3).
                                        try:
                                            if isinstance(steps_detail, list) and steps_detail:
                                                pairs = []
                                                for stp in steps_detail:
                                                    if not isinstance(stp, dict):
                                                        continue
                                                    b = str(stp.get("choice") or stp.get("bid") or "").strip().upper()
                                                    ms = stp.get("elapsed_ms")
                                                    try:
                                                        ms_f = float(ms) if ms is not None else None
                                                    except Exception:
                                                        ms_f = None
                                                    if b and ms_f is not None:
                                                        pairs.append((ms_f, b))
                                                pairs.sort(reverse=True)
                                                top3 = ", ".join([f"{b}:{ms:.0f}ms" for ms, b in pairs[:3]])
                                                base = str(st.session_state.get(ai_model_perf_ss_key) or "").strip()
                                                if top3:
                                                    st.session_state[ai_model_perf_ss_key] = (base + " | slowest=" + top3).strip(" |")
                                        except Exception:
                                            pass
                                elif job.get("status") == "failed":
                                    try:
                                        st.session_state[ai_model_err_ss_key] = str(job.get("error") or "job_failed")
                                    except Exception:
                                        pass
                                    try:
                                        del st.session_state[ai_model_job_ss_key]
                                    except Exception:
                                        pass
                                    try:
                                        if ai_poll_key in st.session_state:
                                            del st.session_state[ai_poll_key]
                                    except Exception:
                                        pass
                                elif job.get("status") == "running":
                                    # Indicate computing state near the top placeholder.
                                    try:
                                        ai_model_auction_placeholder.empty()
                                    except Exception:
                                        pass
                                    # Auto-poll: rerun at most once per second while running.
                                    try:
                                        now_s = float(time.time())
                                        last_s = float(st.session_state.get(ai_poll_key) or 0.0)
                                        if now_s - last_s >= 1.0:
                                            st.session_state[ai_poll_key] = now_s
                                            st.rerun()
                                    except Exception:
                                        pass
                            except Exception:
                                try:
                                    st.session_state[ai_model_err_ss_key] = "status_poll_failed"
                                except Exception:
                                    pass

                ai_model_bt_index: int | None = None
                ai_model_bid: str | None = None
                ai_model_declarer: str | None = None
                ai_model_dd_tricks: int | None = None
                ai_model_dd_score: int | None = None
                ai_model_ev: float | None = None
                if ai_model_auction_text:
                    try:
                        ai_calls = [c.strip() for c in str(ai_model_auction_text).split("-") if c.strip()]
                        ai_model_bid = _final_contract_from_calls(ai_calls)
                        ai_model_declarer = _declarer_direction_from_calls(ai_calls, dealer)
                        ai_dd_score_val = get_dd_score_for_auction(str(ai_model_auction_text), dealer, pinned_deal)
                        ai_model_dd_score = int(ai_dd_score_val) if ai_dd_score_val is not None else None
                        ai_dd_tricks_val = get_dd_tricks_for_auction(str(ai_model_auction_text), dealer, pinned_deal)
                        ai_model_dd_tricks = int(ai_dd_tricks_val) if ai_dd_tricks_val is not None else None
                        ai_ev_val = get_ev_for_auction(str(ai_model_auction_text), dealer, pinned_deal)
                        ai_model_ev = round(float(ai_ev_val), 0) if ai_ev_val is not None else None
                    except Exception:
                        pass
                    try:
                        ai_path_data = api_post("/resolve-auction-path", {"auction": str(ai_model_auction_text)}, timeout=5)
                        ai_path = ai_path_data.get("path", []) or []
                        if ai_path:
                            ai_model_bt_index = ai_path[-1].get("bt_index")
                    except Exception:
                        pass

                ai_model_dd_score_ns: int | None = None
                ai_model_ev_ns: float | None = None
                try:
                    v = _to_ns(ai_model_dd_score, ai_model_declarer)
                    ai_model_dd_score_ns = int(v) if v is not None else None
                except Exception:
                    ai_model_dd_score_ns = None
                try:
                    v = _to_ns(ai_model_ev, ai_model_declarer)
                    ai_model_ev_ns = float(v) if v is not None else None
                except Exception:
                    ai_model_ev_ns = None

                ai_model_row = {
                    "BT Index": ai_model_bt_index if ai_model_auction_text else None,
                    "Source": "AI Model",
                    "Auction": ai_model_auction_text,
                    "Bid": (
                        ai_model_bid
                        if ai_model_auction_text
                        else ("(computing...)" if isinstance(st.session_state.get(ai_model_job_ss_key), str) else "(unavailable)")
                    ),
                    "Declarer": ai_model_declarer,
                    "Tricks": ai_model_dd_tricks,
                    "Result": _result_from_auction_and_dd_tricks(ai_model_auction_text, ai_model_dd_tricks),
                    "DD_Score_NS": ai_model_dd_score_ns,
                    "EV_NS": ai_model_ev_ns,
                    "Score_NS": _score_ns_from_auction_and_dd_tricks(
                        str(ai_model_auction_text),
                        dealer,
                        ai_model_dd_tricks,
                        pinned_deal.get("Vul", pinned_deal.get("Vulnerability")),
                    )
                    if ai_model_auction_text
                    else None,
                }

                completion_rows = [actual_row, built_row, ai_model_row]
                # Always include a Cheater Model row (even if unavailable), so the UI is stable.
                best_dd_score_ns: int | None = None
                best_ev_ns: float | None = None
                if best_auction_text:
                    try:
                        v = _to_ns(best_dd_score, best_declarer)
                        best_dd_score_ns = int(v) if v is not None else None
                    except Exception:
                        best_dd_score_ns = None
                    try:
                        v = _to_ns(best_ev, best_declarer)
                        best_ev_ns = float(v) if v is not None else None
                    except Exception:
                        best_ev_ns = None
                completion_rows.append(
                    {
                        "BT Index": best_bt_index if best_auction_text else None,
                        "Source": "Cheater Model",
                        "Auction": best_auction_text,
                        "Bid": best_bid if best_auction_text else "(unavailable)",
                        "Declarer": best_declarer,
                        "Tricks": best_dd_tricks,
                        "Result": _result_from_auction_and_dd_tricks(best_auction_text, best_dd_tricks),
                        "DD_Score_NS": best_dd_score_ns,
                        "EV_NS": best_ev_ns,
                        "Score_NS": _score_ns_from_auction_and_dd_tricks(
                            str(best_auction_text),
                            dealer,
                            best_dd_tricks,
                            pinned_deal.get("Vul", pinned_deal.get("Vulnerability")),
                        )
                        if best_auction_text
                        else None,
                    }
                )

                # Add IMPs column: IMPs gained/lost vs Actual row's Score_NS
                actual_score_ns: int | None = None
                try:
                    for rr in completion_rows:
                        if str(rr.get("Source", "")).strip().lower() == "actual":
                            v = rr.get("Score_NS")
                            actual_score_ns = int(v) if v is not None else None
                            break
                except Exception:
                    actual_score_ns = None
                for rr in completion_rows:
                    try:
                        s = rr.get("Score_NS")
                        s_i = int(s) if s is not None else None
                    except Exception:
                        s_i = None
                    if actual_score_ns is None or s_i is None:
                        rr["IMPs"] = None
                    else:
                        rr["IMPs"] = _score_diff_to_imps(int(s_i) - int(actual_score_ns))

                # Add row-color bucket based on BT membership.
                # If not in the BT table, use Invalid's color.
                for rr in completion_rows:
                    rr["Bucket"] = "Valid" if (rr.get("BT Index") is not None) else "Invalid"

                # ── Debug dump: write AI Model auction debug JSON for agent access ──
                try:
                    import json as _dbg_json
                    from datetime import datetime as _dbg_dt, timezone as _dbg_tz
                    _dbg_path = pathlib.Path("data/ai_model_debug.json")
                    _dbg_path.parent.mkdir(parents=True, exist_ok=True)

                    # Pinned deal summary (safe subset — no giant arrays)
                    _dbg_deal: dict[str, Any] = {}
                    if pinned_deal:
                        for _dk in [
                            "_row_idx", "index", "Dealer", "Vul", "Vulnerability",
                            "Hand_N", "Hand_E", "Hand_S", "Hand_W",
                            "HCP_N", "HCP_E", "HCP_S", "HCP_W",
                            "Total_Points_N", "Total_Points_E", "Total_Points_S", "Total_Points_W",
                        ]:
                            _dv = pinned_deal.get(_dk)
                            if _dv is not None:
                                _dbg_deal[_dk] = _dv

                    # Completion rows (the 4-row summary table)
                    _dbg_completion = []
                    for _cr in completion_rows:
                        _dbg_completion.append({
                            k: (int(v) if isinstance(v, (int,)) else
                                round(float(v), 2) if isinstance(v, float) else
                                str(v) if v is not None else None)
                            for k, v in _cr.items()
                        })

                    # AI Model per-step scoring (from server response)
                    _dbg_ai_steps = []
                    if ai_model_steps:
                        for _stp in ai_model_steps:
                            if isinstance(_stp, dict):
                                _dbg_ai_steps.append(_stp)

                    # Advanced table rows for current auction prefix (if available in session)
                    _dbg_advanced: list[dict[str, Any]] = []
                    try:
                        _adv_ss = st.session_state.get("_ab_adv_rows_debug")
                        if isinstance(_adv_ss, list):
                            _dbg_advanced = _adv_ss
                    except Exception:
                        pass

                    _dbg_payload = {
                        "timestamp": _dbg_dt.now(_dbg_tz.utc).isoformat(),
                        "pinned_deal": _dbg_deal,
                        "dealer": dealer if 'dealer' in dir() else None,
                        "built_auction": auction_text,
                        "ai_model_auction": ai_model_auction_text,
                        "best_auction": best_auction_text if 'best_auction_text' in dir() else None,
                        "actual_auction": actual_auction if 'actual_auction' in dir() else None,
                        "completion_rows": _dbg_completion,
                        "ai_model_steps": _dbg_ai_steps,
                        "advanced_table_rows": _dbg_advanced,
                        "current_path_bids": [
                            str(s.get("bid", "")) for s in (current_path or []) if isinstance(s, dict)
                        ],
                    }
                    _dbg_path.write_text(
                        _dbg_json.dumps(_dbg_payload, indent=2, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    pass  # Debug dump is best-effort; never break the UI

                # Combined aggregate comparison table: Hand Stats + DD Tricks with Source labels.
                try:
                    _AB_DIRS = ["N", "E", "S", "W"]
                    _AB_STRAINS = ["C", "D", "H", "S", "N"]
                    _ab_stats_cache = st.session_state.setdefault("_auction_builder_source_stats_cache", {})
                    _ab_criteria_stats_cache = st.session_state.setdefault("_auction_builder_criteria_stats_cache", {})

                    def _ab_norm_source_label(src_raw: str) -> str:
                        s = str(src_raw or "").strip().upper()
                        if s == "ACTUAL":
                            return "Actual Auction"
                        if s == "AI MODEL":
                            return "AI Model"
                        return str(src_raw or "").strip() or "Selected"

                    def _ab_deal_rows(deal_obj: dict[str, Any]) -> list[dict[str, Any]]:
                        rows: list[dict[str, Any]] = []
                        for _d in _AB_DIRS:
                            r: dict[str, Any] = {"Source": "Deal", "Direction": _d, "Deals": 1}
                            _hcp = deal_obj.get(f"HCP_{_d}")
                            r["HCP"] = round(float(_hcp), 1) if _hcp is not None else None
                            r["HCP_std"] = None
                            _tp = deal_obj.get(f"Total_Points_{_d}")
                            r["Total_Points"] = round(float(_tp), 1) if _tp is not None else None
                            r["Total_Points_std"] = None

                            # Prefer parsing Hand_ strings (S.H.D.C), fallback to SL columns.
                            _sl_by_suit: dict[str, int | None] = {}
                            _hand = str(deal_obj.get(f"Hand_{_d}", "") or "")
                            _parts = _hand.split(".") if _hand else []
                            if len(_parts) == 4:
                                _sl_by_suit = {
                                    "S": len(_parts[0]),
                                    "H": len(_parts[1]),
                                    "D": len(_parts[2]),
                                    "C": len(_parts[3]),
                                }
                            else:
                                for _s in ["S", "H", "D", "C"]:
                                    _v = deal_obj.get(f"SL_{_s}_{_d}")
                                    _sl_by_suit[_s] = int(_v) if _v is not None else None
                            r["SL_C"] = _sl_by_suit.get("C")
                            r["SL_D"] = _sl_by_suit.get("D")
                            r["SL_H"] = _sl_by_suit.get("H")
                            r["SL_S"] = _sl_by_suit.get("S")

                            for _s in _AB_STRAINS:
                                _v = deal_obj.get(f"DD_{_d}_{_s}")
                                _col = "DD_NT" if _s == "N" else f"DD_{_s}"
                                r[_col] = int(float(_v)) if _v is not None else None
                            rows.append(r)
                        return rows

                    def _ab_rows_from_agg_stats(src_label: str, agg_stats: dict[str, Any]) -> list[dict[str, Any]]:
                        _deal_count = agg_stats.get("deal_count")
                        hand_rows = agg_stats.get("hand_stats") or []
                        dd_rows = agg_stats.get("dd_means") or []
                        hs_map = {
                            str(r.get("Direction") or "").upper(): r
                            for r in hand_rows
                            if isinstance(r, dict)
                        }
                        dd_map = {
                            str(r.get("Declarer") or "").upper(): r
                            for r in dd_rows
                            if isinstance(r, dict)
                        }
                        rows: list[dict[str, Any]] = []
                        for _d in _AB_DIRS:
                            hs = hs_map.get(_d, {})
                            dd = dd_map.get(_d, {})
                            rows.append(
                                {
                                    "Source": src_label,
                                    "Direction": _d,
                                    "Deals": _deal_count,
                                    "HCP": hs.get("HCP"),
                                    "HCP_std": hs.get("HCP_std"),
                                    "SL_C": hs.get("SL_C"),
                                    "SL_D": hs.get("SL_D"),
                                    "SL_H": hs.get("SL_H"),
                                    "SL_S": hs.get("SL_S"),
                                    "Total_Points": hs.get("Total_Points"),
                                    "Total_Points_std": hs.get("Total_Points_std"),
                                    "DD_C": dd.get("C"),
                                    "DD_D": dd.get("D"),
                                    "DD_H": dd.get("H"),
                                    "DD_S": dd.get("S"),
                                    "DD_NT": dd.get("N"),
                                }
                            )
                        return rows

                    def _ab_fetch_agg_stats(auction_text_raw: str) -> dict[str, Any] | None:
                        _auc = str(auction_text_raw or "").strip().upper()
                        if not _auc or _auc in ("(UNAVAILABLE)", "(COMPUTING...)"):
                            return None
                        _k = f"{_auc}|{int(max_matching_deals)}|{int(seed)}"
                        if _k in _ab_stats_cache:
                            return _ab_stats_cache.get(_k)
                        _resp = api_post(
                            "/sample-deals-by-auction-pattern",
                            {
                                "pattern": f"^{_auc}$",
                                "sample_size": int(max_matching_deals),
                                "seed": int(seed),
                                "include_stats": True,
                            },
                            timeout=30,
                        )
                        _stats = _resp.get("actual_auction_stats")
                        _ab_stats_cache[_k] = _stats
                        return _stats

                    def _ab_fetch_criteria_stats_map(bt_indices_raw: list[Any]) -> dict[str, Any]:
                        _bt_keys: list[str] = []
                        for _v in bt_indices_raw:
                            try:
                                _bt_i = int(_v)
                            except Exception:
                                continue
                            _bt_keys.append(str(_bt_i))
                        if not _bt_keys:
                            return {}

                        _missing: list[int] = []
                        for _k in _bt_keys:
                            if _k not in _ab_criteria_stats_cache:
                                try:
                                    _missing.append(int(_k))
                                except Exception:
                                    continue
                        if _missing:
                            _dealer_payload = None
                            if pinned_deal is not None:
                                _dealer_payload = str(pinned_deal.get("Dealer") or "").upper() or None
                            if _dealer_payload is None:
                                return {k: _ab_criteria_stats_cache.get(k) for k in _bt_keys}
                            _resp = api_post(
                                "/criteria-stats-by-bt-indices",
                                {"bt_indices": _missing, "dealer": _dealer_payload},
                                timeout=30,
                            )
                            _stats_map = _resp.get("stats_by_bt") or {}
                            for _k_i in _missing:
                                _k_s = str(int(_k_i))
                                _ab_criteria_stats_cache[_k_s] = _stats_map.get(_k_s)

                        return {k: _ab_criteria_stats_cache.get(k) for k in _bt_keys}

                    _combined_source_rows: list[dict[str, Any]] = []
                    if pinned_deal:
                        _combined_source_rows.extend(_ab_deal_rows(pinned_deal))
                    for rr in completion_rows:
                        _src_label = _ab_norm_source_label(rr.get("Source"))
                        _stats = _ab_fetch_agg_stats(str(rr.get("Auction") or ""))
                        if _stats:
                            _combined_source_rows.extend(_ab_rows_from_agg_stats(_src_label, _stats))

                    # Criteria-only stats from agg_expr matches by BT index (independent of actual/predicted auction text).
                    _bt_to_source: dict[str, str] = {}
                    for rr in completion_rows:
                        _bt_val = rr.get("BT Index")
                        if _bt_val is None:
                            continue
                        try:
                            _bt_key = str(int(_bt_val))
                        except Exception:
                            continue
                        _bt_to_source[_bt_key] = _ab_norm_source_label(rr.get("Source"))

                    if _bt_to_source:
                        _criteria_stats_map = _ab_fetch_criteria_stats_map(list(_bt_to_source.keys()))
                        for _bt_key, _src_label in _bt_to_source.items():
                            _c_stats = _criteria_stats_map.get(_bt_key)
                            if _c_stats:
                                _combined_source_rows.extend(
                                    _ab_rows_from_agg_stats(f"Criteria ({_src_label})", _c_stats)
                                )

                    if _combined_source_rows:
                        _combined_df = pd.DataFrame(_combined_source_rows)
                        _col_order = [
                            "Source",
                            "Direction",
                            "Deals",
                            "HCP",
                            "HCP_std",
                            "SL_C",
                            "SL_D",
                            "SL_H",
                            "SL_S",
                            "Total_Points",
                            "Total_Points_std",
                            "DD_C",
                            "DD_D",
                            "DD_H",
                            "DD_S",
                            "DD_NT",
                        ]
                        _keep_cols = [c for c in _col_order if c in _combined_df.columns]
                        with st.expander("Aggregate Hand Stats + DD Tricks (all sources)", expanded=False):
                            st.dataframe(_combined_df[_keep_cols], width="stretch")
                except Exception as _e_source_agg:
                    st.caption(f"⚠️ Aggregate comparison stats error: {_e_source_agg}")

                completion_df = order_columns(
                    pl.DataFrame(completion_rows),
                    priority_cols=[
                        "BT Index",
                        "Source",
                        "Auction",
                        "Bid",
                        "Declarer",
                        "DD_Score_NS",
                        "EV_NS",
                        # Keep these immediately before Score_NS (per UI request)
                        "Tricks",
                        "Result",
                        "Score_NS",
                        "IMPs",
                    ],
                )
                with completed_summary_placeholder.container():
                    st.subheader(_completed_summary_header)
                    if _completed_summary_rehydrate_ms is not None:
                        st.info(f"Loaded auction criteria in {_completed_summary_rehydrate_ms/1000:.2f}s")
                    completion_sel = render_aggrid(
                        completion_df,
                        key="auction_builder_complete_summary",
                        height=140,
                        table_name="auction_builder_complete_summary",
                        update_on=["selectionChanged", "cellClicked"],
                        fit_columns_to_view=True,
                        show_sql_expander=False,
                        tooltip_cols=["Auction"],
                        row_bucket_col="Bucket",
                    )
                # Clicking a row should update the Auction Builder + Bid-by-Bid detail to that source's auction.
                try:
                    if completion_sel:
                        row0 = completion_sel[0] or {}
                        sel_auc = str(row0.get("Auction") or "").strip()
                        sel_source = str(row0.get("Source") or "").strip()
                        if sel_auc:
                            # Avoid infinite reruns by only acting on changes.
                            last_key = "_auction_builder_complete_summary_last_selected"
                            last_val = st.session_state.get(last_key)
                            if last_val != (sel_source, sel_auc):
                                st.session_state[last_key] = (sel_source, sel_auc)
                                st.session_state["_auction_builder_detail_source"] = sel_source
                                st.session_state["_auction_builder_detail_auction"] = sel_auc
                                st.session_state["_auction_builder_pending_set_auction"] = sel_auc
                                st.rerun()
                except Exception:
                    pass
            else:
                # Without a pinned deal, show seat/pair (direction depends on dealer which we don't have).
                decl = "—"
                if declarer_seat is not None and declarer_pair is not None:
                    decl = f"S{int(declarer_seat)} ({declarer_pair})"
                # Get bt_index for built auction from last step of current_path
                built_bt_index = current_path[-1].get("bt_index") if current_path else None
                # Single row (no actual auction without pinned deal)
                completion_rows = [
                    {
                        "BT Index": built_bt_index,
                        "Source": "Built",
                        "Auction": auction_text or None,
                        "Bid": final_contract,
                        "Declarer": decl,
                        "DD_Tricks": None,
                        "DD_Score": None,
                        "Avg_EV_NV": avg_ev_final_nv,
                        "Avg_EV_V": avg_ev_final_v,
                        # Row-color bucket based on BT membership
                        "Bucket": "Valid" if (built_bt_index is not None) else "Invalid",
                    },
                ]
                # Include AI Model row (same values as Built) for UI consistency.
                try:
                    ai_model_row = dict(completion_rows[0])
                    ai_model_row["Source"] = "AI Model"
                    completion_rows.append(ai_model_row)
                except Exception:
                    pass
                with completed_summary_placeholder.container():
                    st.subheader(_completed_summary_header)
                    if _completed_summary_rehydrate_ms is not None:
                        st.info(f"Loaded auction criteria in {_completed_summary_rehydrate_ms/1000:.2f}s")
                    render_aggrid(
                        pl.DataFrame(completion_rows),
                        key="auction_builder_complete_summary",
                        height=95,
                        table_name="auction_builder_complete_summary",
                        update_on=["selectionChanged", "cellClicked"],
                        fit_columns_to_view=True,
                        show_sql_expander=False,
                        tooltip_cols=["Auction"],
                        row_bucket_col="Bucket",
                    )
        elif pinned_deal and current_path:
            # Auction not complete, but show pinned deal status
            with completed_summary_placeholder.container():
                st.subheader(_completed_summary_header)
                if _completed_summary_rehydrate_ms is not None:
                    st.info(f"Loaded auction criteria in {_completed_summary_rehydrate_ms/1000:.2f}s")
                if pinned_all_pass:
                    st.success("📌 Pinned deal matches ALL criteria in this auction path")
                else:
                    st.warning("📌 Pinned deal FAILS some criteria - see 'Pinned' and 'Failed_Criteria' columns")
        else:
            # No completion grid to show; render just the title in the placeholder
            with completed_summary_placeholder.container():
                st.subheader(_completed_summary_header)
                if _completed_summary_rehydrate_ms is not None:
                    st.info(f"Loaded auction criteria in {_completed_summary_rehydrate_ms/1000:.2f}s")
        
        summary_df = pl.DataFrame(summary_rows)
        summary_df = order_columns(
            summary_df,
            priority_cols=[
                "BT Index",
                "Bid Num",
                "Direction",
                "Seat",
                "Bid",
                "Matches",  # Pattern-based: deals where actual auction matches
                "Deals",    # Criteria-based: deals that qualify based on hand evaluation
                "Avg_EV",   # Avg EV for this bid, chosen based on pinned vul (only meaningful when pinned)
                "EV_NV",    # Pre-computed average EV (Not Vulnerable)
                "EV_V",     # Pre-computed average EV (Vulnerable)
                "Avg_Par",  # Avg Par for this bid, chosen based on pinned vul (meaningful when pinned)
                "Par_NV",   # Avg Par (Not Vulnerable)
                "Par_V",    # Avg Par (Vulnerable)
                "Criteria Count",
                "Agg_Expr",
                "Categories",
                "Complete",
            ],
        )
        # Make summary grid selectable to show deals for a specific auction step
        # Use selectionChanged to trigger rerun when row is selected
        # Use fit_columns_to_view=False to prevent column width changes on rerender
        # Use row_pass_fail_col to colorize rows:
        #   - With pinned deal: green=passes criteria, red=fails criteria
        #   - Without pinned deal: green=has bt_index, red=no bt_index (e.g., leading passes)
        # Hide/show columns depending on whether a deal is pinned:
        # - pinned: hide EV_NV/EV_V, Avg_EV (EV lookup has seat mismatch bug - to be fixed)
        # - not pinned: hide Avg_EV (use EV_NV/EV_V)
        summary_hide_cols = ["Agg_Expr_full", "Categories_full", "_passes"]
        summary_hide_cols += ["_partial_auction"]
        if pinned_deal:
            summary_hide_cols += ["EV_NV", "EV_V", "Avg_EV", "Par_NV", "Par_V"]
        else:
            summary_hide_cols += ["Avg_EV", "Avg_Par"]

        # Display Bid-by-Bid Detail summary table (optionally labeled by selected completed-auction source).
        detail_source = None
        try:
            stored_auc = str(st.session_state.get("_auction_builder_detail_auction") or "").strip()
            stored_src = str(st.session_state.get("_auction_builder_detail_source") or "").strip()
            if stored_auc and stored_src and stored_auc == str(current_auction or "").strip():
                detail_source = stored_src
        except Exception:
            detail_source = None

        st.subheader(f"📋 Bid-by-Bid Detail — {detail_source}" if detail_source else "📋 Bid-by-Bid Detail")
        summary_selection = render_aggrid(
            summary_df,
            key="auction_builder_summary",
            height=calc_grid_height(4),
            table_name="auction_builder_summary",
            hide_cols=summary_hide_cols,
            # cellClicked ensures clicking a cell updates selection immediately (not just row highlight).
            update_on=["selectionChanged", "cellClicked"],
            fit_columns_to_view=False,
            row_pass_fail_col="_passes",
        )
        
        # Handle row selection - show matching deals for selected step
        # render_aggrid returns a list of selected rows directly
        selected_step_auction: str | None = None
        selected_step_row: dict[str, Any] | None = None
        if summary_selection and len(summary_selection) > 0:
            row0 = summary_selection[0] or {}
            if isinstance(row0, dict):
                selected_step_row = dict(row0)
            # Prefer the exact partial auction stored in the row (robust to sorts/filters).
            sel_partial = row0.get("_partial_auction")
            if isinstance(sel_partial, str) and sel_partial.strip():
                selected_step_auction = sel_partial.strip()
            else:
                # Fallback: reconstruct by Bid Num.
                sel_bid_num_raw = row0.get("Bid Num")
                try:
                    sel_bid_num = int(sel_bid_num_raw) if sel_bid_num_raw is not None else None
                except Exception:
                    sel_bid_num = None
                if sel_bid_num is not None and 1 <= sel_bid_num <= len(current_path):
                    selected_step_auction = "-".join([s["bid"] for s in current_path[:sel_bid_num]])
        
        # If no row selected but auction is complete, show matching deals for full auction
        if selected_step_auction is None and is_complete and current_auction:
            selected_step_auction = current_auction

        # Keep a stable anchor in the layout to reduce page jumpiness as selection changes.
        _matching_deals_anchor = st.empty()
        if selected_step_auction is None:
            _matching_deals_anchor.caption("Select a row in Bid-by-Bid Detail to load matching deals.")
        else:
            _matching_deals_anchor.empty()
        
        # Matching Deals section - shown when a row is selected or when auction is complete
        if selected_step_auction is not None:
            display_auction = selected_step_auction
            st.subheader("🎯 Deals Where Auction Matches Actual Auction")
            # Check if the selected auction is a *completed* auction.
            #
            # IMPORTANT:
            # - Deals' actual auctions are completed auctions (end with "-P-P-P").
            # - The current Auction Builder prefix may be incomplete; even if it equals current_auction,
            #   we must treat it as a prefix match unless it is actually complete.
            is_selected_complete = display_auction.upper().endswith("-P-P-P") or (
                bool(is_complete) and display_auction == current_auction
            )
            # NOTE: These are always matching against the deal dataset's *actual auction* column.
            # We vary only the *pattern* based on which auction (Actual/Built/AI/Cheater) the user
            # selected in Completed Auction Summary.
            src_label = str(detail_source or "Selected").strip() or "Selected"
            if is_selected_complete:
                if src_label.upper() == "ACTUAL":
                    st.caption(f"Deals where the deal's actual auction matches: **{display_auction}**")
                else:
                    st.caption(f"Deals where the deal's actual auction matches the {src_label} auction: **{display_auction}**")
            else:
                if src_label.upper() == "ACTUAL":
                    st.caption(f"Deals where the deal's actual auction starts with: **{display_auction}**")
                else:
                    st.caption(f"Deals where the deal's actual auction starts with the {src_label} prefix: **{display_auction}**")

            # Criteria-based range snapshot for the selected partial step.
            # Shows hard bounds from agg_expr (if present) and observed ranges from criteria-matched deals.
            try:
                _range_bid_num: int | None = None
                if selected_step_row is not None:
                    try:
                        _bn_raw = selected_step_row.get("Bid Num")
                        _range_bid_num = int(_bn_raw) if _bn_raw is not None else None
                    except Exception:
                        _range_bid_num = None
                if _range_bid_num is None:
                    _tok_ct = len([t for t in str(display_auction or "").split("-") if str(t).strip()])
                    if 1 <= _tok_ct <= len(current_path):
                        _range_bid_num = int(_tok_ct)

                _range_step: dict[str, Any] | None = None
                if _range_bid_num is not None and 1 <= _range_bid_num <= len(current_path):
                    _range_step = current_path[_range_bid_num - 1]

                _range_bt_idx: int | None = None
                if selected_step_row is not None:
                    try:
                        _bt_raw = selected_step_row.get("BT Index")
                        _range_bt_idx = int(_bt_raw) if _bt_raw not in (None, "") else None
                    except Exception:
                        _range_bt_idx = None
                if _range_bt_idx is None and _range_step is not None:
                    try:
                        _bt_raw = _range_step.get("bt_index")
                        _range_bt_idx = int(_bt_raw) if _bt_raw is not None else None
                    except Exception:
                        _range_bt_idx = None

                _range_criteria: list[str] = []
                if _range_step is not None:
                    _range_criteria = [str(x) for x in (_range_step.get("agg_expr") or []) if str(x).strip()]

                # Active bidder direction for this selected step (requires pinned deal's dealer).
                _range_active_dir: str | None = None
                if pinned_deal and _range_bid_num is not None:
                    try:
                        _dirs = ["N", "E", "S", "W"]
                        _dealer = str(pinned_deal.get("Dealer", "N")).upper()
                        _d_idx = _dirs.index(_dealer) if _dealer in _dirs else 0
                        _seat_1_to_4 = ((_range_bid_num - 1) % 4) + 1
                        _range_active_dir = _dirs[(_d_idx + _seat_1_to_4 - 1) % 4]
                    except Exception:
                        _range_active_dir = None

                # Parse simple numeric criterion bounds for active bidder stats.
                def _criteria_bounds_map(criteria_list: list[str]) -> dict[str, dict[str, float | None]]:
                    out: dict[str, dict[str, float | None]] = {}
                    for metric in ["HCP", "Total_Points", "SL_C", "SL_D", "SL_H", "SL_S"]:
                        out[metric] = {"lb": None, "ub": None}

                    patt = re.compile(r"^\s*(HCP|Total_Points|SL_[SHDC])\s*(<=|>=|==|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
                    for c_raw in criteria_list:
                        c = str(c_raw or "").strip().replace("≥", ">=").replace("≤", "<=")
                        m = patt.match(c)
                        if not m:
                            continue
                        metric_raw, op, val_s = m.group(1), m.group(2), m.group(3)
                        metric = metric_raw
                        if metric.startswith("SL_") and len(metric) == 4:
                            metric = f"SL_{metric[-1]}"
                        if metric not in out:
                            continue
                        try:
                            val = float(val_s)
                        except Exception:
                            continue
                        is_int_like = metric in {"HCP", "Total_Points", "SL_C", "SL_D", "SL_H", "SL_S"}
                        if op == ">":
                            lb = val + 1.0 if is_int_like else val
                            cur = out[metric]["lb"]
                            out[metric]["lb"] = lb if cur is None else max(float(cur), lb)
                        elif op == ">=":
                            cur = out[metric]["lb"]
                            out[metric]["lb"] = val if cur is None else max(float(cur), val)
                        elif op == "<":
                            ub = val - 1.0 if is_int_like else val
                            cur = out[metric]["ub"]
                            out[metric]["ub"] = ub if cur is None else min(float(cur), ub)
                        elif op == "<=":
                            cur = out[metric]["ub"]
                            out[metric]["ub"] = val if cur is None else min(float(cur), val)
                        elif op == "==":
                            out[metric]["lb"] = val
                            out[metric]["ub"] = val
                    return out

                def _fmt_num_range(v: float | None) -> str:
                    if v is None:
                        return "—"
                    if abs(float(v) - round(float(v))) < 1e-9:
                        return str(int(round(float(v))))
                    return f"{float(v):.1f}"

                def _fmt_range(lb: float | None, ub: float | None) -> str:
                    if lb is None and ub is None:
                        return "—"
                    if lb is not None and ub is not None:
                        return f"{_fmt_num_range(lb)}-{_fmt_num_range(ub)}"
                    if lb is not None:
                        return f"{_fmt_num_range(lb)}+"
                    return f"<= {_fmt_num_range(ub)}"

                if _range_bt_idx is not None:
                    _criteria_stats_cache = st.session_state.setdefault("_auction_builder_criteria_stats_cache", {})
                    _k = str(int(_range_bt_idx))
                    if _k not in _criteria_stats_cache:
                        _dealer_payload = None
                        if pinned_deal is not None:
                            _dealer_payload = str(pinned_deal.get("Dealer") or "").upper() or None
                        if _dealer_payload is not None:
                            _resp = api_post(
                                "/criteria-stats-by-bt-indices",
                                {"bt_indices": [int(_range_bt_idx)], "dealer": _dealer_payload},
                                timeout=30,
                            )
                            _criteria_stats_cache[_k] = (_resp.get("stats_by_bt") or {}).get(_k)
                        else:
                            _criteria_stats_cache[_k] = {}

                    _c_stats = _criteria_stats_cache.get(_k) or {}
                    _dealer_for_map = str((pinned_deal or {}).get("Dealer") or "").upper() if pinned_deal else ""
                    _row_by_dir: dict[str, dict[str, Any]] = {}
                    if _dealer_for_map in {"N", "E", "S", "W"}:
                        _hp_cache = st.session_state.setdefault("_auction_builder_bt_hand_profile_cache", {})
                        _hp_key = f"{int(_range_bt_idx)}|{_dealer_for_map}"
                        _hp_rows = _hp_cache.get(_hp_key)
                        if _hp_rows is None:
                            _resp_hp = api_post(
                                "/bt-hand-profile",
                                {"bt_indices": [int(_range_bt_idx)]},
                                timeout=30,
                            )
                            _hp_rows = _resp_hp.get("rows") or []
                            _hp_cache[_hp_key] = _hp_rows

                        _dirs = ["N", "E", "S", "W"]
                        _d0 = _dirs.index(_dealer_for_map)
                        _seat_to_dir = {
                            1: _dirs[_d0],
                            2: _dirs[(_d0 + 1) % 4],
                            3: _dirs[(_d0 + 2) % 4],
                            4: _dirs[(_d0 + 3) % 4],
                        }
                        for _r in _hp_rows:
                            if not isinstance(_r, dict):
                                continue
                            try:
                                _seat_i = int(_r.get("seat"))
                            except Exception:
                                continue
                            _dir_i = _seat_to_dir.get(_seat_i)
                            if _dir_i:
                                _row_by_dir[_dir_i] = {
                                    "Direction": _dir_i,
                                    "Deals": _r.get("n"),
                                    "HCP_min": _r.get("HCP_min"),
                                    "HCP_max": _r.get("HCP_max"),
                                    "HCP_p10": _r.get("HCP_p10"),
                                    "HCP_p90": _r.get("HCP_p90"),
                                    "Total_Points_min": _r.get("Total_Points_min"),
                                    "Total_Points_max": _r.get("Total_Points_max"),
                                    "Total_Points_p10": _r.get("Total_Points_p10"),
                                    "Total_Points_p90": _r.get("Total_Points_p90"),
                                    "SL_C_min": _r.get("SL_C_min"),
                                    "SL_C_max": _r.get("SL_C_max"),
                                    "SL_C_p10": _r.get("SL_C_p10"),
                                    "SL_C_p90": _r.get("SL_C_p90"),
                                    "SL_D_min": _r.get("SL_D_min"),
                                    "SL_D_max": _r.get("SL_D_max"),
                                    "SL_D_p10": _r.get("SL_D_p10"),
                                    "SL_D_p90": _r.get("SL_D_p90"),
                                    "SL_H_min": _r.get("SL_H_min"),
                                    "SL_H_max": _r.get("SL_H_max"),
                                    "SL_H_p10": _r.get("SL_H_p10"),
                                    "SL_H_p90": _r.get("SL_H_p90"),
                                    "SL_S_min": _r.get("SL_S_min"),
                                    "SL_S_max": _r.get("SL_S_max"),
                                    "SL_S_p10": _r.get("SL_S_p10"),
                                    "SL_S_p90": _r.get("SL_S_p90"),
                                }
                    _obs_row = _row_by_dir.get(str(_range_active_dir or "").upper()) if _range_active_dir else None
                    if _obs_row:
                        _bounds = _criteria_bounds_map(_range_criteria)
                        _deals_n = _obs_row.get("Deals")
                        with st.expander(
                            f"📏 Criteria Range Snapshot ({display_auction}) — bidder {_range_active_dir} "
                            f"({_deals_n:,} criteria-matched deals)" if isinstance(_deals_n, int) else
                            f"📏 Criteria Range Snapshot ({display_auction}) — bidder {_range_active_dir}",
                            expanded=False,
                        ):
                            _show_all_dirs = st.checkbox(
                                "All 4 directions",
                                value=False,
                                key=f"ab_range_all_dirs_{display_auction}_{_k}",
                                help="Show ranges for N/E/S/W instead of only the active bidder direction.",
                            )

                            _dirs_to_show = ["N", "E", "S", "W"] if _show_all_dirs else [str(_range_active_dir)]
                            _metric_rows: list[dict[str, Any]] = []
                            for _dir_show in _dirs_to_show:
                                _r = _row_by_dir.get(str(_dir_show).upper())
                                if not _r:
                                    continue
                                for _metric in ["HCP", "Total_Points", "SL_C", "SL_D", "SL_H", "SL_S"]:
                                    _b = _bounds.get(_metric, {"lb": None, "ub": None})
                                    _obs_min = _r.get(f"{_metric}_min")
                                    _obs_max = _r.get(f"{_metric}_max")
                                    _p10 = _r.get(f"{_metric}_p10")
                                    _p90 = _r.get(f"{_metric}_p90")
                                    _metric_rows.append(
                                        {
                                            "Direction": str(_dir_show).upper(),
                                            "Metric": _metric,
                                            "Criteria": _fmt_range(_b.get("lb"), _b.get("ub")),
                                            "Observed Min-Max": _fmt_range(
                                                float(_obs_min) if _obs_min is not None else None,
                                                float(_obs_max) if _obs_max is not None else None,
                                            ),
                                            "Observed P10-P90": _fmt_range(
                                                float(_p10) if _p10 is not None else None,
                                                float(_p90) if _p90 is not None else None,
                                            ),
                                        }
                                    )
                            _range_df = pd.DataFrame(_metric_rows)
                            st.dataframe(_range_df, width="stretch")

                            # Trick Quality: compare multiple trick estimators in one place.
                            try:
                                _calls = [c.strip().upper() for c in str(display_auction or "").split("-") if str(c).strip()]
                                _last_contract: str | None = None
                                for _c in _calls:
                                    if re.match(r"^[1-7][CDHSN]$", _c):
                                        _last_contract = _c
                                if _last_contract:
                                    _level = int(_last_contract[0])
                                    _strain = _last_contract[1]
                                    _book = _level + 6

                                    _declarer_dir: str | None = None
                                    if pinned_deal:
                                        try:
                                            _declarer_dir = _declarer_direction_from_calls(_calls, str(pinned_deal.get("Dealer", "N")))
                                        except Exception:
                                            _declarer_dir = None
                                    if not _declarer_dir:
                                        _declarer_dir = str(_range_active_dir or "").upper() or None

                                    def _conf_label(n: int | None) -> str:
                                        if n is None:
                                            return "Low"
                                        if n >= 1000:
                                            return "High"
                                        if n >= 250:
                                            return "Med"
                                        return "Low"

                                    def _fmt_delta(v: float | None) -> str:
                                        if v is None:
                                            return "—"
                                        return f"{float(v):+.1f}"

                                    _tq_rows: list[dict[str, Any]] = []
                                    _deal_count_all = _c_stats.get("deal_count")
                                    _deal_count_i = int(_deal_count_all) if isinstance(_deal_count_all, (int, float)) else None

                                    # 1) Precomputed trick quality (Phase 1 endpoint)
                                    _dealer_for_tq = str((pinned_deal or {}).get("Dealer") or "").upper() if pinned_deal else ""
                                    if (
                                        _dealer_for_tq in {"N", "E", "S", "W"}
                                        and _declarer_dir in {"N", "E", "S", "W"}
                                    ):
                                        _dirs = ["N", "E", "S", "W"]
                                        _d0 = _dirs.index(_dealer_for_tq)
                                        _dir_to_seat = {
                                            _dirs[_d0]: 1,
                                            _dirs[(_d0 + 1) % 4]: 2,
                                            _dirs[(_d0 + 2) % 4]: 3,
                                            _dirs[(_d0 + 3) % 4]: 4,
                                        }
                                        _seat_i = _dir_to_seat.get(str(_declarer_dir))
                                        _strain_to_i = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}
                                        _strain_i = _strain_to_i.get(str(_strain))
                                        _vul_i = None
                                        if pinned_deal:
                                            _vul_raw = str(
                                                pinned_deal.get("Vul", pinned_deal.get("Vulnerability", ""))
                                            ).upper()
                                            _ns_vul = _vul_raw in ("N_S", "NS", "BOTH", "ALL")
                                            _ew_vul = _vul_raw in ("E_W", "EW", "BOTH", "ALL")
                                            _is_ns_decl = str(_declarer_dir) in ("N", "S")
                                            _vul_i = 1 if ((_is_ns_decl and _ns_vul) or ((not _is_ns_decl) and _ew_vul)) else 0
                                        if _seat_i is not None and _strain_i is not None:
                                            _resp_tq = api_post(
                                                "/bt-trick-quality",
                                                {
                                                    "bt_indices": [int(_range_bt_idx)],
                                                    "seat": int(_seat_i),
                                                    "strain": int(_strain_i),
                                                    "level": int(_level),
                                                    "vul": int(_vul_i) if _vul_i is not None else None,
                                                },
                                                timeout=30,
                                            )
                                            _tq_pre_rows = _resp_tq.get("rows") or []
                                            if _tq_pre_rows:
                                                _pre = _tq_pre_rows[0]
                                                try:
                                                    _pre_exp = float(_pre.get("exp_extra")) if _pre.get("exp_extra") is not None else None
                                                except Exception:
                                                    _pre_exp = None
                                                try:
                                                    _pre_mt = float(_pre.get("mean_tricks")) if _pre.get("mean_tricks") is not None else None
                                                except Exception:
                                                    _pre_mt = None
                                                try:
                                                    _pre_pm = float(_pre.get("p_make")) if _pre.get("p_make") is not None else None
                                                except Exception:
                                                    _pre_pm = None
                                                try:
                                                    _pre_pp1 = float(_pre.get("p_plus1")) if _pre.get("p_plus1") is not None else None
                                                except Exception:
                                                    _pre_pp1 = None
                                                _pre_n = _pre.get("n")
                                                _pre_n_i = int(_pre_n) if isinstance(_pre_n, (int, float)) else None
                                                _vul_label = "V" if _vul_i == 1 else ("NV" if _vul_i == 0 else "?")
                                                _tq_rows.append(
                                                    {
                                                        "Metric": "Precomputed Trick Quality",
                                                        "Tricks +/-": _fmt_delta(_pre_exp),
                                                        "ExpTricks": round(_pre_mt, 1) if _pre_mt is not None else None,
                                                        "Baseline": _book,
                                                        "SampleN": _pre_n_i,
                                                        "Confidence": _conf_label(_pre_n_i),
                                                        "Notes": (
                                                            f"p_make={(_pre_pm * 100.0):.1f}%"
                                                            if _pre_pm is not None
                                                            else ""
                                                        )
                                                        + (
                                                            f", p_plus1={(_pre_pp1 * 100.0):.1f}%"
                                                            if _pre_pp1 is not None
                                                            else ""
                                                        )
                                                        + f", vul={_vul_label}",
                                                    }
                                                )

                                    # 2) Criteria DD mean (full matched set stats)
                                    _dd_rows = _c_stats.get("dd_means") or []
                                    _dd_map = {
                                        str(r.get("Declarer") or "").upper(): r
                                        for r in _dd_rows
                                        if isinstance(r, dict)
                                    }
                                    _dd_decl_row = _dd_map.get(str(_declarer_dir or "").upper(), {})
                                    _dd_mean = _dd_decl_row.get(_strain) if isinstance(_dd_decl_row, dict) else None
                                    try:
                                        _dd_mean_f = float(_dd_mean) if _dd_mean is not None else None
                                    except Exception:
                                        _dd_mean_f = None
                                    if _dd_mean_f is not None:
                                        _tq_rows.append(
                                            {
                                                "Metric": "Criteria DD Mean",
                                                "Tricks +/-": _fmt_delta(_dd_mean_f - float(_book)),
                                                "ExpTricks": round(_dd_mean_f, 1),
                                                "Baseline": _book,
                                                "SampleN": _deal_count_i,
                                                "Confidence": _conf_label(_deal_count_i),
                                                "Notes": f"Declarer {_declarer_dir}, strain {_strain}",
                                            }
                                        )

                                    # Sample deals for percentile/median and bucket estimators (optional; can be slow).
                                    _sample_deals: list[dict[str, Any]] = []
                                    _load_sample_estimators = st.toggle(
                                        "Load sampled estimators (median + historical bucket)",
                                        value=False,
                                        key=f"ab_tq_sample_estimators_{display_auction}_{_k}",
                                        help="Calls /sample-deals-for-bt-indices and may add several seconds.",
                                    )
                                    if _load_sample_estimators:
                                        _bt_sample_cache = st.session_state.setdefault("_auction_builder_bt_sample_cache", {})
                                        _sample_key = f"{int(_range_bt_idx)}|{int(seed)}|{min(int(max_matching_deals), 500)}"
                                        _sample_deals = _bt_sample_cache.get(_sample_key)
                                        if _sample_deals is None:
                                            _resp_sd = api_post(
                                                "/sample-deals-for-bt-indices",
                                                {
                                                    "bt_indices": [int(_range_bt_idx)],
                                                    "sample_size": min(int(max_matching_deals), 500),
                                                    "seed": int(seed),
                                                },
                                                timeout=45,
                                            )
                                            _sample_deals = (
                                                ((_resp_sd.get("deals_by_bt") or {}).get(str(int(_range_bt_idx))) or {})
                                                .get("deals", [])
                                                or []
                                            )
                                            _bt_sample_cache[_sample_key] = _sample_deals

                                    _dd_col = f"DD_{str(_declarer_dir or '').upper()}_{_strain}"
                                    _dd_vals: list[float] = []
                                    for _d in _sample_deals:
                                        try:
                                            _v = _d.get(_dd_col)
                                            if _v is not None:
                                                _dd_vals.append(float(_v))
                                        except Exception:
                                            pass

                                    # 3) Criteria DD median (sampled)
                                    if _dd_vals:
                                        _p50 = float(pd.Series(_dd_vals).median())
                                        _tq_rows.append(
                                            {
                                                "Metric": "Criteria DD Median (sample)",
                                                "Tricks +/-": _fmt_delta(_p50 - float(_book)),
                                                "ExpTricks": round(_p50, 1),
                                                "Baseline": _book,
                                                "SampleN": len(_dd_vals),
                                                "Confidence": _conf_label(len(_dd_vals)),
                                                "Notes": f"P50 from sampled criteria-matched deals ({_dd_col})",
                                            }
                                        )
                                    elif not _load_sample_estimators:
                                        _tq_rows.append(
                                            {
                                                "Metric": "Criteria DD Median (sample)",
                                                "Tricks +/-": "—",
                                                "ExpTricks": None,
                                                "Baseline": _book,
                                                "SampleN": None,
                                                "Confidence": "Low",
                                                "Notes": "Enable 'Load sampled estimators' to compute",
                                            }
                                        )

                                    # 4) Historical bucket (Total_Points + trump length + secondary length)
                                    if pinned_deal and _range_active_dir and _sample_deals:
                                        _ad = str(_range_active_dir).upper()
                                        _tp_cur = pinned_deal.get(f"Total_Points_{_ad}")
                                        try:
                                            _tp_cur_f = float(_tp_cur) if _tp_cur is not None else None
                                        except Exception:
                                            _tp_cur_f = None

                                        def _tp_bin(tp: float | None) -> str | None:
                                            if tp is None:
                                                return None
                                            if tp <= 7:
                                                return "0-7"
                                            if tp <= 10:
                                                return "8-10"
                                            if tp <= 13:
                                                return "11-13"
                                            if tp <= 16:
                                                return "14-16"
                                            if tp <= 19:
                                                return "17-19"
                                            return "20+"

                                        def _slen_bin(x: float | int | None) -> str | None:
                                            if x is None:
                                                return None
                                            try:
                                                xi = int(x)
                                            except Exception:
                                                return None
                                            if xi <= 4:
                                                return "<=4"
                                            if xi == 5:
                                                return "5"
                                            if xi == 6:
                                                return "6"
                                            return "7+"

                                        def _sec_bin(x: float | int | None) -> str | None:
                                            if x is None:
                                                return None
                                            try:
                                                xi = int(x)
                                            except Exception:
                                                return None
                                            if xi <= 3:
                                                return "<=3"
                                            if xi == 4:
                                                return "4"
                                            return "5+"

                                        _suit_for_len = _strain if _strain in {"C", "D", "H", "S"} else None
                                        _cur_sl_by_suit: dict[str, int | None] = {}
                                        for _s in ["S", "H", "D", "C"]:
                                            _v = pinned_deal.get(f"SL_{_s}_{_ad}")
                                            try:
                                                _cur_sl_by_suit[_s] = int(_v) if _v is not None else None
                                            except Exception:
                                                _cur_sl_by_suit[_s] = None
                                        _cur_trump_len = _cur_sl_by_suit.get(_suit_for_len) if _suit_for_len else max(
                                            [_v for _v in _cur_sl_by_suit.values() if isinstance(_v, int)],
                                            default=None,
                                        )
                                        _cur_secondary = None
                                        _vals_sorted = sorted(
                                            [int(_v) for _v in _cur_sl_by_suit.values() if isinstance(_v, int)],
                                            reverse=True,
                                        )
                                        if len(_vals_sorted) >= 2:
                                            _cur_secondary = _vals_sorted[1]

                                        _target_bucket = (_tp_bin(_tp_cur_f), _slen_bin(_cur_trump_len), _sec_bin(_cur_secondary))

                                        _bucket_vals: list[float] = []
                                        for _d in _sample_deals:
                                            _tp = _d.get(f"Total_Points_{_ad}")
                                            try:
                                                _tp_f = float(_tp) if _tp is not None else None
                                            except Exception:
                                                _tp_f = None
                                            _sl_map = {}
                                            for _s in ["S", "H", "D", "C"]:
                                                _sv = _d.get(f"SL_{_s}_{_ad}")
                                                try:
                                                    _sl_map[_s] = int(_sv) if _sv is not None else None
                                                except Exception:
                                                    _sl_map[_s] = None
                                            _trump = _sl_map.get(_suit_for_len) if _suit_for_len else max(
                                                [_v for _v in _sl_map.values() if isinstance(_v, int)],
                                                default=None,
                                            )
                                            _sec = None
                                            _ss = sorted(
                                                [int(_v) for _v in _sl_map.values() if isinstance(_v, int)],
                                                reverse=True,
                                            )
                                            if len(_ss) >= 2:
                                                _sec = _ss[1]
                                            _bucket = (_tp_bin(_tp_f), _slen_bin(_trump), _sec_bin(_sec))
                                            if _bucket != _target_bucket:
                                                continue
                                            try:
                                                _v = _d.get(_dd_col)
                                                if _v is not None:
                                                    _bucket_vals.append(float(_v))
                                            except Exception:
                                                pass

                                        if _bucket_vals:
                                            _bucket_mean = float(pd.Series(_bucket_vals).mean())
                                            _tq_rows.append(
                                                {
                                                    "Metric": "Historical Bucket",
                                                    "Tricks +/-": _fmt_delta(_bucket_mean - float(_book)),
                                                    "ExpTricks": round(_bucket_mean, 1),
                                                    "Baseline": _book,
                                                    "SampleN": len(_bucket_vals),
                                                    "Confidence": _conf_label(len(_bucket_vals)),
                                                    "Notes": (
                                                        f"TP={_target_bucket[0]}, TrumpLen={_target_bucket[1]}, "
                                                        f"Secondary={_target_bucket[2]}"
                                                    ),
                                                }
                                            )

                                    # 5) Actual pinned deal DD for this contract/declarer (if available)
                                    if pinned_deal:
                                        _dd_actual = pinned_deal.get(_dd_col)
                                        try:
                                            _dd_actual_i = int(float(_dd_actual)) if _dd_actual is not None else None
                                        except Exception:
                                            _dd_actual_i = None
                                        if _dd_actual_i is not None:
                                            _tq_rows.append(
                                                {
                                                    "Metric": "Pinned Deal Actual",
                                                    "Tricks +/-": _fmt_delta(float(_dd_actual_i) - float(_book)),
                                                    "ExpTricks": _dd_actual_i,
                                                    "Baseline": _book,
                                                    "SampleN": 1,
                                                    "Confidence": "N/A",
                                                    "Notes": f"Observed DD for pinned deal ({_dd_col})",
                                                }
                                            )

                                    if _tq_rows:
                                        _tq_df = pd.DataFrame(_tq_rows)
                                        st.markdown("**Trick Quality**")
                                        st.dataframe(_tq_df, width="stretch")
                            except Exception as _e_tq:
                                st.caption(f"⚠️ Trick quality error: {_e_tq}")
                                pass
                    elif _range_criteria:
                        st.caption("Range snapshot: pin a deal to resolve bidder direction for this partial step.")
            except Exception as _e_range_snapshot:
                st.caption(f"⚠️ Criteria range snapshot error: {_e_range_snapshot}")
            
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
                    # Rehydrate notes on display so edits show up immediately (without re-fetching deals)
                    notes_df = _auction_builder_load_notes_df(AUCTION_BUILDER_NOTES_CSV)
                    notes_m = _auction_builder_notes_map(notes_df)
                    deals_df = pl.DataFrame(cached_data.get("display_rows", []))
                    if len(deals_df) > 0 and "index" in deals_df.columns:
                        deals_df = deals_df.with_columns(
                            pl.col("index")
                            .cast(pl.Int64, strict=False)
                            .map_elements(lambda x: notes_m.get(int(x), "") if x is not None else "", return_dtype=pl.Utf8)
                            .alias("Note")
                        )
                    render_aggrid(
                        deals_df,
                        key="auction_builder_deals",
                        height=calc_grid_height(len(deals_df), max_height=400),
                        table_name="auction_builder_deals",
                    )

                    # Display actual-auction aggregate stats (hand stats + DD means)
                    _aa_stats = cached_data.get("actual_auction_stats")
                    if _aa_stats:
                        _aa_n = _aa_stats.get("deal_count", total_count)
                        with st.expander(
                            f"Actual Auction Stats ({_aa_n:,} deals)",
                            expanded=False,
                        ):
                            _src_raw = str(detail_source or "Selected").strip()
                            _src_up = _src_raw.upper()
                            if _src_up == "ACTUAL":
                                _src_label = "Actual Auction"
                            elif _src_up == "AI MODEL":
                                _src_label = "AI Model"
                            else:
                                _src_label = _src_raw or "Selected"
                            _aa_hs = _aa_stats.get("hand_stats") or []
                            _aa_dd = _aa_stats.get("dd_means") or []
                            _aa_hs_map = {
                                str(r.get("Direction") or "").upper(): r
                                for r in _aa_hs
                                if isinstance(r, dict)
                            }
                            _aa_dd_map = {
                                str(r.get("Declarer") or "").upper(): r
                                for r in _aa_dd
                                if isinstance(r, dict)
                            }
                            _aa_rows: list[dict[str, Any]] = []
                            for _dir in ["N", "E", "S", "W"]:
                                _hs_r = _aa_hs_map.get(_dir, {})
                                _dd_r = _aa_dd_map.get(_dir, {})
                                _aa_rows.append(
                                    {
                                        "Source": _src_label,
                                        "Direction": _dir,
                                        "Deals": _aa_n,
                                        "HCP": _hs_r.get("HCP"),
                                        "HCP_std": _hs_r.get("HCP_std"),
                                        "SL_C": _hs_r.get("SL_C"),
                                        "SL_D": _hs_r.get("SL_D"),
                                        "SL_H": _hs_r.get("SL_H"),
                                        "SL_S": _hs_r.get("SL_S"),
                                        "Total_Points": _hs_r.get("Total_Points"),
                                        "Total_Points_std": _hs_r.get("Total_Points_std"),
                                        "DD_C": _dd_r.get("C"),
                                        "DD_D": _dd_r.get("D"),
                                        "DD_H": _dd_r.get("H"),
                                        "DD_S": _dd_r.get("S"),
                                        "DD_NT": _dd_r.get("N"),
                                    }
                                )
                            _aa_df = pd.DataFrame(_aa_rows)
                            st.dataframe(_aa_df, width="stretch")
                else:
                    st.info("No deals found matching this auction pattern.")
            elif display_deals_cache_key in st.session_state:
                # Row was selected, load deals
                with st.spinner("Fetching matching deals..."):
                    try:
                        # Lightweight: sample deals by actual-auction regex (no Rules/BT).
                        auction_upper = display_auction.upper()
                        # Check if selected auction is complete (matches full current auction or ends with -P-P-P)
                        if is_selected_complete:
                            arena_pattern = f"^{auction_upper}$"
                        else:
                            arena_pattern = f"^{auction_upper}.*-P-P-P$"
                        deals_data = api_post(
                            "/sample-deals-by-auction-pattern",
                            {"pattern": arena_pattern, "sample_size": int(max_matching_deals), "seed": int(seed), "include_stats": True},
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
                        notes_df = _auction_builder_load_notes_df(AUCTION_BUILDER_NOTES_CSV)
                        notes_m = _auction_builder_notes_map(notes_df)
                        display_rows = []
                        for deal in sample_deals:
                            try:
                                di = int(deal.get("index")) if deal.get("index") is not None else None
                            except Exception:
                                di = None
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
                                "Note": notes_m.get(di, "") if di is not None else "",
                            })
                        
                        # Cache the results including stats, timing, and total count
                        st.session_state[display_deals_cache_key] = {
                            "sample_deals": sample_deals,
                            "display_rows": display_rows,
                            "stats": stats,
                            "actual_auction_stats": deals_data.get("actual_auction_stats"),
                            "total_count": deals_data.get("total_count", len(sample_deals)),
                            "elapsed_sec": round((deals_data.get("_client_elapsed_ms") or deals_data.get("elapsed_ms") or 0) / 1000, 2),
                        }
                        # Make the sampled-deal stats visible in the Bid-by-Bid Summary for this exact partial.
                        step_stats_map = st.session_state.setdefault("_auction_builder_step_stats", {})
                        step_stats_map[str(display_auction)] = {
                            "_sample_size": int(max_matching_deals),
                            "_seed": int(seed),
                            **dict(stats),
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

            # -------------------------------------------------------------------
            # NEW: BT criteria-matching deals (backs the 'Deals' column).
            # -------------------------------------------------------------------
            try:
                if pinned_deal and selected_step_row is not None:
                    deals_raw = selected_step_row.get("Deals")
                    try:
                        deals_i = int(deals_raw) if deals_raw not in ("", None) else None
                    except Exception:
                        deals_i = None

                    toks = [t.strip().upper() for t in str(display_auction or "").split("-") if t.strip()]
                    last_call = toks[-1] if toks else ""
                    is_contract_like = bool(last_call and len(last_call) >= 2 and last_call[0].isdigit())
                    if deals_i is not None and deals_i > 0 and is_contract_like and len(toks) >= 1:
                        prefix_before = "-".join(toks[:-1]).strip("-")
                        bid_call = last_call

                        # Parity with BT: strip leading passes from the prefix before traversal.
                        try:
                            bt_prefix, _lp = _strip_leading_passes(prefix_before)
                        except Exception:
                            bt_prefix = prefix_before

                        st.subheader("🧩 Deals Matching Auction Criteria")
                        st.caption(
                            f"Deals where BT criteria say the next bid `{bid_call}` is valid after prefix `{bt_prefix}`. "
                            f"(Expected count from 'Deals' column: **{deals_i:,}**.)"
                        )

                        bt_cache_key = f"auction_builder_bt_criteria_deals__{bt_prefix}__{bid_call}__{int(max_bt_criteria_deals)}__{int(seed)}"
                        if bt_cache_key not in st.session_state:
                            st.session_state[bt_cache_key] = None

                        if st.session_state.get(bt_cache_key) is None:
                            with st.spinner("Fetching BT criteria-matching deals..."):
                                resp = api_post(
                                    "/deals-matching-next-bid-criteria",
                                    {
                                        "auction_prefix": bt_prefix,
                                        "bid": bid_call,
                                        "max_rows": int(max_bt_criteria_deals),
                                        "seed": int(seed),
                                    },
                                    timeout=60,
                                )
                                st.session_state[bt_cache_key] = resp

                        resp = st.session_state.get(bt_cache_key) or {}
                        rows = resp.get("rows") or []
                        total_count = resp.get("total_count")
                        if rows:
                            try:
                                total_s = f"{int(total_count):,}" if isinstance(total_count, (int, float)) else "?"
                            except Exception:
                                total_s = "?"
                            st.info(f"Showing **{len(rows):,}** of **{total_s}** BT criteria-matching deals.")
                            render_aggrid(
                                pl.DataFrame(rows),
                                key=f"auction_builder_bt_criteria_deals_grid_{bt_prefix}_{bid_call}",
                                height=calc_grid_height(len(rows), max_height=400),
                                table_name="auction_builder_bt_criteria_deals",
                            )
                        else:
                            st.info("No deals returned for this BT criteria set (or request failed).")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main UI – function selector and controls
# ---------------------------------------------------------------------------

st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
func_choice = st.sidebar.selectbox(
    "Function",
    [
        "AI Model Batch Arena",          # Batch compare AI Model vs Actual auctions
        "Auction Builder",               # Build auction step-by-step with BT lookups
        "Deals by Auction Pattern",      # Primary: find deals matching auction criteria
        "Analyze Actual Auctions",       # Group deals by bid column, analyze outcomes
        "Bidding Arena",                 # Head-to-head model comparison
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

# Function descriptions (WIP)
FUNC_DESCRIPTIONS = {
    "Deals by Auction Pattern": "Find deals matching an auction pattern's criteria. Compare Rules contracts vs actual using DD scores and EV.",
    "Analyze Actual Auctions": "Group deals by their actual auction (bid column). Analyze criteria compliance, score deltas, and outcomes.",
    "Bidding Arena": "Head-to-head model comparison. Compare bidding models (Rules, Actual, NN, etc.) with DD scores, EV, and IMP differentials.",
    "AI Model Batch Arena": "Batch compare AI Model auctions against Actual auctions with DD scores, IMP differentials, and running totals.",
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
                    # AgGrid can render empty inside expanders; use Streamlit's native dataframe here.
                    try:
                        st.dataframe(
                            rules_df.to_pandas(),
                            width="stretch",
                            height=calc_grid_height(len(rules_df), max_height=420, row_height=28, header_height=45),
                        )
                    except Exception:
                        # Fallback: show as text if conversion fails
                        st.write(rules_df)
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

else:
    pass

# ---------------------------------------------------------------------------
# Dispatch to appropriate render function based on selection
# ---------------------------------------------------------------------------

# Track “entered Auction Builder” reliably (before dispatch).
prev_choice = st.session_state.get("_last_func_choice")
st.session_state["_last_func_choice"] = func_choice
st.session_state["_entered_auction_builder"] = (func_choice == "Auction Builder" and prev_choice != "Auction Builder")

match func_choice:
    case "Deals by Auction Pattern":
        render_deals_by_auction_pattern(pattern)
    case "Analyze Actual Auctions":
        render_analyze_actual_auctions()
    case "Bidding Arena":
        render_bidding_arena()
    case "AI Model Batch Arena":
        render_ai_model_batch_arena()
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

# ---------------------------------------------------------------------------
# Developer Settings (at bottom of sidebar)
# ---------------------------------------------------------------------------
with st.sidebar.expander("Developer Settings", expanded=False):
    st.checkbox("Show SQL queries", value=False, key="show_sql_queries")
