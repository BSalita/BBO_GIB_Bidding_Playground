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
import polars as pl
import duckdb
import endplay
import fastapi
import uvicorn
import pandas as pd
import time
import requests
from datetime import datetime, timezone
import pathlib
import os
import sys
import re
from typing import Any, Dict

# Add mlBridgeLib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlBridgeLib"))

from bbo_bidding_queries_lib import normalize_auction_pattern


API_BASE = "http://127.0.0.1:8000"


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


def render_aggrid(records: Any, key: str, height: int | None = None) -> None:
    """Render a list-of-dicts or DataFrame using AgGrid."""
    if records is None:
        st.info("No data.")
        return
    if isinstance(records, pl.DataFrame):
        df = records
    else:
        try:
            df = pl.DataFrame(records)
        except Exception:
            st.json(records)
            return
    if df.is_empty():
        st.info("No rows to display.")
        return

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

    gb = GridOptionsBuilder.from_dataframe(df.to_pandas())
    # Disable pagination entirely to allow scrolling within the fixed height
    gb.configure_pagination(enabled=False)
    # Make all columns read-only (not editable), resizable, filterable, sortable
    gb.configure_default_column(resizable=True, filter=True, sortable=True, editable=False)
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

    AgGrid(
        df.to_pandas(),
        gridOptions=grid_options,
        height=height,
        theme="balham",
        key=key,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
    )

if st.session_state.get("first_run", True):
    st.session_state.app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    st.session_state.first_run = False

def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(f"Python:{'.'.join(map(str, sys.version_info[:3]))} API:{fastapi.__version__} Uvicorn:{uvicorn.__version__} pandas:{pl.__version__} polars:{pl.__version__} endplay:{endplay.__version__}")
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

    with st.spinner("Fetching Openings by Deal Index from server. Takes about 20 seconds."):
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
            if opening_bids_df:
                st.write("Opening Bids:")
                # Rank bids best→worst using reverse sort order of Auction (per request)
                try:
                    bids_df = pl.DataFrame(opening_bids_df)
                    if "Auction" in bids_df.columns:
                        # Sort by Seat ascending, then Auction descending
                        if "seat" in bids_df.columns:
                            bids_df = bids_df.sort(["seat", "Auction"], descending=[False, True])
                        else:
                            bids_df = bids_df.sort("Auction", descending=True)
                        bids_df = bids_df.with_row_index("Best_Bid").with_columns(
                            (pl.col("Best_Bid") + 1).alias("Best_Bid")
                        )
                        # Put Best_Bid first for readability
                        front = [
                            c
                            for c in [
                                "Best_Bid",
                                "index",
                                "Auction",
                                "seat",
                                "Dealer",
                                "bid",
                                "Actual_Contract",
                                "AI_Contract",
                                "DD_Score_Declarer",
                                "EV_Score_Declarer",
                                "ParScore",
                                "ParContract",
                                "EV_ParContracts",
                                "Expr",
                            ]
                            if c in bids_df.columns
                        ]
                        rest = [c for c in bids_df.columns if c not in front]
                        bids_df = bids_df.select(front + rest)
                    render_aggrid(bids_df, key=f"bids_{d['dealer']}_{d['index']}")
                except Exception:
                    # Fallback: render raw payload
                    render_aggrid(opening_bids_df, key=f"bids_{d['dealer']}_{d['index']}")
            else:
                st.info("No opening bids found.")

            if d.get("hands"):
                st.write("Hands:")
                hands_dict = d["hands"]
                df_hands = pl.DataFrame([{
                    "Hand_N": hands_dict.get("Hand_N"),
                    "Hand_E": hands_dict.get("Hand_E"),
                    "Hand_S": hands_dict.get("Hand_S"),
                    "Hand_W": hands_dict.get("Hand_W"),
                }])
                render_aggrid(df_hands, key=f"hands_{d['dealer']}_{d['index']}")
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
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: {s['auction']}")
            render_aggrid(s["sequence"], key=f"seq_random_{i}")
            st.divider()


def render_find_auction_sequences(pattern: str | None):
    """Search for auction sequences matching a regex pattern."""
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)

    # Random seed at bottom of sidebar
    st.sidebar.divider()
    seed = int(st.sidebar.number_input("Random Seed (0=random)", value=0, min_value=0, key="seed_find"))

    payload = {"pattern": pattern, "n_samples": int(n_samples), "seed": seed}
    with st.spinner("Fetching auctions from server. Takes about 10 seconds."):
        data = api_post("/auction-sequences-matching", payload)

    st.caption(f"Effective pattern: {data.get('pattern', pattern)}")
    samples = data.get("samples", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    if not samples:
        st.info(f"No auctions matched the pattern. ({elapsed_ms/1000:.1f}s)")
    else:
        st.success(f"Found {len(samples)} matching auction(s) in {elapsed_ms/1000:.1f}s")
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: {s['auction']}")
            render_aggrid(s["sequence"], key=f"seq_pattern_{i}")
            st.divider()
    
    criteria_rejected = data.get("criteria_rejected", [])
    if criteria_rejected:
        with st.expander(f"🚫 Rejected auctions due to custom criteria. {len(criteria_rejected)} shown.", expanded=False):
            st.caption("Rows filtered out by bbo_custom_auction_criteria.csv rules.")
            try:
                rejected_df = pl.DataFrame(criteria_rejected)
                st.dataframe(rejected_df.to_pandas(), width='stretch')
            except Exception as e:
                st.warning(f"Could not render as table: {e}")
                st.json(criteria_rejected)


def render_deals_by_auction_pattern(pattern: str | None):
    """Find deals matching an auction pattern's criteria."""
    n_auction_samples = st.sidebar.number_input("Auction Samples", value=2, min_value=1, max_value=10)
    n_deal_samples = st.sidebar.number_input("Deal Samples per Auction", value=10, min_value=1, max_value=1000)

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
        "n_auction_samples": int(n_auction_samples),
        "n_deal_samples": int(n_deal_samples),
        "seed": seed,
        "dist_pattern": deal_dist_pattern if deal_dist_pattern else None,
        "sorted_shape": deal_sorted_shape if deal_sorted_shape else None,
        "dist_direction": deal_dist_direction,
    }

    with st.spinner("Fetching Deals Matching Auction from server. Takes about 10 seconds."):
        data = api_post("/deals-matching-auction", payload)

    elapsed_ms = data.get("elapsed_ms", 0)
    st.caption(f"Effective pattern: {data.get('pattern', pattern)}")
    
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
            seat_roles = {1: "Opener/Dealer", 2: "LHO", 3: "Partner", 4: "RHO"}
            st.caption(f"ℹ️ Row seat={row_seat}, Actual final seat={actual_final_seat}. "
                      f"Seat 1=Dealer, Seat 2=LHO, Seat 3=Partner, Seat 4=RHO")
            
            if missing:
                with st.expander(f"⚠️ Missing Criteria ({sum(len(v) for v in missing.values())} total)", expanded=True):
                    st.warning("These criteria could not be matched to pre-computed bitmaps - filtering may be incomplete!")
                    for key, criteria_list in missing.items():
                        seat_num = int(key.split('_')[1]) if '_' in key else 0
                        role = seat_roles.get(seat_num, "")
                        st.write(f"**{key}** ({role}): {', '.join(criteria_list)}")
            if found:
                with st.expander(f"✅ Applied Criteria ({sum(len(v) for v in found.values())} total)", expanded=False):
                    for key in sorted(found.keys()):
                        criteria_list = found[key]
                        seat_num = int(key.split('_')[1]) if '_' in key else 0
                        role = seat_roles.get(seat_num, "")
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
                    makes_diff_str = f"+{makes_diff:.1f}%" if makes_diff >= 0 else f"{makes_diff:.1f}%"
                    stats_rows.append({"Metric": "DD Makes", "AI": f"{ai_makes_pct:.1f}%", "Actual": f"{contract_makes_pct:.1f}%", "Diff": makes_diff_str})
                
                # Par Achieved row (only if data exists)
                if ai_par > 0 or contract_par > 0:
                    par_diff = ai_par_pct - contract_par_pct
                    par_diff_str = f"+{par_diff:.1f}%" if par_diff >= 0 else f"{par_diff:.1f}%"
                    stats_rows.append({"Metric": "Par Achieved", "AI": f"{ai_par_pct:.1f}%", "Actual": f"{contract_par_pct:.1f}%", "Diff": par_diff_str})
                
                # EV row (only if data exists)
                if avg_ev_ai is not None and avg_ev_contract is not None:
                    ev_diff_str = f"+{avg_ev_diff:.2f}" if avg_ev_diff >= 0 else f"{avg_ev_diff:.2f}"
                    stats_rows.append({"Metric": "Avg EV", "AI": f"{avg_ev_ai:.2f}", "Actual": f"{avg_ev_contract:.2f}", "Diff": ev_diff_str})
                
                if stats_rows:
                    stats_df = pl.DataFrame(stats_rows)
                    render_aggrid(stats_df, key=f"stats_{i}", height=150)
            
            # Contract summary table (detailed breakdown by contract)
            contract_summary = a.get("contract_summary", [])
            if contract_summary:
                st.write("**Contract Summary:**")
                summary_df = pl.DataFrame(contract_summary)
                render_aggrid(summary_df, key=f"contract_summary_{i}", height=200)
            
            deals = a.get("deals", [])
            if deals:
                deals_df = pl.DataFrame(deals)
                st.write(f"**Matching Deals:** (showing {len(deals_df)})")
                render_aggrid(deals_df, key=f"deals_{i}")
            else:
                st.info("No matching deals (criteria may be too restrictive or distribution filter removed all).")
            st.divider()
    
    # Show criteria-rejected rows for debugging (from criteria.csv)
    criteria_rejected = data.get("criteria_rejected", [])
    if criteria_rejected:
        with st.expander(f"🚫 Rejected auctions due to custom criteria. {len(criteria_rejected)} shown.", expanded=False):
            st.caption("Rows filtered out by bbo_custom_auction_criteria.csv rules.")
            try:
                rejected_df = pl.DataFrame(criteria_rejected)
                st.dataframe(rejected_df.to_pandas(), width='stretch')
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
    auction_pattern = normalize_auction_pattern(raw_auction_pattern)
    if auction_pattern != raw_auction_pattern:
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
            render_aggrid(display_df, key="bt_stats_grid", height=500)
            
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
        
        with st.expander(f"**Deal {deal_idx + 1}**: {pbn[:50]}..." if len(pbn) > 50 else f"**Deal {deal_idx + 1}**: {pbn}", expanded=(deal_idx == 0)):
            if par_score is not None:
                st.info(f"🎯 **Par Score**: {par_score} ({par_contracts}) | Vul: {vul_option}")
            
            st.write("**Hands:**")
            hands_row = {f"Hand_{d}": hands.get(d, "") for d in "NESW"}
            render_aggrid(pl.DataFrame([hands_row]), key=f"hands_deal_{deal_idx}", height=80)
            
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
                render_aggrid(pl.DataFrame(stats_rows), key=f"stats_deal_{deal_idx}", height=150)
            
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
                render_aggrid(matches_df, key=f"matches_deal_{deal_idx}", height=min(400, 50 + len(matches) * 35))
        
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
    
    if count > 0:
        st.success(f"✅ PBN found in database! ({elapsed/1000:.1f}s, searched {total:,} rows)")
        
        # Use first returned match as the primary deal info
        deal_info = matches[0] if matches and isinstance(matches[0], dict) else {}
        if deal_info:
            st.write("**Deal Information:**")
            
            key_cols = ["PBN", "Vul", "Dealer", "bid", "Contract", "Result", "Tricks", "Score", "ParScore", "DD_Score_Declarer"]
            key_info = {k: v for k, v in deal_info.items() if k in key_cols and v is not None}
            if key_info:
                render_aggrid(pl.DataFrame([key_info]), key="pbn_lookup_key", height=80)
            
            with st.expander("Full Deal Data", expanded=False):
                all_info_df = pl.DataFrame([deal_info])
                render_aggrid(all_info_df, key="pbn_lookup_full", height=200)

        # If multiple matches exist, show a small sample list for clarity
        if count > 1:
            with st.expander(f"All matches ({count:,})", expanded=False):
                render_aggrid(pl.DataFrame(matches), key="pbn_lookup_matches", height=250)
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
        help="Regex pattern to filter auctions. Trailing '-p-p-p' appended if not present. Use .* for all.",
        key="group_by_bid_regex"
    )
    auction_regex = normalize_auction_pattern(raw_auction_regex)
    if auction_regex != raw_auction_regex:
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
        
        label = f"**{bid_auction}**"
        if bt_auction and bt_auction != bid_auction:
            label += f" → {bt_auction}"
        label += f" ({deal_count:,} deals, {sample_count} shown)"
        
        with st.expander(label, expanded=(i == 0)):
            if bt_info:
                seat_roles = {1: "Opener/Dealer", 2: "LHO", 3: "Partner", 4: "RHO"}
                
                total_criteria = 0
                for s in range(1, 5):
                    agg_col = f"Agg_Expr_Seat_{s}"
                    if agg_col in bt_info and bt_info[agg_col]:
                        total_criteria += len(bt_info[agg_col])
                
                with st.expander(f"📋 Bidding Table Criteria ({total_criteria} total)", expanded=False):
                    for s in range(1, 5):
                        agg_col = f"Agg_Expr_Seat_{s}"
                        role = seat_roles.get(s, "")
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
                        dir_df = dir_df.with_columns(pl.col("HCP_avg").cast(pl.Float64).round(1))
                    if "Total_Points_avg" in dir_df.columns:
                        dir_df = dir_df.with_columns(pl.col("Total_Points_avg").cast(pl.Float64).round(1))
                    render_aggrid(dir_df, key=f"group_by_bid_stats_{i}", height=160)

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
                    render_aggrid(score_df, key=f"group_by_bid_score_stats_{i}", height=140)
            
            if deals:
                st.write(f"**Sample Deals** ({len(deals)} shown):")
                deals_df = pl.DataFrame(deals)
                
                priority_cols = ["PBN", "Hand_S", "Hand_N", "Vul", "Contract", "Result", "Score", 
                               "Score_MP", "Score_MP_Pct", "DD_Score_Declarer", "ParScore"]
                available_priority = [c for c in priority_cols if c in deals_df.columns]
                other_cols = [c for c in deals_df.columns if c not in priority_cols]
                ordered_cols = available_priority + other_cols
                    
                deals_df = deals_df.select(ordered_cols)
                
                drop_cols = [c for c in ["Auction", "Board_ID"] if c in deals_df.columns]
                if drop_cols:
                    deals_df = deals_df.drop(drop_cols)
                
                render_aggrid(deals_df, key=f"group_by_bid_deals_{i}", height=min(400, 50 + len(deals) * 35))


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
        "Analyze Deal (PBN/LIN)",        # Input a deal, find matching auctions
        "Bidding Table Explorer",        # Browse bt_df with statistics
        "Find Auction Sequences",        # Regex search bt_df
        "PBN Database Lookup",           # Check if PBN exists in deal_df
        "Random Auction Samples",        # Random completed auctions
        "Opening Bids by Deal",          # Browse deals, see opening bid matches
    ],
)

# Function descriptions (WIP)
FUNC_DESCRIPTIONS = {
    "Deals by Auction Pattern": "Find deals matching an auction pattern's criteria. Compare AI contracts vs actual using DD scores and EV.",
    "Analyze Actual Auctions": "Group deals by their actual auction (bid column). Analyze criteria compliance, score deltas, and outcomes.",
    "Analyze Deal (PBN/LIN)": "Input a PBN/LIN deal and find which bidding table auctions match the hand characteristics.",
    "Bidding Table Explorer": "Browse bidding table entries with aggregate statistics (min/max ranges) for hand criteria per auction.",
    "Find Auction Sequences": "Search for auction sequences matching a regex pattern. Shows criteria per seat.",
    "PBN Database Lookup": "Check if a specific PBN deal exists in the database. Returns game results if found.",
    "Random Auction Samples": "View random completed auction sequences from the bidding table.",
    "Opening Bids by Deal": "Browse deals by index and see which opening bids match based on pre-computed criteria.",
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
    case "Analyze Deal (PBN/LIN)":
        render_analyze_deal()
    case "Bidding Table Explorer":
        render_bidding_table_explorer()
    case "Find Auction Sequences":
        render_find_auction_sequences(pattern)
    case "PBN Database Lookup":
        render_pbn_database_lookup()
    case "Random Auction Samples":
        render_random_auction_samples()
    case "Opening Bids by Deal":
        render_opening_bids_by_deal()
    case _:
        st.error(f"Unknown function: {func_choice}")

