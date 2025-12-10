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


def normalize_auction_pattern(pattern: str) -> str:
    """
    Normalize auction regex pattern by appending implied trailing passes.
    
    Bridge auctions end with 3 consecutive passes after any bid/double/redouble.
    If the pattern doesn't end with '-p-p-p', append it (handling regex anchors).
    
    Examples:
        '1n-p-3n' → '1n-p-3n-p-p-p'
        '^1N-p-3N$' → '^1N-p-3N-p-p-p$'
        '1c-p-p-d' → '1c-p-p-d-p-p-p'
        'p-p-p-p' → 'p-p-p-p' (pass-out, already complete)
        '1n-p-3n-p-p-p' → '1n-p-3n-p-p-p' (already complete)
        '.*-3n' → '.*-3n-p-p-p' (wildcards supported)
    """
    import re
    
    if not pattern or not pattern.strip():
        return pattern
    
    pattern = pattern.strip()
    
    # Check for end anchor and temporarily remove it
    has_end_anchor = pattern.endswith('$')
    if has_end_anchor:
        pattern = pattern[:-1]
    
    # Check if already ends with -p-p-p (case insensitive)
    if re.search(r'-[pP]-[pP]-[pP]$', pattern):
        return pattern + ('$' if has_end_anchor else '')
    
    # Check for pass-out pattern (p-p-p-p)
    if re.search(r'^[\^]?[pP]-[pP]-[pP]-[pP]$', pattern):
        return pattern + ('$' if has_end_anchor else '')
    
    # Don't append if pattern ends with open-ended wildcards that could match passes
    # e.g., '.*', '.+', '[^-]*' at the end
    if re.search(r'(\.\*|\.\+|\[[^\]]*\]\*|\[[^\]]*\]\+)$', pattern):
        return pattern + ('$' if has_end_anchor else '')
    
    # Append the trailing passes
    pattern = pattern + '-p-p-p'
    
    return pattern + ('$' if has_end_anchor else '')


def api_get(path: str) -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE}{path}")
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE}{path}", json=payload)
    resp.raise_for_status()
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
    gb.configure_default_column(resizable=True, filter=True, sortable=True)
    # Explicitly set row/header heights to ensure consistent sizing
    gb.configure_grid_options(rowHeight=28, headerHeight=32)
    grid_options = gb.build()

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
st.caption(app_info())

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
            st.write(f"✅ **{file_name}**: {row_count:,} rows")
    
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
# Main UI – function selector and controls
# ---------------------------------------------------------------------------

st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
st.sidebar.header("Settings")
func_choice = st.sidebar.selectbox(
    "Function",
    [
        "Openings by Deal Index",
        "Random Auction Sequences",
        "Auction Sequences Matching",
        "Deals Matching Auction",
        "Bidding Table Statistics",
        "Bid PBN",
    ],
)

# Auction Regex input - shown for pattern-based functions
pattern = None
if func_choice in ["Auction Sequences Matching", "Deals Matching Auction"]:
    raw_pattern = st.sidebar.text_input("Auction Regex", value="^1N-p-3N$",
        help="Trailing '-p-p-p' is assumed if not present (e.g., '1N-p-3N' → '1N-p-3N-p-p-p')")
    pattern = normalize_auction_pattern(raw_pattern)
    if pattern != raw_pattern:
        st.sidebar.caption(f"→ {pattern}")
    st.sidebar.divider()
else:
    st.sidebar.divider()

# Random seed: 0 = non-reproducible, any other value = reproducible
seed = int(st.sidebar.number_input("Random Seed (0=random-random)", value=0, min_value=0))

if func_choice == "Openings by Deal Index":
    #st.sidebar.subheader("Opening Bid Filters")
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

    payload = {
        "sample_size": int(sample_size),
        "seats": seats,
        "directions": directions,
        "opening_directions": opening_directions,
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

            # Display bt_df rows for opening bids (the key data!)
            opening_bids_df = d.get("opening_bids_df", [])
            if opening_bids_df:
                st.write("Opening Bids:")
                render_aggrid(opening_bids_df, key=f"bids_{d['dealer']}_{d['index']}")
            else:
                st.info("No opening bids found.")

            # Display hands as a single row with all 4 hand columns
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

elif func_choice == "Random Auction Sequences":
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)

    payload = {"n_samples": int(n_samples), "seed": seed}
    with st.spinner("Fetching bidding sequences from server. Takes about 10 seconds."):
        data = api_post("/random-auction-sequences", payload)

    samples = data.get("samples", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    if not samples:
        st.info(f"No completed auctions found. ({elapsed_ms/1000:.1f}s)")
    else:
        st.success(f"Found {len(samples)} auction(s) in {elapsed_ms/1000:.1f}s")
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: {s['auction']}")
            render_aggrid(s["sequence"], key=f"seq_random_{i}")
            st.divider()

elif func_choice == "Auction Sequences Matching":
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)

    payload = {"pattern": pattern, "n_samples": int(n_samples), "seed": seed}
    with st.spinner("Fetching auctions from server. Takes about 10 seconds."):
        data = api_post("/auction-sequences-matching", payload)

    st.caption(f"Effective pattern: {data.get('pattern', pattern)}")
    samples = data.get("samples", [])
    elapsed_ms = data.get("elapsed_ms", 0)
    if not samples:
        st.info(f"No auctions matched the pattern. ({elapsed_ms/1000:.1f}s)")
    else:
        st.success(f"Found {len(samples)} auction(s) in {elapsed_ms/1000:.1f}s")
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: {s['auction']}")
            render_aggrid(s["sequence"], key=f"seq_pattern_{i}")
            st.divider()

elif func_choice == "Deals Matching Auction":
    n_auction_samples = st.sidebar.number_input("Auction Samples", value=2, min_value=1)
    n_deal_samples = st.sidebar.number_input("Deal Samples per Auction", value=10, min_value=1)

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
        st.success(f"Found {len(auctions)} auction(s) in {elapsed_ms/1000:.1f}s")
        for i, a in enumerate(auctions, start=1):
            st.subheader(f"Auction {i}: {a['auction']}")
            expr = a.get("expr")
            if expr:
                with st.expander(f"Expr criteria ({len(expr) if isinstance(expr, list) else 1})"):
                    if isinstance(expr, list):
                        df_expr = pl.DataFrame({"Expr": expr})
                        render_aggrid(df_expr, key=f"expr_{i}", height=200)
                    else:
                        st.write(expr)
            criteria_by_seat = a.get("criteria_by_seat")
            if criteria_by_seat:
                rows = []
                for seat, crit_list in criteria_by_seat.items():
                    rows.append({"Seat": seat, "Criteria": crit_list})
                st.write("Criteria by seat:")
                df_criteria = pl.DataFrame(rows)
                render_aggrid(df_criteria, key=f"criteria_{i}", height=220)
            
            # Show distribution SQL if returned
            dist_sql = a.get("dist_sql_query")
            if dist_sql:
                with st.expander("🔍 Distribution SQL Query", expanded=False):
                    st.code(dist_sql, language="sql")
            
            deals = a.get("deals", [])
            if deals:
                deals_df = pl.DataFrame(deals)
                st.write(f"Matching deals (showing {len(deals_df)}):")
                render_aggrid(deals_df, key=f"deals_{i}")
            else:
                st.info("No matching deals (criteria may be too restrictive or distribution filter removed all).")
            st.divider()

elif func_choice == "Bidding Table Statistics":
    st.header("Bidding Table Statistics Viewer")
    st.caption("View bidding table entries with aggregate statistics (mean, std, min, max) for matching deals.")
    
    # Sidebar controls
    st.sidebar.subheader("Statistics Filters")
    
    # Auction regex filter
    raw_auction_pattern = st.sidebar.text_input("Auction Regex", value="^1N-p-3N$",
        help="Trailing '-p-p-p' is assumed if not present (e.g., '1N-p-3N' → '1N-p-3N-p-p-p')")
    auction_pattern = normalize_auction_pattern(raw_auction_pattern)
    if auction_pattern != raw_auction_pattern:
        st.sidebar.caption(f"→ {auction_pattern}")
    
    sample_size = st.sidebar.number_input("Sample Size", value=100, min_value=1, max_value=10000)
    
    # Filter by minimum matching deals (0 = all)
    min_matches = st.sidebar.number_input("Min Matching Deals (0=all)", value=0, min_value=0, max_value=100000)
    
    st.sidebar.divider()
    st.sidebar.subheader("Distribution Filter")
    
    # Seat selector for distribution filter
    dist_seat = st.sidebar.selectbox("Filter Seat", [1, 2, 3, 4], index=0,
        help="Which seat's distribution to filter (S1=opener in most auctions)")
    
    # Ordered distribution pattern input
    dist_pattern = st.sidebar.text_input("Ordered Distribution (S-H-D-C)", value="",
        placeholder="e.g., 5-4-3-1, 5+-4-x-x",
        help="Filter by exact suit order. Leave empty for no filter.")
    
    # Validate and show parsed result for ordered distribution
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
    
    # Sorted shape input (any suit order)
    sorted_shape = st.sidebar.text_input("Sorted Shape (any suit order)", value="",
        placeholder="e.g., 5431, 4432, 5332",
        help="Filter by shape regardless of suit. E.g., '5431' matches any 5-4-3-1 hand.")
    
    # Validate sorted shape
    if sorted_shape:
        parsed_shape = parse_sorted_shape(sorted_shape)
        if parsed_shape:
            st.sidebar.caption(f"→ shape {''.join(map(str, parsed_shape))} (any suits)")
        else:
            st.sidebar.warning("Invalid sorted shape (must be 4 digits summing to 13)")
    
    # Show help in expander
    with st.sidebar.expander("Distribution notation help"):
        st.markdown(format_distribution_help())
    
    # Build payload with distribution filter params (server-side filtering)
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
    
    # Show data availability info
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
            
            # Show distribution filter SQL if applied
            dist_sql_query = data.get("dist_sql_query")
            if dist_sql_query:
                st.info(f"📐 Distribution filter applied (Seat {dist_seat})")
                with st.expander("🔍 Distribution SQL Query", expanded=False):
                    st.code(dist_sql_query, language="sql")
            
            # Reorder columns: put key columns first
            key_cols = ["row_idx", "original_idx", "Auction"]
            if "matching_deal_count" in display_df.columns:
                key_cols.append("matching_deal_count")
            
            other_cols = [c for c in display_df.columns if c not in key_cols]
            display_df = display_df.select([c for c in key_cols if c in display_df.columns] + sorted(other_cols))
            
            # Display summary
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
            
            # Display the data
            st.subheader("Bidding Table with Statistics")
            render_aggrid(display_df, key="bt_stats_grid", height=500)
            
            # Show column info in expander
            with st.expander("Column Descriptions"):
                st.markdown("""
                **Key Columns**:
                - `row_idx`: Display row index
                - `original_idx`: Original row index in source files
                - `Auction`: The auction sequence
                - `matching_deal_count`: Number of deals matching the criteria
                
                **Criteria Columns** (from bt_criteria.parquet):
                - `{col}_min_S{seat}`: Minimum value constraint (e.g. HCP_min_S1)
                - `{col}_max_S{seat}`: Maximum value constraint (e.g. HCP_max_S1)
                
                **Aggregate Statistics** (from bbo_bt_aggregate.parquet):
                - `{col}_mean_S{seat}`: Mean value across matching deals (e.g. HCP_mean_S1)
                - `{col}_std_S{seat}`: Standard deviation (e.g. HCP_std_S1)
                - `{col}_min_S{seat}`: Minimum observed value
                - `{col}_max_S{seat}`: Maximum observed value
                
                **Columns**: HCP, SL_C, SL_D, SL_H, SL_S, Total_Points (for each of 4 seats: S1-S4)
                """)

elif func_choice == "Bid PBN":
    st.header("Bid PBN - Analyze Deal and Find Matching Auctions")
    st.caption("Enter a PBN deal string or URL to analyze the deal and find matching auctions from the bidding table.")
    
    # Input options
    input_type = st.sidebar.radio("Input Type", ["PBN String", "PBN URL"])
    
    if input_type == "PBN String":
        pbn_input = st.sidebar.text_area(
            "PBN Deal String",
            value="N:AK65.KQ2.A54.K32 Q82.JT95.K62.A87 JT97.A843.QJ3.54 43.76.T987.QJT96",
            height=100,
            help="Enter a PBN deal string like 'N:AK65.KQ2.A54.K32 Q82.JT95.K62.A87 ...'"
        )
    else:
        pbn_input = st.sidebar.text_input(
            "PBN URL",
            value="",
            placeholder="https://example.com/deals.pbn",
            help="Enter a URL to a PBN file (processes ALL deals)"
        )
    
    # Seat selection for auction matching
    match_seat = st.sidebar.selectbox("Match Auction Seat", [1, 2, 3, 4], index=0,
        help="Which seat's criteria to match against the deal")
    
    max_auctions = st.sidebar.number_input("Max Auctions to Show", value=50, min_value=1, max_value=500)
    
    # Checkbox for par calculation
    include_par = st.sidebar.checkbox("Calculate Par Score", value=True,
        help="Calculate par score using double-dummy analysis (server-side)")
    
    # Vulnerability for par calculation
    vul_option = st.sidebar.selectbox("Vulnerability", ["None", "Both", "NS", "EW"], index=0,
        help="Vulnerability for par score calculation")
    
    # Process button
    if st.sidebar.button("Analyze Deal(s)", type="primary"):
        if not pbn_input:
            st.warning("Please enter a PBN deal string or URL.")
        else:
            # Call the API to process PBN (handles both string and URL)
            with st.spinner("Processing PBN deal(s)..."):
                try:
                    pbn_payload = {
                        "pbn": pbn_input,
                        "include_par": include_par,
                        "vul": vul_option,
                    }
                    pbn_data = api_post("/process-pbn", pbn_payload)
                except Exception as e:
                    st.error(f"Failed to process PBN: {e}")
                    st.stop()
            
            deals = pbn_data.get("deals", [])
            if not deals:
                st.warning("No valid deals found.")
                st.stop()
            
            st.success(f"Processed {len(deals)} deal(s) in {pbn_data.get('elapsed_ms', 0)/1000:.1f}s")
            
            # Process each deal
            for deal_idx, deal in enumerate(deals):
                # Check for errors
                if "error" in deal:
                    st.warning(f"Deal {deal_idx + 1}: {deal['error']}")
                    continue
                
                dealer = deal.get('Dealer', '?')
                
                # Display deal header
                st.divider()
                st.subheader(f"📋 Deal {deal_idx + 1}: Dealer {dealer}")
                
                # Display par info if available
                if include_par and 'Par_Score' in deal:
                    par_score = deal.get('Par_Score')
                    par_contract = deal.get('Par_Contract', 'N/A')
                    st.write(f"**Par:** {par_score} ({par_contract})")
                
                # Display hands in a compact format
                hands_str = " | ".join([f"{d}: {deal.get(f'Hand_{d}', 'N/A')}" for d in 'NESW'])
                st.write(f"**Hands:** {hands_str}")
                
                # Display full deal DataFrame
                deal_df = pl.DataFrame([deal])
                with st.expander(f"📊 Deal {deal_idx + 1} Features", expanded=(deal_idx == 0)):
                    render_aggrid(deal_df, key=f"deal_features_{deal_idx}")
                
                # Find matching auctions using API
                # Determine which direction corresponds to the seat
                directions = ['N', 'E', 'S', 'W']
                dealer_idx_val = directions.index(dealer) if dealer in directions else 0
                match_direction = directions[(dealer_idx_val + match_seat - 1) % 4]
                
                # Get hand features for the matched direction
                hcp = deal.get(f'HCP_{match_direction}')
                sl_s = deal.get(f'SL_S_{match_direction}')
                sl_h = deal.get(f'SL_H_{match_direction}')
                sl_d = deal.get(f'SL_D_{match_direction}')
                sl_c = deal.get(f'SL_C_{match_direction}')
                tp = deal.get(f'Total_Points_{match_direction}')
                
                if all(v is not None for v in [hcp, sl_s, sl_h, sl_d, sl_c, tp]):
                    try:
                        auction_payload = {
                            "hcp": hcp,
                            "sl_s": sl_s,
                            "sl_h": sl_h,
                            "sl_d": sl_d,
                            "sl_c": sl_c,
                            "total_points": tp,
                            "seat": match_seat,
                            "max_results": max_auctions,
                        }
                        auction_data = api_post("/find-matching-auctions", auction_payload)
                        
                        # Display SQL query
                        sql_query = auction_data.get("sql_query", "")
                        with st.expander(f"🔍 SQL Query (Seat {match_seat} = {match_direction})", expanded=False):
                            st.code(sql_query, language="sql")
                            st.caption(f"Criteria: HCP={hcp}, SL_S={sl_s}, SL_H={sl_h}, SL_D={sl_d}, SL_C={sl_c}, TP={tp}")
                        
                        auctions = auction_data.get("auctions", [])
                        if auctions:
                            st.success(f"Found {len(auctions)} matching auctions for Deal {deal_idx + 1} ({auction_data.get('elapsed_ms', 0)/1000:.1f}s)")
                            auctions_df = pl.DataFrame(auctions)
                            render_aggrid(auctions_df, key=f"matching_auctions_{deal_idx}", height=300)
                        else:
                            st.info(f"No matching auctions found for Deal {deal_idx + 1}")
                    except Exception as e:
                        st.error(f"Error finding auctions for Deal {deal_idx + 1}: {e}")
                else:
                    st.warning(f"Missing hand features for Deal {deal_idx + 1}, cannot match auctions")
