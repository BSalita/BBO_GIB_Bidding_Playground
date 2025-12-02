"""Streamlit frontend for BBO GIB Bidding Playground.

All heavy work is delegated to the FastAPI service defined in
`bbo_bidding_queries_api.py`.

Run the backend first (takes about 8-10 minutes to complete):

    python bbo_bidding_queries_api.py

Then run this app (may wait for the backend to finish loading data):

    streamlit run bbo_bidding_queries_streamlit.py
"""

from __future__ import annotations

import time
from typing import Any, Dict
import os
import sys
import pandas as pd
import requests
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from datetime import datetime, timezone
import pathlib

import polars as pl
import endplay
import fastapi
import uvicorn


API_BASE = "http://127.0.0.1:8000"


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
        fit_columns_on_grid_load=True,
    )

if st.session_state.get("first_run", True):
    st.session_state.app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
    st.session_state.first_run = False

def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(f"Python:{'.'.join(map(str, sys.version_info[:3]))} API:{fastapi.__version__} Uvicorn:{uvicorn.__version__} pandas:{pl.__version__} polars:{pl.__version__} endplay:{endplay.__version__}")
    return

st.set_page_config(layout="wide")
st.title("BBO GIB Bidding Playground")
st.caption(app_info())

# ---------------------------------------------------------------------------
# Backend initialization / maintenance gate
# ---------------------------------------------------------------------------

try:
    status = api_get("/status")
except Exception as exc:  # pragma: no cover - connectivity issues
    st.error(f"Cannot reach backend at {API_BASE}: {exc}")
    st.stop()

if not status["initialized"]:
    st.header("Maintenance in progress")
    st.write("Backend is loading data; please wait...")
    st.json(status)

    # Auto-poll: wait then rerun to check if ready
    with st.spinner("Waiting for backend to finish loading..."):
        time.sleep(30)
    st.rerun()

if status.get("error"):
    st.warning(f"Backend reported an initialization error: {status['error']}")

# Display dataset info
bt_df_rows = status.get("bt_df_rows")
deal_df_rows = status.get("deal_df_rows")
if bt_df_rows is not None and deal_df_rows is not None:
    st.info(f"📊 Loaded data: **{deal_df_rows:,}** deals, **{bt_df_rows:,}** bidding table entries")

# ---------------------------------------------------------------------------
# Main UI – function selector and controls
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
func_choice = st.sidebar.selectbox(
    "Function",
    [
        "Opening Bid Details",
        "Bidding Sequences (Random)",
        "Auctions Matching Pattern",
        "Deals for Auction Pattern",
    ],
)

# Optional fixed seed: when disabled, backend will use nondeterministic sampling
use_seed = st.sidebar.checkbox("Use fixed random seed", value=True)
seed = None
if use_seed:
    seed = int(st.sidebar.number_input("Random Seed", value=42))

if func_choice == "Opening Bid Details":
    st.sidebar.subheader("Opening Bid Filters")
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

    with st.spinner("Fetching opening bid details from backend. Takes about 20 seconds..."):
        data = api_post("/opening-bid-details", payload)

    deals = data.get("deals", [])
    if not deals:
        st.info("No deals matched the specified filters.")
    else:
        for d in deals:
            st.subheader(f"Dealer {d['dealer']} – deal index {d['index']}")
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

elif func_choice == "Bidding Sequences (Random)":
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)

    payload = {"n_samples": int(n_samples), "seed": seed}
    with st.spinner("Fetching bidding sequences from backend..."):
        data = api_post("/bidding-sequences", payload)

    samples = data.get("samples", [])
    if not samples:
        st.info("No completed auctions found.")
    else:
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: Auction='{s['auction']}', seat={s['seat']}")
            render_aggrid(s["sequence"], key=f"seq_random_{i}")
            st.divider()

elif func_choice == "Auctions Matching Pattern":
    pattern = st.sidebar.text_input("Auction Pattern (Prefix or Regex)", value="^1N-p-3N$")
    n_samples = st.sidebar.number_input("Number of Samples", value=5, min_value=1)

    payload = {"pattern": pattern, "n_samples": int(n_samples), "seed": seed}
    with st.spinner("Fetching auctions from backend..."):
        data = api_post("/auctions-matching", payload)

    st.caption(f"Effective pattern: {data.get('pattern', pattern)}")
    samples = data.get("samples", [])
    if not samples:
        st.info("No auctions matched the pattern.")
    else:
        for i, s in enumerate(samples, start=1):
            st.subheader(f"Sample {i}: Auction='{s['auction']}', seat={s['seat']}")
            render_aggrid(s["sequence"], key=f"seq_pattern_{i}")
            st.divider()

elif func_choice == "Deals for Auction Pattern":
    pattern = st.sidebar.text_input("Auction Pattern", value="^1N-p-3N$")
    n_auction_samples = st.sidebar.number_input("Auction Samples", value=2, min_value=1)
    n_deal_samples = st.sidebar.number_input("Deal Samples per Auction", value=10, min_value=1)

    payload = {
        "pattern": pattern,
        "n_auction_samples": int(n_auction_samples),
        "n_deal_samples": int(n_deal_samples),
        "seed": seed,
    }

    with st.spinner("Fetching deals for auction pattern from backend..."):
        data = api_post("/deals-for-auction", payload)

    st.caption(f"Effective pattern: {data.get('pattern', pattern)}")
    auctions = data.get("auctions", [])
    if not auctions:
        st.info("No auctions matched the pattern.")
    else:
        for i, a in enumerate(auctions, start=1):
            st.subheader(f"Auction {i}: '{a['auction']}' (seat {a['seat']})")
            expr = a.get("expr")
            if expr:
                # Expr is typically a list of criteria strings; show as a small DataFrame
                # inside an expander so it is easier to scan and sort.
                with st.expander(f"Expr criteria ({len(expr) if isinstance(expr, list) else 1})"):
                    if isinstance(expr, list):
                        df_expr = pl.DataFrame({"Expr": expr})
                        render_aggrid(df_expr, key=f"expr_{i}", height=200)
                    else:
                        st.write(expr)
            criteria_by_seat = a.get("criteria_by_seat")
            if criteria_by_seat:
                # Convert criteria_by_seat dict into rows: Seat, Criteria (list)
                rows = []
                for seat, crit_list in criteria_by_seat.items():
                    # seat keys may be strings; keep as-is for display
                    rows.append({"Seat": seat, "Criteria": crit_list})
                st.write("Criteria by seat:")
                df_criteria = pl.DataFrame(rows)
                render_aggrid(df_criteria, key=f"criteria_{i}", height=220)
            deals = a.get("deals", [])
            if deals:
                st.write(f"Matching deals (showing up to {len(deals)}):")
                render_aggrid(deals, key=f"deals_{i}")
            else:
                st.info("No matching deals (criteria may be too restrictive).")
            st.divider()
