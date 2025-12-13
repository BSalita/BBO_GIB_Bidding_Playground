"""
FastAPI server for BBO bidding queries.

This service loads bidding data (bt_df and deal_df), exposes HTTP endpoints that Streamlit (or other clients) can call.

Heavy initialization (loading Parquet files, building criteria bitmaps,
computing opening-bid candidates) is performed once in the background and the
results are kept in-process. Startup takes ~8-10 minutes.

Usage:
    python bbo_bidding_queries_api.py
"""

from __future__ import annotations

import gc
import os
import random
import re
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import pathlib

# Add mlBridgeLib to path so its internal imports work (append, not insert, to avoid shadowing the package)
sys.path.append(os.path.join(os.path.dirname(__file__), "mlBridgeLib"))

import psutil


def _fast_exit_handler(signum, frame):
    """Exit immediately without Python's slow garbage collection cleanup."""
    print("\n[shutdown] Received signal, exiting immediately (skipping GC cleanup). Takes about 20 seconds.")
    os._exit(0)


# Register fast exit for SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, _fast_exit_handler)
signal.signal(signal.SIGTERM, _fast_exit_handler)


def _log_memory(label: str) -> None:
    """Log current memory usage."""
    process = psutil.Process()
    mem = process.memory_info()
    rss_gb = mem.rss / (1024 ** 3)
    vms_gb = mem.vms / (1024 ** 3)
    print(f"[mem] {label}: RSS={rss_gb:.1f}GB, VMS={vms_gb:.1f}GB")

from contextlib import asynccontextmanager

import polars as pl
import duckdb  # pyright: ignore[reportMissingImports]
import requests as http_requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from endplay.types import Deal, Vul, Player
from endplay.dds import calc_dd_table, par

from mlBridgeLib.mlBridgeBiddingLib import (
    DIRECTIONS,
    load_bt_df,
    load_deal_df,
    load_execution_plan_data,
    directional_to_directionless,
    build_or_load_directional_criteria_bitmaps,
    process_opening_bids,
)

from bbo_bidding_queries_lib import (
    calculate_imp,
    normalize_auction_pattern,
    parse_contract_from_auction,
    get_declarer_for_auction,
    get_ai_contract,
    get_dd_score_for_auction,
    get_ev_for_auction,
    compute_hand_features,
    compute_par_score,
    parse_pbn_deal,
    parse_distribution_pattern,
    parse_sorted_shape,
    build_distribution_sql_for_bt,
    build_distribution_sql_for_deals,
    add_suit_length_columns,
    evaluate_criterion_for_hand,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_seed(seed: int | None) -> int | None:
    """Convert seed=0 to None (non-reproducible) for Polars .sample()."""
    if seed is None:
        return None
    return None if seed == 0 else seed


# ---------------------------------------------------------------------------
# Data directory resolution (supports --data-dir command line arg)
# ---------------------------------------------------------------------------

def _parse_data_dir_arg() -> pathlib.Path | None:
    """Parse --data-dir from sys.argv early (before full argparse)."""
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            return pathlib.Path(sys.argv[i + 1])
        if arg.startswith("--data-dir="):
            return pathlib.Path(arg.split("=", 1)[1])
    return None


def _parse_deal_rows_arg() -> int | None:
    """Parse --deal-rows from sys.argv early (before full argparse).
    
    Use this to limit deal_df rows for faster debugging startup.
    """
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == "--deal-rows" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith("--deal-rows="):
            return int(arg.split("=", 1)[1])
    return None


# Check for --data-dir command line argument
_cli_data_dir = _parse_data_dir_arg()
# Check for --deal-rows (limit deal_df rows for faster debugging)
_cli_deal_rows = _parse_deal_rows_arg()
if _cli_deal_rows is not None:
    print(f"DEBUG MODE: Limiting deal_df to {_cli_deal_rows:,} rows")

# Default to 'data' subdirectory if --data-dir not specified
dataPath = _cli_data_dir if _cli_data_dir is not None else pathlib.Path("data")

if not dataPath.exists():
    print(f"WARNING: Data directory does not exist: {dataPath}")
else:
    print(f"dataPath: {dataPath} (exists)")


# ---------------------------------------------------------------------------
# Required files check
# ---------------------------------------------------------------------------

exec_plan_file = dataPath.joinpath("bbo_bt_execution_plan_data.pkl")
bbo_mldf_augmented_file = dataPath.joinpath("bbo_mldf_augmented.parquet")
bbo_bidding_table_augmented_file = dataPath.joinpath("bbo_bt_augmented.parquet")
bt_aggregates_file = dataPath.joinpath("bbo_bt_aggregate.parquet")
auction_criteria_file = dataPath.joinpath("bbo_custom_auction_criteria.csv")


# ---------------------------------------------------------------------------
# Dynamic auction criteria filtering (from criteria.csv)
# ---------------------------------------------------------------------------

def _load_auction_criteria() -> list[tuple[str, list[str]]]:
    """Load criteria.csv and return list of (partial_auction, [criteria...]).
    
    CSV format: partial_auction,criterion1,criterion2,...
    Example row: 1c,SL_S >= SL_H,HCP >= 12
    """
    if not auction_criteria_file.exists():
        return []
    
    import csv
    criteria_list = []
    try:
        with open(auction_criteria_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'):  # Skip empty lines and comments
                    continue
                partial_auction = row[0].strip().lower()
                criteria = [c.strip() for c in row[1:] if c.strip()]
                if partial_auction and criteria:
                    criteria_list.append((partial_auction, criteria))
    except Exception as e:
        print(f"[auction-criteria] Error loading {auction_criteria_file}: {e}")
    
    return criteria_list


def _get_direction_for_partial_auction(partial_auction: str, dealer: str) -> str:
    """Get the direction (N/E/S/W) for the bidder of the last bid in partial_auction.
    
    The seat number = number of dashes + 1
    Direction = ['N', 'E', 'S', 'W'][(dealer_index + seat - 1) % 4]
    """
    directions = ['N', 'E', 'S', 'W']
    dealer_idx = directions.index(dealer.upper()) if dealer.upper() in directions else 0
    num_dashes = partial_auction.count('-')
    seat = num_dashes + 1  # 1-based seat number
    direction_idx = (dealer_idx + seat - 1) % 4
    return directions[direction_idx]


def _parse_criterion_to_polars(criterion: str, direction: str) -> pl.Expr | None:
    """Parse a criterion string like 'SL_S >= SL_H' into a Polars expression.
    
    Supports:
    - Column comparisons: SL_S >= SL_H, HCP > Total_Points
    - Value comparisons: HCP >= 12, SL_S <= 5
    - Operators: >=, <=, >, <, ==, !=
    """
    
    # Match: COL OP COL or COL OP VALUE
    pattern = r'(\w+)\s*(>=|<=|>|<|==|!=)\s*(\w+|\d+)'
    match = re.match(pattern, criterion.strip())
    if not match:
        print(f"[auction-criteria] Could not parse criterion: {criterion}")
        return None
    
    left, op, right = match.groups()
    
    # Append direction to column names if they're bridge columns
    bridge_cols = ['SL_S', 'SL_H', 'SL_D', 'SL_C', 'HCP', 'Total_Points']
    
    left_col = f"{left}_{direction}" if left in bridge_cols else left
    left_expr = pl.col(left_col)
    
    # Check if right is a number or column name
    right_val: float | None = None
    try:
        right_val = float(right)
    except ValueError:
        pass
    
    if right_val is not None:
        # Compare against a numeric value
        if op == '>=':
            return left_expr >= right_val
        elif op == '<=':
            return left_expr <= right_val
        elif op == '>':
            return left_expr > right_val
        elif op == '<':
            return left_expr < right_val
        elif op == '==':
            return left_expr == right_val
        elif op == '!=':
            return left_expr != right_val
    else:
        # Compare against another column
        right_col = f"{right}_{direction}" if right in bridge_cols else right
        right_expr = pl.col(right_col)
        if op == '>=':
            return left_expr >= right_expr
        elif op == '<=':
            return left_expr <= right_expr
        elif op == '>':
            return left_expr > right_expr
        elif op == '<':
            return left_expr < right_expr
        elif op == '==':
            return left_expr == right_expr
        elif op == '!=':
            return left_expr != right_expr
    
    return None


def _apply_auction_criteria(
    df: pl.DataFrame, 
    dealer_col: str = 'Dealer', 
    auction_col: str = 'Auction',
    track_rejected: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Apply dynamic auction criteria from criteria.csv to filter the DataFrame.
    
    For each row, checks if Auction starts with any partial_auction in criteria.csv.
    If so, applies the additional criteria filters based on the seat's direction.
    
    Returns:
        (filtered_df, rejected_df) - rejected_df is None if track_rejected=False
        rejected_df has columns: Auction, Partial_Auction, Failed_Criteria, Dealer, Seat, Direction
    """
    criteria_list = _load_auction_criteria()
    if not criteria_list:
        return df, None
    
    if auction_col not in df.columns or dealer_col not in df.columns:
        return df, None
    
    # Track rejected rows for debugging
    rejected_rows: list[dict] = []
    
    # For each criterion set, filter out rows that match the partial
    # auction but don't satisfy the criteria
    for partial_auction, criteria in criteria_list:
        auction_lower = pl.col(auction_col).cast(pl.Utf8).str.to_lowercase()
        matches_partial = auction_lower.str.starts_with(partial_auction)
        
        # For each dealer, build the criteria check
        for dealer in ['N', 'E', 'S', 'W']:
            direction = _get_direction_for_partial_auction(partial_auction, dealer)
            num_dashes = partial_auction.count('-')
            seat = num_dashes + 1
            
            criteria_exprs = []
            criteria_strs = []
            for criterion in criteria:
                criterion_expr = _parse_criterion_to_polars(criterion, direction)
                if criterion_expr is not None:
                    criteria_exprs.append((criterion, criterion_expr))
                    criteria_strs.append(criterion)
            
            if criteria_exprs:
                # Find rows that match partial auction and dealer
                matching_mask = matches_partial & (pl.col(dealer_col) == dealer)
                matching_rows = df.filter(matching_mask)
                
                if track_rejected and matching_rows.height > 0:
                    # For each matching row, check which criteria fail
                    for row_idx in range(matching_rows.height):
                        row = matching_rows.row(row_idx, named=True)
                        failed_criteria = []
                        
                        for criterion_str, criterion_expr in criteria_exprs:
                            # Check if this single row passes the criterion
                            try:
                                single_row_df = matching_rows.slice(row_idx, 1)
                                passes = single_row_df.filter(criterion_expr).height > 0
                                if not passes:
                                    failed_criteria.append(criterion_str)
                            except Exception:
                                failed_criteria.append(f"{criterion_str} (eval error)")
                        
                        if failed_criteria:
                            rejected_rows.append({
                                'Auction': str(row.get(auction_col, '')),
                                'Partial_Auction': str(partial_auction),
                                'Failed_Criteria': ', '.join(failed_criteria),  # Convert list to string for display
                                'Dealer': str(dealer),
                                'Seat': int(seat),
                                'Direction': str(direction),
                            })
                
                # Build combined filter
                combined_criteria = criteria_exprs[0][1]
                for _, expr in criteria_exprs[1:]:
                    combined_criteria = combined_criteria & expr
                
                # Keep row if: not (matches_partial AND dealer matches AND NOT criteria_satisfied)
                df = df.filter(
                    ~(matches_partial & (pl.col(dealer_col) == dealer)) | combined_criteria
                )
    
    # Build rejected DataFrame
    rejected_df = None
    if track_rejected and rejected_rows:
        rejected_df = pl.DataFrame(rejected_rows)
    
    return df, rejected_df


REQUIRED_FILES = [
    exec_plan_file,
    bbo_mldf_augmented_file,
    bbo_bidding_table_augmented_file,
    bt_aggregates_file,
]


def _check_required_files() -> list[str]:
    """Check that all required data files exist. Returns list of missing files."""
    missing = []
    for f in REQUIRED_FILES:
        if not f.exists():
            missing.append(str(f))
    return missing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - starts initialization on startup."""
    # Startup: start initialization in background thread
    with _STATE_LOCK:
        if not STATE["initialized"] and not STATE["initializing"]:
            STATE["initializing"] = True
            STATE["error"] = None
            print("[startup] Starting initialization in background thread...")
            thread = threading.Thread(target=_heavy_init, daemon=True)
            thread.start()
    
    yield  # Server runs here
    
    # Shutdown: nothing special needed (fast exit handler takes care of it)


app = FastAPI(title="BBO Bidding Queries API", lifespan=lifespan)


class StatusResponse(BaseModel):
    initialized: bool
    initializing: bool
    warming: bool = False  # True while pre-warming endpoints
    error: Optional[str]
    bt_df_rows: Optional[int] = None
    deal_df_rows: Optional[int] = None
    loading_step: Optional[str] = None  # Current loading step
    loaded_files: Optional[Dict[str, Any]] = None  # File name -> row count (int or str like "100 of 1000")


class InitResponse(BaseModel):
    status: str


class OpeningsByDealIndexRequest(BaseModel):
    sample_size: int = 6
    seats: Optional[List[int]] = None
    directions: Optional[List[str]] = None
    opening_directions: Optional[List[str]] = None


class RandomAuctionSequencesRequest(BaseModel):
    n_samples: int = 5
    # seed=0 means non-reproducible, any other value is reproducible
    seed: Optional[int] = 0


class AuctionSequencesMatchingRequest(BaseModel):
    pattern: str
    n_samples: int = 5
    seed: Optional[int] = 0


class DealsMatchingAuctionRequest(BaseModel):
    pattern: str
    n_auction_samples: int = 2
    n_deal_samples: int = 10
    seed: Optional[int] = 0
    # Distribution filter for deals
    dist_pattern: Optional[str] = None  # Ordered distribution (S-H-D-C), e.g., "5-4-3-1"
    sorted_shape: Optional[str] = None  # Sorted shape (any suit), e.g., "5431"
    dist_direction: str = "N"  # Which hand to filter (N/E/S/W)


class BiddingTableStatisticsRequest(BaseModel):
    auction_pattern: str = "^1N-p-3N$"
    sample_size: int = 100
    min_matches: int = 0  # 0 = no minimum
    seed: Optional[int] = 0
    # Distribution filter
    dist_pattern: Optional[str] = None  # Ordered distribution (S-H-D-C), e.g., "5-4-3-1"
    sorted_shape: Optional[str] = None  # Sorted shape (any suit), e.g., "5431"
    dist_seat: int = 1  # Which seat to filter (1-4)


class ProcessPBNRequest(BaseModel):
    pbn: str  # PBN or LIN string, or URL to .pbn/.lin file
    include_par: bool = True
    vul: str = "None"  # None, Both, NS, EW


class FindMatchingAuctionsRequest(BaseModel):
    hcp: int
    sl_s: int
    sl_h: int
    sl_d: int
    sl_c: int
    total_points: int
    seat: int = 1  # Which seat to match (1-4)
    max_results: int = 50


class PBNLookupRequest(BaseModel):
    pbn: str  # PBN deal string to look up
    max_results: int = 100


class GroupByBidRequest(BaseModel):
    """Request for grouping deals by their actual auction (bid column)."""
    auction_pattern: str = ".*"  # Regex pattern to filter auctions
    n_auction_groups: int = 10  # Number of unique auctions to show
    n_deals_per_group: int = 5  # Number of sample deals per auction
    seed: Optional[int] = 0  # Random seed (0 = non-reproducible)
    min_deals: int = 1  # Minimum deals required per auction group


# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------


STATE: Dict[str, Any] = {
    "initialized": False,
    "initializing": False,
    "warming": False,  # True while pre-warming endpoints
    "error": None,
    "loading_step": None,  # Current loading step description
    "loaded_files": {},  # File name -> row count (updated as files load)
    "deal_df": None,
    "bt_df": None,
    "bt_completed_df": None,  # Cached: bt_df.filter(is_completed_auction)
    "deal_criteria_by_seat_dfs": None,
    "deal_criteria_by_direction_dfs": None,
    "results": None,
    "bt_criteria": None,  # bt_criteria.parquet
    "bt_aggregates": None,  # bbo_bt_aggregate.parquet
}

# Additional optional data file paths
bt_criteria_file = dataPath.joinpath("bbo_bt_criteria.parquet")

_STATE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Heavy initialization
# ---------------------------------------------------------------------------


def _heavy_init() -> None:
    """Perform all heavy loading and preprocessing.

    This roughly mirrors `bbo_bidding_queries`, but instead of
    printing and updating globals for an interactive notebook, it populates
    the shared STATE dict for use by the API endpoints.
    """

    # Check required files exist before starting
    print("[init] Checking required files...")
    missing_files = _check_required_files()
    if missing_files:
        error_msg = f"Missing required files:\n" + "\n".join(f"  - {f}" for f in missing_files)
        print(f"[init] ERROR: {error_msg}")
        with _STATE_LOCK:
            STATE["initialized"] = False
            STATE["initializing"] = False
            STATE["error"] = error_msg
        return

    t0 = time.time()
    print("[init] Starting initialization...")
    _log_memory("start")
    
    TOTAL_STEPS = 10  # Total number of loading steps
    
    def _update_loading_status(step_num: int, step: str, file_name: str | None = None, row_count: int | str | None = None):
        """Update loading progress in STATE (thread-safe)."""
        with _STATE_LOCK:
            STATE["loading_step"] = f"[{step_num}/{TOTAL_STEPS}] {step}"
            if file_name and row_count is not None:
                STATE["loaded_files"][file_name] = row_count
    
    try:
        _update_loading_status(1, "Loading execution plan...")
        (
            directionless_criteria_cols,
            expr_map_by_direction,
            valid_deal_columns,
            pythonized_exprs_by_direction,
        ) = load_execution_plan_data(exec_plan_file)
        _log_memory("after load_execution_plan_data")

        # Add additional columns needed for PBN Lookup (game results)
        # DD_Score_Declarer = double-dummy score for the actual contract played
        # Contract = actual contract played (e.g., "4SN" = 4S by North)
        additional_cols = ['PBN', 'Vul', 'Declarer', 'bid', 'Contract', 'Result', 'Tricks', 'Score', 
                           'ParScore', 'DD_Score_Declarer', 'ParContracts']
        
        # Add DD_Score columns for all contracts (for DD_Score_AI computation)
        # Format: DD_Score_{level}{strain}_{direction} e.g. DD_Score_3N_N
        for level in range(1, 8):
            for strain in ['C', 'D', 'H', 'S', 'N']:
                for direction in ['N', 'E', 'S', 'W']:
                    additional_cols.append(f"DD_Score_{level}{strain}_{direction}")
        
        # Add EV (Expected Value) columns for the actual contract played
        # EV_Score_Declarer = EV for the actual contract (analogous to DD_Score_Declarer)
        additional_cols.append('EV_Score_Declarer')
        
        # Add EV columns for all contracts (for EV_AI computation)
        # Format: EV_{pair}_{declarer}_{strain}_{level}_{vul}
        # e.g. EV_NS_N_S_3_NV = EV for 3S by North, NS pair, Not Vulnerable
        for pair in ['NS', 'EW']:
            # Declarers for each pair
            declarers = ['N', 'S'] if pair == 'NS' else ['E', 'W']
            for declarer in declarers:
                for strain in ['C', 'D', 'H', 'S', 'N']:
                    for level in range(1, 8):
                        for vul in ['NV', 'V']:
                            additional_cols.append(f"EV_{pair}_{declarer}_{strain}_{level}_{vul}")
        
        for col in additional_cols:
            if col not in valid_deal_columns:
                valid_deal_columns.append(col)

        # Load deals (optionally limited by --deal-rows for faster debugging)
        n_rows_msg = f" (limited to {_cli_deal_rows:,} rows)" if _cli_deal_rows else ""
        _update_loading_status(2, f"Loading deal_df (bbo_mldf_augmented.parquet){n_rows_msg}...")
        deal_df = load_deal_df(bbo_mldf_augmented_file, valid_deal_columns, mldf_n_rows=_cli_deal_rows)
        
        # Get total row count for display (when using --deal-rows)
        if _cli_deal_rows:
            import pyarrow.parquet as pq
            total_rows = pq.read_metadata(bbo_mldf_augmented_file).num_rows
            row_info = f"{deal_df.height:,} of {total_rows:,}"
        else:
            row_info = f"{deal_df.height:,}"
        _update_loading_status(3, "Building criteria bitmaps...", "deal_df", row_info)
        _log_memory("after load_deal_df")

        # todo: do this earlier in the pipeline?
        # Convert 'bid' from pl.List(pl.Utf8) to pl.Utf8 by joining with '-'
        if 'bid' in deal_df.columns and deal_df['bid'].dtype == pl.List(pl.Utf8):
            deal_df = deal_df.with_columns(pl.col('bid').list.join('-'))
        
        # Build criteria bitmaps and derive per-seat/per-dealer views
        criteria_deal_dfs_directional = build_or_load_directional_criteria_bitmaps(
            deal_df,
            pythonized_exprs_by_direction,
            expr_map_by_direction,
        )
        _log_memory("after build_or_load_directional_criteria_bitmaps")

        _update_loading_status(4, "Processing criteria views...")
        deal_criteria_by_direction_dfs, deal_criteria_by_seat_dfs = directional_to_directionless(
            criteria_deal_dfs_directional, expr_map_by_direction
        )
        _log_memory("after directional_to_directionless")

        # We no longer need these large helper objects
        del criteria_deal_dfs_directional, pythonized_exprs_by_direction, directionless_criteria_cols
        gc.collect()
        _log_memory("after gc.collect (criteria cleanup)")

        # Load bidding table
        _update_loading_status(5, "Loading bt_df (bbo_bt_augmented.parquet)...")
        bt_df = load_bt_df(bbo_bidding_table_augmented_file, include_expr_and_sequences=True)
        _log_memory("after load_bt_df")

        # Compute opening-bid candidates for all (dealer, seat) combinations
        _update_loading_status(6, "Processing opening bids (may take several minutes)...", "bt_df", bt_df.height)
        results = process_opening_bids(
            deal_df,
            bt_df,
            deal_criteria_by_seat_dfs,
            bbo_bidding_table_augmented_file,
        )
        _update_loading_status(7, "Loading optional files...")
        _log_memory("after process_opening_bids")

        # Load optional aggregates files (non-blocking if missing)
        bt_criteria = None
        bt_aggregates = None
        
        if bt_criteria_file.exists():
            _update_loading_status(7, "Loading bt_criteria.parquet...")
            print(f"[init] Loading bt_criteria from {bt_criteria_file}...")
            bt_criteria = pl.read_parquet(bt_criteria_file)
            _update_loading_status(7, "Loading bt_criteria.parquet...", "bt_criteria", bt_criteria.height)
            _log_memory("after load bt_criteria")
        else:
            print(f"[init] bt_criteria not found at {bt_criteria_file} (optional)")
        
        if bt_aggregates_file.exists():
            _update_loading_status(8, "Loading bt_aggregates.parquet...")
            print(f"[init] Loading bt_aggregates from {bt_aggregates_file}...")
            bt_aggregates = pl.read_parquet(bt_aggregates_file)
            _update_loading_status(8, "Loading bt_aggregates.parquet...", "bt_aggregates", bt_aggregates.height)
            _log_memory("after load bt_aggregates")
        else:
            print(f"[init] bt_aggregates not found at {bt_aggregates_file} (optional)")

        # Cache the completed auctions filter (expensive: ~2 min on 541M rows)
        bt_completed_df = None
        if "is_completed_auction" in bt_df.columns:
            _update_loading_status(9, "Filtering completed auctions (bt_completed_df)...")
            print("[init] Caching bt_completed_df (is_completed_auction filter)...")
            bt_completed_df = bt_df.filter(pl.col("is_completed_auction"))
            _update_loading_status(9, "Filtering completed auctions...", "bt_completed_df", bt_completed_df.height)
            print(f"[init] bt_completed_df: {bt_completed_df.height:,} rows (from {bt_df.height:,})")
            _log_memory("after bt_completed_df filter")

        with _STATE_LOCK:
            STATE["deal_df"] = deal_df
            STATE["bt_df"] = bt_df
            STATE["bt_completed_df"] = bt_completed_df
            STATE["deal_criteria_by_seat_dfs"] = deal_criteria_by_seat_dfs
            STATE["deal_criteria_by_direction_dfs"] = deal_criteria_by_direction_dfs
            STATE["results"] = results
            STATE["bt_criteria"] = bt_criteria
            STATE["bt_aggregates"] = bt_aggregates
            STATE["initialized"] = True  # Required for _ensure_ready() in pre-warming
            STATE["warming"] = True  # Still warming - not ready for users yet
            STATE["error"] = None
        _log_memory("after STATE update")

        # ------------------------------------------------------------------
        # Pre-warm selected query paths so the first user request is faster.
        # This does a tiny sample query on each endpoint's core logic.
        # ------------------------------------------------------------------
        _update_loading_status(10, "Pre-warming endpoints...")
        try:
            print("[init] Pre-warming openings-by-deal-index endpoint ...")
            _ = openings_by_deal_index(OpeningsByDealIndexRequest(sample_size=1))

            print("[init] Pre-warming random-auction-sequences endpoint ...")
            _ = random_auction_sequences(RandomAuctionSequencesRequest(n_samples=1, seed=42))

            print("[init] Pre-warming auction-sequences-matching endpoint ...")
            _ = auction_sequences_matching(AuctionSequencesMatchingRequest(pattern="^1N-p-3N$", n_samples=1, seed=0))

            print("[init] Pre-warming deals-matching-auction endpoint ...")
            _ = deals_matching_auction(
                DealsMatchingAuctionRequest(
                    pattern="^1N-p-3N$",
                    n_auction_samples=1,
                    n_deal_samples=3,
                    seed=0,
                )
            )

            print("[init] Pre-warming bidding-table-statistics endpoint ...")
            _ = bidding_table_statistics(
                BiddingTableStatisticsRequest(
                    auction_pattern="^1N-p-3N$",
                    sample_size=1,
                    seed=42,
                )
            )

            print("[init] Pre-warming process-pbn endpoint ...")
            _ = process_pbn(
                ProcessPBNRequest(
                    pbn="N:AKQ2.KQ2.AK2.AK2 T987.987.987.987 J654.654.654.654 3.JT53.QJT53.QJT5",
                    include_par=True,
                    vul="None",
                )
            )

            print("[init] Pre-warming find-matching-auctions endpoint ...")
            _ = find_matching_auctions(
                FindMatchingAuctionsRequest(
                    hcp=15, sl_s=4, sl_h=3, sl_d=3, sl_c=3,
                    total_points=17, seat=1, max_results=1,
                )
            )

            print("[init] Pre-warming group-by-bid endpoint ...")
            _ = group_by_bid(
                GroupByBidRequest(
                    auction_pattern="^1N-p-3N$",
                    n_auction_groups=1,
                    n_deals_per_group=1,
                    seed=42,
                )
            )
        except Exception as warm_exc:  # pragma: no cover - best-effort prewarm
            print("[init] WARNING: pre-warm step failed:", warm_exc)

        # Mark warming complete - now ready for users
        with _STATE_LOCK:
            STATE["warming"] = False
            STATE["initializing"] = False

        _log_memory("after pre-warm")
        elapsed = time.time() - t0
        print(f"[init] Completed heavy initialization (including pre-warm) in {elapsed:.1f}s")

    except Exception as exc:  # pragma: no cover - defensive
        with _STATE_LOCK:
            STATE["initialized"] = False
            STATE["initializing"] = False
            STATE["error"] = f"{type(exc).__name__}: {exc}"
        print("[init] ERROR during initialization:", exc)


# ---------------------------------------------------------------------------
# API endpoints: status & init
# ---------------------------------------------------------------------------


@app.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    with _STATE_LOCK:
        bt_df_rows = None
        deal_df_rows = None
        if STATE["initialized"]:
            if STATE["bt_df"] is not None:
                bt_df_rows = STATE["bt_df"].height
            if STATE["deal_df"] is not None:
                deal_df_rows = STATE["deal_df"].height
        return StatusResponse(
            initialized=bool(STATE["initialized"]),
            initializing=bool(STATE["initializing"]),
            warming=bool(STATE.get("warming", False)),
            error=STATE["error"],
            bt_df_rows=bt_df_rows,
            deal_df_rows=deal_df_rows,
            loading_step=STATE.get("loading_step"),
            loaded_files=STATE.get("loaded_files") or None,
        )


@app.post("/init", response_model=InitResponse)
def start_init(background_tasks: BackgroundTasks) -> InitResponse:
    with _STATE_LOCK:
        if STATE["initialized"]:
            return InitResponse(status="already_initialized")
        if STATE["initializing"]:
            return InitResponse(status="already_initializing")
        STATE["initializing"] = True
        STATE["error"] = None

    background_tasks.add_task(_heavy_init)
    return InitResponse(status="started")


def _ensure_ready() -> Tuple[pl.DataFrame, pl.DataFrame, Dict[int, Dict[str, pl.DataFrame]], Dict[Tuple[str, int], Dict[str, Any]]]:
    with _STATE_LOCK:
        if not STATE["initialized"]:
            if STATE["initializing"]:
                raise HTTPException(status_code=503, detail="Initialization in progress")
            raise HTTPException(status_code=503, detail="Service not initialized")
        deal_df = STATE["deal_df"]
        bt_df = STATE["bt_df"]
        deal_criteria_by_seat_dfs = STATE["deal_criteria_by_seat_dfs"]
        results = STATE["results"]

    assert isinstance(deal_df, pl.DataFrame)
    assert isinstance(bt_df, pl.DataFrame)
    return deal_df, bt_df, deal_criteria_by_seat_dfs, results


# ---------------------------------------------------------------------------
# API: opening bid details
# ---------------------------------------------------------------------------


@app.post("/openings-by-deal-index")
def openings_by_deal_index(req: OpeningsByDealIndexRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    deal_df, bt_df, _deal_criteria_by_seat_dfs, results = _ensure_ready()

    dir_to_idx = {d: i for i, d in enumerate(DIRECTIONS)}

    seats_to_process = req.seats if req.seats is not None else [1, 2, 3, 4]
    directions_to_process = req.directions if req.directions is not None else list(DIRECTIONS)
    opening_dirs_filter = req.opening_directions

    # Sample the first N deals (consistent with the notebook usage)
    sample_df = deal_df.head(req.sample_size)
    target_rows = sample_df.filter(pl.col("Dealer").is_in(directions_to_process))

    dealer_col = target_rows["Dealer"]
    index_col = target_rows["index"]

    out_deals: List[Dict[str, Any]] = []

    for row_idx in range(target_rows.height):
        dealer = dealer_col[row_idx]
        idx_val = int(index_col[row_idx])
        dealer_idx = dir_to_idx[dealer]
        original_pos = idx_val - 1

        opening_bids: List[int] = []
        opening_seat_num: Optional[int] = None

        for seat in seats_to_process:
            key = (dealer, seat)
            if key not in results:
                continue

            orig_indices = results[key]["original_indices"]
            pos = orig_indices.search_sorted(original_pos)

            if pos < len(orig_indices) and int(orig_indices[pos]) == original_pos:
                bids = results[key]["candidates"]["candidate_bids"][pos]
                if bids is not None and len(bids) > 0:
                    opener_for_seat = DIRECTIONS[(dealer_idx + seat - 1) % 4]
                    if opening_dirs_filter is None or opener_for_seat in opening_dirs_filter:
                        if opening_seat_num is None:
                            opening_seat_num = seat
                        opening_bids.extend(int(b) for b in bids)

        if not opening_bids:
            continue

        opening_seat = DIRECTIONS[(dealer_idx + opening_seat_num - 1) % 4] if opening_seat_num else None
        if opening_dirs_filter is not None:
            if opening_seat is None or opening_seat not in opening_dirs_filter:
                continue

        # Remove duplicate bid indices while preserving order
        opening_bids_unique = list(dict.fromkeys(opening_bids))

        # Fetch bt_df rows for these opening bid indices
        opening_bids_df_rows: List[Dict[str, Any]] = []
        if opening_bids_unique:
            # Select relevant columns from bt_df for display
            bt_display_cols = ["index", "Auction", "seat", "Expr"]
            available_bt_cols = [c for c in bt_display_cols if c in bt_df.columns]
            filtered_bt = bt_df.filter(pl.col("index").is_in(opening_bids_unique)).select(available_bt_cols)
            opening_bids_df_rows = filtered_bt.to_dicts()

        row_slice = target_rows[row_idx]
        hands = {
            "Hand_N": row_slice["Hand_N"][0] if "Hand_N" in row_slice.columns else None,
            "Hand_E": row_slice["Hand_E"][0] if "Hand_E" in row_slice.columns else None,
            "Hand_S": row_slice["Hand_S"][0] if "Hand_S" in row_slice.columns else None,
            "Hand_W": row_slice["Hand_W"][0] if "Hand_W" in row_slice.columns else None,
        }

        out_deals.append(
            {
                "index": idx_val,
                "dealer": dealer,
                "opening_seat": opening_seat,
                "opening_bid_indices": opening_bids_unique,
                "opening_bids_df": opening_bids_df_rows,
                "hands": hands,
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[openings-by-deal-index] {elapsed_ms:.1f}ms")
    return {"deals": out_deals, "elapsed_ms": round(elapsed_ms, 1)}


# ---------------------------------------------------------------------------
# API: bidding sequences (random samples of completed auctions)
# ---------------------------------------------------------------------------


@app.post("/random-auction-sequences")
def random_auction_sequences(req: RandomAuctionSequencesRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    _, bt_df, _, _ = _ensure_ready()

    # Use cached completed auctions filter (computed once at startup)
    with _STATE_LOCK:
        completed_df = STATE.get("bt_completed_df")
    
    if completed_df is None:
        raise HTTPException(status_code=500, detail="bt_completed_df not cached (is_completed_auction column missing?)")

    if completed_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[random-auction-sequences] {elapsed_ms:.1f}ms (empty)")
        return {"samples": [], "elapsed_ms": round(elapsed_ms, 1)}

    sample_n = min(req.n_samples, completed_df.height)
    effective_seed = _effective_seed(req.seed)
    sampled_df = completed_df.sample(n=sample_n, seed=effective_seed)

    agg_expr_cols = [f"Agg_Expr_Seat_{i}" for i in range(1, 5)]
    extra_cols = ["Expr"] + agg_expr_cols
    display_cols = ["index", "Auction", "seat"]
    lookup_cols = display_cols + [c for c in extra_cols if c in bt_df.columns]
    available_display_cols = [c for c in lookup_cols if c in bt_df.columns]

    # Batch collect previous indices
    all_prev_indices = set()
    for prev_list in sampled_df["previous_bid_indices"].to_list():
        if prev_list:
            all_prev_indices.update(prev_list)

    if all_prev_indices:
        prev_rows_df = bt_df.filter(pl.col("index").is_in(list(all_prev_indices))).select(available_display_cols)
        prev_rows_lookup = {row["index"]: row for row in prev_rows_df.iter_rows(named=True)}
    else:
        prev_rows_lookup = {}

    out_samples: List[Dict[str, Any]] = []

    for row in sampled_df.iter_rows(named=True):
        auction = row["Auction"]
        seat = int(row["seat"])
        prev_indices = row["previous_bid_indices"]

        sequence_data: List[Dict[str, Any]] = []
        if prev_indices:
            sequence_data.extend(prev_rows_lookup[idx] for idx in prev_indices if idx in prev_rows_lookup)
        sequence_data.append({c: row[c] for c in available_display_cols})

        seq_df = pl.DataFrame(sequence_data).sort("index")
        if all(col in seq_df.columns for col in agg_expr_cols):
            seq_df = seq_df.with_columns(
                pl.when(pl.col("seat") == 1)
                .then(pl.col("Agg_Expr_Seat_1"))
                .when(pl.col("seat") == 2)
                .then(pl.col("Agg_Expr_Seat_2"))
                .when(pl.col("seat") == 3)
                .then(pl.col("Agg_Expr_Seat_3"))
                .when(pl.col("seat") == 4)
                .then(pl.col("Agg_Expr_Seat_4"))
                .alias("Agg_Expr")
            )
            out_cols = ["index", "Auction", "seat"]
            if "Expr" in seq_df.columns:
                out_cols.append("Expr")
            out_cols.append("Agg_Expr")
            seq_df = seq_df.select(out_cols)

        out_samples.append(
            {
                "auction": auction,
                "seat": seat,
                "sequence": seq_df.to_dicts(),
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[random-auction-sequences] {elapsed_ms:.1f}ms ({len(out_samples)} samples)")
    return {"samples": out_samples, "elapsed_ms": round(elapsed_ms, 1)}


# ---------------------------------------------------------------------------
# API: auctions matching a pattern
# ---------------------------------------------------------------------------


@app.post("/auction-sequences-matching")
def auction_sequences_matching(req: AuctionSequencesMatchingRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    _, bt_df, _, _ = _ensure_ready()

    if "previous_bid_indices" not in bt_df.columns:
        raise HTTPException(status_code=500, detail="Column 'previous_bid_indices' not found in bt_df")

    # Use cached completed auctions filter
    with _STATE_LOCK:
        base_df = STATE.get("bt_completed_df")
    if base_df is None:
        base_df = bt_df  # Fallback if not cached

    pattern = normalize_auction_pattern(req.pattern)

    is_regex = pattern.startswith("^") or pattern.endswith("$")
    if is_regex:
        # Use (?i) for case-insensitive matching
        regex_pattern = f"(?i){pattern}"
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(regex_pattern))
    else:
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(f"(?i){pattern}"))

    # Apply dynamic auction criteria from criteria.csv (track rejected for debugging)
    filtered_df, rejected_df = _apply_auction_criteria(filtered_df, track_rejected=True)

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[auction-sequences-matching] {elapsed_ms:.1f}ms (no matches)")
        result = {"samples": [], "pattern": pattern, "elapsed_ms": round(elapsed_ms, 1)}
        if rejected_df is not None and rejected_df.height > 0:
            result["criteria_rejected"] = rejected_df.to_dicts()
        return result

    sample_n = min(req.n_samples, filtered_df.height)
    effective_seed = _effective_seed(req.seed)
    sampled_df = filtered_df.sample(n=sample_n, seed=effective_seed)

    agg_expr_cols = [f"Agg_Expr_Seat_{i}" for i in range(1, 5)]
    extra_cols = ["Expr"] + agg_expr_cols
    display_cols = ["index", "Auction", "seat"]
    lookup_cols = display_cols + [c for c in extra_cols if c in bt_df.columns]
    available_display_cols = [c for c in lookup_cols if c in bt_df.columns]

    all_prev_indices = set()
    for prev_list in sampled_df["previous_bid_indices"].to_list():
        if prev_list:
            all_prev_indices.update(prev_list)

    if all_prev_indices:
        prev_rows_df = bt_df.filter(pl.col("index").is_in(list(all_prev_indices))).select(available_display_cols)
        prev_rows_lookup = {row["index"]: row for row in prev_rows_df.iter_rows(named=True)}
    else:
        prev_rows_lookup = {}

    out_samples: List[Dict[str, Any]] = []

    for row in sampled_df.iter_rows(named=True):
        auction = row["Auction"]
        seat = int(row["seat"])
        prev_indices = row["previous_bid_indices"]

        sequence_data: List[Dict[str, Any]] = []
        if prev_indices:
            sequence_data.extend(prev_rows_lookup[idx] for idx in prev_indices if idx in prev_rows_lookup)
        sequence_data.append({c: row[c] for c in available_display_cols})

        seq_df = pl.DataFrame(sequence_data).sort("index")
        if all(col in seq_df.columns for col in agg_expr_cols):
            seq_df = seq_df.with_columns(
                pl.when(pl.col("seat") == 1)
                .then(pl.col("Agg_Expr_Seat_1"))
                .when(pl.col("seat") == 2)
                .then(pl.col("Agg_Expr_Seat_2"))
                .when(pl.col("seat") == 3)
                .then(pl.col("Agg_Expr_Seat_3"))
                .when(pl.col("seat") == 4)
                .then(pl.col("Agg_Expr_Seat_4"))
                .alias("Agg_Expr")
            )
            out_cols = ["index", "Auction", "seat"]
            if "Expr" in seq_df.columns:
                out_cols.append("Expr")
            out_cols.append("Agg_Expr")
            seq_df = seq_df.select(out_cols)

        out_samples.append(
            {
                "auction": auction,
                "seat": seat,
                "sequence": seq_df.to_dicts(),
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[auction-sequences-matching] {elapsed_ms:.1f}ms ({len(out_samples)} samples)")
    result = {"pattern": pattern, "samples": out_samples, "elapsed_ms": round(elapsed_ms, 1)}
    if rejected_df is not None and rejected_df.height > 0:
        result["criteria_rejected"] = rejected_df.to_dicts()
    return result


# ---------------------------------------------------------------------------
# API: deals for auction pattern
# ---------------------------------------------------------------------------


@app.post("/deals-matching-auction")
def deals_matching_auction(req: DealsMatchingAuctionRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    deal_df, bt_df, deal_criteria_by_seat_dfs, _results = _ensure_ready()

    # Use cached completed auctions filter
    with _STATE_LOCK:
        base_df = STATE.get("bt_completed_df")
    if base_df is None:
        base_df = bt_df  # Fallback if not cached

    pattern = normalize_auction_pattern(req.pattern)

    is_regex = pattern.startswith("^") or pattern.endswith("$")
    if is_regex:
        # Use (?i) for case-insensitive matching
        regex_pattern = f"(?i){pattern}"
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(regex_pattern))
    else:
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(f"(?i){pattern}"))

    # Apply dynamic auction criteria from criteria.csv (track rejected for debugging)
    filtered_df, rejected_df = _apply_auction_criteria(filtered_df, track_rejected=True)

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[deals-matching-auction] {elapsed_ms:.1f}ms (no matches)")
        result = {"pattern": pattern, "auctions": [], "elapsed_ms": round(elapsed_ms, 1)}
        if rejected_df is not None and rejected_df.height > 0:
            result["criteria_rejected"] = rejected_df.to_dicts()
        return result

    sample_n = min(req.n_auction_samples, filtered_df.height)
    effective_seed = _effective_seed(req.seed)
    sampled_auctions = filtered_df.sample(n=sample_n, seed=effective_seed)

    deal_display_cols = ["index", "Dealer", "bid", "Contract", "Hand_N", "Hand_E", "Hand_S", "Hand_W", 
                         "Declarer", "Result", "Tricks", "Score", "DD_Score_Declarer", "EV_Score_Declarer",
                         "ParScore", "ParContracts"]

    out_auctions: List[Dict[str, Any]] = []

    for auction_row in sampled_auctions.iter_rows(named=True):
        auction = auction_row["Auction"]
        seat = int(auction_row["seat"])

        auction_info: Dict[str, Any] = {
            "auction": auction,
            "seat": seat,
            "expr": auction_row.get("Expr"),
            "criteria_by_seat": {},
            "deals": [],
            "criteria_debug": {},  # Debug info for criteria matching
        }

        # Criteria by seat
        for s in range(1, 5):
            agg_col = f"Agg_Expr_Seat_{s}"
            if agg_col in auction_row:
                crit_list = auction_row[agg_col]
                if crit_list:
                    auction_info["criteria_by_seat"][str(s)] = crit_list

        matching_deals: List[pl.DataFrame] = []
        total_matching_count = 0  # Count ALL matching deals before sampling
        
        # Track criteria matching for debugging (once per seat, not per-dealer)
        # Criteria are the same for all dealers - only which direction plays each seat changes
        criteria_found: Dict[str, List[str]] = {}
        criteria_missing: Dict[str, List[str]] = {}
        
        # Determine the actual final seat by checking which Agg_Expr_Seat_s columns have content
        actual_final_seat = 0
        for s in range(1, 5):
            agg_col = f"Agg_Expr_Seat_{s}"
            if agg_col in auction_row and auction_row[agg_col]:
                actual_final_seat = s
        
        # Pre-populate criteria debug info (before dealer loop, since criteria are same for all)
        for s in range(1, actual_final_seat + 1):
            agg_col = f"Agg_Expr_Seat_{s}"
            seat_key = f"Seat_{s}"
            criteria_found[seat_key] = []
            criteria_missing[seat_key] = []
            
            if agg_col in auction_row:
                criteria_list = auction_row[agg_col]
                if criteria_list:
                    # Check against first available dealer's criteria df to see what's available
                    for dealer in DIRECTIONS:
                        if s in deal_criteria_by_seat_dfs and dealer in deal_criteria_by_seat_dfs[s]:
                            seat_criteria_df = deal_criteria_by_seat_dfs[s][dealer]
                            for criterion in criteria_list:
                                if criterion in seat_criteria_df.columns:
                                    if criterion not in criteria_found[seat_key]:
                                        criteria_found[seat_key].append(criterion)
                                else:
                                    if criterion not in criteria_missing[seat_key]:
                                        criteria_missing[seat_key].append(criterion)
                            break  # Only need to check one dealer's df

        for dealer in DIRECTIONS:
            dealer_mask = deal_df["Dealer"] == dealer
            if not dealer_mask.any():
                continue

            combined_mask = dealer_mask.clone()

            # Loop through ALL seats that have Agg_Expr, not just up to `seat`
            for s in range(1, actual_final_seat + 1):
                agg_col = f"Agg_Expr_Seat_{s}"
                if agg_col not in auction_row:
                    continue
                criteria_list = auction_row[agg_col]
                if not criteria_list:
                    continue

                if s in deal_criteria_by_seat_dfs and dealer in deal_criteria_by_seat_dfs[s]:
                    seat_criteria_df = deal_criteria_by_seat_dfs[s][dealer]
                    for criterion in criteria_list:
                        if criterion in seat_criteria_df.columns:
                            criterion_values = seat_criteria_df[criterion]
                            combined_mask = combined_mask & criterion_values

            matching_idx = combined_mask.arg_true()
            if len(matching_idx) > 0:
                # Track total matching before sampling
                total_matching_count += len(matching_idx)
                
                # Randomly sample from matching indices using the seed
                if len(matching_idx) > req.n_deal_samples:
                    rng = random.Random(effective_seed)
                    sampled_indices = rng.sample(list(matching_idx), req.n_deal_samples)
                else:
                    sampled_indices = list(matching_idx)
                for idx in sampled_indices:
                    matching_deals.append(deal_df[idx])
        
        # Add debug info - include all seats up to actual_final_seat, even if empty
        auction_info["criteria_debug"] = {
            "row_seat": seat,  # The seat field from the bt_df row
            "actual_final_seat": actual_final_seat,  # Determined by which Agg_Expr columns have content
            "found": criteria_found,  # Include all seats (even empty) for transparency
            "missing": {k: v for k, v in criteria_missing.items() if v},
        }
        
        # Record total matching deals (before sampling)
        auction_info["total_matching_deals"] = total_matching_count

        if matching_deals:
            combined_df = pl.concat(matching_deals[: req.n_deal_samples])
            
            # Add a column showing which direction is the opener (seat 1)
            # Seat 1 direction = Dealer
            # Seat 2 direction = (Dealer + 1) % 4, etc.
            if "Dealer" in combined_df.columns:
                combined_df = combined_df.with_columns(
                    pl.col("Dealer").alias("Opener_Direction")
                )
            
            # Apply distribution filter if specified
            if req.dist_pattern or req.sorted_shape:
                direction = req.dist_direction.upper()
                if direction in 'NESW':
                    # Add suit length columns for filtering
                    combined_df = add_suit_length_columns(combined_df, direction)
                    
                    # Build and apply filter
                    dist_where = build_distribution_sql_for_deals(
                        req.dist_pattern, req.sorted_shape, direction
                    )
                    if dist_where:
                        try:
                            dist_sql = f"SELECT * FROM combined_df WHERE {dist_where}"
                            combined_df = duckdb.sql(dist_sql).pl()
                            auction_info["dist_sql_query"] = dist_sql
                        except Exception as e:
                            print(f"[deals-matching-auction] Distribution filter error: {e}")
                    
                    # Remove the added SL columns for cleaner output
                    sl_cols = [f"SL_{s}_{direction}" for s in ['S', 'H', 'D', 'C']]
                    combined_df = combined_df.drop([c for c in sl_cols if c in combined_df.columns])
            
            # Compute AI (Auction Intelligence) derived columns for each deal
            deals_list = combined_df.to_dicts()
            for deal_row in deals_list:
                dealer = deal_row.get("Dealer", "N")
                # AI contract (final contract including declarer and doubles/redoubles)
                deal_row["AI_Contract"] = get_ai_contract(auction, dealer)
                # DD score for the AI's contract
                dd_score_ai = get_dd_score_for_auction(auction, dealer, deal_row)
                deal_row["DD_Score_AI"] = dd_score_ai
                
                # EV (Expected Value) for the AI's contract
                ev_ai = get_ev_for_auction(auction, dealer, deal_row)
                deal_row["EV_AI"] = ev_ai
                
                # IMP difference between AI contract DD score and actual contract DD score
                # Positive = AI contract scores better, Negative = actual contract scores better
                dd_score_actual = deal_row.get("DD_Score_Declarer")
                if dd_score_actual is not None and dd_score_ai is not None:
                    score_diff = int(dd_score_ai) - int(dd_score_actual)
                    imp_diff = calculate_imp(abs(score_diff))
                    deal_row["IMP_AI_vs_Actual"] = imp_diff if score_diff >= 0 else -imp_diff
                else:
                    deal_row["IMP_AI_vs_Actual"] = None
                
                # Convert ParContracts list of structs to readable contract strings
                # Each struct has: Level, Strain, Double, Pair_Direction, Result
                par_contracts = deal_row.get("ParContracts")
                if par_contracts is not None and isinstance(par_contracts, list):
                    formatted_contracts = []
                    for c in par_contracts:
                        if isinstance(c, dict):
                            level = c.get("Level", "")
                            strain = c.get("Strain", "")
                            double = c.get("Double", "")
                            pair_dir = c.get("Pair_Direction", "")
                            result = c.get("Result", "")
                            # Format: "4S N +1" or "3NX EW -2" or "4HXX NS ="
                            contract_str = f"{level}{strain}{double}"
                            if pair_dir:
                                contract_str += f" {pair_dir}"
                            if result is not None and result != "":
                                # Result is typically an int: 0 = "=", positive = "+N", negative = "-N"
                                if isinstance(result, int):
                                    if result == 0:
                                        contract_str += " ="
                                    elif result > 0:
                                        contract_str += f" +{result}"
                                    else:
                                        contract_str += f" {result}"
                                else:
                                    contract_str += f" {result}"
                            formatted_contracts.append(contract_str.strip())
                        else:
                            formatted_contracts.append(str(c))
                    deal_row["ParContracts"] = ", ".join(formatted_contracts)
            
            # Create summary by Contract
            deals_with_computed = pl.DataFrame(deals_list)
            if "Contract" in deals_with_computed.columns and "IMP_AI_vs_Actual" in deals_with_computed.columns:
                # Build aggregation list - DD stats
                agg_exprs = [
                    pl.len().alias("Count"),
                    pl.col("IMP_AI_vs_Actual").mean().alias("Avg_IMP_AI"),
                    # Percentage where Contract makes (DD_Score_Declarer >= 0)
                    (pl.col("DD_Score_Declarer").cast(pl.Int64, strict=False).ge(0).sum() * 100.0 / pl.len()).alias("Contract_Made%"),
                    # Percentage where AI_Contract makes (DD_Score_AI >= 0)
                    (pl.col("DD_Score_AI").cast(pl.Int64, strict=False).ge(0).sum() * 100.0 / pl.len()).alias("AI_Made%"),
                    # Percentage where Contract achieves par (DD_Score_Declarer == ParScore)
                    (pl.col("DD_Score_Declarer").cast(pl.Int64, strict=False).eq(pl.col("ParScore").cast(pl.Int64, strict=False)).sum() * 100.0 / pl.len()).alias("Contract_Par%"),
                    # Percentage where AI_Contract achieves par (DD_Score_AI == ParScore)
                    (pl.col("DD_Score_AI").cast(pl.Int64, strict=False).eq(pl.col("ParScore").cast(pl.Int64, strict=False)).sum() * 100.0 / pl.len()).alias("AI_Par%"),
                ]
                
                # Add EV stats if columns exist
                has_ev_contract = "EV_Score_Declarer" in deals_with_computed.columns
                has_ev_ai = "EV_AI" in deals_with_computed.columns
                
                if has_ev_contract:
                    agg_exprs.append(pl.col("EV_Score_Declarer").cast(pl.Float64, strict=False).mean().alias("Avg_EV_Contract"))
                if has_ev_ai:
                    agg_exprs.append(pl.col("EV_AI").cast(pl.Float64, strict=False).mean().alias("Avg_EV_AI"))
                if has_ev_contract and has_ev_ai:
                    # EV difference: AI EV - Actual Contract EV (positive = AI better)
                    agg_exprs.append(
                        (pl.col("EV_AI").cast(pl.Float64, strict=False) - pl.col("EV_Score_Declarer").cast(pl.Float64, strict=False))
                        .mean().alias("Avg_EV_Diff")
                    )
                
                contract_summary = (
                    deals_with_computed
                    .group_by("Contract")
                    .agg(agg_exprs)
                    .sort("Count", descending=True)
                )
                
                # Round to 1 decimal place
                round_cols = [
                    pl.col("Avg_IMP_AI").round(1),
                    pl.col("Contract_Made%").round(1),
                    pl.col("AI_Made%").round(1),
                    pl.col("Contract_Par%").round(1),
                    pl.col("AI_Par%").round(1),
                ]
                if has_ev_contract:
                    round_cols.append(pl.col("Avg_EV_Contract").round(2))
                if has_ev_ai:
                    round_cols.append(pl.col("Avg_EV_AI").round(2))
                if has_ev_contract and has_ev_ai:
                    round_cols.append(pl.col("Avg_EV_Diff").round(2))
                
                contract_summary = contract_summary.with_columns(round_cols)
                auction_info["contract_summary"] = contract_summary.to_dicts()
                
                # Calculate grand totals across all deals
                total_deals = deals_with_computed.height
                imp_values = deals_with_computed["IMP_AI_vs_Actual"].cast(pl.Int64, strict=False)
                total_imp = imp_values.sum()
                auction_info["total_imp_ai"] = int(total_imp) if total_imp is not None else 0
                auction_info["total_deals"] = total_deals
                
                # IMP breakdown: cumulative IMPs where AI won vs where Actual won
                imp_ai_wins = imp_values.filter(imp_values > 0).sum()
                imp_actual_wins = (-imp_values.filter(imp_values < 0)).sum()
                auction_info["imp_ai_advantage"] = int(imp_ai_wins) if imp_ai_wins is not None else 0
                auction_info["imp_actual_advantage"] = int(imp_actual_wins) if imp_actual_wins is not None else 0
                
                # Count where contracts make (DD score >= 0)
                dd_actual = deals_with_computed["DD_Score_Declarer"].cast(pl.Int64, strict=False)
                dd_ai = deals_with_computed["DD_Score_AI"].cast(pl.Int64, strict=False)
                par_score = deals_with_computed["ParScore"].cast(pl.Int64, strict=False)
                
                auction_info["contract_makes_count"] = int((dd_actual >= 0).sum())
                auction_info["ai_makes_count"] = int((dd_ai >= 0).sum())
                
                # Count where contracts achieve par
                auction_info["contract_par_count"] = int((dd_actual == par_score).sum())
                auction_info["ai_par_count"] = int((dd_ai == par_score).sum())
                
                # EV grand totals
                if has_ev_contract:
                    ev_contract = deals_with_computed["EV_Score_Declarer"].cast(pl.Float64, strict=False)
                    ev_contract_mean = ev_contract.mean()
                    auction_info["avg_ev_contract"] = round(float(ev_contract_mean), 2) if ev_contract_mean is not None else None  # type: ignore[arg-type]
                
                if has_ev_ai:
                    ev_ai_col = deals_with_computed["EV_AI"].cast(pl.Float64, strict=False)
                    ev_ai_mean = ev_ai_col.mean()
                    auction_info["avg_ev_ai"] = round(float(ev_ai_mean), 2) if ev_ai_mean is not None else None  # type: ignore[arg-type]
                
                if has_ev_contract and has_ev_ai:
                    ev_diff = (deals_with_computed["EV_AI"].cast(pl.Float64, strict=False) - 
                               deals_with_computed["EV_Score_Declarer"].cast(pl.Float64, strict=False))
                    ev_diff_mean = ev_diff.mean()
                    auction_info["avg_ev_diff"] = round(float(ev_diff_mean), 2) if ev_diff_mean is not None else None  # type: ignore[arg-type]
            
            # Filter to display columns (keeping computed columns we just added)
            display_cols_set = set(deal_display_cols) | {"DD_Score_AI", "EV_AI", "AI_Contract", "IMP_AI_vs_Actual"}
            auction_info["deals"] = [
                {k: v for k, v in d.items() if k in display_cols_set}
                for d in deals_list
            ]

        out_auctions.append(auction_info)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_deals = sum(len(a.get("deals", [])) for a in out_auctions)
    print(f"[deals-matching-auction] {elapsed_ms:.1f}ms ({len(out_auctions)} auctions, {total_deals} deals)")
    
    response = {"pattern": pattern, "auctions": out_auctions, "elapsed_ms": round(elapsed_ms, 1)}
    if req.dist_pattern or req.sorted_shape:
        response["dist_filter"] = {
            "dist_pattern": req.dist_pattern,
            "sorted_shape": req.sorted_shape,
            "direction": req.dist_direction,
        }
    if rejected_df is not None and rejected_df.height > 0:
        response["criteria_rejected"] = rejected_df.to_dicts()
    return response


# ---------------------------------------------------------------------------
# API: bidding table statistics
# ---------------------------------------------------------------------------


@app.post("/bidding-table-statistics")
def bidding_table_statistics(req: BiddingTableStatisticsRequest) -> Dict[str, Any]:
    """Get bidding table entries with aggregate statistics filtered by auction pattern."""
    t0 = time.perf_counter()
    _, bt_df, _, _ = _ensure_ready()
    
    with _STATE_LOCK:
        bt_criteria = STATE.get("bt_criteria")
        bt_aggregates = STATE.get("bt_aggregates")
        bt_completed_df = STATE.get("bt_completed_df")
    
    # Use cached completed auctions filter (to match bt_criteria/bt_aggregates row indices)
    # bt_criteria and bt_aggregates were generated from is_completed_auction rows only
    if bt_completed_df is not None:
        base_df = bt_completed_df.with_row_index("_idx")
    else:
        base_df = bt_df.with_row_index("_idx")
    
    # OPTIMIZATION 1: Use Polars native str.contains() instead of Python loop
    # This is vectorized and much faster than iterating millions of rows
    try:
        # Use case-insensitive regex matching with Polars
        # Cast to Utf8 and use (?i) flag for case-insensitive matching (same as other endpoints)
        regex_pattern = f"(?i){req.auction_pattern}"
        matched_df = base_df.filter(
            pl.col("Auction").cast(pl.Utf8).str.contains(regex_pattern)
        )
    except Exception as e:
        import traceback
        error_detail = f"Regex error: {e}\n{traceback.format_exc()}"
        print(f"[bidding-table-statistics] ERROR: {error_detail}")
        raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {e}")
    
    t1 = time.perf_counter()
    
    if matched_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "pattern": req.auction_pattern,
            "total_matches": 0,
            "rows": [],
            "has_criteria": bt_criteria is not None,
            "has_aggregates": bt_aggregates is not None,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    
    # Filter by min matching deals if aggregates available
    if bt_aggregates is not None and req.min_matches > 0:
        # OPTIMIZATION 2: Use join instead of filtering for each index
        valid_agg_df = bt_aggregates.filter(
            pl.col("matching_deal_count") >= req.min_matches
        ).select(pl.col("bt_row_idx").alias("_idx"))
        
        matched_df = matched_df.join(valid_agg_df, on="_idx", how="inner")
        
        if matched_df.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "pattern": req.auction_pattern,
                "total_matches": 0,
                "rows": [],
                "has_criteria": bt_criteria is not None,
                "has_aggregates": bt_aggregates is not None,
                "message": f"No matching auctions have >= {req.min_matches} matching deals",
                "elapsed_ms": round(elapsed_ms, 1),
            }
    
    total_matches = matched_df.height
    t2 = time.perf_counter()
    
    # Sample using Polars native sampling
    sample_n = min(req.sample_size, total_matches)
    effective_seed = None if (req.seed is None or req.seed == 0) else req.seed
    sampled_df = matched_df.sample(n=sample_n, seed=effective_seed).sort("_idx")
    
    # Get the sampled indices
    sampled_indices = sampled_df["_idx"].to_list()
    
    t3 = time.perf_counter()
    
    # OPTIMIZATION 3: Build result using Polars joins instead of row-by-row access
    # sampled_df already has the Auction column from base_df (which was filtered for is_completed_auction)
    # Use sampled_df directly instead of re-indexing into bt_df (which would use wrong indices)
    result_df = sampled_df.select([
        pl.col("_idx").alias("original_idx"),
        pl.col("Auction"),
    ]).with_row_index("row_idx")
    
    # OPTIMIZATION 4: Join with aggregates instead of per-row filtering
    if bt_aggregates is not None:
        agg_subset = bt_aggregates.filter(
            pl.col("bt_row_idx").is_in(sampled_indices)
        ).rename({"bt_row_idx": "original_idx"})
        
        result_df = result_df.join(agg_subset, on="original_idx", how="left")
    
    # OPTIMIZATION 5: Join with criteria instead of per-row access
    if bt_criteria is not None:
        # Add row index to criteria for joining
        criteria_with_idx = bt_criteria.with_row_index("original_idx")
        criteria_subset = criteria_with_idx.filter(
            pl.col("original_idx").is_in(sampled_indices)
        )
        
        result_df = result_df.join(criteria_subset, on="original_idx", how="left")
    
    t4 = time.perf_counter()
    
    # Apply distribution filter if specified
    dist_sql_query = None
    pre_dist_count = len(result_df)
    if (req.dist_pattern or req.sorted_shape) and bt_criteria is not None:
        dist_where = build_distribution_sql_for_bt(
            req.dist_pattern, req.sorted_shape, req.dist_seat, result_df.columns
        )
        if dist_where:
            dist_sql_query = f"SELECT * FROM result_df WHERE {dist_where}"
            try:
                result_df = duckdb.sql(dist_sql_query).pl()
            except Exception as e:
                print(f"[bidding-table-statistics] Distribution filter error: {e}")
    
    t5 = time.perf_counter()
    
    # Convert to list of dicts
    result_rows = result_df.to_dicts()
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[bidding-table-statistics] {elapsed_ms:.1f}ms ({len(result_rows)} rows from {total_matches} matches)")
    print(f"  regex: {(t1-t0)*1000:.0f}ms, filter: {(t2-t1)*1000:.0f}ms, sample: {(t3-t2)*1000:.0f}ms, join: {(t4-t3)*1000:.0f}ms, dist: {(t5-t4)*1000:.0f}ms")
    if dist_sql_query:
        print(f"  distribution filter: {pre_dist_count} → {len(result_rows)} rows")
    
    return {
        "pattern": req.auction_pattern,
        "total_matches": total_matches,
        "sample_size": len(result_rows),
        "rows": result_rows,
        "has_criteria": bt_criteria is not None,
        "has_aggregates": bt_aggregates is not None,
        "dist_sql_query": dist_sql_query,
        "elapsed_ms": round(elapsed_ms, 1),
    }



# Distribution filter utilities and helper functions are now imported from bbo_bidding_queries_lib


# ---------------------------------------------------------------------------
# API: process PBN / LIN
# ---------------------------------------------------------------------------

def _parse_file_with_endplay(content: str, is_lin: bool = False) -> tuple[list[str], dict[int, str]]:
    """
    Parse PBN or LIN file content using endplay and extract all deals with vulnerabilities.
    Returns (list of PBN strings, dict of deal_idx -> vulnerability).
    """
    from endplay.parsers import pbn as pbn_parser, lin as lin_parser
    
    pbn_deals = []
    deal_vuls = {}
    
    try:
        # Parse using appropriate endplay parser
        if is_lin:
            boards = lin_parser.loads(content)
        else:
            boards = pbn_parser.loads(content)
        
        for board in boards:
            if board.deal is None:
                continue
            
            # Get PBN string from deal
            pbn_str = board.deal.to_pbn()
            
            # Get vulnerability
            vul_map = {0: 'None', 1: 'NS', 2: 'EW', 3: 'Both'}  # endplay Vul enum values
            try:
                vul = vul_map.get(int(board.vul), 'None') if board.vul is not None else 'None'
            except Exception:
                vul = 'None'
            
            deal_idx = len(pbn_deals)
            pbn_deals.append(pbn_str)
            deal_vuls[deal_idx] = vul
    except Exception as e:
        print(f"[parse-file] endplay parsing failed: {e}")
    
    return pbn_deals, deal_vuls



# PBN parsing, hand features, auction parsing, and par score functions 
# are now imported from bbo_bidding_queries_lib


@app.post("/process-pbn")
def process_pbn(req: ProcessPBNRequest) -> Dict[str, Any]:
    """Process PBN or LIN deal(s) and compute features including optional par score."""
    t0 = time.perf_counter()
    
    pbn_input = req.pbn.strip()
    pbn_deals = []
    
    # Track vulnerability per deal (for PBN files)
    deal_vuls: dict[int, str] = {}
    
    # Track detected input type for user feedback
    input_type = "unknown"
    input_source = ""
    
    # Auto-detect input type: URL, local file path, or direct PBN/LIN string
    # Check if it's a URL
    if pbn_input.startswith('http://') or pbn_input.startswith('https://'):
        # Convert GitHub blob URLs to raw URLs
        url = pbn_input
        if 'github.com' in url and '/blob/' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        try:
            response = http_requests.get(url, timeout=30)
            response.raise_for_status()
            file_content = response.text
            
            # Detect file type from URL or content
            is_lin = url.lower().endswith('.lin') or 'md|' in file_content[:500]
            
            # Use endplay to parse PBN or LIN file
            pbn_deals, deal_vuls = _parse_file_with_endplay(file_content, is_lin=is_lin)
            
            input_type = "LIN URL" if is_lin else "PBN URL"
            input_source = url
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch/parse URL: {e}")
    
    # Check if it's a local file path
    elif os.path.isfile(pbn_input) or (
        (pbn_input.lower().endswith('.pbn') or pbn_input.lower().endswith('.lin')) and 
        (pbn_input.startswith('/') or (len(pbn_input) > 2 and pbn_input[1] == ':'))
    ):
        try:
            file_path = pbn_input
            if not os.path.isfile(file_path):
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Detect file type from extension or content
            is_lin = file_path.lower().endswith('.lin') or 'md|' in file_content[:500]
            
            # Use endplay to parse PBN or LIN file
            pbn_deals, deal_vuls = _parse_file_with_endplay(file_content, is_lin=is_lin)
            
            input_type = "LIN file" if is_lin else "PBN file"
            input_source = file_path
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read/parse file: {e}")
    
    # Check if it's LIN content (not URL or file)
    elif 'md|' in pbn_input and '|' in pbn_input:
        # LIN string content
        pbn_deals, deal_vuls = _parse_file_with_endplay(pbn_input, is_lin=True)
        input_type = "LIN string"
        input_source = f"{len(pbn_input)} chars"
    
    else:
        # Single PBN string
        pbn_deals = [pbn_input]
        input_type = "PBN string"
        input_source = f"{len(pbn_input)} chars"
    
    if not pbn_deals:
        raise HTTPException(status_code=400, detail="No valid PBN/LIN deals found")
    
    results = []
    for deal_idx, pbn_str in enumerate(pbn_deals):
        deal = parse_pbn_deal(pbn_str)
        if not deal:
            results.append({"error": f"Invalid PBN: {pbn_str[:50]}..."})
            continue

        # Ensure these keys exist for downstream UI and par-score computation.
        deal.setdefault("pbn", pbn_str)
        deal.setdefault("Dealer", "N")
        
        # Add features for each hand
        for direction in 'NESW':
            hand_col = f'Hand_{direction}'
            if hand_col in deal:
                features = compute_hand_features(deal[hand_col])
                for key, value in features.items():
                    deal[f'{key}_{direction}'] = value
        
        # Compute par if requested
        if req.include_par:
            # Use vulnerability from PBN file if available, otherwise use request
            vul = deal_vuls.get(deal_idx, req.vul)
            deal['Vulnerability'] = vul
            par_info = compute_par_score(pbn_str, str(deal.get('Dealer', 'N')), vul)
            deal.update(par_info)
        
        # Try to find matching deal in deal_df by PBN hands
        # This adds game result columns if the deal exists in bbo_mldf_augmented
        try:
            deal_df, _, _, _ = _ensure_ready()
            
            # Match by all 4 hand strings (exact PBN match)
            match_criteria = pl.lit(True)
            for direction in 'NESW':
                hand_col = f'Hand_{direction}'
                if hand_col in deal and hand_col in deal_df.columns:
                    match_criteria = match_criteria & (pl.col(hand_col) == deal[hand_col])
            
            matching_deals = deal_df.filter(match_criteria)
            
            if matching_deals.height > 0:
                first_match = matching_deals.row(0, named=True)
                
                # Override Dealer and Vulnerability from parquet if matched
                if 'Dealer' in first_match:
                    deal['Dealer'] = first_match['Dealer']
                if 'Vul' in first_match:
                    deal['Vulnerability'] = first_match['Vul']
                
                # Get game result columns from matched deal
                game_result_cols = ['bid', 'Declarer', 'Result', 'Tricks', 'Score', 'ParScore', 'DD_Tricks']
                for col in game_result_cols:
                    if col in first_match:
                        deal[col] = first_match[col]
                deal['matching_deals_in_db'] = matching_deals.height
        except Exception as e:
            # Don't fail if lookup fails
            print(f"[process-pbn] Deal lookup failed: {e}")
        
        results.append(deal)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[process-pbn] {elapsed_ms:.1f}ms ({len(results)} deals, type={input_type})")
    
    return {
        "deals": results,
        "count": len(results),
        "input_type": input_type,
        "input_source": input_source,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# API: find matching auctions
# ---------------------------------------------------------------------------

# evaluate_criterion_for_hand is now imported from bbo_bidding_queries_lib

def _filter_auctions_by_hand_criteria(
    df: pl.DataFrame, 
    hand_values: Dict[str, int],
    seat: int,
    auction_col: str = 'Auction'
) -> tuple[pl.DataFrame, list[dict]]:
    """
    Filter auctions based on custom criteria from bbo_custom_auction_criteria.csv,
    evaluated against a specific hand's values.
    
    Returns (filtered_df, list of rejected auction info for debugging)
    """
    criteria_list = _load_auction_criteria()
    if not criteria_list:
        return df, []
    
    if auction_col not in df.columns:
        return df, []
    
    rejected_info = []
    rows_to_keep = []
    
    for row_idx in range(df.height):
        row = df.row(row_idx, named=True)
        auction = str(row.get(auction_col, '')).lower()
        keep_row = True
        failed_criteria = []
        matched_partial = None
        
        # Check each criterion set
        for partial_auction, criteria in criteria_list:
            if auction.startswith(partial_auction):
                # This auction matches this partial - check criteria
                # Determine which seat made the last bid of the partial auction
                num_dashes = partial_auction.count('-')
                criteria_seat = num_dashes + 1  # 1-based seat number
                
                # Only apply criteria if the bidder matches our seat
                if criteria_seat == seat:
                    matched_partial = partial_auction
                    for criterion in criteria:
                        if not evaluate_criterion_for_hand(criterion, hand_values):
                            keep_row = False
                            failed_criteria.append(criterion)
                    
                    if not keep_row:
                        break  # No need to check more partials
        
        if keep_row:
            rows_to_keep.append(row_idx)
        else:
            rejected_info.append({
                'Auction': str(row.get(auction_col, '')),
                'Partial_Auction': str(matched_partial) if matched_partial else '',
                'Failed_Criteria': ', '.join(failed_criteria),  # Convert list to string for display
                'Seat': int(seat),
            })
    
    if len(rows_to_keep) == df.height:
        return df, rejected_info  # No filtering needed
    
    filtered_df = df[rows_to_keep]
    return filtered_df, rejected_info


@app.post("/find-matching-auctions")
def find_matching_auctions(req: FindMatchingAuctionsRequest) -> Dict[str, Any]:
    """Find auctions from bt_df that match the given hand criteria."""
    t0 = time.perf_counter()
    _, bt_df, _, _ = _ensure_ready()
    
    with _STATE_LOCK:
        bt_criteria = STATE.get("bt_criteria")
        bt_aggregates = STATE.get("bt_aggregates")
        bt_completed_df = STATE.get("bt_completed_df")
    
    seat = req.seat
    
    # Check if criteria columns are available
    if bt_criteria is None:
        raise HTTPException(
            status_code=503, 
            detail="bt_criteria not loaded. Run bt_criteria_extractor.py first."
        )
    
    # Load auction criteria for info
    criteria_list = _load_auction_criteria()
    criteria_loaded = len(criteria_list)
    
    # Use cached completed auctions filter
    if bt_completed_df is not None:
        base_df = bt_completed_df.with_row_index("_idx")
    else:
        base_df = bt_df.with_row_index("_idx")
    
    # Build SQL WHERE clause
    conditions = []
    conditions.append(f'"HCP_min_S{seat}" <= {req.hcp} AND "HCP_max_S{seat}" >= {req.hcp}')
    conditions.append(f'"SL_S_min_S{seat}" <= {req.sl_s} AND "SL_S_max_S{seat}" >= {req.sl_s}')
    conditions.append(f'"SL_H_min_S{seat}" <= {req.sl_h} AND "SL_H_max_S{seat}" >= {req.sl_h}')
    conditions.append(f'"SL_D_min_S{seat}" <= {req.sl_d} AND "SL_D_max_S{seat}" >= {req.sl_d}')
    conditions.append(f'"SL_C_min_S{seat}" <= {req.sl_c} AND "SL_C_max_S{seat}" >= {req.sl_c}')
    conditions.append(f'"Total_Points_min_S{seat}" <= {req.total_points} AND "Total_Points_max_S{seat}" >= {req.total_points}')
    
    where_clause = " AND ".join(conditions)
    
    t1 = time.perf_counter()
    
    # Join bt_df with criteria
    criteria_with_idx = bt_criteria.with_row_index("_idx")
    joined_df = base_df.join(criteria_with_idx, on="_idx", how="inner")
    
    # Also join with aggregates if available
    if bt_aggregates is not None:
        agg_with_idx = bt_aggregates.rename({"bt_row_idx": "_idx"})
        joined_df = joined_df.join(agg_with_idx, on="_idx", how="left")
    
    t2 = time.perf_counter()
    
    # Apply filter using DuckDB (get more than needed to account for criteria filtering)
    sql_query = f"SELECT * FROM joined_df WHERE {where_clause} LIMIT {req.max_results * 3}"
    
    try:
        matching_df = duckdb.sql(sql_query).pl()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e}")
    
    t3 = time.perf_counter()
    
    # Apply custom auction criteria from CSV
    hand_values = {
        'HCP': req.hcp,
        'SL_S': req.sl_s,
        'SL_H': req.sl_h,
        'SL_D': req.sl_d,
        'SL_C': req.sl_c,
        'Total_Points': req.total_points,
    }
    pre_criteria_count = matching_df.height
    matching_df, rejected_auctions = _filter_auctions_by_hand_criteria(
        matching_df, hand_values, seat
    )
    post_criteria_count = matching_df.height
    
    # Limit to requested max
    if matching_df.height > req.max_results:
        matching_df = matching_df.head(req.max_results)
    
    t4 = time.perf_counter()
    
    # Build result columns - Auction first, then key metrics
    result_cols = ["Auction"]
    if "matching_deal_count" in matching_df.columns:
        result_cols.append("matching_deal_count")
    
    # Add Agg_Expr for all 4 seats
    for s in range(1, 5):
        agg_expr_col = f"Agg_Expr_Seat_{s}"
        if agg_expr_col in matching_df.columns:
            result_cols.append(agg_expr_col)
    
    # Add criteria columns for all seats (S1-S4)
    for s in range(1, 5):
        for col in sorted(matching_df.columns):
            if f"_S{s}" in col and col not in result_cols:
                result_cols.append(col)
    
    result_cols = [c for c in result_cols if c in matching_df.columns]
    result_df = matching_df.select(result_cols)  # Include all columns
    
    result_rows = result_df.to_dicts()
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    criteria_filtered = pre_criteria_count - post_criteria_count
    print(f"[find-matching-auctions] {elapsed_ms:.1f}ms ({len(result_rows)} matches, {criteria_filtered} filtered by CSV criteria)")
    print(f"  join: {(t2-t1)*1000:.0f}ms, sql: {(t3-t2)*1000:.0f}ms, csv-criteria: {(t4-t3)*1000:.0f}ms")
    
    response = {
        "sql_query": f"WHERE {where_clause}",
        "auctions": result_rows,
        "total_matches": len(result_rows),
        "seat": seat,
        "criteria": {
            "HCP": req.hcp,
            "SL_S": req.sl_s,
            "SL_H": req.sl_h,
            "SL_D": req.sl_d,
            "SL_C": req.sl_c,
            "Total_Points": req.total_points,
        },
        "auction_criteria_loaded": criteria_loaded,
        "auction_criteria_filtered": criteria_filtered,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    
    # Include rejected auctions for debugging (limit to 10)
    if rejected_auctions:
        response["criteria_rejected"] = rejected_auctions[:10]
    
    return response


# ---------------------------------------------------------------------------
# API: PBN Lookup
# ---------------------------------------------------------------------------

@app.get("/pbn-sample")
def get_pbn_sample() -> Dict[str, Any]:
    """Get a sample PBN from the first row of deal_df for testing."""
    t0 = time.perf_counter()
    deal_df, _, _, _ = _ensure_ready()
    
    # Get first row and construct PBN string
    if deal_df.height == 0:
        raise HTTPException(status_code=404, detail="No deals found in dataset")
    
    first_row = deal_df.row(0, named=True)
    
    # Construct PBN string from Hand_N, Hand_E, Hand_S, Hand_W
    dealer = first_row.get('Dealer', 'N')
    directions = ['N', 'E', 'S', 'W']
    dealer_idx = directions.index(dealer) if dealer in directions else 0
    
    hands = []
    for i in range(4):
        d = directions[(dealer_idx + i) % 4]
        hand = first_row.get(f'Hand_{d}', '')
        hands.append(hand)
    
    pbn = f"{dealer}:" + " ".join(hands)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "pbn": pbn,
        "dealer": dealer,
        "elapsed_ms": round(elapsed_ms, 1),
    }


@app.get("/pbn-random")
def get_pbn_random() -> Dict[str, Any]:
    """Get a random PBN from deal_df (YOLO mode)."""
    t0 = time.perf_counter()
    deal_df, _, _, _ = _ensure_ready()
    
    if deal_df.height == 0:
        raise HTTPException(status_code=404, detail="No deals found in dataset")
    
    # Pick a random row
    random_idx = random.randint(0, deal_df.height - 1)
    random_row = deal_df.row(random_idx, named=True)
    
    # Construct PBN string from Hand_N, Hand_E, Hand_S, Hand_W
    dealer = random_row.get('Dealer', 'N')
    directions = ['N', 'E', 'S', 'W']
    dealer_idx = directions.index(dealer) if dealer in directions else 0
    
    hands = []
    for i in range(4):
        d = directions[(dealer_idx + i) % 4]
        hand = random_row.get(f'Hand_{d}', '')
        hands.append(hand)
    
    pbn = f"{dealer}:" + " ".join(hands)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "pbn": pbn,
        "dealer": dealer,
        "row_idx": random_idx,
        "elapsed_ms": round(elapsed_ms, 1),
    }


@app.post("/pbn-lookup")
def pbn_lookup(req: PBNLookupRequest) -> Dict[str, Any]:
    """Look up a PBN deal in bbo_mldf_augmented.parquet and return matching rows."""
    t0 = time.perf_counter()
    deal_df, _, _, _ = _ensure_ready()
    
    pbn_input = req.pbn.strip()
    
    # Parse PBN to get individual hands
    parsed = parse_pbn_deal(pbn_input)
    if not parsed:
        raise HTTPException(status_code=400, detail=f"Invalid PBN format: {pbn_input[:100]}")
    
    # Build match criteria based on all 4 hands
    match_criteria = pl.lit(True)
    for direction in 'NESW':
        hand_col = f'Hand_{direction}'
        if hand_col in parsed and hand_col in deal_df.columns:
            match_criteria = match_criteria & (pl.col(hand_col) == parsed[hand_col])
    
    # Find matching rows
    matching = deal_df.filter(match_criteria)
    
    # Limit results
    if matching.height > req.max_results:
        matching = matching.head(req.max_results)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[pbn-lookup] Found {matching.height} matches in {elapsed_ms:.1f}ms")
    
    return {
        "matches": matching.to_dicts(),
        "count": matching.height,
        "total_in_df": deal_df.height,
        "pbn_searched": pbn_input,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# API: Group by Bid
# ---------------------------------------------------------------------------

# calculate_imp is now imported from bbo_bidding_queries_lib

@app.post("/group-by-bid")
def group_by_bid(req: GroupByBidRequest) -> Dict[str, Any]:
    """
    Group deals by their actual auction sequence (bid column) and show deal characteristics.
    
    This joins deal_df with bt_df to get bidding table info (Expr, Agg_Expr_Seat_*).
    """
    t0 = time.perf_counter()
    deal_df, bt_df, _, _ = _ensure_ready()
    
    with _STATE_LOCK:
        bt_completed_df = STATE.get("bt_completed_df")
    
    # Check if 'bid' column exists
    if 'bid' not in deal_df.columns:
        raise HTTPException(status_code=500, detail="Column 'bid' not found in deal_df")
    
    # Filter by auction pattern
    pattern = normalize_auction_pattern(req.auction_pattern)
    
    # Build a dash-joined auction string for filtering/grouping.
    # Note: `bid` is usually List[str], but can contain nulls (and we want to be robust to any odd rows).
    try:
        bid_dtype = deal_df.schema.get("bid")
        if bid_dtype == pl.List(pl.Utf8):
            deal_df_with_str = deal_df.with_columns(pl.col("bid").list.join("-").alias("bid_str"))
        elif bid_dtype == pl.Utf8:
            deal_df_with_str = deal_df.with_columns(pl.col("bid").fill_null("").alias("bid_str"))
        else:
            # Fallback for unexpected schemas
            bid_str_expr = pl.col("bid").map_elements(
                lambda x: "-".join(map(str, x)) if isinstance(x, list) else (str(x) if x is not None else ""),
                return_dtype=pl.Utf8,
            )
            deal_df_with_str = deal_df.with_columns(bid_str_expr.alias("bid_str"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build bid_str from 'bid' column: {e}")

    try:
        # Case-insensitive regex matching
        regex_pattern = f"(?i){pattern}"
        filtered_df = deal_df_with_str.filter(pl.col("bid_str").str.contains(regex_pattern))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {e}")
    
    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "pattern": pattern,
            "auction_groups": [],
            "total_matching_deals": 0,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    
    t1 = time.perf_counter()
    
    # Group by dash-joined auction string and count
    bid_counts = (
        filtered_df.group_by("bid_str")
        .agg(pl.len().alias("deal_count"))
        .sort("deal_count", descending=True)
    )
    
    # Filter by minimum deals
    if req.min_deals > 1:
        bid_counts = bid_counts.filter(pl.col("deal_count") >= req.min_deals)
    
    # Sample auction groups
    effective_seed = _effective_seed(req.seed)
    n_groups = min(req.n_auction_groups, bid_counts.height)
    if n_groups < bid_counts.height:
        sampled_bids = bid_counts.sample(n=n_groups, seed=effective_seed)
    else:
        sampled_bids = bid_counts
    
    t2 = time.perf_counter()
    
    # Define columns to include
    deal_cols = [
        "index", "Dealer", "Vul", "Declarer","bid",
        "Result", "Tricks", "Score", "ParScore",
        "Hand_N", "Hand_E", "Hand_S", "Hand_W",
        "HCP_N", "HCP_E", "HCP_S", "HCP_W",
        "SL_S_N", "SL_H_N", "SL_D_N", "SL_C_N",
        "SL_S_E", "SL_H_E", "SL_D_E", "SL_C_E",
        "SL_S_S", "SL_H_S", "SL_D_S", "SL_C_S",
        "SL_S_W", "SL_H_W", "SL_D_W", "SL_C_W",
        "Total_Points_N", "Total_Points_E", "Total_Points_S", "Total_Points_W",
    ]
    available_deal_cols = [c for c in deal_cols if c in filtered_df.columns]
    
    # Get bt_df columns for joining (Expr, Agg_Expr_Seat_*)
    # todo: add "Announcement" column after 'Auction' column.
    bt_cols = ["Auction", "seat", "Expr", "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4"]
    available_bt_cols = [c for c in bt_cols if c in bt_df.columns]
    
    # Use completed auctions for bt lookup
    bt_lookup_df = bt_completed_df if bt_completed_df is not None else bt_df
    
    auction_groups = []
    
    for row in sampled_bids.iter_rows(named=True):
        bid_auction = row["bid_str"]
        deal_count = row["deal_count"]
        
        # Get all deals for this auction (before sampling)
        group_all_deals = filtered_df.filter(pl.col("bid_str") == bid_auction).select(available_deal_cols)
        
        # Sample deals within group for display
        n_samples = min(req.n_deals_per_group, group_all_deals.height)
        if n_samples < group_all_deals.height:
            group_deals = group_all_deals.sample(n=n_samples, seed=effective_seed)
        else:
            group_deals = group_all_deals
        
        # Identify board columns for duplicate / matchpoint analysis
        board_cols = ["Dealer", "Vul", "Hand_N", "Hand_E", "Hand_S", "Hand_W"]
        has_board_cols = all(col in group_deals.columns for col in board_cols)
        max_dup_all = max_dup_sample = 0
        boards_with_dups_all = boards_with_dups_sample = 0
        boards_total_all = boards_total_sample = 0
        
        if has_board_cols and group_all_deals.height > 0:
            board_counts_all = group_all_deals.group_by(board_cols).len()
            boards_total_all = board_counts_all.height
            if boards_total_all > 0:
                max_val_all = board_counts_all["len"].max()
                if isinstance(max_val_all, (int, float)):
                    max_dup_all = int(max_val_all)
                boards_with_dups_all = board_counts_all.filter(pl.col("len") > 1).height
        
        if has_board_cols and group_deals.height > 0:
            board_counts_sample = group_deals.group_by(board_cols).len()
            boards_total_sample = board_counts_sample.height
            if boards_total_sample > 0:
                max_val_sample = board_counts_sample["len"].max()
                if isinstance(max_val_sample, (int, float)):
                    max_dup_sample = int(max_val_sample)
                boards_with_dups_sample = board_counts_sample.filter(pl.col("len") > 1).height
        
        # Try to find matching bt_df row for this auction
        bt_info = None
        bt_auction = None  # The matched Auction from bt_df (standardized format)
        if available_bt_cols:
            # Normalize auction for matching (add -p-p-p if needed for completed auction)
            auction_normalized = bid_auction.lower() if bid_auction else ""
            
            # Try exact match first
            bt_match = bt_lookup_df.filter(
                pl.col("Auction").cast(pl.Utf8).str.to_lowercase() == auction_normalized
            )
            
            # If no match, try with -p-p-p suffix
            if bt_match.height == 0 and not auction_normalized.endswith("-p-p-p"):
                auction_with_passes = auction_normalized + "-p-p-p"
                bt_match = bt_lookup_df.filter(
                    pl.col("Auction").cast(pl.Utf8).str.to_lowercase() == auction_with_passes
                )
            
            if bt_match.height > 0:
                bt_row = bt_match.row(0, named=True)
                bt_info = {c: bt_row.get(c) for c in available_bt_cols if c in bt_row}
                bt_auction = bt_row.get("Auction")  # Get the exact Auction string from bt_df
        
        # Add 'Auction' column from bt_df to each deal row
        if bt_auction:
            group_deals = group_deals.with_columns(pl.lit(bt_auction).alias("Auction"))
        
        # Check validation: does this deal satisfy the criteria for this auction?
        # Only if we have criteria (Agg_Expr_Seat_*) and hand features (HCP_*, SL_*_*)
        if bt_info:
            # Determine which seat opened the bidding (from Auction string)
            # Standard GIB format: dealers rotate, but we need to map hand columns to seats
            # Seat 1 = Dealer
            # Seat 2 = (Dealer + 1) % 4
            # Seat 3 = (Dealer + 2) % 4
            # Seat 4 = (Dealer + 3) % 4
            
            # We'll compute validation for each row
            validation_results = []
            violations_list = []
            
            for deal_row in group_deals.iter_rows(named=True):
                dealer = deal_row.get("Dealer", "N")
                directions = ["N", "E", "S", "W"]
                dealer_idx = directions.index(dealer) if dealer in directions else 0
                
                is_valid = True
                violations = []
                
                # Check criteria for each seat
                for seat in range(1, 5):
                    agg_col = f"Agg_Expr_Seat_{seat}"
                    criteria = bt_info.get(agg_col)
                    if not criteria:
                        continue
                        
                    # Map seat to direction for this deal
                    # Seat 1 is Dealer, Seat 2 is LHO, etc.
                    direction = directions[(dealer_idx + seat - 1) % 4]
                    
                    # Build hand values dict for this direction
                    hand_values = {
                        'HCP': deal_row.get(f"HCP_{direction}"),
                        'Total_Points': deal_row.get(f"Total_Points_{direction}"),
                        'SL_S': deal_row.get(f"SL_S_{direction}"),
                        'SL_H': deal_row.get(f"SL_H_{direction}"),
                        'SL_D': deal_row.get(f"SL_D_{direction}"),
                        'SL_C': deal_row.get(f"SL_C_{direction}"),
                    }
                    
                    # Validate each criterion
                    for criterion in criteria:
                        if not evaluate_criterion_for_hand(criterion, hand_values):  # type: ignore[arg-type]
                            is_valid = False
                            # Format violation: "HCP(14) < 15"
                            # Parse criterion to get LHS
                            match = re.match(r'(\w+)\s*(>=|<=|>|<|==|!=)\s*(\w+|\d+)', criterion)
                            if match:
                                lhs, op, rhs = match.groups()
                                actual_val = hand_values.get(lhs)
                                if actual_val is not None:
                                    violations.append(f"{lhs}_{direction}({actual_val}) violated {criterion}")
                                else:
                                    violations.append(f"{criterion} (missing data)")
                            else:
                                violations.append(f"{criterion} (parse error)")
                
                validation_results.append(is_valid)
                violations_list.append("; ".join(violations) if violations else "")
            
            # Add validation columns
            group_deals = group_deals.with_columns([
                pl.Series("Auctions_Match", validation_results),
                pl.Series("Criteria_Violations", violations_list)
            ])
        
        # Calculate Score_Delta (Score - ParScore) and Score_IMP
        # (outside bt_info block - we want stats even without bidding table match)
        if "Score" in group_deals.columns and "ParScore" in group_deals.columns:
            group_deals = group_deals.with_columns(
                (
                    pl.col("Score").cast(pl.Int64, strict=False) - 
                    pl.col("ParScore").cast(pl.Int64, strict=False)
                ).alias("Score_Delta")
            )
            
            # Calculate IMPs from Score_Delta
            group_deals = group_deals.with_columns(
                pl.col("Score_Delta").map_elements(
                    lambda x: calculate_imp(x) * (1 if x >= 0 else -1) if x is not None else None, 
                    return_dtype=pl.Int64
                ).alias("Score_IMP")
            )

        # Calculate matchpoint-style scores when duplicate boards exist (vectorized, no Python row loops)
        mp_board_count = 0
        if has_board_cols and "Score" in group_deals.columns and group_deals.height > 0:
            # Boards with duplicates in this sample
            board_sizes = group_deals.group_by(board_cols).len()
            mp_board_count = board_sizes.filter(pl.col("len") > 1).height

            if mp_board_count > 0:
                df_mp = group_deals.with_row_index("__row").with_columns(
                    pl.col("Score").cast(pl.Float64, strict=False).alias("__Score_f")
                )

                # Aggregate counts per (board, score)
                levels = (
                    df_mp.group_by(board_cols + ["__Score_f"])
                    .agg(pl.len().alias("__cnt"))
                    .with_columns(pl.col("__cnt").sum().over(board_cols).alias("__n"))
                    .sort(board_cols + ["__Score_f"])
                    .with_columns(pl.col("__cnt").cum_sum().over(board_cols).alias("__cum"))
                    .with_columns((pl.col("__cum") - pl.col("__cnt")).alias("__beats"))
                    .with_columns((pl.col("__cnt") - 1).alias("__ties"))
                    .with_columns((pl.col("__beats") * 2 + pl.col("__ties")).alias("__mp"))
                    .with_columns((2 * (pl.col("__n") - 1)).alias("__max_mp"))
                    .with_columns(
                        pl.when(pl.col("__n") >= 2)
                        .then(pl.col("__mp") / pl.col("__max_mp"))
                        .otherwise(None)
                        .alias("__pct")
                    )
                    .select(board_cols + ["__Score_f", "__mp", "__pct"])
                )

                df_mp = df_mp.join(levels, on=board_cols + ["__Score_f"], how="left").select(
                    ["__row", pl.col("__mp").alias("Score_MP"), pl.col("__pct").alias("Score_MP_Pct")]
                )

                group_deals = (
                    group_deals.with_row_index("__row")
                    .join(df_mp, on="__row", how="left")
                    .drop("__row")
                )

        # Compute statistics for this auction group
        stats = {}
        if has_board_cols:
            stats["Max_Duplicates_All"] = max_dup_all
            stats["Boards_With_Duplicates_All"] = boards_with_dups_all
            stats["Boards_Total_All"] = boards_total_all
            stats["Max_Duplicates_Sample"] = max_dup_sample
            stats["Boards_With_Duplicates_Sample"] = boards_with_dups_sample
            stats["Boards_Total_Sample"] = boards_total_sample
            stats["Boards_With_MP_Data"] = mp_board_count
        
        match_rows = None
        non_match_rows = None
        if "Auctions_Match" in group_deals.columns:
            match_rows = group_deals.filter(pl.col("Auctions_Match"))
            non_match_rows = group_deals.filter(~pl.col("Auctions_Match"))
        
        # Score Delta & IMP stats
        # Helper to safely round values that might be None
        def safe_round(val, decimals=1):
            return round(float(val), decimals) if val is not None else None
        
        if "Score_Delta" in group_deals.columns:
            # Overall avg & stddev
            delta_vals = group_deals["Score_Delta"].drop_nulls().cast(pl.Float64)
            if len(delta_vals) > 0:
                stats["Score_Delta_Avg"] = safe_round(delta_vals.mean())
                stats["Score_Delta_StdDev"] = safe_round(delta_vals.std())
            
            # IMP stats
            if "Score_IMP" in group_deals.columns:
                imp_vals = group_deals["Score_IMP"].drop_nulls().cast(pl.Float64)
                if len(imp_vals) > 0:
                    stats["Score_IMP_Avg"] = safe_round(imp_vals.mean())
                    stats["Score_IMP_StdDev"] = safe_round(imp_vals.std())
            
            # By compliance
            if match_rows is not None and non_match_rows is not None:
                # Match stats
                match_deltas = match_rows["Score_Delta"].drop_nulls().cast(pl.Float64)
                if len(match_deltas) > 0:
                    stats["Score_Delta_Match_Avg"] = safe_round(match_deltas.mean())
                    stats["Score_Delta_Match_StdDev"] = safe_round(match_deltas.std())
                    stats["Match_Count"] = len(match_deltas)
                    
                    if "Score_IMP" in match_rows.columns:
                        match_imps = match_rows["Score_IMP"].drop_nulls().cast(pl.Float64)
                        if len(match_imps) > 0:
                            stats["Score_IMP_Match_Avg"] = safe_round(match_imps.mean())
                
                # No Match stats
                non_match_deltas = non_match_rows["Score_Delta"].drop_nulls().cast(pl.Float64)
                if len(non_match_deltas) > 0:
                    stats["Score_Delta_NoMatch_Avg"] = safe_round(non_match_deltas.mean())
                    stats["Score_Delta_NoMatch_StdDev"] = safe_round(non_match_deltas.std())
                    stats["NoMatch_Count"] = len(non_match_deltas)
                    
                    if "Score_IMP" in non_match_rows.columns:
                        non_match_imps = non_match_rows["Score_IMP"].drop_nulls().cast(pl.Float64)
                        if len(non_match_imps) > 0:
                            stats["Score_IMP_NoMatch_Avg"] = safe_round(non_match_imps.mean())

        # Matchpoint stats
        if "Score_MP" in group_deals.columns:
            mp_vals = group_deals["Score_MP"].drop_nulls().cast(pl.Float64)
            if len(mp_vals) > 0:
                stats["Score_MP_Avg"] = safe_round(mp_vals.mean())
                stats["Score_MP_StdDev"] = safe_round(mp_vals.std())
            if match_rows is not None and non_match_rows is not None:
                match_mp_vals = match_rows["Score_MP"].drop_nulls().cast(pl.Float64)
                if len(match_mp_vals) > 0:
                    stats["Score_MP_Match_Avg"] = safe_round(match_mp_vals.mean())
                    stats["Score_MP_Match_StdDev"] = safe_round(match_mp_vals.std())
                non_match_mp_vals = non_match_rows["Score_MP"].drop_nulls().cast(pl.Float64)
                if len(non_match_mp_vals) > 0:
                    stats["Score_MP_NoMatch_Avg"] = safe_round(non_match_mp_vals.mean())
                    stats["Score_MP_NoMatch_StdDev"] = safe_round(non_match_mp_vals.std())
        
        if "Score_MP_Pct" in group_deals.columns:
            mp_pct_vals = group_deals["Score_MP_Pct"].drop_nulls().cast(pl.Float64)
            if len(mp_pct_vals) > 0:
                mp_pct_mean = mp_pct_vals.mean()
                mp_pct_std = mp_pct_vals.std()
                stats["Score_MP_Pct_Avg"] = safe_round(float(mp_pct_mean) * 100) if mp_pct_mean is not None else None  # type: ignore[arg-type]
                stats["Score_MP_Pct_StdDev"] = safe_round(float(mp_pct_std) * 100) if mp_pct_std is not None else None  # type: ignore[arg-type]
            if match_rows is not None and non_match_rows is not None:
                match_pct_vals = match_rows["Score_MP_Pct"].drop_nulls().cast(pl.Float64)
                if len(match_pct_vals) > 0:
                    match_pct_mean = match_pct_vals.mean()
                    stats["Score_MP_Match_Pct_Avg"] = safe_round(float(match_pct_mean) * 100) if match_pct_mean is not None else None  # type: ignore[arg-type]
                non_match_pct_vals = non_match_rows["Score_MP_Pct"].drop_nulls().cast(pl.Float64)
                if len(non_match_pct_vals) > 0:
                    non_match_pct_mean = non_match_pct_vals.mean()
                    stats["Score_MP_NoMatch_Pct_Avg"] = safe_round(float(non_match_pct_mean) * 100) if non_match_pct_mean is not None else None  # type: ignore[arg-type]

        for direction in "NESW":
            hcp_col = f"HCP_{direction}"
            tp_col = f"Total_Points_{direction}"
            if hcp_col in group_deals.columns:
                hcp_values = group_deals[hcp_col].drop_nulls().cast(pl.Float64)
                if len(hcp_values) > 0:
                    hcp_mean = hcp_values.mean()
                    hcp_min = hcp_values.min()
                    hcp_max = hcp_values.max()
                    if hcp_mean is not None:
                        stats[f"HCP_{direction}_avg"] = round(hcp_mean, 1)  # type: ignore[arg-type]
                    if hcp_min is not None:
                        stats[f"HCP_{direction}_min"] = int(hcp_min)  # type: ignore[arg-type]
                    if hcp_max is not None:
                        stats[f"HCP_{direction}_max"] = int(hcp_max)  # type: ignore[arg-type]
            if tp_col in group_deals.columns:
                tp_values = group_deals[tp_col].drop_nulls().cast(pl.Float64)
                if len(tp_values) > 0:
                    tp_mean = tp_values.mean()
                    if tp_mean is not None:
                        stats[f"TP_{direction}_avg"] = round(tp_mean, 1)  # type: ignore[arg-type]
        
        auction_groups.append({
            "auction": bid_auction,  # From deal_df 'bid' column
            "bt_auction": bt_auction,  # Matched Auction from bt_df (with -p-p-p if completed)
            "deal_count": deal_count,
            "sample_count": group_deals.height,
            "bt_info": bt_info,
            "stats": stats,
            "deals": group_deals.to_dicts(),
        })
    
    t3 = time.perf_counter()
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_deals = sum(g["sample_count"] for g in auction_groups)
    print(f"[group-by-bid] {elapsed_ms:.1f}ms ({len(auction_groups)} groups, {total_deals} deals)")
    print(f"  filter: {(t1-t0)*1000:.0f}ms, group: {(t2-t1)*1000:.0f}ms, build: {(t3-t2)*1000:.0f}ms")
    
    return {
        "pattern": pattern,
        "auction_groups": auction_groups,
        "total_matching_deals": filtered_df.height,
        "unique_auctions": bid_counts.height,
        "elapsed_ms": round(elapsed_ms, 1),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="BBO Bidding Queries API")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory containing parquet files (overrides auto-detection)",
    )
    parser.add_argument(
        "--deal-rows",
        type=int,
        default=1000,
        help="Limit deal_df rows for faster startup (default: 1000) or None for all rows",
    )
    args = parser.parse_args()

    print("Starting API server...")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
