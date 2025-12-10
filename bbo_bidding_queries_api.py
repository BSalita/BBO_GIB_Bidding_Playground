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
import duckdb
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

rootPath = pathlib.Path("e:/bridge/data")
if rootPath.exists():
    bboPath = rootPath.joinpath("bbo")
    dataPath = bboPath.joinpath("data")
    biddingPath = bboPath.joinpath("bidding")
else:
    rootPath = pathlib.Path('.')
    if not rootPath.exists():
        raise ValueError(f'rootPath does not exist: {rootPath}')
    bboPath = rootPath.joinpath("")
    dataPath = bboPath.joinpath("data")
    biddingPath = bboPath.joinpath("data")
print(f'rootPath: {rootPath}')
print(f'bboPath: {bboPath}')
print(f'dataPath: {dataPath}')
print(f'biddingPath: {biddingPath}')


# ---------------------------------------------------------------------------
# Required files check
# ---------------------------------------------------------------------------

exec_plan_file = biddingPath.joinpath("bbo_bt_execution_plan_data.pkl") # todo: rename to bbo_bt_execution_plan_data.pkl
bbo_mldf_augmented_file = dataPath.joinpath("bbo_mldf_augmented.parquet")
bbo_bidding_table_augmented_file = biddingPath.joinpath("bbo_bt_augmented.parquet")
bt_aggregates_file = dataPath.joinpath("bbo_bt_aggregate.parquet")

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
    pbn: str  # PBN string or URL
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


# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------


STATE: Dict[str, Any] = {
    "initialized": False,
    "initializing": False,
    "warming": False,  # True while pre-warming endpoints
    "error": None,
    "deal_df": None,
    "bt_df": None,
    "bt_completed_df": None,  # Cached: bt_df.filter(is_completed_auction)
    "deal_criteria_by_seat_dfs": None,
    "deal_criteria_by_direction_dfs": None,
    "results": None,
    "bt_criteria": None,  # bt_criteria.parquet
    "bt_aggregates": None,  # bbo_bt_aggregate.parquet
}

# Additional data file paths (bt_aggregates_file defined above with REQUIRED_FILES)
bt_criteria_file = biddingPath.joinpath("bt_criteria.parquet")

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
    try:
        (
            directionless_criteria_cols,
            expr_map_by_direction,
            valid_deal_columns,
            pythonized_exprs_by_direction,
        ) = load_execution_plan_data(exec_plan_file)
        _log_memory("after load_execution_plan_data")

        # Load deals
        deal_df = load_deal_df(bbo_mldf_augmented_file, valid_deal_columns, mldf_n_rows=None)
        _log_memory("after load_deal_df")

        # Build criteria bitmaps and derive per-seat/per-dealer views
        criteria_deal_dfs_directional = build_or_load_directional_criteria_bitmaps(
            deal_df,
            pythonized_exprs_by_direction,
            expr_map_by_direction,
        )
        _log_memory("after build_or_load_directional_criteria_bitmaps")

        deal_criteria_by_direction_dfs, deal_criteria_by_seat_dfs = directional_to_directionless(
            criteria_deal_dfs_directional, expr_map_by_direction
        )
        _log_memory("after directional_to_directionless")

        # We no longer need these large helper objects
        del criteria_deal_dfs_directional, pythonized_exprs_by_direction, directionless_criteria_cols
        gc.collect()
        _log_memory("after gc.collect (criteria cleanup)")

        # Load bidding table
        bt_df = load_bt_df(bbo_bidding_table_augmented_file, include_expr_and_sequences=True)
        _log_memory("after load_bt_df")

        # Compute opening-bid candidates for all (dealer, seat) combinations
        results = process_opening_bids(
            deal_df,
            bt_df,
            deal_criteria_by_seat_dfs,
            bbo_bidding_table_augmented_file,
        )
        _log_memory("after process_opening_bids")

        # Load optional aggregates files (non-blocking if missing)
        bt_criteria = None
        bt_aggregates = None
        
        if bt_criteria_file.exists():
            print(f"[init] Loading bt_criteria from {bt_criteria_file}...")
            bt_criteria = pl.read_parquet(bt_criteria_file)
            _log_memory("after load bt_criteria")
        else:
            print(f"[init] bt_criteria not found at {bt_criteria_file} (optional)")
        
        if bt_aggregates_file.exists():
            print(f"[init] Loading bt_aggregates from {bt_aggregates_file}...")
            bt_aggregates = pl.read_parquet(bt_aggregates_file)
            _log_memory("after load bt_aggregates")
        else:
            print(f"[init] bt_aggregates not found at {bt_aggregates_file} (optional)")

        # Cache the completed auctions filter (expensive: ~2 min on 541M rows)
        bt_completed_df = None
        if "is_completed_auction" in bt_df.columns:
            print("[init] Caching bt_completed_df (is_completed_auction filter)...")
            bt_completed_df = bt_df.filter(pl.col("is_completed_auction"))
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

    dirs = ["N", "E", "S", "W"]
    dir_to_idx = {d: i for i, d in enumerate(dirs)}

    seats_to_process = req.seats if req.seats is not None else [1, 2, 3, 4]
    directions_to_process = req.directions if req.directions is not None else dirs
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
                    opener_for_seat = dirs[(dealer_idx + seat - 1) % 4]
                    if opening_dirs_filter is None or opener_for_seat in opening_dirs_filter:
                        if opening_seat_num is None:
                            opening_seat_num = seat
                        opening_bids.extend(int(b) for b in bids)

        if not opening_bids:
            continue

        opening_seat = dirs[(dealer_idx + opening_seat_num - 1) % 4] if opening_seat_num else None
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
    # seed=0 means non-reproducible (None), any other value is reproducible
    effective_seed = None if req.seed == 0 else req.seed
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

    pattern = req.pattern
    three_passes = "-p-p-p"
    pattern_core = pattern.rstrip("$")
    if not pattern_core.endswith(three_passes):
        if pattern.endswith("$"):
            pattern = pattern[:-1] + three_passes + "$"
        else:
            pattern = pattern + three_passes

    is_regex = pattern.startswith("^") or pattern.endswith("$")
    if is_regex:
        # Use (?i) for case-insensitive matching
        regex_pattern = f"(?i){pattern}"
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(regex_pattern))
    else:
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(f"(?i){pattern}"))

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[auction-sequences-matching] {elapsed_ms:.1f}ms (no matches)")
        return {"samples": [], "pattern": pattern, "elapsed_ms": round(elapsed_ms, 1)}

    sample_n = min(req.n_samples, filtered_df.height)
    # seed=0 means non-reproducible (None), any other value is reproducible
    effective_seed = None if req.seed == 0 else req.seed
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
    return {"pattern": pattern, "samples": out_samples, "elapsed_ms": round(elapsed_ms, 1)}


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

    pattern = req.pattern
    three_passes = "-p-p-p"
    pattern_core = pattern.rstrip("$")
    if not pattern_core.endswith(three_passes):
        if pattern.endswith("$"):
            pattern = pattern[:-1] + three_passes + "$"
        else:
            pattern = pattern + three_passes

    is_regex = pattern.startswith("^") or pattern.endswith("$")
    if is_regex:
        # Use (?i) for case-insensitive matching
        regex_pattern = f"(?i){pattern}"
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(regex_pattern))
    else:
        filtered_df = base_df.filter(pl.col("Auction").cast(pl.Utf8).str.contains(f"(?i){pattern}"))

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[deals-matching-auction] {elapsed_ms:.1f}ms (no matches)")
        return {"pattern": pattern, "auctions": [], "elapsed_ms": round(elapsed_ms, 1)}

    sample_n = min(req.n_auction_samples, filtered_df.height)
    # seed=0 means non-reproducible (None), any other value is reproducible
    effective_seed = None if req.seed == 0 else req.seed
    sampled_auctions = filtered_df.sample(n=sample_n, seed=effective_seed)

    dirs = ["N", "E", "S", "W"]
    deal_display_cols = ["index", "Dealer", "Hand_N", "Hand_E", "Hand_S", "Hand_W"]

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
        }

        # Criteria by seat
        for s in range(1, 5):
            agg_col = f"Agg_Expr_Seat_{s}"
            if agg_col in auction_row:
                crit_list = auction_row[agg_col]
                if crit_list:
                    auction_info["criteria_by_seat"][str(s)] = crit_list

        matching_deals: List[pl.DataFrame] = []

        for dealer in dirs:
            dealer_mask = deal_df["Dealer"] == dealer
            if not dealer_mask.any():
                continue

            combined_mask = dealer_mask.clone()

            for s in range(1, seat + 1):
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
                for idx in matching_idx[: req.n_deal_samples]:
                    matching_deals.append(deal_df[idx])

        if matching_deals:
            combined_df = pl.concat(matching_deals[: req.n_deal_samples])
            
            # Apply distribution filter if specified
            if req.dist_pattern or req.sorted_shape:
                direction = req.dist_direction.upper()
                if direction in 'NESW':
                    # Add suit length columns for filtering
                    combined_df = _add_suit_length_columns(combined_df, direction)
                    
                    # Build and apply filter
                    dist_where = _build_distribution_sql_for_deals(
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
            
            auction_info["deals"] = combined_df.select(
                [c for c in deal_display_cols if c in combined_df.columns]
            ).to_dicts()

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
    return response


# ---------------------------------------------------------------------------
# API: bidding table statistics
# ---------------------------------------------------------------------------

import random as random_module


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
        dist_where = _build_distribution_sql_for_bt(
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


# ---------------------------------------------------------------------------
# Distribution filter utilities
# ---------------------------------------------------------------------------

def _parse_distribution_pattern(pattern: str) -> dict | None:
    """Parse a suit distribution pattern into filter criteria (S-H-D-C order)."""
    import re
    
    if not pattern or not pattern.strip():
        return None
    
    pattern = pattern.strip()
    suits = ['S', 'H', 'D', 'C']
    result: dict[str, tuple[int, int] | None] = {s: None for s in suits}
    
    # Try compact numeric format (e.g., '4333', '5332')
    if re.match(r'^[0-9]{4}$', pattern):
        for i, suit in enumerate(suits):
            val = int(pattern[i])
            result[suit] = (val, val)
        return result
    
    # Split by dash for other formats
    parts = pattern.split('-')
    if len(parts) != 4:
        return None
    
    for i, (part, suit) in enumerate(zip(parts, suits)):
        part = part.strip()
        
        if part.lower() == 'x' or part in ('.*', '.+', '*', ''):
            result[suit] = None
            continue
        
        bracket_match = re.match(r'^\[(\d+)[-:](\d+)\]$', part)
        if bracket_match:
            result[suit] = (int(bracket_match.group(1)), int(bracket_match.group(2)))
            continue
        
        colon_match = re.match(r'^(\d+):(\d+)$', part)
        if colon_match:
            result[suit] = (int(colon_match.group(1)), int(colon_match.group(2)))
            continue
        
        plus_match = re.match(r'^(\d+)\+$', part)
        if plus_match:
            result[suit] = (int(plus_match.group(1)), 13)
            continue
        
        minus_match = re.match(r'^(\d+)-$', part)
        if minus_match:
            result[suit] = (0, int(minus_match.group(1)))
            continue
        
        if re.match(r'^\d+$', part):
            val = int(part)
            result[suit] = (val, val)
            continue
        
        result[suit] = None
    
    return result


def _parse_sorted_shape(pattern: str) -> list[int] | None:
    """Parse a sorted shape pattern (e.g., '5431', '4432')."""
    import re
    
    if not pattern or not pattern.strip():
        return None
    
    pattern = pattern.strip()
    
    if re.match(r'^[0-9]{4}$', pattern):
        lengths = [int(c) for c in pattern]
        if sum(lengths) == 13:
            return sorted(lengths, reverse=True)
        return None
    
    parts = pattern.split('-')
    if len(parts) == 4:
        try:
            lengths = [int(p.strip()) for p in parts]
            if sum(lengths) == 13:
                return sorted(lengths, reverse=True)
        except ValueError:
            pass
    
    return None


def _build_distribution_sql_for_bt(
    dist_pattern: str | None,
    sorted_shape: str | None,
    seat: int,
    available_columns: list[str]
) -> str:
    """Build SQL WHERE clause for bt_df distribution filtering."""
    from itertools import permutations
    
    conditions = []
    suits = ['S', 'H', 'D', 'C']
    
    if dist_pattern:
        parsed = _parse_distribution_pattern(dist_pattern)
        if parsed:
            for suit, constraint in parsed.items():
                if constraint is None:
                    continue
                min_val, max_val = constraint
                min_col = f"SL_{suit}_min_S{seat}"
                max_col = f"SL_{suit}_max_S{seat}"
                
                if min_col in available_columns and max_col in available_columns:
                    conditions.append(f'"{min_col}" <= {max_val}')
                    conditions.append(f'"{max_col}" >= {min_val}')
    
    if sorted_shape:
        shape = _parse_sorted_shape(sorted_shape)
        if shape:
            perm_conditions = []
            unique_perms = set(permutations(shape))
            
            for perm in unique_perms:
                perm_parts = []
                for suit, expected_len in zip(suits, perm):
                    min_col = f"SL_{suit}_min_S{seat}"
                    max_col = f"SL_{suit}_max_S{seat}"
                    if min_col in available_columns and max_col in available_columns:
                        perm_parts.append(
                            f'("{min_col}" <= {expected_len} AND "{max_col}" >= {expected_len})'
                        )
                if len(perm_parts) == 4:
                    perm_conditions.append(f"({' AND '.join(perm_parts)})")
            
            if perm_conditions:
                conditions.append(f"({' OR '.join(perm_conditions)})")
    
    return " AND ".join(conditions) if conditions else ""


def _build_distribution_sql_for_deals(
    dist_pattern: str | None,
    sorted_shape: str | None,
    direction: str
) -> str:
    """Build SQL WHERE clause for deal_df distribution filtering (Hand_* columns)."""
    from itertools import permutations
    
    conditions = []
    suits = ['S', 'H', 'D', 'C']
    
    if dist_pattern:
        parsed = _parse_distribution_pattern(dist_pattern)
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
    
    if sorted_shape:
        shape = _parse_sorted_shape(sorted_shape)
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
    
    return " AND ".join(conditions) if conditions else ""


def _add_suit_length_columns(df: pl.DataFrame, direction: str) -> pl.DataFrame:
    """Add suit length columns for a specific direction's hand."""
    hand_col = f"Hand_{direction}"
    if hand_col not in df.columns:
        return df
    
    suits = ['S', 'H', 'D', 'C']
    for suit_idx, suit in enumerate(suits):
        col_name = f"SL_{suit}_{direction}"
        df = df.with_columns(
            pl.col(hand_col).str.split('.').list.get(suit_idx).str.len_chars().alias(col_name)
        )
    
    return df


# ---------------------------------------------------------------------------
# API: process PBN
# ---------------------------------------------------------------------------

def _parse_pbn_deal(pbn_str: str) -> dict | None:
    """Parse a PBN deal string into hands and dealer."""
    try:
        pbn_str = pbn_str.strip()
        if ':' not in pbn_str:
            return None
        
        dealer = pbn_str[0].upper()
        if dealer not in 'NESW':
            return None
        
        hands_str = pbn_str[2:].strip()
        hands = hands_str.split()
        if len(hands) != 4:
            return None
        
        dealer_order = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        directions = ['N', 'E', 'S', 'W']
        start_idx = dealer_order[dealer]
        
        result = {'Dealer': dealer, 'PBN': pbn_str}
        for i, hand in enumerate(hands):
            direction = directions[(start_idx + i) % 4]
            result[f'Hand_{direction}'] = hand
        
        return result
    except Exception:
        return None


def _compute_hand_features(hand_str: str) -> dict:
    """Compute HCP, suit lengths, and total points for a hand."""
    hcp_values = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    
    suits = hand_str.split('.')
    if len(suits) != 4:
        return {}
    
    suit_names = ['S', 'H', 'D', 'C']
    result = {}
    total_hcp = 0
    dp = 0  # Distribution points
    
    for suit_name, suit_cards in zip(suit_names, suits):
        length = len(suit_cards)
        result[f'SL_{suit_name}'] = length
        
        suit_hcp = sum(hcp_values.get(c.upper(), 0) for c in suit_cards)
        total_hcp += suit_hcp
        
        # Distribution points: void=3, singleton=2, doubleton=1
        if length == 0:
            dp += 3
        elif length == 1:
            dp += 2
        elif length == 2:
            dp += 1
    
    result['HCP'] = total_hcp
    result['Total_Points'] = total_hcp + dp
    
    return result


def _compute_par_score(pbn_str: str, dealer: str, vul: str = "None") -> dict:
    """Compute par score and contracts using endplay."""
    try:
        deal = Deal(pbn_str)
        
        dealer_map = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}
        dealer_player = dealer_map.get(dealer, Player.north)
        
        vul_map = {
            'None': Vul.none, 'Both': Vul.both, 'All': Vul.both,
            'NS': Vul.ns, 'N-S': Vul.ns,
            'EW': Vul.ew, 'E-W': Vul.ew
        }
        vul_enum = vul_map.get(vul, Vul.none)
        
        dd_table = calc_dd_table(deal)
        parlist = par(dd_table, vul_enum, dealer_player)
        
        par_score = parlist.score
        
        # endplay Denom enum: spades=0, hearts=1, diamonds=2, clubs=3, nt=4
        strain_map = {0: 'S', 1: 'H', 2: 'D', 3: 'C', 4: 'N'}
        contracts_list = []
        for contract in parlist:
            level = contract.level
            strain = strain_map.get(int(contract.denom), '?')
            declarer = contract.declarer.abbr
            penalty = contract.penalty.abbr if contract.penalty.abbr != 'U' else ''
            result = contract.result
            
            result_str = f"+{result}" if result > 0 else str(result) if result < 0 else "="
            contract_str = f"{level}{strain}{penalty} {declarer} {result_str}"
            contracts_list.append(contract_str)
        
        par_contracts = ", ".join(contracts_list) if contracts_list else "Pass"
        
        return {'Par_Score': par_score, 'Par_Contract': par_contracts}
    except Exception as e:
        return {'Par_Score': None, 'Par_Contract': f"Error: {e}"}


@app.post("/process-pbn")
def process_pbn(req: ProcessPBNRequest) -> Dict[str, Any]:
    """Process PBN deal(s) and compute features including optional par score."""
    t0 = time.perf_counter()
    
    pbn_input = req.pbn.strip()
    pbn_deals = []
    
    # Track vulnerability per deal (for PBN files)
    deal_vuls: dict[int, str] = {}
    
    # Check if it's a URL
    if pbn_input.startswith('http://') or pbn_input.startswith('https://'):
        # Convert GitHub blob URLs to raw URLs
        url = pbn_input
        if 'github.com' in url and '/blob/' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        try:
            response = http_requests.get(url, timeout=30)
            response.raise_for_status()
            pbn_content = response.text
            
            # Parse PBN file - extract deals with their vulnerabilities
            import re
            
            # Split into blocks (separated by blank lines or consecutive tags)
            current_vul = "None"
            deal_pattern = re.compile(r'\[Deal\s+"([NESW]:[^"]+)"\]')
            vul_pattern = re.compile(r'\[Vulnerable\s+"([^"]+)"\]')
            
            # Process line by line to pair vulnerabilities with deals
            for line in pbn_content.split('\n'):
                line = line.strip()
                vul_match = vul_pattern.search(line)
                if vul_match:
                    vul_str = vul_match.group(1)
                    # Normalize vulnerability
                    if vul_str.lower() in ('none', '-'):
                        current_vul = "None"
                    elif vul_str.lower() in ('all', 'both'):
                        current_vul = "Both"
                    elif vul_str.upper() in ('NS', 'N-S'):
                        current_vul = "NS"
                    elif vul_str.upper() in ('EW', 'E-W'):
                        current_vul = "EW"
                    else:
                        current_vul = "None"
                
                deal_match = deal_pattern.search(line)
                if deal_match:
                    deal_idx = len(pbn_deals)
                    pbn_deals.append(deal_match.group(1))
                    deal_vuls[deal_idx] = current_vul
            
            # Fallback if no [Deal] tags found
            if not pbn_deals:
                lines = [l.strip() for l in pbn_content.split('\n') 
                        if l.strip() and ':' in l and l[0] in 'NESW']
                pbn_deals = lines
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    else:
        # Single PBN string
        pbn_deals = [pbn_input]
    
    if not pbn_deals:
        raise HTTPException(status_code=400, detail="No valid PBN deals found")
    
    results = []
    for deal_idx, pbn_str in enumerate(pbn_deals):
        deal = _parse_pbn_deal(pbn_str)
        if not deal:
            results.append({"error": f"Invalid PBN: {pbn_str[:50]}..."})
            continue
        
        # Add features for each hand
        for direction in 'NESW':
            hand_col = f'Hand_{direction}'
            if hand_col in deal:
                features = _compute_hand_features(deal[hand_col])
                for key, value in features.items():
                    deal[f'{key}_{direction}'] = value
        
        # Compute par if requested
        if req.include_par:
            # Use vulnerability from PBN file if available, otherwise use request
            vul = deal_vuls.get(deal_idx, req.vul)
            deal['Vulnerability'] = vul
            par_info = _compute_par_score(pbn_str, deal['Dealer'], vul)
            deal.update(par_info)
        
        results.append(deal)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[process-pbn] {elapsed_ms:.1f}ms ({len(results)} deals)")
    
    return {
        "deals": results,
        "count": len(results),
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# API: find matching auctions
# ---------------------------------------------------------------------------

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
    
    # Apply filter using DuckDB
    sql_query = f"SELECT * FROM joined_df WHERE {where_clause} LIMIT {req.max_results}"
    
    try:
        matching_df = duckdb.sql(sql_query).pl()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e}")
    
    t3 = time.perf_counter()
    
    # Build result columns - Auction first, then key metrics
    result_cols = ["Auction"]
    if "matching_deal_count" in matching_df.columns:
        result_cols.append("matching_deal_count")
    
    # Add criteria columns for the matched seat
    for col in sorted(matching_df.columns):
        if f"_S{seat}" in col and col not in result_cols:
            result_cols.append(col)
    
    result_cols = [c for c in result_cols if c in matching_df.columns]
    result_df = matching_df.select(result_cols[:25])  # Limit columns
    
    result_rows = result_df.to_dicts()
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[find-matching-auctions] {elapsed_ms:.1f}ms ({len(result_rows)} matches)")
    print(f"  join: {(t2-t1)*1000:.0f}ms, filter: {(t3-t2)*1000:.0f}ms")
    
    return {
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
    args = parser.parse_args()

    print("Starting API server...")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
