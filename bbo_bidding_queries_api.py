"""
FastAPI server for BBO bidding queries.

This service loads bidding data using the helpers, then exposes
HTTP endpoints that Streamlit (or other clients) can call.

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
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

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

exec_plan_file = biddingPath.joinpath("execution_plan_data.pkl")
bbo_mldf_augmented_file = dataPath.joinpath("bbo_mldf_augmented.parquet")
bbo_bidding_table_augmented_file = biddingPath.joinpath("bbo_bidding_table_augmented.parquet")

REQUIRED_FILES = [
    exec_plan_file,
    bbo_mldf_augmented_file,
    bbo_bidding_table_augmented_file,
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


# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------


STATE: Dict[str, Any] = {
    "initialized": False,
    "initializing": False,
    "error": None,
    "deal_df": None,
    "bt_df": None,
    "deal_criteria_by_seat_dfs": None,
    "deal_criteria_by_direction_dfs": None,
    "results": None,
}

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
            unique_criteria_cols_l,
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
        del criteria_deal_dfs_directional, pythonized_exprs_by_direction, unique_criteria_cols_l
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

        with _STATE_LOCK:
            STATE["deal_df"] = deal_df
            STATE["bt_df"] = bt_df
            STATE["deal_criteria_by_seat_dfs"] = deal_criteria_by_seat_dfs
            STATE["deal_criteria_by_direction_dfs"] = deal_criteria_by_direction_dfs
            STATE["results"] = results
            STATE["initialized"] = True
            STATE["initializing"] = False
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
        except Exception as warm_exc:  # pragma: no cover - best-effort prewarm
            print("[init] WARNING: pre-warm step failed:", warm_exc)

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

    if "is_completed_auction" not in bt_df.columns:
        raise HTTPException(status_code=500, detail="Column 'is_completed_auction' not found in bt_df")

    completed_df = bt_df.filter(pl.col("is_completed_auction"))
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

    if "is_completed_auction" in bt_df.columns:
        base_df = bt_df.filter(pl.col("is_completed_auction"))
    else:
        base_df = bt_df

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

    if "is_completed_auction" in bt_df.columns:
        base_df = bt_df.filter(pl.col("is_completed_auction"))
    else:
        base_df = bt_df

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
            auction_info["deals"] = combined_df.select(
                [c for c in deal_display_cols if c in combined_df.columns]
            ).to_dicts()

        out_auctions.append(auction_info)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_deals = sum(len(a.get("deals", [])) for a in out_auctions)
    print(f"[deals-matching-auction] {elapsed_ms:.1f}ms ({len(out_auctions)} auctions, {total_deals} deals)")
    return {"pattern": pattern, "auctions": out_auctions, "elapsed_ms": round(elapsed_ms, 1)}


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
