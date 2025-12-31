"""
FastAPI server for BBO bidding queries.

This service loads bidding data (bt_seat1_df and deal_df), exposes HTTP endpoints that Streamlit (or other clients) can call.

Heavy initialization (loading Parquet files, building criteria bitmaps,
computing opening-bid candidates) is performed once in the background and the
results are kept in-process. Startup takes ~8-10 minutes.

Usage:
    python bbo_bidding_queries_api.py
"""

from __future__ import annotations

import csv
import gc
import operator
import os
import re
import signal
import sys
import threading
import time
import traceback
from functools import wraps
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, TypeVar
import pathlib

import pyarrow.parquet as pq

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
    load_deal_df,
    load_execution_plan_data,
    directional_to_directionless,
    build_or_load_directional_criteria_bitmaps,
)

# NOTE: imported as a module so static analysis doesn't get confused about symbol exports.
import mlBridgeLib.mlBridgeBiddingLib as mlBridgeBiddingLib

from bbo_bidding_queries_lib import (
    evaluate_criterion_for_hand,
)

# ---------------------------------------------------------------------------
# Hot-reloadable handlers module
# ---------------------------------------------------------------------------

import importlib

# Track mtime of plugins directory for hot-reload
_PLUGINS_DIR = pathlib.Path(__file__).parent / "plugins"
_plugins_mtime: float = 0.0
_plugins_last_reload_epoch_s: float | None = None
_plugins_lock = threading.Lock()

# Global registry of loaded plugins
PLUGINS: Dict[str, Any] = {}

def _reload_plugins() -> dict[str, object]:
    """Reload all modules in the plugins directory if any have been modified.
    
    This allows editing handler code without restarting the server.
    """
    global _plugins_mtime, _plugins_last_reload_epoch_s
    
    # Check if plugins directory exists
    if not _PLUGINS_DIR.exists():
        return {"reloaded": False, "mtime": None, "reloaded_at": None}
    
    # Check mtime of the plugins directory itself and all .py files inside
    # (max mtime of any file or the dir)
    try:
        current_mtime = _PLUGINS_DIR.stat().st_mtime
        for p in _PLUGINS_DIR.glob("*.py"):
            t = p.stat().st_mtime
            if t > current_mtime:
                current_mtime = t
    except FileNotFoundError:
        # A file might have been deleted during iteration
        return {"reloaded": False, "mtime": None, "reloaded_at": None}
    
    with _plugins_lock:
        if current_mtime > _plugins_mtime:
            _plugins_last_reload_epoch_s = time.time()
            print(f"[hot-reload] Reloading plugins (mtime {current_mtime})")
            
            # Ensure plugins dir is in path or package importable
            # Since 'plugins' is a package (has __init__.py), we can import 'plugins.module_name'
            
            # Reload all .py files in plugins/
            for p in _PLUGINS_DIR.glob("*.py"):
                if p.name == "__init__.py":
                    continue
                
                module_name = p.stem
                full_module_name = f"plugins.{module_name}"
                
                try:
                    if full_module_name in sys.modules:
                        print(f"  - Reloading {full_module_name}")
                        module = importlib.reload(sys.modules[full_module_name])
                    else:
                        print(f"  - Importing {full_module_name}")
                        module = importlib.import_module(full_module_name)
                    
                    PLUGINS[module_name] = module
                except Exception as e:
                    print(f"  ! Error loading {full_module_name}: {e}")
            
            _plugins_mtime = current_mtime
            return {
                "reloaded": True,
                "mtime": current_mtime,
                "reloaded_at": _plugins_last_reload_epoch_s,
            }

    return {
        "reloaded": False,
        "mtime": _plugins_mtime if _plugins_mtime else None,
        "reloaded_at": _plugins_last_reload_epoch_s,
    }

# Initialize plugins on startup
_reload_plugins()

# Alias for backward compatibility with existing handler calls
# This assumes bbo_bidding_queries_api_handlers is present in plugins/
if "bbo_bidding_queries_api_handlers" in PLUGINS:
    bbo_bidding_queries_api_handlers = PLUGINS["bbo_bidding_queries_api_handlers"]
else:
    print("WARNING: bbo_bidding_queries_api_handlers plugin not found!")
    # Define a dummy object to prevent startup crash if file missing, 
    # though it will crash on access
    class DummyHandler:
        def __getattr__(self, name):
            raise ImportError("bbo_bidding_queries_api_handlers plugin not loaded")
    bbo_bidding_queries_api_handlers = DummyHandler()



def _attach_hot_reload_info(resp: Dict[str, Any], reload_info: dict[str, object]) -> Dict[str, Any]:
    """Attach hot-reload info to a JSON response payload."""
    # Always include the boolean so callers can cheaply display a message.
    resp["hot_reload"] = bool(reload_info.get("reloaded", False))
    if resp["hot_reload"]:
        resp["hot_reload_mtime"] = reload_info.get("mtime")
        resp["hot_reload_at"] = reload_info.get("reloaded_at")
    return resp


def _log_and_raise(endpoint: str, e: Exception) -> NoReturn:
    """Log exception with traceback and raise HTTPException with details."""
    tb = traceback.format_exc()
    print(f"[{endpoint}] ERROR: {e}\n{tb}")
    raise HTTPException(status_code=500, detail=tb)


def _get_handler_module():
    """Get the handler module, raising ImportError if not found."""
    handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
    if not handler_module:
        raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")
    return handler_module


def _prepare_handler_call() -> Tuple[Dict[str, Any], dict, Any]:
    """Common setup for endpoint handlers.
    
    Returns:
        (state_copy, reload_info, handler_module)
    
    Raises:
        HTTPException: If service not ready
        ImportError: If handler plugin not loaded
    """
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    handler_module = _get_handler_module()
    return state, reload_info, handler_module


# ---------------------------------------------------------------------------
# Data directory resolution (supports --data-dir command line arg)
# ---------------------------------------------------------------------------

def _parse_data_dir_arg() -> pathlib.Path | None:
    """Parse --data-dir from sys.argv early (before full argparse)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            return pathlib.Path(sys.argv[i + 1])
        if arg.startswith("--data-dir="):
            return pathlib.Path(arg.split("=", 1)[1])
    return None


def _parse_deal_rows_arg() -> int | None:
    """Parse --deal-rows from sys.argv early.
    
    0 or None means 'all rows'.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--deal-rows" and i + 1 < len(sys.argv):
            val = int(sys.argv[i + 1])
            return None if val <= 0 else val
        if arg.startswith("--deal-rows="):
            val = int(arg.split("=", 1)[1])
            return None if val <= 0 else val
    return None # Default to all rows if not specified


# ---------------------------------------------------------------------------
# Optional init knobs (parsed early from sys.argv)
# ---------------------------------------------------------------------------

def _parse_prewarm_arg() -> bool:
    """Parse --prewarm / --no-prewarm from sys.argv early.
    
    Default: True (on) so endpoints are pre-warmed unless explicitly disabled.
    """
    prewarm = True
    for arg in sys.argv:
        if arg == "--prewarm":
            prewarm = True
        elif arg == "--no-prewarm":
            prewarm = False
        elif arg.startswith("--prewarm="):
            v = arg.split("=", 1)[1].strip().lower()
            prewarm = v in ("1", "true", "yes", "y", "on")
    return prewarm


# Check for --data-dir command line argument
_cli_data_dir = _parse_data_dir_arg()
# Check for --deal-rows (limit deal_df rows for faster debugging)
_cli_deal_rows = _parse_deal_rows_arg()
_cli_prewarm = _parse_prewarm_arg()
if not _cli_prewarm:
    print("[config] Pre-warming disabled (--no-prewarm)")
if _cli_deal_rows is not None:
    print(f"[config] Limiting deal_df to {_cli_deal_rows:,} rows (--deal-rows)")

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
bbo_mldf_augmented_file = dataPath.joinpath("bbo_mldf_augmented_matches.parquet")
bt_seat1_file = dataPath.joinpath("bbo_bt_seat1.parquet")  # Clean seat-1-only table
bt_augmented_file = dataPath.joinpath("bbo_bt_augmented.parquet")  # Full bidding table (all seats/prefixes)
bt_aggregates_file = dataPath.joinpath("bbo_bt_aggregate.parquet")
auction_criteria_file = dataPath.joinpath("bbo_custom_auction_criteria.csv")
# Merged rules file: learned criteria with deduplication (for Rules model)
merged_rules_file = dataPath.joinpath("bbo_bt_merged_rules.parquet")

# ---------------------------------------------------------------------------
# Constants for hand criteria
# ---------------------------------------------------------------------------

# Valid column names for hand criteria expressions (base names; direction suffix appended where applicable)
# NOTE: Keep this list aligned with what actually exists in the precomputed deal_df.
HAND_CRITERIA_COLUMNS = frozenset(['HCP', 'SL_S', 'SL_H', 'SL_D', 'SL_C', 'Total_Points'])

# Bridge directions
DIRECTIONS_LIST = ['N', 'E', 'S', 'W']

# Valid seat numbers (1-based)
VALID_SEATS = range(1, 5)

# Comparison operators and their implementations (works with Polars expressions via __ge__ etc.)
_COMPARISON_OPS = {
    '>=': operator.ge,
    '<=': operator.le,
    '>': operator.gt,
    '<': operator.lt,
    '==': operator.eq,
    '!=': operator.ne,
}

# Vulnerability enum mapping (endplay library convention)
VUL_MAP = {0: 'None', 1: 'NS', 2: 'EW', 3: 'Both'}

# DD Score column generation constants
DD_SCORE_LEVELS = range(1, 8)  # Bridge levels 1-7
DD_SCORE_STRAINS = ('C', 'D', 'H', 'S', 'N')
DD_SCORE_VULS = ('NV', 'V')

# Initialization step counts (for loading progress display)
INIT_STEPS_WITH_PREWARM = 7
INIT_STEPS_WITHOUT_PREWARM = 6


def _normalize_auction_expr(col_name: str = "Auction") -> pl.Expr:
    """Create a Polars expression that normalizes auction strings.
    
    Normalizes by: UPPERCASE (canonical), strip leading passes (P-).
    This is used consistently across all auction matching code.
    """
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace(r"^(P-)+", "")
    )


# ---------------------------------------------------------------------------
# Dynamic auction criteria filtering (from criteria.csv)
# ---------------------------------------------------------------------------


def _strip_inline_comment(s: str) -> str:
    """Strip inline comments from a string (everything after #)."""
    pos = s.find('#')
    if pos != -1:
        s = s[:pos]
    return s.strip()


def _load_auction_criteria() -> list[tuple[str, list[str]]]:
    """Load criteria.csv and return list of (partial_auction, [criteria...]).
    
    CSV format: partial_auction,criterion1,criterion2,...
    Example row: 1c,SL_S >= SL_H,HCP >= 12
    
    Comments:
    - Lines starting with # are skipped entirely
    - Inline comments (text after #) are stripped from each cell
    
    Complex expressions with &, |, parentheses are supported:
    - (SL_D > SL_H | SL_D > SL_S) - OR logic
    - SL_D >= SL_C & SL_D > SL_H - AND logic
    - Parentheses for grouping
    """
    if not auction_criteria_file.exists():
        return []
    
    criteria_list = []
    try:
        with open(auction_criteria_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Skip lines where first cell is a comment
                first_cell = row[0].strip()
                if first_cell.startswith('#'):
                    continue
                # Strip inline comments from first cell (partial auction) - canonical UPPERCASE
                partial_auction = _strip_inline_comment(first_cell).upper()
                if not partial_auction:
                    continue
                # Strip inline comments from each criterion
                criteria = []
                for c in row[1:]:
                    c_clean = _strip_inline_comment(c)
                    if c_clean:
                        criteria.append(c_clean)
                if criteria:
                    criteria_list.append((partial_auction, criteria))
    except Exception as e:
        print(f"[auction-criteria] Error loading {auction_criteria_file}: {e}")
    
    return criteria_list


def _normalize_criterion_string(s: str) -> str:
    """Normalize a criterion expression for matching against bitmap column names.
    
    The execution-plan expressions sometimes differ only by whitespace around operators.
    We normalize by stripping and removing whitespace around comparison operators.
    """
    s = (s or "").strip()
    # Remove whitespace around comparison operators
    s = re.sub(r"\s*(>=|<=|==|!=|>|<)\s*", r"\1", s)
    # Collapse internal whitespace (e.g. between tokens like 'and', though uncommon here)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_custom_criteria_overlay(available_criteria_names: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an in-memory overlay from bbo_custom_auction_criteria.csv.
    
    Overlay row format:
        {"partial": str, "seat": int, "criteria": list[str]}
    
    Notes:
    - `partial` is expected to be in seat-1 view (no leading 'p-').
    - We compute `seat` as (num_dashes + 1), consistent with existing logic.
    - This overlay is applied *on-the-fly* in handlers; we do NOT mutate bt_seat1_df.
    """
    criteria_list = _load_auction_criteria()
    overlay: list[dict[str, Any]] = []
    unknown_criteria: list[str] = []
    for partial_auction, criteria in criteria_list:
        seat = partial_auction.count("-") + 1
        if seat not in VALID_SEATS:
            continue
        normalized_criteria: list[str] = []
        for c in criteria:
            c0 = (c or "").strip()
            if not c0:
                continue
            if available_criteria_names is None:
                # No reference set available; keep as-is
                normalized_criteria.append(c0)
                continue
            if c0 in available_criteria_names:
                normalized_criteria.append(c0)
                continue
            c_norm = _normalize_criterion_string(c0)
            if c_norm in available_criteria_names:
                normalized_criteria.append(c_norm)
                continue
            # Keep original but track as unknown (will likely be ignored by bitmap checks)
            normalized_criteria.append(c0)
            unknown_criteria.append(c0)

        overlay.append(
            {"partial": partial_auction.strip().upper(), "seat": int(seat), "criteria": normalized_criteria}  # Canonical UPPERCASE
        )

    # Provide a UI-friendly "rules" list for `/custom-criteria-info`.
    # (Older UI expects stats["rules"] and stats["rules_applied"].)
    rules_for_ui = [
        {"partial": r["partial"], "seat": r["seat"], "criteria": r["criteria"]}
        for r in overlay
    ]

    stats: dict[str, Any] = {
        "criteria_file": str(auction_criteria_file),
        "criteria_file_exists": auction_criteria_file.exists(),
        "rule_count": len(overlay),
        "rules_applied": len(overlay),  # backward compatibility with older UI wording
        "overlay_enabled": True,
        "unknown_criteria_count": len(set(unknown_criteria)),
        "rules": rules_for_ui,
    }
    return overlay, stats


def _get_direction_for_partial_auction(partial_auction: str, dealer: str) -> str:
    """Get the direction (N/E/S/W) for the bidder of the last bid in partial_auction.
    
    The seat number = number of dashes + 1
    Direction = DIRECTIONS_LIST[(dealer_index + seat - 1) % 4]
    """
    dealer_upper = dealer.upper()
    dealer_idx = DIRECTIONS_LIST.index(dealer_upper) if dealer_upper in DIRECTIONS_LIST else 0
    num_dashes = partial_auction.count('-')
    seat = num_dashes + 1  # 1-based seat number
    direction_idx = (dealer_idx + seat - 1) % 4
    return DIRECTIONS_LIST[direction_idx]


def _parse_criterion_to_polars(criterion: str, direction: str) -> pl.Expr | None:
    """Parse a criterion string like 'SL_S >= SL_H' into a Polars expression.
    
    Supports:
    - Column comparisons: SL_S >= SL_H, HCP > Total_Points
    - Value comparisons: HCP >= 12, SL_S <= 5
    - Operators: >=, <=, >, <, ==, !=
    """
    # Match: COL OP COL or COL OP VALUE (allow floats like 12.5)
    pattern = r'(\w+)\s*(>=|<=|>|<|==|!=)\s*(\w+|\d+\.?\d*)'
    match = re.match(pattern, criterion.strip())
    if match is None:
        print(f"[auction-criteria] Could not parse criterion: {criterion}")
        return None
    
    left, op, right = match.groups()
    
    if op not in _COMPARISON_OPS:
        print(f"[auction-criteria] Unknown operator '{op}' in criterion: {criterion}")
        return None
    
    # Append direction to column names if they're bridge columns
    left_col = f"{left}_{direction}" if left in HAND_CRITERIA_COLUMNS else left
    left_expr = pl.col(left_col)
    
    # Check if right is a number or column name
    try:
        right_val = float(right)
        return _COMPARISON_OPS[op](left_expr, right_val)
    except ValueError:
        # Compare against another column
        right_col = f"{right}_{direction}" if right in HAND_CRITERIA_COLUMNS else right
        right_expr = pl.col(right_col)
        return _COMPARISON_OPS[op](left_expr, right_expr)


def _build_criteria_expressions(
    criteria: list[str], 
    direction: str
) -> list[tuple[str, pl.Expr]]:
    """Parse a list of criterion strings into (name, expression) pairs."""
    result = []
    for criterion in criteria:
        expr = _parse_criterion_to_polars(criterion, direction)
        if expr is not None:
            result.append((criterion, expr))
    return result


def _combine_criteria_expressions(exprs: list[tuple[str, pl.Expr]]) -> pl.Expr:
    """Combine multiple criteria expressions with AND logic."""
    combined = exprs[0][1]
    for _, expr in exprs[1:]:
        combined = combined & expr
    return combined


def _check_row_criteria(
    row_df: pl.DataFrame,
    criteria_exprs: list[tuple[str, pl.Expr]]
) -> list[str]:
    """Check which criteria fail for a single row. Returns list of failed criterion names."""
    failed = []
    for criterion_str, criterion_expr in criteria_exprs:
        try:
            passes = row_df.filter(criterion_expr).height > 0
            if not passes:
                failed.append(criterion_str)
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.ComputeError) as e:
            failed.append(f"{criterion_str} (error: {type(e).__name__})")
    return failed


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
    if len(criteria_list) == 0:
        return df, None
    
    if auction_col not in df.columns or dealer_col not in df.columns:
        return df, None
    
    rejected_rows: list[dict] = []
    
    for partial_auction, criteria in criteria_list:
        # Use consistent auction normalization
        matches_partial = _normalize_auction_expr(auction_col).str.starts_with(partial_auction)
        num_dashes = partial_auction.count('-')
        seat = num_dashes + 1
            
        for dealer in DIRECTIONS_LIST:
            direction = _get_direction_for_partial_auction(partial_auction, dealer)
            criteria_exprs = _build_criteria_expressions(criteria, direction)
            
            if len(criteria_exprs) == 0:
                continue
            
            matching_mask = matches_partial & (pl.col(dealer_col) == dealer)
            matching_rows = df.filter(matching_mask)
                
            # Track rejected rows if requested
            if track_rejected and matching_rows.height > 0:
                for row_idx in range(matching_rows.height):
                    row = matching_rows.row(row_idx, named=True)
                    single_row_df = matching_rows.slice(row_idx, 1)
                    failed_criteria = _check_row_criteria(single_row_df, criteria_exprs)
                        
                    if len(failed_criteria) > 0:
                        rejected_rows.append({
                            'Auction': str(row.get(auction_col, '')),
                            'Partial_Auction': str(partial_auction),
                            'Failed_Criteria': ', '.join(failed_criteria),
                            'Dealer': str(dealer),
                            'Seat': int(seat),
                            'Direction': str(direction),
                        })
                
            # Apply filter: keep rows that don't match OR satisfy criteria
            combined_criteria = _combine_criteria_expressions(criteria_exprs)
            try:
                df = df.filter(~matching_mask | combined_criteria)
            except (pl.exceptions.ColumnNotFoundError, pl.exceptions.ComputeError) as e:
                # Defensive: a criterion may reference a column not present in df.
                # Skip applying this criterion set rather than crashing the endpoint.
                print(
                    "[auction-criteria] WARNING: skipping criteria due to filter error: "
                    f"partial={partial_auction}, dealer={dealer}, err={type(e).__name__}: {e}"
                )
                continue
    
    rejected_df = pl.DataFrame(rejected_rows) if track_rejected and len(rejected_rows) > 0 else None
    return df, rejected_df


def _check_required_files() -> list[str]:
    """Check that required data files exist. Returns list of missing files.
    
    NOTE:
    - `bbo_bt_seat1.parquet` is a REQUIRED pipeline artifact for this API. If it's missing,
      something is wrong upstream and we hard-fail rather than falling back.
    - `bbo_bt_criteria_seat1_df.parquet` is now also REQUIRED at runtime; it provides the
      completed-auction criteria/aggregate stats (`bt_stats_df`) used by several endpoints
      (Bidding Table Explorer, Find Matching Auctions, etc.). If it's missing, we fail fast
      instead of running with partially functional endpoints.
    - We intentionally do NOT load the full `bbo_bt_augmented.parquet` into memory; all heavy
      preprocessing should be done offline by `bbo_bt_build_seat1.py` and friends.
    """
    required: list[pathlib.Path] = [
        exec_plan_file,
        bbo_mldf_augmented_file,
        bt_seat1_file,
        bt_criteria_seat1_file,
    ]

    missing: list[str] = []
    for f in required:
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
    seed: Optional[int] = 0


class RandomAuctionSequencesRequest(BaseModel):
    n_samples: int = 5
    # seed=0 means non-reproducible, any other value is reproducible
    seed: Optional[int] = 0


class AuctionSequencesMatchingRequest(BaseModel):
    pattern: str
    n_samples: int = 5
    seed: Optional[int] = 0
    # If True, treat the pattern as applying to all seats (ignore leading initial passes in stored auctions).
    # If False, treat the pattern as a literal regex against the raw Auction string.
    allow_initial_passes: bool = True


class AuctionSequencesByIndexRequest(BaseModel):
    """Request for fetching auction sequences by bt_index values."""
    indices: List[int]
    # If True, expand each result into 4 dealer positions (p- prefix variants), matching Find Auction Sequences behavior.
    allow_initial_passes: bool = True


class DealCriteriaCheck(BaseModel):
    seat: int
    criteria: List[str] = []


class DealCriteriaEvalBatchRequest(BaseModel):
    """Request to evaluate criteria for a given deal row index (batch).

    Returns per-seat: passed, failed, untracked.
    """
    deal_row_idx: int
    dealer: str  # N/E/S/W
    checks: List[DealCriteriaCheck]


class DealsMatchingAuctionRequest(BaseModel):
    pattern: str
    n_auction_samples: int = 2
    n_deal_samples: int = 10
    seed: Optional[int] = 0
    allow_initial_passes: bool = True
    # Distribution filter for deals
    dist_pattern: Optional[str] = None  # Ordered distribution (S-H-D-C), e.g., "5-4-3-1"
    sorted_shape: Optional[str] = None  # Sorted shape (any suit), e.g., "5431"
    dist_direction: str = "N"  # Which hand to filter (N/E/S/W)
    # Wrong bid filter: "all" (default), "no_wrong" (only conforming bids), "only_wrong" (only non-conforming)
    wrong_bid_filter: str = "all"


class BiddingTableStatisticsRequest(BaseModel):
    auction_pattern: str = "^1N-p-3N$"
    sample_size: int = 100
    min_matches: int = 0  # 0 = no minimum
    seed: Optional[int] = 0
    allow_initial_passes: bool = True
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


class BTSeatStatsRequest(BaseModel):
    """Request for on-the-fly seat stats for a single bt_seat1 row."""
    bt_index: int  # bt_seat1.bt_index
    # seat: 1-4 for specific seat, 0 for all seats
    seat: int = 0
    # Optional cap on number of deals to aggregate (0 = all); currently unused.
    max_deals: Optional[int] = 0


class ExecuteSQLRequest(BaseModel):
    """Request for executing user SQL queries."""
    sql: str
    max_rows: int = 10000


class WrongBidStatsRequest(BaseModel):
    """Request for aggregate wrong bid statistics."""
    auction_pattern: Optional[str] = None  # Optional filter pattern
    seat: Optional[int] = None  # Filter by specific seat (1-4), None = all seats


class FailedCriteriaSummaryRequest(BaseModel):
    """Request for criterion failure analysis."""
    auction_pattern: Optional[str] = None  # Optional filter pattern
    top_n: int = 20  # Number of top failing criteria to return
    seat: Optional[int] = None  # Filter by specific seat (1-4), None = all seats


class WrongBidLeaderboardRequest(BaseModel):
    """Request for wrong bid leaderboard."""
    top_n: int = 20  # Number of bids to return
    seat: Optional[int] = None  # Filter by specific seat (1-4), None = all seats


class CustomCriteriaRule(BaseModel):
    """A single custom criteria rule."""
    partial_auction: str  # e.g., "1c", "1n-p-3n"
    criteria: List[str]  # e.g., ["HCP >= 12", "SL_C >= 3"]


class CustomCriteriaSaveRequest(BaseModel):
    """Request to save all custom criteria rules."""
    rules: List[CustomCriteriaRule]


class CustomCriteriaValidateRequest(BaseModel):
    """Request to validate a criteria expression."""
    expression: str  # e.g., "HCP >= 12"


class CustomCriteriaPreviewRequest(BaseModel):
    """Request to preview impact of a rule."""
    partial_auction: str


class BiddingArenaRequest(BaseModel):
    """Request for Bidding Arena: head-to-head model comparison."""
    model_a: str = "Rules"  # First model to compare
    model_b: str = "Actual"  # Second model to compare
    auction_pattern: Optional[str] = None  # Optional filter pattern
    sample_size: int = 1000  # Number of deals to compare
    seed: Optional[int] = 0  # Random seed for sampling
    # Custom deals source: file path or URL to parquet/csv file
    # If None, uses the default deal_df loaded at startup
    deals_uri: Optional[str] = None
    # Optional pinned deal indexes to force-include in the sample (shown in Sample Deal Comparisons)
    deal_indices: Optional[List[int]] = None
    # If True, the Rules model will search all completed BT rows (can be extremely slow).
    # Only relevant when the deal rows do not contain precomputed Matched_BT_Indices.
    search_all_bt_rows: bool = False


class AuctionDDAnalysisRequest(BaseModel):
    """Request for Auction DD Analysis: get DD columns for deals matching an auction's criteria."""
    auction: str  # Partial or complete auction string (e.g., "1N-p", "1N-p-3N-p-p-p")
    max_deals: int = 1000  # Maximum number of matching deals to return
    seed: Optional[int] = 0  # Random seed for sampling (0 = non-reproducible)
    vul_filter: Optional[str] = None  # Filter by vulnerability: None, Both, NS, EW (None = all)
    include_hands: bool = True  # Include Hand_N/E/S/W columns in output
    include_scores: bool = True  # Include DD_Score columns in addition to DD tricks


class RankBidsByEVRequest(BaseModel):
    """Request for Rank Next Bids by EV: rank all possible next bids after an auction by EV."""
    auction: str = ""  # Auction prefix (empty = opening bids)
    max_deals: int = 500  # Max deals to sample for EV calculation
    seed: Optional[int] = 0  # Random seed for sampling (0 = non-reproducible)
    vul_filter: Optional[str] = None  # Filter by vulnerability: None, Both, NS, EW (None = all)
    include_hands: bool = True  # Include Hand_N/E/S/W columns in output
    include_scores: bool = True  # Include DD_Score columns in deal data


class ContractEVDealsRequest(BaseModel):
    """Request for deals matching a selected next bid + contract EV row."""
    auction: str = ""          # Auction prefix (same as Rank Next Bids by EV)
    next_bid: str              # Bid from Next Bid Rankings (e.g. "4N")
    contract: str              # Contract string (e.g. "4H", "3N")
    declarer: str              # N/E/S/W
    seat: Optional[int] = None # Optional seat number 1-4 (relative to Dealer). If provided, server computes per-deal declarer direction from seat.
    vul: str                   # NV or V
    max_deals: int = 500
    seed: Optional[int] = 0
    include_hands: bool = True

# Import model registry for Bidding Arena
from bbo_bidding_models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------


# NOTE: Handlers receive a shallow copy of STATE via `dict(STATE)`.
# This is safe because:
#   - Polars DataFrames are immutable (operations return new DataFrames)
#   - Handlers should NOT mutate nested dicts/lists in state
# If mutation is needed, use deepcopy or redesign the data flow.
STATE: Dict[str, Any] = {
    "initialized": False,
    "initializing": False,
    "warming": False,  # True while pre-warming endpoints
    "error": None,
    "loading_step": None,  # Current loading step description
    "loaded_files": {},  # File name -> row count (updated as files load)
    "deal_df": None,
    "bt_seat1_df": None,  # Clean seat-1-only table (bbo_bt_seat1.parquet)
    "bt_openings_df": None,  # Tiny opening-bid lookup table (built from bt_seat1_df)
    "deal_criteria_by_seat_dfs": None,
    "deal_criteria_by_direction_dfs": None,
    "results": None,
    # Criteria / aggregate statistics for completed auctions (seat-1 view).
    # Built from bbo_bt_criteria.parquet + bbo_bt_aggregate.parquet and keyed by bt_index.
    "bt_stats_df": None,
    "duckdb_conn": None,
    # Hot-reloadable overlay rules loaded from bbo_custom_auction_criteria.csv.
    # These rules are applied on-the-fly to BT rows when serving responses and when building criteria masks.
    "custom_criteria_overlay": [],
    # Merged rules lookup: bt_index -> Merged_Rules list (for Rules model in Bidding Arena)
    # Loaded from bbo_bt_merged_rules.parquet (optional - if missing, Rules model is disabled)
    "merged_rules_lookup": None,
    "custom_criteria_stats": {},
    # Set of available criterion names (from deal_criteria_by_direction_dfs) for normalizing CSV criteria strings.
    "available_criteria_names": None,
}

# Additional optional data file paths
bt_criteria_file = dataPath.joinpath("bbo_bt_criteria.parquet")
# Pre-joined completed-auction criteria/aggregate table (preferred at runtime).
bt_criteria_seat1_file = dataPath.joinpath("bbo_bt_criteria_seat1_df.parquet")

_STATE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# DuckDB Connection for SQL queries
# ---------------------------------------------------------------------------

# Global DuckDB connection - will be initialized after DataFrames are loaded
DUCKDB_CONN: Optional[duckdb.DuckDBPyConnection] = None
_DUCKDB_LOCK = threading.Lock()


def _register_duckdb_tables(
    deal_df: pl.DataFrame,
    bt_seat1_df: pl.DataFrame,
    bt_stats_df: Optional[pl.DataFrame] = None,
) -> None:
    """Register DataFrames with DuckDB for SQL queries."""
    global DUCKDB_CONN
    with _DUCKDB_LOCK:
        if DUCKDB_CONN is None:
            DUCKDB_CONN = duckdb.connect()
        
        # Register Polars DataFrames directly with DuckDB (Zero-copy)
        # DuckDB can see Polars objects in the local scope if we use the same name,
        # or we can use the register() method.
        DUCKDB_CONN.register("deals", deal_df)
        DUCKDB_CONN.register("auctions", bt_seat1_df)
        
        if bt_stats_df is not None:
            DUCKDB_CONN.register("auction_stats", bt_stats_df)
            print(
                "[duckdb] Registered Polars DataFrames (Zero-copy): "
                f"deals ({deal_df.height:,} rows), "
                f"auctions ({bt_seat1_df.height:,} rows), "
                f"auction_stats ({bt_stats_df.height:,} rows)"
            )
        else:
            print(
                "[duckdb] Registered Polars DataFrames (Zero-copy): "
                f"deals ({deal_df.height:,} rows), "
                f"auctions ({bt_seat1_df.height:,} rows)"
            )


# ---------------------------------------------------------------------------
# Heavy initialization helpers
# ---------------------------------------------------------------------------


def _build_additional_deal_columns() -> List[str]:
    """Build list of additional columns needed for PBN Lookup and DD/EV computation."""
    additional_cols = [
        'PBN', 'Vul', 'Declarer', 'bid', 'Contract', 'Result', 'Tricks', 'Score',
        'ParScore', 'DD_Score_Declarer', 'EV_Score_Declarer', 'ParContracts'
    ]
    
    # Add DD_Score columns for all contracts (for DD_Score_AI computation)
    # Format: DD_Score_{level}{strain}_{direction} e.g. DD_Score_3N_N
    for level in DD_SCORE_LEVELS:
        for strain in DD_SCORE_STRAINS:
            for direction in DIRECTIONS_LIST:
                additional_cols.append(f"DD_Score_{level}{strain}_{direction}")
    
    # Add EV columns for all contracts (for EV_AI computation)
    # Format: EV_{pair}_{declarer}_{strain}_{level}_{vul}
    for pair in ['NS', 'EW']:
        declarers = ['N', 'S'] if pair == 'NS' else ['E', 'W']
        for declarer in declarers:
            for strain in DD_SCORE_STRAINS:
                for level in DD_SCORE_LEVELS:
                    for vul in DD_SCORE_VULS:
                        additional_cols.append(f"EV_{pair}_{declarer}_{strain}_{level}_{vul}")
    
    return additional_cols


def _load_bt_seat1_df() -> pl.DataFrame:
    """Load bt_seat1_df with only the columns needed at runtime."""
    print(f"[init] Loading bt_seat1_df from {bt_seat1_file} (selective columns)...")
    bt_seat1_scan = pl.scan_parquet(bt_seat1_file)
    available_cols = bt_seat1_scan.collect_schema().names()
    
    # Hard fail if 'bt_index' is missing
    if "bt_index" not in available_cols:
        raise ValueError(f"REQUIRED column 'bt_index' missing from {bt_seat1_file}. Pipeline error.")
    
    # Columns required for API operations
    required_cols = [
        "bt_index", "Auction", "is_opening_bid", "is_completed_auction",
        "seat", "candidate_bid", "npasses", "auction_len", "Expr",
        "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
        "previous_bid_indices", "next_bid_indices",
        "matching_deal_count",
    ]
    cols_to_load = [c for c in required_cols if c in available_cols]
    print(f"[init] Loading {len(cols_to_load)} of {len(available_cols)} columns...")
    
    return bt_seat1_scan.select(cols_to_load).collect()


def _load_merged_rules() -> Dict[int, List[str]] | None:
    """Load merged rules for Rules model (optional).
    
    IMPORTANT: We key this lookup by `bt_index` (int) to avoid fragile auction-string
    normalization/matching issues.
    """
    if not merged_rules_file.exists():
        print(f"[init] Merged rules file not found: {merged_rules_file}")
        print("[init] Rules model will be disabled in Bidding Arena (only Raw_Rules available)")
        return None
    
    try:
        print(f"[init] Loading merged rules from {merged_rules_file}...")
        merged_rules_df = pl.read_parquet(merged_rules_file)
        if "bt_index" not in merged_rules_df.columns:
            raise ValueError("Merged rules parquet missing required column: bt_index")
        if "Merged_Rules" not in merged_rules_df.columns:
            raise ValueError("Merged rules parquet missing required column: Merged_Rules")

        lookup: dict[int, list[str]] = {}
        for bt_index, merged_rules in (
            merged_rules_df
            .select(["bt_index", "Merged_Rules"])
            .drop_nulls(["bt_index", "Merged_Rules"])
            .iter_rows()
        ):
            try:
                bt_idx_i = int(bt_index)
            except Exception:
                continue
            if not merged_rules:
                continue
            lookup[bt_idx_i] = list(merged_rules)

        print(f"[init] Loaded {len(lookup):,} merged rules entries (Rules model enabled)")
        return lookup
    except Exception as e:
        print(f"[init] WARNING: Failed to load merged rules from {merged_rules_file}: {e}")
        print("[init] Rules model will be disabled in Bidding Arena (only Raw_Rules available)")
        return None


def _prewarm_all_endpoints(bt_seat1_df: pl.DataFrame) -> None:
    """Pre-warm all endpoints to speed up first user request."""
    
    def _prewarm_endpoint(name: str, fn, *args, **kwargs):
        """Run a prewarm call and log its duration."""
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            elapsed_s = time.perf_counter() - t0
            print(f"[init] Pre-warmed {name}: {elapsed_s:.2f}s")
            return result
        except Exception as e:
            elapsed_s = time.perf_counter() - t0
            print(f"[init] Pre-warm {name} FAILED ({elapsed_s:.2f}s): {e}")
            return None
    
    prewarm_t0 = time.perf_counter()
    
    _prewarm_endpoint("openings-by-deal-index",
        openings_by_deal_index, OpeningsByDealIndexRequest(sample_size=1))

    _prewarm_endpoint("random-auction-sequences",
        random_auction_sequences, RandomAuctionSequencesRequest(n_samples=1, seed=42))

    _prewarm_endpoint("auction-sequences-matching",
        auction_sequences_matching,
        AuctionSequencesMatchingRequest(pattern="^1N-p-3N$", n_samples=1, seed=0))

    _prewarm_endpoint("deals-matching-auction",
        deals_matching_auction,
        DealsMatchingAuctionRequest(pattern="^1N-p-3N$", n_auction_samples=1, n_deal_samples=3, seed=0))

    _prewarm_endpoint("bidding-table-statistics",
        bidding_table_statistics,
        BiddingTableStatisticsRequest(auction_pattern="^1N-p-3N$", sample_size=1, seed=42))

    _prewarm_endpoint("process-pbn",
        process_pbn,
        ProcessPBNRequest(
            pbn="N:AKQ2.KQ2.AK2.AK2 T987.987.987.987 J654.654.654.654 3.JT53.QJT53.QJT5",
            include_par=True, vul="None"))

    _prewarm_endpoint("find-matching-auctions",
        find_matching_auctions,
        FindMatchingAuctionsRequest(hcp=15, sl_s=4, sl_h=3, sl_d=3, sl_c=3, total_points=17, seat=1, max_results=1))

    _prewarm_endpoint("group-by-bid",
        group_by_bid,
        GroupByBidRequest(auction_pattern="^1N-p-3N$", n_auction_groups=1, n_deals_per_group=1, seed=42))

    sample_resp = _prewarm_endpoint("pbn-sample", get_pbn_sample)
    _prewarm_endpoint("pbn-random", get_pbn_random)

    sample_pbn = getattr(sample_resp, "pbn", None) or (sample_resp.get("pbn", "") if sample_resp else "")
    if sample_pbn:
        _prewarm_endpoint("pbn-lookup", pbn_lookup, PBNLookupRequest(pbn=sample_pbn, max_results=1))

    _prewarm_endpoint("execute-sql", execute_sql, ExecuteSQLRequest(sql="SELECT 1 AS x", max_rows=1))

    # Pre-warm bt-seat-stats endpoint
    first_bt_index = None
    if bt_seat1_df.height > 0:
        try:
            first_bt_index = int(bt_seat1_df.select("bt_index").head(1).item())
        except Exception:
            pass
    if first_bt_index is not None:
        _prewarm_endpoint("bt-seat-stats", bt_seat_stats, BTSeatStatsRequest(bt_index=first_bt_index, seat=0, max_deals=0))

    _prewarm_endpoint("wrong-bid-stats", wrong_bid_stats, WrongBidStatsRequest(auction_pattern=None, seat=None))
    _prewarm_endpoint("failed-criteria-summary", failed_criteria_summary, FailedCriteriaSummaryRequest(auction_pattern=None, top_n=5, seat=None))
    _prewarm_endpoint("wrong-bid-leaderboard", wrong_bid_leaderboard, WrongBidLeaderboardRequest(top_n=5, seat=None))
    _prewarm_endpoint("bidding-models", list_bidding_models)
    _prewarm_endpoint("bidding-arena", bidding_arena, BiddingArenaRequest(model_a="Rules", model_b="Actual", sample_size=10, seed=42))
    _prewarm_endpoint("rank-bids-by-ev", rank_bids_by_ev, RankBidsByEVRequest(auction="", max_deals=10, seed=42))

    total_prewarm_s = time.perf_counter() - prewarm_t0
    print(f"[init] All endpoints pre-warmed in {total_prewarm_s:.2f}s")


# ---------------------------------------------------------------------------
# Heavy initialization (main)
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
    
    # Keep this in sync with the _update_loading_status() calls below.
    total_steps = INIT_STEPS_WITH_PREWARM if _cli_prewarm else INIT_STEPS_WITHOUT_PREWARM
    
    def _update_loading_status(step_num: int, step: str, file_name: str | None = None, row_count: int | str | None = None):
        """Update loading progress in STATE (thread-safe)."""
        with _STATE_LOCK:
            STATE["loading_step"] = f"[{step_num}/{total_steps}] {step}"
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

        # Add additional columns needed for PBN Lookup and DD/EV computation
        additional_cols = _build_additional_deal_columns()
        for col in additional_cols:
            if col not in valid_deal_columns:
                valid_deal_columns.append(col)

        # Load deals (optionally limited by --deal-rows for faster debugging)
        n_rows_msg = f" (limited to {_cli_deal_rows:,} rows)" if _cli_deal_rows else ""
        _update_loading_status(2, f"Loading deal_df ({bbo_mldf_augmented_file.name}){n_rows_msg}...")
        deal_df = load_deal_df(bbo_mldf_augmented_file, valid_deal_columns, mldf_n_rows=_cli_deal_rows)
        
        # Get total row count for display (when using --deal-rows)
        if _cli_deal_rows:
            total_rows = pq.read_metadata(bbo_mldf_augmented_file).num_rows
            row_info = f"{deal_df.height:,} of {total_rows:,}"
        else:
            row_info = f"{deal_df.height:,}"
        _update_loading_status(3, "Building criteria bitmaps...", "deal_df", row_info)
        _log_memory("after load_deal_df")

        # todo: do this earlier in the pipeline?
        # Convert 'bid' from pl.List(pl.Utf8) to pl.Utf8 by joining with '-'
        bid_dtype = deal_df.schema.get("bid")
        if bid_dtype == pl.List(pl.Utf8):
            deal_df = deal_df.with_columns(pl.col('bid').list.join('-'))
        
        # Load criteria bitmaps (must be pre-built by pipeline)
        # Bitmap file is named based on deal file: {stem}_criteria_bitmaps.parquet
        bitmap_file = bbo_mldf_augmented_file.parent / f"{bbo_mldf_augmented_file.stem.replace('_matches', '')}_criteria_bitmaps.parquet"
        if not bitmap_file.exists():
            raise FileNotFoundError(
                f"Criteria bitmaps file not found: {bitmap_file}\n"
                "Run the pipeline first: python bbo_bt_deal_matches.py"
            )
        
        _update_loading_status(3, f"Loading criteria bitmaps ({bitmap_file.name})...")
        criteria_deal_dfs_directional = build_or_load_directional_criteria_bitmaps(
            deal_df,
            pythonized_exprs_by_direction,
            expr_map_by_direction,
            deal_file=bbo_mldf_augmented_file,
            exec_plan_file=exec_plan_file,
        )
        _log_memory("after load_directional_criteria_bitmaps")

        _update_loading_status(4, "Processing criteria views...")
        deal_criteria_by_direction_dfs, deal_criteria_by_seat_dfs = directional_to_directionless(
            criteria_deal_dfs_directional, expr_map_by_direction
        )
        _log_memory("after directional_to_directionless")

        # Capture the canonical set of criterion names (as used by the bitmap DataFrames).
        # These are "original expressions" (directionless) after directional_to_directionless renaming.
        # We'll use this set to normalize CSV criteria strings on reload (mainly whitespace differences).
        try:
            ref_dir = "N" if "N" in deal_criteria_by_direction_dfs else next(iter(deal_criteria_by_direction_dfs.keys()))
            available_criteria_names = set(deal_criteria_by_direction_dfs[ref_dir].columns)
        except Exception:
            available_criteria_names = None

        # We no longer need these large helper objects
        del criteria_deal_dfs_directional, pythonized_exprs_by_direction, directionless_criteria_cols
        gc.collect()
        _log_memory("after gc.collect (criteria cleanup)")

        # Load clean seat-1-only table (used for pattern matching - no p- prefix issues)
        bt_seat1_df = _load_bt_seat1_df()
        
        # IMPORTANT:
        # Do NOT strip leading 'p-' prefixes here.
        # Those prefixes encode seat/turn order and are required for correct declarer/contract logic
        # (AI/DD/IMP computations). Matching code strips prefixes at query time instead.
        _update_loading_status(5, "Loading bt_seat1_df...", "bt_seat1_df", bt_seat1_df.height)
        print(f"[init] bt_seat1_df: {bt_seat1_df.height:,} rows (clean seat-1 data)")
        
        # Load hot-reloadable criteria overlay (does NOT mutate bt_seat1_df)
        overlay, custom_criteria_stats = _build_custom_criteria_overlay(available_criteria_names)
        _log_memory("after load custom criteria overlay")

        # Load completed-auction stats table (criteria + aggregates) keyed by bt_index.
        # This file is REQUIRED for the Bidding Table Explorer and other tools.
        print(f"[init] Loading bt_stats_df from {bt_criteria_seat1_file}...")
        if not bt_criteria_seat1_file.exists():
            raise FileNotFoundError(f"REQUIRED stats file missing: {bt_criteria_seat1_file}. Pipeline error.")
        
        try:
            bt_stats_df = pl.read_parquet(bt_criteria_seat1_file)
            print(f"[init] bt_stats_df: {bt_stats_df.height:,} rows (completed auctions with criteria/aggregates)")
            # Track in loaded_files so the UI "Files loaded" list includes stats.
            _update_loading_status(5, "Loading bt_seat1_df and bt_stats_df...", "bt_stats_df", bt_stats_df.height)
        except Exception as e:
            print(f"[init] ERROR: Failed to load bt_stats_df from {bt_criteria_seat1_file}: {e}")
            raise
        _log_memory("after load bt_stats_df")

        # Load merged rules for Rules model (optional - not required for startup)
        merged_rules_lookup = _load_merged_rules()
        if merged_rules_lookup:
            _update_loading_status(5, "Loading merged rules...", "merged_rules", len(merged_rules_lookup))
        _log_memory("after load merged_rules")

        # Compute opening-bid candidates for all (dealer, seat) combinations
        _update_loading_status(6, "Processing opening bids (seat1-only)...", "bt_seat1_df", bt_seat1_df.height)
        results, bt_openings_df = mlBridgeBiddingLib.process_opening_bids_from_bt_seat1(
            deal_df=deal_df,
            bt_seat1_df=bt_seat1_df,
            deal_criteria_by_seat_dfs=deal_criteria_by_seat_dfs,
        )
        _update_loading_status(6, "Preparing to serve..." if not _cli_prewarm else "Preparing to serve (pre-warm next)...")
        _log_memory("after process_opening_bids")

        with _STATE_LOCK:
            STATE["deal_df"] = deal_df
            STATE["bt_seat1_df"] = bt_seat1_df
            STATE["bt_openings_df"] = bt_openings_df
            STATE["deal_criteria_by_seat_dfs"] = deal_criteria_by_seat_dfs
            STATE["deal_criteria_by_direction_dfs"] = deal_criteria_by_direction_dfs
            STATE["results"] = results
            STATE["bt_stats_df"] = bt_stats_df
            STATE["custom_criteria_overlay"] = overlay
            STATE["custom_criteria_stats"] = custom_criteria_stats
            STATE["available_criteria_names"] = available_criteria_names
            STATE["merged_rules_lookup"] = merged_rules_lookup
            STATE["initialized"] = True  # Required for _ensure_ready() in pre-warming
            STATE["warming"] = bool(_cli_prewarm)  # Only true if we will actually pre-warm
            STATE["error"] = None
        _log_memory("after STATE update")

        # Register DataFrames with DuckDB for SQL queries
        _register_duckdb_tables(deal_df, bt_seat1_df, bt_stats_df)
        with _STATE_LOCK:
            STATE["duckdb_conn"] = DUCKDB_CONN
        _log_memory("after DuckDB registration")

        # If prewarm is disabled, we are ready immediately.
        if not _cli_prewarm:
            with _STATE_LOCK:
                STATE["warming"] = False
                STATE["initializing"] = False
            elapsed = time.time() - t0
            print(f"[init] Completed heavy initialization (pre-warm disabled) in {elapsed:.1f}s")
            return

        # Pre-warm selected query paths so the first user request is faster
        _update_loading_status(7, "Pre-warming endpoints...")
        try:
            _prewarm_all_endpoints(bt_seat1_df)
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
            if STATE["bt_seat1_df"] is not None:
                bt_df_rows = STATE["bt_seat1_df"].height
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


@app.get("/custom-criteria-info")
def get_custom_criteria_info() -> Dict[str, Any]:
    """Get info about custom auction criteria loaded from CSV (hot-reloadable overlay)."""
    with _STATE_LOCK:
        if not STATE["initialized"]:
            return {"initialized": False, "stats": None}
        stats = STATE.get("custom_criteria_stats", {})
    return {
        "initialized": True,
        "criteria_file": str(auction_criteria_file),
        "stats": stats,
    }


@app.get("/custom-criteria-rules")
def get_custom_criteria_rules() -> Dict[str, Any]:
    """Get all custom criteria rules from the CSV file."""
    criteria_list = _load_auction_criteria()
    rules = [
        {"partial_auction": partial, "criteria": criteria}
        for partial, criteria in criteria_list
    ]
    return {
        "file_path": str(auction_criteria_file),
        "file_exists": auction_criteria_file.exists(),
        "rules": rules,
        "rule_count": len(rules),
    }


@app.post("/custom-criteria-rules")
def save_custom_criteria_rules(req: CustomCriteriaSaveRequest) -> Dict[str, Any]:
    """Save all custom criteria rules to the CSV file."""
    try:
        with open(auction_criteria_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Write header comment
            writer.writerow(["# Custom auction criteria - format: partial_auction,criterion1,criterion2,..."])
            for rule in req.rules:
                row = [rule.partial_auction.strip().upper()] + [c.strip() for c in rule.criteria]  # Canonical UPPERCASE
                writer.writerow(row)
        
        return {
            "success": True,
            "file_path": str(auction_criteria_file),
            "rules_saved": len(req.rules),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")


@app.post("/custom-criteria-validate")
def validate_custom_criteria(req: CustomCriteriaValidateRequest) -> Dict[str, Any]:
    """Validate a criteria expression syntax."""
    expression = req.expression.strip()
    
    # Check basic format: COL OP VALUE or COL OP COL
    pattern = r'^(\w+)\s*(>=|<=|>|<|==|!=)\s*(\w+|\d+\.?\d*)$'
    match = re.match(pattern, expression)
    
    if not match:
        return {
            "valid": False,
            "expression": expression,
            "error": "Invalid format. Expected: COLUMN OPERATOR VALUE (e.g., 'HCP >= 12', 'SL_S > SL_H')",
        }
    
    left, op, right = match.groups()
    
    # Check if left is a known column
    if left not in HAND_CRITERIA_COLUMNS:
        return {
            "valid": False,
            "expression": expression,
            "error": f"Unknown column '{left}'. Valid columns: {sorted(HAND_CRITERIA_COLUMNS)}",
            "warning": True,  # Warning, not error - might be valid for advanced use
        }
    
    # Check if right is a number or valid column
    try:
        float(right)
        is_number = True
    except ValueError:
        is_number = False
    
    if not is_number and right not in HAND_CRITERIA_COLUMNS:
        return {
            "valid": False,
            "expression": expression,
            "error": f"Right side '{right}' is neither a number nor a known column",
            "warning": True,
        }
    
    return {
        "valid": True,
        "expression": expression,
        "parsed": {"left": left, "operator": op, "right": right},
    }


@app.post("/custom-criteria-preview")
def preview_custom_criteria(req: CustomCriteriaPreviewRequest) -> Dict[str, Any]:
    """Preview how many auctions would be affected by a rule."""
    _ensure_ready()
    
    partial = req.partial_auction.strip().upper()  # Canonical UPPERCASE
    
    with _STATE_LOCK:
        bt_seat1_df = STATE["bt_seat1_df"]
    
    # Find auctions that start with this partial (ignore leading passes)
    matches = bt_seat1_df.filter(_normalize_auction_expr().str.starts_with(partial))
    
    # Determine which seat this affects
    num_dashes = partial.count('-')
    seat = num_dashes + 1
    
    # Get sample auctions
    sample_auctions = matches.head(10).select("Auction").to_series().to_list()
    
    return {
        "partial_auction": partial,
        "seat_affected": seat,
        "auctions_affected": matches.height,
        "sample_auctions": sample_auctions,
    }


@app.post("/custom-criteria-reload")
def reload_custom_criteria() -> Dict[str, Any]:
    """Hot-reload custom criteria overlay from CSV (no bt/deal reload)."""
    _ensure_ready()
    try:
        with _STATE_LOCK:
            available_criteria_names = STATE.get("available_criteria_names")
        overlay, stats = _build_custom_criteria_overlay(available_criteria_names)
        with _STATE_LOCK:
            STATE["custom_criteria_overlay"] = overlay
            STATE["custom_criteria_stats"] = stats
        return {"success": True, "message": "Custom criteria overlay reloaded successfully", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


def _ensure_ready() -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    Dict[int, Dict[str, pl.DataFrame]],
    Dict[Tuple[str, int], Dict[str, Any]],
]:
    with _STATE_LOCK:
        if not STATE["initialized"]:
            if STATE["initializing"]:
                raise HTTPException(status_code=503, detail="Initialization in progress")
            raise HTTPException(status_code=503, detail="Service not initialized")
        deal_df = STATE["deal_df"]
        bt_seat1_df = STATE["bt_seat1_df"]
        bt_openings_df = STATE["bt_openings_df"]
        deal_criteria_by_seat_dfs = STATE["deal_criteria_by_seat_dfs"]
        results = STATE["results"]

    assert isinstance(deal_df, pl.DataFrame)
    assert isinstance(bt_seat1_df, pl.DataFrame)
    assert isinstance(bt_openings_df, pl.DataFrame)
    return deal_df, bt_seat1_df, bt_openings_df, deal_criteria_by_seat_dfs, results


# ---------------------------------------------------------------------------
# API: opening bid details
# ---------------------------------------------------------------------------


@app.post("/openings-by-deal-index")
def openings_by_deal_index(req: OpeningsByDealIndexRequest) -> Dict[str, Any]:
    """Opening bids by deal index - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()  # Validate state is ready
    
    with _STATE_LOCK:
        state = dict(STATE)  # Shallow copy for handler
    
    try:
        # Access plugin dynamically to get fresh version
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")
            
        resp = handler_module.handle_openings_by_deal_index(
            state=state,
            sample_size=req.sample_size,
            seats=req.seats,
            directions=req.directions,
            opening_directions=req.opening_directions,
            seed=req.seed,
            load_auction_criteria_fn=_load_auction_criteria,
            filter_auctions_by_hand_criteria_fn=_filter_auctions_by_hand_criteria,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("openings-by-deal-index", e)


# ---------------------------------------------------------------------------
# API: bidding sequences (random samples of completed auctions)
# ---------------------------------------------------------------------------


@app.post("/random-auction-sequences")
def random_auction_sequences(req: RandomAuctionSequencesRequest) -> Dict[str, Any]:
    """Random auction sequences - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()

    with _STATE_LOCK:
        state = dict(STATE)
    
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_random_auction_sequences(
            state=state,
            n_samples=req.n_samples,
            seed=req.seed,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("random-auction-sequences", e)


# ---------------------------------------------------------------------------
# API: auctions matching a pattern
# ---------------------------------------------------------------------------


@app.post("/auction-sequences-matching")
def auction_sequences_matching(req: AuctionSequencesMatchingRequest) -> Dict[str, Any]:
    """Auction sequences matching pattern - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()

    with _STATE_LOCK:
        state = dict(STATE)
    
    if state.get("bt_seat1_df") is None or "previous_bid_indices" not in state["bt_seat1_df"].columns:
        raise HTTPException(status_code=500, detail="Column 'previous_bid_indices' not found in bt_seat1_df")
    
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_auction_sequences_matching(
            state=state,
            pattern=req.pattern,
            allow_initial_passes=req.allow_initial_passes,
            n_samples=req.n_samples,
            seed=req.seed,
            apply_auction_criteria_fn=_apply_auction_criteria,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("auction-sequences-matching", e)


@app.post("/auction-sequences-by-index")
def auction_sequences_by_index(req: AuctionSequencesByIndexRequest) -> Dict[str, Any]:
    """Auction sequences by bt_index list - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()

    with _STATE_LOCK:
        state = dict(STATE)

    if state.get("bt_seat1_df") is None or "previous_bid_indices" not in state["bt_seat1_df"].columns:
        raise HTTPException(status_code=500, detail="Column 'previous_bid_indices' not found in bt_seat1_df")

    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_auction_sequences_by_index(
            state=state,
            indices=req.indices,
            allow_initial_passes=req.allow_initial_passes,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("auction-sequences-by-index", e)


@app.post("/deal-criteria-eval-batch")
def deal_criteria_eval_batch(req: DealCriteriaEvalBatchRequest) -> Dict[str, Any]:
    """Evaluate criteria for a specific deal row index, for multiple (seat, criteria) checks.

    Returns per-seat: passed, failed, untracked.
    """
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_deal_criteria_failures_batch(
            state=state,
            deal_row_idx=int(req.deal_row_idx),
            dealer=str(req.dealer),
            checks=[c.model_dump() for c in req.checks],
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("deal-criteria-eval-batch", e)


# ---------------------------------------------------------------------------
# API: deals for auction pattern
# ---------------------------------------------------------------------------


@app.post("/deals-matching-auction")
def deals_matching_auction(req: DealsMatchingAuctionRequest) -> Dict[str, Any]:
    """Deals matching auction pattern - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_deals_matching_auction(
            state=state,
            pattern=req.pattern,
            allow_initial_passes=req.allow_initial_passes,
            n_auction_samples=req.n_auction_samples,
            n_deal_samples=req.n_deal_samples,
            seed=req.seed,
            dist_pattern=req.dist_pattern,
            sorted_shape=req.sorted_shape,
            dist_direction=req.dist_direction,
            wrong_bid_filter=req.wrong_bid_filter,
            apply_auction_criteria_fn=_apply_auction_criteria,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("deals-matching-auction", e)


# ---------------------------------------------------------------------------
# API: bidding table statistics
# ---------------------------------------------------------------------------


@app.post("/bidding-table-statistics")
def bidding_table_statistics(req: BiddingTableStatisticsRequest) -> Dict[str, Any]:
    """Bidding table statistics - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_bidding_table_statistics(
            state=state,
            auction_pattern=req.auction_pattern,
            allow_initial_passes=req.allow_initial_passes,
            sample_size=req.sample_size,
            min_matches=req.min_matches,
            seed=req.seed,
            dist_pattern=req.dist_pattern,
            sorted_shape=req.sorted_shape,
            dist_seat=req.dist_seat,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("bidding-table-statistics", e)


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
            try:
                vul = VUL_MAP.get(int(board.vul), 'None') if board.vul is not None else 'None'
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
    """Process PBN or LIN deal(s) - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_process_pbn(
            state=state,
            pbn_input=req.pbn,
            include_par=req.include_par,
            default_vul=req.vul,
            parse_file_with_endplay_fn=_parse_file_with_endplay,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("process-pbn", e)


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
    
    # Criteria matching is defined in "seat-1 view" (no leading P-), so strip leading passes for matching.
    _pass_prefix_re = re.compile(r"^(P-)+")
    
    for row_idx in range(df.height):
        row = df.row(row_idx, named=True)
        auction = str(row.get(auction_col, '')).upper()  # Canonical UPPERCASE
        auction_norm = _pass_prefix_re.sub("", auction)
        keep_row = True
        failed_criteria = []
        matched_partial = None
        
        # Check each criterion set
        for partial_auction, criteria in criteria_list:
            if auction_norm.startswith(partial_auction):
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
    
    # Select rows in a type-checker-friendly way (and preserve order).
    idx_df = pl.DataFrame({"_i": rows_to_keep, "_pos": list(range(len(rows_to_keep)))})
    filtered_df = (
        df.with_row_index("_i")
        .join(idx_df, on="_i", how="inner")
        .sort("_pos")
        .drop(["_i", "_pos"])
    )
    return filtered_df, rejected_info


@app.post("/find-matching-auctions")
def find_matching_auctions(req: FindMatchingAuctionsRequest) -> Dict[str, Any]:
    """Find matching auctions - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    
    if state.get("bt_seat1_df") is None:
        # This should never happen: bt_seat1 is a required file and init hard-fails if missing.
        raise HTTPException(status_code=503, detail="bt_seat1_df not loaded (pipeline error).")
    
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_find_matching_auctions(
            state=state,
            hcp=req.hcp, sl_s=req.sl_s, sl_h=req.sl_h, sl_d=req.sl_d, sl_c=req.sl_c,
            total_points=req.total_points, seat=req.seat, max_results=req.max_results,
            load_auction_criteria_fn=_load_auction_criteria,
            filter_auctions_by_hand_criteria_fn=_filter_auctions_by_hand_criteria,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("find-matching-auctions", e)


# ---------------------------------------------------------------------------
# API: PBN Lookup
# ---------------------------------------------------------------------------

@app.get("/pbn-sample")
def get_pbn_sample() -> Dict[str, Any]:
    """Get a sample PBN from the first row of deal_df for testing."""
    state, reload_info, handler = _prepare_handler_call()
    try:
        resp = handler.handle_pbn_sample(state)
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _log_and_raise("pbn-sample", e)


@app.get("/pbn-random")
def get_pbn_random() -> Dict[str, Any]:
    """Get a random PBN from deal_df (YOLO mode)."""
    state, reload_info, handler = _prepare_handler_call()
    try:
        resp = handler.handle_pbn_random(state)
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _log_and_raise("pbn-random", e)


@app.post("/pbn-lookup")
def pbn_lookup(req: PBNLookupRequest) -> Dict[str, Any]:
    """Look up a PBN deal in bbo_mldf_augmented.parquet and return matching rows."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_pbn_lookup(state, req.pbn, req.max_results)
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("pbn-lookup", e)


# ---------------------------------------------------------------------------
# API: Group by Bid
# ---------------------------------------------------------------------------

# calculate_imp is now imported from bbo_bidding_queries_lib

@app.post("/group-by-bid")
def group_by_bid(req: GroupByBidRequest) -> Dict[str, Any]:
    """Group deals by bid - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    
    if 'bid' not in state["deal_df"].columns:
        raise HTTPException(status_code=500, detail="Column 'bid' not found in deal_df")
    
    try:
        # Access plugin dynamically
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_group_by_bid(
            state=state,
            auction_pattern=req.auction_pattern,
            n_auction_groups=req.n_auction_groups,
            n_deals_per_group=req.n_deals_per_group,
            seed=req.seed,
            min_deals=req.min_deals,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("group-by-bid", e)


# ---------------------------------------------------------------------------
# API: On-the-fly bt seat stats
# ---------------------------------------------------------------------------


@app.post("/bt-seat-stats")
def bt_seat_stats(req: BTSeatStatsRequest) -> Dict[str, Any]:
    """Compute on-the-fly hand stats for a single bt_seat1 row and seat (or all seats)."""
    reload_info = _reload_plugins()
    _ensure_ready()

    with _STATE_LOCK:
        state = dict(STATE)
        bt_seat1_df = STATE.get("bt_seat1_df")

    if bt_seat1_df is None:
        raise HTTPException(status_code=503, detail="bt_seat1_df not loaded (pipeline error).")

    try:
        row_df = bt_seat1_df.filter(pl.col("bt_index") == req.bt_index)
        if row_df.height == 0:
            raise HTTPException(status_code=404, detail=f"bt_index {req.bt_index} not found in bt_seat1_df")
        bt_row = row_df.row(0, named=True)

        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_bt_seat_stats(
            state=state,
            bt_row=bt_row,
            seat=req.seat,
            max_deals=req.max_deals or None,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("bt-seat-stats", e)


# ---------------------------------------------------------------------------
# API: Execute SQL
# ---------------------------------------------------------------------------


@app.post("/execute-sql")
def execute_sql(req: ExecuteSQLRequest) -> Dict[str, Any]:
    """Execute user-provided SQL against registered tables (deals, auctions)."""
    _ensure_ready()
    
    if DUCKDB_CONN is None:
        raise HTTPException(status_code=503, detail="DuckDB not initialized")
    
    t0 = time.time()
    sql = req.sql.strip()
    
    # Remove trailing semicolons (they break subquery wrapping)
    sql = sql.rstrip(';').strip()
    
    # Security: only allow SELECT queries
    sql_lower = sql.lower()
    if any(kw in sql_lower for kw in ['drop', 'delete', 'update', 'insert', 'alter', 'create', 'truncate']):
        raise HTTPException(status_code=400, detail="Only SELECT queries allowed")
    
    try:
        with _DUCKDB_LOCK:
            # Execute with row limit for safety
            if req.max_rows > 0:
                limited_sql = f"SELECT * FROM ({sql}) AS _subq LIMIT {req.max_rows}"
            else:
                limited_sql = sql
            result = DUCKDB_CONN.execute(limited_sql).pl()
        
        elapsed_ms = (time.time() - t0) * 1000
        print(f"[execute-sql] {result.height} rows in {elapsed_ms:.1f}ms")
        
        return {
            "rows": result.to_dicts(),
            "columns": result.columns,
            "row_count": result.height,
            "sql": sql,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL Error: {e}")


# ---------------------------------------------------------------------------
# API: Wrong Bid Statistics
# ---------------------------------------------------------------------------


@app.post("/wrong-bid-stats")
def wrong_bid_stats(req: WrongBidStatsRequest) -> Dict[str, Any]:
    """Aggregate statistics about wrong bids across the dataset."""
    state, reload_info, handler = _prepare_handler_call()
    try:
        resp = handler.handle_wrong_bid_stats(
            state=state,
            auction_pattern=req.auction_pattern,
            seat=req.seat,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("wrong-bid-stats", e)


@app.post("/failed-criteria-summary")
def failed_criteria_summary(req: FailedCriteriaSummaryRequest) -> Dict[str, Any]:
    """Analysis of which criteria fail most often."""
    state, reload_info, handler = _prepare_handler_call()
    try:
        resp = handler.handle_failed_criteria_summary(
            state=state,
            auction_pattern=req.auction_pattern,
            top_n=req.top_n,
            seat=req.seat,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("failed-criteria-summary", e)


@app.post("/wrong-bid-leaderboard")
def wrong_bid_leaderboard(req: WrongBidLeaderboardRequest) -> Dict[str, Any]:
    """Leaderboard of bids with highest error rates."""
    state, reload_info, handler = _prepare_handler_call()
    try:
        resp = handler.handle_wrong_bid_leaderboard(
            state=state,
            top_n=req.top_n,
            seat=req.seat,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("wrong-bid-leaderboard", e)


# ---------------------------------------------------------------------------
# API: Bidding Models Registry
# ---------------------------------------------------------------------------


@app.get("/bidding-models")
def list_bidding_models() -> Dict[str, Any]:
    """List all available bidding models for the Bidding Arena."""
    models = MODEL_REGISTRY.list_models()
    return {
        "models": models,
        "count": len(models),
        "help": "Use these model names in /bidding-arena requests",
    }


# ---------------------------------------------------------------------------
# API: Bidding Arena (Model Comparison)
# ---------------------------------------------------------------------------


@app.post("/bidding-arena")
def bidding_arena(req: BiddingArenaRequest) -> Dict[str, Any]:
    """Bidding Arena: Head-to-head comparison between two bidding models."""
    reload_info = _reload_plugins()
    _ensure_ready()
    
    # Validate model names against registry
    if not MODEL_REGISTRY.is_valid_model(req.model_a):
        available = [m["name"] for m in MODEL_REGISTRY.list_models()]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_a: '{req.model_a}'. Available models: {available}"
        )
    if not MODEL_REGISTRY.is_valid_model(req.model_b):
        available = [m["name"] for m in MODEL_REGISTRY.list_models()]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_b: '{req.model_b}'. Available models: {available}"
        )
    if req.model_a == req.model_b:
        raise HTTPException(
            status_code=400,
            detail="model_a and model_b must be different"
        )
    
    with _STATE_LOCK:
        state = dict(STATE)
    
    # Check model availability
    model_a = MODEL_REGISTRY.get(req.model_a)
    model_b = MODEL_REGISTRY.get(req.model_b)
    if model_a and not model_a.is_available(state):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model_a}' is not available (required data not loaded)"
        )
    if model_b and not model_b.is_available(state):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model_b}' is not available (required data not loaded)"
        )
    
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_bidding_arena(
            state=state,
            model_a=req.model_a,
            model_b=req.model_b,
            auction_pattern=req.auction_pattern,
            sample_size=req.sample_size,
            seed=req.seed,
            deals_uri=req.deals_uri,
            deal_indices=req.deal_indices,
            search_all_bt_rows=bool(req.search_all_bt_rows),
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("bidding-arena", e)


# ---------------------------------------------------------------------------
# API: auction DD analysis (deals matching auction with DD columns)
# ---------------------------------------------------------------------------


@app.post("/auction-dd-analysis")
def auction_dd_analysis(req: AuctionDDAnalysisRequest) -> Dict[str, Any]:
    """Get DD columns for deals matching an auction's criteria.
    
    Finds the auction in BT, matches its Agg_Expr_Seat_[1-4] criteria against
    the deal bitmaps, and returns DD_[NESW]_[CDHSN] columns for matched deals.
    """
    reload_info = _reload_plugins()
    _ensure_ready()
    
    with _STATE_LOCK:
        state = dict(STATE)
    
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_auction_dd_analysis(
            state=state,
            auction=req.auction,
            max_deals=req.max_deals,
            seed=req.seed,
            vul_filter=req.vul_filter,
            include_hands=req.include_hands,
            include_scores=req.include_scores,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("auction-dd-analysis", e)


@app.post("/rank-bids-by-ev")
def rank_bids_by_ev(req: RankBidsByEVRequest) -> Dict[str, Any]:
    """Rank all possible next bids after an auction by Expected Value (EV).
    
    Given an auction prefix (or empty for opening bids), finds all possible next bids
    using next_bid_indices and computes average EV for each bid across matching deals.
    """
    reload_info = _reload_plugins()
    _ensure_ready()
    
    with _STATE_LOCK:
        state = dict(STATE)
    
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_rank_bids_by_ev(
            state=state,
            auction=req.auction,
            max_deals=req.max_deals,
            seed=req.seed,
            vul_filter=req.vul_filter,
            include_hands=req.include_hands,
            include_scores=req.include_scores,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("rank-bids-by-ev", e)


@app.post("/contract-ev-deals")
def contract_ev_deals(req: ContractEVDealsRequest) -> Dict[str, Any]:
    """Return deals matching a selected next bid and a specific contract EV row."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")
        resp = handler_module.handle_contract_ev_deals(
            state=state,
            auction=req.auction,
            next_bid=req.next_bid,
            contract=req.contract,
            declarer=req.declarer,
            seat=req.seat,
            vul=req.vul,
            max_deals=req.max_deals,
            seed=req.seed,
            include_hands=req.include_hands,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("contract-ev-deals", e)


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
        default=0,
        help="Limit deal_df rows for faster startup (default: 0, which means all rows)",
    )
    args = parser.parse_args()

    print("Starting API server...")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
