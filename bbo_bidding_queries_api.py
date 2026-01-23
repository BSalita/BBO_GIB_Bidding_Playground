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

# Suppress Intel Fortran runtime's Ctrl+C handler (must be set before importing numpy/scipy)
# This prevents the ugly "forrtl: error (200): program aborting due to control-C event" on Windows
import os
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import csv
import gc
import logging
import operator
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, TypeVar
import pathlib
import pyarrow.parquet as pq
import psutil


def _fast_exit_handler(signum, frame):
    """Exit immediately without Python's slow garbage collection cleanup."""
    print("\n[shutdown] Received signal, exiting immediately (skipping python GC cleanup). Takes about 30 seconds to clean system pagefile cache.")
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
from datetime import datetime, timezone

import polars as pl
import numpy as np
import duckdb  # pyright: ignore[reportMissingImports]
import requests as http_requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from endplay.types import Deal, Vul, Player
from endplay.dds import calc_dd_table, par

from mlBridge.mlBridgeBiddingLib import (
    DIRECTIONS,
    load_deal_df,
    load_execution_plan_data,
    directional_to_directionless,
    build_or_load_directional_criteria_bitmaps,
)

# NOTE: imported as a module so static analysis doesn't get confused about symbol exports.
import mlBridge.mlBridgeBiddingLib as mlBridgeBiddingLib

from bbo_bidding_queries_lib import (
    evaluate_criterion_for_hand,
    format_elapsed,
    is_regex_pattern,
    get_cached_regex,
    pattern_matches,
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
                        mod = sys.modules[full_module_name]
                        if hasattr(mod, "__spec__") and mod.__spec__ is not None:
                            module = importlib.reload(mod)
                        else:
                            # Fallback: if spec is missing, try to import again
                            module = importlib.import_module(full_module_name)
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
    """Attach hot-reload info to a JSON response payload.
    
    Also enforces a UI-friendly invariant: lists of deal-like dict rows are
    returned sorted by `index` unless the endpoint explicitly implements a
    different ordering (in which case rows typically do not include `index`).
    """
    def _safe_int(x: Any) -> int:
        try:
            # Handle numpy scalars, strings, etc.
            return int(x)  # type: ignore[arg-type]
        except Exception:
            return 2**31 - 1

    def _sort_lists_by_index(obj: Any) -> Any:
        # Recursively sort list[dict] by "index" when present.
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = _sort_lists_by_index(v)
            return obj
        if isinstance(obj, list):
            # Recurse first so nested structures get normalized.
            for i in range(len(obj)):
                obj[i] = _sort_lists_by_index(obj[i])
            if obj and all(isinstance(e, dict) for e in obj):
                has_any_index = any(("index" in e) for e in obj)  # type: ignore[operator]
                if has_any_index:
                    # Stable, deterministic ordering for display.
                    obj.sort(key=lambda d: (0 if "index" in d else 1, _safe_int(d.get("index"))))  # type: ignore[call-arg]
            return obj
        return obj

    # Apply sorting normalization before attaching metadata.
    resp = _sort_lists_by_index(resp)
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


_AGG_EXPR_SEAT_COLS = {
    "Agg_Expr_Seat_1",
    "Agg_Expr_Seat_2",
    "Agg_Expr_Seat_3",
    "Agg_Expr_Seat_4",
}

# If Agg_Expr columns appear on a huge bt_seat1_df, we will thrash pagefile.
# This is a safety rail to prevent accidental regressions.
_MAX_BT_ROWS_ALLOWED_WITH_AGG_EXPR_LOADED = 5_000_000


def _sanity_check_state_for_memory(state: Dict[str, Any]) -> None:
    """Fail-fast checks to prevent catastrophic memory/pagefile thrashing.

    Invariants:
    - `STATE["bt_seat1_df"]` should NOT include Agg_Expr_Seat_* columns for the full BT
      (461M rows). Those columns must be loaded on-demand for small bt_index subsets.
    - If any endpoint needs on-demand Agg_Expr loading, `STATE["bt_seat1_file"]` must exist.
    """
    bt_df = state.get("bt_seat1_df")
    if isinstance(bt_df, pl.DataFrame):
        if _AGG_EXPR_SEAT_COLS.intersection(set(bt_df.columns)):
            if bt_df.height > _MAX_BT_ROWS_ALLOWED_WITH_AGG_EXPR_LOADED:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "FATAL: bt_seat1_df was loaded with Agg_Expr_Seat_* columns for a large BT.\n"
                        f"Rows={bt_df.height:,} (limit={_MAX_BT_ROWS_ALLOWED_WITH_AGG_EXPR_LOADED:,}).\n"
                        "This will cause massive RAM/pagefile thrashing.\n"
                        "Fix: do not load Agg_Expr_Seat_* in _load_bt_seat1_df; load on-demand by bt_index."
                    ),
                )

    # If we are running the lightweight BT (no Agg_Expr in memory), ensure the file path is present.
    # This supports on-demand loading in handler utilities.
    bt_file = state.get("bt_seat1_file")
    if bt_file is None:
        # Not all endpoints need it, but missing it usually indicates a state wiring bug.
        # Keep as a warning-only check to avoid breaking read-only endpoints.
        # (Handlers that require on-demand Agg_Expr will raise a clearer error.)
        pass


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
    _sanity_check_state_for_memory(state)
    handler_module = _get_handler_module()
    return state, reload_info, handler_module


def _resolve_deal_row_idx_from_index(state: Dict[str, Any], deal_index: int) -> int:
    """Resolve user-facing deal `index` -> row position used by bitmaps (Invariant B).
    
    Requires STATE["deal_index_monotonic"] and STATE["deal_index_arr"] built at startup.
    """
    idx_arr = state.get("deal_index_arr")
    if idx_arr is None:
        raise HTTPException(status_code=500, detail="deal_index_arr missing from state (startup cache not built)")
    import numpy as np
    deal_index_i = int(deal_index)

    # Fast-path: monotonic index enables O(log n) lookup via binary search.
    if bool(state.get("deal_index_monotonic", False)):
        pos = int(np.searchsorted(idx_arr, deal_index_i))
        if pos < 0 or pos >= len(idx_arr) or int(idx_arr[pos]) != deal_index_i:
            raise HTTPException(status_code=404, detail=f"Deal index {deal_index_i} not found")
        return int(pos)

    # Fallback: non-monotonic `index` column. Use a vectorized equality scan against the cached numpy array.
    # This is O(n) but is typically fast in NumPy and only used for small pinned-deal lookups.
    try:
        matches = np.flatnonzero(idx_arr == deal_index_i)
    except Exception:
        # Some pipelines may produce an object/float array; try a best-effort cast for comparison.
        try:
            matches = np.flatnonzero(idx_arr.astype(np.int64, copy=False) == deal_index_i)
        except Exception:
            matches = np.array([], dtype=np.int64)

    if matches.size == 0:
        raise HTTPException(status_code=404, detail=f"Deal index {deal_index_i} not found")

    # If duplicates exist, choose the first row position deterministically.
    return int(matches[0])


def _resolve_deal_row_indices_from_indices(state: Dict[str, Any], indices: list[int]) -> list[int]:
    """Resolve a list of deal `index` -> row positions, preserving order (skips missing)."""
    out: list[int] = []
    for x in indices:
        try:
            out.append(_resolve_deal_row_idx_from_index(state, int(x)))
        except HTTPException:
            # Skip missing indices for batch use-cases; caller can decide how to report.
            continue
    return out


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
    return 1000000  # Default to 1M rows for faster startup


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
    raise FileNotFoundError(f"Data directory does not exist: {dataPath}")
print(f"dataPath: {dataPath} (exists)")


# ---------------------------------------------------------------------------
# Required files check
# ---------------------------------------------------------------------------

exec_plan_file = dataPath.joinpath("bbo_bt_execution_plan_data.pkl")
# Use source deals file (16M rows) instead of CPU-pipeline matches file (10K rows)
# Deal→BT mapping comes from bbo_deal_to_bt_verified.parquet (GPU pipeline output)
bbo_mldf_augmented_file = dataPath.joinpath("bbo_mldf_augmented.parquet")
bt_seat1_file = dataPath.joinpath("bbo_bt_compiled.parquet")  # Pre-compiled BT with learned rules baked in
bt_augmented_file = dataPath.joinpath("bbo_bt_augmented.parquet")  # Full bidding table (all seats/prefixes)
bt_aggregates_file = dataPath.joinpath("bbo_bt_aggregate.parquet")
bt_categories_file = dataPath.joinpath("bbo_bt_categories.parquet")  # 103 bid-category boolean flags (Phase 4)
bt_ev_stats_file = dataPath.joinpath("bt_ev_par_stats_gpu.parquet")  # Per-BT EV/Par stats (GPU pipeline output)
auction_criteria_file = dataPath.joinpath("bbo_custom_auction_criteria.csv")
# New rules file: detailed rule discovery metrics (detailed view)
new_rules_file = dataPath.joinpath("bbo_bt_new_rules.parquet")
# Precomputed deal-to-BT verified index (GPU pipeline output, optional)
deal_to_bt_verified_file = dataPath.joinpath("bbo_deal_to_bt_verified.parquet")

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
        # Hybrid matching: literal prefix for simple patterns, regex for complex ones
        if is_regex_pattern(partial_auction):
            # Regex path - auto-anchor for prefix matching
            regex_pat = partial_auction if partial_auction.startswith('^') else f'^{partial_auction}'
            matches_partial = _normalize_auction_expr(auction_col).str.contains(f'(?i){regex_pat}')
        else:
            # Fast literal path
            matches_partial = _normalize_auction_expr(auction_col).str.starts_with(partial_auction)
        
        # Calculate seat from partial auction (used in both paths)
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
    - `bbo_bt_compiled.parquet` is a REQUIRED pipeline artifact for this API. If it's missing,
      something is wrong upstream and we hard-fail rather than falling back.
    - `bbo_bt_criteria_seat1_df.parquet` is now also REQUIRED at runtime; it provides the
      completed-auction criteria/aggregate stats (`bt_stats_df`) used by several endpoints
      (Bidding Table Explorer, Find Matching Auctions, etc.). If it's missing, we fail fast
      instead of running with partially functional endpoints.
    - We intentionally do NOT load the full `bbo_bt_augmented.parquet` into memory; all heavy
      preprocessing should be done offline by `bbo_bt_compile_rules.py` and friends.
    """
    required: list[pathlib.Path] = [
        exec_plan_file,
        bbo_mldf_augmented_file,
        bt_seat1_file,
        bt_criteria_seat1_file,
        bt_categories_file,
        bt_completed_agg_file,
        deal_to_bt_verified_file,
        new_rules_file,
        auction_criteria_file,
    ]

    missing: list[str] = []
    for f in required:
        if not f.exists():
            missing.append(str(f))
    return missing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - starts initialization on startup."""
    # Fast-fail check: ensure all required files exist before starting background thread
    print("[init] Checking required files...")
    missing_files = _check_required_files()
    if missing_files:
        error_msg = f"FATAL: Missing required files:\n" + "\n".join(f"  - {f}" for f in missing_files)
        print(f"\n{'='*60}\n{error_msg}\n{'='*60}\n")
        # Hard fail - prevents server from starting or serving requests in broken state
        os._exit(1)

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


@app.middleware("http")
async def _access_log_with_elapsed(request, call_next):
    """Access log with elapsed time (replaces uvicorn access log).
    
    We intentionally disable uvicorn's built-in access_log and emit our own line
    that includes elapsed seconds.
    """
    t0 = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        try:
            elapsed_s = time.perf_counter() - t0
            client = getattr(request, "client", None)
            client_host = getattr(client, "host", "?")
            client_port = getattr(client, "port", "?")
            method = getattr(request, "method", "?")
            http_version = request.scope.get("http_version", "1.1")
            path = request.url.path
            if request.url.query:
                path = f"{path}?{request.url.query}"
            status_code = int(getattr(response, "status_code", 0) or 0)
            try:
                from http import HTTPStatus
                phrase = HTTPStatus(status_code).phrase
            except Exception:
                phrase = ""
            logger = logging.getLogger("uvicorn.access")
            # Match uvicorn's default shape but add elapsed at the end.
            logger.info(
                '%s:%s - "%s %s HTTP/%s" %s %s (%.2fs)',
                client_host,
                client_port,
                method,
                path,
                http_version,
                status_code,
                phrase,
                elapsed_s,
            )
        except Exception:
            # Never fail the request because logging failed.
            pass


class StatusResponse(BaseModel):
    initialized: bool
    initializing: bool
    warming: bool = False  # True while pre-warming endpoints
    prewarm_progress: Optional[Dict[str, Any]] = None  # {"i": 1, "n": 12, "name": "endpoint-name"}
    error: Optional[str]
    bt_df_rows: Optional[int] = None
    deal_df_rows: Optional[int] = None
    loading_step: Optional[str] = None  # Current loading step
    loaded_files: Optional[Dict[str, Any]] = None  # File name -> info string (e.g., "1,234 rows × 45 cols (100.5 MB)")


def _format_bytes(size_bytes: int | float) -> str:
    """Format bytes to human-readable string (KB/MB/GB)."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / 1024:.1f} KB"


def _format_file_info(
    df: pl.DataFrame | None = None,
    file_path: pathlib.Path | None = None,
    row_count: int | str | None = None,
    col_count: int | None = None,
    elapsed_secs: float | None = None,
    rss_delta_bytes: int | None = None,
) -> str:
    """Format file loading info showing shape, file size, memory size, and load time.
    
    Examples:
        "16,234,567 rows × 45 cols | disk: 1.2 GB | mem(est): 5.3 GB | rssΔ: 5.1 GB | 5.3s"
        "123,456 entries | disk: 50.3 MB | mem: 200.1 MB | 1.2s"
    """
    parts = []
    
    # Get shape from DataFrame if provided
    if df is not None:
        rows = f"{df.height:,}"
        cols = df.width
        parts.append(f"{rows} rows × {cols} cols")
    elif row_count is not None:
        if isinstance(row_count, int):
            parts.append(f"{row_count:,} entries")
        else:
            parts.append(str(row_count))
    
    # Get file size on disk if path provided
    if file_path is not None and file_path.exists():
        disk_size = file_path.stat().st_size
        parts.append(f"disk: {_format_bytes(disk_size)}")
    
    # Get memory size if DataFrame provided
    if df is not None:
        mem_size = df.estimated_size()
        parts.append(f"mem(est): {_format_bytes(mem_size)}")

    # Include actual process RSS delta if available (ground truth-ish on Windows)
    if rss_delta_bytes is not None:
        try:
            parts.append(f"rssΔ: {_format_bytes(rss_delta_bytes)}")
        except Exception:
            pass
    
    # Add elapsed time if provided
    if elapsed_secs is not None:
        parts.append(f"{elapsed_secs:.1f}s")
    
    return " | ".join(parts) if parts else "loaded"


class InitResponse(BaseModel):
    status: str


class OpeningsByDealIndexRequest(BaseModel):
    sample_size: int = 6
    seats: Optional[List[int]] = None
    directions: Optional[List[str]] = None
    opening_directions: Optional[List[str]] = None
    seed: Optional[int] = 0


class RandomAuctionSequencesRequest(BaseModel):
    n_samples: int = 10
    # seed=0 means non-reproducible, any other value is reproducible
    seed: Optional[int] = 0
    # If True, only sample from completed auctions (ended in final contract)
    completed_only: bool = True
    # If True, only sample from partial auctions (not completed)
    partial_only: bool = False


class AuctionSequencesMatchingRequest(BaseModel):
    # Allow None from UI; treat it as empty string server-side.
    pattern: str | None = ""
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
    # Primary key used by criteria bitmaps and deal_to_bt indices (row position in deals file).
    deal_row_idx: int
    # Optional deal `index` (user-facing). If provided, server maps it to deal_row_idx using
    # the monotonic index cache built at startup.
    deal_index: Optional[int] = None
    dealer: str  # N/E/S/W
    checks: List[DealCriteriaCheck]


class DealsByIndexRequest(BaseModel):
    """Fetch deals by user-facing `index` values (monotonic fast-path).
    
    Returns rows in the same order as the input list.
    """
    indices: List[int]
    max_rows: int = 200  # safety cap
    # Optional list of columns to return (None = default subset)
    columns: Optional[List[str]] = None


class BTCategoriesByIndexRequest(BaseModel):
    """Fetch bid-category flags (Phase 4) by bt_index values.

    Returns rows in the same order as the input list.
    """
    indices: List[int]
    max_rows: int = 500  # safety cap


class ResolveAuctionPathRequest(BaseModel):
    """Resolve a full auction string into a path of detailed step info.

    Used by Auction Builder to fetch criteria and metadata for an entire path in one call.
    """
    auction: str


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


class SampleDealsByAuctionPatternRequest(BaseModel):
    """Sample deals whose *actual auction* matches a regex pattern.

    This is intentionally lightweight (no Rules matching / BT criteria work) and is used by
    Streamlit's Auction Builder for fast "Show Matching Deals".
    """

    pattern: str
    sample_size: int = 25
    seed: Optional[int] = 42


class AuctionPatternCountsRequest(BaseModel):
    """Return counts for multiple *actual auction* regex patterns.

    This is the batch version of /sample-deals-by-auction-pattern when you only need counts.
    It avoids N separate HTTP calls from Streamlit (less overhead, cleaner logs).
    """

    patterns: List[str]


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
    # Optional: include 100+ bid-category boolean flags (from bbo_bt_categories.parquet)
    include_categories: bool = False


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


class NewRulesLookupRequest(BaseModel):
    """Request for detailed new rules metrics for a specific auction step."""
    auction: str
    bt_index: Optional[int] = None


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


class ListNextBidsRequest(BaseModel):
    """Request for List Next Bids: fast lookup of available next bids using BT's next_bid_indices."""
    auction: str = ""  # Auction prefix (empty = opening bids)


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


class BestAuctionsLookaheadRequest(BaseModel):
    """Request for server-side DFS to find best completed auctions by DD or EV."""
    deal_row_idx: int          # Deal row index (from deal_df)
    auction_prefix: str = ""   # Current auction prefix (empty = opening)
    metric: str = "DD"         # "DD" or "EV"
    max_depth: int = 20        # Max search depth
    max_results: int = 10      # Max results to return
    # If True, Pass bids are always treated as valid even if their criteria fails.
    permissive_pass: bool = True


class BestAuctionsLookaheadStartRequest(BaseModel):
    """Request to start an async best-auctions lookahead job (avoid client timeouts)."""
    deal_row_idx: int
    auction_prefix: str = ""
    metric: str = "DD"
    max_depth: int = 20
    max_results: int = 10
    # Controls for long-running searches (bounded server-side in handler as well).
    deadline_s: float = 1000.0
    max_nodes: int = 200000
    beam_width: int = 50
    # If True, Pass bids are always treated as valid even if their criteria fails.
    permissive_pass: bool = True


class DealMatchedBTSampleRequest(BaseModel):
    """Sample BT rows that match a specific deal using the GPU-verified deal→BT index."""
    # Primary key used by criteria bitmaps and deal_to_bt indices (row position in deals file).
    deal_row_idx: int
    # Optional deal `index` (user-facing). If provided, server maps it to deal_row_idx using
    # the monotonic index cache built at startup.
    deal_index: Optional[int] = None
    n_samples: int = 25
    seed: Optional[int] = 0  # 0 = non-reproducible
    metric: str = "DD"       # "DD" or "EV" (controls which score column is emphasized)
    # If True, Pass bids are always treated as valid even if their criteria fails.
    # TODO: Revisit whether a pass should always be valid. Perhaps it should be
    # rejected but still show in 'Best Bids Ranked by Model'.
    permissive_pass: bool = True


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
# ---------------------------------------------------------------------------
# Gemini-3.2 CSR Traversal Index (Direct-Address CSR)
# ---------------------------------------------------------------------------

def get_bid_vocab():
    """Consistent bid encoding for Gemini-3.2 index."""
    code_to_bid = [""]
    bid_to_code = {"": 0}
    def add(b):
        b = b.upper()
        if b not in bid_to_code:
            bid_to_code[b] = len(code_to_bid); code_to_bid.append(b)
    for b in ["P", "X", "XX", "D", "R"]: add(b)
    for level in range(1, 8):
        for strain in ["C", "D", "H", "S", "N"]:
            add(f"{level}{strain}")
    # Add lowercase mappings for robustness
    for b in ["p", "x", "xx", "d", "r"]:
        if b.upper() in bid_to_code:
            bid_to_code[b] = bid_to_code[b.upper()]
    return bid_to_code, code_to_bid

BID_TO_CODE, CODE_TO_BID = get_bid_vocab()

@dataclass
class G3Index:
    offsets: np.ndarray    # uint32: [bt_index] -> edges start
    children: np.ndarray   # uint32: flat child indices
    bidcodes: np.ndarray   # uint8: flat bid codes
    openings: dict[str, int]
    
    def walk(self, auction: str) -> int | None:
        """O(1) direct-address traversal of the bidding table."""
        # Normalize: strip leading passes for seat-1 view traversal
        auction_norm = re.sub(r"(?i)^(P-)+", "", auction).rstrip("-")
        tokens = [t.upper() for t in auction_norm.split("-") if t]
        if not tokens: return None
        
        curr = self.openings.get(tokens[0])
        if curr is None: return None
        
        for tok in tokens[1:]:
            code = BID_TO_CODE.get(tok, 0)
            start, end = self.offsets[curr], self.offsets[curr + 1]
            found = False
            for i in range(start, end):
                if self.bidcodes[i] == code:
                    curr = int(self.children[i]); found = True; break
            if not found: return None
        return curr

    def list_next_bids(self, bt_index: int) -> list[str]:
        """Fetch all available next bids for a given bt_index."""
        if bt_index >= len(self.offsets) - 1: return []
        start, end = self.offsets[bt_index], self.offsets[bt_index + 1]
        codes = self.bidcodes[start:end]
        # Filter for unique codes and convert to bid strings
        return sorted(list(set([CODE_TO_BID[c] for c in codes if c > 0])))

STATE: Dict[str, Any] = {
    "initialized": False,
    "initializing": False,
    "warming": False,  # True while pre-warming endpoints
    "prewarm_progress": None,  # Populated during endpoint pre-warm
    "error": None,
    "loading_step": None,  # Current loading step description
    "loaded_files": {},  # File name -> row count (updated as files load)
    "deal_df": None,
    "bt_seat1_df": None,  # Pre-compiled BT table (bbo_bt_compiled.parquet)
    "bt_openings_df": None,  # Tiny opening-bid lookup table (built from bt_seat1_df)
    "g3_index": None,       # Gemini-3.2 CSR Traversal Index (built on startup)
    "deal_criteria_by_seat_dfs": None,
    "deal_criteria_by_direction_dfs": None,
    "results": None,
    # Criteria / aggregate statistics for completed auctions (seat-1 view).
    # Built from bbo_bt_criteria.parquet + bbo_bt_aggregate.parquet and keyed by bt_index.
    "bt_stats_df": None,
    # Completed auctions with Agg_Expr only (63MB, ~975K rows) for fast wrong-bid-stats
    "bt_completed_agg_df": None,
    # Optional: bid-category boolean flags from bbo_bt_categories.parquet (Phase 4).
    "bt_categories_df": None,
    "bt_category_cols": None,  # list[str]
    # Fast bt_index -> row position mapping for lightweight BT
    "bt_index_arr": None,      # numpy array of UInt32
    "bt_index_monotonic": False,
    "duckdb_conn": None,
    # Hot-reloadable overlay rules loaded from bbo_custom_auction_criteria.csv.
    # These rules are applied on-the-fly to BT rows when serving responses and when building criteria masks.
    "custom_criteria_overlay": [],
    # Loaded from bbo_bt_new_rules.parquet (optional - for detailed rule inspection)
    "new_rules_df": None,
    "custom_criteria_stats": {},
    # Set of available criterion names (from deal_criteria_by_direction_dfs) for normalizing CSV criteria strings.
    "available_criteria_names": None,
}

# ---------------------------------------------------------------------------
# Async jobs: best auctions lookahead (timeout-safe)
# ---------------------------------------------------------------------------

_BEST_AUCTIONS_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="best-auctions",
)
_BEST_AUCTIONS_JOBS_LOCK = threading.Lock()
_BEST_AUCTIONS_JOBS: Dict[str, Dict[str, Any]] = {}
_BEST_AUCTIONS_JOBS_MAX = 200
_BEST_AUCTIONS_JOBS_TTL_S = 30 * 60  # 30 minutes


def _best_auctions_jobs_gc(now_s: float | None = None) -> None:
    """Remove old/completed jobs to cap memory growth."""
    now = float(now_s if now_s is not None else time.time())
    with _BEST_AUCTIONS_JOBS_LOCK:
        # TTL-based cleanup
        to_del: list[str] = []
        for job_id, job in _BEST_AUCTIONS_JOBS.items():
            created_at = float(job.get("created_at_s") or now)
            if now - created_at > _BEST_AUCTIONS_JOBS_TTL_S:
                to_del.append(str(job_id))
        for job_id in to_del:
            _BEST_AUCTIONS_JOBS.pop(job_id, None)

        # Size-based cleanup (drop oldest)
        if len(_BEST_AUCTIONS_JOBS) > _BEST_AUCTIONS_JOBS_MAX:
            # Sort by created time asc, drop oldest extras
            items = sorted(
                _BEST_AUCTIONS_JOBS.items(),
                key=lambda kv: float(kv[1].get("created_at_s") or now),
            )
            overflow = len(_BEST_AUCTIONS_JOBS) - _BEST_AUCTIONS_JOBS_MAX
            for i in range(max(0, overflow)):
                _BEST_AUCTIONS_JOBS.pop(items[i][0], None)

# Additional optional data file paths
bt_criteria_file = dataPath.joinpath("bbo_bt_criteria.parquet")
# Pre-joined completed-auction criteria/aggregate table (preferred at runtime).
bt_criteria_seat1_file = dataPath.joinpath("bbo_bt_criteria_seat1_df.parquet")
# Completed auctions with Agg_Expr only (63MB, ~975K rows) for fast wrong-bid-stats lookups
bt_completed_agg_file = dataPath.joinpath("bbo_bt_completed_agg_expr.parquet")

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
    """Register DataFrames and Views with DuckDB for SQL queries.
    
    For very large tables (like auctions), we use a View pointing directly to 
    the Parquet file to avoid memory bloat from DuckDB's Polars scanner.
    """
    global DUCKDB_CONN
    with _DUCKDB_LOCK:
        if DUCKDB_CONN is None:
            DUCKDB_CONN = duckdb.connect()
        
        # Register Polars DataFrames directly with DuckDB (Zero-copy)
        DUCKDB_CONN.register("deals", deal_df)
        
        # For the massive auctions table, use a View pointing to the Parquet file 
        # instead of the in-memory Polars DataFrame to save memory.
        # DuckDB can query the Parquet file directly with high performance.
        DUCKDB_CONN.execute(f"CREATE OR REPLACE VIEW auctions AS SELECT * FROM read_parquet('{bt_seat1_file}')")
        
        if bt_stats_df is not None:
            DUCKDB_CONN.register("auction_stats", bt_stats_df)
            print(
                "[duckdb] Registered Polars DataFrames and Views: "
                f"deals ({deal_df.height:,} rows), "
                f"auctions (View on {bt_seat1_file.name}), "
                f"auction_stats ({bt_stats_df.height:,} rows)"
            )
        else:
            print(
                "[duckdb] Registered Polars DataFrames and Views: "
                f"deals ({deal_df.height:,} rows), "
                f"auctions (View on {bt_seat1_file.name})"
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
    
    # Add raw DD trick columns (4 directions × 5 strains = 20 columns)
    # Format: DD_{direction}_{strain} e.g. DD_N_C, DD_E_S
    for direction in DIRECTIONS_LIST:
        for strain in DD_SCORE_STRAINS:
            additional_cols.append(f"DD_{direction}_{strain}")
    
    # Add DD_Score columns for all contracts (for DD_Score_AI computation)
    # Format: DD_Score_{level}{strain}_{direction} e.g. DD_Score_3N_N
    for level in DD_SCORE_LEVELS:
        for strain in DD_SCORE_STRAINS:
            for direction in DIRECTIONS_LIST:
                additional_cols.append(f"DD_Score_{level}{strain}_{direction}")
    
    # Add EV columns for all contracts (for EV computation)
    # Format: EV_{pair}_{declarer}_{strain}_{level} (no vulnerability suffix)
    for pair in ['NS', 'EW']:
        declarers = ['N', 'S'] if pair == 'NS' else ['E', 'W']
        for declarer in declarers:
            for strain in DD_SCORE_STRAINS:
                for level in DD_SCORE_LEVELS:
                    additional_cols.append(f"EV_{pair}_{declarer}_{strain}_{level}")
    
    return additional_cols


def _load_bt_seat1_df() -> pl.DataFrame:
    """Load bt_seat1_df with only the LIGHTWEIGHT columns needed at runtime.
    
    IMPORTANT: We intentionally do NOT load the heavy Agg_Expr_Seat_X columns here.
    These 4 columns contain List(String) data that explodes to 100+ GB in RAM for 461M rows.
    Instead, they are loaded on-demand from the Parquet file only for the specific rows
    needed by each query (see build_opening_bids_table_from_bt_seat1 for the pattern).
    """
    print(f"[init] Loading bt_seat1_df from {bt_seat1_file} (lightweight columns only)...")
    
    # Use scan for efficient column selection
    bt_seat1_scan = pl.scan_parquet(bt_seat1_file)
    available_cols = bt_seat1_scan.collect_schema().names()
    
    # Hard fail if 'bt_index' is missing
    if "bt_index" not in available_cols:
        raise ValueError(f"REQUIRED column 'bt_index' missing from {bt_seat1_file}. Pipeline error.")
    
    # LIGHTWEIGHT columns only - no Agg_Expr_Seat_X (those are loaded on-demand)
    required_cols = [
        "bt_index", "Auction", "is_opening_bid", "is_completed_auction",
        "seat", "candidate_bid", "npasses", "auction_len", "Expr",
        # Agg_Expr_Seat_1..4 are INTENTIONALLY EXCLUDED - too heavy (100+ GB)
        "previous_bid_indices", "next_bid_indices",
        "matching_deal_count",
    ]
    cols_to_load = [c for c in required_cols if c in available_cols]
    print(f"[init] Loading {len(cols_to_load)} of {len(available_cols)} columns (excluding heavy Agg_Expr columns)...")
    
    # Optimization: Categoricalize repeated strings and downcast integers
    # Note: We normalize Auction to uppercase during load to avoid expensive runtime 
    # string operations on 461M rows.
    df = (
        bt_seat1_scan.select(cols_to_load)
        .with_columns([
            pl.col("bt_index").cast(pl.UInt32),
            pl.col("Auction").str.to_uppercase(),
            pl.col("candidate_bid").cast(pl.Categorical),
            pl.col("seat").cast(pl.UInt8),
            pl.col("npasses").cast(pl.UInt8),
            pl.col("auction_len").cast(pl.UInt8),
            pl.col("matching_deal_count").cast(pl.UInt32),
        ])
        .collect()
    )
    
    return df


def _load_or_build_bt_can_complete_sidecar(
    bt_parquet_file: pathlib.Path,
    out_file: pathlib.Path,
    max_bt_index_inclusive: int,
) -> np.ndarray:
    """Sidecar build/load for BT reachability to a completed auction.

    can_complete[bt_index] == 1 iff bt_index is on a path to any completed auction.

    Build: iterate completed rows and mark each row + its previous_bid_indices chain.
    Store: uint8 NumPy array (0/1) at `out_file` for fast mmap load.
    """
    if out_file.exists():
        print(f"[init] Loading can_complete sidecar: {out_file}")
        return np.load(out_file, mmap_mode="r")

    t0 = time.perf_counter()
    start_dt = datetime.now(timezone.utc)
    print(f"[init] Building can_complete sidecar (start={start_dt.isoformat()})")
    print(f"[init]   bt_parquet_file={bt_parquet_file}")
    print(f"[init]   out_file={out_file}")
    print(f"[init]   max_bt_index={max_bt_index_inclusive:,}")

    can_complete = np.zeros(int(max_bt_index_inclusive) + 1, dtype=np.uint8)

    file_path_sql = str(bt_parquet_file).replace("\\", "/")
    conn = duckdb.connect(":memory:")
    try:
        rel = conn.execute(
            f"""
            SELECT bt_index, previous_bid_indices
            FROM read_parquet('{file_path_sql}')
            WHERE is_completed_auction = true
            """
        )

        batch_size = 20_000
        processed = 0
        last_print = time.perf_counter()
        while True:
            rows = rel.fetchmany(batch_size)
            if not rows:
                break
            processed += len(rows)
            prev_accum: list[int] = []
            for bt_idx, prev_list in rows:
                if bt_idx is None:
                    continue
                try:
                    bi = int(bt_idx)
                except Exception:
                    continue
                if 0 <= bi <= max_bt_index_inclusive:
                    can_complete[bi] = 1
                if prev_list:
                    try:
                        prev_accum.extend(int(x) for x in prev_list if x is not None)
                    except Exception:
                        pass
            if prev_accum:
                try:
                    arr = np.asarray(prev_accum, dtype=np.int64)
                    arr = arr[(arr >= 0) & (arr <= max_bt_index_inclusive)]
                    if arr.size:
                        can_complete[arr] = 1
                except Exception:
                    pass
            now = time.perf_counter()
            if (now - last_print) > 5.0:
                last_print = now
                elapsed_s = now - t0
                print(f"[init] can_complete progress: processed_completed={processed:,}, elapsed={elapsed_s:.1f}s")
    finally:
        conn.close()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, can_complete, allow_pickle=False)
    end_dt = datetime.now(timezone.utc)
    elapsed_s = time.perf_counter() - t0
    print(f"[init] Built can_complete sidecar (end={end_dt.isoformat()}, elapsed={elapsed_s:.1f}s)")

    return np.load(out_file, mmap_mode="r")


def _prewarm_all_endpoints(bt_seat1_df: pl.DataFrame) -> None:
    """Pre-warm all endpoints to speed up first user request."""
    
    def _set_prewarm_progress(i: int, n: int, name: str) -> None:
        with _STATE_LOCK:
            STATE["prewarm_progress"] = {"i": int(i), "n": int(n), "name": str(name)}
            # Also decorate loading_step for clients that only show this string
            ls = str(STATE.get("loading_step") or "Pre-warming endpoints...")
            if "Pre-warming endpoints" in ls:
                # Preserve the outer [step/total] prefix if present
                prefix = ""
                rest = ls
                if ls.startswith("[") and "]" in ls:
                    prefix = ls[: ls.index("]") + 1] + " "
                    rest = ls[ls.index("]") + 1 :].strip()
                STATE["loading_step"] = f"{prefix}{rest} ({i}/{n}) {name}"

    def _clear_prewarm_progress() -> None:
        with _STATE_LOCK:
            STATE["prewarm_progress"] = None

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

    # Pick a stable sample deal index for warm-ups that require one
    sample_deal_row_idx = 0
    try:
        with _STATE_LOCK:
            if STATE.get("deal_df") is not None and STATE["deal_df"].height > 1:
                sample_deal_row_idx = 1
    except Exception:
        sample_deal_row_idx = 0

    # Pre-warm bt-seat-stats endpoint + list-next-bids helpers
    first_bt_index = None
    if bt_seat1_df.height > 0:
        try:
            first_bt_index = int(bt_seat1_df.select("bt_index").head(1).item())
        except Exception:
            pass

    # Build task list so we can display i/n counts during warm-up.
    # Keep payloads tiny so prewarm is fast and stable.
    warm_pbn = "N:AKQ2.KQ2.AK2.AK2 T987.987.987.987 J654.654.654.654 3.JT53.QJT53.QJT5"
    tasks: list[tuple[str, Any]] = [
        ("openings-by-deal-index", (openings_by_deal_index, OpeningsByDealIndexRequest(sample_size=1))),
        ("random-auction-sequences", (random_auction_sequences, RandomAuctionSequencesRequest(n_samples=1, seed=42))),
        ("auction-sequences-matching", (auction_sequences_matching, AuctionSequencesMatchingRequest(pattern="^1N-p-3N$", n_samples=1, seed=0))),
        ("deals-matching-auction", (deals_matching_auction, DealsMatchingAuctionRequest(pattern="^1N-p-3N$", n_auction_samples=1, n_deal_samples=1, seed=0))),
        ("bidding-table-statistics", (bidding_table_statistics, BiddingTableStatisticsRequest(auction_pattern="^1N-p-3N$", sample_size=1, seed=42))),
        ("process-pbn", (process_pbn, ProcessPBNRequest(pbn=warm_pbn, include_par=True, vul="None"))),
        ("find-matching-auctions", (find_matching_auctions, FindMatchingAuctionsRequest(hcp=15, sl_s=4, sl_h=3, sl_d=3, sl_c=3, total_points=17, seat=1, max_results=1))),
        ("group-by-bid", (group_by_bid, GroupByBidRequest(auction_pattern="^1N-p-3N$", n_auction_groups=1, n_deals_per_group=1, seed=42))),
        ("pbn-sample", (get_pbn_sample, None)),
        ("pbn-random", (get_pbn_random, None)),
        ("pbn-lookup", (pbn_lookup, PBNLookupRequest(pbn=warm_pbn, max_results=1))),
        ("execute-sql", (execute_sql, ExecuteSQLRequest(sql="SELECT 1 AS x", max_rows=1))),
    ]
    if first_bt_index is not None:
        tasks.append(("bt-seat-stats", (bt_seat_stats, BTSeatStatsRequest(bt_index=first_bt_index, seat=0, max_deals=0))))

    tasks.extend([
        ("wrong-bid-stats", (wrong_bid_stats, WrongBidStatsRequest(auction_pattern=None, seat=None))),
        ("failed-criteria-summary", (failed_criteria_summary, FailedCriteriaSummaryRequest(auction_pattern=None, top_n=5, seat=None))),
        ("wrong-bid-leaderboard", (wrong_bid_leaderboard, WrongBidLeaderboardRequest(top_n=5, seat=None))),
        ("bidding-models", (list_bidding_models, None)),
        ("bidding-arena", (bidding_arena, BiddingArenaRequest(model_a="Rules", model_b="Actual", sample_size=10, seed=42))),
        ("rank-bids-by-ev", (rank_bids_by_ev, RankBidsByEVRequest(auction="", max_deals=10, seed=42))),
        ("new-rules-lookup", (new_rules_lookup, NewRulesLookupRequest(auction="1S-p-2C"))),
        # Commonly used by the Streamlit app / auction builder
        ("resolve-auction-path", (resolve_auction_path, ResolveAuctionPathRequest(auction="1N-p-3N"))),
        ("list-next-bids", (list_next_bids, ListNextBidsRequest(auction="1N-p-3N"))),
        ("deal-criteria-eval-batch", (deal_criteria_eval_batch, DealCriteriaEvalBatchRequest(deal_row_idx=sample_deal_row_idx, dealer="N", checks=[DealCriteriaCheck(seat=1, criteria=[])]))),
        ("auction-pattern-counts", (auction_pattern_counts, AuctionPatternCountsRequest(patterns=["^1N-p-3N$"]))),
        ("auction-dd-analysis", (auction_dd_analysis, AuctionDDAnalysisRequest(auction="1N-p-3N", max_deals=1, seed=42, include_hands=False, include_scores=False))),
        ("best-auctions-lookahead", (best_auctions_lookahead, BestAuctionsLookaheadRequest(deal_row_idx=sample_deal_row_idx, auction_prefix="1N-p-3N", metric="DD", max_depth=3, max_results=2))),
    ])

    n = len(tasks)
    for i, (name, call) in enumerate(tasks, start=1):
        _set_prewarm_progress(i, n, name)
        fn, req = call
        if req is None:
            _prewarm_endpoint(name, fn)
        else:
            _prewarm_endpoint(name, fn, req)
    _clear_prewarm_progress()

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
        t0_deal = time.perf_counter()
        deal_df = load_deal_df(bbo_mldf_augmented_file, valid_deal_columns, mldf_n_rows=_cli_deal_rows)
        elapsed_deal = time.perf_counter() - t0_deal
        
        # Get total row count for display (when using --deal-rows)
        if _cli_deal_rows:
            total_rows = pq.read_metadata(bbo_mldf_augmented_file).num_rows
            row_info = f"{deal_df.height:,} of {total_rows:,} rows × {deal_df.width} cols in {elapsed_deal:.1f}s"
        else:
            row_info = _format_file_info(df=deal_df, file_path=bbo_mldf_augmented_file, elapsed_secs=elapsed_deal)
        _update_loading_status(3, "Building criteria bitmaps...", "deal_df", row_info)
        _log_memory("after load_deal_df")

        # todo: do this earlier in the pipeline?
        # Convert 'bid' from pl.List(pl.Utf8) to pl.Utf8 by joining with '-'
        bid_dtype = deal_df.schema.get("bid")
        if bid_dtype == pl.List(pl.Utf8):
            deal_df = deal_df.with_columns(pl.col('bid').list.join('-'))

        # -------------------------------------------------------------------
        # Deal index monotonic optimization (Invariant B)
        # -------------------------------------------------------------------
        # Many pipelines preserve the deal file row order and keep `index` monotonic.
        # If so, we can map deal `index` -> row position via binary search, which
        # enables fast pinned-deal lookups without scanning/joining the full deal_df.
        #
        # WARNING: We do NOT assume `index == row_position`. We only leverage
        # monotonicity for O(log n) lookup.
        try:
            if "index" in deal_df.columns:
                idx_arr = deal_df.get_column("index").to_numpy()
                # Best-effort monotonic check (non-decreasing). This is O(n) but done once at startup.
                is_mono = bool(np.all(idx_arr[1:] >= idx_arr[:-1])) if len(idx_arr) > 1 else True
                with _STATE_LOCK:
                    STATE["deal_index_arr"] = idx_arr
                    STATE["deal_index_monotonic"] = is_mono
                print(f"[init] deal_df index monotonic: {is_mono} (n={len(idx_arr):,})")
        except Exception as e:
            print(f"[init] WARNING: failed to build deal index monotonic cache: {e}")
        
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

        # Slice bitmap DataFrames to match deal_df row count (for --deal-rows support)
        n_deals = deal_df.height
        for direction in list(deal_criteria_by_direction_dfs.keys()):
            df = deal_criteria_by_direction_dfs[direction]
            if df.height > n_deals:
                deal_criteria_by_direction_dfs[direction] = df.head(n_deals)
        for seat in list(deal_criteria_by_seat_dfs.keys()):
            for direction in list(deal_criteria_by_seat_dfs[seat].keys()):
                df = deal_criteria_by_seat_dfs[seat][direction]
                if df.height > n_deals:
                    deal_criteria_by_seat_dfs[seat][direction] = df.head(n_deals)
        if _cli_deal_rows:
            print(f"[init] Sliced criteria bitmaps to {n_deals:,} rows to match deal_df")

        # TEMPORARY FIX: Force unknown criteria columns to True until bitmaps are regenerated.
        # These criteria exist in BT data but weren't in the bitmap generation pipeline.
        # ALWAYS overwrite (even if column exists with False values).
        # Remove this block after regenerating bitmaps with the updated mlBridgeAugmentLib.py.
        _unknown_criteria_cols = [
            pl.lit(True).alias("Forcing_One_Round"),
            pl.lit(True).alias("Opponents_Cannot_Play_Undoubled_Below_2N"),
            pl.lit(True).alias("Forcing_To_2N"),
            pl.lit(True).alias("Forcing_To_3N"),
        ]
        for direction in list(deal_criteria_by_direction_dfs.keys()):
            df = deal_criteria_by_direction_dfs[direction]
            # ALWAYS overwrite these columns with True (even if they exist with False)
            deal_criteria_by_direction_dfs[direction] = df.with_columns(_unknown_criteria_cols)
        for seat in list(deal_criteria_by_seat_dfs.keys()):
            for direction in list(deal_criteria_by_seat_dfs[seat].keys()):
                df = deal_criteria_by_seat_dfs[seat][direction]
                deal_criteria_by_seat_dfs[seat][direction] = df.with_columns(_unknown_criteria_cols)
        print("[init] TEMPORARY: Forced unknown criteria columns (Forcing_One_Round, Forcing_To_2N, etc.) to True")

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
        _proc = psutil.Process()
        rss0_bt = _proc.memory_info().rss
        t0_bt = time.perf_counter()
        bt_seat1_df = _load_bt_seat1_df()
        elapsed_bt = time.perf_counter() - t0_bt
        rss1_bt = _proc.memory_info().rss
        rss_delta_bt = int(rss1_bt - rss0_bt)
        
        # IMPORTANT:
        # Do NOT strip leading 'p-' prefixes here.
        # Those prefixes encode seat/turn order and are required for correct declarer/contract logic
        # (AI/DD/IMP computations). Matching code strips prefixes at query time instead.
        bt_seat1_info = _format_file_info(
            df=bt_seat1_df,
            file_path=bt_seat1_file,
            elapsed_secs=elapsed_bt,
            rss_delta_bytes=rss_delta_bt,
        )
        _update_loading_status(5, "Loading bt_seat1_df...", "bt_seat1_df", bt_seat1_info)
        print(f"[init] bt_seat1_df: {bt_seat1_info} (clean seat-1 data)")
        
        # -------------------------------------------------------------------
        # Build Gemini-3.2 CSR Traversal Index (Invariant C)
        # -------------------------------------------------------------------
        # Since the build takes < 60s, we run it on every startup. This provides 
        # sub-millisecond BT traversal (O(1) jumps) and replaces slow regex scans.
        try:
            _update_loading_status(6, "Building Gemini-3.2 CSR index...")
            t0_g3 = time.perf_counter()
            
            # 1. Vectorized Bid Mapping
            max_bt_index_opt = bt_seat1_df["bt_index"].max()
            if max_bt_index_opt is None:
                raise ValueError("bt_seat1_df is empty or missing bt_index")
            max_idx = int(max_bt_index_opt) # type: ignore
            
            bid_codes_map = np.zeros(int(max_idx) + 1, dtype=np.uint8)
            unique_candidate_bids = bt_seat1_df["candidate_bid"].unique().to_list()
            for bid_val in unique_candidate_bids:
                if not bid_val: continue
                # Match code regardless of case
                code = BID_TO_CODE.get(str(bid_val).upper()) or BID_TO_CODE.get(str(bid_val).lower())
                if code:
                    matching_indices = bt_seat1_df.filter(pl.col("candidate_bid") == bid_val)["bt_index"].to_numpy()
                    bid_codes_map[matching_indices] = code

            # 2. Vectorized CSR Build (Explode next_bid_indices)
            parent_info = (
                bt_seat1_df.filter(pl.col("next_bid_indices").list.len() > 0)
                .select(["bt_index", "next_bid_indices"])
                .explode("next_bid_indices")
                .rename({"bt_index": "p", "next_bid_indices": "c"})
                .sort("p")
            )
            p_indices = parent_info["p"].to_numpy().astype(np.uint32)
            flat_children = parent_info["c"].to_numpy().astype(np.uint32)
            flat_bidcodes = bid_codes_map[flat_children]
            
            unique_p, counts = np.unique(p_indices, return_counts=True)
            degrees = np.zeros(int(max_idx) + 1, dtype=np.uint32)
            degrees[unique_p] = counts.astype(np.uint32)
            
            offsets = np.zeros(int(max_idx) + 2, dtype=np.uint32)
            offsets[1:] = np.cumsum(degrees, dtype=np.uint32)
            
            # 3. Map Openings (Using is_opening_bid column)
            openings = {}
            opening_df = bt_seat1_df.filter(pl.col("is_opening_bid") == True).select(["Auction", "bt_index"])
            for row in opening_df.to_dicts():
                auc = str(row["Auction"]).upper()
                openings[auc] = int(row["bt_index"])
            
            g3_index = G3Index(offsets, flat_children, flat_bidcodes, openings)
            with _STATE_LOCK:
                STATE["g3_index"] = g3_index
            
            elapsed_g3 = time.perf_counter() - t0_g3
            print(f"[init] Gemini-3.2 CSR index built in {elapsed_g3:.1f}s (openings={len(openings)})")
            _update_loading_status(6, "Built Gemini-3.2 CSR index", "g3_index", f"{len(flat_children):,} edges in {elapsed_g3:.1f}s")
            
        except Exception as e:
            print(f"[init] ERROR: Failed to build Gemini-3.2 index: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------------
        # Build/load can_complete sidecar (Reachability to completed auctions)
        # -------------------------------------------------------------------
        try:
            _update_loading_status(7, "Building can_complete sidecar...")
            t0_cc = time.perf_counter()
            max_bt_index_opt = bt_seat1_df["bt_index"].max()
            if max_bt_index_opt is None:
                raise ValueError("bt_seat1_df missing bt_index max; cannot build can_complete")
            max_idx_cc = int(max_bt_index_opt)  # type: ignore[arg-type]
            cc_file = dataPath.joinpath("bt_can_complete_u8.npy")
            cc_arr = _load_or_build_bt_can_complete_sidecar(bt_seat1_file, cc_file, max_idx_cc)
            with _STATE_LOCK:
                STATE["bt_can_complete"] = cc_arr
                STATE["bt_can_complete_file"] = cc_file
            elapsed_cc = time.perf_counter() - t0_cc
            print(f"[init] can_complete loaded/built in {elapsed_cc:.1f}s ({cc_file})")
            _update_loading_status(7, "Built can_complete sidecar", "bt_can_complete", f"{cc_file.name} in {elapsed_cc:.1f}s")
        except Exception as e:
            print(f"[init] ERROR: Failed to build/load can_complete sidecar: {e}")
            traceback.print_exc()

        # Load hot-reloadable criteria overlay (does NOT mutate bt_seat1_df)
        overlay, custom_criteria_stats = _build_custom_criteria_overlay(available_criteria_names)
        _log_memory("after load custom criteria overlay")

        # Load completed-auction stats table (criteria + aggregates) keyed by bt_index.
        # This file is REQUIRED for the Bidding Table Explorer and other tools.
        print(f"[init] Loading bt_stats_df from {bt_criteria_seat1_file}...")
        if not bt_criteria_seat1_file.exists():
            raise FileNotFoundError(f"REQUIRED stats file missing: {bt_criteria_seat1_file}. Pipeline error.")
        
        try:
            t0_stats = time.perf_counter()
            bt_stats_df = pl.read_parquet(bt_criteria_seat1_file)
            elapsed_stats = time.perf_counter() - t0_stats
            bt_stats_info = _format_file_info(df=bt_stats_df, file_path=bt_criteria_seat1_file, elapsed_secs=elapsed_stats)
            print(f"[init] bt_stats_df: {bt_stats_info} (completed auctions with criteria/aggregates)")
            # Track in loaded_files so the UI "Files loaded" list includes stats.
            _update_loading_status(5, "Loading bt_seat1_df and bt_stats_df...", "bt_stats_df", bt_stats_info)
        except Exception as e:
            print(f"[init] ERROR: Failed to load bt_stats_df from {bt_criteria_seat1_file}: {e}")
            raise
        _log_memory("after load bt_stats_df")

        # Load completed-auctions Agg_Expr lookup (optional, for fast wrong-bid-stats)
        # This is a 63MB file with ~975K rows - much faster than querying 46GB Parquet
        bt_completed_agg_df: Optional[pl.DataFrame] = None
        if bt_completed_agg_file.exists():
            try:
                print(f"[init] Loading bt_completed_agg_df from {bt_completed_agg_file}...")
                t0_agg = time.perf_counter()
                bt_completed_agg_df = pl.read_parquet(bt_completed_agg_file)
                elapsed_agg = time.perf_counter() - t0_agg
                agg_info = f"{bt_completed_agg_df.height:,} rows in {elapsed_agg:.1f}s"
                print(f"[init] bt_completed_agg_df: {agg_info} (for fast wrong-bid-stats)")
            except Exception as e:
                print(f"[init] WARNING: Failed to load bt_completed_agg_df: {e} (will use slow fallback)")
                bt_completed_agg_df = None
        else:
            print("[init] bt_completed_agg_df not found (run: create bbo_bt_completed_agg_expr.parquet)")
        _log_memory("after load bt_completed_agg_df")

        # Load bid-category flags (optional, Phase 4 output).
        bt_categories_df = None
        bt_category_cols: list[str] | None = None
        if bt_categories_file.exists():
            try:
                print(f"[init] Loading bt_categories_df from {bt_categories_file} (category flags)...")
                t0_cats = time.perf_counter()
                scan = pl.scan_parquet(bt_categories_file)
                cols = scan.collect_schema().names()
                # Category flags are named like is_Preempt, is_Artificial, ...
                bt_category_cols = sorted([c for c in cols if c.startswith("is_")])
                cols_to_load = (["bt_index"] if "bt_index" in cols else []) + bt_category_cols
                if not cols_to_load:
                    raise ValueError(f"{bt_categories_file} is missing required join key column 'bt_index'")
                bt_categories_df = (
                    scan.select(cols_to_load)
                    .drop_nulls(subset=["bt_index"])
                    .with_columns(pl.col("bt_index").cast(pl.UInt32))
                    .unique(subset=["bt_index"])
                    .sort("bt_index")  # Required for binary search in /bt-categories-by-index
                    .collect()
                )
                elapsed_cats = time.perf_counter() - t0_cats
                cats_info = _format_file_info(
                    df=bt_categories_df,
                    file_path=pathlib.Path(bt_categories_file),
                    elapsed_secs=elapsed_cats,
                )
                _update_loading_status(5, "Loading bt_categories_df...", "bt_categories_df", cats_info)
                print(f"[init] bt_categories_df: {cats_info} ({len(bt_category_cols):,} category flags)")
            except Exception as e:
                print(f"[init] WARNING: Failed to load bt_categories_df: {e}")
                bt_categories_df = None
                bt_category_cols = None
        else:
            print("[init] bt_categories_df not found (optional): bbo_bt_categories.parquet")
        _log_memory("after load bt_categories_df")

        # Load new rules detailed metrics (optional)
        new_rules_df = None
        if new_rules_file.exists():
            try:
                print(f"[init] Loading new rules metrics from {new_rules_file}...")
                t0_new = time.perf_counter()
                new_rules_df = pl.read_parquet(new_rules_file)
                if "step_auction" in new_rules_df.columns:
                    new_rules_df = new_rules_df.with_columns(pl.col("step_auction").str.to_uppercase())
                elapsed_new = time.perf_counter() - t0_new
                new_rules_info = _format_file_info(df=new_rules_df, file_path=new_rules_file, elapsed_secs=elapsed_new)
                # Key by step_auction for fast lookup
                _update_loading_status(5, "Loading new rules metrics...", "new_rules", new_rules_info)
            except Exception as e:
                print(f"[init] WARNING: Failed to load new rules from {new_rules_file}: {e}")
        _log_memory("after load new_rules")

        # Load precomputed BT EV/Par stats (optional, GPU pipeline output)
        # This provides Avg_EV_S{seat} and Avg_Par_S{seat} for each bt_index
        bt_ev_stats_df: Optional[pl.DataFrame] = None
        if bt_ev_stats_file.exists():
            try:
                print(f"[init] Loading BT EV/Par stats from {bt_ev_stats_file}...")
                t0_ev = time.perf_counter()
                bt_ev_stats_df = pl.read_parquet(bt_ev_stats_file)
                elapsed_ev = time.perf_counter() - t0_ev
                ev_info = _format_file_info(df=bt_ev_stats_df, file_path=bt_ev_stats_file, elapsed_secs=elapsed_ev)
                _update_loading_status(5, "Loading BT EV stats...", "bt_ev_stats", ev_info)
                print(f"[init] bt_ev_stats_df: {ev_info} (pre-computed Avg_EV/Avg_Par per seat)")
            except Exception as e:
                print(f"[init] WARNING: Failed to load BT EV stats from {bt_ev_stats_file}: {e}")
                bt_ev_stats_df = None
        else:
            print("[init] No BT EV stats found (run bbo_bt_ev_gpu.py)")
        _log_memory("after load bt_ev_stats")
        
        # Load precomputed deal-to-BT verified index (optional, from GPU pipeline)
        # This enables O(1) lookup of which BT rows match each deal
        deal_to_bt_index_df: Optional[pl.DataFrame] = None
        if deal_to_bt_verified_file.exists():
            try:
                print(f"[init] Loading precomputed deal-to-BT index from {deal_to_bt_verified_file}...")
                t0_idx = time.perf_counter()
                # Load DataFrame and sort by deal_idx for binary search lookups
                deal_to_bt_index_df = (
                    pl.read_parquet(deal_to_bt_verified_file, columns=["deal_idx", "Matched_BT_Indices"])
                    .sort("deal_idx")
                )
                elapsed_idx = time.perf_counter() - t0_idx
                idx_info = f"{deal_to_bt_index_df.height:,} deals in {elapsed_idx:.1f}s"
                _update_loading_status(5, "Loading deal-to-BT index...", "deal_to_bt_index", idx_info)
                print(f"[init] deal_to_bt_index: {idx_info} (GPU-verified, sorted for fast lookup)")
            except Exception as e:
                print(f"[init] WARNING: Failed to load deal-to-BT index from {deal_to_bt_verified_file}: {e}")
                deal_to_bt_index_df = None
        else:
            print("[init] No precomputed deal-to-BT index found (run bbo_bt_filter_by_bitmap.py)")
        _log_memory("after load deal_to_bt_index")

        # Compute opening-bid candidates for all (dealer, seat) combinations
        _update_loading_status(6, "Processing opening bids (seat1-only)...", "bt_openings", "computing...")
        t0_openings = time.perf_counter()
        # NOTE: `bt_parquet_file` is used to load heavy Agg_Expr columns on-demand.
        # Some type-checkers may have stale signatures for mlBridgeBiddingLib; runtime signature supports it.
        results, bt_openings_df = mlBridgeBiddingLib.process_opening_bids_from_bt_seat1(
            deal_df=deal_df,
            bt_seat1_df=bt_seat1_df,
            deal_criteria_by_seat_dfs=deal_criteria_by_seat_dfs,
            bt_parquet_file=bt_seat1_file,  # type: ignore[call-arg]
        )
        elapsed_openings = time.perf_counter() - t0_openings
        bt_openings_info = f"{bt_openings_df.height:,} rows × {bt_openings_df.width} cols in {elapsed_openings:.1f}s"
        _update_loading_status(6, "Preparing to serve...", "bt_openings", bt_openings_info)
        _log_memory("after process_opening_bids")

        # Build fast bt_index lookup array
        print("[init] Building bt_index lookup array...")
        t0_bt_idx = time.perf_counter()
        import numpy as np_local
        bt_index_arr = bt_seat1_df["bt_index"].to_numpy()
        # Avoid np.diff on 461M rows (it can allocate multi-GB temp arrays).
        # Instead, check monotonicity in chunks with small, bounded temporaries.
        def _is_monotonic_non_decreasing_uint32(arr: Any, chunk: int = 5_000_000) -> bool:
            n = len(arr)
            if n <= 1:
                return True
            prev = int(arr[0])
            # Compare adjacent chunks: arr[i0:i1] >= arr[i0-1:i1-1]
            for i0 in range(1, n, chunk):
                i1 = min(n, i0 + chunk)
                # First element in chunk must be >= prev
                if int(arr[i0]) < prev:
                    return False
                # Check within the chunk
                a = arr[i0:i1]
                b = arr[i0 - 1 : i1 - 1]
                if not bool((a >= b).all()):
                    return False
                prev = int(arr[i1 - 1])
            return True

        bt_index_monotonic = _is_monotonic_non_decreasing_uint32(bt_index_arr)
        # If bt_index isn't monotonic in the source file, DO NOT sort bt_seat1_df (461M rows!).
        # Instead, build a sorted bt_index array + a row-position permutation so we can still do
        # O(log n) lookups via np.searchsorted.
        bt_index_sorted_arr = bt_index_arr
        bt_index_sorted_pos: Any = None
        if not bt_index_monotonic:
            print("[init] bt_index is NOT monotonic; building sorted index permutation (one-time cost)...")
            t0_sort = time.perf_counter()
            # argsort returns int64; cast to uint32 to save memory (461M fits in uint32)
            order = np_local.argsort(bt_index_arr, kind="mergesort")
            bt_index_sorted_arr = bt_index_arr[order]
            bt_index_sorted_pos = order.astype(np_local.uint32, copy=False)
            elapsed_sort = time.perf_counter() - t0_sort
            print(f"[init] bt_index permutation built in {elapsed_sort:.1f}s")
        elapsed_bt_idx = (time.perf_counter() - t0_bt_idx) * 1000
        print(f"[init] bt_index lookup arrays built in {elapsed_bt_idx:.1f}ms (monotonic={bt_index_monotonic})")

        with _STATE_LOCK:
            STATE["deal_df"] = deal_df
            STATE["bt_seat1_df"] = bt_seat1_df
            STATE["bt_seat1_file"] = bt_seat1_file  # For on-demand Agg_Expr loading in handlers
            STATE["bt_openings_df"] = bt_openings_df
            STATE["deal_criteria_by_seat_dfs"] = deal_criteria_by_seat_dfs
            STATE["deal_criteria_by_direction_dfs"] = deal_criteria_by_direction_dfs
            STATE["results"] = results
            STATE["bt_stats_df"] = bt_stats_df
            STATE["bt_completed_agg_df"] = bt_completed_agg_df  # For fast wrong-bid-stats (63MB in-memory)
            STATE["bt_categories_df"] = bt_categories_df
            STATE["bt_category_cols"] = bt_category_cols
            STATE["bt_index_arr"] = bt_index_arr
            STATE["bt_index_monotonic"] = bt_index_monotonic
            STATE["bt_index_sorted_arr"] = bt_index_sorted_arr
            STATE["bt_index_sorted_pos"] = bt_index_sorted_pos
            STATE["custom_criteria_overlay"] = overlay
            STATE["custom_criteria_stats"] = custom_criteria_stats
            STATE["available_criteria_names"] = available_criteria_names
            STATE["new_rules_df"] = new_rules_df
            STATE["deal_to_bt_index_df"] = deal_to_bt_index_df  # Precomputed deal→[bt_indices] DataFrame (or None)
            STATE["bt_ev_stats_df"] = bt_ev_stats_df  # Precomputed Avg_EV/Avg_Par per bt_index per seat (or None)
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


@app.post("/new-rules-lookup")
def new_rules_lookup(req: NewRulesLookupRequest) -> Dict[str, Any]:
    """Look up detailed new rules metrics for a specific auction sequence."""
    state, reload_info, handler = _prepare_handler_call()
    try:
        resp = handler.handle_new_rules_lookup(
            state=state,
            auction=req.auction,
            bt_index=req.bt_index,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("new-rules-lookup", e)


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
            prewarm_progress=STATE.get("prewarm_progress"),
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
    
    # Hybrid matching: literal prefix for simple patterns, regex for complex ones
    if is_regex_pattern(partial):
        regex_pat = partial if partial.startswith('^') else f'^{partial}'
        matches = bt_seat1_df.filter(_normalize_auction_expr().str.contains(f'(?i){regex_pat}'))
    else:
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
            completed_only=req.completed_only,
            partial_only=req.partial_only,
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
    # Allow callers to pass the user-facing deal `index` instead of deal_row_idx.
    # This is fast iff STATE["deal_index_monotonic"] is True.
    deal_row_idx = int(req.deal_row_idx)
    if req.deal_index is not None:
        deal_row_idx = _resolve_deal_row_idx_from_index(state, int(req.deal_index))
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_deal_criteria_failures_batch(
            state=state,
            deal_row_idx=deal_row_idx,
            dealer=str(req.dealer),
            checks=[c.model_dump() for c in req.checks],
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("deal-criteria-eval-batch", e)


@app.post("/deals-by-index")
def deals_by_index(req: DealsByIndexRequest) -> Dict[str, Any]:
    """Fetch deal rows by user-facing `index` values using monotonic fast-path."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    deal_df = state.get("deal_df")
    if not isinstance(deal_df, pl.DataFrame):
        raise HTTPException(status_code=500, detail="deal_df not loaded")

    # Safety caps
    want = [int(x) for x in (req.indices or []) if x is not None]
    want = want[: max(0, int(req.max_rows or 0))]
    if not want:
        return _attach_hot_reload_info({"rows": [], "count": 0}, reload_info)

    row_positions = _resolve_deal_row_indices_from_indices(state, want)
    if not row_positions:
        return _attach_hot_reload_info({"rows": [], "count": 0}, reload_info)

    # Select columns (include HCP/Total_Points for criteria evaluation)
    default_cols = [
        "index", "Dealer", "Vul",
        "Hand_N", "Hand_E", "Hand_S", "Hand_W",
        "HCP_N", "HCP_E", "HCP_S", "HCP_W",
        "Total_Points_N", "Total_Points_E", "Total_Points_S", "Total_Points_W",
        "ParScore", "Contract", "Score", "Result", "bid",
    ]
    cols = req.columns or default_cols
    cols = [str(c) for c in cols if c]  # normalize
    # Columns present in the in-memory deal_df
    cols_in_mem = [c for c in cols if c in deal_df.columns]
    if not cols_in_mem:
        cols_in_mem = ["index"] if "index" in deal_df.columns else [deal_df.columns[0]]
    # Columns requested but not loaded into memory (e.g. heavy Probs_*).
    cols_missing = [c for c in cols if c not in deal_df.columns]

    # Take rows by row position, then include _row_idx for downstream callers.
    out_df = deal_df.select(cols_in_mem)
    out_df = out_df.with_row_index("_row_idx")
    from plugins.bbo_handlers_common import take_rows_by_index as _take  # avoid polars version issues
    out_df = _take(out_df, row_positions)

    # If callers requested columns that are not loaded into deal_df, fetch them directly
    # from the source parquet file for just these row positions.
    #
    # This is critical for "wide" columns like Probs_* which are too large to keep in RAM
    # but are needed for per-deal computations (e.g. doubled/redoubled EV).
    if cols_missing:
        # Hard requirement: deals parquet must exist (no fallback logic).
        if not bbo_mldf_augmented_file.exists():
            raise HTTPException(status_code=500, detail=f"deals parquet not found: {bbo_mldf_augmented_file}")

        # Only fetch missing columns that actually exist in the parquet schema.
        try:
            pf = pq.ParquetFile(str(bbo_mldf_augmented_file))
            schema_names = set(pf.schema.names)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to open deals parquet: {e}")

        cols_missing = [c for c in cols_missing if c in schema_names]

        if cols_missing:
            # Map absolute row position -> rank in requested output order
            pos_rank = {int(pos): i for i, pos in enumerate(row_positions)}

            # Precompute row group start offsets to map positions -> row groups.
            rg_starts: list[int] = []
            total = 0
            try:
                for rg_i in range(pf.num_row_groups):
                    rg_starts.append(total)
                    total += int(pf.metadata.row_group(rg_i).num_rows)  # type: ignore[union-attr]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read parquet row group metadata: {e}")

            import bisect as _bisect
            by_rg: dict[int, list[int]] = {}
            for pos in row_positions:
                p = int(pos)
                # rightmost start <= p
                rg_i = int(_bisect.bisect_right(rg_starts, p) - 1)
                if rg_i < 0:
                    continue
                by_rg.setdefault(rg_i, []).append(p)

            parts: list[pl.DataFrame] = []
            for rg_i, poss in by_rg.items():
                start = rg_starts[rg_i]
                try:
                    tbl = pf.read_row_group(int(rg_i), columns=cols_missing)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to read row group {rg_i}: {e}")
                rg_df_any = pl.from_arrow(tbl)
                rg_df: pl.DataFrame
                if isinstance(rg_df_any, pl.Series):
                    rg_df = rg_df_any.to_frame()
                else:
                    rg_df = rg_df_any
                for p in poss:
                    off = int(p - start)
                    if off < 0 or off >= rg_df.height:
                        continue
                    parts.append(
                        rg_df.slice(off, 1).with_columns(
                            pl.lit(int(p)).alias("_row_idx"),
                            pl.lit(int(pos_rank.get(int(p), 0))).alias("_rank"),
                        )
                    )

            if parts:
                extra_df = pl.concat(parts, how="vertical").sort("_rank").drop("_rank")
                # Join on _row_idx (safe, unique for these rows)
                out_df = out_df.join(extra_df, on="_row_idx", how="left")

    rows = out_df.to_dicts()
    resp = {"rows": rows, "count": len(rows)}
    return _attach_hot_reload_info(resp, reload_info)


@app.post("/bt-categories-by-index")
def bt_categories_by_index(req: BTCategoriesByIndexRequest) -> Dict[str, Any]:
    """Fetch bid-category flags (Phase 4) for bt_index values.

    Returns `categories_true` as a list of category names (without the `is_` prefix).
    """
    reload_info = _reload_plugins()
    _ensure_ready()
    t0 = time.perf_counter()

    with _STATE_LOCK:
        state = dict(STATE)

    bt_categories_df: pl.DataFrame | None = state.get("bt_categories_df")
    bt_category_cols: list[str] = state.get("bt_category_cols") or []
    if bt_categories_df is None or not bt_category_cols:
        raise HTTPException(
            status_code=400,
            detail="bt_categories_df not loaded (generate bbo_bt_categories.parquet via bbo_bt_classify_bids.py)",
        )

    indices = [int(x) for x in (req.indices or []) if x is not None]
    indices = indices[: max(0, int(req.max_rows or 0))]
    if not indices:
        return _attach_hot_reload_info({"rows": [], "missing_indices": [], "elapsed_ms": 0.0}, reload_info)

    try:
        # Optimization: Use binary search for fast bt_index lookup in giant 461M-row DF
        import numpy as np_local
        cat_idx_arr = bt_categories_df["bt_index"].to_numpy()
        
        rows_by_idx: dict[int, dict[str, Any]] = {}
        for idx_val in indices:
            pos = int(np_local.searchsorted(cat_idx_arr, int(idx_val)))
            if 0 <= pos < len(cat_idx_arr):
                val_at_pos = cat_idx_arr[pos]
                # Handle NaN/None (Polars/NumPy bridge sometimes produces floats with NaNs)
                try:
                    if not np_local.isnan(val_at_pos) and int(val_at_pos) == int(idx_val):
                        # Found it! Pluck categories efficiently.
                        cats_true: list[str] = []
                        for c in bt_category_cols:
                            # Direct access by row position is much faster than filter()
                            if bool(bt_categories_df[c][pos]):
                                cats_true.append(c[3:] if c.startswith("is_") else c)
                        rows_by_idx[int(idx_val)] = {"bt_index": int(idx_val), "categories_true": cats_true}
                except (ValueError, TypeError):
                    continue

        out_rows: list[dict[str, Any]] = []
        missing: list[int] = []
        for idx in indices:
            rr = rows_by_idx.get(int(idx))
            if rr is None:
                missing.append(int(idx))
            else:
                out_rows.append(rr)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        out = {"rows": out_rows, "missing_indices": missing, "elapsed_ms": round(elapsed_ms, 1)}
        return _attach_hot_reload_info(out, reload_info)
    except Exception as e:
        _log_and_raise("bt-categories-by-index", e)


@app.post("/resolve-auction-path")
def resolve_auction_path(req: ResolveAuctionPathRequest) -> Dict[str, Any]:
    """Resolve a full auction string - delegated to hot-reloadable handler."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_resolve_auction_path(
            state=state,
            auction=req.auction,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("resolve-auction-path", e)


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


@app.post("/sample-deals-by-auction-pattern")
def sample_deals_by_auction_pattern(req: SampleDealsByAuctionPatternRequest) -> Dict[str, Any]:
    """Fast sampling of deals by actual-auction regex (no BT / Rules)."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_sample_deals_by_auction_pattern(
            state=state,
            pattern=req.pattern,
            sample_size=int(req.sample_size),
            seed=req.seed,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("sample-deals-by-auction-pattern", e)


@app.post("/auction-pattern-counts")
def auction_pattern_counts(req: AuctionPatternCountsRequest) -> Dict[str, Any]:
    """Batch counts for actual-auction regex patterns (no BT / Rules)."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_auction_pattern_counts(
            state=state,
            patterns=list(req.patterns),
        )
        return _attach_hot_reload_info(resp, reload_info)
    except Exception as e:
        _log_and_raise("auction-pattern-counts", e)


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
            include_categories=req.include_categories,
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
            # Hybrid matching: literal prefix for simple patterns, regex for complex ones
            if pattern_matches(partial_auction, auction_norm):
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

    if not rows_to_keep:
        # IMPORTANT: If the keep-list is empty, Polars will infer Null dtype for the join key,
        # causing `u64` vs `null` join failures. Return an empty frame with the same schema.
        return df.head(0), rejected_info
    
    # Select rows in a type-checker-friendly way (and preserve order).
    idx_df = pl.DataFrame(
        {
            "_i": pl.Series(rows_to_keep, dtype=pl.UInt64),
            "_pos": pl.Series(list(range(len(rows_to_keep))), dtype=pl.UInt32),
        }
    )
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
        print(f"[execute-sql] {result.height} rows in {format_elapsed(elapsed_ms)}")
        
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


@app.post("/list-next-bids")
def list_next_bids(req: ListNextBidsRequest) -> Dict[str, Any]:
    """Fast lookup of available next bids using BT's next_bid_indices.
    
    Given an auction prefix (or empty for opening bids), returns all valid next bids
    with their bt_index and Agg_Expr criteria. Uses the sorted BT structure for
    efficient O(log n) lookups instead of regex scanning.
    """
    reload_info = _reload_plugins()
    _ensure_ready()
    
    with _STATE_LOCK:
        state = dict(STATE)
    
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_list_next_bids(
            state=state,
            auction=req.auction,
        )
        if reload_info:
            resp["_reload_info"] = reload_info
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("list-next-bids", e)


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


@app.post("/best-auctions-lookahead")
def best_auctions_lookahead(req: BestAuctionsLookaheadRequest) -> Dict[str, Any]:
    """Server-side DFS to find best completed auctions by DD or EV.
    
    Uses CSR index for O(1) next-bid traversal and bitmap DFs for O(1) criteria eval.
    Single request replaces dozens of client-side API calls.
    """
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)
    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")
        resp = handler_module.handle_best_auctions_lookahead(
            state=state,
            deal_row_idx=req.deal_row_idx,
            auction_prefix=req.auction_prefix,
            metric=req.metric,
            max_depth=req.max_depth,
            max_results=req.max_results,
            permissive_pass=req.permissive_pass,
        )
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("best-auctions-lookahead", e)


@app.post("/best-auctions-lookahead/start")
def best_auctions_lookahead_start(req: BestAuctionsLookaheadStartRequest) -> Dict[str, Any]:
    """Start a best-auctions search asynchronously (avoids client read timeouts)."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)

    handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
    if not handler_module:
        raise HTTPException(status_code=500, detail="Plugin 'bbo_bidding_queries_api_handlers' not found")

    job_id = str(uuid.uuid4())
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "running",  # running|completed|failed
        "created_at_s": time.time(),
        "finished_at_s": None,
        "request": req.model_dump(),
        "result": None,
        "error": None,
    }

    _best_auctions_jobs_gc(job["created_at_s"])
    with _BEST_AUCTIONS_JOBS_LOCK:
        _BEST_AUCTIONS_JOBS[job_id] = job

    def _run_job() -> None:
        try:
            t_wall0 = time.perf_counter()
            t_cpu0 = time.process_time()
            # Delegate to hot-reloadable handler
            resp = handler_module.handle_best_auctions_lookahead(
                state=state,
                deal_row_idx=int(req.deal_row_idx),
                auction_prefix=str(req.auction_prefix or ""),
                metric=str(req.metric or "DD"),
                max_depth=int(req.max_depth),
                max_results=int(req.max_results),
                deadline_s=float(req.deadline_s),
                max_nodes=int(req.max_nodes),
                beam_width=int(req.beam_width),
                permissive_pass=bool(req.permissive_pass),
            )
            t_wall1 = time.perf_counter()
            t_cpu1 = time.process_time()
            with _BEST_AUCTIONS_JOBS_LOCK:
                j = _BEST_AUCTIONS_JOBS.get(job_id)
                if j is not None:
                    j["status"] = "completed"
                    j["result"] = resp
                    j["wall_elapsed_s"] = round(t_wall1 - t_wall0, 3)
                    j["cpu_elapsed_s"] = round(t_cpu1 - t_cpu0, 3)
                    j["finished_at_s"] = time.time()
        except Exception as e:
            with _BEST_AUCTIONS_JOBS_LOCK:
                j = _BEST_AUCTIONS_JOBS.get(job_id)
                if j is not None:
                    j["status"] = "failed"
                    j["error"] = f"{e}"
                    j["finished_at_s"] = time.time()

    _BEST_AUCTIONS_EXECUTOR.submit(_run_job)
    return _attach_hot_reload_info({"job_id": job_id, "status": "running"}, reload_info)


@app.get("/best-auctions-lookahead/status/{job_id}")
def best_auctions_lookahead_status(job_id: str) -> Dict[str, Any]:
    """Poll status/results for an async best-auctions job."""
    reload_info = _reload_plugins()
    _ensure_ready()
    _best_auctions_jobs_gc()
    with _BEST_AUCTIONS_JOBS_LOCK:
        job = _BEST_AUCTIONS_JOBS.get(str(job_id))
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        # Shallow copy so we don't leak executor internals / allow mutation
        out = dict(job)
    return _attach_hot_reload_info(out, reload_info)


@app.post("/deal-matched-bt-sample")
def deal_matched_bt_sample(req: DealMatchedBTSampleRequest) -> Dict[str, Any]:
    """Return a random sample of BT rows that match a pinned deal (GPU-verified index)."""
    reload_info = _reload_plugins()
    _ensure_ready()
    with _STATE_LOCK:
        state = dict(STATE)

    # Allow callers to pass the user-facing deal `index` instead of deal_row_idx.
    deal_row_idx = int(req.deal_row_idx)
    if req.deal_index is not None:
        deal_row_idx = _resolve_deal_row_idx_from_index(state, int(req.deal_index))

    try:
        handler_module = PLUGINS.get("bbo_bidding_queries_api_handlers")
        if not handler_module:
            raise ImportError("Plugin 'bbo_bidding_queries_api_handlers' not found")

        resp = handler_module.handle_deal_matched_bt_sample(
            state=state,
            deal_row_idx=deal_row_idx,
            n_samples=int(req.n_samples),
            seed=int(req.seed or 0),
            metric=str(req.metric or "DD"),
            permissive_pass=bool(req.permissive_pass),
        )
        return _attach_hot_reload_info(resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log_and_raise("deal-matched-bt-sample", e)


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
        default=1000000,
        help="Limit deal_df rows for faster startup (default: 1000000, use 0 for all rows)",
    )
    args = parser.parse_args()

    print("Starting API server...")
    try:
        source_file = pathlib.Path(__file__)
        source_mtime = source_file.stat().st_mtime
        source_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(source_mtime))
        print(f"[init] Source file: {source_file.name} (modified: {source_date})")
    except Exception:
        pass
    # Disable uvicorn's built-in access log; we emit our own line with elapsed seconds.
    uvicorn.run(app, host=args.host, port=args.port, reload=False, access_log=False)
