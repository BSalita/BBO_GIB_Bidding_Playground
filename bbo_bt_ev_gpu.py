"""
BT EV/Par Statistics + Hand Stats + DD Means - GPU Accelerated (NV/V Split)

Installation: (has been problematic)
  pip uninstall torch torchvision
  pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128

Estimated runtime: ~12-16 hours on RTX 5080 with 16GB VRAM. ~100-115GB peak RAM.

Single-pass architecture (2026-02-10, optimized from v2 two-pass):
  8 vul-split cubes, 47 ops/batch — EV + hand stats + DD means in one traversal.
  NV+V stats are summed in the combination step.  Combined cubes eliminated.
  Cube tensors uploaded to GPU once per (direction, vul) and reused across all
  BT-row chunks — eliminates hundreds of redundant CPU→GPU transfers.
  Batch size 4000 (up from 1000) for better GPU utilization.

Memory optimization (2026-02-07):
  Processes one cube at a time instead of all 8 simultaneously.
  With 461M BT rows, this reduces peak RAM from ~1.1 TB to ~96 GB per task
  (~100-115 GB total including bounds cache + cubes + overhead).
  dd_sums use int32 (max value ≈ 208M, fits int32) saving ~37 GB per task.
  dim_sums use int32 and min/max use uint8 to further reduce RAM + checkpoint IO.

Computes per-BT-row aggregate statistics using GPU-accelerated histogram cube
matching. Consolidates work previously done by bbo_bt_compute_stats.py and
bbo_build_cube.py into a single pipeline.

Features:
- GPU acceleration via PyTorch (RTX 5080)
- Histogram cube for efficient aggregation (16M deals → ~7K buckets per vul)
- Vulnerability split: separate NV and V statistics for EV/Par/Count per seat
- Hand statistics: mean/std/min/max for HCP, SL, Total_Points per seat (not vul-split)
- DD trick means: average DD tricks per declarer per strain per seat (seat-relative, not vul-split)
- Total_Points criteria matching (6th cube dimension, deterministic from HCP+SL)
- Progress indicators (tqdm)
- Restartable via checkpoints
- Memory-efficient chunked processing

Expected runtime: ~12-16 hours (single merged pass on 8 vul-split cubes)

Output Schema (~205 columns):
- bt_index: BT row identifier
- Count_S{1-4}_NV, Count_S{1-4}_V: Deal counts per seat, split by vulnerability
- Avg_Par_S{1-4}_NV, Avg_Par_S{1-4}_V: Average ParScore per seat, split by vulnerability
- Avg_EV_S{1-4}_NV, Avg_EV_S{1-4}_V: Average EV per seat, split by vulnerability
- matching_deal_count_S{1-4}: Combined deal count per seat (NV+V, all directions)
- {HCP,SL_C,SL_D,SL_H,SL_S,Total_Points}_{mean,std}_S{1-4}: Hand stat means/stds (Float32)
- {HCP,SL_C,SL_D,SL_H,SL_S,Total_Points}_{min,max}_S{1-4}: Hand stat min/max (UInt8)
- DD_S{1-4}_{C,D,H,S,N}_mean_S{1-4}: Mean DD tricks per declarer-seat/strain (Float32, seat-relative)

Output Files:
- bt_ev_par_stats_gpu_v3.parquet: Per-BT stats with NV/V splits + hand stats + seat-relative DD means

Usage:
    python bbo_bt_ev_gpu.py                           # Full pipeline (~12-16 hours)
    python bbo_bt_ev_gpu.py --resume                  # Resume from checkpoint
    python bbo_bt_ev_gpu.py --max-bt-rows 1000000     # Test run
    python bbo_bt_ev_gpu.py --build-index             # DEPRECATED/disabled (ignored)

-------------------------------------------------------------------------------
INVERTED INDEX: DEPRECATED - To be removed in future version
-------------------------------------------------------------------------------

Status: DEPRECATED (2026-01-16)
    The --build-index feature is deprecated and will be removed in a future version.
    The code remains for reference but should not be used in production.

Original Design Goal:
    The inverted index was designed to answer "given a deal's hand characteristics,
    find all BT rows whose criteria match." It maps histogram buckets (grouped by
    HCP + suit lengths) to lists of matching bt_index values.

Why It Failed:
    1. MASSIVE OUTPUT SIZE
       - ~220M BT rows × ~1000 matching buckets each = ~200 billion pairs
       - At 6 bytes per pair = 1+ TB of raw data
       - With 8 cubes (4 directions × 2 vul states), this multiplies further

    2. DISK I/O BOTTLENECK  
       - Streaming billions of pairs to disk during GPU processing is very slow
       - Consolidation step (grouping by bucket) adds hours more

    3. SUPERSEDED BY BETTER SOLUTION
       - The primary use case (deal → matching BT rows) is already solved by
         bbo_deal_to_bt_verified.parquet, which was built by a different pipeline
         (bbo_bt_deal_matches_gpu.py + bbo_bt_filter_by_bitmap.py)
       - That pipeline uses exact auction matching, which is more accurate

What To Use Instead:
    bbo_deal_to_bt_verified.parquet provides O(1) lookup of deal_idx → bt_indices
    for deals that actually occurred in the dataset. This is loaded by the API
    server at startup and used for all deal-to-BT queries.

Summary:
    The stats (Avg_EV, Avg_Par, Count per seat/vul) are the primary value of this
    pipeline. The inverted index code will be removed once confirmed unused.
-------------------------------------------------------------------------------
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl

# Configure Polars threads
import multiprocessing
_num_cores = multiprocessing.cpu_count()
_polars_threads = min(16, max(1, _num_cores - 4))
os.environ["POLARS_MAX_THREADS"] = str(_polars_threads)

try:
    import torch  # type: ignore[import-not-found]
    HAS_TORCH = True
except ImportError:
    torch: Any = None
    HAS_TORCH = False
    print("WARNING: PyTorch not available. GPU acceleration disabled.")

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
    HAS_TQDM = True
except ImportError:
    tqdm: Any = None
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = ["N", "E", "S", "W"]
METRICS = ["HCP", "SL_S", "SL_H", "SL_D", "SL_C", "Total_Points"]  # 6 dimensions for cube
CHECKPOINT_INTERVAL = 1_000_000  # Save every 1M BT rows
BATCH_SIZE = 4_000  # BT rows per GPU batch — larger batches reduce per-batch Python/CUDA overhead
BT_CHUNK_SIZE = 5_000_000  # Process BT in 5M row chunks to avoid memory explosion
BOUNDS_VALIDATE_ROWS = 2_000  # Fail-fast sample check for bounds extraction correctness

# DD trick columns: 4 declarers x 5 strains = 20 columns
DD_DECLARERS = ["N", "E", "S", "W"]
DD_STRAINS = ["C", "D", "H", "S", "N"]  # N = NoTrump
DD_COLS = [f"DD_{decl}_{strain}" for decl in DD_DECLARERS for strain in DD_STRAINS]

# Default paths
DEFAULT_DEALS_FILE = Path("E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet")
DEFAULT_BT_FILE = Path("E:/bridge/data/bbo/bidding/bbo_bt_seat1.parquet")
DEFAULT_OUTPUT_FILE = Path("E:/bridge/data/bbo/bidding/bt_ev_par_stats_gpu_v3.parquet")
DEFAULT_CHECKPOINT_DIR = Path("E:/bridge/data/bbo/bidding/checkpoints_ev_gpu")
DEFAULT_INDEX_DIR = Path("E:/bridge/data/bbo/bidding/inverted_index")

# Pipeline identity + required checkpoint contents (resume safety)
PIPELINE_VERSION = "ev_gpu_v3_single_pass_opt_2026-02-11"
REQUIRED_VUL_RESULT_KEYS = {
    "bt_indices",
    "counts",
    "par_sums",
    "ev_sums",
    "dim_sums",
    "dim_sq_sums",
    "dim_mins",
    "dim_maxs",
    "dd_sums",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def fmt_datetime() -> str:
    """Current datetime as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """Print with timestamp."""
    print(f"[{fmt_datetime()}] {msg}", flush=True)


class ProgressBar:
    """Progress bar wrapper that works with or without tqdm."""
    
    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        
        if HAS_TQDM:
            self.pbar = tqdm(total=total, desc=desc, unit=unit, ncols=100)
        else:
            self.pbar = None
            self._last_pct = -1
    
    def update(self, n: int = 1) -> None:
        self.current += n
        if self.pbar:
            self.pbar.update(n)
        else:
            pct = int(100 * self.current / self.total) if self.total > 0 else 0
            if pct >= self._last_pct + 5:
                elapsed = time.time() - self.start_time
                rate = self.current / elapsed if elapsed > 0 else 0
                eta = (self.total - self.current) / rate if rate > 0 else 0
                print(f"  {self.desc}: {pct}% ({self.current:,}/{self.total:,}) "
                      f"| {rate:,.0f} {self.unit}/s | ETA {fmt_time(eta)}", flush=True)
                self._last_pct = pct
    
    def close(self) -> None:
        if self.pbar:
            self.pbar.close()
        else:
            elapsed = time.time() - self.start_time
            print(f"  {self.desc}: 100% ({self.total:,}) in {fmt_time(elapsed)}", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages checkpoints for restartability."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = checkpoint_dir / "state.json"
        self.results_dir = checkpoint_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def load_state(self) -> Optional[Dict]:
        """Load checkpoint state if exists."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return None
    
    def save_state(self, state: Dict) -> None:
        """Save checkpoint state."""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def results_file_vul(self, direction: str, vul: str, seat: int) -> Path:
        return self.results_dir / f"results_{direction}_{vul}_S{seat}.npz"

    def peek_results_vul_keys(self, direction: str, vul: str, seat: int) -> Optional[set[str]]:
        """Return key set in a vul checkpoint without loading arrays."""
        filename = self.results_file_vul(direction, vul, seat)
        if not filename.exists():
            return None
        try:
            with np.load(filename) as data:
                return set(data.files)
        except Exception as e:
            log(f"  WARNING: Failed to read checkpoint keys: {filename} ({e})")
            return None

    def has_results_vul(
        self,
        direction: str,
        vul: str,
        seat: int,
        required_keys: set[str] = REQUIRED_VUL_RESULT_KEYS,
    ) -> bool:
        keys = self.peek_results_vul_keys(direction, vul, seat)
        return keys is not None and required_keys.issubset(keys)
    
    def save_results(self, direction: str, seat: int, bt_indices: np.ndarray,
                     counts: np.ndarray, par_sums: np.ndarray, ev_sums: np.ndarray) -> None:
        """Save intermediate results for a direction/seat."""
        filename = self.results_dir / f"results_{direction}_S{seat}.npz"
        np.savez_compressed(filename, 
                           bt_indices=bt_indices,
                           counts=counts,
                           par_sums=par_sums,
                           ev_sums=ev_sums)
    
    def load_results(self, direction: str, seat: int) -> Optional[Dict]:
        """Load intermediate results if exists."""
        filename = self.results_dir / f"results_{direction}_S{seat}.npz"
        if filename.exists():
            data = np.load(filename)
            return {k: data[k] for k in data.files}
        return None
    
    def save_results_vul(self, direction: str, vul: str, seat: int, bt_indices: np.ndarray,
                         counts: np.ndarray, par_sums: np.ndarray, ev_sums: np.ndarray,
                         dim_sums: Optional[np.ndarray] = None,
                         dim_sq_sums: Optional[np.ndarray] = None,
                         dim_mins: Optional[np.ndarray] = None,
                         dim_maxs: Optional[np.ndarray] = None,
                         dd_sums: Optional[np.ndarray] = None) -> None:
        """Save intermediate results for a direction/vul/seat.
        
        Extended arrays (all optional for backward compat):
            dim_sums: (6, n_bt) int32 — weighted sums of 6 cube dimensions
            dim_sq_sums: (6, n_bt) int64 — weighted sums of squared dimensions
            dim_mins: (6, n_bt) uint8 — min of each dimension (255 sentinel = no data)
            dim_maxs: (6, n_bt) uint8 — max of each dimension (0 sentinel = no data)
            dd_sums: (20, n_bt) int32 — sums of DD trick values
        """
        filename = self.results_dir / f"results_{direction}_{vul}_S{seat}.npz"
        save_dict = dict(
            bt_indices=bt_indices, counts=counts,
            par_sums=par_sums, ev_sums=ev_sums,
        )
        if dim_sums is not None:
            save_dict["dim_sums"] = dim_sums
        if dim_sq_sums is not None:
            save_dict["dim_sq_sums"] = dim_sq_sums
        if dim_mins is not None:
            save_dict["dim_mins"] = dim_mins
        if dim_maxs is not None:
            save_dict["dim_maxs"] = dim_maxs
        if dd_sums is not None:
            save_dict["dd_sums"] = dd_sums
        np.savez_compressed(filename, **save_dict)  # type: ignore[arg-type]
    
    def load_results_vul(self, direction: str, vul: str, seat: int) -> Optional[Dict]:
        """Load intermediate results if exists (with vul state)."""
        filename = self.results_dir / f"results_{direction}_{vul}_S{seat}.npz"
        if filename.exists():
            data = np.load(filename)
            return {k: data[k] for k in data.files}
        return None
    
    def save_results_combined(self, direction: str, seat: int, bt_indices: np.ndarray,
                              counts: np.ndarray,
                              dim_sums: np.ndarray, dim_sq_sums: np.ndarray,
                              dim_mins: np.ndarray, dim_maxs: np.ndarray,
                              dd_sums: np.ndarray) -> None:
        """Save combined (non-vul-split) stats for a direction/seat."""
        filename = self.results_dir / f"stats_combined_{direction}_S{seat}.npz"
        np.savez_compressed(filename,
                            bt_indices=bt_indices, counts=counts,
                            dim_sums=dim_sums, dim_sq_sums=dim_sq_sums,
                            dim_mins=dim_mins, dim_maxs=dim_maxs,
                            dd_sums=dd_sums)
    
    def load_results_combined(self, direction: str, seat: int) -> Optional[Dict]:
        """Load combined stats if exists."""
        filename = self.results_dir / f"stats_combined_{direction}_S{seat}.npz"
        if filename.exists():
            data = np.load(filename)
            return {k: data[k] for k in data.files}
        return None
    
    def clear(self) -> None:
        """Clear all checkpoints."""
        if self.checkpoint_dir.exists():
            try:
                shutil.rmtree(self.checkpoint_dir)
            except PermissionError as e:
                log(f"  WARNING: Could not clear checkpoint dir: {e}")
                log(f"  Attempting to remove files individually...")
                # Try to remove files individually
                for f in self.checkpoint_dir.rglob("*"):
                    try:
                        if f.is_file():
                            f.unlink()
                    except PermissionError:
                        pass
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Histogram Cube Builder
# ---------------------------------------------------------------------------

def _compute_dp(sl: int) -> int:
    """Compute distribution points for a single suit length."""
    if sl == 0:
        return 3
    elif sl == 1:
        return 2
    elif sl == 2:
        return 1
    return 0


def build_histogram_cube(
    deals_file: Path,
    direction: str,
    max_deals: Optional[int] = None,
    vul_filter: Optional[str] = None,  # "NV" or "V" - filters by vulnerability for this direction's side
) -> pl.DataFrame:
    """
    Build histogram cube for a direction.
    
    Groups deals by (HCP, SL_S, SL_H, SL_D, SL_C, Total_Points) and pre-aggregates
    count, ParScore sum, EV sum, and DD trick sums.
    
    Total_Points is the 6th cube dimension. Since TP = HCP + distribution_points
    and DP is deterministic from suit lengths, adding TP as a GROUP BY column
    adds zero new buckets. This enables TP criteria matching (previously silently
    ignored) and TP statistics computation.
    
    DD trick sums (20 columns) are aggregated per bucket for computing mean DD
    tricks per BT node in the GPU kernel.
    
    Args:
        deals_file: Path to deals parquet file
        direction: N, E, S, or W
        max_deals: Optional limit on deals to process
        vul_filter: Optional "NV" or "V" to filter by vulnerability for this direction's side
                   - N/S directions: NV = Vul NOT IN ('N_S', 'Both'), V = Vul IN ('N_S', 'Both')
                   - E/W directions: NV = Vul NOT IN ('E_W', 'Both'), V = Vul IN ('E_W', 'Both')
    
    Returns DataFrame with ~200K-500K unique buckets (same as before since TP is deterministic).
    """
    t0 = time.time()
    
    # 6 dimension columns (including Total_Points)
    dim_cols = [f"HCP_{direction}", f"SL_{direction}_S", f"SL_{direction}_H",
                f"SL_{direction}_D", f"SL_{direction}_C"]
    tp_col = f"Total_Points_{direction}"
    
    # Check if columns exist
    schema = pl.scan_parquet(deals_file).collect_schema()
    available_cols = schema.names()
    
    # Total_Points: load from parquet if available, otherwise compute from HCP + DP
    has_tp = tp_col in available_cols
    
    # Use available ParScore/EV columns
    par_col = "ParScore" if "ParScore" in available_cols else None
    ev_col = "EV_Score_Declarer" if "EV_Score_Declarer" in available_cols else None
    
    if par_col is None:
        log(f"  WARNING: ParScore column not found, using zeros")
    if ev_col is None:
        log(f"  WARNING: EV_Score_Declarer column not found, using zeros")
    
    # DD trick columns (deal-level, same across all directions)
    available_dd_cols = [col for col in DD_COLS if col in available_cols]
    if len(available_dd_cols) < 20:
        log(f"  WARNING: Only {len(available_dd_cols)}/20 DD trick columns found")
    
    # Build aggregation expressions
    agg_exprs = [pl.len().alias("count")]
    if par_col:
        agg_exprs.append(pl.col(par_col).sum().alias("par_sum"))
    else:
        agg_exprs.append(pl.lit(0).alias("par_sum"))
    if ev_col:
        agg_exprs.append(pl.col(ev_col).sum().alias("ev_sum"))
    else:
        agg_exprs.append(pl.lit(0).alias("ev_sum"))
    
    # DD trick sums
    for dd_col in available_dd_cols:
        agg_exprs.append(pl.col(dd_col).cast(pl.Int64).sum().alias(f"{dd_col}_sum"))
    
    # Select columns to read
    read_cols = dim_cols.copy()
    if has_tp:
        read_cols.append(tp_col)
    if par_col:
        read_cols.append(par_col)
    if ev_col:
        read_cols.append(ev_col)
    read_cols.extend(available_dd_cols)
    
    # Add Vul column if filtering by vulnerability
    if vul_filter and "Vul" in available_cols:
        read_cols.append("Vul")
    
    # Build cube
    query = pl.scan_parquet(deals_file).select(read_cols)
    if max_deals:
        query = query.head(max_deals)
    
    # Compute Total_Points if not in parquet
    if not has_tp:
        sl_s_col = f"SL_{direction}_S"
        sl_h_col = f"SL_{direction}_H"
        sl_d_col = f"SL_{direction}_D"
        sl_c_col = f"SL_{direction}_C"
        
        def _dp_expr(col_name: str) -> pl.Expr:
            c = pl.col(col_name)
            return (
                pl.when(c == 0).then(3)
                .when(c == 1).then(2)
                .when(c == 2).then(1)
                .otherwise(0)
            )
        
        query = query.with_columns(
            (pl.col(f"HCP_{direction}")
             + _dp_expr(sl_s_col) + _dp_expr(sl_h_col)
             + _dp_expr(sl_d_col) + _dp_expr(sl_c_col)
            ).cast(pl.UInt8).alias(tp_col)
        )
    
    # Apply vulnerability filter if specified
    if vul_filter and "Vul" in available_cols:
        # Determine which Vul values make this direction vulnerable
        if direction in ("N", "S"):
            # NS side: vulnerable when Vul is N_S or Both
            vul_values = ["N_S", "Both"]
        else:
            # EW side: vulnerable when Vul is E_W or Both
            vul_values = ["E_W", "Both"]
        
        if vul_filter == "V":
            query = query.filter(pl.col("Vul").is_in(vul_values))
        else:  # NV
            query = query.filter(~pl.col("Vul").is_in(vul_values))
    
    # GROUP BY 6 dimensions (including Total_Points)
    group_cols = dim_cols + [tp_col]
    
    cube = (
        query
        .group_by(group_cols)
        .agg(agg_exprs)
        .collect()
    )
    
    # Rename columns for consistency
    rename_map = {
        dim_cols[0]: "HCP",
        dim_cols[1]: "SL_S",
        dim_cols[2]: "SL_H",
        dim_cols[3]: "SL_D",
        dim_cols[4]: "SL_C",
        tp_col: "Total_Points",
    }
    cube = cube.rename(rename_map)
    
    vul_suffix = f"_{vul_filter}" if vul_filter else ""
    elapsed = time.time() - t0
    log(f"  Built cube for {direction}{vul_suffix}: {cube.height:,} buckets, "
        f"{len(available_dd_cols)} DD cols in {fmt_time(elapsed)}")
    
    return cube


# ---------------------------------------------------------------------------
# BT Criteria Parser
# ---------------------------------------------------------------------------

def parse_bt_criteria(expr_str: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse Agg_Expr string to extract bounds for each metric.
    
    Example: "HCP >= 12 AND HCP <= 17 AND SL_S >= 4"
    Returns: {"HCP": (12, 17), "SL_S": (4, 13), ...}
    """
    bounds = {
        "HCP": [0, 40],
        "SL_S": [0, 13],
        "SL_H": [0, 13],
        "SL_D": [0, 13],
        "SL_C": [0, 13],
        "Total_Points": [0, 50],
    }
    
    if not expr_str or expr_str == "None" or expr_str == "null":
        return {k: (v[0], v[1]) for k, v in bounds.items()}
    
    # Parse >= and <=
    for match in re.finditer(r"(HCP|SL_[SHDC]|Total_Points)\s*>=\s*(\d+)", expr_str):
        metric, val = match.groups()
        bounds[metric][0] = max(bounds[metric][0], int(val))
    
    for match in re.finditer(r"(HCP|SL_[SHDC]|Total_Points)\s*<=\s*(\d+)", expr_str):
        metric, val = match.groups()
        bounds[metric][1] = min(bounds[metric][1], int(val))
    
    return {k: (v[0], v[1]) for k, v in bounds.items()}


def extract_bt_bounds_polars(bt_df: pl.DataFrame, seat: int) -> Dict[str, np.ndarray]:
    """
    Extract bounds arrays from BT DataFrame using Polars expressions.
    
    MEMORY-EFFICIENT: Uses Polars regex extraction, no Python object creation.
    
    Returns dict with lo/hi arrays for each metric (including Total_Points).
    """
    expr_col = f"Agg_Expr_Seat_{seat}"
    n = bt_df.height
    
    # Default bounds for all 6 dimensions
    defaults = {
        "HCP": (0, 40),
        "SL_S": (0, 13),
        "SL_H": (0, 13),
        "SL_D": (0, 13),
        "SL_C": (0, 13),
        "Total_Points": (0, 50),
    }
    
    if expr_col not in bt_df.columns:
        log(f"  WARNING: {expr_col} not found, using defaults")
        result = {}
        for metric, (default_lo, default_hi) in defaults.items():
            result[f"{metric}_lo"] = np.full(n, default_lo, dtype=np.int16)
            result[f"{metric}_hi"] = np.full(n, default_hi, dtype=np.int16)
        return result
    
    # Join list of strings into single string per row (stays in Polars, no Python objects!)
    joined = bt_df.select(
        pl.col(expr_col).list.join(" AND ").alias("expr_str")
    )
    
    # Build extraction expressions using Polars regex
    extract_exprs = []
    for metric, (default_lo, default_hi) in defaults.items():
        # Extract >= value (lower bound)
        lo_pattern = f"{metric}\\s*>=\\s*(\\d+)"
        extract_exprs.append(
            pl.col("expr_str")
            .str.extract(lo_pattern, group_index=1)
            .cast(pl.Int16)
            .fill_null(default_lo)
            .alias(f"{metric}_lo")
        )
        
        # Extract <= value (upper bound)
        hi_pattern = f"{metric}\\s*<=\\s*(\\d+)"
        extract_exprs.append(
            pl.col("expr_str")
            .str.extract(hi_pattern, group_index=1)
            .cast(pl.Int16)
            .fill_null(default_hi)
            .alias(f"{metric}_hi")
        )
    
    # Execute extraction (all in Polars, memory efficient)
    bounds_df = joined.select(extract_exprs)
    
    # Convert to numpy arrays
    result = {}
    for metric in defaults.keys():
        result[f"{metric}_lo"] = bounds_df[f"{metric}_lo"].to_numpy()
        result[f"{metric}_hi"] = bounds_df[f"{metric}_hi"].to_numpy()
    
    return result


def _validate_bt_bounds_sample(
    bt_df: pl.DataFrame,
    seat: int,
    bounds: Dict[str, np.ndarray],
    sample_n: int = BOUNDS_VALIDATE_ROWS,
) -> None:
    """Fail-fast correctness check for bounds extraction.

    Compares Polars-extracted bounds against the Python regex parser on a small
    sample.  This catches cases where multiple constraints for the same metric
    appear in a single expression (e.g. two '>=' clauses) which would otherwise
    be silently mis-parsed.
    """
    expr_col = f"Agg_Expr_Seat_{seat}"
    if expr_col not in bt_df.columns:
        return

    n_rows = bt_df.height
    if n_rows <= 0:
        return

    n = int(min(sample_n, n_rows))
    if n <= 0:
        return

    # Validate two slices (front + back) to increase coverage without sampling.
    offsets = [0]
    if n_rows > n:
        offsets.append(max(0, n_rows - n))

    for off in offsets:
        sample_lists = (
            bt_df.select(pl.col(expr_col).slice(off, n))
            .to_series()
            .to_list()
        )
        for i, parts in enumerate(sample_lists):
            # parts is typically List[str]
            expr_str = " AND ".join(parts) if isinstance(parts, list) else (str(parts) if parts else "")
            py_bounds = parse_bt_criteria(expr_str)
            row_idx = off + i
            for metric in METRICS:
                lo_key = f"{metric}_lo"
                hi_key = f"{metric}_hi"
                pol_lo = int(bounds[lo_key][row_idx])
                pol_hi = int(bounds[hi_key][row_idx])
                py_lo, py_hi = py_bounds[metric]
                if pol_lo != py_lo or pol_hi != py_hi:
                    raise ValueError(
                        f"Bounds extraction mismatch seat={seat} row={row_idx} metric={metric}: "
                        f"polars=({pol_lo},{pol_hi}) python=({py_lo},{py_hi}) expr={parts!r}"
                    )


def extract_bt_bounds(bt_df: pl.DataFrame, seat: int) -> Dict[str, np.ndarray]:
    """
    Extract bounds arrays from BT DataFrame for a specific seat.
    
    Uses Polars-based extraction for memory efficiency.
    """
    bounds = extract_bt_bounds_polars(bt_df, seat)
    _validate_bt_bounds_sample(bt_df, seat, bounds)
    return bounds


# ---------------------------------------------------------------------------
# GPU Matching
# ---------------------------------------------------------------------------


def prepare_cube_gpu(
    cube: pl.DataFrame,
    device: Any,
    compute_stats: bool = True,
) -> Dict[str, Any]:
    """Upload cube data to GPU tensors once.  Reuse across all chunks/seats.

    Returns a dict of GPU-resident tensors that ``query_cube_batch`` consumes.
    """
    cube_gpu: Dict[str, Any] = {}

    # 6 dimension columns (int16)
    dims = {}
    for metric in METRICS:
        if metric in cube.columns:
            dims[metric] = torch.tensor(
                cube[metric].to_numpy(), device=device, dtype=torch.int16
            )
    cube_gpu["dims"] = dims

    cube_count = torch.tensor(cube["count"].to_numpy(), device=device, dtype=torch.int64)
    cube_gpu["count"] = cube_count
    cube_gpu["par"] = torch.tensor(cube["par_sum"].to_numpy(), device=device, dtype=torch.int64)
    cube_gpu["ev"] = torch.tensor(cube["ev_sum"].to_numpy(), device=device, dtype=torch.int64)

    # DD trick sum columns
    dd_sum_cols = [c for c in cube.columns if c.startswith("DD_") and c.endswith("_sum")]
    cube_gpu["dd_sum_cols"] = dd_sum_cols
    dd_gpu = []
    if compute_stats:
        for col in dd_sum_cols:
            dd_gpu.append(
                torch.tensor(cube[col].to_numpy(), device=device, dtype=torch.int64)
            )
    cube_gpu["dd"] = dd_gpu

    # Precomputed weighted dimension values (for mean/std)
    weighted_dims: List[Optional[Any]] = []
    weighted_sq_dims: List[Optional[Any]] = []
    if compute_stats:
        for metric in METRICS:
            if metric in dims:
                d = dims[metric].to(torch.int64)
                weighted_dims.append(d * cube_count)
                weighted_sq_dims.append(d * d * cube_count)
            else:
                weighted_dims.append(None)
                weighted_sq_dims.append(None)
    cube_gpu["weighted_dims"] = weighted_dims
    cube_gpu["weighted_sq_dims"] = weighted_sq_dims
    cube_gpu["n_buckets"] = cube.height

    return cube_gpu


def free_cube_gpu(cube_gpu: Dict[str, Any], empty_cache: bool = True) -> None:
    """Release GPU tensors held by a prepared cube."""
    for v in cube_gpu.get("dims", {}).values():
        del v
    for k in ("count", "par", "ev"):
        if k in cube_gpu:
            del cube_gpu[k]
    for t in cube_gpu.get("dd", []):
        del t
    for t in cube_gpu.get("weighted_dims", []):
        if t is not None:
            del t
    for t in cube_gpu.get("weighted_sq_dims", []):
        if t is not None:
            del t
    cube_gpu.clear()
    if empty_cache:
        torch.cuda.empty_cache()


def query_cube_batch(
    cube_gpu: Dict[str, Any],
    bt_bounds: Dict[str, np.ndarray],
    bt_indices: np.ndarray,
    device: Any,
    batch_size: int = BATCH_SIZE,
    compute_stats: bool = True,
) -> Dict[str, Any]:
    """Run GPU matching against a *pre-uploaded* cube (from ``prepare_cube_gpu``).

    Same semantics as the old ``gpu_query_cube`` but avoids re-uploading cube
    tensors on every call.
    """
    n_bt = len(bt_indices)

    cube_dims_gpu = cube_gpu["dims"]
    cube_count = cube_gpu["count"]
    cube_par = cube_gpu["par"]
    cube_ev = cube_gpu["ev"]
    cube_dd_gpu = cube_gpu["dd"]
    weighted_dims = cube_gpu["weighted_dims"]
    weighted_sq_dims = cube_gpu["weighted_sq_dims"]
    dd_sum_cols = cube_gpu["dd_sum_cols"]
    n_dd = len(dd_sum_cols)
    n_dims = len(METRICS)

    # Upload BT bounds to GPU
    bounds_lo_gpu = {}
    bounds_hi_gpu = {}
    for metric in METRICS:
        lo_key = f"{metric}_lo"
        hi_key = f"{metric}_hi"
        if lo_key in bt_bounds:
            bounds_lo_gpu[metric] = torch.tensor(bt_bounds[lo_key], device=device, dtype=torch.int16)
            bounds_hi_gpu[metric] = torch.tensor(bt_bounds[hi_key], device=device, dtype=torch.int16)

    # Result arrays (CPU)
    # counts fit in int32 (<= ~16M deals per cube); keep int32 to reduce RAM/IO.
    result_counts = np.zeros(n_bt, dtype=np.int32)
    result_par = np.zeros(n_bt, dtype=np.int64)
    result_ev = np.zeros(n_bt, dtype=np.int64)

    result_dim_sums: np.ndarray = np.empty(0)
    result_dim_sq_sums: np.ndarray = np.empty(0)
    result_dim_mins: np.ndarray = np.empty(0)
    result_dim_maxs: np.ndarray = np.empty(0)
    result_dd_sums: np.ndarray = np.empty(0)
    if compute_stats:
        # dim_sums fit in int32 per (direction,vul) cube; dim_sq_sums must stay int64
        result_dim_sums = np.zeros((n_dims, n_bt), dtype=np.int32)
        result_dim_sq_sums = np.zeros((n_dims, n_bt), dtype=np.int64)
        # Store min/max as uint8 with sentinels: min=255 (no data), max=0 (no data)
        result_dim_mins = np.full((n_dims, n_bt), 255, dtype=np.uint8)
        result_dim_maxs = np.zeros((n_dims, n_bt), dtype=np.uint8)
        # dd_sums fit in int32 (<= 13 * deals_in_cube)
        result_dd_sums = np.zeros((n_dd, n_bt), dtype=np.int32)

    sentinel_min_i16 = torch.tensor(9999, dtype=torch.int16, device=device)
    sentinel_max_i16 = torch.tensor(-1, dtype=torch.int16, device=device)

    # Process in batches
    with torch.inference_mode():
        for i in range(0, n_bt, batch_size):
            end = min(i + batch_size, n_bt)

            # Build 6D mask: (batch_size, n_buckets)
            mask = None
            for metric in METRICS:
                if metric not in cube_dims_gpu or metric not in bounds_lo_gpu:
                    continue
                batch_lo = bounds_lo_gpu[metric][i:end].unsqueeze(1)
                batch_hi = bounds_hi_gpu[metric][i:end].unsqueeze(1)
                dim_check = (cube_dims_gpu[metric] >= batch_lo) & (cube_dims_gpu[metric] <= batch_hi)
                mask = dim_check if mask is None else (mask & dim_check)

            if mask is None:
                continue

            # ---- EV aggregations (count, par, ev) — always computed ----
            mask_int = mask.to(torch.int64)
            result_counts[i:end] = (mask_int * cube_count).sum(dim=1).to(torch.int32).cpu().numpy()
            result_par[i:end] = (mask_int * cube_par).sum(dim=1).cpu().numpy()
            result_ev[i:end] = (mask_int * cube_ev).sum(dim=1).cpu().numpy()

            if compute_stats:
                # ---- Hand stats: weighted sums / squared sums ----
                for dim_idx in range(n_dims):
                    if weighted_dims[dim_idx] is not None:
                        result_dim_sums[dim_idx, i:end] = (
                            (mask_int * weighted_dims[dim_idx]).sum(dim=1).to(torch.int32).cpu().numpy()
                        )
                        result_dim_sq_sums[dim_idx, i:end] = (
                            (mask_int * weighted_sq_dims[dim_idx]).sum(dim=1).cpu().numpy()
                        )

                # ---- Hand stats: min / max ----
                for dim_idx, metric in enumerate(METRICS):
                    if metric not in cube_dims_gpu:
                        continue
                    cube_dim = cube_dims_gpu[metric]

                    masked_min = torch.where(mask, cube_dim, sentinel_min_i16)
                    mins_u8 = torch.clamp(masked_min.min(dim=1).values, 0, 255).to(torch.uint8)
                    result_dim_mins[dim_idx, i:end] = mins_u8.cpu().numpy()

                    masked_max = torch.where(mask, cube_dim, sentinel_max_i16)
                    maxs_u8 = torch.clamp(masked_max.max(dim=1).values, 0, 255).to(torch.uint8)
                    result_dim_maxs[dim_idx, i:end] = maxs_u8.cpu().numpy()

                    del masked_min, masked_max, mins_u8, maxs_u8

                # ---- DD trick sums ----
                for dd_idx in range(n_dd):
                    result_dd_sums[dd_idx, i:end] = (
                        (mask_int * cube_dd_gpu[dd_idx]).sum(dim=1).to(torch.int32).cpu().numpy()
                    )

    result: Dict[str, Any] = {
        "counts": result_counts,
        "par_sums": result_par,
        "ev_sums": result_ev,
    }
    if compute_stats:
        result["dim_sums"] = result_dim_sums
        result["dim_sq_sums"] = result_dim_sq_sums
        result["dim_mins"] = result_dim_mins
        result["dim_maxs"] = result_dim_maxs
        result["dd_sums"] = result_dd_sums
    return result


def gpu_query_cube(
    cube: pl.DataFrame,
    bt_bounds: Dict[str, np.ndarray],
    bt_indices: np.ndarray,
    device: Any,  # torch.device
    batch_size: int = BATCH_SIZE,
    start_idx: int = 0,
    pbar: Optional[ProgressBar] = None,
    collect_index: bool = False,
    index_file: Optional[Path] = None,
    compute_stats: bool = True,
) -> Dict[str, Any]:
    """Legacy wrapper — uploads cube to GPU, queries, and frees.

    NOTE: collect_index argument is ignored in v3 pipeline.
    
    New code should use ``prepare_cube_gpu`` + ``query_cube_batch`` directly
    to keep the cube resident across multiple calls.
    """
    cube_gpu = prepare_cube_gpu(cube, device, compute_stats=compute_stats)
    try:
        return query_cube_batch(
            cube_gpu, bt_bounds, bt_indices, device, batch_size,
            compute_stats=compute_stats,
        )
    finally:
        free_cube_gpu(cube_gpu)


# ---------------------------------------------------------------------------
# Inverted Index Consolidation
# ---------------------------------------------------------------------------

def consolidate_inverted_index(
    index_dir: Path,
    cube: pl.DataFrame,
    direction: str,
    seat: int,
) -> None:
    """
    Consolidate raw binary index files into per-bucket parquet files.
    
    Raw format: interleaved (uint16 bucket_idx, uint32 bt_idx) pairs
    Output: One parquet per bucket with bt_indices list
    """
    raw_dir = index_dir / "raw" / f"{direction}_S{seat}"
    output_dir = index_dir / "consolidated" / f"{direction}_S{seat}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_buckets = cube.height
    
    # Collect all matches per bucket
    bucket_matches: Dict[int, List[int]] = {i: [] for i in range(n_buckets)}
    
    # Read all raw files
    raw_files = sorted(raw_dir.glob("chunk_*.bin"))
    if not raw_files:
        log(f"    No raw index files found for {direction}/S{seat}")
        return
    
    log(f"    Reading {len(raw_files)} raw files...")
    total_pairs = 0
    
    for raw_file in raw_files:
        file_size = raw_file.stat().st_size
        if file_size == 0:
            continue
        
        # Read as structured array: (bucket_idx: u2, bt_idx: u4)
        # But numpy doesn't pack u2+u4 well, so read separately
        with open(raw_file, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Parse interleaved data: 2 bytes bucket, 4 bytes bt_idx
        # Actually our format is column_stack which makes it [bucket, bt_idx] per row
        # Let me fix: we wrote column_stack which is row-major
        # So pairs.tofile writes: bucket0_u16, bt0_u32, bucket1_u16, bt1_u32...
        # This is tricky because of different sizes
        
        # Simpler: re-read as structured array
        dt = np.dtype([('bucket', np.uint16), ('bt_idx', np.uint32)])
        try:
            pairs = np.fromfile(raw_file, dtype=dt)
            for bucket_idx, bt_idx in pairs:
                bucket_matches[bucket_idx].append(int(bt_idx))
            total_pairs += len(pairs)
        except Exception as e:
            log(f"    Warning: Could not read {raw_file}: {e}")
    
    log(f"    Total pairs: {total_pairs:,}")
    
    # Write per-bucket files
    log(f"    Writing {n_buckets} bucket files...")
    non_empty = 0
    for bucket_idx in range(n_buckets):
        bt_list = bucket_matches[bucket_idx]
        if bt_list:
            bt_arr = np.array(bt_list, dtype=np.uint32)
            bt_arr.sort()  # Sort for efficient lookup later
            np.save(output_dir / f"bucket_{bucket_idx:05d}.npy", bt_arr)
            non_empty += 1
    
    log(f"    Written {non_empty} non-empty bucket files")
    
    # Save bucket metadata (the cube values for lookup)
    cube.write_parquet(output_dir / "bucket_metadata.parquet")


def consolidate_inverted_index_vul(
    index_dir: Path,
    cube: pl.DataFrame,
    direction: str,
    vul: str,
    seat: int,
) -> None:
    """
    Consolidate raw binary index files into per-bucket parquet files (with vul state).
    
    Raw format: interleaved (uint16 bucket_idx, uint32 bt_idx) pairs
    Output: One parquet per bucket with bt_indices list
    """
    raw_dir = index_dir / "raw" / f"{direction}_{vul}_S{seat}"
    output_dir = index_dir / "consolidated" / f"{direction}_{vul}_S{seat}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_buckets = cube.height
    
    # Collect all matches per bucket
    bucket_matches: Dict[int, List[int]] = {i: [] for i in range(n_buckets)}
    
    # Read all raw files
    raw_files = sorted(raw_dir.glob("chunk_*.bin"))
    if not raw_files:
        log(f"    No raw index files found for {direction}_{vul}/S{seat}")
        return
    
    log(f"    Reading {len(raw_files)} raw files...")
    total_pairs = 0
    
    for raw_file in raw_files:
        file_size = raw_file.stat().st_size
        if file_size == 0:
            continue
        
        dt = np.dtype([('bucket', np.uint16), ('bt_idx', np.uint32)])
        try:
            pairs = np.fromfile(raw_file, dtype=dt)
            for bucket_idx, bt_idx in pairs:
                bucket_matches[bucket_idx].append(int(bt_idx))
            total_pairs += len(pairs)
        except Exception as e:
            log(f"    Warning: Could not read {raw_file}: {e}")
    
    log(f"    Total pairs: {total_pairs:,}")
    
    # Write per-bucket files
    log(f"    Writing {n_buckets} bucket files...")
    non_empty = 0
    for bucket_idx in range(n_buckets):
        bt_list = bucket_matches[bucket_idx]
        if bt_list:
            bt_arr = np.array(bt_list, dtype=np.uint32)
            bt_arr.sort()  # Sort for efficient lookup later
            np.save(output_dir / f"bucket_{bucket_idx:05d}.npy", bt_arr)
            non_empty += 1
    
    log(f"    Written {non_empty} non-empty bucket files")
    
    # Save bucket metadata (the cube values for lookup)
    cube.write_parquet(output_dir / "bucket_metadata.parquet")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    deals_file: Path,
    bt_file: Path,
    output_file: Path,
    checkpoint_dir: Path,
    index_dir: Optional[Path] = None,
    resume: bool = False,
    max_bt_rows: Optional[int] = None,
    max_deals: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    build_index: bool = True,
) -> None:
    """
    Run the full EV computation pipeline.
    
    If build_index=True, the inverted index is deprecated/ignored in v3.
    """
    
    pipeline_start = time.time()
    log("=" * 70)
    log("BT EV/Par Stats - GPU Accelerated Pipeline")
    if build_index:
        log("  WARNING: --build-index is DEPRECATED and will be removed.")
        log("  Use bbo_deal_to_bt_verified.parquet for deal→BT lookups instead.")
    log("=" * 70)
    log(f"Start: {fmt_datetime()}")
    log(f"Deals file: {deals_file}")
    log(f"BT file: {bt_file}")
    log(f"Output: {output_file}")
    log(f"Checkpoint dir: {checkpoint_dir}")
    if build_index and index_dir:
        log(f"Index dir: {index_dir}")
    log(f"Resume: {resume}")
    if max_bt_rows:
        log(f"Max BT rows: {max_bt_rows:,}")
    if max_deals:
        log(f"Max deals: {max_deals:,}")
    log("=" * 70)
    
    # Check GPU
    if not HAS_TORCH:
        log("ERROR: PyTorch not available. Please install: pip install torch")
        return
    
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available. GPU required for this pipeline.")
        return
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Initialize checkpoint manager
    ckpt = CheckpointManager(checkpoint_dir)
    
    if resume:
        state = ckpt.load_state()
        if state:
            log(f"Resuming from checkpoint: {state.get('last_completed', 'none')}")
        else:
            log("No checkpoint found, starting fresh")
            state = {"completed": [], "last_completed": None}
    else:
        ckpt.clear()
        state = {"completed": [], "last_completed": None}

    # --- Resume safety / pipeline versioning ---
    if not isinstance(state, dict):
        state = {"completed": [], "last_completed": None}
    if not isinstance(state.get("completed"), list):
        state["completed"] = []
    if "last_completed" not in state:
        state["last_completed"] = None

    prev_ver = state.get("pipeline_version")
    if prev_ver != PIPELINE_VERSION:
        if resume:
            log(f"  WARNING: checkpoint pipeline_version={prev_ver!r} → {PIPELINE_VERSION!r}; validating checkpoints")
        state["pipeline_version"] = PIPELINE_VERSION

    # When resuming, rebuild completion state from what's actually on disk:
    # - cube_{direction}_{vul}.parquet present → cube task complete
    # - results_{direction}_{vul}_S{seat}.npz contains REQUIRED_VUL_RESULT_KEYS → query task complete
    if resume:
        completed_set = set(state.get("completed", []))

        # Cube files
        for direction in DIRECTIONS:
            for vul in ("NV", "V"):
                cube_file = checkpoint_dir / f"cube_{direction}_{vul}.parquet"
                if cube_file.exists():
                    completed_set.add(f"cube_{direction}_{vul}")

        # Query tasks (strip stale ones first, then re-add verified)
        completed_set = {k for k in completed_set if not str(k).startswith("query_")}
        n_verified = 0
        for seat in range(1, 5):
            for direction in DIRECTIONS:
                for vul in ("NV", "V"):
                    task_key = f"query_{direction}_{vul}_S{seat}"
                    if ckpt.has_results_vul(direction, vul, seat, required_keys=REQUIRED_VUL_RESULT_KEYS):
                        completed_set.add(task_key)
                        n_verified += 1

        state["completed"] = sorted(completed_set)
        if state.get("last_completed") not in completed_set:
            state["last_completed"] = None
        ckpt.save_state(state)
        log(f"  Resume validation: {n_verified}/32 query tasks have full stats checkpoints")
    
    # ---------------------------------------------------------------------------
    # Step 1: Load BT core data (small - just indices)
    # ---------------------------------------------------------------------------
    # Force disable build_index as it's not supported in v3 pipeline
    if build_index:
        log("  WARNING: --build-index is disabled in v3 pipeline. Use bbo_deal_to_bt_verified.parquet.")
        build_index = False

    total_steps = 5
    log(f"\n[1/{total_steps}] Loading BT core data...")
    step_start = time.time()
    
    # Only load tiny columns now - expressions loaded per-seat later
    bt_core = pl.read_parquet(bt_file, columns=["bt_index"])
    if max_bt_rows:
        bt_core = bt_core.head(max_bt_rows)
    
    n_bt = bt_core.height
    bt_indices = bt_core["bt_index"].to_numpy()
    del bt_core  # Free immediately
    
    elapsed = time.time() - step_start
    log(f"  Loaded {n_bt:,} BT indices in {fmt_time(elapsed)}")
    
    # ---------------------------------------------------------------------------
    # Step 2: Build histogram cubes (8 vul-split only)
    # ---------------------------------------------------------------------------
    # Only 8 vul-split cubes needed — single merged pass computes EV + stats on
    # the same cubes.  Combined cubes are no longer built; NV+V stats are summed
    # in the combination step instead.
    log(f"\n[2/{total_steps}] Building histogram cubes (8 vul-split)...")
    step_start = time.time()
    
    VUL_STATES = ["NV", "V"]
    
    cubes_vul = {}  # Key: (direction, vul) tuple
    for direction in DIRECTIONS:
        for vul in VUL_STATES:
            cube_key = f"cube_{direction}_{vul}"
            if cube_key in state.get("completed", []):
                log(f"  Skipping {direction}_{vul} (already built)")
                cube_file = checkpoint_dir / f"cube_{direction}_{vul}.parquet"
                if cube_file.exists():
                    cubes_vul[(direction, vul)] = pl.read_parquet(cube_file)
                else:
                    cubes_vul[(direction, vul)] = build_histogram_cube(deals_file, direction, max_deals, vul_filter=vul)
            else:
                cubes_vul[(direction, vul)] = build_histogram_cube(deals_file, direction, max_deals, vul_filter=vul)
                cube_file = checkpoint_dir / f"cube_{direction}_{vul}.parquet"
                cubes_vul[(direction, vul)].write_parquet(cube_file)
                state["completed"].append(cube_key)
                ckpt.save_state(state)
    
    elapsed = time.time() - step_start
    log(f"  Cubes built in {fmt_time(elapsed)} (8 vul-split cubes)")
    
    # ---------------------------------------------------------------------------
    # Step 3: Merged GPU pass — EV + hand stats + DD means on 8 vul-split cubes
    # ---------------------------------------------------------------------------
    # Single pass computes all 47 ops (3 EV + 6×4 dim + 6×4 dim² + 6 min + 6 max
    # + 20 DD sums) on each vul-split cube.  The combination step (step 4) sums
    # NV+V for hand stats and DD means.  This eliminates the 4 combined cubes and
    # avoids a second traversal of the cube data.
    #
    # 32 tasks total: 4 seats × 4 directions × 2 vul states.
    # Cube tensors are uploaded to GPU once per (direction, vul) and reused across
    # all BT chunks for that cube.
    #
    # Peak RAM per task: ~96 GB (stats accumulators) — same as before but only one
    # set of accumulators is live at a time.
    # ---------------------------------------------------------------------------
    log(f"\n[3/{total_steps}] Merged GPU pass — EV + stats on 8 vul-split cubes...")
    step_start = time.time()
    
    n_dims = len(METRICS)
    _sample_cube_key = next(iter(cubes_vul))
    n_dd = sum(1 for c in cubes_vul[_sample_cube_key].columns if c.startswith("DD_") and c.endswith("_sum"))
    if n_dd != len(DD_DECLARERS) * len(DD_STRAINS):
        raise RuntimeError(
            f"Expected {len(DD_DECLARERS) * len(DD_STRAINS)} DD columns but found {n_dd}. "
            f"DD rotation logic requires exactly 4 declarers × 5 strains = 20 columns."
        )
    
    # Upload cube tensors to GPU once per (direction, vul) and reuse across all seats/chunks.
    log("  Uploading 8 cubes to GPU (keep resident across all tasks)...")
    upload_start = time.time()
    cube_gpu_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for direction in DIRECTIONS:
        for vul in VUL_STATES:
            cube_gpu_cache[(direction, vul)] = prepare_cube_gpu(
                cubes_vul[(direction, vul)],
                device,
                compute_stats=True,
            )
    log(f"  Cube upload complete in {fmt_time(time.time() - upload_start)}")

    chunk_size = BT_CHUNK_SIZE
    n_chunks = (n_bt + chunk_size - 1) // chunk_size
    
    # 32 tasks: 4 seats × 4 dirs × 2 vul
    total_tasks = len(DIRECTIONS) * len(VUL_STATES) * 4
    tasks_done = sum(1 for d in DIRECTIONS for v in VUL_STATES for s in range(1, 5)
                     if f"query_{d}_{v}_S{s}" in state.get("completed", []))
    
    for seat in range(1, 5):
        seat_tasks = [f"query_{d}_{v}_S{seat}" for d in DIRECTIONS for v in VUL_STATES]
        if all(t in state.get("completed", []) for t in seat_tasks):
            log(f"  Skipping seat {seat} (all completed)")
            continue
        
        log(f"  Seat {seat}: {n_bt:,} BT rows, {n_chunks} chunks of {chunk_size:,}")
        
        # Cache bounds once per seat (reused across all 8 cubes)
        log(f"    Loading Agg_Expr_Seat_{seat} bounds...")
        bounds_start = time.time()
        seat_df = pl.read_parquet(bt_file, columns=["bt_index", f"Agg_Expr_Seat_{seat}"])
        if max_bt_rows:
            seat_df = seat_df.head(max_bt_rows)
        all_bounds = extract_bt_bounds(seat_df, seat)
        del seat_df; gc.collect()
        log(f"    Bounds cached: {sum(a.nbytes for a in all_bounds.values()) / 1e9:.1f} GB "
            f"in {fmt_time(time.time() - bounds_start)}")
        
        for direction in DIRECTIONS:
            for vul in VUL_STATES:
                task_key = f"query_{direction}_{vul}_S{seat}"
                if task_key in state.get("completed", []):
                    log(f"    Skipping {direction}_{vul}/S{seat} (completed)")
                    continue
                
                pair_start = time.time()
                
                # Full accumulators: EV (3 arrays) + stats (dims + DD)
                results: Dict[str, Any] = {
                    "counts": np.zeros(n_bt, dtype=np.int32),
                    "par_sums": np.zeros(n_bt, dtype=np.int64),
                    "ev_sums": np.zeros(n_bt, dtype=np.int64),
                    "dim_sums": np.zeros((n_dims, n_bt), dtype=np.int32),
                    "dim_sq_sums": np.zeros((n_dims, n_bt), dtype=np.int64),
                    "dim_mins": np.full((n_dims, n_bt), 255, dtype=np.uint8),
                    "dim_maxs": np.zeros((n_dims, n_bt), dtype=np.uint8),
                    "dd_sums": np.zeros((n_dd, n_bt), dtype=np.int32),
                }
                accum_bytes = sum(a.nbytes for a in results.values())
                log(f"    {direction}_{vul}/S{seat}: allocated {accum_bytes / 1e9:.1f} GB accumulators")
                
                cube_gpu = cube_gpu_cache[(direction, vul)]
                
                for chunk_idx in range(n_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, n_bt)
                    
                    if chunk_idx % 10 == 0 or chunk_idx == n_chunks - 1:
                        log(f"      Chunk {chunk_idx + 1}/{n_chunks}")
                    
                    chunk_bounds = {k: arr[chunk_start:chunk_end] for k, arr in all_bounds.items()}
                    chunk_bt = bt_indices[chunk_start:chunk_end]
                    
                    chunk_results = query_cube_batch(
                        cube_gpu, chunk_bounds, chunk_bt, device, batch_size,
                        compute_stats=True,
                    )
                    
                    results["counts"][chunk_start:chunk_end] = chunk_results["counts"]
                    results["par_sums"][chunk_start:chunk_end] = chunk_results["par_sums"]
                    results["ev_sums"][chunk_start:chunk_end] = chunk_results["ev_sums"]
                    results["dim_sums"][:, chunk_start:chunk_end] = chunk_results["dim_sums"]
                    results["dim_sq_sums"][:, chunk_start:chunk_end] = chunk_results["dim_sq_sums"]
                    results["dim_mins"][:, chunk_start:chunk_end] = chunk_results["dim_mins"]
                    results["dim_maxs"][:, chunk_start:chunk_end] = chunk_results["dim_maxs"]
                    results["dd_sums"][:, chunk_start:chunk_end] = chunk_results["dd_sums"]
                    
                    del chunk_results
                
                # Save all results (EV + stats) in one checkpoint
                ckpt.save_results_vul(
                    direction, vul, seat, bt_indices,
                    results["counts"], results["par_sums"], results["ev_sums"],
                    dim_sums=results["dim_sums"],
                    dim_sq_sums=results["dim_sq_sums"],
                    dim_mins=results["dim_mins"],
                    dim_maxs=results["dim_maxs"],
                    dd_sums=results["dd_sums"],
                )
                
                state["completed"].append(task_key)
                state["last_completed"] = task_key
                tasks_done += 1
                ckpt.save_state(state)
                
                pair_elapsed = time.time() - pair_start
                log(f"    {direction}_{vul}/S{seat} done in {fmt_time(pair_elapsed)} "
                    f"({tasks_done}/{total_tasks} tasks)")
                
                del results; gc.collect()
        
        del all_bounds; gc.collect()
    
    # Release cube tensors after all tasks complete
    for _cube_gpu in cube_gpu_cache.values():
        free_cube_gpu(_cube_gpu, empty_cache=False)
    cube_gpu_cache.clear()
    torch.cuda.empty_cache()

    gpu_elapsed = time.time() - step_start
    log(f"  Merged GPU pass completed in {fmt_time(gpu_elapsed)}")
    
    # ---------------------------------------------------------------------------
    # Step 4: Combine results and compute averages (NV/V split + hand stats + DD means)
    # ---------------------------------------------------------------------------
    log(f"\n[4/{total_steps}] Combining results (NV/V split + hand stats + DD means)...")
    step_start = time.time()
    
    # bt_indices already set from Step 1
    n = len(bt_indices)
    
    # Determine DD column count from any vul-split cube
    _sample_cube_key = next(iter(cubes_vul))
    dd_sum_col_names = [c for c in cubes_vul[_sample_cube_key].columns
                        if c.startswith("DD_") and c.endswith("_sum")]
    n_dd = len(dd_sum_col_names)
    n_dims = len(METRICS)
    
    result_cols: Dict[str, Any] = {"bt_index": bt_indices}
    
    for seat in range(1, 5):
        log(f"  Aggregating seat {seat}...")
        
        # ---- Per-vul aggregation (Count, Avg_Par, Avg_EV) from EV checkpoints ----
        for vul in VUL_STATES:
            total_count = np.zeros(n, dtype=np.int32)
            total_par = np.zeros(n, dtype=np.int64)
            total_ev = np.zeros(n, dtype=np.int64)
            
            for direction in DIRECTIONS:
                data = ckpt.load_results_vul(direction, vul, seat)
                if data is None:
                    raise RuntimeError(f"Missing checkpoint for {direction}_{vul}/S{seat} (cannot combine results)")
                for k in ("counts", "par_sums", "ev_sums"):
                    if k not in data:
                        raise RuntimeError(f"Checkpoint missing key {k!r} for {direction}_{vul}/S{seat}")
                total_count += data["counts"]
                total_par += data["par_sums"]
                total_ev += data["ev_sums"]
                del data
            
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_par = np.where(total_count > 0, total_par / total_count, np.nan)
                avg_ev = np.where(total_count > 0, total_ev / total_count, np.nan)
            
            result_cols[f"Count_S{seat}_{vul}"] = total_count.astype(np.uint32)
            result_cols[f"Avg_Par_S{seat}_{vul}"] = avg_par.astype(np.float32)
            result_cols[f"Avg_EV_S{seat}_{vul}"] = avg_ev.astype(np.float32)
        
        # ---- Combined stats: sum NV + V vul checkpoints across 4 directions ----
        # Each vul checkpoint now contains stats arrays (dim_sums, dd_sums, etc.)
        # from the merged pass.  We sum across all 8 (4 dirs × 2 vul) checkpoints.
        combined_count = np.zeros(n, dtype=np.int32)
        combined_dim_sums = np.zeros((n_dims, n), dtype=np.int64)
        combined_dim_sq_sums = np.zeros((n_dims, n), dtype=np.int64)
        combined_dim_mins = np.full((n_dims, n), 255, dtype=np.uint8)
        combined_dim_maxs = np.zeros((n_dims, n), dtype=np.uint8)
        # dd_sums accumulates 8 int32 checkpoints; max per-ckpt ≈ 208M → 8*208M = 1.66B,
        # which is close to int32 max (2.1B).  Use int64 for safe accumulation.
        combined_dd_sums = np.zeros((n_dd, n), dtype=np.int64)
        
        for direction in DIRECTIONS:
            for vul in VUL_STATES:
                data = ckpt.load_results_vul(direction, vul, seat)
                if data is None:
                    raise RuntimeError(f"Missing checkpoint for {direction}_{vul}/S{seat} (cannot combine stats)")
                missing = REQUIRED_VUL_RESULT_KEYS.difference(data.keys())
                if missing:
                    raise RuntimeError(
                        f"Checkpoint for {direction}_{vul}/S{seat} missing required keys: {sorted(missing)}"
                    )
                
                combined_count += data["counts"]
                combined_dim_sums += data["dim_sums"]
                combined_dim_sq_sums += data["dim_sq_sums"]
                combined_dim_mins = np.minimum(combined_dim_mins, data["dim_mins"])
                combined_dim_maxs = np.maximum(combined_dim_maxs, data["dim_maxs"])

                # Rotate DD sums to seat-relative positions before accumulating.
                # DD columns are in blocks of 5 (one per declarer): N=0-4, E=5-9, S=10-14, W=15-19.
                # For the N-cube (dir_idx=0): N is seat S, so no rotation.
                # For the E-cube (dir_idx=1): E is seat S, rotate by -5 so E→position 0.
                # For the S-cube (dir_idx=2): S is seat S, rotate by -10.
                # For the W-cube (dir_idx=3): W is seat S, rotate by -15.
                dir_idx = DIRECTIONS.index(direction)
                shift = dir_idx * len(DD_STRAINS)
                if shift == 0:
                    combined_dd_sums += data["dd_sums"]
                else:
                    # Equivalent to np.roll(dd_sums, -shift, axis=0) but avoids allocating a huge copy.
                    combined_dd_sums[: n_dd - shift] += data["dd_sums"][shift:]
                    combined_dd_sums[n_dd - shift :] += data["dd_sums"][:shift]
                del data
        
        # matching_deal_count (combined count across all directions)
        result_cols[f"matching_deal_count_S{seat}"] = combined_count.astype(np.uint32)
        
        # Hand stats: mean, std, min, max for each of 6 dimensions
        count_1d = combined_count.astype(np.float64)
        count_2d = count_1d[np.newaxis, :]  # (1, n) for broadcasting with (n_dims, n)
        has_data = count_1d > 0
        has_data_2d = has_data[np.newaxis, :]
        
        # Column name mapping for output: METRICS index → output name
        # METRICS = ["HCP", "SL_S", "SL_H", "SL_D", "SL_C", "Total_Points"]
        metric_output_names = {
            "HCP": "HCP",
            "SL_S": "SL_S",
            "SL_H": "SL_H",
            "SL_D": "SL_D",
            "SL_C": "SL_C",
            "Total_Points": "Total_Points",
        }
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dim_means = np.where(has_data_2d,
                                 combined_dim_sums.astype(np.float64) / count_2d,
                                 np.nan)  # (n_dims, n)
            dim_mean_sq = np.where(has_data_2d,
                                   combined_dim_sq_sums.astype(np.float64) / count_2d,
                                   np.nan)
            dim_vars = dim_mean_sq - dim_means ** 2
            dim_stds = np.sqrt(np.maximum(0, dim_vars))

        # Downcast once; store views to avoid per-column copies.
        dim_means_f32 = dim_means.astype(np.float32)
        dim_stds_f32 = dim_stds.astype(np.float32)
        del dim_means, dim_stds, dim_mean_sq, dim_vars
        
        for dim_idx, metric in enumerate(METRICS):
            out_name = metric_output_names[metric]
            
            result_cols[f"{out_name}_mean_S{seat}"] = dim_means_f32[dim_idx]
            result_cols[f"{out_name}_std_S{seat}"] = dim_stds_f32[dim_idx]
            
            # Min/max: replace sentinel values with 0 where no data
            mins = combined_dim_mins[dim_idx].copy()
            mins[~has_data] = 0
            result_cols[f"{out_name}_min_S{seat}"] = mins
            
            maxs = combined_dim_maxs[dim_idx].copy()
            maxs[~has_data] = 0
            result_cols[f"{out_name}_max_S{seat}"] = maxs
        
        # DD trick means — compute in float64 then downcast to float32
        with np.errstate(divide='ignore', invalid='ignore'):
            dd_means = combined_dd_sums.astype(np.float64) / count_2d
            dd_means[:, ~has_data] = np.nan  # (n_dd, n)
        dd_means_f32 = dd_means.astype(np.float32)
        del dd_means
        
        # After rotation, DD positions are seat-relative:
        #   positions 0-4  = seat S (self) declares strains C,D,H,S,N
        #   positions 5-9  = seat S+1 (LHO) declares strains C,D,H,S,N
        #   positions 10-14 = seat S+2 (partner) declares strains C,D,H,S,N
        #   positions 15-19 = seat S+3 (RHO) declares strains C,D,H,S,N
        for dd_idx in range(n_dd):
            decl_seat_offset = dd_idx // len(DD_STRAINS)  # 0=self, 1=next, 2=opp, 3=prev
            decl_seat = ((seat - 1 + decl_seat_offset) % 4) + 1  # 1-indexed
            strain = DD_STRAINS[dd_idx % len(DD_STRAINS)]
            result_cols[f"DD_S{decl_seat}_{strain}_mean_S{seat}"] = dd_means_f32[dd_idx]
    
    result_df = pl.DataFrame(result_cols)
    result_df = result_df.sort("bt_index")
    
    n_total_cols = len(result_cols)
    elapsed = time.time() - step_start
    log(f"  Combined {n:,} rows in {fmt_time(elapsed)} ({n_total_cols} columns: "
        f"24 EV/Par + {4 * (n_dims * 4 + 1)} hand stats + {4 * n_dd} DD means)")
    
    # ---------------------------------------------------------------------------
    # Step 5: Write stats output
    # ---------------------------------------------------------------------------
    log(f"\n[5/{total_steps}] Writing stats output...")
    step_start = time.time()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_file)
    
    file_size = output_file.stat().st_size / 1e6
    elapsed = time.time() - step_start
    log(f"  Written {output_file} ({file_size:.1f} MB) in {fmt_time(elapsed)}")
    
    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)
    log(f"Start:   {datetime.fromtimestamp(pipeline_start).strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"End:     {fmt_datetime()}")
    log(f"Elapsed: {fmt_time(total_elapsed)}")
    log(f"Stats:   {output_file}")
    log(f"Columns: {n_total_cols}")
    log(f"Rows:    {n:,}")
    log("=" * 70)
    
    # Sample output
    log("\nSample results (first 5):")
    sample = result_df.head(5)
    for row in sample.iter_rows(named=True):
        count_nv = row.get('Count_S1_NV', 0)
        count_v = row.get('Count_S1_V', 0)
        ev_nv = row.get('Avg_EV_S1_NV')
        ev_v = row.get('Avg_EV_S1_V')
        hcp_mean = row.get('HCP_mean_S1')
        match_count = row.get('matching_deal_count_S1', 0)
        ev_nv_str = f"{ev_nv:.1f}" if ev_nv is not None and not (isinstance(ev_nv, float) and ev_nv != ev_nv) else "N/A"
        ev_v_str = f"{ev_v:.1f}" if ev_v is not None and not (isinstance(ev_v, float) and ev_v != ev_v) else "N/A"
        hcp_str = f"{hcp_mean:.1f}" if hcp_mean is not None and not (isinstance(hcp_mean, float) and hcp_mean != hcp_mean) else "N/A"
        log(f"  bt_index={row['bt_index']}, "
            f"Count_NV={count_nv:,}, Count_V={count_v:,}, "
            f"EV_NV={ev_nv_str}, EV_V={ev_v_str}, "
            f"HCP_mean={hcp_str}, match_count={match_count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute BT EV/Par statistics using GPU acceleration"
    )
    parser.add_argument(
        "--deals-file", type=Path, default=DEFAULT_DEALS_FILE,
        help="Path to deals parquet file"
    )
    parser.add_argument(
        "--bt-file", type=Path, default=DEFAULT_BT_FILE,
        help="Path to BT parquet file"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_FILE,
        help="Output parquet file"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory for restartability"
    )
    parser.add_argument(
        "--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
        help="Directory for inverted index output"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--build-index", action="store_true",
        help="DEPRECATED/disabled: ignored in v3 pipeline"
    )
    parser.add_argument(
        "--max-bt-rows", type=int, default=None,
        help="Maximum BT rows to process (for testing)"
    )
    parser.add_argument(
        "--max-deals", type=int, default=None,
        help="Maximum deals to include in cube (for testing)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"GPU batch size (default: {BATCH_SIZE})"
    )
    
    args = parser.parse_args()
    
    # Validate files
    if not args.deals_file.exists():
        print(f"ERROR: Deals file not found: {args.deals_file}")
        sys.exit(1)
    
    if not args.bt_file.exists():
        print(f"ERROR: BT file not found: {args.bt_file}")
        sys.exit(1)
    
    run_pipeline(
        deals_file=args.deals_file,
        bt_file=args.bt_file,
        output_file=args.output,
        checkpoint_dir=args.checkpoint_dir,
        index_dir=args.index_dir,
        resume=args.resume,
        build_index=args.build_index,
        max_bt_rows=args.max_bt_rows,
        max_deals=args.max_deals,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
