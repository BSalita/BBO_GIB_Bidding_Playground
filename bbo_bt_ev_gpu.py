"""
BT EV/Par Statistics + Inverted Index - GPU Accelerated (NV/V Split)

Takes 6 hours to run on a RTX 5080 with 16GB VRAM. 100GB of RAM.

Computes per-BT-row aggregate statistics SPLIT BY VULNERABILITY using 
GPU-accelerated histogram cube matching.

Features:
- GPU acceleration via PyTorch (RTX 5080)
- Histogram cube for efficient aggregation (16M deals → ~7K buckets per vul)
- Vulnerability split: separate NV and V statistics per seat
- Progress indicators (tqdm)
- Restartable via checkpoints
- Memory-efficient chunked processing

Expected runtime: ~7 hours for stats only

Output Schema (24 columns):
- bt_index: BT row identifier
- Count_S{1-4}_NV, Count_S{1-4}_V: Deal counts per seat, split by vulnerability
- Avg_Par_S{1-4}_NV, Avg_Par_S{1-4}_V: Average ParScore per seat, split by vulnerability
- Avg_EV_S{1-4}_NV, Avg_EV_S{1-4}_V: Average EV per seat, split by vulnerability

Output Files:
- bt_ev_par_stats_gpu.parquet: Per-BT stats with NV/V splits

Usage:
    python bbo_bt_ev_gpu.py                           # Stats only (~7 hrs)
    python bbo_bt_ev_gpu.py --resume                  # Resume from checkpoint
    python bbo_bt_ev_gpu.py --max-bt-rows 1000000     # Test run
    python bbo_bt_ev_gpu.py --build-index             # DEPRECATED - do not use

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
METRICS = ["HCP", "SL_S", "SL_H", "SL_D", "SL_C"]  # 5 dimensions for cube
CHECKPOINT_INTERVAL = 1_000_000  # Save every 1M BT rows
BATCH_SIZE = 1_000  # BT rows per GPU batch (tune based on VRAM)
BT_CHUNK_SIZE = 5_000_000  # Process BT in 5M row chunks to avoid memory explosion

# Default paths
DEFAULT_DEALS_FILE = Path("E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet")
DEFAULT_BT_FILE = Path("E:/bridge/data/bbo/bidding/bbo_bt_seat1.parquet")
DEFAULT_OUTPUT_FILE = Path("E:/bridge/data/bbo/bidding/bt_ev_par_stats_gpu.parquet")
DEFAULT_CHECKPOINT_DIR = Path("E:/bridge/data/bbo/bidding/checkpoints_ev_gpu")
DEFAULT_INDEX_DIR = Path("E:/bridge/data/bbo/bidding/inverted_index")

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
                         counts: np.ndarray, par_sums: np.ndarray, ev_sums: np.ndarray) -> None:
        """Save intermediate results for a direction/vul/seat."""
        filename = self.results_dir / f"results_{direction}_{vul}_S{seat}.npz"
        np.savez_compressed(filename, 
                           bt_indices=bt_indices,
                           counts=counts,
                           par_sums=par_sums,
                           ev_sums=ev_sums)
    
    def load_results_vul(self, direction: str, vul: str, seat: int) -> Optional[Dict]:
        """Load intermediate results if exists (with vul state)."""
        filename = self.results_dir / f"results_{direction}_{vul}_S{seat}.npz"
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

def build_histogram_cube(
    deals_file: Path,
    direction: str,
    max_deals: Optional[int] = None,
    vul_filter: Optional[str] = None,  # "NV" or "V" - filters by vulnerability for this direction's side
) -> pl.DataFrame:
    """
    Build histogram cube for a direction.
    
    Groups deals by (HCP, SL_S, SL_H, SL_D, SL_C) and pre-aggregates
    count, ParScore sum, and EV sum.
    
    Args:
        deals_file: Path to deals parquet file
        direction: N, E, S, or W
        max_deals: Optional limit on deals to process
        vul_filter: Optional "NV" or "V" to filter by vulnerability for this direction's side
                   - N/S directions: NV = Vul NOT IN ('N_S', 'Both'), V = Vul IN ('N_S', 'Both')
                   - E/W directions: NV = Vul NOT IN ('E_W', 'Both'), V = Vul IN ('E_W', 'Both')
    
    Returns DataFrame with ~200K-500K unique buckets.
    """
    t0 = time.time()
    
    # Columns needed
    dim_cols = [f"HCP_{direction}", f"SL_{direction}_S", f"SL_{direction}_H",
                f"SL_{direction}_D", f"SL_{direction}_C"]
    value_cols = ["ParScore", "EV_Score_Declarer"]
    
    # Check if columns exist
    schema = pl.scan_parquet(deals_file).collect_schema()
    available_cols = schema.names()
    
    # Use available ParScore/EV columns
    par_col = "ParScore" if "ParScore" in available_cols else None
    ev_col = "EV_Score_Declarer" if "EV_Score_Declarer" in available_cols else None
    
    if par_col is None:
        log(f"  WARNING: ParScore column not found, using zeros")
    if ev_col is None:
        log(f"  WARNING: EV_Score_Declarer column not found, using zeros")
    
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
    
    # Select columns to read
    read_cols = dim_cols.copy()
    if par_col:
        read_cols.append(par_col)
    if ev_col:
        read_cols.append(ev_col)
    
    # Add Vul column if filtering by vulnerability
    if vul_filter and "Vul" in available_cols:
        read_cols.append("Vul")
    
    # Build cube
    query = pl.scan_parquet(deals_file).select(read_cols)
    if max_deals:
        query = query.head(max_deals)
    
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
    
    cube = (
        query
        .group_by(dim_cols)
        .agg(agg_exprs)
        .collect()
    )
    
    # Rename columns for consistency
    cube = cube.rename({
        dim_cols[0]: "HCP",
        dim_cols[1]: "SL_S",
        dim_cols[2]: "SL_H",
        dim_cols[3]: "SL_D",
        dim_cols[4]: "SL_C",
    })
    
    vul_suffix = f"_{vul_filter}" if vul_filter else ""
    elapsed = time.time() - t0
    log(f"  Built cube for {direction}{vul_suffix}: {cube.height:,} buckets in {fmt_time(elapsed)}")
    
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
    }
    
    if not expr_str or expr_str == "None" or expr_str == "null":
        return {k: (v[0], v[1]) for k, v in bounds.items()}
    
    # Parse >= and <=
    for match in re.finditer(r"(HCP|SL_[SHDC])\s*>=\s*(\d+)", expr_str):
        metric, val = match.groups()
        bounds[metric][0] = max(bounds[metric][0], int(val))
    
    for match in re.finditer(r"(HCP|SL_[SHDC])\s*<=\s*(\d+)", expr_str):
        metric, val = match.groups()
        bounds[metric][1] = min(bounds[metric][1], int(val))
    
    return {k: (v[0], v[1]) for k, v in bounds.items()}


def extract_bt_bounds_polars(bt_df: pl.DataFrame, seat: int) -> Dict[str, np.ndarray]:
    """
    Extract bounds arrays from BT DataFrame using Polars expressions.
    
    MEMORY-EFFICIENT: Uses Polars regex extraction, no Python object creation.
    
    Returns dict with lo/hi arrays for each metric.
    """
    expr_col = f"Agg_Expr_Seat_{seat}"
    n = bt_df.height
    
    if expr_col not in bt_df.columns:
        log(f"  WARNING: {expr_col} not found, using defaults")
        return {
            "HCP_lo": np.zeros(n, dtype=np.int16),
            "HCP_hi": np.full(n, 40, dtype=np.int16),
            "SL_S_lo": np.zeros(n, dtype=np.int16),
            "SL_S_hi": np.full(n, 13, dtype=np.int16),
            "SL_H_lo": np.zeros(n, dtype=np.int16),
            "SL_H_hi": np.full(n, 13, dtype=np.int16),
            "SL_D_lo": np.zeros(n, dtype=np.int16),
            "SL_D_hi": np.full(n, 13, dtype=np.int16),
            "SL_C_lo": np.zeros(n, dtype=np.int16),
            "SL_C_hi": np.full(n, 13, dtype=np.int16),
        }
    
    # Join list of strings into single string per row (stays in Polars, no Python objects!)
    joined = bt_df.select(
        pl.col(expr_col).list.join(" AND ").alias("expr_str")
    )
    
    # Default bounds
    defaults = {
        "HCP": (0, 40),
        "SL_S": (0, 13),
        "SL_H": (0, 13),
        "SL_D": (0, 13),
        "SL_C": (0, 13),
    }
    
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


def extract_bt_bounds(bt_df: pl.DataFrame, seat: int) -> Dict[str, np.ndarray]:
    """
    Extract bounds arrays from BT DataFrame for a specific seat.
    
    Uses Polars-based extraction for memory efficiency.
    """
    return extract_bt_bounds_polars(bt_df, seat)


# ---------------------------------------------------------------------------
# GPU Matching
# ---------------------------------------------------------------------------

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Query histogram cube for all BT rows using GPU.
    
    Returns (counts, par_sums, ev_sums) for each BT row.
    
    If collect_index=True, also streams (bucket_idx, bt_idx) pairs to index_file.
    """
    n_bt = len(bt_indices)
    n_buckets = cube.height
    
    # Move cube data to GPU
    cube_hcp = torch.tensor(cube["HCP"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_s = torch.tensor(cube["SL_S"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_h = torch.tensor(cube["SL_H"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_d = torch.tensor(cube["SL_D"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_c = torch.tensor(cube["SL_C"].to_numpy(), device=device, dtype=torch.int16)
    cube_count = torch.tensor(cube["count"].to_numpy(), device=device, dtype=torch.int64)
    cube_par = torch.tensor(cube["par_sum"].to_numpy(), device=device, dtype=torch.int64)
    cube_ev = torch.tensor(cube["ev_sum"].to_numpy(), device=device, dtype=torch.int64)
    
    # Move BT bounds to GPU
    hcp_lo = torch.tensor(bt_bounds["HCP_lo"], device=device, dtype=torch.int16)
    hcp_hi = torch.tensor(bt_bounds["HCP_hi"], device=device, dtype=torch.int16)
    sl_s_lo = torch.tensor(bt_bounds["SL_S_lo"], device=device, dtype=torch.int16)
    sl_s_hi = torch.tensor(bt_bounds["SL_S_hi"], device=device, dtype=torch.int16)
    sl_h_lo = torch.tensor(bt_bounds["SL_H_lo"], device=device, dtype=torch.int16)
    sl_h_hi = torch.tensor(bt_bounds["SL_H_hi"], device=device, dtype=torch.int16)
    sl_d_lo = torch.tensor(bt_bounds["SL_D_lo"], device=device, dtype=torch.int16)
    sl_d_hi = torch.tensor(bt_bounds["SL_D_hi"], device=device, dtype=torch.int16)
    sl_c_lo = torch.tensor(bt_bounds["SL_C_lo"], device=device, dtype=torch.int16)
    sl_c_hi = torch.tensor(bt_bounds["SL_C_hi"], device=device, dtype=torch.int16)
    
    # Results
    result_counts = np.zeros(n_bt, dtype=np.int64)
    result_par = np.zeros(n_bt, dtype=np.int64)
    result_ev = np.zeros(n_bt, dtype=np.int64)
    
    # For inverted index: stream matches to file
    index_handle = None
    if collect_index and index_file:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index_handle = open(index_file, 'ab')  # Append binary
    
    # Process in batches
    for i in range(start_idx, n_bt, batch_size):
        end = min(i + batch_size, n_bt)
        
        # Get batch bounds (broadcast to match cube)
        batch_hcp_lo = hcp_lo[i:end].unsqueeze(1)
        batch_hcp_hi = hcp_hi[i:end].unsqueeze(1)
        batch_sl_s_lo = sl_s_lo[i:end].unsqueeze(1)
        batch_sl_s_hi = sl_s_hi[i:end].unsqueeze(1)
        batch_sl_h_lo = sl_h_lo[i:end].unsqueeze(1)
        batch_sl_h_hi = sl_h_hi[i:end].unsqueeze(1)
        batch_sl_d_lo = sl_d_lo[i:end].unsqueeze(1)
        batch_sl_d_hi = sl_d_hi[i:end].unsqueeze(1)
        batch_sl_c_lo = sl_c_lo[i:end].unsqueeze(1)
        batch_sl_c_hi = sl_c_hi[i:end].unsqueeze(1)
        
        # 5D mask: (batch_size, n_buckets)
        mask = (
            (cube_hcp >= batch_hcp_lo) & (cube_hcp <= batch_hcp_hi) &
            (cube_sl_s >= batch_sl_s_lo) & (cube_sl_s <= batch_sl_s_hi) &
            (cube_sl_h >= batch_sl_h_lo) & (cube_sl_h <= batch_sl_h_hi) &
            (cube_sl_d >= batch_sl_d_lo) & (cube_sl_d <= batch_sl_d_hi) &
            (cube_sl_c >= batch_sl_c_lo) & (cube_sl_c <= batch_sl_c_hi)
        )
        
        # Aggregate stats
        mask_int = mask.to(torch.int64)
        batch_counts = (mask_int * cube_count).sum(dim=1)
        batch_par = (mask_int * cube_par).sum(dim=1)
        batch_ev = (mask_int * cube_ev).sum(dim=1)
        
        # Copy back to CPU
        result_counts[i:end] = batch_counts.cpu().numpy()
        result_par[i:end] = batch_par.cpu().numpy()
        result_ev[i:end] = batch_ev.cpu().numpy()
        
        # Collect inverted index matches
        if index_handle is not None:
            # Find (bt_row, bucket) pairs where mask is True
            mask_cpu = mask.cpu().numpy()
            batch_bt_indices = bt_indices[i:end]
            
            # Get indices where mask is True: (row_in_batch, bucket_idx)
            match_rows, match_buckets = np.where(mask_cpu)
            
            if len(match_rows) > 0:
                # Convert batch-relative row to global bt_index
                matched_bt_indices = batch_bt_indices[match_rows].astype(np.uint32)
                matched_bucket_indices = match_buckets.astype(np.uint16)
                
                # Write as structured array for proper binary format
                dt = np.dtype([('bucket', np.uint16), ('bt_idx', np.uint32)])
                pairs = np.empty(len(match_rows), dtype=dt)
                pairs['bucket'] = matched_bucket_indices
                pairs['bt_idx'] = matched_bt_indices
                pairs.tofile(index_handle)
        
        if pbar:
            pbar.update(end - i)
    
    if index_handle is not None:
        index_handle.close()
    
    return result_counts, result_par, result_ev


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
    
    If build_index=True, also builds inverted index (bucket → BT indices).
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
    
    # ---------------------------------------------------------------------------
    # Step 1: Load BT core data (small - just indices)
    # ---------------------------------------------------------------------------
    log("\n[1/4] Loading BT core data...")
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
    # Step 2: Build histogram cubes (one per direction)
    # ---------------------------------------------------------------------------
    # Build 8 cubes: 2 per direction (NV and V splits)
    log("\n[2/4] Building histogram cubes (NV/V split)...")
    step_start = time.time()
    
    VUL_STATES = ["NV", "V"]
    cubes = {}  # Key: (direction, vul) tuple
    for direction in DIRECTIONS:
        for vul in VUL_STATES:
            cube_key = f"cube_{direction}_{vul}"
            if cube_key in state.get("completed", []):
                log(f"  Skipping {direction}_{vul} (already built)")
                # Load from saved parquet
                cube_file = checkpoint_dir / f"cube_{direction}_{vul}.parquet"
                if cube_file.exists():
                    cubes[(direction, vul)] = pl.read_parquet(cube_file)
                else:
                    cubes[(direction, vul)] = build_histogram_cube(deals_file, direction, max_deals, vul_filter=vul)
            else:
                cubes[(direction, vul)] = build_histogram_cube(deals_file, direction, max_deals, vul_filter=vul)
                # Save cube for restart
                cube_file = checkpoint_dir / f"cube_{direction}_{vul}.parquet"
                cubes[(direction, vul)].write_parquet(cube_file)
                state["completed"].append(cube_key)
                ckpt.save_state(state)
    
    elapsed = time.time() - step_start
    log(f"  Cubes built in {fmt_time(elapsed)} (8 cubes: 4 directions x 2 vul states)")
    
    # ---------------------------------------------------------------------------
    # Step 3: GPU query - organized by SEAT first (memory optimization)
    # This way we load one Agg_Expr_Seat_X column at a time instead of all 4
    # Now queries 8 cubes (4 directions x 2 vul states) per seat
    # ---------------------------------------------------------------------------
    log("\n[3/4] GPU matching (this is the long step)...")
    step_start = time.time()
    
    # Tasks now include vul state: query_{direction}_{vul}_S{seat}
    total_tasks = len(DIRECTIONS) * len(VUL_STATES) * 4  # 4 seats x 4 directions x 2 vul = 32 tasks
    completed_tasks = sum(1 for d in DIRECTIONS for v in VUL_STATES for s in range(1, 5) 
                         if f"query_{d}_{v}_S{s}" in state.get("completed", []))
    
    for seat in range(1, 5):
        # Check if ALL (direction, vul) pairs for this seat are done
        seat_tasks = [f"query_{d}_{v}_S{seat}" for d in DIRECTIONS for v in VUL_STATES]
        if all(t in state.get("completed", []) for t in seat_tasks):
            log(f"  Skipping seat {seat} (all direction/vul combos completed)")
            continue
        
        log(f"  Processing seat {seat} in chunks of {BT_CHUNK_SIZE:,} rows...")
        
        # Initialize result accumulators for all (direction, vul) pairs
        # Key: (direction, vul) tuple
        dir_vul_results = {}
        for d in DIRECTIONS:
            for v in VUL_STATES:
                if f"query_{d}_{v}_S{seat}" not in state.get("completed", []):
                    dir_vul_results[(d, v)] = {
                        "counts": np.zeros(n_bt, dtype=np.int64),
                        "par_sums": np.zeros(n_bt, dtype=np.int64),
                        "ev_sums": np.zeros(n_bt, dtype=np.int64),
                    }
        
        if not dir_vul_results:
            log(f"  All direction/vul combos for seat {seat} already completed")
            continue
        
        # Process BT in chunks to avoid memory explosion
        chunk_size = BT_CHUNK_SIZE
        n_chunks = (n_bt + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_bt)
            chunk_len = chunk_end - chunk_start
            
            log(f"    Chunk {chunk_idx + 1}/{n_chunks}: rows {chunk_start:,}-{chunk_end:,}")
            
            # Load ONLY this chunk's expression column
            chunk_df = pl.read_parquet(
                bt_file,
                columns=["bt_index", f"Agg_Expr_Seat_{seat}"]
            ).slice(chunk_start, chunk_len)
            
            # Parse bounds for this chunk (uses Polars, memory-efficient)
            chunk_bounds = extract_bt_bounds(chunk_df, seat)
            chunk_bt_indices = chunk_df["bt_index"].to_numpy()
            del chunk_df
            gc.collect()
            
            # Query each (direction, vul) cube with this chunk
            for (direction, vul) in dir_vul_results.keys():
                cube = cubes[(direction, vul)]
                
                # Set up index file for this chunk (if building index)
                # Index files now include vul state
                chunk_index_file = None
                if build_index and index_dir:
                    raw_index_dir = index_dir / "raw" / f"{direction}_{vul}_S{seat}"
                    raw_index_dir.mkdir(parents=True, exist_ok=True)
                    chunk_index_file = raw_index_dir / f"chunk_{chunk_idx:04d}.bin"
                
                # GPU query for this chunk (no progress bar for chunks)
                counts, par_sums, ev_sums = gpu_query_cube(
                    cube, chunk_bounds, chunk_bt_indices, device, batch_size,
                    collect_index=build_index,
                    index_file=chunk_index_file,
                )
                
                # Accumulate into results
                dir_vul_results[(direction, vul)]["counts"][chunk_start:chunk_end] = counts
                dir_vul_results[(direction, vul)]["par_sums"][chunk_start:chunk_end] = par_sums
                dir_vul_results[(direction, vul)]["ev_sums"][chunk_start:chunk_end] = ev_sums
                
                del counts, par_sums, ev_sums
            
            # Free chunk data
            del chunk_bounds, chunk_bt_indices
            gc.collect()
            torch.cuda.empty_cache()
        
        # Save results for each (direction, vul) pair
        for (direction, vul), results in dir_vul_results.items():
            task_key = f"query_{direction}_{vul}_S{seat}"
            
            log(f"  Saving {direction}_{vul}/S{seat}...")
            # Use modified save that includes vul in the key
            ckpt.save_results_vul(
                direction, vul, seat, bt_indices,
                results["counts"], results["par_sums"], results["ev_sums"]
            )
            
            state["completed"].append(task_key)
            state["last_completed"] = task_key
            completed_tasks += 1
        
        ckpt.save_state(state)
        log(f"  Seat {seat} completed ({completed_tasks}/{total_tasks} tasks)")
        
        # Free all results for this seat
        del dir_vul_results
        gc.collect()
    
    elapsed = time.time() - step_start
    log(f"  All GPU queries completed in {fmt_time(elapsed)}")
    
    # ---------------------------------------------------------------------------
    # Step 4: Combine results and compute averages (NV/V split)
    # ---------------------------------------------------------------------------
    log("\n[4/4] Combining results (NV/V split)...")
    step_start = time.time()
    
    # bt_indices already set from Step 1
    n = len(bt_indices)
    
    result_cols = {"bt_index": bt_indices}
    
    for seat in range(1, 5):
        for vul in VUL_STATES:
            log(f"  Aggregating seat {seat} {vul}...")
            # Aggregate across all directions for this seat/vul combo
            total_count = np.zeros(n, dtype=np.int64)
            total_par = np.zeros(n, dtype=np.int64)
            total_ev = np.zeros(n, dtype=np.int64)
            
            for direction in DIRECTIONS:
                # Load results from disk (streaming, not holding all in memory)
                data = ckpt.load_results_vul(direction, vul, seat)
                if data is not None:
                    total_count += data["counts"]
                    total_par += data["par_sums"]
                    total_ev += data["ev_sums"]
                    del data  # Free immediately
                else:
                    log(f"    WARNING: Missing results for {direction}_{vul}/S{seat}")
            
            # Compute averages (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_par = np.where(total_count > 0, total_par / total_count, np.nan)
                avg_ev = np.where(total_count > 0, total_ev / total_count, np.nan)
            
            result_cols[f"Count_S{seat}_{vul}"] = total_count.astype(np.uint32)
            result_cols[f"Avg_Par_S{seat}_{vul}"] = avg_par.astype(np.float32)
            result_cols[f"Avg_EV_S{seat}_{vul}"] = avg_ev.astype(np.float32)
    
    result_df = pl.DataFrame(result_cols)
    result_df = result_df.sort("bt_index")
    
    elapsed = time.time() - step_start
    log(f"  Combined {n:,} rows in {fmt_time(elapsed)} (24 columns: 4 seats x 2 vul x 3 metrics)")
    
    # ---------------------------------------------------------------------------
    # Step 5: Write stats output
    # ---------------------------------------------------------------------------
    total_steps = 6 if build_index else 5
    log(f"\n[5/{total_steps}] Writing stats output...")
    step_start = time.time()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_file)
    
    file_size = output_file.stat().st_size / 1e6
    elapsed = time.time() - step_start
    log(f"  Written {output_file} ({file_size:.1f} MB) in {fmt_time(elapsed)}")
    
    # ---------------------------------------------------------------------------
    # Step 6: Consolidate inverted index (if enabled)
    # ---------------------------------------------------------------------------
    if build_index and index_dir:
        log(f"\n[6/{total_steps}] Consolidating inverted index...")
        step_start = time.time()
        
        for seat in range(1, 5):
            for direction in DIRECTIONS:
                for vul in VUL_STATES:
                    log(f"  Processing {direction}_{vul}/S{seat}...")
                    consolidate_inverted_index_vul(index_dir, cubes[(direction, vul)], direction, vul, seat)
        
        elapsed = time.time() - step_start
        log(f"  Index consolidation completed in {fmt_time(elapsed)}")
    
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
    if build_index and index_dir:
        log(f"Index:   {index_dir}")
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
        ev_nv_str = f"{ev_nv:.1f}" if ev_nv is not None and not (isinstance(ev_nv, float) and ev_nv != ev_nv) else "N/A"
        ev_v_str = f"{ev_v:.1f}" if ev_v is not None and not (isinstance(ev_v, float) and ev_v != ev_v) else "N/A"
        log(f"  bt_index={row['bt_index']}, "
            f"Count_NV={count_nv:,}, Count_V={count_v:,}, "
            f"EV_NV={ev_nv_str}, EV_V={ev_v_str}")


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
        help="DEPRECATED: Build inverted index (~500GB+ disk, to be removed)"
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
