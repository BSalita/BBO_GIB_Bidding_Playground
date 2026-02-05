#!/usr/bin/env python3
"""
GPU-Accelerated Deal-to-BT Matching (Memory-Optimized v3)

For each deal, finds all completed BT auctions whose criteria match.
Uses histogram cube approach for efficiency (same as bbo_bt_ev_gpu.py).

Two-phase approach:
1. GPU: Build bucket→[bt_indices] mapping via histogram cube matching
2. CPU: Group deals by bucket-tuple, compute intersection once per group

Criteria model / invariant:
- This pipeline assumes BT criteria in `Agg_Expr_Seat_1..4` are **AND-only**: a list of atomic predicates that must all be true.
- Phase 1 extracts only simple `HCP/SL_* >=/<=` bounds from those predicates and uses them for range matching.
- If OR/complex predicates were introduced into BT `Agg_Expr_*`, Phase 1 could become unsound (miss valid matches).
  Complex OR logic is expected to be enforced downstream (e.g., `data/bbo_custom_auction_criteria.csv` in the API layer),
  not baked into the GPU pipeline.

Memory optimization v3:
- Uses NumPy arrays instead of Python dicts/lists
- Loads one Agg_Expr column at a time
- Groups deals by bucket-tuple for O(groups) instead of O(deals) intersections
- Sorts bt_ids within buckets for fast intersection (assume_unique=True)

Performance (RTX 5080, 16GB VRAM, 192GB RAM):
- Runtime: ~2.8 hours
- Peak memory: ~180 GB
- Step 3 (GPU matching): ~1.2 hours
- Step 4 (intersections): ~1.3 hours at 230 groups/sec

Output: deal_to_bt_gpu.parquet
    Columns:
        - deal_idx (UInt32): Index into bbo_mldf_augmented.parquet (0 to 15,994,826)
        - Matched_BT_Indices (List[UInt32]): List of bt_index values from 
          bbo_bt_seat1.parquet that match this deal's hand criteria for all 4 seats
    
    Statistics (full run):
        - 15,994,827 rows (one per deal)
        - Average 1,522.8 matches per deal
        - Max 20,201 matches per deal
        - 0% deals with zero matches (all deals match some completed auctions)
        - File size: ~89 GB

Usage:
    python bbo_bt_deal_matches_gpu.py                    # Full run
    python bbo_bt_deal_matches_gpu.py --max-deals 10000  # Test run
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

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
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)

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
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}

# Processing parameters
BATCH_SIZE = 2000  # BT rows per GPU batch

# Default paths
DEFAULT_DEALS_FILE = Path("E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet")
DEFAULT_BT_FILE = Path("E:/bridge/data/bbo/bidding/bbo_bt_seat1.parquet")
DEFAULT_OUTPUT_FILE = Path("E:/bridge/data/bbo/bidding/bbo_deal_to_bt_gpu.parquet")
DEFAULT_CHECKPOINT_DIR = Path("E:/bridge/data/bbo/bidding/checkpoints_deal_matches_gpu")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def fmt_mem(bytes_val: int) -> str:
    """Format bytes as human-readable."""
    if bytes_val < 1e9:
        return f"{bytes_val/1e6:.1f} MB"
    return f"{bytes_val/1e9:.1f} GB"

def get_memory_usage() -> int:
    """Get current process memory usage in bytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0

# ---------------------------------------------------------------------------
# Histogram Cube Builder
# ---------------------------------------------------------------------------

def build_histogram_cube(
    deals_file: Path,
    direction: str,
    max_deals: Optional[int] = None,
) -> pl.DataFrame:
    """
    Build histogram cube for a direction.
    Groups deals by (HCP, SL_S, SL_H, SL_D, SL_C).
    """
    dim_cols = [f"HCP_{direction}", f"SL_{direction}_S", f"SL_{direction}_H",
                f"SL_{direction}_D", f"SL_{direction}_C"]
    
    query = pl.scan_parquet(deals_file).select(dim_cols)
    if max_deals:
        query = query.head(max_deals)
    
    cube = (
        query
        .group_by(dim_cols)
        .agg([pl.len().alias("count")])
        .collect()
    )
    
    # Rename for consistency
    cube = cube.rename({
        dim_cols[0]: "HCP",
        dim_cols[1]: "SL_S",
        dim_cols[2]: "SL_H",
        dim_cols[3]: "SL_D",
        dim_cols[4]: "SL_C",
    })
    
    return cube

# ---------------------------------------------------------------------------
# BT Bounds Extraction - from single Agg_Expr column
# ---------------------------------------------------------------------------

def extract_bounds_from_agg_expr(
    bt_file: Path,
    bt_indices_filter: np.ndarray,
    seat: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load one Agg_Expr_Seat_X column and extract bounds.
    """
    expr_col = f"Agg_Expr_Seat_{seat}"
    
    bt_df = (
        pl.scan_parquet(bt_file)
        .select(["bt_index", expr_col])
        .filter(pl.col("bt_index").is_in(bt_indices_filter.tolist()))
        .collect()
    )
    
    n = bt_df.height
    bt_indices = bt_df["bt_index"].to_numpy().astype(np.uint32)
    
    joined = bt_df.select(
        pl.col(expr_col).list.join(" AND ").alias("expr_str")
    )
    
    defaults = {
        "HCP": (0, 40),
        "SL_S": (0, 13),
        "SL_H": (0, 13),
        "SL_D": (0, 13),
        "SL_C": (0, 13),
    }
    
    extract_exprs = []
    for metric, (default_lo, default_hi) in defaults.items():
        lo_pattern = f"{metric}\\s*>=\\s*(\\d+)"
        extract_exprs.append(
            pl.col("expr_str")
            .str.extract(lo_pattern, group_index=1)
            .cast(pl.Int16)
            .fill_null(default_lo)
            .alias(f"{metric}_lo")
        )
        hi_pattern = f"{metric}\\s*<=\\s*(\\d+)"
        extract_exprs.append(
            pl.col("expr_str")
            .str.extract(hi_pattern, group_index=1)
            .cast(pl.Int16)
            .fill_null(default_hi)
            .alias(f"{metric}_hi")
        )
    
    bounds_df = joined.select(extract_exprs)
    
    result = {}
    for metric in defaults.keys():
        result[f"{metric}_lo"] = bounds_df[f"{metric}_lo"].to_numpy()
        result[f"{metric}_hi"] = bounds_df[f"{metric}_hi"].to_numpy()
    
    del bt_df, joined, bounds_df
    gc.collect()
    
    return bt_indices, result

# ---------------------------------------------------------------------------
# GPU Bucket-to-BT Matching (NumPy-based storage)
# ---------------------------------------------------------------------------

def gpu_match_bucket_to_bt_numpy(
    cube: pl.DataFrame,
    bt_bounds: Dict[str, np.ndarray],
    bt_indices: np.ndarray,
    device: Any,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU match: for each bucket, find which BT rows match.
    
    Returns: (bucket_ids, bt_ids) - parallel numpy arrays
    """
    n_bt = len(bt_indices)
    
    # Move cube data to GPU
    cube_hcp = torch.tensor(cube["HCP"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_s = torch.tensor(cube["SL_S"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_h = torch.tensor(cube["SL_H"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_d = torch.tensor(cube["SL_D"].to_numpy(), device=device, dtype=torch.int16)
    cube_sl_c = torch.tensor(cube["SL_C"].to_numpy(), device=device, dtype=torch.int16)
    
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
    
    all_bucket_ids = []
    all_bt_ids = []
    
    for i in range(0, n_bt, batch_size):
        end = min(i + batch_size, n_bt)
        
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
        
        mask = (
            (cube_hcp >= batch_hcp_lo) & (cube_hcp <= batch_hcp_hi) &
            (cube_sl_s >= batch_sl_s_lo) & (cube_sl_s <= batch_sl_s_hi) &
            (cube_sl_h >= batch_sl_h_lo) & (cube_sl_h <= batch_sl_h_hi) &
            (cube_sl_d >= batch_sl_d_lo) & (cube_sl_d <= batch_sl_d_hi) &
            (cube_sl_c >= batch_sl_c_lo) & (cube_sl_c <= batch_sl_c_hi)
        )
        
        mask_cpu = mask.cpu().numpy()
        batch_bt_indices = bt_indices[i:end]
        
        match_rows, match_buckets = np.where(mask_cpu)
        
        if len(match_rows) > 0:
            matched_bt_ids = batch_bt_indices[match_rows]
            all_bucket_ids.append(match_buckets.astype(np.int32))
            all_bt_ids.append(matched_bt_ids.astype(np.uint32))
    
    if all_bucket_ids:
        bucket_ids = np.concatenate(all_bucket_ids)
        bt_ids = np.concatenate(all_bt_ids)
    else:
        bucket_ids = np.array([], dtype=np.int32)
        bt_ids = np.array([], dtype=np.uint32)
    
    return bucket_ids, bt_ids


def build_bucket_bt_index(bucket_ids: np.ndarray, bt_ids: np.ndarray, n_buckets: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an index for fast bucket→bt_ids lookup.
    bt_ids within each bucket are SORTED for fast intersection.
    """
    if len(bucket_ids) == 0:
        return np.zeros(n_buckets + 1, dtype=np.int64), np.array([], dtype=np.uint32)
    
    # First sort by bucket
    sort_idx = np.argsort(bucket_ids)
    bucket_ids_sorted = bucket_ids[sort_idx]
    bt_ids_sorted = bt_ids[sort_idx]
    
    counts = np.bincount(bucket_ids_sorted, minlength=n_buckets)
    
    offsets = np.zeros(n_buckets + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    
    # Now sort bt_ids WITHIN each bucket for fast intersection
    for bucket_idx in range(n_buckets):
        start = offsets[bucket_idx]
        end = offsets[bucket_idx + 1]
        if end > start:
            bt_ids_sorted[start:end] = np.sort(bt_ids_sorted[start:end])
    
    return offsets, bt_ids_sorted

# ---------------------------------------------------------------------------
# Deal to Bucket Assignment
# ---------------------------------------------------------------------------

def load_deal_bucket_indices(
    deals_file: Path,
    cubes: Dict[str, pl.DataFrame],
    max_deals: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Load deals and assign each to buckets in all 4 direction cubes.
    """
    cols_needed = ["Dealer"]
    for d in DIRECTIONS:
        cols_needed.extend([f"HCP_{d}", f"SL_{d}_S", f"SL_{d}_H", f"SL_{d}_D", f"SL_{d}_C"])
    
    query = pl.scan_parquet(deals_file).select(cols_needed)
    if max_deals:
        query = query.head(max_deals)
    deals = query.collect()
    
    n_deals = deals.height
    deal_indices = np.arange(n_deals, dtype=np.uint32)
    
    dealer_map = {"N": 0, "E": 1, "S": 2, "W": 3}
    dealers = deals["Dealer"].map_elements(lambda x: dealer_map.get(x, 0), return_dtype=pl.Int8).to_numpy()
    
    bucket_indices: Dict[str, np.ndarray] = {}
    
    for direction in DIRECTIONS:
        cube = cubes[direction]
        n_buckets = cube.height
        
        bucket_lookup = {}
        for idx in range(n_buckets):
            key = (
                int(cube["HCP"][idx]),
                int(cube["SL_S"][idx]),
                int(cube["SL_H"][idx]),
                int(cube["SL_D"][idx]),
                int(cube["SL_C"][idx]),
            )
            bucket_lookup[key] = idx
        
        hcp = deals[f"HCP_{direction}"].to_numpy()
        sl_s = deals[f"SL_{direction}_S"].to_numpy()
        sl_h = deals[f"SL_{direction}_H"].to_numpy()
        sl_d = deals[f"SL_{direction}_D"].to_numpy()
        sl_c = deals[f"SL_{direction}_C"].to_numpy()
        
        bucket_arr = np.zeros(n_deals, dtype=np.int32)
        for i in range(n_deals):
            key = (int(hcp[i]), int(sl_s[i]), int(sl_h[i]), int(sl_d[i]), int(sl_c[i]))
            bucket_arr[i] = bucket_lookup.get(key, -1)
        
        bucket_indices[direction] = bucket_arr
    
    return deal_indices, bucket_indices, dealers

# ---------------------------------------------------------------------------
# Fast intersection using sorted arrays
# ---------------------------------------------------------------------------

def intersect_sorted_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    """Intersect multiple sorted uint32 arrays. Arrays must be pre-sorted!"""
    if not arrays:
        return np.array([], dtype=np.uint32)
    
    # Start with smallest array for efficiency
    arrays = sorted(arrays, key=len)
    
    result = arrays[0]
    for arr in arrays[1:]:
        if len(result) == 0:
            return np.array([], dtype=np.uint32)
        # assume_unique=True is MUCH faster for sorted arrays
        result = np.intersect1d(result, arr, assume_unique=True)
    
    return result

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    deals_file: Path,
    bt_file: Path,
    output_file: Path,
    checkpoint_dir: Path,
    max_deals: Optional[int] = None,
    max_bt_rows: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Run the GPU deal-to-BT matching pipeline."""
    
    pipeline_start = time.time()
    log("=" * 70)
    log("GPU Deal-to-BT Matching Pipeline (Memory-Optimized v3)")
    log("=" * 70)
    log(f"Deals: {deals_file}")
    log(f"BT: {bt_file}")
    log(f"Output: {output_file}")
    if max_deals:
        log(f"Max deals: {max_deals:,}")
    if max_bt_rows:
        log(f"Max BT rows: {max_bt_rows:,}")
    log(f"Initial memory: {fmt_mem(get_memory_usage())}")
    log("=" * 70)
    
    # Check GPU
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # -------------------------------------------------------------------------
    # Step 1: Get completed auction bt_indices (lightweight load)
    # -------------------------------------------------------------------------
    log("\n[1/5] Loading completed auction indices...")
    step_start = time.time()
    
    schema = pl.read_parquet_schema(bt_file)
    if "is_completed_auction" in schema:
        bt_indices_df = (
            pl.scan_parquet(bt_file)
            .filter(pl.col("is_completed_auction"))
            .select(["bt_index"])
            .collect()
        )
    else:
        bt_indices_df = (
            pl.scan_parquet(bt_file)
            .filter(pl.col("Auction").str.ends_with("-p-p-p") | (pl.col("Auction") == "p-p-p-p"))
            .select(["bt_index"])
            .collect()
        )
    
    if max_bt_rows:
        bt_indices_df = bt_indices_df.head(max_bt_rows)
    
    n_bt = bt_indices_df.height
    bt_indices_all = bt_indices_df["bt_index"].to_numpy().astype(np.uint32)
    
    del bt_indices_df
    gc.collect()
    
    log(f"  Found {n_bt:,} completed auctions in {fmt_time(time.time() - step_start)}")
    log(f"  Memory: {fmt_mem(get_memory_usage())}")
    
    # -------------------------------------------------------------------------
    # Step 2: Build histogram cubes (one per direction)
    # -------------------------------------------------------------------------
    log("\n[2/5] Building histogram cubes...")
    step_start = time.time()
    
    cubes = {}
    for direction in DIRECTIONS:
        cubes[direction] = build_histogram_cube(deals_file, direction, max_deals)
        log(f"  {direction}: {cubes[direction].height:,} buckets")
    
    log(f"  Cubes built in {fmt_time(time.time() - step_start)}")
    log(f"  Memory: {fmt_mem(get_memory_usage())}")
    
    # -------------------------------------------------------------------------
    # Step 3: GPU match buckets to BT (per seat) - load one Agg_Expr at a time
    # -------------------------------------------------------------------------
    log("\n[3/5] GPU matching buckets to BT...")
    step_start = time.time()
    
    bucket_bt_index: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {
        d: {} for d in DIRECTIONS
    }
    
    total_pairs = 0
    
    for seat in range(1, 5):
        log(f"  Processing seat {seat}...")
        seat_start = time.time()
        
        bt_indices, bounds = extract_bounds_from_agg_expr(bt_file, bt_indices_all, seat)
        log(f"    Loaded Agg_Expr_Seat_{seat}, memory: {fmt_mem(get_memory_usage())}")
        
        for direction in DIRECTIONS:
            cube = cubes[direction]
            n_buckets = cube.height
            
            bucket_ids, bt_ids = gpu_match_bucket_to_bt_numpy(
                cube, bounds, bt_indices, device, batch_size
            )
            
            n_pairs = len(bucket_ids)
            total_pairs += n_pairs
            
            offsets, bt_ids_sorted = build_bucket_bt_index(bucket_ids, bt_ids, n_buckets)
            bucket_bt_index[direction][seat] = (offsets, bt_ids_sorted)
            
            del bucket_ids, bt_ids
        
        del bounds, bt_indices
        gc.collect()
        torch.cuda.empty_cache()
        log(f"    Seat {seat} done in {fmt_time(time.time() - seat_start)}, memory: {fmt_mem(get_memory_usage())}")
    
    log(f"  Total (bucket, bt) pairs: {total_pairs:,}")
    log(f"  GPU matching completed in {fmt_time(time.time() - step_start)}")
    log(f"  Memory: {fmt_mem(get_memory_usage())}")
    
    # -------------------------------------------------------------------------
    # Step 4: Load deals, GROUP by bucket-tuple, compute intersection per group
    # -------------------------------------------------------------------------
    log("\n[4/5] Loading deals and matching (grouped by bucket-tuple)...")
    step_start = time.time()
    
    deal_indices, bucket_indices, dealers = load_deal_bucket_indices(
        deals_file, cubes, max_deals
    )
    n_deals = len(deal_indices)
    
    log(f"  Loaded {n_deals:,} deals")
    log(f"  Memory: {fmt_mem(get_memory_usage())}")
    
    # Precompute seat→direction mapping for each dealer
    # seat_to_dir[dealer_idx] = [dir for seat 1, dir for seat 2, dir for seat 3, dir for seat 4]
    seat_to_dir = {}
    for dealer_idx, dealer in enumerate(DIRECTIONS):
        seat_to_dir[dealer_idx] = [
            DIRECTIONS[(dealer_idx + seat - 1) % 4] for seat in range(1, 5)
        ]
    
    # -------------------------------------------------------------------------
    # KEY OPTIMIZATION: Group deals by (dealer, bucket_seat1, bucket_seat2, bucket_seat3, bucket_seat4)
    # All deals in the same group have identical matching BT indices!
    # -------------------------------------------------------------------------
    log("  Grouping deals by bucket-tuple...")
    group_start = time.time()
    
    # Build bucket-tuple for each deal
    # For deal i with dealer d:
    #   seat 1 uses direction seat_to_dir[d][0], bucket = bucket_indices[dir][i]
    #   etc.
    
    # Create tuple keys: (dealer, b1, b2, b3, b4)
    tuple_keys = []
    for i in range(n_deals):
        d = int(dealers[i])
        dirs = seat_to_dir[d]
        b1 = int(bucket_indices[dirs[0]][i])
        b2 = int(bucket_indices[dirs[1]][i])
        b3 = int(bucket_indices[dirs[2]][i])
        b4 = int(bucket_indices[dirs[3]][i])
        tuple_keys.append((d, b1, b2, b3, b4))
    
    # Group deals by tuple
    tuple_to_deals: Dict[Tuple[int, int, int, int, int], List[int]] = defaultdict(list)
    for i, key in enumerate(tuple_keys):
        tuple_to_deals[key].append(i)
    
    n_groups = len(tuple_to_deals)
    log(f"  Created {n_groups:,} unique bucket-tuples from {n_deals:,} deals")
    log(f"  Average deals per group: {n_deals / n_groups:.1f}")
    log(f"  Grouping took {fmt_time(time.time() - group_start)}")
    log(f"  Memory: {fmt_mem(get_memory_usage())}")
    
    # -------------------------------------------------------------------------
    # Compute intersection ONCE per group (not once per deal!)
    # -------------------------------------------------------------------------
    log("  Computing intersections per group...")
    
    # Pre-allocate result array
    match_lists: List[List[int]] = [[] for _ in range(n_deals)]
    
    pbar = tqdm(total=n_groups, desc="Groups", unit="grp") if HAS_TQDM else None
    
    groups_with_matches = 0
    total_matches = 0
    
    for (dealer_idx, b1, b2, b3, b4), deal_list in tuple_to_deals.items():
        # Check for invalid buckets
        if b1 < 0 or b2 < 0 or b3 < 0 or b4 < 0:
            # Deals with invalid buckets get empty match list
            if pbar:
                pbar.update(1)
            continue
        
        dirs = seat_to_dir[dealer_idx]
        buckets = [b1, b2, b3, b4]
        
        # Get BT indices for each seat's bucket
        seat_matches = []
        for seat_idx in range(4):
            direction = dirs[seat_idx]
            bucket_idx = buckets[seat_idx]
            
            offsets, bt_ids_sorted = bucket_bt_index[direction][seat_idx + 1]
            start = offsets[bucket_idx]
            end = offsets[bucket_idx + 1]
            bt_list = bt_ids_sorted[start:end]
            seat_matches.append(bt_list)
        
        # Intersection of all 4 seats
        if all(len(s) > 0 for s in seat_matches):
            matching = intersect_sorted_arrays(seat_matches)
            matching_list = matching.tolist()
            
            if len(matching_list) > 0:
                groups_with_matches += 1
                total_matches += len(matching_list) * len(deal_list)
                
                # Assign same result to ALL deals in this group
                for deal_idx in deal_list:
                    match_lists[deal_idx] = matching_list
        
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    log(f"  Groups with matches: {groups_with_matches:,} ({100*groups_with_matches/n_groups:.1f}%)")
    log(f"  Deal matching completed in {fmt_time(time.time() - step_start)}")
    log(f"  Memory: {fmt_mem(get_memory_usage())}")
    
    # -------------------------------------------------------------------------
    # Step 5: Save output
    # -------------------------------------------------------------------------
    log("\n[5/5] Saving output...")
    step_start = time.time()
    
    # Stats
    match_counts = [len(m) for m in match_lists]
    avg_matches = sum(match_counts) / len(match_counts) if match_counts else 0
    max_matches = max(match_counts) if match_counts else 0
    zero_matches = sum(1 for c in match_counts if c == 0)
    
    log(f"  Match statistics:")
    log(f"    Average matches per deal: {avg_matches:.1f}")
    log(f"    Max matches per deal: {max_matches:,}")
    log(f"    Deals with zero matches: {zero_matches:,} ({100*zero_matches/n_deals:.1f}%)")
    
    output_df = pl.DataFrame({
        "deal_idx": deal_indices,
        "Matched_BT_Indices": pl.Series(match_lists, dtype=pl.List(pl.UInt32)),
    })
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_file)
    
    file_size = output_file.stat().st_size / 1e6
    log(f"  Written {output_file} ({file_size:.1f} MB) in {fmt_time(time.time() - step_start)}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)
    log(f"Elapsed: {fmt_time(total_elapsed)}")
    log(f"Output: {output_file}")
    log(f"Deals: {n_deals:,}")
    log(f"Unique bucket-tuples: {n_groups:,}")
    log(f"Avg matches per deal: {avg_matches:.1f}")
    log(f"Peak memory: {fmt_mem(get_memory_usage())}")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated deal-to-BT matching (memory-optimized v3)"
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
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--max-deals", type=int, default=None,
        help="Maximum deals to process (for testing)"
    )
    parser.add_argument(
        "--max-bt-rows", type=int, default=None,
        help="Maximum BT rows to process (for testing)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"GPU batch size (default: {BATCH_SIZE})"
    )
    
    args = parser.parse_args()
    
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
        max_deals=args.max_deals,
        max_bt_rows=args.max_bt_rows,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
