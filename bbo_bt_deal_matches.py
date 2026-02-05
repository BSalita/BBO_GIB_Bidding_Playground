"""
BT Deal Matches Pipeline (Optimized)

For each deal, finds all completed BT auctions whose criteria match.

Optimizations:
1. Block-Matrix Filtering: Processes ranges in cache-friendly blocks to minimize memory bandwidth.
2. Flattened Feature Arrays: Pre-converts all deal features and range limits to dense int8 matrices.
3. Chunked Disk Persistence: Writes results immediately to disk to keep RAM usage constant.
4. Zero-Copy Bitmaps: Uses memory-mapped files for large bitmap tables.
5. Restartable: Resumes from last checkpoint if interrupted (use --no-resume to force fresh start).
"""

import argparse
import pathlib
import sys
import os
import time
import re
import json
import pickle
import tempfile
import concurrent.futures
import shutil
from concurrent.futures import FIRST_COMPLETED
from datetime import datetime
from typing import Dict, List, Tuple, Any, cast
import polars as pl
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = ["N", "E", "S", "W"]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}
# Order matters: must match how we flatten feature arrays
# 4 seats * 6 metrics = 24 features per deal
RANGE_METRICS = ["HCP", "SL_S", "SL_H", "SL_D", "SL_C", "Total_Points"]

DEFAULT_RANGES = {
    "HCP": (0, 37), "SL_S": (0, 13), "SL_H": (0, 13),
    "SL_D": (0, 13), "SL_C": (0, 13), "Total_Points": (0, 40),
}

PATTERN_GE = re.compile(r"(\w+)\s*>=\s*(\d+)")
PATTERN_LE = re.compile(r"(\w+)\s*<=\s*(\d+)")
PATTERN_EQ = re.compile(r"(\w+)\s*==\s*(\d+)")
PATTERN_GT = re.compile(r"(\w+)\s*>\s*(\d+)")
PATTERN_LT = re.compile(r"(\w+)\s*<\s*(\d+)")

# ---------------------------------------------------------------------------
# Worker Shared State
# ---------------------------------------------------------------------------
_worker_data = {}

def init_worker(
    resource_file: str,
    bitmap_file: str,
    bitmap_shape: Tuple[int, int],
    criteria_file: str,
    criteria_count: int,
    offset_file: str,
    offset_count: int,
    range_matrix_file: str,
    range_matrix_shape: Tuple[int, int, int]  # (num_groups, 24, 2)
):
    """Initialize worker process by attaching to all shared memmaps."""
    global _worker_data
    with open(resource_file, 'rb') as f:
        _worker_data.update(pickle.load(f))
    
    # 1. Attach to bitmaps (bool)
    _worker_data['bitmap_memmap'] = np.memmap(
        bitmap_file, dtype=bool, mode='r', shape=bitmap_shape
    )
    
    # 2. Attach to Flat Criteria Indices (uint16)
    _worker_data['crit_indices'] = np.memmap(criteria_file, dtype=np.uint16, mode='r', shape=(criteria_count,))
    
    # 3. Attach to Offsets (uint32)
    _worker_data['crit_offsets'] = np.memmap(offset_file, dtype=np.uint32, mode='r', shape=(offset_count,))

    # 4. Attach to Dense Range Matrix (int8)
    # Shape: (num_groups, 24, 2) -> 24 features (4 seats * 6 metrics), 2 values (lo, hi)
    _worker_data['range_matrix'] = np.memmap(range_matrix_file, dtype=np.int8, mode='r', shape=range_matrix_shape)

def process_one_chunk(chunk_args):
    """Worker function: read deal chunk -> block filter -> bitmap verify -> write parquet."""
    chunk_idx, chunk_start, chunk_len, out_parquet = chunk_args
    
    group_to_bt_indices = _worker_data['group_to_bt_indices']
    bt_idx_to_auction = _worker_data['bt_idx_to_auction']
    bt_idx_map = _worker_data['bt_idx_map'] # bt_index -> internal_0_to_N_idx
    bitmap_memmap = _worker_data['bitmap_memmap']
    range_matrix = _worker_data['range_matrix'] # (G, 24, 2)
    crit_indices = _worker_data['crit_indices']
    crit_offsets = _worker_data['crit_offsets']
    deal_file = _worker_data["deal_file"]
    matching_cols = _worker_data["matching_cols"]
    
    # 1. Read Deal Chunk
    chunk_df = (
        pl.scan_parquet(deal_file)
        .select(matching_cols)
        .slice(chunk_start, chunk_len)
        .collect()
    )
    num_deals = chunk_df.height
    num_groups = range_matrix.shape[0]
    
    # 2. Flatten Deal Features into dense (N, 24) int8 array
    # Order: S1_HCP, S1_SL_S, ..., S4_TP
    # We normalized seats relative to Dealer: N=S1, E=S2...
    deal_features = np.zeros((num_deals, 24), dtype=np.int8)
    
    # Normalize features relative to Dealer
    # To do this fast, we can use Polars expressions to create the 24 cols, then to_numpy
    norm_exprs = []
    for seat in range(1, 5):
        for metric in RANGE_METRICS:
            expr = (
                pl.when(pl.col("Dealer") == "N").then(pl.col(get_col_for_metric(metric, get_seat_direction("N", seat))))
                .when(pl.col("Dealer") == "E").then(pl.col(get_col_for_metric(metric, get_seat_direction("E", seat))))
                .when(pl.col("Dealer") == "S").then(pl.col(get_col_for_metric(metric, get_seat_direction("S", seat))))
                .when(pl.col("Dealer") == "W").then(pl.col(get_col_for_metric(metric, get_seat_direction("W", seat))))
                .otherwise(pl.lit(0))
                .alias(f"__S{seat}_{metric}")
            )
            norm_exprs.append(expr)
    
    feat_df = chunk_df.select(norm_exprs)
    # This gives us (N, 24) array aligned with range_matrix columns
    deal_features = feat_df.to_numpy().astype(np.int8)
    
    # 3. Blocked Range Filtering
    # Instead of (N_deals x N_groups) full boolean matrix, we do blocks to fit in L2/L3 cache.
    # A block of 256 deals x 1024 groups = 256KB mask, fits easily.
    
    BLOCK_DEALS = 512
    BLOCK_GROUPS = 4096
    
    # Pre-fetch bit arrays to avoid dict lookups in hot loop
    chunk_bitmaps = bitmap_memmap[chunk_start : chunk_start + num_deals]
    results = [[] for _ in range(num_deals)]
    dealers = chunk_df.get_column("Dealer").to_list()
    # Cache dir indices: dealer_dirs[dealer_str] -> [0,1,2,3] for N,E,S,W indices
    dealer_dirs_map = {d: [DIR_TO_IDX[get_seat_direction(d, s)] for s in range(1, 5)] for d in DIRECTIONS}
    dealers_indices = [dealer_dirs_map[d] for d in dealers]

    # Iterate Deal Blocks
    for d_start in range(0, num_deals, BLOCK_DEALS):
        d_end = min(d_start + BLOCK_DEALS, num_deals)
        d_slice = deal_features[d_start:d_end, :] # (BD, 24)
        
        # Iterate Group Blocks
        for g_start in range(0, num_groups, BLOCK_GROUPS):
            g_end = min(g_start + BLOCK_GROUPS, num_groups)
            
            # Broadcast Comparison: (BD, 1, 24) vs (1, BG, 24)
            # This creates a (BD, BG, 24) bool array. We can reduce immediately.
            # Memory optimization: compare lo, compare hi, AND them.
            
            g_slice = range_matrix[g_start:g_end] # (BG, 24, 2)
            g_lo = g_slice[:, :, 0] # (BG, 24)
            g_hi = g_slice[:, :, 1] # (BG, 24)
            
            # (BD, 1, 24) >= (1, BG, 24)
            # This uses ~ (BD*BG*24) bytes. 512*4096*24 = 50MB. Safe.
            # pass_lo = (d_slice[:, None, :] >= g_lo[None, :, :])
            # pass_hi = (d_slice[:, None, :] <= g_hi[None, :, :])
            # mask = np.all(pass_lo & pass_hi, axis=2) # (BD, BG)
            
            # Optimized numpy:
            # Check LO
            mask = np.all(d_slice[:, None, :] >= g_lo[None, :, :], axis=2)
            # If any fail LO, we don't need to check HI (short circuit not possible in pure numpy, but AND is fast)
            if np.any(mask):
                mask &= np.all(d_slice[:, None, :] <= g_hi[None, :, :], axis=2)
            
            # Find hits
            # rows (deals) and cols (groups) where mask is True
            hit_rows, hit_cols = np.where(mask)
            
            if len(hit_rows) == 0:
                continue
                
            # 4. Verify Candidates
            # For each hit, check the bitmaps
            # hit_rows are local to d_start
            # hit_cols are local to g_start
            
            for r, c in zip(hit_rows, hit_cols):
                deal_local_idx = d_start + r
                group_global_idx = g_start + c
                
                # Get candidates for this group
                candidates = group_to_bt_indices.get(group_global_idx, [])
                if not candidates: continue
                
                deal_dir_indices = dealers_indices[deal_local_idx]
                
                for bt_idx in candidates:
                    internal_bt_idx = bt_idx_map.get(bt_idx)
                    if internal_bt_idx is None: continue
                    
                    # Bitmap Check
                    all_pass = True
                    for seat_idx in range(4): # 0=S1
                        dir_idx = deal_dir_indices[seat_idx]
                        offset_idx = (internal_bt_idx * 16) + (seat_idx * 4) + dir_idx
                        c_start = crit_offsets[offset_idx]
                        c_end = crit_offsets[offset_idx + 1]
                        
                        # Empty criteria pass automatically
                        if c_start == c_end: continue
                        
                        # Check bits
                        for k in range(c_start, c_end):
                            col_idx = crit_indices[k]
                            if not chunk_bitmaps[deal_local_idx, col_idx]:
                                all_pass = False; break
                        if not all_pass: break
                    
                    if all_pass:
                        results[deal_local_idx].append(bt_idx)

    # Sort results
    for r in results:
        if r:
            r.sort(key=lambda idx: bt_idx_to_auction.get(idx, ""), reverse=True)

    # Persist
    out_df = pl.DataFrame(
        {
            "_idx": np.arange(chunk_start, chunk_start + num_deals, dtype=np.uint32),
            "Matched_BT_Indices": pl.Series(results, dtype=pl.List(pl.UInt32)),
        }
    )
    out_df.write_parquet(out_parquet)
    return out_parquet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_elapsed(seconds: float) -> str:
    if seconds < 60: return f"{seconds:.1f}s"
    elif seconds < 3600: return f"{seconds / 60:.1f}m ({seconds:.0f}s)"
    else: return f"{seconds / 3600:.1f}h ({seconds / 60:.0f}m)"

def _format_hhmmss(seconds: float) -> str:
    if seconds < 0: seconds = 0
    s = int(seconds)
    return f"{s//3600:d}:{(s%3600)//60:02d}:{s%60:02d}"

def _step_done(t_step_start: float) -> None:
    elapsed = time.time() - t_step_start
    print(f"  Completed in {_format_elapsed(elapsed)} ({elapsed:.1f}s)")

def get_seat_direction(dealer: str, seat: int) -> str:
    dealer_idx = DIRECTIONS.index(dealer)
    return DIRECTIONS[(dealer_idx + seat - 1) % 4]

def get_col_for_metric(metric: str, direction: str) -> str:
    if metric.startswith("SL_"): return f"SL_{direction}_{metric[3:]}"
    return f"{metric}_{direction}"

def parse_criteria_to_ranges(criteria_list: List[str] | None) -> Dict[str, Tuple[int, int]]:
    if not criteria_list: return {m: DEFAULT_RANGES[m] for m in RANGE_METRICS}
    mins = {m: [] for m in RANGE_METRICS}; maxs = {m: [] for m in RANGE_METRICS}
    for expr in criteria_list:
        expr = expr.strip()
        if m := PATTERN_GE.match(expr):
            metric, val = m.groups()
            if metric in mins: mins[metric].append(int(val))
        elif m := PATTERN_LE.match(expr):
            metric, val = m.groups()
            if metric in maxs: maxs[metric].append(int(val))
        elif m := PATTERN_EQ.match(expr):
            metric, val = m.groups()
            if metric in mins: mins[metric].append(int(val)); maxs[metric].append(int(val))
        elif m := PATTERN_GT.match(expr):
            metric, val = m.groups()
            if metric in mins: mins[metric].append(int(val) + 1)
        elif m := PATTERN_LT.match(expr):
            metric, val = m.groups()
            if metric in maxs: maxs[metric].append(int(val) - 1)
    return {m: (max(mins[m]) if mins[m] else DEFAULT_RANGES[m][0],
                min(maxs[m]) if maxs[m] else DEFAULT_RANGES[m][1])
            for m in RANGE_METRICS}

def build_bt_ranges_cube(bt_completed: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[int, List[int]]]:
    bt_indices = bt_completed.get_column("bt_index").to_list()
    num_rows = len(bt_indices)
    raw_data = {"bt_index": [int(idx) for idx in bt_indices if idx is not None]}
    
    # Initialize defaults
    for seat in range(1, 5):
        for metric in RANGE_METRICS:
            lo_def, hi_def = DEFAULT_RANGES[metric]
            raw_data[f"S{seat}_{metric}_Lo"] = [int(lo_def)] * num_rows
            raw_data[f"S{seat}_{metric}_Hi"] = [int(hi_def)] * num_rows
    
    seat_data = {s: bt_completed.get_column(f"Agg_Expr_Seat_{s}").to_list() for s in range(1, 5)}
    for i in range(num_rows):
        for s_idx in range(1, 5):
            if criteria := seat_data[s_idx][i]:
                ranges = parse_criteria_to_ranges(criteria)
                for m, (lo, hi) in ranges.items():
                    raw_data[f"S{s_idx}_{m}_Lo"][i] = int(lo)
                    raw_data[f"S{s_idx}_{m}_Hi"][i] = int(hi)
    
    full_ranges_df = pl.DataFrame({k: pl.Series(k, v, dtype=pl.UInt32 if k == "bt_index" else pl.Int8) for k, v in raw_data.items()})
    range_cols = [c for c in full_ranges_df.columns if c != "bt_index"]
    cube_df = full_ranges_df.group_by(range_cols).agg(pl.col("bt_index").alias("bt_indices")).with_row_index("group_id")
    return cube_df.select(["group_id"] + range_cols), {
        int(gid): [int(idx) for idx in idxs]
        for gid, idxs in zip(cube_df.get_column("group_id"), cube_df.get_column("bt_indices"))
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deal-rows", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=10000) # Increased due to better memory mgmt
    parser.add_argument("--deal-file", type=str, default="E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet")
    parser.add_argument("--bt-file", type=str, default="E:/bridge/data/bbo/bidding/bbo_bt_seat1.parquet")
    parser.add_argument("--bitmap-file", type=str, default="E:/bridge/data/bbo/data/bbo_mldf_augmented_criteria_bitmaps.parquet")
    parser.add_argument("--exec-plan-file", type=str, default="E:/bridge/data/bbo/bidding/bbo_bt_execution_plan_data.pkl")
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--progress-file", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from previous run if chunks exist (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Force fresh start, delete existing chunks")
    args = parser.parse_args()
    
    dt_start = datetime.now()
    t_pipeline_start = time.time()
    deal_file = pathlib.Path(args.deal_file)
    bt_file = pathlib.Path(args.bt_file)
    bitmap_file = pathlib.Path(args.bitmap_file)
    exec_plan_file = pathlib.Path(args.exec_plan_file)
    output_file = pathlib.Path(args.output) if args.output else deal_file.parent / "bbo_mldf_augmented_matches.parquet"

    print("=" * 70)
    print(f"BT Deal Matches Pipeline (OPT) | Start: {dt_start.isoformat(sep=' ', timespec='seconds')}")
    print("=" * 70)
    
    # 1. Load BT
    print("\n[1/6] Loading BT...")
    t_step = time.time()
    bt_needed_cols = ["bt_index", "Auction", "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4"]
    bt_schema = pl.read_parquet_schema(bt_file)
    bt_select_cols = [c for c in bt_needed_cols if c in bt_schema]
    if "is_completed_auction" in bt_schema:
        bt_select_cols = bt_select_cols + ["is_completed_auction"]
    bt_q = pl.scan_parquet(bt_file).select(bt_select_cols)
    if "is_completed_auction" in bt_schema:
        bt_q = bt_q.filter(pl.col("is_completed_auction"))
    bt_completed = bt_q.collect()
    bt_idx_list = bt_completed.get_column("bt_index").to_list()
    bt_auction_list = bt_completed.get_column("Auction").to_list()
    bt_idx_to_auction = {int(i): str(a) for i, a in zip(bt_idx_list, bt_auction_list)}
    _step_done(t_step)
    
    # 2. Build Cube
    print("\n[2/6] Building range cube and dense binary criteria...")
    t_step = time.time()
    unique_ranges_df, group_to_bt_indices = build_bt_ranges_cube(bt_completed)
    
    # Build Dense Range Matrix (NumGroups, 24, 2)
    # 24 features: S1_HCP, S1_SL_S ... S4_TP
    # 2 values: Lo, Hi
    num_groups = unique_ranges_df.height
    print(f"  Cube size: {num_groups:,} groups")
    
    range_matrix_file = pathlib.Path(tempfile.gettempdir()) / f"ranges_{os.getpid()}.bin"
    range_matrix_shape = (num_groups, 24, 2)
    range_mm = np.memmap(range_matrix_file, dtype=np.int8, mode="w+", shape=range_matrix_shape)
    
    # Fill range matrix
    feat_idx = 0
    for seat in range(1, 5):
        for metric in RANGE_METRICS:
            # Fetch Lo/Hi columns, cast to int8, assign to memmap
            # Note: int8 handles -128..127. Bridge metrics (0-40) fit easily.
            lo_vals = unique_ranges_df.get_column(f"S{seat}_{metric}_Lo").to_numpy().astype(np.int8)
            hi_vals = unique_ranges_df.get_column(f"S{seat}_{metric}_Hi").to_numpy().astype(np.int8)
            range_mm[:, feat_idx, 0] = lo_vals
            range_mm[:, feat_idx, 1] = hi_vals
            feat_idx += 1
    range_mm.flush()
    del range_mm # Close write handle
    
    # Load Lib
    import importlib.util
    lib_path = pathlib.Path(__file__).parent / "mlBridge"
    sys.path.insert(0, str(lib_path))
    spec = importlib.util.spec_from_file_location("mlBridgeBiddingLib", lib_path / "mlBridgeBiddingLib.py")
    if spec is None or spec.loader is None: sys.exit("ERROR: No lib")
    mlBridgeBiddingLib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mlBridgeBiddingLib)
    
    (dir_cols, expr_map, valid_deal_columns, py_exprs) = mlBridgeBiddingLib.load_execution_plan_data(exec_plan_file)
    
    total_deal_rows = pl.scan_parquet(deal_file).select(pl.len()).collect().item()
    target_rows = args.deal_rows if args.deal_rows else total_deal_rows
    
    if bitmap_file.exists():
        if pl.scan_parquet(bitmap_file).select(pl.len()).collect().item() < target_rows:
            bitmap_file.unlink()

    if not bitmap_file.exists():
        cols_for_builder = sorted(set(valid_deal_columns).union({"Dealer"}))
        builder_df = pl.read_parquet(deal_file, columns=cols_for_builder, n_rows=args.deal_rows)
        mlBridgeBiddingLib.build_or_load_directional_criteria_bitmaps(builder_df, py_exprs, expr_map, deal_file=deal_file, exec_plan_file=exec_plan_file)

    bitmap_schema = pl.read_parquet_schema(bitmap_file)
    col_to_idx = {name: i for i, name in enumerate(bitmap_schema.keys())}
    
    # Flatten Criteria Indices
    print("  Flattening criteria indices...")
    num_auctions = bt_completed.height
    bt_idx_map = {}
    flat_indices = []
    offsets = [0]
    
    bt_indices_list = bt_completed.get_column("bt_index").to_list()
    seat_criteria_data = {
        s: bt_completed.get_column(f"Agg_Expr_Seat_{s}").to_list()
        for s in range(1, 5)
    }
    
    total_criteria_found = 0
    for i in range(num_auctions):
        bt_idx = int(bt_indices_list[i])
        bt_idx_map[bt_idx] = i
        for seat_idx in range(4): # S1..S4
            exprs = seat_criteria_data[seat_idx+1][i]
            for d in DIRECTIONS:
                if exprs:
                    indices: List[int] = []
                    em = expr_map.get(d, {})
                    for crit in exprs:
                        mapped = em.get(crit)
                        if mapped is None: continue
                        col_name = f"DIR_{d}_{mapped}"
                        idx = col_to_idx.get(col_name)
                        if idx is None: continue
                        indices.append(idx)
                    flat_indices.extend(indices)
                    total_criteria_found += len(indices)
                offsets.append(len(flat_indices))
                
    if total_criteria_found == 0:
        raise RuntimeError("CRITICAL: 0 criteria indices matched bitmap columns.")
    
    criteria_file = pathlib.Path(tempfile.gettempdir()) / f"crit_{os.getpid()}.bin"
    offset_file = pathlib.Path(tempfile.gettempdir()) / f"offs_{os.getpid()}.bin"
    np.array(flat_indices, dtype=np.uint16).tofile(str(criteria_file))
    np.array(offsets, dtype=np.uint32).tofile(str(offset_file))
    _step_done(t_step)
    
    # 3. Shared Bitmaps
    print("\n[3/6] Preparing bitmap memmap (bool)...")
    t_step = time.time()
    num_deals = args.deal_rows if args.deal_rows else pl.scan_parquet(bitmap_file).select(pl.len()).collect().item()
    num_cols = len(bitmap_schema)
    bitmap_bin = pathlib.Path(tempfile.gettempdir()) / f"bitmaps_{os.getpid()}.bin"
    mm = np.memmap(bitmap_bin, dtype=np.bool_, mode="w+", shape=(num_deals, num_cols))
    build_chunk = 20000
    for start in range(0, num_deals, build_chunk):
        end = min(start + build_chunk, num_deals)
        df_chunk = pl.scan_parquet(bitmap_file).slice(start, end - start).collect()
        mm[start:end, :] = df_chunk.to_numpy().astype(bool)
    mm.flush()
    del mm
    _step_done(t_step)

    # 4. Load Deal Data
    print("\n[4/6] Deal data will be read by workers...")
    t_step = time.time()
    matching_cols = ["Dealer"] + [get_col_for_metric(m, d) for d in DIRECTIONS for m in RANGE_METRICS]
    _step_done(t_step)
    
    # 5. Parallel Processing with Restart Support
    tmp_chunk_dir = deal_file.parent / "tmp_chunks"
    state_file = tmp_chunk_dir / "_run_state.json"
    
    # Calculate total chunks and build chunk info list
    total_chunks = (num_deals + args.chunk_size - 1) // args.chunk_size
    all_chunks = [
        {
            "idx": i,
            "start": s,
            "length": min(args.chunk_size, num_deals - s),
            "file": str(tmp_chunk_dir / f"chunk_{i:05d}.parquet")
        }
        for i, s in enumerate(range(0, num_deals, args.chunk_size))
    ]
    
    # Handle resume logic
    skipped_rows = 0
    completed_chunk_indices = set()
    
    if tmp_chunk_dir.exists():
        if args.resume and state_file.exists():
            # Validate state file matches current run parameters
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                
                params_match = (
                    saved_state.get("num_deals") == num_deals and
                    saved_state.get("chunk_size") == args.chunk_size and
                    saved_state.get("deal_file") == str(deal_file) and
                    saved_state.get("total_chunks") == total_chunks
                )
                
                if params_match:
                    # Find completed chunks by checking which files exist and are valid
                    for chunk_info in all_chunks:
                        chunk_file = pathlib.Path(chunk_info["file"])
                        if chunk_file.exists():
                            try:
                                # Validate file is readable and has expected row count
                                chunk_df = pl.scan_parquet(chunk_file).select(pl.len()).collect()
                                if chunk_df.item() == chunk_info["length"]:
                                    completed_chunk_indices.add(chunk_info["idx"])
                                    skipped_rows += chunk_info["length"]
                            except Exception:
                                # Corrupt file, will be reprocessed
                                chunk_file.unlink()
                    
                    if completed_chunk_indices:
                        print(f"\n  RESUME: Found {len(completed_chunk_indices):,} completed chunks ({skipped_rows:,} rows).", flush=True)
                else:
                    print(f"\n  RESUME: State mismatch detected, starting fresh.", flush=True)
                    shutil.rmtree(tmp_chunk_dir)
            except Exception as e:
                print(f"\n  RESUME: Could not read state file ({e}), starting fresh.", flush=True)
                shutil.rmtree(tmp_chunk_dir)
        elif not args.resume:
            print(f"\n  --no-resume specified, starting fresh.", flush=True)
            shutil.rmtree(tmp_chunk_dir)
    
    tmp_chunk_dir.mkdir(exist_ok=True)
    
    # Save/update state file
    with open(state_file, 'w') as f:
        json.dump({
            "num_deals": num_deals,
            "chunk_size": args.chunk_size,
            "deal_file": str(deal_file),
            "total_chunks": total_chunks,
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)
    
    # Build list of pending chunks
    pending_chunks = [c for c in all_chunks if c["idx"] not in completed_chunk_indices]
    pending_rows = sum(c["length"] for c in pending_chunks)
    
    num_workers = args.parallel or max(1, (os.cpu_count() or 4) - 4)
    print(f"\n[5/6] Processing {num_deals:,} deals with {num_workers} workers...")
    if skipped_rows > 0:
        print(f"  Skipping {skipped_rows:,} already-completed rows ({len(completed_chunk_indices)} chunks).", flush=True)
        print(f"  Remaining: {pending_rows:,} rows ({len(pending_chunks)} chunks).", flush=True)
    t_step = time.time()
    
    # Handle 100% resumed case - all chunks already complete
    new_rows = 0
    done_rows = skipped_rows
    res_file = None
    
    if not pending_chunks:
        print(f"  All chunks already complete! Skipping to merge.", flush=True)
    else:
        print(f"  Progress will print every ~{args.progress_every:,} rows.", flush=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            res_file = tmp.name
            pickle.dump({
                'unique_ranges_df': unique_ranges_df, 'group_to_bt_indices': group_to_bt_indices,
                'bt_idx_to_auction': bt_idx_to_auction,
                'bt_idx_map': bt_idx_map,
                'deal_file': str(deal_file),
                'matching_cols': matching_cols,
            }, tmp)

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, initializer=init_worker, 
                initargs=(res_file, str(bitmap_bin), (num_deals, num_cols),
                          str(criteria_file), len(flat_indices), str(offset_file), len(offsets),
                          str(range_matrix_file), range_matrix_shape)
            ) as executor:
                # Only submit pending chunks
                task_iter = iter([
                    (c["idx"], c["start"], c["length"], c["file"])
                    for c in pending_chunks
                ])

                in_flight: Dict[concurrent.futures.Future, int] = {}
                done_rows = skipped_rows  # Start from skipped rows for overall progress
                new_rows = 0  # Track newly processed rows for rate calculation
                last_report_rows = done_rows - max(1, args.progress_every)
                t_step5_progress_start = time.time()
                last_heartbeat = time.time()

                # Prime
                backlog = max(1, num_workers * 2)
                for _ in range(backlog):
                    try:
                        args4 = next(task_iter)
                    except StopIteration:
                        break
                    fut = executor.submit(process_one_chunk, args4)
                    in_flight[fut] = int(args4[2])

                while in_flight:
                    timeout = max(1, int(args.heartbeat_seconds))
                    done_set, _ = concurrent.futures.wait(
                        in_flight.keys(), timeout=timeout, return_when=FIRST_COMPLETED
                    )

                    now = time.time()
                    if not done_set:
                        elapsed = now - t_step5_progress_start
                        if (now - last_heartbeat) >= timeout:
                            print(f"  Heartbeat: {done_rows:,}/{num_deals:,} ({(done_rows/num_deals*100):.2f}%) | elapsed {_format_hhmmss(elapsed)}", flush=True)
                            last_heartbeat = now
                        continue

                    for fut in done_set:
                        chunk_len = in_flight.pop(fut)
                        _outp = fut.result()
                        done_rows += chunk_len
                        new_rows += chunk_len

                        try:
                            args4 = next(task_iter)
                            nfut = executor.submit(process_one_chunk, args4)
                            in_flight[nfut] = int(args4[2])
                        except StopIteration:
                            pass

                        should_report = (done_rows - last_report_rows) >= max(1, args.progress_every) or done_rows >= num_deals
                        if should_report:
                            elapsed = time.time() - t_step5_progress_start
                            rate = (new_rows / elapsed) if elapsed > 0 else 0.0
                            remaining = num_deals - done_rows
                            eta_s = (remaining / rate) if rate > 0 else float("inf")
                            pct = (done_rows / num_deals) * 100
                            eta_txt = _format_hhmmss(eta_s) if eta_s != float("inf") else "?:??"
                            print(f"  Progress: {done_rows:,}/{num_deals:,} ({pct:.2f}%) | ETA {eta_txt} | {rate:,.0f} rows/s", flush=True)
                            last_report_rows = done_rows
                            last_heartbeat = time.time()
                            if args.progress_file:
                                with open(args.progress_file.strip(' "\'\\/'), 'w') as f:
                                    json.dump({
                                        "progress": pct, 
                                        "processed": done_rows, 
                                        "total": num_deals,
                                        "skipped": skipped_rows,
                                        "new_this_run": new_rows,
                                        "eta_seconds": None if eta_s == float('inf') else eta_s, 
                                        "timestamp": datetime.now().isoformat()
                                    }, f)
        finally:
            # Clean up temp resource file used by workers
            if res_file: 
                pathlib.Path(res_file).unlink(missing_ok=True)
    
    # Clean up temp files (created in steps 2-3, need cleanup regardless of processing)
    for f in [bitmap_bin, criteria_file, offset_file, range_matrix_file]:
        if f: 
            pathlib.Path(f).unlink(missing_ok=True)

    _step_done(t_step)
    if skipped_rows > 0:
        print(f"  (Resumed run: {skipped_rows:,} rows from cache, {new_rows:,} newly processed)", flush=True)

    # 6. Final Merge
    print("\n[6/6] Finalizing merge (Streaming Sink)...")
    t_step = time.time()
    
    # Verify all chunks exist before merging
    missing_chunks = [c for c in all_chunks if not pathlib.Path(c["file"]).exists()]
    if missing_chunks:
        print(f"  ERROR: {len(missing_chunks)} chunks missing! Run again to complete.", flush=True)
        for mc in missing_chunks[:5]:
            print(f"    - chunk_{mc['idx']:05d} (rows {mc['start']:,}-{mc['start']+mc['length']:,})", flush=True)
        if len(missing_chunks) > 5:
            print(f"    ... and {len(missing_chunks)-5} more", flush=True)
        sys.exit(1)
    
    q_results = pl.scan_parquet(tmp_chunk_dir / "*.parquet")
    q_original = pl.scan_parquet(deal_file).slice(0, num_deals).with_row_index("_idx")
    
    q_original.join(q_results, on="_idx", how="left") \
              .with_columns([pl.col("Matched_BT_Indices").list.len().cast(pl.UInt16).alias("Matched_BT_Count")]) \
              .drop("_idx") \
              .sink_parquet(output_file)
    
    # Clean up only after successful merge
    shutil.rmtree(tmp_chunk_dir)
    _step_done(t_step)
    
    dt_end = datetime.now()
    print("=" * 70)
    print(f"BT Deal Matches Pipeline (OPT) | End:   {dt_end.isoformat(sep=' ', timespec='seconds')}")
    print(f"Total elapsed: {_format_elapsed(time.time() - t_pipeline_start)}")
    print(f"Output: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()

