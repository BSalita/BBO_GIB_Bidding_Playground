#!/usr/bin/env python3
"""
SUPERHUMAN GPU Bitmap Filter for Deal-to-BT Matches

Performance: ~25 minutes for 16M deals on RTX 5080.
VRAM Target: ~11-12GB Peak.

Strategy:
1.  Deduplicate BT criteria into "Logical Templates" (completed auctions only).
2.  Build a Dense Template Matrix (N_Cols x Unique_Templates) on GPU (Pinned).
3.  For each large RAM batch (50k deals), perform GPU MatMul in sub-chunks (500 deals).
4.  Collect passed (Deal, Template) coordinates on GPU and move to CPU (Sparse).
5.  Vectorized filtering in Polars (explode/join/regroup) on the whole batch.

Criteria model / invariant:
- BT `Agg_Expr_Seat_1..4` are treated as **AND-only** lists of atomic predicates.
- Verification only enforces predicates that can be mapped to bitmap columns via `criteria_to_bitmap_column(...)`.
  If new criterion forms are added without corresponding bitmap columns, the verifier becomes permissive for those predicates
  unless the pipeline is updated.
"""

import argparse
import gc
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import torch  # type: ignore[import-not-found]

# --- Constants ---
DIRECTIONS = ["N", "E", "S", "W"]
DIR_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}

DEFAULT_GPU_OUTPUT = Path("E:/bridge/data/bbo/bidding/bbo_deal_to_bt_gpu.parquet")
DEFAULT_BITMAP_FILE = Path("E:/bridge/data/bbo/data/bbo_mldf_augmented_criteria_bitmaps.parquet")
DEFAULT_BT_FILE = Path("E:/bridge/data/bbo/bidding/bbo_bt_seat1.parquet")
DEFAULT_DEALS_FILE = Path("E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet")
DEFAULT_OUTPUT_FILE = Path("E:/bridge/data/bbo/bidding/bbo_deal_to_bt_verified.parquet")

DEAL_CHUNK_SIZE = 500  # Sub-chunks for GPU MatMul (Keep < 1GB transient)
RAM_BATCH_SIZE = 50000 # Large batches for efficient Parquet I/O and Polars joins

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

try:
    from tqdm import tqdm # type: ignore[import-not-found]
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable=None, **kwargs): return iterable

def fmt_time(seconds: float) -> str:
    if seconds < 60: return f"{seconds:.1f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"

def get_seat_direction(dealer: str, seat: int) -> str:
    dealer_idx = DIRECTIONS.index(dealer)
    return DIRECTIONS[(dealer_idx + seat - 1) % 4]

def criteria_to_bitmap_column(criteria: str, direction: str) -> Optional[str]:
    criteria = criteria.strip()
    match = re.match(r'^(SL)_([SHDC])\s*(>=|<=|==|>|<)\s*(\d+)$', criteria)
    if match:
        prefix, suit, op, value = match.groups()
        return f"DIR_{direction}_{prefix}_{direction}_{suit} {op} {value}"
    match = re.match(r'^(\w+)\s*(>=|<=|==|>|<)\s*(\d+)$', criteria)
    if match:
        metric, op, value = match.groups()
        return f"DIR_{direction}_{metric}_{direction} {op} {value}"
    match = re.match(r'^(\w+)_([SHDC])$', criteria)
    if match:
        prefix, suit = match.groups()
        return f"DIR_{direction}_{prefix}_{direction}_{suit}"
    if re.match(r'^[A-Z]\w+$', criteria):
        return f"DIR_{direction}_{criteria}_{direction}"
    return None

def run_pipeline(
    gpu_output: Path,
    bitmap_file: Path,
    bt_file: Path,
    deals_file: Path,
    output_file: Path,
    max_deals: Optional[int] = None,
) -> None:
    start_time = time.time()
    log("=" * 70)
    log("SUPERHUMAN GPU Bitmap Filter (12GB VRAM Profile)")
    log("=" * 70)

    device = torch.device("cuda:0")
    log(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Metadata
    log("[1/5] Loading metadata...")
    bitmap_schema = pl.scan_parquet(bitmap_file).collect_schema()
    bitmap_columns = list(bitmap_schema.names())
    col_to_idx = {col: i for i, col in enumerate(bitmap_columns)}
    n_total_deals = pl.scan_parquet(gpu_output).select(pl.len()).collect().item()
    if max_deals: n_total_deals = min(n_total_deals, max_deals)

    # 2. Deduplicate BT Logic into Templates
    log("[2/5] Deduplicating BT logic templates (all completed auctions)...")
    t0 = time.time()
    
    bt_logic = (
        pl.scan_parquet(bt_file)
        .filter(pl.col("is_completed_auction"))
        .select(["bt_index", "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4"])
        .collect()
    )
    
    # Create logic keys
    bt_logic = bt_logic.with_columns([
        pl.col(f"Agg_Expr_Seat_{i}").list.join("|").alias(f"logic_{i}")
        for i in range(1, 5)
    ])
    bt_logic = bt_logic.with_columns(
        pl.concat_str([f"logic_{i}" for i in range(1, 5)], separator="||").alias("full_logic")
    )
    
    unique_templates = bt_logic.select("full_logic").unique().with_row_index("template_id")
    bt_to_template = bt_logic.join(unique_templates, on="full_logic").select(["bt_index", "template_id"])
    
    template_defs = bt_logic.join(unique_templates, on="full_logic").unique(subset="template_id").sort("template_id")
    n_templates = len(template_defs)
    log(f"  Found {n_templates:,} unique logic templates for {len(bt_logic):,} BT rows in {time.time()-t0:.1f}s")

    # 3. Build Template-to-Column Matrices
    log("[3/5] Building GPU Template Matrices...")
    t0 = time.time()
    template_matrices_cpu = {} 
    template_sums_cpu = {}

    for dealer in DIRECTIONS:
        rows, cols = [], []
        sums = []
        for row in template_defs.iter_rows(named=True):
            tid = row["template_id"]
            current_cols = []
            for seat in range(1, 5):
                direction = get_seat_direction(dealer, seat)
                criteria_list = row[f"Agg_Expr_Seat_{seat}"]
                if criteria_list:
                    for crit in criteria_list:
                        col_name = criteria_to_bitmap_column(crit, direction)
                        if col_name and col_name in col_to_idx:
                            current_cols.append(col_to_idx[col_name])
            
            rows.extend([tid] * len(current_cols))
            cols.extend(current_cols)
            sums.append(len(current_cols))
        
        matrix = torch.zeros((n_templates, len(bitmap_columns)), device="cpu", dtype=torch.float16)
        matrix[rows, cols] = 1.0
        template_matrices_cpu[dealer] = matrix.T.contiguous()
        template_sums_cpu[dealer] = torch.tensor(sums, device="cpu", dtype=torch.float16)

    log(f"  CPU Template Matrices ready in {time.time()-t0:.1f}s")

    # 4. Verification Loop
    log("[4/5] Running Matrix Verification...")
    
    # Move Template Matrices to GPU (9.72 GB total)
    log("  Pinning template matrices to GPU (9.7 GB allocated)...")
    template_matrices_gpu = {d: m.to(device) for d, m in template_matrices_cpu.items()}
    template_sums_gpu = {d: s.to(device) for d, s in template_sums_cpu.items()}
    
    del template_matrices_cpu, template_sums_cpu, bt_logic, template_defs
    gc.collect()

    all_results = []
    total_before, total_after = 0, 0
    
    n_ram_batches = (n_total_deals + RAM_BATCH_SIZE - 1) // RAM_BATCH_SIZE
    pbar = tqdm(total=n_total_deals, desc="Verifying deals", unit="deals") if HAS_TQDM else None

    gpu_lf = pl.scan_parquet(gpu_output)
    deals_lf = pl.scan_parquet(deals_file)
    bitmap_lf = pl.scan_parquet(bitmap_file)

    for batch_idx in range(n_ram_batches):
        b_start = batch_idx * RAM_BATCH_SIZE
        b_size = min(RAM_BATCH_SIZE, n_total_deals - b_start)
        
        gpu_batch = gpu_lf.slice(b_start, b_size).collect()
        deals_batch = deals_lf.slice(b_start, b_size).select("Dealer").collect()
        bitmap_batch = bitmap_lf.slice(b_start, b_size).collect().to_numpy()
        
        n_sub_chunks = (b_size + DEAL_CHUNK_SIZE - 1) // DEAL_CHUNK_SIZE
        
        # Collect passed coordinates (local_deal_idx, template_id)
        passed_deal_indices = []
        passed_template_ids = []
        
        for sc_idx in range(n_sub_chunks):
            sc_start = sc_idx * DEAL_CHUNK_SIZE
            sc_end = min(sc_start + DEAL_CHUNK_SIZE, b_size)
            sc_size = sc_end - sc_start
            
            d_bitmap_gpu = torch.tensor(bitmap_batch[sc_start:sc_end], device=device, dtype=torch.float16)
            sc_dealers = deals_batch.slice(sc_start, sc_size)["Dealer"].to_list()
            
            for dealer in DIRECTIONS:
                d_indices = [i for i, d in enumerate(sc_dealers) if d == dealer]
                if not d_indices: continue
                
                d_idx_tensor = torch.tensor(d_indices, device=device)
                d_sub = d_bitmap_gpu[d_idx_tensor]
                
                # MatMul Intersection
                res = torch.matmul(d_sub, template_matrices_gpu[dealer])
                
                # Comparison
                chunk_matches = (res == template_sums_gpu[dealer])
                
                # Sparser coordinate extraction (Prevents 38GB VRAM allocation)
                coords = torch.nonzero(chunk_matches)
                if coords.shape[0] > 0:
                    # Offset local row indices back to the batch context
                    batch_row_indices = d_idx_tensor[coords[:, 0]] + sc_start
                    passed_deal_indices.append(batch_row_indices.cpu().numpy())
                    passed_template_ids.append(coords[:, 1].cpu().numpy())
                
                del res, d_sub, chunk_matches
            
            del d_bitmap_gpu
        
        # --- Vectorized Filtering in Polars ---
        if passed_deal_indices:
            passed_df = pl.DataFrame({
                "local_idx": np.concatenate(passed_deal_indices).astype(np.uint32),
                "template_id": np.concatenate(passed_template_ids).astype(np.uint32)
            }).unique()
        else:
            passed_df = pl.DataFrame({"local_idx": [], "template_id": []}, 
                                     schema={"local_idx": pl.UInt32, "template_id": pl.UInt32})

        gpu_batch = gpu_batch.with_row_index("local_idx")
        total_before += gpu_batch["Matched_BT_Indices"].list.len().sum()
        
        exploded = gpu_batch.explode("Matched_BT_Indices")
        joined = exploded.join(bt_to_template, left_on="Matched_BT_Indices", right_on="bt_index", how="left")
        
        verified_exploded = joined.join(passed_df, on=["local_idx", "template_id"], how="inner")
        
        chunk_verified = (
            verified_exploded.group_by("deal_idx")
            .agg(pl.col("Matched_BT_Indices"))
        )
        
        batch_results = (
            gpu_batch.select("deal_idx")
            .join(chunk_verified, on="deal_idx", how="left")
            .with_columns(pl.col("Matched_BT_Indices").fill_null([]))
        )
        
        total_after += batch_results["Matched_BT_Indices"].list.len().sum()
        all_results.append(batch_results)

        if pbar: pbar.update(b_size)
        
        del passed_df, bitmap_batch, passed_deal_indices, passed_template_ids
        # Transient cache clearing
        torch.cuda.empty_cache()

    if pbar: pbar.close()

    # 5. Save Output
    log("[5/5] Saving results...")
    final_df = pl.concat(all_results)
    final_df.write_parquet(output_file)
    
    reduction = (1 - total_after / total_before) * 100 if total_before > 0 else 0
    log("=" * 70)
    log(f"PIPELINE COMPLETE in {fmt_time(time.time() - start_time)}")
    log(f"Matches before: {total_before:,}, After: {total_after:,} ({reduction:.1f}% reduction)")
    log(f"Output: {output_file}")
    log("=" * 70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-deals", type=int, default=None)
    args = parser.parse_args()
    
    run_pipeline(
        gpu_output=DEFAULT_GPU_OUTPUT,
        bitmap_file=DEFAULT_BITMAP_FILE,
        bt_file=DEFAULT_BT_FILE,
        deals_file=DEFAULT_DEALS_FILE,
        output_file=DEFAULT_OUTPUT_FILE,
        max_deals=args.max_deals
    )

if __name__ == "__main__":
    main()
