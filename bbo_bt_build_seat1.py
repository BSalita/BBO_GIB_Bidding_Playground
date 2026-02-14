#!/usr/bin/env python3
"""
Build a seat-1-only bidding table parquet file (V3 - Fast Stats Integrated).

This script creates 'bbo_bt_seat1.parquet' by joining the augmented bidding table
with statistics from one of two sources:

1. bbo_bt_stats/ directory (from bbo_bt_compute_stats.py) — the original source
2. bt_ev_par_stats_gpu_v3.parquet (from bbo_bt_ev_gpu.py) — the consolidated source
   which also includes EV/Par stats and seat-relative DD trick means

Process:
1. Scans bt_augmented (541M rows).
2. Filters for Auctions that don't start with 'p-' (opening bids).
3. Joins with stats (from either source) directly on 'index'/'bt_index'.
4. Writes out the final seat1 table.

Usage:
    python bbo_bt_build_seat1.py                        # Use bbo_bt_stats/ (default)
    python bbo_bt_build_seat1.py --ev-gpu-stats path    # Use ev_gpu v3 output

Latest observed full run (2026-02-13, v3 input):
  python bbo_bt_build_seat1.py --ev-gpu-stats E:\\bridge\\data\\bbo\\bidding\\bt_ev_par_stats_gpu_v3.parquet
  Elapsed: 51.6m (3097s)
  Wide table size: 47,387.1 MB
  Stats table size: 220.7 MB
"""

import argparse
import pathlib
import time
from datetime import datetime
import polars as pl


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m ({seconds:.0f}s)"
    else:
        return f"{seconds / 3600:.1f}h ({seconds / 60:.0f}m)"


def _log_step(step_name: str, step_start: float, min_seconds: float = 60.0) -> None:
    """Log step time if it exceeds min_seconds."""
    elapsed = time.time() - step_start
    if elapsed >= min_seconds:
        print(f"  [step] {step_name}: {_format_elapsed(elapsed)}")

def build_seat1_table_fast(
    bt_augmented_path: pathlib.Path,
    stats_dir: pathlib.Path,
    output_path: pathlib.Path,
    n_rows: int | None = None,
    ev_gpu_stats_path: pathlib.Path | None = None,
) -> None:
    build_start = time.time()
    print("=" * 60)
    print("Building Seat-1-Only Bidding Table (Fast Stats Integration)")
    print("Latest full-run baseline (v3 input): 51.6m (3097s)")
    print("=" * 60)
    
    if not bt_augmented_path.exists():
        print(f"ERROR: Missing {bt_augmented_path}")
        return

    # 1. Scan augmented table
    step_start = time.time()
    print(f"Scanning bt_augmented: {bt_augmented_path}")
    lf = pl.scan_parquet(bt_augmented_path)
    if n_rows:
        lf = lf.head(n_rows)
        print(f"  Debug limit: {n_rows:,} rows")

    # Filter to openers only
    lf = lf.filter(~pl.col("Auction").str.starts_with("p-"))
    _log_step("scan/filter bt_augmented", step_start, min_seconds=0.0)
    
    # 2. Scan stats from chosen source
    step_start = time.time()
    if ev_gpu_stats_path is not None:
        # Use ev_gpu v3 output as stats source
        if not ev_gpu_stats_path.exists():
            print(f"ERROR: Missing ev_gpu stats file {ev_gpu_stats_path}")
            return
        print(f"Scanning stats from ev_gpu: {ev_gpu_stats_path}")
        stats_lf = pl.scan_parquet(ev_gpu_stats_path)
        # The ev_gpu output uses 'bt_index' which corresponds to 'index' in bt_augmented
        # Rename for join compatibility
        stats_lf = stats_lf.rename({"bt_index": "index"}).with_columns(
            pl.col("index").cast(pl.UInt64)
        )
        print("  (using bt_index → index rename for join)")
    else:
        # Use legacy stats directory from bbo_bt_compute_stats.py
        if not stats_dir.exists():
            print(f"ERROR: Missing stats directory {stats_dir}")
            return
        print(f"Scanning stats: {stats_dir}")
        stats_lf = pl.scan_parquet(stats_dir / "*.parquet")
    _log_step("scan stats source", step_start, min_seconds=0.0)
    
    # 3. Join on index
    # Both are keyed by global 'index' from augmented table
    step_start = time.time()
    lf = lf.join(stats_lf, on="index", how="left")
    _log_step("join augmented + stats", step_start, min_seconds=0.0)
    
    # 4. Final selection and renaming
    # Downstream expects 'bt_index' and 'matching_deal_count' (from Seat 1)
    # Perform final downcasting of core columns
    lf = lf.rename({
        "index": "bt_index",
        "matching_deal_count_S1": "matching_deal_count"
    }).with_columns([
        pl.col("bt_index").cast(pl.UInt32),
        pl.col("seat").cast(pl.UInt8),
    ])
    _log_step("rename/downcast core columns", step_start, min_seconds=0.0)
    
    # Ensure all stats columns that might have been joined are Float32/UInt8
    # (In case the stats folder had legacy Float64 files)
    step_start = time.time()
    all_cols = lf.collect_schema().names()
    cast_ops = []
    for c in all_cols:
        if any(x in c for x in ["_mean", "_std"]):
            cast_ops.append(pl.col(c).fill_nan(None).cast(pl.Float32))
        elif any(x in c for x in ["_min", "_max"]):
            cast_ops.append(pl.col(c).fill_nan(None).fill_null(0).cast(pl.UInt8))
    if cast_ops:
        lf = lf.with_columns(cast_ops)
    _log_step("schema scan + stats column casts", step_start, min_seconds=0.0)
    
    # 5. Write results
    # We produce TWO files:
    # A. The full wide Seat 1 table
    # B. A compact stats-only table for the API Explorer (Completed Auctions only)
    
    stats_output_path = output_path.parent / "bbo_bt_criteria_seat1_df.parquet"
    
    print(f"Streaming Wide Table to {output_path}...")
    print(f"Streaming Compact Stats to {stats_output_path}...")
    
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Wide table sink
    step_start = time.time()
    lf.sink_parquet(output_path)
    _log_step("write wide seat1 parquet", step_start, min_seconds=0.0)
    
    # Compact stats sink (filter to completed auctions and select only stats/index)
    # The API expects 'bt_index' as the key.
    stats_lf = pl.scan_parquet(output_path).filter(pl.col("is_completed_auction"))
    
    # Select index and all stats columns (matching_deal_count, _min, _max, _mean, _std)
    step_start = time.time()
    stats_cols = ["bt_index"] + [c for c in stats_lf.collect_schema().names() 
                                if any(x in c for x in ["matching_deal_count", "_S1", "_S2", "_S3", "_S4"])]
    stats_lf.select(stats_cols).sink_parquet(stats_output_path)
    _log_step("write compact criteria stats parquet", step_start, min_seconds=0.0)
    
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.")
    print(f"  Wide Table: {output_path.stat().st_size / 1024**2:.1f} MB")
    print(f"  Stats Table: {stats_output_path.stat().st_size / 1024**2:.1f} MB")
    print(f"  Total build elapsed: {_format_elapsed(time.time() - build_start)}")

def main():
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("=" * 70)
    print("BBO SEAT-1 BIDDING TABLE BUILDER")
    print("=" * 70)
    print(f"Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    parser = argparse.ArgumentParser(description="Build seat-1-only bidding table (Fast)")
    parser.add_argument("--bt-rows", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--ev-gpu-stats", type=str, default=None,
                        help="Path to ev_gpu v2/v3 parquet (replaces bbo_bt_stats/ directory)")
    args = parser.parse_args()
    
    data_dir = pathlib.Path("e:/bridge/data/bbo/bidding")
    bt_augmented_path = data_dir / "bbo_bt_augmented.parquet"
    stats_dir = data_dir / "bbo_bt_stats"
    
    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        output_path = data_dir / "bbo_bt_seat1.parquet"
    
    ev_gpu_stats_path = pathlib.Path(args.ev_gpu_stats) if args.ev_gpu_stats else None
    
    build_seat1_table_fast(
        bt_augmented_path=bt_augmented_path,
        stats_dir=stats_dir,
        output_path=output_path,
        n_rows=args.bt_rows,
        ev_gpu_stats_path=ev_gpu_stats_path,
    )
    
    # Final summary
    end_datetime = datetime.now()
    total_elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"End:     {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {_format_elapsed(total_elapsed)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
