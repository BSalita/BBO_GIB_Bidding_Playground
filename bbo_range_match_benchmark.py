"""
Range Match Benchmark Script

Tests performance of matching deal_df rows against bt_df range criteria.
bt_df has 12 columns per seat: min/max ranges for HCP, SL_[CDHS], Total_Points
deal_df has 6 columns: HCP, SL_[CDHS], Total_Points
Match is True if all deal_df values are between corresponding bt_df min/max.

Usage:
    python range_match_benchmark.py --seat 1
    python range_match_benchmark.py --seat 1 --bt-rows 10000 --deal-rows 100000
    python range_match_benchmark.py --seat 1 --workers 4
"""

import argparse
import pathlib
import random
import sys
import time
from multiprocessing import Pool, cpu_count
import polars as pl


# Column definitions
CRITERIA_COLS = ["HCP", "SL_C", "SL_D", "SL_H", "SL_S", "Total_Points"]

# bt_df: 12 columns (6 value pairs with min/max) - standard names after renaming
BT_COLS = [
    "HCP_min", "HCP_max",
    "SL_C_min", "SL_C_max",
    "SL_D_min", "SL_D_max",
    "SL_H_min", "SL_H_max",
    "SL_S_min", "SL_S_max",
    "Total_Points_min", "Total_Points_max",
]

# deal_df: 6 columns matching bt criteria
DEAL_COLS = ["HCP", "SL_C", "SL_D", "SL_H", "SL_S", "Total_Points"]

# Realistic value ranges for bridge data (for synthetic generation)
VALUE_RANGES = {
    "HCP": (0, 37),
    "SL_C": (0, 13),
    "SL_D": (0, 13),
    "SL_H": (0, 13),
    "SL_S": (0, 13),
    "Total_Points": (0, 40),
}

# Global variable for worker processes to access deal_df
_worker_deal_df: pl.DataFrame | None = None


def _init_worker(deal_df_bytes: bytes):
    """Initialize worker process with deal_df."""
    global _worker_deal_df
    _worker_deal_df = pl.read_ipc(deal_df_bytes)


def _process_bt_chunk(bt_chunk_bytes: bytes) -> int:
    """Process a chunk of bt_df rows and return total matches."""
    global _worker_deal_df
    assert _worker_deal_df is not None, "Worker not initialized"
    
    deal_df = _worker_deal_df  # Local reference for type checker
    bt_chunk = pl.read_ipc(bt_chunk_bytes)
    total_matches = 0
    
    for row in bt_chunk.iter_rows(named=True):
        filter_expr = pl.lit(True)
        for col in CRITERIA_COLS:
            min_val = row[f"{col}_min"]
            max_val = row[f"{col}_max"]
            filter_expr = filter_expr & (
                (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
            )
        
        match_count = deal_df.lazy().filter(filter_expr).select(pl.len()).collect().item()
        total_matches += match_count
    
    return total_matches


def scan_bt_criteria(parquet_path: pathlib.Path, seat: int, n_rows: int | None = None) -> pl.LazyFrame:
    """
    Lazily scan bt_criteria parquet file for a specific seat.
    
    The parquet has columns like: HCP_min_S1, HCP_max_S1, etc.
    This function selects columns for the specified seat and renames them to standard names.
    """
    # Build column mapping: HCP_min_S1 -> HCP_min
    seat_cols = []
    rename_map = {}
    for col in CRITERIA_COLS:
        min_col = f"{col}_min_S{seat}"
        max_col = f"{col}_max_S{seat}"
        seat_cols.extend([min_col, max_col])
        rename_map[min_col] = f"{col}_min"
        rename_map[max_col] = f"{col}_max"
    
    lf = pl.scan_parquet(parquet_path)
    lf = lf.select(seat_cols)
    
    if n_rows is not None:
        lf = lf.head(n_rows)
    
    lf = lf.rename(rename_map)
    
    return lf


def generate_bt_df(n_rows: int, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic bt_df with min/max range columns (fallback)."""
    random.seed(seed)
    
    data = {}
    for col in CRITERIA_COLS:
        vmin, vmax = VALUE_RANGES[col]
        mins = [random.randint(vmin, vmax - 2) for _ in range(n_rows)]
        maxs = [min(m + random.randint(2, min(8, vmax - vmin)), vmax) for m in mins]
        data[f"{col}_min"] = mins
        data[f"{col}_max"] = maxs
    
    return pl.DataFrame(data).cast({col: pl.UInt8 for col in BT_COLS})


def get_deal_col_mapping() -> dict[str, str]:
    """Map our column names to actual parquet column names (North's hand)."""
    return {
        "HCP": "HCP_N",
        "SL_C": "SL_N_C",
        "SL_D": "SL_N_D", 
        "SL_H": "SL_N_H",
        "SL_S": "SL_N_S",
        "Total_Points": "Total_Points_N",
    }


def scan_deal_df(parquet_path: pathlib.Path, n_rows: int | None = None) -> pl.LazyFrame:
    """Lazily scan deal_df from parquet file."""
    col_mapping = get_deal_col_mapping()
    actual_cols = list(col_mapping.values())
    
    lf = pl.scan_parquet(parquet_path)
    lf = lf.select(actual_cols)
    
    if n_rows is not None:
        lf = lf.head(n_rows)
    
    rename_map = {v: k for k, v in col_mapping.items()}
    lf = lf.rename(rename_map)
    lf = lf.cast({col: pl.UInt8 for col in DEAL_COLS})
    
    return lf


def generate_deal_df(n_rows: int, seed: int = 123) -> pl.DataFrame:
    """Generate synthetic deal_df (fallback if parquet not available)."""
    random.seed(seed)
    
    data = {}
    for col in DEAL_COLS:
        vmin, vmax = VALUE_RANGES[col]
        data[col] = [random.randint(vmin, vmax) for _ in range(n_rows)]
    
    return pl.DataFrame(data).cast({col: pl.UInt8 for col in DEAL_COLS})


def run_benchmark(
    bt_lf: pl.LazyFrame, 
    deal_lf: pl.LazyFrame,
    batch_size: int = 1000,
    workers: int = 1
) -> dict:
    """
    Run benchmark using Polars lazy evaluation.
    
    Args:
        bt_lf: LazyFrame with bt criteria
        deal_lf: LazyFrame with deal data
        batch_size: Batch size for progress reporting
        workers: Number of parallel workers (1 = sequential)
    """
    t0 = time.time()
    
    # Collect both dataframes
    deal_df = deal_lf.with_row_index("_deal_idx").collect()
    bt_df = bt_lf.collect()
    n_deals = len(deal_df)
    n_bt = len(bt_df)
    
    prep_time = time.time() - t0
    print(f"  Collected deal_df: {n_deals:,} rows, bt_df: {n_bt:,} rows in {prep_time:.2f}s")
    
    t1 = time.time()
    
    if workers == 1:
        # Sequential processing
        total_matches = _run_sequential(bt_df, deal_df, n_bt, batch_size, t1)
    else:
        # Parallel processing
        total_matches = _run_parallel(bt_df, deal_df, n_bt, workers, t1)
    
    match_time = time.time() - t1
    total_time = time.time() - t0
    
    return {
        "prep_time": prep_time,
        "match_time": match_time,
        "total_time": total_time,
        "total_matches": total_matches,
        "bt_rows": n_bt,
        "deal_rows": n_deals,
        "batch_size": batch_size,
        "workers": workers,
    }


def _run_sequential(bt_df: pl.DataFrame, deal_df: pl.DataFrame, n_bt: int, batch_size: int, t1: float) -> int:
    """Run benchmark sequentially (single worker)."""
    total_matches = 0
    
    for batch_start in range(0, n_bt, batch_size):
        batch_end = min(batch_start + batch_size, n_bt)
        bt_batch = bt_df.slice(batch_start, batch_end - batch_start)
        
        batch_matches = 0
        for row in bt_batch.iter_rows(named=True):
            filter_expr = pl.lit(True)
            for col in CRITERIA_COLS:
                min_val = row[f"{col}_min"]
                max_val = row[f"{col}_max"]
                filter_expr = filter_expr & (
                    (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
                )
            
            match_count = deal_df.lazy().filter(filter_expr).select(pl.len()).collect().item()
            batch_matches += match_count
        
        total_matches += batch_matches
        
        # Progress
        if (batch_start // batch_size) % 10 == 0 or batch_end == n_bt:
            elapsed = time.time() - t1
            pct = batch_end / n_bt * 100
            rate = batch_end / elapsed if elapsed > 0 else 0
            eta = (n_bt - batch_end) / rate if rate > 0 else 0
            print(f"  Processed {batch_end:,}/{n_bt:,} bt rows ({pct:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s - ETA: {eta:.0f}s")
    
    return total_matches


def _run_parallel(bt_df: pl.DataFrame, deal_df: pl.DataFrame, n_bt: int, workers: int, t1: float) -> int:
    """Run benchmark in parallel across multiple workers."""
    print(f"  Starting {workers} workers...")
    
    # Serialize deal_df for workers
    deal_df_bytes = deal_df.write_ipc(None).getvalue()
    
    # Split bt_df into chunks for workers
    chunk_size = max(1, n_bt // (workers * 4))  # Create ~4 chunks per worker for better load balancing
    chunks = []
    for i in range(0, n_bt, chunk_size):
        chunk = bt_df.slice(i, min(chunk_size, n_bt - i))
        chunks.append(chunk.write_ipc(None).getvalue())
    
    print(f"  Split into {len(chunks)} chunks of ~{chunk_size:,} rows each")
    
    # Process chunks in parallel
    total_matches = 0
    chunks_done = 0
    
    with Pool(workers, initializer=_init_worker, initargs=(deal_df_bytes,)) as pool:
        for chunk_matches in pool.imap_unordered(_process_bt_chunk, chunks):
            total_matches += chunk_matches
            chunks_done += 1
            
            # Progress
            if chunks_done % max(1, len(chunks) // 10) == 0 or chunks_done == len(chunks):
                elapsed = time.time() - t1
                pct = chunks_done / len(chunks) * 100
                print(f"  Completed {chunks_done}/{len(chunks)} chunks ({pct:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s - Matches so far: {total_matches:,}")
    
    return total_matches


def main():
    parser = argparse.ArgumentParser(description="Range Match Benchmark")
    parser.add_argument("--seat", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Seat number to use for bt criteria (1-4)")
    parser.add_argument("--bt-rows", type=int, default=None, 
                        help="Number of bt_df rows to use (None=all)")
    parser.add_argument("--deal-rows", type=int, default=None, 
                        help="Number of deal_df rows to load (None=all)")
    parser.add_argument("--batch-size", type=int, default=1000, 
                        help="Batch size for progress reporting (sequential mode)")
    parser.add_argument("--workers", type=int, default=1,
                        help=f"Number of parallel workers (1=sequential, max={cpu_count()})")
    parser.add_argument("--bt-parquet", type=str, default=None, 
                        help="Path to bt_criteria parquet file")
    parser.add_argument("--deal-parquet", type=str, default=None, 
                        help="Path to deal parquet file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Range Match Benchmark")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Seat: {args.seat}")
    print(f"  bt_df rows: {args.bt_rows if args.bt_rows else 'all'}")
    print(f"  deal_df rows: {args.deal_rows if args.deal_rows else 'all'}")
    print(f"  Workers: {args.workers}")
    if args.workers == 1:
        print(f"  Batch size: {args.batch_size}")
    print()
    
    # Find bt_criteria parquet file
    if args.bt_parquet:
        bt_parquet_path = pathlib.Path(args.bt_parquet)
    else:
        bt_parquet_path = pathlib.Path("E:/bridge/data/bbo/bidding/bt_criteria.parquet")
    
    # Load bt_df
    print("Loading bt_df...")
    t0 = time.time()
    
    if bt_parquet_path.exists():
        print(f"  Loading from: {bt_parquet_path} (Seat {args.seat})")
        bt_lf = scan_bt_criteria(bt_parquet_path, args.seat, args.bt_rows)
        n_bt_rows = bt_lf.select(pl.len()).collect().item()
        print(f"  Scanned lazy frame: {n_bt_rows:,} rows in {time.time() - t0:.2f}s")
    else:
        print(f"  ERROR: bt_criteria parquet not found: {bt_parquet_path}")
        sys.exit(1)
    print()
    
    # Find deal parquet file
    if args.deal_parquet:
        deal_parquet_path = pathlib.Path(args.deal_parquet)
    else:
        deal_parquet_path = pathlib.Path("E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet")
    
    # Load deal_df
    print("Loading deal_df...")
    t0 = time.time()
    
    if deal_parquet_path.exists():
        print(f"  Loading from: {deal_parquet_path}")
        deal_lf = scan_deal_df(deal_parquet_path, args.deal_rows)
        n_deal_rows = deal_lf.select(pl.len()).collect().item()
        print(f"  Scanned lazy frame: {n_deal_rows:,} rows in {time.time() - t0:.2f}s")
    else:
        print(f"  ERROR: Deal parquet not found: {deal_parquet_path}")
        sys.exit(1)
    
    print(f"  Columns: {DEAL_COLS}")
    print()
    
    # Run benchmark
    print("Running benchmark...")
    print("-" * 40)
    
    results = run_benchmark(bt_lf, deal_lf, batch_size=args.batch_size, workers=args.workers)
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  bt_df rows:     {results['bt_rows']:,}")
    print(f"  deal_df rows:   {results['deal_rows']:,}")
    print(f"  Workers:        {results['workers']}")
    print(f"  Total matches:  {results['total_matches']:,}")
    print(f"  Avg matches/bt: {results['total_matches'] / results['bt_rows']:.1f}")
    print()
    print(f"  Prep time:      {results['prep_time']:.2f}s")
    print(f"  Match time:     {results['match_time']:.2f}s")
    print(f"  Total time:     {results['total_time']:.2f}s")
    print()
    
    # Extrapolate to full dataset
    if results['bt_rows'] < 2_000_000 or results['deal_rows'] < 16_000_000:
        scale_factor = (2_000_000 / results['bt_rows']) * (16_000_000 / results['deal_rows'])
        estimated_full = results['match_time'] * scale_factor
        print(f"  Estimated time for 2M x 16M: {estimated_full:.0f}s ({estimated_full/3600:.1f} hours)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
