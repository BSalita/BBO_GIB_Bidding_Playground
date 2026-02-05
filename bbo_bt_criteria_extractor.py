"""

Takes 3m-5m to extract criteria for 2.1M rows.

BT Criteria Extractor Script

Reads bbo_bt_augmented.parquet. extracts min/max criteria columns. creates bbo_bt_criteria.parquet.
from Agg_Expr_Seat_[1-4] columns.

Each Agg_Expr_Seat column contains a list of expression strings like:
  ["HCP <= 12", "HCP >= 8", "SL_H == 3", "SL_C <= 5", ...]

This script extracts HCP, SL_[CDHS], Total_Points constraints and creates
12 columns per seat (6 value pairs × min/max) = 48 total columns.

IMPORTANT: Filters to is_completed_auction=True first (541M → 2.1M rows)

Usage:
    python bt_criteria_extractor.py
    python bt_criteria_extractor.py --bt-rows 10000
    python bt_criteria_extractor.py --no-filter  # Include all rows (slow!)

previous:
    BBO-bidding/bbo_create_execution_plan.py
next:
    bbo_bt_aggregate.py
    bbo_bidding_queries_api.py
    bbo_bidding_queries_streamlit.py
"""

import argparse
import pathlib
import re
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


# Column definitions for extracted criteria
CRITERIA_COLS = ["HCP", "SL_C", "SL_D", "SL_H", "SL_S", "Total_Points"]

# Default min/max values when no constraint specified
DEFAULT_RANGES = {
    "HCP": (0, 40),
    "SL_C": (0, 13),
    "SL_D": (0, 13),
    "SL_H": (0, 13),
    "SL_S": (0, 13),
    "Total_Points": (0, 50),
}

# Precompiled regex patterns
PATTERN_LE = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*<=\s*(\d+)$")
PATTERN_GE = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*>=\s*(\d+)$")
PATTERN_EQ = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*==\s*(\d+)$")
PATTERN_LT = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*<\s*(\d+)$")
PATTERN_GT = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*>\s*(\d+)$")


def parse_expressions(expr_list: list[str] | None) -> dict[str, tuple[int, int]]:
    """
    Parse a list of expression strings and extract min/max for each criteria column.
    """
    if expr_list is None:
        return {col: DEFAULT_RANGES[col] for col in CRITERIA_COLS}
    
    mins = {col: [] for col in CRITERIA_COLS}
    maxs = {col: [] for col in CRITERIA_COLS}
    
    for expr in expr_list:
        expr = expr.strip()
        
        match = PATTERN_GE.match(expr)
        if match:
            col, val = match.groups()
            mins[col].append(int(val))
            continue
        
        match = PATTERN_LE.match(expr)
        if match:
            col, val = match.groups()
            maxs[col].append(int(val))
            continue
        
        match = PATTERN_EQ.match(expr)
        if match:
            col, val = match.groups()
            val = int(val)
            mins[col].append(val)
            maxs[col].append(val)
            continue
        
        match = PATTERN_GT.match(expr)
        if match:
            col, val = match.groups()
            mins[col].append(int(val) + 1)
            continue
        
        match = PATTERN_LT.match(expr)
        if match:
            col, val = match.groups()
            maxs[col].append(int(val) - 1)
            continue
    
    result = {}
    for col in CRITERIA_COLS:
        default_min, default_max = DEFAULT_RANGES[col]
        min_val = max(mins[col]) if mins[col] else default_min
        max_val = min(maxs[col]) if maxs[col] else default_max
        result[col] = (min_val, max_val)
    
    return result


def parse_expr_to_tuple(expr_list: list[str] | None) -> tuple:
    """Parse expressions and return flat tuple of (min, max) pairs for all criteria."""
    parsed = parse_expressions(expr_list)
    # Return flat tuple: (HCP_min, HCP_max, SL_C_min, SL_C_max, ...)
    result = []
    for col in CRITERIA_COLS:
        result.extend(parsed[col])
    return tuple(result)


def extract_criteria_for_seat_fast(bt_df: pl.DataFrame, seat: int) -> pl.DataFrame:
    """
    Extract criteria columns for a specific seat using Polars map_elements.
    """
    col_name = f"Agg_Expr_Seat_{seat}"
    
    if col_name not in bt_df.columns:
        raise ValueError(f"Column {col_name} not found in bt_df")
    
    # Use map_elements to parse each row - returns struct with all values
    n_rows = len(bt_df)
    print(f"    Processing {n_rows:,} rows...")
    
    # Define output column names
    output_cols = []
    for col in CRITERIA_COLS:
        output_cols.append(f"{col}_min_S{seat}")
        output_cols.append(f"{col}_max_S{seat}")
    
    # Use map_elements with return type
    t0 = time.time()
    
    # Process using map_elements - returns list of tuples
    parsed_series = bt_df[col_name].map_elements(
        parse_expr_to_tuple,
        return_dtype=pl.List(pl.UInt8)
    )
    
    print(f"    Parsed expressions in {time.time() - t0:.1f}s")
    
    # Convert to DataFrame columns
    t1 = time.time()
    
    # Extract each value from the list into separate columns
    result_df = parsed_series.to_frame("parsed").select([
        pl.col("parsed").list.get(i).alias(output_cols[i])
        for i in range(len(output_cols))
    ])
    
    print(f"    Built columns in {time.time() - t1:.1f}s")
    
    return result_df


def extract_all_criteria(bt_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract criteria columns for all 4 seats.
    """
    result_dfs = []
    
    for seat in range(1, 5):
        print(f"  Seat {seat}:")
        t0 = time.time()
        seat_df = extract_criteria_for_seat_fast(bt_df, seat)
        print(f"    Total: {time.time() - t0:.1f}s")
        result_dfs.append(seat_df)
    
    return pl.concat(result_dfs, how="horizontal")


def main():
    start_time = time.time()
    start_datetime = datetime.now()
    
    parser = argparse.ArgumentParser(description="BT Criteria Extractor")
    parser.add_argument("--bt-rows", type=int, default=None, help="Number of bt_df rows to process (None=all)")
    parser.add_argument("--bt-parquet", type=str, default=None, help="Path to bt parquet file")
    parser.add_argument("--output", type=str, default=None, help="Output parquet file path")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter to completed auctions (slow!)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("BT CRITERIA EXTRACTOR")
    print("=" * 70)
    print(f"Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Find bt parquet file
    bt_parquet_path = None
    candidates = [
        pathlib.Path("e:/bridge/data/bbo/bidding/bbo_bt_augmented.parquet"),
    ]
    
    if args.bt_parquet:
        bt_parquet_path = pathlib.Path(args.bt_parquet)
    else:
        for p in candidates:
            if p.exists():
                bt_parquet_path = p
                break
    
    if not bt_parquet_path or not bt_parquet_path.exists():
        print("ERROR: bt parquet file not found!")
        print("Tried:", [args.bt_parquet] if args.bt_parquet else candidates)
        return
    
    print(f"Loading bt_df from: {bt_parquet_path}")
    
    # Load required columns (include is_completed_auction for filtering)
    t0 = time.time()
    cols_to_load = [
        "is_completed_auction",
        "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4"
    ]
    
    if args.bt_rows:
        bt_df = pl.read_parquet(bt_parquet_path, columns=cols_to_load, n_rows=args.bt_rows)
    else:
        bt_df = pl.read_parquet(bt_parquet_path, columns=cols_to_load)
    
    print(f"  Loaded {len(bt_df):,} rows in {time.time() - t0:.2f}s")
    
    # Filter to completed auctions (reduces 541M to ~2.1M)
    if not args.no_filter:
        t0 = time.time()
        original_count = len(bt_df)
        bt_df = bt_df.filter(pl.col("is_completed_auction"))
        print(f"  Filtered to completed auctions: {original_count:,} → {len(bt_df):,} rows in {time.time() - t0:.2f}s")
    
    # Drop the filter column
    bt_df = bt_df.drop("is_completed_auction")
    print()
    
    # Extract criteria
    print("Extracting criteria columns...")
    t0 = time.time()
    criteria_df = extract_all_criteria(bt_df)
    print(f"\nExtracted {len(criteria_df.columns)} columns in {time.time() - t0:.1f}s total")
    print()
    
    # Show sample
    print("Sample of extracted criteria (first 5 rows):")
    print(criteria_df.head(5))
    print()
    
    # Show column names
    print(f"Columns ({len(criteria_df.columns)}):")
    for i, col in enumerate(criteria_df.columns):
        if i < 12:
            print(f"  {col}")
    print("  ...")
    print()
    
    # Save output
    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        n_rows_str = f"_{args.bt_rows}" if args.bt_rows else ""
        filter_str = "_all" if args.no_filter else ""
        output_path = pathlib.Path(f"e:/bridge/data/bbo/bidding/bbo_bt_criteria{n_rows_str}{filter_str}.parquet")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    criteria_df.write_parquet(output_path)
    print(f"Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Rows: {len(criteria_df):,}")
    
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
