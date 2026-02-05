"""

takes 2h for 541m bt rows and 16m deal rows.

Fast Bidding Table Stats Computer (Gemini 3 Pro) - Chunked Version

Computes aggregate statistics (min/max/mean/std) for every row in the bidding table
by matching criteria against a pre-computed 'Deal Cube'.

Process:
1. Loads the Deal Cube.
2. Processes the Bidding Table in memory-safe chunks (e.g., 5M rows).
3. For each chunk:
   a. Extracts criteria.
   b. Computes stats (vectorized).
   c. Joins stats.
   d. Writes chunk to disk.

Output: A folder 'E:/bridge/data/bbo/bidding/bbo_bt_stats' containing partitioned parquet files.
This approach avoids memory explosion/pagefile thrashing.

Usage:
    python bbo_bt_compute_stats.py --chunk-size 5000000
"""

import argparse
import pathlib
import time
from datetime import datetime
import re
import shutil
import numpy as np
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

# ---------------------------------------------------------------------------
# Criteria Parsing Logic
# ---------------------------------------------------------------------------

CRITERIA_COLS = ["HCP", "SL_C", "SL_D", "SL_H", "SL_S", "Total_Points"]

DEFAULT_RANGES = {
    "HCP": (0, 40),
    "SL_C": (0, 13),
    "SL_D": (0, 13),
    "SL_H": (0, 13),
    "SL_S": (0, 13),
    "Total_Points": (0, 50),
}

# Regex patterns
PATTERN_LE = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*<=\s*(\d+)$")
PATTERN_GE = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*>=\s*(\d+)$")
PATTERN_EQ = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*==\s*(\d+)$")
PATTERN_LT = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*<\s*(\d+)$")
PATTERN_GT = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*>\s*(\d+)$")

def parse_expression_list(expr_list: list[str] | None) -> tuple:
    """
    Parse a list of expression strings and return flat tuple of (min, max) for all cols.
    """
    mins = {c: [] for c in CRITERIA_COLS}
    maxs = {c: [] for c in CRITERIA_COLS}
    
    if expr_list:
        for expr in expr_list:
            expr = expr.strip()
            # >=
            m = PATTERN_GE.match(expr)
            if m:
                c, v = m.groups()
                mins[c].append(int(v))
                continue
            # <=
            m = PATTERN_LE.match(expr)
            if m:
                c, v = m.groups()
                maxs[c].append(int(v))
                continue
            # ==
            m = PATTERN_EQ.match(expr)
            if m:
                c, v = m.groups()
                mins[c].append(int(v))
                maxs[c].append(int(v))
                continue
            # >
            m = PATTERN_GT.match(expr)
            if m:
                c, v = m.groups()
                mins[c].append(int(v) + 1)
                continue
            # <
            m = PATTERN_LT.match(expr)
            if m:
                c, v = m.groups()
                maxs[c].append(int(v) - 1)
                continue

    result = []
    for col in CRITERIA_COLS:
        d_min, d_max = DEFAULT_RANGES[col]
        final_min = max(mins[col]) if mins[col] else d_min
        final_max = min(maxs[col]) if maxs[col] else d_max
        result.append(final_min)
        result.append(final_max)
        
    return tuple(result)

# ---------------------------------------------------------------------------
# Stats Computation Logic
# ---------------------------------------------------------------------------

class CubeStatsComputer:
    def __init__(self, cube_df: pl.DataFrame):
        self.cube_df = cube_df
        self.n_buckets = len(cube_df)
        
        # Prepare numpy arrays for vectorized filtering
        self.cube_vals = {}
        for col in CRITERIA_COLS:
            # Cast to float64 to avoid overflow during squaring/aggregation
            self.cube_vals[col] = cube_df[col].to_numpy().astype(np.float64)
            
        self.counts = cube_df["count"].to_numpy().astype(np.float64)
        print(f"  Initialized CubeStatsComputer with {self.n_buckets:,} buckets")

    def compute_stats_for_unique_criteria(self, unique_df: pl.DataFrame, progress_prefix: str = "") -> pl.DataFrame:
        n_crit = len(unique_df)
        
        # Load criteria into numpy (N_crit, 1) for broadcasting
        c_min = {}
        c_max = {}
        for col in CRITERIA_COLS:
            c_min[col] = unique_df[f"{col}_min"].to_numpy()[:, None]
            c_max[col] = unique_df[f"{col}_max"].to_numpy()[:, None]
            
        # chunk_size here is for unique criteria expressions per batch.
        # With a Deal Cube of ~10K buckets, the mask is only ~50MB at chunk=5000.
        chunk_size = 5000
        
        # Output columns
        out_cols = ["matching_deal_count"]
        for col in CRITERIA_COLS:
            for stat in ["mean", "std", "min", "max"]:
                out_cols.append(f"{col}_{stat}")
        
        # Initialize result storage (nan initialized, float64)
        res_matrix = np.full((n_crit, len(out_cols)), np.nan, dtype=np.float64)
        col_to_idx = {c: i for i, c in enumerate(out_cols)}
        
        t0 = time.time()
        
        for i in range(0, n_crit, chunk_size):
            end = min(i + chunk_size, n_crit)
            current_chunk_len = end - i
            
            # Print ETA rarely (only if batch is huge)
            if i > 0 and n_crit > 20000:
                print(f"      {progress_prefix}Processed {i:,}/{n_crit:,} criteria...")

            # Allocate mask (C_chunk, B_cube)
            chunk_mask = np.ones((current_chunk_len, self.n_buckets), dtype=bool)
            
            for col in CRITERIA_COLS:
                col_vals = self.cube_vals[col] # (B,)
                chunk_min = c_min[col][i:end]  # (C, 1)
                chunk_max = c_max[col][i:end]  # (C, 1)
                
                # Apply filters one by one to save temporary memory
                # (This avoids creating multiple temp arrays)
                chunk_mask &= (col_vals >= chunk_min)
                chunk_mask &= (col_vals <= chunk_max)
            
            # Count
            deal_counts = (chunk_mask * self.counts).sum(axis=1) # (C,)
            res_matrix[i:end, col_to_idx["matching_deal_count"]] = deal_counts
            
            valid_rows = deal_counts > 0
            if not np.any(valid_rows):
                continue
                
            for col in CRITERIA_COLS:
                vals = self.cube_vals[col] # (B,)
                weighted_vals = vals * self.counts # (B,)
                col_sums = (chunk_mask * weighted_vals).sum(axis=1) # (C,)
                
                # Mean
                means = np.zeros_like(col_sums)
                np.divide(col_sums, deal_counts, out=means, where=valid_rows)
                res_matrix[i:end, col_to_idx[f"{col}_mean"]] = means
                
                # Std
                weighted_sq = (vals ** 2) * self.counts
                sq_sums = (chunk_mask * weighted_sq).sum(axis=1)
                mean_sq = np.zeros_like(sq_sums)
                np.divide(sq_sums, deal_counts, out=mean_sq, where=valid_rows)
                vars = mean_sq - (means ** 2)
                vars = np.maximum(vars, 0)
                stds = np.sqrt(vars)
                res_matrix[i:end, col_to_idx[f"{col}_std"]] = stds
                
                # Min / Max
                vals_b = np.broadcast_to(vals, (current_chunk_len, self.n_buckets))
                masked_vals = np.where(chunk_mask, vals_b, np.inf)
                mins = masked_vals.min(axis=1)
                mins = np.where(np.isinf(mins), np.nan, mins)
                res_matrix[i:end, col_to_idx[f"{col}_min"]] = mins
                
                masked_vals = np.where(chunk_mask, vals_b, -np.inf)
                maxs = masked_vals.max(axis=1)
                maxs = np.where(np.isinf(maxs), np.nan, maxs)
                res_matrix[i:end, col_to_idx[f"{col}_max"]] = maxs

        # Convert matrix to Polars
        res_dict = {"crit_hash": unique_df["crit_hash"]}
        for col in out_cols:
            res_dict[col] = res_matrix[:, col_to_idx[col]]
            
        return pl.DataFrame(res_dict)

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def compute_fast_stats_chunked(
    bt_parquet: pathlib.Path,
    cube_parquet: pathlib.Path,
    output_dir: pathlib.Path,
    chunk_size: int,
    seats: list[int] = [1, 2, 3, 4],
    debug_limit: int = 0
):
    print("="*60)
    print("Fast Bidding Table Stats Computer (Chunked)")
    print("="*60)
    
    # 1. Load Cube
    print(f"Loading Cube: {cube_parquet}")
    cube_df = pl.read_parquet(cube_parquet)
    computer = CubeStatsComputer(cube_df)
    
    # 2. Prepare Input/Output
    print(f"Scanning BT: {bt_parquet}")
    
    # Clean output dir
    if output_dir.exists():
        if debug_limit > 0:
            print(f"  Debug mode: Clearing {output_dir}...")
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        else:
            print(f"  Output dir {output_dir} exists.")
            # We assume user wants to overwrite or we should error?
            # Let's clean it to avoid mixing old/new parts
            print("  Clearing output directory to ensure fresh start...")
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
    else:
        output_dir.mkdir(parents=True)

    # Get total rows
    total_rows = pl.scan_parquet(bt_parquet).select(pl.len()).collect().item()
    if debug_limit > 0:
        total_rows = min(total_rows, debug_limit)
        
    n_chunks = (total_rows + chunk_size - 1) // chunk_size
    print(f"Total Rows: {total_rows:,}")
    print(f"Chunk Size: {chunk_size:,}")
    print(f"Total Chunks: {n_chunks}")
    
    # Columns to load
    cols_to_keep = ["index"] + [f"Agg_Expr_Seat_{s}" for s in seats]
    
    start_time = time.time()
    
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * chunk_size
        current_len = min(chunk_size, total_rows - offset)
        
        chunk_t0 = time.time()
        print(f"\nProcessing Chunk {chunk_idx+1}/{n_chunks} (Offset: {offset:,}, Len: {current_len:,})...")
        
        # Read chunk using slice (efficient in parquet - uses row groups)
        df_chunk = (
            pl.scan_parquet(bt_parquet)
            .select(cols_to_keep)
            .slice(offset, current_len)
            .collect()
        )
        
        # Base for this chunk
        df_result = df_chunk.select(cols_to_keep)
        
        # Process seats for this chunk
        for seat in seats:
            col_name = f"Agg_Expr_Seat_{seat}"
            
            # Hash
            # Optimization: We can compute hash on unique list items only?
            # No, we need hash for every row to join.
            # Polars list join hash is fast enough for 5M rows.
            hashed_chunk = df_chunk.select(
                pl.col("index"),
                pl.col(col_name).list.join("|").hash().alias("crit_hash")
            )
            
            # Unique Criteria
            unique_hashes = hashed_chunk.select("crit_hash").unique()
            # We need the expression too.
            # GroupBy hash and take first expression? Or join back?
            # Or just unique on hash and col.
            # Note: df_chunk is in memory.
            unique_exprs = (
                df_chunk.select(
                    pl.col(col_name).list.join("|").hash().alias("crit_hash"),
                    pl.col(col_name)
                )
                .unique(subset=["crit_hash"])
            )
            
            n_u = len(unique_exprs)
            # print(f"  Seat {seat}: {n_u} unique criteria")
            
            # Parse
            parsed_data = []
            hashes = unique_exprs["crit_hash"].to_list()
            exprs = unique_exprs[col_name].to_list()
            
            for h, e in zip(hashes, exprs):
                ranges = parse_expression_list(e)
                row = {"crit_hash": h}
                idx = 0
                for c in CRITERIA_COLS:
                    row[f"{c}_min"] = ranges[idx]
                    row[f"{c}_max"] = ranges[idx+1]
                    idx += 2
                parsed_data.append(row)
            
            parsed_df = pl.DataFrame(parsed_data)
            
            # Compute Stats
            stats_df = computer.compute_stats_for_unique_criteria(parsed_df, progress_prefix=f"C{chunk_idx+1} S{seat}: ")
            
            # Rename
            rename_map = {c: f"{c}_S{seat}" for c in stats_df.columns if c != "crit_hash"}
            stats_df = stats_df.rename(rename_map)
            
            # Join stats back to chunk rows
            # df_result has 'index'. hashed_chunk has 'index', 'crit_hash'.
            # stats_df has 'crit_hash'.
            
            # Helper to join
            stats_mapped = hashed_chunk.join(stats_df, on="crit_hash", how="left").drop("crit_hash")
            
            # Join to accumulated result
            df_result = df_result.join(stats_mapped, on="index", how="left")
            
        # --- DOWNCASTING FIX ---
        # 1. matching_deal_count -> UInt32 (up to 4B deals)
        # 2. _min, _max -> UInt8 (Bridge values are 0-50)
        # 3. _mean, _std -> Float32 (saves 50% RAM, enough precision)
        cast_ops = {}
        for c in df_result.columns:
            if "matching_deal_count" in c:
                cast_ops[c] = pl.col(c).fill_nan(None).fill_null(0).cast(pl.UInt32)
            elif any(x in c for x in ["_min", "_max"]):
                cast_ops[c] = pl.col(c).fill_nan(None).fill_null(0).cast(pl.UInt8)
            elif any(x in c for x in ["_mean", "_std"]):
                cast_ops[c] = pl.col(c).fill_nan(None).cast(pl.Float32)
        
        if cast_ops:
            df_result = df_result.with_columns(**cast_ops)
        # -----------------------

        # Drop the Agg_Expr_Seat_* columns - they're already in bt_augmented
        # and would create duplicate "_right" columns after the join
        agg_expr_cols = [c for c in df_result.columns if c.startswith("Agg_Expr_Seat_")]
        if agg_expr_cols:
            df_result = df_result.drop(agg_expr_cols)

        # Write Chunk
        out_file = output_dir / f"part_{chunk_idx:05d}.parquet"
        df_result.write_parquet(out_file)
        
        chunk_elapsed = time.time() - chunk_t0
        total_elapsed = time.time() - start_time
        avg_chunk_time = total_elapsed / (chunk_idx + 1)
        eta = avg_chunk_time * (n_chunks - (chunk_idx + 1))
        
        print(f"  Chunk saved to {out_file} ({chunk_elapsed:.1f}s)")
        print(f"  Progress: {chunk_idx+1}/{n_chunks}. Global ETA: {eta/60:.1f}m")

    print("\n" + "="*60)
    print("DONE. Dataset written to:")
    print(str(output_dir))
    print("="*60)


def main():
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("=" * 70)
    print("BBO BIDDING TABLE STATS COMPUTER (Chunked)")
    print("=" * 70)
    print(f"Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    parser = argparse.ArgumentParser(description="Fast Bidding Table Stats Computer (Chunked)")
    parser.add_argument("--bt-parquet", type=str, 
                        default="E:/bridge/data/bbo/bidding/bbo_bt_augmented.parquet")
    parser.add_argument("--cube-parquet", type=str,
                        default="E:/bridge/data/bbo/bidding/bbo_deal_cube.parquet")
    # Output is now a DIRECTORY
    parser.add_argument("--output", type=str,
                        default="E:/bridge/data/bbo/bidding/bbo_bt_stats")
    parser.add_argument("--limit", type=int, default=0, help="Debug row limit (0 = no limit)")
    parser.add_argument("--seats", type=str, default="1,2,3,4", help="Comma-separated list of seats")
    parser.add_argument("--chunk-size", type=int, default=5_000_000, help="Rows per chunk (default 5M)")
    
    args = parser.parse_args()
    
    seats_list = [int(s) for s in args.seats.split(",")]
    
    compute_fast_stats_chunked(
        pathlib.Path(args.bt_parquet),
        pathlib.Path(args.cube_parquet),
        pathlib.Path(args.output),
        chunk_size=args.chunk_size,
        seats=seats_list,
        debug_limit=args.limit
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
