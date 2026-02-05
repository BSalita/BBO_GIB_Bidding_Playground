"""
Bidding Table Rules Compiler - Chunked Streaming

takes 4h30m

Processes BT in 5M-row chunks to keep memory under 10GB.
Each chunk is fully processed then written to output before loading the next.
"""
import polars as pl
import pyarrow.parquet as pq
import pathlib
import time
import os
import gc
import argparse
from datetime import datetime
from typing import Dict
from tqdm import tqdm

DEFAULT_BASE_DIR = pathlib.Path(r"e:\bridge\data\bbo\bidding")
DEFAULT_INPUT_BT_NAME = "bbo_bt_seat1.parquet"
DEFAULT_INPUT_MERGED_NAME = "bbo_bt_merged_rules.parquet"
DEFAULT_OUTPUT_NAME = "bbo_bt_compiled.parquet"

CHUNK_SIZE = 5_000_000  # 5M rows per chunk


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile base + learned rules into a single BT parquet (chunked streaming).")
    p.add_argument(
        "--base-dir",
        type=str,
        default=os.environ.get("BBO_BIDDING_DATA_DIR", str(DEFAULT_BASE_DIR)),
        help="Directory containing input/output parquet files. Can also set env var BBO_BIDDING_DATA_DIR.",
    )
    p.add_argument("--input-bt", type=str, default=DEFAULT_INPUT_BT_NAME, help="Base BT parquet filename.")
    p.add_argument("--input-merged", type=str, default=DEFAULT_INPUT_MERGED_NAME, help="Merged rules parquet filename.")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT_NAME, help="Output compiled BT parquet filename.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in rows.")
    p.add_argument("--no-inner-progress", action="store_true", help="Disable per-chunk inner progress bar.")
    return p.parse_args()

def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def process_chunk(chunk: pl.DataFrame, prefix_dfs_by_len: Dict[int, pl.DataFrame], chunk_num: int = 0, show_progress: bool = True) -> pl.DataFrame:
    """Process a single chunk: apply all prefix-length joins and combine rules."""
    
    # Normalize auction and tokenize
    chunk = chunk.with_columns(
        pl.col("Auction").str.to_uppercase()
        .str.replace_all(r"^(P-)+", "")
        .alias("_norm")
    )
    chunk = chunk.with_columns(
        pl.col("_norm").str.split("-").alias("_tokens")
    )
    chunk = chunk.with_columns(
        pl.col("_tokens").list.len().alias("_len")
    )
    
    # Convert seat columns to list type if needed
    for seat in range(1, 5):
        col = f"Agg_Expr_Seat_{seat}"
        if col in chunk.columns:
            dtype = chunk[col].dtype
            if dtype == pl.Utf8 or dtype == pl.String:
                chunk = chunk.with_columns(
                    pl.col(col).str.split(";").list.eval(
                        pl.element().str.strip_chars().filter(pl.element().str.len_chars() > 0)
                    )
                )
    
    # Add row index for joins within chunk
    chunk = chunk.with_row_index("_idx")
    
    # Initialize merged columns as empty lists
    for seat in range(1, 5):
        chunk = chunk.with_columns(
            pl.lit([]).cast(pl.List(pl.Utf8)).alias(f"_m{seat}")
        )
    
    # Process each prefix length
    lengths = sorted(prefix_dfs_by_len.keys())
    total_matches = 0
    length_iter = tqdm(lengths, desc=f"    Chunk {chunk_num} lengths", leave=False, disable=not show_progress)
    for length in length_iter:
        prefix_df = prefix_dfs_by_len[length]
        # Filter to rows with enough tokens
        mask = chunk["_len"] >= length
        if mask.sum() == 0:
            continue
        
        # Extract prefix key
        candidates = chunk.filter(mask).select([
            "_idx",
            pl.col("_tokens").list.slice(0, length).list.join("-").alias("_key")
        ])
        
        # Join with prefix rules
        matches = candidates.join(
            prefix_df.rename({"prefix": "_key"}),
            on="_key",
            how="inner"
        )
        
        if matches.height == 0:
            del candidates, matches
            continue
        
        total_matches += matches.height
        length_iter.set_postfix({"len": length, "matches": f"{total_matches:,}"})
        
        # Aggregate by row and seat
        agg = matches.group_by(["_idx", "seat"]).agg(
            pl.col("rules_list").flatten().unique().alias("_rules")
        )
        
        del candidates, matches
        
        # Merge into chunk for each seat
        for seat in range(1, 5):
            seat_agg = agg.filter(pl.col("seat") == seat).select(["_idx", "_rules"])
            if seat_agg.height == 0:
                continue
            
            mcol = f"_m{seat}"
            chunk = chunk.join(
                seat_agg.rename({"_rules": "_inc"}),
                on="_idx",
                how="left"
            )
            chunk = chunk.with_columns(
                pl.when(pl.col("_inc").is_null())
                .then(pl.col(mcol))
                .otherwise(pl.concat_list([pl.col(mcol), pl.col("_inc")]))
                .alias(mcol)
            )
            chunk = chunk.drop("_inc")
        
        del agg
    
    # Combine base + merged for each seat
    for seat in range(1, 5):
        base_col = f"Agg_Expr_Seat_{seat}"
        mcol = f"_m{seat}"
        chunk = chunk.with_columns(
            pl.concat_list([pl.col(base_col), pl.col(mcol)])
            .list.eval(pl.element().filter(pl.element().is_not_null()))
            .list.unique()
            .alias(base_col)
        )
    
    # Drop temp columns
    drop_cols = ["_idx", "_norm", "_tokens", "_len"] + [f"_m{s}" for s in range(1, 5)]
    chunk = chunk.drop([c for c in drop_cols if c in chunk.columns])
    
    return chunk


def compile_bt() -> None:
    args = _parse_args()

    base_dir = pathlib.Path(args.base_dir)
    input_bt = base_dir / str(args.input_bt)
    input_merged = base_dir / str(args.input_merged)
    output_compiled = base_dir / str(args.output)
    chunk_size = int(args.chunk_size)
    show_inner_progress = not bool(args.no_inner_progress)

    start_dt = datetime.now()
    print(f"{'='*60}")
    print(f"Bidding Table Rules Compiler - Chunked Streaming")
    print(f"Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if not input_bt.exists():
        print(f"ERROR: Base BT not found: {input_bt}")
        return
    
    print(f"\nBase dir:      {base_dir}")
    print(f"Input BT:      {input_bt} ({format_size(os.path.getsize(input_bt))})")
    print(f"Input Merged:  {input_merged} ({format_size(os.path.getsize(input_merged)) if input_merged.exists() else 'N/A'})")
    print(f"Output:        {output_compiled}")
    print(f"Chunk size:    {chunk_size:,} rows")
    
    # =========================================================================
    # PHASE 1: Build prefix -> rules mapping
    # =========================================================================
    print(f"\n{'─'*60}")
    print("PHASE 1: Building prefix -> rules mapping")
    print(f"{'─'*60}")
    
    t0 = time.time()
    
    print("  Loading merged rules...")
    merged_df = pl.read_parquet(input_merged)
    
    mr_dtype = merged_df["Merged_Rules"].dtype
    if mr_dtype == pl.Utf8 or mr_dtype == pl.String:
        merged_df = merged_df.with_columns(
            pl.col("Merged_Rules").str.split(";").list.eval(
                pl.element().str.strip_chars().filter(pl.element().str.len_chars() > 0)
            ).alias("rules_list")
        )
    else:
        merged_df = merged_df.with_columns(
            pl.col("Merged_Rules").list.eval(
                pl.element().cast(pl.Utf8).str.strip_chars().filter(pl.element().str.len_chars() > 0)
            ).alias("rules_list")
        )
    
    merged_df = merged_df.filter(pl.col("rules_list").list.len() > 0)
    merged_bt_indices = set(merged_df["bt_index"].to_list())
    print(f"    → {len(merged_bt_indices):,} bt_indices have merged rules")
    
    print("  Loading auctions for merged bt_indices...")
    auc_df = pl.read_parquet(input_bt, columns=["bt_index", "Auction"])
    auc_df = auc_df.filter(pl.col("bt_index").is_in(list(merged_bt_indices)))
    
    auc_df = auc_df.with_columns(
        pl.col("Auction").str.to_uppercase()
        .str.replace_all(r"^(P-)+", "")
        .alias("prefix")
    ).filter(pl.col("prefix").str.len_chars() > 0)
    
    # IMPORTANT: Include 'seat' from merged_df - it tells which seat the learned rules apply to.
    # Previously we calculated seat from prefix dash count, which was WRONG - it gave the seat
    # that made the last bid, not the seat the rules describe.
    prefix_rules_join = auc_df.select(["bt_index", "prefix"]).join(
        merged_df.select(["bt_index", "seat", "rules_list"]),
        on="bt_index"
    )
    
    prefix_rules_join = prefix_rules_join.with_columns(
        (pl.col("prefix").str.count_matches("-") + 1).alias("prefix_len"),
        # Cast seat to Int32 for consistency (it comes from merged_df as Int64)
        pl.col("seat").cast(pl.Int32)
    )
    
    max_len_series = prefix_rules_join["prefix_len"].max()
    max_len: int = 0 if max_len_series is None else int(str(max_len_series))
    print(f"    → Max prefix length: {max_len} tokens")
    
    prefix_dfs_by_len: Dict[int, pl.DataFrame] = {}
    for length in range(1, max_len + 1):
        len_df = prefix_rules_join.filter(pl.col("prefix_len") == length).select([
            "prefix", "seat", "rules_list"
        ])
        if len_df.height > 0:
            prefix_dfs_by_len[length] = len_df
    
    print(f"    → {len(prefix_dfs_by_len)} prefix lengths with rules")
    
    del merged_df, auc_df, prefix_rules_join, merged_bt_indices
    gc.collect()
    
    print(f"  Phase 1 completed in {time.time() - t0:.1f}s")
    
    # =========================================================================
    # PHASE 2: Process chunks and stream to output
    # =========================================================================
    print(f"\n{'─'*60}")
    print("PHASE 2: Processing chunks")
    print(f"{'─'*60}")
    
    t0 = time.time()
    
    parquet_file = pq.ParquetFile(input_bt)
    total_rows = parquet_file.metadata.num_rows
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    print(f"  Total rows: {total_rows:,}")
    print(f"  Chunks: {num_chunks}")
    
    writer = None
    rows_processed = 0
    chunk_num = 0
    
    pbar = tqdm(total=total_rows, desc="  Processing", unit="rows")
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk_num += 1
        chunk_start = time.time()
        
        # Convert to Polars
        chunk_data = pl.from_arrow(batch)
        assert isinstance(chunk_data, pl.DataFrame)
        chunk: pl.DataFrame = chunk_data
        chunk_size = len(chunk)
        
        # Process this chunk
        processed = process_chunk(chunk, prefix_dfs_by_len, chunk_num=chunk_num, show_progress=show_inner_progress)
        
        # Write to output
        chunk_arrow = processed.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(output_compiled), chunk_arrow.schema)
        writer.write_table(chunk_arrow)
        
        rows_processed += chunk_size
        
        # Cleanup
        del chunk, processed, chunk_arrow
        gc.collect()
        
        pbar.update(chunk_size)
        
        chunk_elapsed = time.time() - chunk_start
        rows_per_sec = chunk_size / chunk_elapsed
        pbar.set_postfix({"rows/s": f"{rows_per_sec:,.0f}"})
    
    pbar.close()
    
    if writer:
        writer.close()
    
    print(f"  Phase 2 completed in {time.time() - t0:.1f}s")
    print(f"  Average: {rows_processed / (time.time() - t0):,.0f} rows/s")
    
    # =========================================================================
    # DONE
    # =========================================================================
    output_size = os.path.getsize(output_compiled) if output_compiled.exists() else 0
    end_dt = datetime.now()
    elapsed = end_dt - start_dt
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"Output:  {output_compiled}")
    print(f"Size:    {format_size(output_size)}")
    print(f"Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {elapsed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    compile_bt()
