#!/usr/bin/env python3
"""
Invert Deal→Completed-BT Matches

Takes the deal→completed_bt mapping and inverts it to bt→deals.
Only includes completed auctions (not intermediate bids).
Limits each BT row to first N matching deals to keep output manageable.

Input:
  - bbo_mldf_augmented_matches.parquet (deal_idx, Matched_BT_Indices)

Output:
  - completed_bt_to_deals.parquet (bt_index, Deal_Indices as list, capped)

Estimated runtime: 5-10 minutes
Estimated output size: ~400 MB (975K rows × 100 deals max)
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

import numpy as np
import polars as pl

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MATCHES_FILE = Path("E:/bridge/data/bbo/data/bbo_mldf_augmented_matches.parquet")
DEFAULT_OUTPUT_FILE = Path("E:/bridge/data/bbo/bidding/completed_bt_to_deals.parquet")

# Maximum deals to store per BT row (keeps output size manageable)
MAX_DEALS_PER_BT = 100


def log(msg: str) -> None:
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def run_pipeline(
    matches_file: Path = DEFAULT_MATCHES_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    max_deals: Optional[int] = None,
    max_deals_per_bt: int = MAX_DEALS_PER_BT,
) -> None:
    """Main pipeline - invert deal→bt to bt→deals for completed auctions only."""
    start_time = datetime.now()
    log(f"=" * 70)
    log(f"Invert Completed BT Matches | Start: {start_time}")
    log(f"=" * 70)
    log(f"Matches file: {matches_file}")
    log(f"Output: {output_file}")
    log(f"Max deals per BT: {max_deals_per_bt}")
    
    # Step 1: Load matches
    log("[1/3] Loading matches...")
    step_start = time.time()
    
    if max_deals:
        matches_df = pl.scan_parquet(matches_file).head(max_deals).collect()
    else:
        matches_df = pl.read_parquet(matches_file)
    
    n_deals = len(matches_df)
    log(f"  Loaded {n_deals:,} deals with matches in {time.time() - step_start:.1f}s")
    
    # Get columns
    if 'deal_idx' in matches_df.columns:
        deal_indices = matches_df['deal_idx'].to_numpy()
    elif '_idx' in matches_df.columns:
        deal_indices = matches_df['_idx'].to_numpy()
    else:
        deal_indices = np.arange(n_deals, dtype=np.uint32)
    
    matched_bt_lists = matches_df['Matched_BT_Indices'].to_list()
    
    # Step 2: Invert mapping
    log("[2/3] Inverting mapping...")
    step_start = time.time()
    
    # bt_to_deals: bt_idx → list of deal_indices (capped)
    bt_to_deals: dict[int, list[int]] = defaultdict(list)
    
    total_pairs = sum(len(m) if m else 0 for m in matched_bt_lists)
    log(f"  Processing {total_pairs:,} (deal, completed_bt) pairs...")
    
    pbar = tqdm(total=n_deals, desc="Inverting", unit="deals") if HAS_TQDM else None
    
    pairs_added = 0
    pairs_capped = 0
    
    for deal_idx, bt_list in zip(deal_indices, matched_bt_lists):
        if bt_list is None or len(bt_list) == 0:
            if pbar:
                pbar.update(1)
            continue
        
        deal_idx_int = int(deal_idx)
        
        for bt_idx in bt_list:
            bt_idx_int = int(bt_idx)
            deals_list = bt_to_deals[bt_idx_int]
            
            if len(deals_list) < max_deals_per_bt:
                deals_list.append(deal_idx_int)
                pairs_added += 1
            else:
                pairs_capped += 1
        
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    log(f"  Inverted in {time.time() - step_start:.1f}s")
    log(f"  Pairs added: {pairs_added:,}")
    log(f"  Pairs capped: {pairs_capped:,}")
    log(f"  Unique completed BT rows: {len(bt_to_deals):,}")
    
    # Step 3: Save output
    log("[3/3] Saving output...")
    step_start = time.time()
    
    # Sort lists and prepare output
    bt_idx_list = sorted(bt_to_deals.keys())
    deal_lists = [sorted(bt_to_deals[bt_idx]) for bt_idx in bt_idx_list]
    
    # Stats
    deal_counts = [len(d) for d in deal_lists]
    avg_deals = sum(deal_counts) / len(deal_counts) if deal_counts else 0
    max_deals_actual = max(deal_counts) if deal_counts else 0
    at_cap = sum(1 for c in deal_counts if c >= max_deals_per_bt)
    
    log(f"  Stats:")
    log(f"    Completed BT rows: {len(bt_idx_list):,}")
    log(f"    Avg deals per BT: {avg_deals:.1f}")
    log(f"    Max deals per BT: {max_deals_actual:,}")
    log(f"    BT rows at cap ({max_deals_per_bt}): {at_cap:,}")
    
    # Create output DataFrame
    output_df = pl.DataFrame({
        'bt_index': pl.Series(bt_idx_list, dtype=pl.UInt32),
        'Deal_Indices': pl.Series(deal_lists, dtype=pl.List(pl.UInt32)),
    })
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_file)
    
    file_size_mb = output_file.stat().st_size / 1e6
    log(f"  Saved {file_size_mb:.1f} MB in {time.time() - step_start:.1f}s")
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    log(f"=" * 70)
    log(f"Invert Completed BT Matches | End: {end_time}")
    log(f"Total elapsed: {elapsed}")
    log(f"=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Invert deal→completed-BT matches")
    parser.add_argument(
        "--matches-file", type=Path, default=DEFAULT_MATCHES_FILE,
        help=f"Input matches parquet (default: {DEFAULT_MATCHES_FILE})"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_FILE,
        help=f"Output parquet file (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--max-deals", type=int, default=None,
        help="Maximum deals to process (for testing)"
    )
    parser.add_argument(
        "--max-deals-per-bt", type=int, default=MAX_DEALS_PER_BT,
        help=f"Max deals to store per BT row (default: {MAX_DEALS_PER_BT})"
    )
    
    args = parser.parse_args()
    
    if not args.matches_file.exists():
        print(f"ERROR: Matches file not found: {args.matches_file}")
        print("Run bbo_bt_deal_matches.py first to create it.")
        sys.exit(1)
    
    run_pipeline(
        matches_file=args.matches_file,
        output_file=args.output,
        max_deals=args.max_deals,
        max_deals_per_bt=args.max_deals_per_bt,
    )


if __name__ == "__main__":
    main()
