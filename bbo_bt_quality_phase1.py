"""
BT Quality Monitor - Phase 1: Contradiction Detection

Finds logically impossible criteria combinations in the bidding table.

Fast vectorized implementation using Polars - runs in ~1-2 minutes on 66K rows.

Hardware target: 192GB RAM, 32 cores, fast SSD

Usage:
    python bbo_bt_quality_phase1.py
    python bbo_bt_quality_phase1.py --input E:/bridge/data/bbo/bidding/bbo_bt_merged_rules.parquet
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import re

# Configure Polars thread count before importing
# Use min(16, num_cores - 4) to leave headroom for OS and other processes
import multiprocessing
_num_cores = multiprocessing.cpu_count()
_polars_threads = min(16, max(1, _num_cores - 4))
os.environ["POLARS_MAX_THREADS"] = str(_polars_threads)

import polars as pl
from tqdm import tqdm

print(f"[config] CPU cores: {_num_cores}, Polars threads: {_polars_threads}")


# ============================================================================
# TIMING UTILITIES
# ============================================================================

def format_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_datetime(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("E:/bridge/data/bbo/bidding")
DEFAULT_INPUT = DATA_DIR / "bbo_bt_merged_rules.parquet"
OUTPUT_DIR = Path("E:/bridge/data/bbo/bidding/quality_reports")

# Metrics and their valid ranges
METRIC_RANGES = {
    "HCP": (0, 37),
    "Total_Points": (0, 40),
    "SL_S": (0, 13),
    "SL_H": (0, 13),
    "SL_D": (0, 13),
    "SL_C": (0, 13),
}

# Cross-metric constraints
# Total_Points >= HCP always (distribution points can only add)
# Sum of suit lengths = 13

# ============================================================================
# PARSING FUNCTIONS (Vectorized where possible)
# ============================================================================

CRITERION_PATTERN = re.compile(r'^(\w+)\s*(>=|<=|>|<|==)\s*(\d+)$')

def parse_criterion(criterion: str) -> Tuple[str, str, int] | None:
    """Parse a single criterion string."""
    if not criterion:
        return None
    m = CRITERION_PATTERN.match(criterion.strip())
    if m:
        return (m.group(1), m.group(2), int(m.group(3)))
    return None


def extract_bounds_from_criteria(criteria: List[str]) -> Dict[str, Tuple[int, int]]:
    """Extract min/max bounds for each metric from a list of criteria."""
    bounds: Dict[str, Tuple[int, int]] = {}
    
    for c in criteria or []:
        parsed = parse_criterion(str(c))
        if not parsed:
            continue
        
        metric, op, val = parsed
        lo, hi = bounds.get(metric, (0, 99))
        
        if op == '>=':
            lo = max(lo, val)
        elif op == '>':
            lo = max(lo, val + 1)
        elif op == '<=':
            hi = min(hi, val)
        elif op == '<':
            hi = min(hi, val - 1)
        elif op == '==':
            lo = max(lo, val)
            hi = min(hi, val)
        
        bounds[metric] = (lo, hi)
    
    return bounds


def detect_contradictions_in_row(criteria: List[str]) -> List[Dict[str, Any]]:
    """Detect all contradictions in a single row's criteria."""
    issues = []
    bounds = extract_bounds_from_criteria(criteria)
    
    # Issue Type 1: Impossible range (min > max)
    for metric, (lo, hi) in bounds.items():
        if lo > hi:
            issues.append({
                "type": "IMPOSSIBLE_RANGE",
                "metric": metric,
                "detail": f"{metric} >= {lo} AND {metric} <= {hi}",
                "severity": "ERROR",
            })
    
    # Issue Type 2: HCP > Total_Points (impossible)
    hcp_lo = bounds.get("HCP", (0, 37))[0]
    tp_hi = bounds.get("Total_Points", (0, 40))[1]
    if hcp_lo > tp_hi:
        issues.append({
            "type": "HCP_EXCEEDS_TP",
            "metric": "HCP/Total_Points",
            "detail": f"HCP >= {hcp_lo} but Total_Points <= {tp_hi}",
            "severity": "ERROR",
        })
    
    # Issue Type 3: Suit lengths sum > 13 (impossible)
    sl_mins = sum(bounds.get(f"SL_{s}", (0, 13))[0] for s in "SHDC")
    if sl_mins > 13:
        issues.append({
            "type": "SUIT_LENGTHS_EXCEED_13",
            "metric": "SL_*",
            "detail": f"Min suit lengths sum to {sl_mins} > 13",
            "severity": "ERROR",
        })
    
    # Issue Type 4: Suit lengths sum < 13 (impossible if all maxes set)
    sl_maxes = sum(bounds.get(f"SL_{s}", (0, 13))[1] for s in "SHDC")
    # Only flag if all 4 suits have explicit upper bounds
    all_suits_bounded = all(f"SL_{s}" in bounds for s in "SHDC")
    if all_suits_bounded and sl_maxes < 13:
        issues.append({
            "type": "SUIT_LENGTHS_BELOW_13",
            "metric": "SL_*",
            "detail": f"Max suit lengths sum to {sl_maxes} < 13",
            "severity": "ERROR",
        })
    
    # Issue Type 5: Very suspicious ranges (warnings, not errors)
    for metric, (lo, hi) in bounds.items():
        if metric == "HCP" and lo >= 20 and hi <= 21:
            # Very narrow HCP range - suspicious but valid
            pass
        if metric.startswith("SL_") and lo > 10:
            issues.append({
                "type": "EXTREME_SUIT_LENGTH",
                "metric": metric,
                "detail": f"{metric} >= {lo} (11+ cards in one suit)",
                "severity": "WARNING",
            })
    
    return issues


# ============================================================================
# VECTORIZED POLARS PROCESSING
# ============================================================================

def _safe_extract_bounds(crit: Any) -> Dict:
    """Safely extract bounds, handling None/empty values."""
    if crit is None:
        return {}
    if isinstance(crit, (list, tuple)) and len(crit) == 0:
        return {}
    # Cast to list of strings for type safety
    criteria_list: List[str] = list(crit) if isinstance(crit, (list, tuple)) else []
    return extract_bounds_from_criteria(criteria_list)


def _safe_detect_issues(crit: Any) -> List:
    """Safely detect issues, handling None/empty values."""
    if crit is None:
        return []
    if isinstance(crit, (list, tuple)) and len(crit) == 0:
        return []
    # Cast to list of strings for type safety
    criteria_list: List[str] = list(crit) if isinstance(crit, (list, tuple)) else []
    return detect_contradictions_in_row(criteria_list)


def _count_errors(issues) -> int:
    """Count ERROR severity issues."""
    if issues is None:
        return 0
    return len([i for i in issues if i.get("severity") == "ERROR"])


def _count_warnings(issues) -> int:
    """Count WARNING severity issues."""
    if issues is None:
        return 0
    return len([i for i in issues if i.get("severity") == "WARNING"])


def process_criteria_column_vectorized(df: pl.DataFrame, criteria_col: str) -> pl.DataFrame:
    """
    Process criteria column using Polars expressions where possible.
    
    For complex logic, we use map_elements but with efficient batching.
    """
    
    # Use map_elements with return_dtype for efficiency
    # Polars will parallelize this across all cores
    
    result = df.with_columns([
        # Extract bounds as struct for each row
        pl.col(criteria_col).map_elements(
            _safe_extract_bounds,
            return_dtype=pl.Object,
        ).alias("_bounds"),
        
        # Detect contradictions
        pl.col(criteria_col).map_elements(
            _safe_detect_issues,
            return_dtype=pl.Object,
        ).alias("_issues"),
    ])
    
    # Add has_issues boolean for fast filtering
    result = result.with_columns([
        pl.col("_issues").map_elements(
            lambda issues: len(issues) > 0 if issues else False,
            return_dtype=pl.Boolean,
        ).alias("has_issues"),
        
        pl.col("_issues").map_elements(
            _count_errors,
            return_dtype=pl.UInt32,
        ).alias("error_count"),
        
        pl.col("_issues").map_elements(
            _count_warnings,
            return_dtype=pl.UInt32,
        ).alias("warning_count"),
    ])
    
    return result


def analyze_sibling_consistency(df: pl.DataFrame) -> pl.DataFrame:
    """
    Find sibling bids (same prefix, different next_bid) and check consistency.
    
    Uses Polars group_by for efficient aggregation.
    """
    
    # Group by step_auction (prefix) to find siblings
    sibling_groups = df.group_by("step_auction").agg([
        pl.col("next_bid").alias("sibling_bids"),
        pl.col("Merged_Rules").alias("sibling_criteria"),
        pl.col("bt_index").alias("sibling_indices"),
        pl.len().alias("sibling_count"),
    ])
    
    # Filter to prefixes with multiple bids (actual siblings)
    multi_bid = sibling_groups.filter(pl.col("sibling_count") > 1)
    
    return multi_bid


def check_sibling_strength_consistency(siblings: Dict[str, List[str]]) -> List[Dict]:
    """Check if higher-level bids have appropriately higher strength requirements."""
    issues = []
    
    # Parse all bounds
    bounds_by_bid = {}
    for bid, criteria in siblings.items():
        bounds_by_bid[bid] = extract_bounds_from_criteria(criteria)
    
    # Compare bids in same strain at different levels
    strain_bids: Dict[str, List[Tuple[int, str]]] = {}
    for bid in siblings.keys():
        if len(bid) >= 2 and bid[0].isdigit():
            level = int(bid[0])
            strain = bid[1:]
            if strain not in strain_bids:
                strain_bids[strain] = []
            strain_bids[strain].append((level, bid))
    
    for strain, bids in strain_bids.items():
        if len(bids) < 2:
            continue
        
        sorted_bids = sorted(bids, key=lambda x: x[0])
        
        for i in range(len(sorted_bids) - 1):
            lower_level, lower_bid = sorted_bids[i]
            higher_level, higher_bid = sorted_bids[i + 1]
            
            lower_bounds = bounds_by_bid.get(lower_bid, {})
            higher_bounds = bounds_by_bid.get(higher_bid, {})
            
            # Higher level bid should generally have higher HCP floor
            lower_hcp_lo = lower_bounds.get("HCP", (0, 40))[0]
            higher_hcp_lo = higher_bounds.get("HCP", (0, 40))[0]
            
            # Exception: preempts (higher level = weaker hand)
            is_preempt_like = higher_level >= 3 and higher_bounds.get("HCP", (0, 40))[1] <= 10
            
            if not is_preempt_like and higher_hcp_lo < lower_hcp_lo:
                issues.append({
                    "type": "INVERTED_STRENGTH",
                    "bids": (lower_bid, higher_bid),
                    "detail": f"{higher_bid} (level {higher_level}) has LOWER HCP floor ({higher_hcp_lo}) than {lower_bid} ({lower_hcp_lo})",
                    "severity": "WARNING",
                })
    
    return issues


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def run_phase1_analysis(input_file: Path, output_dir: Path) -> Dict[str, Any]:
    """Run full Phase 1 quality analysis."""
    
    start_time = datetime.now()
    t0 = time.perf_counter()
    
    print("=" * 70)
    print("BT Quality Monitor - Phase 1: Contradiction Detection")
    print("=" * 70)
    print(f"Start:  {format_datetime(start_time)}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    print(f"Loading {input_file}...")
    t_load_start = time.perf_counter()
    df = pl.read_parquet(input_file)
    t_load = time.perf_counter() - t_load_start
    print(f"  Loaded {df.height:,} rows, {df.width} columns")
    print(f"  Columns: {df.columns}")
    print(f"  Load time: {format_elapsed(t_load)}")
    print()
    
    # Determine criteria column
    criteria_col = None
    for col in ["Merged_Rules", "criteria_with_metrics", "Agg_Expr"]:
        if col in df.columns:
            criteria_col = col
            break
    
    if criteria_col is None:
        print("ERROR: No criteria column found!")
        return {"error": "No criteria column"}
    
    print(f"Using criteria column: {criteria_col}")
    
    # Process for contradictions
    print("\nDetecting contradictions (vectorized, parallel)...")
    t1 = time.perf_counter()
    
    result = process_criteria_column_vectorized(df, criteria_col)
    
    t_process = time.perf_counter() - t1
    print(f"  Processing time: {format_elapsed(t_process)}")
    
    # Collect statistics
    total_rows = result.height
    rows_with_issues = result.filter(pl.col("has_issues")).height
    rows_with_errors = result.filter(pl.col("error_count") > 0).height
    rows_with_warnings = result.filter(pl.col("warning_count") > 0).height
    total_errors = result.select(pl.col("error_count").sum()).item()
    total_warnings = result.select(pl.col("warning_count").sum()).item()
    
    print(f"\n{'=' * 50}")
    print("CONTRADICTION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total rows analyzed:     {total_rows:>10,}")
    print(f"Rows with any issues:    {rows_with_issues:>10,} ({100*rows_with_issues/total_rows:.2f}%)")
    print(f"Rows with ERRORS:        {rows_with_errors:>10,} ({100*rows_with_errors/total_rows:.2f}%)")
    print(f"Rows with WARNINGS:      {rows_with_warnings:>10,} ({100*rows_with_warnings/total_rows:.2f}%)")
    print(f"Total ERROR count:       {total_errors:>10,}")
    print(f"Total WARNING count:     {total_warnings:>10,}")
    
    # Get issue breakdown
    print(f"\n{'=' * 50}")
    print("ISSUE TYPE BREAKDOWN")
    print(f"{'=' * 50}")
    
    issue_type_counts: Dict[str, int] = {}
    error_rows = result.filter(pl.col("has_issues"))
    error_row_count = error_rows.height
    
    # Use progress bar if many error rows
    if error_row_count > 5000:
        iterator = tqdm(
            error_rows.select(["_issues"]).iter_rows(),
            total=error_row_count,
            desc="  Counting issue types",
            unit="rows",
        )
    else:
        iterator = error_rows.select(["_issues"]).iter_rows()
    
    for row in iterator:
        issues = row[0]
        for issue in issues:
            itype = issue["type"]
            issue_type_counts[itype] = issue_type_counts.get(itype, 0) + 1
    
    for itype, count in sorted(issue_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {itype:30s}: {count:>6,}")
    
    # Sample errors for inspection
    print(f"\n{'=' * 50}")
    print("SAMPLE ERRORS (first 20)")
    print(f"{'=' * 50}")
    
    sample_errors = error_rows.head(20).select([
        "step_auction", "next_bid", "bt_index", "_issues"
    ])
    
    for row in sample_errors.iter_rows(named=True):
        auction = row.get("step_auction", "")
        next_bid = row.get("next_bid", "")
        issues = row.get("_issues", [])
        print(f"\n  Auction: {auction} → {next_bid}")
        for issue in issues:
            severity = issue["severity"]
            detail = issue["detail"]
            print(f"    [{severity}] {detail}")
    
    # Sibling consistency analysis
    print(f"\n{'=' * 50}")
    print("SIBLING CONSISTENCY ANALYSIS")
    print(f"{'=' * 50}")
    
    t2 = time.perf_counter()
    sibling_groups = analyze_sibling_consistency(df.select([
        "step_auction", "next_bid", "bt_index", criteria_col
    ]).rename({criteria_col: "Merged_Rules"}))
    
    sibling_issues = []
    sibling_count = sibling_groups.height
    
    # Use progress bar if this might take a while (> 1000 groups)
    if sibling_count > 1000:
        iterator = tqdm(
            sibling_groups.iter_rows(named=True),
            total=sibling_count,
            desc="  Checking siblings",
            unit="prefixes",
        )
    else:
        iterator = sibling_groups.iter_rows(named=True)
    
    for row in iterator:
        siblings = dict(zip(row["sibling_bids"], row["sibling_criteria"]))
        issues = check_sibling_strength_consistency(siblings)
        if issues:
            sibling_issues.append({
                "prefix": row["step_auction"],
                "issues": issues,
            })
    
    t_sibling = time.perf_counter() - t2
    print(f"  Sibling analysis time: {format_elapsed(t_sibling)}")
    print(f"  Prefixes with sibling issues: {len(sibling_issues)}")
    
    if sibling_issues:
        print(f"\n  Sample sibling issues (first 10):")
        for item in sibling_issues[:10]:
            prefix = item["prefix"]
            print(f"\n    Prefix: '{prefix}'")
            for issue in item["issues"]:
                print(f"      [{issue['severity']}] {issue['detail']}")
    
    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save error rows to parquet
    error_output = output_dir / f"phase1_errors_{timestamp}.parquet"
    error_rows.select([
        "step_auction", "next_bid", "bt_index", 
        "has_issues", "error_count", "warning_count"
    ]).write_parquet(error_output)
    print(f"\nSaved error rows to: {error_output}")
    
    # Final timing
    end_time = datetime.now()
    total_time = time.perf_counter() - t0
    
    # Save summary to JSON
    summary = {
        "start_time": format_datetime(start_time),
        "end_time": format_datetime(end_time),
        "elapsed_time": format_elapsed(total_time),
        "elapsed_seconds": round(total_time, 2),
        "timestamp": timestamp,
        "input_file": str(input_file),
        "total_rows": total_rows,
        "rows_with_issues": rows_with_issues,
        "rows_with_errors": rows_with_errors,
        "rows_with_warnings": rows_with_warnings,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "issue_type_counts": issue_type_counts,
        "sibling_issues_count": len(sibling_issues),
    }
    
    summary_output = output_dir / f"phase1_summary_{timestamp}.json"
    with open(summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_output}")
    
    # Final report
    print(f"\n{'=' * 70}")
    print("COMPLETED")
    print(f"{'=' * 70}")
    print(f"Start:   {format_datetime(start_time)}")
    print(f"End:     {format_datetime(end_time)}")
    print(f"Elapsed: {format_elapsed(total_time)}")
    print(f"{'=' * 70}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="BT Quality Monitor - Phase 1")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    run_phase1_analysis(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
