"""
Criteria-Category Correlation Analysis

Analyzes relationships between bid categories and criteria patterns to:
1. Discover which criteria predict which categories
2. Find anomalies where categories and criteria don't align
3. Suggest improvements to both systems

Input: data/bbo_bt_categories.parquet (must have both categories and criteria)
Output: quality_reports/criteria_category_analysis_YYYYMMDD_HHMMSS.json

Usage:
    python bbo_bt_criteria_category_analysis.py
"""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Configure Polars threads
import multiprocessing
_num_cores = multiprocessing.cpu_count()
_polars_threads = min(16, max(1, _num_cores - 4))
os.environ["POLARS_MAX_THREADS"] = str(_polars_threads)

import polars as pl
from tqdm import tqdm

print(f"[config] CPU cores: {_num_cores}, Polars threads: {_polars_threads}")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("E:/bridge/data/bbo/bidding")
INPUT_FILE = DATA_DIR / "bbo_bt_categories.parquet"
MERGED_RULES_FILE = DATA_DIR / "bbo_bt_merged_rules.parquet"
OUTPUT_DIR = Path("E:/bridge/data/bbo/bidding/quality_reports")

# Categories we expect to correlate with specific criteria
EXPECTED_CORRELATIONS = {
    "is_Raise": ["SL_H >= 3", "SL_S >= 3", "SL_D >= 3", "SL_C >= 3"],
    "is_Preempt": ["HCP <=", "SL_"],
    "is_Strong": ["HCP >= 20", "HCP >= 22"],
    "is_Weak": ["HCP <="],
    "is_BalancedShowing": ["SL_"],
    "is_GameForcing": ["HCP >="],
}

# Criteria patterns to extract
CRITERION_PATTERN = re.compile(r'(\w+)\s*(>=|<=|>|<|==)\s*(\d+)')


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def extract_criteria_features(criteria_list: List[str]) -> Dict[str, Any]:
    """Extract structured features from a list of criteria strings."""
    if not criteria_list:
        return {}
    
    features = {
        "hcp_min": None,
        "hcp_max": None,
        "tp_min": None,
        "tp_max": None,
        "suit_lengths": {},  # {suit: (min, max)}
        "raw_criteria": criteria_list,
    }
    
    for crit in criteria_list:
        if not isinstance(crit, str):
            continue
        m = CRITERION_PATTERN.match(crit.strip())
        if not m:
            continue
        
        metric, op, val = m.group(1), m.group(2), int(m.group(3))
        
        if metric == "HCP":
            if op in (">=", ">"):
                features["hcp_min"] = max(features["hcp_min"] or 0, val)
            elif op in ("<=", "<"):
                features["hcp_max"] = min(features["hcp_max"] or 40, val)
        elif metric == "Total_Points":
            if op in (">=", ">"):
                features["tp_min"] = max(features["tp_min"] or 0, val)
            elif op in ("<=", "<"):
                features["tp_max"] = min(features["tp_max"] or 50, val)
        elif metric.startswith("SL_"):
            suit = metric[3:]
            lo, hi = features["suit_lengths"].get(suit, (0, 13))
            if op in (">=", ">"):
                lo = max(lo, val)
            elif op in ("<=", "<"):
                hi = min(hi, val)
            features["suit_lengths"][suit] = (lo, hi)
    
    return features


def analyze_category_criteria(df: pl.DataFrame, category: str, criteria_col: str) -> Dict:
    """Analyze criteria patterns for rows where a category is True."""
    
    cat_rows = df.filter(pl.col(category))
    if cat_rows.height == 0:
        return {"count": 0, "patterns": {}, "anomalies": []}
    
    # Collect criteria statistics
    criteria_counts = defaultdict(int)
    hcp_mins = []
    hcp_maxes = []
    suit_length_mins = defaultdict(list)
    
    for row in cat_rows.select([criteria_col]).iter_rows():
        criteria = row[0]
        if not criteria:
            continue
        
        features = extract_criteria_features(criteria)
        
        if features.get("hcp_min") is not None:
            hcp_mins.append(features["hcp_min"])
        if features.get("hcp_max") is not None:
            hcp_maxes.append(features["hcp_max"])
        
        for suit, (lo, hi) in features.get("suit_lengths", {}).items():
            if lo > 0:
                suit_length_mins[suit].append(lo)
        
        # Count individual criteria
        for crit in criteria:
            if isinstance(crit, str):
                # Normalize criteria for counting
                normalized = crit.strip()
                criteria_counts[normalized] += 1
    
    # Compute statistics
    result = {
        "count": cat_rows.height,
        "top_criteria": sorted(criteria_counts.items(), key=lambda x: -x[1])[:20],
        "hcp_min_stats": {
            "count": len(hcp_mins),
            "min": min(hcp_mins) if hcp_mins else None,
            "max": max(hcp_mins) if hcp_mins else None,
            "avg": sum(hcp_mins) / len(hcp_mins) if hcp_mins else None,
        },
        "hcp_max_stats": {
            "count": len(hcp_maxes),
            "min": min(hcp_maxes) if hcp_maxes else None,
            "max": max(hcp_maxes) if hcp_maxes else None,
            "avg": sum(hcp_maxes) / len(hcp_maxes) if hcp_maxes else None,
        },
        "suit_length_requirements": {
            suit: {
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "avg": sum(vals) / len(vals),
            }
            for suit, vals in suit_length_mins.items()
            if vals
        },
    }
    
    return result


def find_anomalies(df: pl.DataFrame, criteria_col: str) -> List[Dict]:
    """Find rows where categories and criteria seem inconsistent."""
    anomalies = []
    
    # Check is_Raise without appropriate criteria
    # Suit raises need SL_X >= 3, NT raises need HCP/TP criteria
    raise_rows = df.filter(pl.col("is_Raise"))
    for row in raise_rows.select(["step_auction", "next_bid", criteria_col]).head(200).iter_rows():
        auction, next_bid, criteria = row
        if not criteria:
            continue
        
        features = extract_criteria_features(criteria)
        is_nt_raise = next_bid and next_bid.upper().endswith('N')
        
        if is_nt_raise:
            # NT raises should have HCP or Total_Points criteria
            has_point_req = (
                features.get("hcp_min") is not None or 
                features.get("hcp_max") is not None or
                features.get("tp_min") is not None or
                features.get("tp_max") is not None
            )
            if not has_point_req:
                anomalies.append({
                    "type": "NT_RAISE_NO_POINTS",
                    "auction": auction,
                    "next_bid": next_bid,
                    "criteria": criteria,
                    "note": "NT raise but no HCP/Total_Points criterion",
                })
        # NOTE: Removed SUIT_RAISE_NO_SUPPORT check - suit raises may require 
        # 2, 3, or 4 cards depending on context. We rely on criteria correctness.
        
        if len(anomalies) >= 30:
            break
    
    # Check is_Preempt with high HCP
    if "is_Preempt" in df.columns:
        preempt_rows = df.filter(pl.col("is_Preempt"))
        for row in preempt_rows.select(["step_auction", "next_bid", criteria_col]).head(50).iter_rows():
            auction, next_bid, criteria = row
            if not criteria:
                continue
            
            features = extract_criteria_features(criteria)
            if features.get("hcp_min") and features["hcp_min"] >= 12:
                anomalies.append({
                    "type": "PREEMPT_HIGH_HCP",
                    "auction": auction,
                    "next_bid": next_bid,
                    "hcp_min": features["hcp_min"],
                    "note": "is_Preempt=True but HCP >= 12",
                })
    
    # Check is_Strong without high HCP
    # Note: HCP >= 19 is acceptable for 2C opening (strong, artificial)
    if "is_Strong" in df.columns:
        strong_rows = df.filter(pl.col("is_Strong"))
        for row in strong_rows.select(["step_auction", "next_bid", criteria_col]).head(50).iter_rows():
            auction, next_bid, criteria = row
            if not criteria:
                continue
            
            features = extract_criteria_features(criteria)
            # Threshold 19 allows for 2C opening which typically requires 19+ HCP
            if not features.get("hcp_min") or features["hcp_min"] < 19:
                anomalies.append({
                    "type": "STRONG_LOW_HCP",
                    "auction": auction,
                    "next_bid": next_bid,
                    "hcp_min": features.get("hcp_min"),
                    "note": "is_Strong=True but HCP min < 19",
                })
    
    return anomalies


def compute_category_correlations(df: pl.DataFrame) -> Dict[str, List[Tuple[str, float, int]]]:
    """Compute correlations between category pairs."""
    cat_cols = [c for c in df.columns if c.startswith("is_")]
    
    # Filter to non-zero categories
    active_cats = []
    for col in cat_cols:
        if df.filter(pl.col(col)).height > 0:
            active_cats.append(col)
    
    correlations = {}
    
    for cat in active_cats[:30]:  # Limit for performance
        cat_count = df.filter(pl.col(cat)).height
        if cat_count == 0:
            continue
        
        co_occurrences = []
        for other_cat in active_cats:
            if other_cat == cat:
                continue
            
            both_count = df.filter(pl.col(cat) & pl.col(other_cat)).height
            if both_count > 0:
                # Jaccard-like similarity
                other_count = df.filter(pl.col(other_cat)).height
                union = cat_count + other_count - both_count
                similarity = both_count / union if union > 0 else 0
                co_occurrences.append((other_cat, round(similarity, 3), both_count))
        
        # Sort by similarity
        co_occurrences.sort(key=lambda x: -x[1])
        correlations[cat] = co_occurrences[:10]
    
    return correlations


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = datetime.now()
    t0 = time.perf_counter()
    
    print("=" * 70)
    print("Criteria-Category Correlation Analysis")
    print("=" * 70)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print(f"Loading {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)
    print(f"  Loaded {df.height:,} rows, {df.width} columns")
    
    # Also load merged rules for criteria
    print(f"Loading {MERGED_RULES_FILE}...")
    merged_df = pl.read_parquet(MERGED_RULES_FILE)
    
    # Determine criteria column
    criteria_col = None
    for col in ["Merged_Rules", "criteria_with_metrics", "Agg_Expr"]:
        if col in merged_df.columns:
            criteria_col = col
            break
    
    if not criteria_col:
        print("ERROR: No criteria column found!")
        return
    
    print(f"  Using criteria column: {criteria_col}")
    
    # Join criteria to categories
    print("\nJoining criteria to categories...")
    df = df.join(
        merged_df.select(["step_auction", "next_bid", criteria_col]),
        on=["step_auction", "next_bid"],
        how="left"
    )
    
    # Get category columns
    cat_cols = [c for c in df.columns if c.startswith("is_")]
    active_cats = [c for c in cat_cols if df.filter(pl.col(c)).height > 0]
    print(f"  Active categories: {len(active_cats)}")
    
    # Analyze each category
    print("\nAnalyzing category-criteria relationships...")
    category_analysis = {}
    
    for cat in tqdm(active_cats, desc="  Categories"):
        category_analysis[cat] = analyze_category_criteria(df, cat, criteria_col)
    
    # Find anomalies
    print("\nFinding anomalies...")
    anomalies = find_anomalies(df, criteria_col)
    print(f"  Found {len(anomalies)} anomalies")
    
    # Compute category correlations
    print("\nComputing category correlations...")
    correlations = compute_category_correlations(df)
    
    # Generate report
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Top insights
    print("\n### Key Category-Criteria Patterns ###\n")
    
    key_categories = ["is_Raise", "is_Preempt", "is_Strong", "is_Weak", 
                      "is_Opening", "is_Response", "is_GameForcing"]
    
    for cat in key_categories:
        if cat not in category_analysis:
            continue
        analysis = category_analysis[cat]
        if analysis["count"] == 0:
            continue
        
        print(f"\n{cat} ({analysis['count']:,} rows):")
        
        # Top criteria
        if analysis["top_criteria"]:
            print("  Top criteria:")
            for crit, count in analysis["top_criteria"][:5]:
                pct = 100 * count / analysis["count"]
                print(f"    {crit}: {count:,} ({pct:.1f}%)")
        
        # HCP stats
        hcp_min = analysis["hcp_min_stats"]
        if hcp_min["count"] > 0:
            print(f"  HCP minimum: avg={hcp_min['avg']:.1f}, range=[{hcp_min['min']}, {hcp_min['max']}]")
        
        # Suit length requirements
        if analysis["suit_length_requirements"]:
            print("  Suit length requirements:")
            for suit, stats in analysis["suit_length_requirements"].items():
                print(f"    {suit}: avg>={stats['avg']:.1f} ({stats['count']} rows)")
    
    # Anomalies
    if anomalies:
        print(f"\n### Anomalies ({len(anomalies)} found) ###\n")
        for anom in anomalies[:10]:
            print(f"  [{anom['type']}] {anom['auction']} -> {anom['next_bid']}")
            print(f"    {anom['note']}")
    
    # Category correlations
    print("\n### Category Correlations (top pairs) ###\n")
    for cat in ["is_Raise", "is_Preempt", "is_Opening", "is_GameForcing"]:
        if cat not in correlations:
            continue
        pairs = correlations[cat][:5]
        if pairs:
            print(f"  {cat} correlates with:")
            for other, sim, count in pairs:
                print(f"    {other}: {sim:.2f} ({count:,} co-occurrences)")
    
    # Save full results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare JSON-serializable output
    output = {
        "timestamp": timestamp,
        "total_rows": df.height,
        "active_categories": len(active_cats),
        "anomalies": anomalies,
        "correlations": {k: [(o, s, c) for o, s, c in v] for k, v in correlations.items()},
        "category_summaries": {
            cat: {
                "count": analysis["count"],
                "top_criteria": analysis["top_criteria"][:10],
                "hcp_min_stats": analysis["hcp_min_stats"],
                "hcp_max_stats": analysis["hcp_max_stats"],
                "suit_length_requirements": analysis["suit_length_requirements"],
            }
            for cat, analysis in category_analysis.items()
            if analysis["count"] > 0
        },
    }
    
    output_file = OUTPUT_DIR / f"criteria_category_analysis_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full report to: {output_file}")
    
    # Final timing
    end_time = datetime.now()
    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
