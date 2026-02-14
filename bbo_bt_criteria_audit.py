"""
BT Criteria Audit Tool

Scans the Bidding Table for insufficiently specified criteria and produces
a structured report identifying nodes that need attention.

Deficiency categories detected:
  1. EMPTY_CRITERIA          — Agg_Expr is null or empty list for the acting seat
  2. NO_STRENGTH             — Has criteria but lacks both HCP and Total_Points
  3. UNBOUNDED_HCP           — Has HCP lower bound but no upper bound (or vice versa)
  4. UNBOUNDED_TOTAL_POINTS  — Has Total_Points lower bound but no upper bound (or vice versa)
  5. NO_SUIT_LENGTH          — Non-Pass bid at level 1+ with no suit-length constraint
  6. WIDE_HCP_RANGE          — HCP range spans 15+ points (very imprecise)
  7. WIDE_TP_RANGE           — Total_Points range spans 15+ points (very imprecise)
  8. DEAD_END                — No children and not a completed auction
  9. NO_CRITERIA_FOR_SEAT    — Criteria exist for some seats but not the acting seat

Output:  quality_reports/bt_criteria_audit_YYYYMMDD_HHMMSS.json
         quality_reports/bt_criteria_audit_YYYYMMDD_HHMMSS_summary.txt

Usage:
    python bbo_bt_criteria_audit.py [--max-nodes N] [--batch-size N]

Performance:
    The BT has ~461M rows. Agg_Expr columns are loaded on-demand in batches
    via DuckDB to avoid loading 100+ GB into memory. Default batch size is
    50,000 bt_indices. Full scan takes ~2-4 hours.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure Polars threads before import
import multiprocessing

_num_cores = multiprocessing.cpu_count()
_polars_threads = min(16, max(1, _num_cores - 4))
os.environ["POLARS_MAX_THREADS"] = str(_polars_threads)

import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("E:/bridge/data/bbo/bidding")
BT_FILE = DATA_DIR / "bbo_bt_compiled.parquet"
OUTPUT_DIR = Path("quality_reports")

# Criteria parsing
_CRITERION_PATTERN = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*(>=|<=|==|>|<)\s*(\d+)$")
_PASS_BIDS = {"P", "PASS"}
_STRAIN_BIDS = re.compile(r"^[1-7](C|D|H|S|NT?)$", re.IGNORECASE)

# HCP range considered "wide" (imprecise)
_WIDE_HCP_THRESHOLD = 15


# ---------------------------------------------------------------------------
# Criteria feature extraction
# ---------------------------------------------------------------------------

def extract_criteria_features(criteria_list: list[str]) -> dict[str, Any]:
    """Extract structured features from a criteria expression list.

    Returns dict with:
        hcp_min, hcp_max, tp_min, tp_max,
        suit_lengths: {suit: {"min": int|None, "max": int|None}},
        named_criteria: [str, ...],
        has_hcp, has_tp, has_sl, has_named,
        raw: [str, ...]
    """
    features: dict[str, Any] = {
        "hcp_min": None,
        "hcp_max": None,
        "tp_min": None,
        "tp_max": None,
        "suit_lengths": {},
        "named_criteria": [],
        "has_hcp": False,
        "has_tp": False,
        "has_sl": False,
        "has_named": False,
        "raw": list(criteria_list) if criteria_list else [],
    }
    if not criteria_list:
        return features

    for crit in criteria_list:
        if not isinstance(crit, str):
            continue
        crit_s = crit.strip()
        m = _CRITERION_PATTERN.match(crit_s)
        if m:
            var, op, val_s = m.group(1), m.group(2), int(m.group(3))
            if var == "HCP":
                features["has_hcp"] = True
                if op in (">=", ">"):
                    lo = val_s if op == ">=" else val_s + 1
                    if features["hcp_min"] is None or lo > features["hcp_min"]:
                        features["hcp_min"] = lo
                elif op in ("<=", "<"):
                    hi = val_s if op == "<=" else val_s - 1
                    if features["hcp_max"] is None or hi < features["hcp_max"]:
                        features["hcp_max"] = hi
                elif op == "==":
                    features["hcp_min"] = val_s
                    features["hcp_max"] = val_s
            elif var == "Total_Points":
                features["has_tp"] = True
                if op in (">=", ">"):
                    lo = val_s if op == ">=" else val_s + 1
                    if features["tp_min"] is None or lo > features["tp_min"]:
                        features["tp_min"] = lo
                elif op in ("<=", "<"):
                    hi = val_s if op == "<=" else val_s - 1
                    if features["tp_max"] is None or hi < features["tp_max"]:
                        features["tp_max"] = hi
                elif op == "==":
                    features["tp_min"] = val_s
                    features["tp_max"] = val_s
            elif var.startswith("SL_"):
                suit = var  # e.g. "SL_S"
                features["has_sl"] = True
                if suit not in features["suit_lengths"]:
                    features["suit_lengths"][suit] = {"min": None, "max": None}
                sl = features["suit_lengths"][suit]
                if op in (">=", ">"):
                    lo = val_s if op == ">=" else val_s + 1
                    if sl["min"] is None or lo > sl["min"]:
                        sl["min"] = lo
                elif op in ("<=", "<"):
                    hi = val_s if op == "<=" else val_s - 1
                    if sl["max"] is None or hi < sl["max"]:
                        sl["max"] = hi
                elif op == "==":
                    sl["min"] = val_s
                    sl["max"] = val_s
        else:
            # Named or complex criterion
            features["named_criteria"].append(crit_s)
            features["has_named"] = True

    return features


# ---------------------------------------------------------------------------
# Deficiency detection
# ---------------------------------------------------------------------------

def detect_deficiencies(
    bt_index: int,
    auction: str,
    candidate_bid: str,
    seat: int,
    is_completed: bool,
    has_children: bool,
    criteria_by_seat: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Detect criteria deficiencies for a single BT node.

    Returns a list of deficiency dicts, each with:
        deficiency, severity, detail, bt_index, auction, bid, seat
    """
    deficiencies: list[dict[str, Any]] = []
    bid_upper = str(candidate_bid or "").strip().upper()
    is_pass = bid_upper in _PASS_BIDS

    def _add(deficiency: str, severity: str, detail: str) -> None:
        deficiencies.append({
            "deficiency": deficiency,
            "severity": severity,
            "detail": detail,
            "bt_index": bt_index,
            "auction": auction,
            "bid": bid_upper,
            "seat": seat,
        })

    # Dead end check
    if not is_completed and not has_children:
        _add("DEAD_END", "high", "No children and not a completed auction")

    # Skip further criteria checks for Pass bids (they're always permissive)
    if is_pass:
        return deficiencies

    # Get acting seat criteria
    acting_col = f"Agg_Expr_Seat_{seat}"
    acting_criteria = criteria_by_seat.get(acting_col, [])

    # Check if other seats have criteria but acting seat doesn't
    other_seats_have = False
    for s in range(1, 5):
        if s == seat:
            continue
        other_col = f"Agg_Expr_Seat_{s}"
        other_crits = criteria_by_seat.get(other_col, [])
        if other_crits:
            other_seats_have = True
            break

    # 1. Empty criteria
    if not acting_criteria:
        if other_seats_have:
            _add("NO_CRITERIA_FOR_SEAT", "high",
                 f"Seat {seat} has no criteria but other seats do")
        else:
            _add("EMPTY_CRITERIA", "high", "No criteria for any seat")
        return deficiencies  # No further checks possible

    # Extract features
    feat = extract_criteria_features(acting_criteria)

    # 2. No strength indicator (missing both HCP and Total_Points)
    if not feat["has_hcp"] and not feat["has_tp"]:
        _add("NO_STRENGTH", "medium",
             f"No HCP or Total_Points constraint; has: {feat['raw']}")

    # 3. Unbounded HCP
    if feat["has_hcp"]:
        if feat["hcp_min"] is not None and feat["hcp_max"] is None:
            _add("UNBOUNDED_HCP", "low",
                 f"HCP >= {feat['hcp_min']} but no upper bound")
        elif feat["hcp_max"] is not None and feat["hcp_min"] is None:
            _add("UNBOUNDED_HCP", "low",
                 f"HCP <= {feat['hcp_max']} but no lower bound")

    # 4. Unbounded Total_Points
    if feat["has_tp"]:
        if feat["tp_min"] is not None and feat["tp_max"] is None:
            _add("UNBOUNDED_TOTAL_POINTS", "low",
                 f"Total_Points >= {feat['tp_min']} but no upper bound")
        elif feat["tp_max"] is not None and feat["tp_min"] is None:
            _add("UNBOUNDED_TOTAL_POINTS", "low",
                 f"Total_Points <= {feat['tp_max']} but no lower bound")

    # 5. Wide HCP range
    if feat["hcp_min"] is not None and feat["hcp_max"] is not None:
        hcp_range = feat["hcp_max"] - feat["hcp_min"]
        if hcp_range >= _WIDE_HCP_THRESHOLD:
            _add("WIDE_HCP_RANGE", "low",
                 f"HCP range {feat['hcp_min']}-{feat['hcp_max']} "
                 f"spans {hcp_range} points")

    # 6. Wide Total_Points range
    if feat["tp_min"] is not None and feat["tp_max"] is not None:
        tp_range = feat["tp_max"] - feat["tp_min"]
        if tp_range >= _WIDE_HCP_THRESHOLD:
            _add("WIDE_TP_RANGE", "low",
                 f"Total_Points range {feat['tp_min']}-{feat['tp_max']} "
                 f"spans {tp_range} points")

    # 5. No suit-length constraint for a strain bid at level 1+
    if _STRAIN_BIDS.match(bid_upper) and not feat["has_sl"]:
        # Extract strain from bid
        strain = bid_upper[-1] if bid_upper[-1] in "CDHS" else None
        if strain:
            # NT bids don't necessarily need suit-length
            _add("NO_SUIT_LENGTH", "low",
                 f"Strain bid {bid_upper} has no suit-length constraint")

    return deficiencies


# ---------------------------------------------------------------------------
# Batch loading of Agg_Expr via DuckDB
# ---------------------------------------------------------------------------

def load_agg_expr_batch(
    bt_indices: list[int],
    bt_parquet_file: Path,
) -> dict[int, dict[str, list[str]]]:
    """Load Agg_Expr_Seat_1..4 for a batch of bt_indices using DuckDB."""
    if not bt_indices:
        return {}

    try:
        import duckdb
        conn = duckdb.connect(":memory:")
        in_list = ", ".join(str(x) for x in bt_indices)
        file_path = str(bt_parquet_file).replace("\\", "/")
        query = f"""
            SELECT bt_index, Agg_Expr_Seat_1, Agg_Expr_Seat_2,
                   Agg_Expr_Seat_3, Agg_Expr_Seat_4
            FROM read_parquet('{file_path}')
            WHERE bt_index IN ({in_list})
        """
        try:
            result_rel = conn.execute(query)
            rows = result_rel.fetchall()
            col_names = [desc[0] for desc in result_rel.description]
        finally:
            conn.close()

        result: dict[int, dict[str, list[str]]] = {}
        for row in rows:
            row_dict = dict(zip(col_names, row))
            bt_idx = row_dict["bt_index"]
            result[bt_idx] = {}
            for s in range(1, 5):
                col = f"Agg_Expr_Seat_{s}"
                val = row_dict.get(col)
                result[bt_idx][col] = list(val) if val else []
        return result

    except Exception as e:
        print(f"  [WARN] DuckDB failed ({e}), falling back to Polars scan...")
        try:
            df = (
                pl.scan_parquet(bt_parquet_file)
                .filter(pl.col("bt_index").is_in(bt_indices))
                .select(["bt_index", "Agg_Expr_Seat_1", "Agg_Expr_Seat_2",
                          "Agg_Expr_Seat_3", "Agg_Expr_Seat_4"])
                .collect()
            )
            result = {}
            for row in df.to_dicts():
                bt_idx = row["bt_index"]
                result[bt_idx] = {
                    k: (v if v is not None else [])
                    for k, v in row.items() if k != "bt_index"
                }
            return result
        except Exception as e2:
            print(f"  [FATAL] Polars fallback also failed: {e2}")
            return {}


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(
    bt_file: Path = BT_FILE,
    output_dir: Path = OUTPUT_DIR,
    max_nodes: int | None = None,
    batch_size: int = 50_000,
    exclude_dead_ends: bool = True,
) -> Path:
    """Run the BT criteria audit.

    Args:
        bt_file: Path to compiled BT parquet.
        output_dir: Directory for output reports.
        max_nodes: If set, audit only the first N nodes (for testing).
        batch_size: Number of bt_indices to load Agg_Expr for per batch.
        exclude_dead_ends: If True (default), exclude nodes that are dead ends
            (no children and not a completed auction). These nodes are
            unreachable in practice and add noise. If False, include all nodes
            (dead ends will be flagged as DEAD_END deficiencies).

    Returns:
        Path to the generated JSON report.
    """
    t0 = time.perf_counter()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    scope = "exclude dead ends" if exclude_dead_ends else "all nodes (including dead ends)"
    print(f"[audit] BT Criteria Audit starting at {datetime.now().isoformat()}")
    print(f"[audit] BT file: {bt_file}")
    print(f"[audit] Scope: {scope}")
    print(f"[audit] Batch size: {batch_size:,}")
    if max_nodes:
        print(f"[audit] Max nodes: {max_nodes:,}")

    # ------------------------------------------------------------------
    # 1. Load lightweight BT columns
    # ------------------------------------------------------------------
    print("[audit] Loading BT lightweight columns...")
    t1 = time.perf_counter()

    cols_needed = [
        "bt_index", "Auction", "candidate_bid", "seat",
        "is_completed_auction", "next_bid_indices",
    ]
    bt_df = pl.scan_parquet(bt_file).select(cols_needed)
    if max_nodes:
        bt_df = bt_df.head(max_nodes)
    bt_df = bt_df.collect()

    # Compute has_children column
    bt_df = bt_df.with_columns(
        pl.col("next_bid_indices").list.len().alias("n_children")
    )

    # Exclude dead-end nodes if requested
    if exclude_dead_ends:
        before = len(bt_df)
        bt_df = bt_df.filter(
            (pl.col("is_completed_auction") == True)  # noqa: E712
            | (pl.col("n_children") > 0)
        )
        excluded = before - len(bt_df)
        print(f"[audit] Excluded {excluded:,} dead-end nodes ({excluded / max(1, before) * 100:.1f}%)")

    total_nodes = len(bt_df)
    print(f"[audit] Loaded {total_nodes:,} nodes in {time.perf_counter() - t1:.1f}s")

    # ------------------------------------------------------------------
    # 2. Scan in batches
    # ------------------------------------------------------------------
    all_deficiencies: list[dict[str, Any]] = []
    deficiency_counts: Counter = Counter()
    severity_counts: Counter = Counter()
    nodes_scanned = 0
    nodes_with_issues = 0

    # Convert to dicts for row-level access
    bt_indices_all = bt_df["bt_index"].to_list()
    n_batches = (total_nodes + batch_size - 1) // batch_size

    print(f"[audit] Scanning {total_nodes:,} nodes in {n_batches:,} batches...")

    for batch_i in range(n_batches):
        batch_start = batch_i * batch_size
        batch_end = min(batch_start + batch_size, total_nodes)
        batch_indices = bt_indices_all[batch_start:batch_end]

        t_batch = time.perf_counter()

        # Load Agg_Expr for this batch
        agg_data = load_agg_expr_batch(batch_indices, bt_file)

        # Slice the DataFrame for this batch
        batch_df = bt_df.slice(batch_start, batch_end - batch_start)
        batch_rows = batch_df.to_dicts()

        batch_issues = 0
        for row in batch_rows:
            bt_idx = row["bt_index"]
            auction = str(row.get("Auction", "") or "")
            bid = str(row.get("candidate_bid", "") or "")
            seat = int(row.get("seat", 1) or 1)
            is_completed = bool(row.get("is_completed_auction", False))
            has_children = int(row.get("n_children", 0) or 0) > 0

            criteria_by_seat = agg_data.get(bt_idx, {})

            issues = detect_deficiencies(
                bt_index=bt_idx,
                auction=auction,
                candidate_bid=bid,
                seat=seat,
                is_completed=is_completed,
                has_children=has_children,
                criteria_by_seat=criteria_by_seat,
            )

            if issues:
                all_deficiencies.extend(issues)
                batch_issues += 1
                nodes_with_issues += 1
                for iss in issues:
                    deficiency_counts[iss["deficiency"]] += 1
                    severity_counts[iss["severity"]] += 1

            nodes_scanned += 1

        batch_ms = (time.perf_counter() - t_batch) * 1000
        elapsed = time.perf_counter() - t0
        pct = (batch_end / total_nodes) * 100
        rate = nodes_scanned / elapsed if elapsed > 0 else 0
        eta_s = (total_nodes - batch_end) / rate if rate > 0 else 0

        print(
            f"  Batch {batch_i + 1}/{n_batches}: "
            f"{batch_end:,}/{total_nodes:,} ({pct:.1f}%) | "
            f"{batch_issues:,} issues | "
            f"{batch_ms:.0f}ms | "
            f"{rate:,.0f} nodes/s | "
            f"ETA {eta_s / 60:.1f}min"
        )

        # Free memory
        del agg_data, batch_df, batch_rows
        gc.collect()

    elapsed_total = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 3. Build summary
    # ------------------------------------------------------------------
    summary = {
        "audit_timestamp": ts,
        "bt_file": str(bt_file),
        "scope": "exclude_dead_ends" if exclude_dead_ends else "all_nodes",
        "total_nodes_scanned": nodes_scanned,
        "nodes_with_issues": nodes_with_issues,
        "pct_with_issues": round(nodes_with_issues / max(1, nodes_scanned) * 100, 2),
        "total_deficiencies": len(all_deficiencies),
        "deficiency_counts": dict(deficiency_counts.most_common()),
        "severity_counts": dict(severity_counts.most_common()),
        "elapsed_seconds": round(elapsed_total, 1),
    }

    # Top auctions with most issues (for targeted fixing)
    auction_issue_counts: Counter = Counter()
    for d in all_deficiencies:
        # Use auction prefix (first 3 bids) for grouping
        auc = str(d.get("auction", ""))
        parts = auc.split("-")
        prefix = "-".join(parts[:3]) if len(parts) > 3 else auc
        auction_issue_counts[prefix] += 1

    summary["top_auction_prefixes_with_issues"] = dict(
        auction_issue_counts.most_common(50)
    )

    # Bid-level breakdown
    bid_issue_counts: Counter = Counter()
    for d in all_deficiencies:
        bid_issue_counts[d.get("bid", "?")] += 1
    summary["top_bids_with_issues"] = dict(bid_issue_counts.most_common(30))

    # Deficiency × severity matrix
    def_sev_matrix: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for d in all_deficiencies:
        def_sev_matrix[d["deficiency"]][d["severity"]] += 1
    summary["deficiency_severity_matrix"] = {
        k: dict(v) for k, v in def_sev_matrix.items()
    }

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------
    json_path = output_dir / f"bt_criteria_audit_{ts}.json"
    txt_path = output_dir / f"bt_criteria_audit_{ts}_summary.txt"

    # Sort deficiencies by severity (high, medium, low) to prioritize important ones
    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_deficiencies_sorted = sorted(
        all_deficiencies,
        key=lambda d: (severity_order.get(d["severity"], 99), d["bt_index"])
    )

    # JSON report (full deficiency list + summary)
    # Include all deficiencies regardless of severity (removed cap to include low-severity items)
    report = {
        "summary": summary,
        "deficiencies": all_deficiencies_sorted,  # Include all deficiencies, sorted by severity
        "deficiencies_truncated": False,  # No longer truncating
        "deficiencies_total": len(all_deficiencies),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[audit] JSON report: {json_path} ({json_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Text summary (human-readable)
    lines = [
        "=" * 72,
        "BT CRITERIA AUDIT SUMMARY",
        "=" * 72,
        f"Timestamp:        {ts}",
        f"BT file:          {bt_file}",
        f"Nodes scanned:    {nodes_scanned:,}",
        f"Nodes with issues: {nodes_with_issues:,} ({summary['pct_with_issues']:.1f}%)",
        f"Total deficiencies: {len(all_deficiencies):,}",
        f"Elapsed:          {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)",
        "",
        "-" * 72,
        "DEFICIENCY BREAKDOWN",
        "-" * 72,
    ]
    for def_name, count in deficiency_counts.most_common():
        pct = count / max(1, len(all_deficiencies)) * 100
        lines.append(f"  {def_name:<30s} {count:>10,}  ({pct:5.1f}%)")

    lines.extend([
        "",
        "-" * 72,
        "SEVERITY BREAKDOWN",
        "-" * 72,
    ])
    for sev, count in severity_counts.most_common():
        pct = count / max(1, len(all_deficiencies)) * 100
        lines.append(f"  {sev:<30s} {count:>10,}  ({pct:5.1f}%)")

    lines.extend([
        "",
        "-" * 72,
        "TOP 30 BIDS WITH ISSUES",
        "-" * 72,
    ])
    for bid, count in bid_issue_counts.most_common(30):
        lines.append(f"  {bid:<10s} {count:>10,}")

    lines.extend([
        "",
        "-" * 72,
        "TOP 50 AUCTION PREFIXES WITH ISSUES",
        "-" * 72,
    ])
    for prefix, count in auction_issue_counts.most_common(50):
        lines.append(f"  {prefix:<30s} {count:>10,}")

    lines.extend([
        "",
        "-" * 72,
        "SAMPLE DEFICIENCIES (first 20, sorted by severity)",
        "-" * 72,
    ])
    for d in all_deficiencies_sorted[:20]:
        lines.append(
            f"  [{d['severity'].upper():>6s}] {d['deficiency']:<25s} "
            f"bt={d['bt_index']:>10,}  {d['auction']:<30s}  "
            f"bid={d['bid']:<6s} seat={d['seat']}  "
            f"{d['detail']}"
        )

    lines.append("")
    lines.append("=" * 72)

    txt_content = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"[audit] Text summary: {txt_path}")
    print()
    print(txt_content)

    return json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BT Criteria Audit — scan for insufficiently specified criteria"
    )
    parser.add_argument(
        "--bt-file", type=Path, default=BT_FILE,
        help=f"Path to compiled BT parquet (default: {BT_FILE})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory for reports (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=None,
        help="Audit only the first N nodes (for testing)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000,
        help="Number of bt_indices per Agg_Expr loading batch (default: 50000)",
    )
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--exclude-dead-ends", action="store_true", default=True,
        help="Exclude dead-end nodes (no children, not completed) from audit (default)",
    )
    scope_group.add_argument(
        "--include-dead-ends", action="store_true",
        help="Include dead-end nodes (they will be flagged as DEAD_END deficiencies)",
    )

    args = parser.parse_args()

    if not args.bt_file.exists():
        print(f"[FATAL] BT file not found: {args.bt_file}")
        sys.exit(1)

    run_audit(
        bt_file=args.bt_file,
        output_dir=args.output_dir,
        max_nodes=args.max_nodes,
        batch_size=args.batch_size,
        exclude_dead_ends=not args.include_dead_ends,
    )


if __name__ == "__main__":
    main()
