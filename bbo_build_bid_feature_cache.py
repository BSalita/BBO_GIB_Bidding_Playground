"""
Build offline bid-feature cache for AI model acceleration.

Start/End datetime and progress are printed during execution.

Estimated resources (depends on file sizes):
- Typical elapsed: ~1-10 minutes
- Peak RAM: ~8-40 GB

Latest observed full run (2026-02-14):
- Elapsed: 70.9s
- Output: 461,681,310 rows x 11 cols

If expected runtime exceeds one hour, use --checkpoint-dir + --resume to avoid
redoing completed work after interruptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import polars as pl


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _file_sig(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
        "sha256": _sha256_file(path),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bid_to_code_expr(col_name: str = "candidate_bid") -> pl.Expr:
    bid_to_code: Dict[str, int] = {"P": 1, "X": 2, "XX": 3}
    code = 4
    for level in range(1, 8):
        for strain in ["C", "D", "H", "S", "N"]:
            bid_to_code[f"{level}{strain}"] = code
            code += 1

    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.to_uppercase()
        .str.replace_all(r"NT$", "N")
        .replace_strict(bid_to_code, default=0)
        .cast(pl.UInt8)
    )


def _seat_pick_expr(prefix: str, vul: str, out_name: str, default_dtype: Any = pl.Float32) -> pl.Expr:
    """Pick seat-specific column (S1..S4) based on `seat` integer."""
    seat_cols = [f"{prefix}_S1_{vul}", f"{prefix}_S2_{vul}", f"{prefix}_S3_{vul}", f"{prefix}_S4_{vul}"]
    expr = (
        pl.when(pl.col("seat") == 1).then(pl.col(seat_cols[0]))
        .when(pl.col("seat") == 2).then(pl.col(seat_cols[1]))
        .when(pl.col("seat") == 3).then(pl.col(seat_cols[2]))
        .when(pl.col("seat") == 4).then(pl.col(seat_cols[3]))
        .otherwise(None)
        .cast(default_dtype, strict=False)
        .alias(out_name)
    )
    return expr


def build_bid_feature_cache(
    *,
    bt_file: Path,
    stats_file: Path,
    out_file: Path,
    manifest_file: Path,
    checkpoint_dir: Path,
    resume: bool,
) -> None:
    started_at = _utc_now()
    t0 = time.perf_counter()
    print(f"[bbo_build_bid_feature_cache] start_utc={started_at}")

    sources = [_file_sig(bt_file), _file_sig(stats_file)]
    ckpt_meta_file = checkpoint_dir / "bid_feature_cache_checkpoint.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume and out_file.exists() and manifest_file.exists() and ckpt_meta_file.exists():
        try:
            old = _load_json(ckpt_meta_file)
            if old.get("sources") == sources and bool(old.get("completed")):
                print("[resume] matching completed checkpoint found; nothing to do.")
                return
        except Exception:
            pass

    print("[step] scanning schemas...")
    bt_schema = pl.scan_parquet(bt_file).collect_schema().names()
    stats_schema = pl.scan_parquet(stats_file).collect_schema().names()

    required_bt = ["bt_index", "candidate_bid", "seat"]
    for c in required_bt:
        if c not in bt_schema:
            raise RuntimeError(f"Required BT column missing: {c}")

    metric_cols = [c for c in stats_schema if re.match(r"^(Avg_Par|Avg_EV|Count)_S[1-4]_(NV|V)$", c)]
    if not metric_cols:
        raise RuntimeError("Stats file missing required EV/Par/Count columns.")

    bt_cols = ["bt_index", "candidate_bid", "seat"]
    if "matching_deal_count" in bt_schema:
        bt_cols.append("matching_deal_count")

    print("[step] building lazy join plan...")
    bt_lf = (
        pl.scan_parquet(bt_file)
        .select([pl.col(c) for c in bt_cols])
        .with_columns([
            pl.col("bt_index").cast(pl.UInt32, strict=False),
            pl.col("seat").cast(pl.UInt8, strict=False),
        ])
    )
    stats_lf = (
        pl.scan_parquet(stats_file)
        .select([pl.col("bt_index")] + [pl.col(c) for c in metric_cols if c != "bt_index"])
        .with_columns(pl.col("bt_index").cast(pl.UInt32, strict=False))
    )
    lf = bt_lf.join(stats_lf, on="bt_index", how="left")

    print("[step] deriving cache columns...")
    lf = lf.with_columns(
        [
            _bid_to_code_expr("candidate_bid").alias("bid_code"),
            _seat_pick_expr("Avg_Par", "NV", "mean_par_nv", pl.Float32),
            _seat_pick_expr("Avg_Par", "V", "mean_par_v", pl.Float32),
            _seat_pick_expr("Avg_EV", "NV", "mean_ev_nv", pl.Float32),
            _seat_pick_expr("Avg_EV", "V", "mean_ev_v", pl.Float32),
            _seat_pick_expr("Count", "NV", "count_nv", pl.UInt32),
            _seat_pick_expr("Count", "V", "count_v", pl.UInt32),
            pl.col("matching_deal_count").cast(pl.UInt32, strict=False).fill_null(0).alias("matching_deal_count")
            if "matching_deal_count" in bt_cols
            else pl.lit(0, dtype=pl.UInt32).alias("matching_deal_count"),
        ]
    ).select(
        [
            "bt_index",
            "candidate_bid",
            "bid_code",
            "seat",
            "matching_deal_count",
            "mean_par_nv",
            "mean_par_v",
            "mean_ev_nv",
            "mean_ev_v",
            "count_nv",
            "count_v",
        ]
    )

    print("[step] collecting and writing parquet...")
    df = lf.collect()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_file)

    ended_at = _utc_now()
    elapsed_s = time.perf_counter() - t0
    print(f"[done] rows={df.height:,} cols={df.width} elapsed={elapsed_s:.1f}s")

    manifest = {
        "artifact": str(out_file.resolve()),
        "artifact_type": "bbo_bid_feature_cache",
        "builder": "bbo_build_bid_feature_cache.py",
        "version": 1,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": round(elapsed_s, 3),
        "sources": sources,
        "row_count": int(df.height),
        "column_count": int(df.width),
    }
    _write_json(manifest_file, manifest)
    _write_json(
        ckpt_meta_file,
        {
            "completed": True,
            "sources": sources,
            "artifact": str(out_file.resolve()),
            "manifest": str(manifest_file.resolve()),
            "ended_at_utc": ended_at,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build bbo bid feature cache parquet.")
    p.add_argument("--bt-file", type=Path, default=Path("E:/bridge/data/bbo/bidding/bbo_bt_compiled.parquet"))
    p.add_argument("--stats-file", type=Path, default=Path("E:/bridge/data/bbo/bidding/bbo_bt_criteria_seat1_df.parquet"))
    p.add_argument("--out-file", type=Path, default=Path("E:/bridge/data/bbo/bidding/bbo_bid_feature_cache.parquet"))
    p.add_argument("--manifest-file", type=Path, default=Path("E:/bridge/data/bbo/bidding/bbo_bid_feature_cache_manifest.json"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("E:/bridge/data/bbo/bidding/cache_build_checkpoints/bbo_bid_feature_cache"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    build_bid_feature_cache(
        bt_file=args.bt_file,
        stats_file=args.stats_file,
        out_file=args.out_file,
        manifest_file=args.manifest_file,
        checkpoint_dir=args.checkpoint_dir,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
