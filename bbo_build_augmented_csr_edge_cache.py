"""
Build augmented CSR edge metadata cache.

Start/End datetime and progress are printed during execution.

Estimated resources:
- Typical elapsed: ~1-10 minutes
- Peak RAM: ~8-48 GB

Latest observed full run (2026-02-14):
- Elapsed: 71.4s
- Output: 461,681,275 rows x 5 cols
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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
        "path": path.name,  # filename only – portable across machines
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
        "sha256": _sha256_file(path),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bid_to_code_expr(col_name: str) -> pl.Expr:
    mapping: Dict[str, int] = {"P": 1, "X": 2, "XX": 3}
    code = 4
    for level in range(1, 8):
        for strain in ["C", "D", "H", "S", "N"]:
            mapping[f"{level}{strain}"] = code
            code += 1
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.to_uppercase()
        .str.replace_all(r"NT$", "N")
        .replace_strict(mapping, default=0)
        .cast(pl.UInt8)
    )


def build_augmented_csr_edge_cache(
    *,
    bt_file: Path,
    out_file: Path,
    manifest_file: Path,
) -> None:
    started_at = _utc_now()
    t0 = time.perf_counter()
    print(f"[bbo_build_augmented_csr_edge_cache] start_utc={started_at}")

    source_sig = _file_sig(bt_file)
    schema = pl.scan_parquet(bt_file).collect_schema().names()
    required = ["bt_index", "next_bid_indices", "candidate_bid"]
    for c in required:
        if c not in schema:
            raise RuntimeError(f"Required BT column missing: {c}")

    print("[step] reading parent-child edge list...")
    parent_edges = (
        pl.scan_parquet(bt_file)
        .select(
            [
                pl.col("bt_index").cast(pl.UInt32, strict=False).alias("parent_bt_index"),
                pl.col("next_bid_indices"),
            ]
        )
        .filter(pl.col("next_bid_indices").list.len() > 0)
        .explode("next_bid_indices")
        .with_columns(pl.col("next_bid_indices").cast(pl.UInt32, strict=False).alias("child_bt_index"))
        .select(["parent_bt_index", "child_bt_index"])
    )

    print("[step] reading child metadata...")
    child_cols = ["bt_index", "candidate_bid"]
    if "matching_deal_count" in schema:
        child_cols.append("matching_deal_count")
    child_meta = (
        pl.scan_parquet(bt_file)
        .select([pl.col(c) for c in child_cols])
        .with_columns(
            [
                pl.col("bt_index").cast(pl.UInt32, strict=False).alias("child_bt_index"),
                _bid_to_code_expr("candidate_bid").alias("bid_code"),
                pl.col("matching_deal_count").cast(pl.UInt32, strict=False).fill_null(0).alias("matching_deal_count")
                if "matching_deal_count" in child_cols
                else pl.lit(0, dtype=pl.UInt32).alias("matching_deal_count"),
            ]
        )
        .select(["child_bt_index", "candidate_bid", "bid_code", "matching_deal_count"])
    )

    print("[step] joining edges with metadata...")
    df = (
        parent_edges.join(child_meta, on="child_bt_index", how="left")
        .with_columns(
            [
                pl.col("candidate_bid").cast(pl.Utf8, strict=False).fill_null(""),
                pl.col("bid_code").cast(pl.UInt8, strict=False).fill_null(0),
                pl.col("matching_deal_count").cast(pl.UInt32, strict=False).fill_null(0),
            ]
        )
        .select(
            [
                "parent_bt_index",
                "child_bt_index",
                "candidate_bid",
                "bid_code",
                "matching_deal_count",
            ]
        )
        .collect()
    )

    print("[step] writing parquet...")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_file)

    elapsed_s = time.perf_counter() - t0
    ended_at = _utc_now()
    print(f"[done] rows={df.height:,} cols={df.width} elapsed={elapsed_s:.1f}s")

    manifest = {
        "artifact": out_file.name,
        "artifact_type": "bbo_augmented_csr_edge_cache",
        "builder": "bbo_build_augmented_csr_edge_cache.py",
        "version": 1,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": round(elapsed_s, 3),
        "sources": [source_sig],
        "row_count": int(df.height),
        "column_count": int(df.width),
    }
    _write_json(manifest_file, manifest)


def main() -> None:
    p = argparse.ArgumentParser(description="Build augmented CSR edge cache parquet.")
    p.add_argument("--bt-file", type=Path, default=Path("data/bbo_bt_compiled.parquet"))
    p.add_argument("--out-file", type=Path, default=Path("data/bbo_augmented_csr_edge_cache.parquet"))
    p.add_argument("--manifest-file", type=Path, default=Path("data/bbo_augmented_csr_edge_cache_manifest.json"))
    args = p.parse_args()
    build_augmented_csr_edge_cache(
        bt_file=args.bt_file,
        out_file=args.out_file,
        manifest_file=args.manifest_file,
    )


if __name__ == "__main__":
    main()
