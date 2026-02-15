"""
Build optional terminal-auction cheater cache (bounded scope).

This cache intentionally avoids full (deal_idx x prefix) expansion. It stores
terminal BT rows only for fast lookup and optional runtime memo seeding.

Estimated resources:
- Typical elapsed: ~20-120 seconds
- Peak RAM: ~2-12 GB

Latest observed full run (2026-02-14):
- Elapsed: 28.9s
- Output: 974,989 rows x 5 cols
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


def build_cheater_terminal_cache(*, bt_file: Path, out_file: Path, manifest_file: Path) -> None:
    started_at = _utc_now()
    t0 = time.perf_counter()
    print(f"[bbo_build_cheater_terminal_cache] start_utc={started_at}")

    source_sig = _file_sig(bt_file)
    schema = pl.scan_parquet(bt_file).collect_schema().names()
    required = ["bt_index", "Auction", "is_completed_auction"]
    for c in required:
        if c not in schema:
            raise RuntimeError(f"Required BT column missing: {c}")

    cols = ["bt_index", "Auction", "is_completed_auction"]
    if "matching_deal_count" in schema:
        cols.append("matching_deal_count")
    if "candidate_bid" in schema:
        cols.append("candidate_bid")

    print("[step] filtering terminal rows...")
    df = (
        pl.scan_parquet(bt_file)
        .select([pl.col(c) for c in cols])
        .filter(pl.col("is_completed_auction") == True)
        .with_columns(
            [
                pl.col("bt_index").cast(pl.UInt32, strict=False),
                pl.col("matching_deal_count").cast(pl.UInt32, strict=False).fill_null(0).alias("matching_deal_count")
                if "matching_deal_count" in cols
                else pl.lit(0, dtype=pl.UInt32).alias("matching_deal_count"),
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
        "artifact_type": "bbo_cheater_terminal_cache",
        "builder": "bbo_build_cheater_terminal_cache.py",
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
    p = argparse.ArgumentParser(description="Build optional terminal-auction cheater cache parquet.")
    p.add_argument("--bt-file", type=Path, default=Path("data/bbo_bt_compiled.parquet"))
    p.add_argument("--out-file", type=Path, default=Path("data/bbo_cheater_terminal_cache.parquet"))
    p.add_argument("--manifest-file", type=Path, default=Path("data/bbo_cheater_terminal_cache_manifest.json"))
    args = p.parse_args()
    build_cheater_terminal_cache(
        bt_file=args.bt_file,
        out_file=args.out_file,
        manifest_file=args.manifest_file,
    )


if __name__ == "__main__":
    main()
