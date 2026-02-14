"""
Offline trick-quality and hand-profile artifact builder.

This script implements proposal phases T1/T2/T3/T5:
- T1: load and validate inputs (fail-fast on missing DD trick columns)
- T2: build bt_index x seat x strain DD distributions + quantiles
- T3: build bt_index x seat hand-profile histograms + quantiles
- T5: derive level-based trick quality rows (book, exp_extra, p_make, p_plus1)

This implementation supports CPU and GPU backends and is designed to be
deterministic, restartable, and fail-fast.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


DEALER_DIRS = ("N", "E", "S", "W")
STRAINS = ("C", "D", "H", "S", "N")

DD_REQUIRED_COLS = [f"DD_{d}_{s}" for d in DEALER_DIRS for s in STRAINS]
HAND_REQUIRED_COLS = (
    [f"HCP_{d}" for d in DEALER_DIRS]
    + [f"Total_Points_{d}" for d in DEALER_DIRS]
    + [f"SL_{d}_{s}" for d in DEALER_DIRS for s in ("S", "H", "D", "C")]
)
CHECKPOINT_VERSION = "tq_v1"


def _get_torch_cuda():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise RuntimeError("GPU backend requires PyTorch to be installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU backend requested but CUDA is not available")
    return torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(str(tmp), compression="zstd")
    tmp.replace(path)


def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"


def _input_signature(deals_path: Path, verified_map_path: Path, bt_metadata_path: Path) -> str:
    raw = "||".join([_file_sig(deals_path), _file_sig(verified_map_path), _file_sig(bt_metadata_path)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: Path,
        resume: bool,
        deals_path: Path,
        verified_map_path: Path,
        bt_metadata_path: Path,
    ) -> None:
        self.dir = checkpoint_dir
        self.meta_file = self.dir / "checkpoint_meta.json"
        sig = _input_signature(deals_path, verified_map_path, bt_metadata_path)
        self.meta = {"version": CHECKPOINT_VERSION, "input_signature": sig}

        if resume:
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.meta_file.exists():
                prev = json.loads(self.meta_file.read_text(encoding="utf-8"))
                if prev.get("version") != CHECKPOINT_VERSION or prev.get("input_signature") != sig:
                    raise ValueError(
                        "Checkpoint metadata mismatch (version or input files changed). "
                        "Delete checkpoint dir or rerun without --resume."
                    )
            else:
                self.meta_file.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        else:
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            self.meta_file.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.parquet"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def read(self, key: str) -> pl.DataFrame:
        return pl.read_parquet(str(self.path(key)))

    def write(self, key: str, df: pl.DataFrame) -> None:
        _atomic_write_parquet(df, self.path(key))


def _rotate_dirs_from_dealer(dealer: str) -> tuple[str, str, str, str]:
    idx = DEALER_DIRS.index(dealer)
    return (
        DEALER_DIRS[idx],
        DEALER_DIRS[(idx + 1) % 4],
        DEALER_DIRS[(idx + 2) % 4],
        DEALER_DIRS[(idx + 3) % 4],
    )


def _ensure_required_columns(schema_names: list[str], required: list[str], label: str) -> None:
    missing = [c for c in required if c not in schema_names]
    if missing:
        raise ValueError(
            f"Missing required {label} columns ({len(missing)}): {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )


def _compute_hist_quantiles(df: pl.DataFrame, hist_cols: list[str], out_prefix: str) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(
            [
                pl.lit(None, dtype=pl.Float32).alias(f"{out_prefix}p10"),
                pl.lit(None, dtype=pl.Float32).alias(f"{out_prefix}p25"),
                pl.lit(None, dtype=pl.Float32).alias(f"{out_prefix}p50"),
                pl.lit(None, dtype=pl.Float32).alias(f"{out_prefix}p75"),
                pl.lit(None, dtype=pl.Float32).alias(f"{out_prefix}p90"),
            ]
        )

    n_arr = df["n"].to_numpy().astype(np.int64, copy=False)
    hist_mat = np.column_stack([df[c].to_numpy() for c in hist_cols]).astype(np.int64, copy=False)
    cdf = np.cumsum(hist_mat, axis=1, dtype=np.int64)

    def _quantile_idx(pct: float) -> np.ndarray:
        out = np.full(df.height, np.nan, dtype=np.float32)
        valid = n_arr > 0
        if not np.any(valid):
            return out
        target = np.ceil((pct / 100.0) * n_arr[valid]).astype(np.int64)
        target[target < 1] = 1
        cdf_valid = cdf[valid]
        ge = cdf_valid >= target[:, None]
        idx = np.argmax(ge, axis=1).astype(np.int64)
        miss = cdf_valid[:, -1] < target
        idx[miss] = hist_mat.shape[1] - 1
        out[valid] = idx.astype(np.float32)
        return out

    p10 = _quantile_idx(10.0)
    p25 = _quantile_idx(25.0)
    p50 = _quantile_idx(50.0)
    p75 = _quantile_idx(75.0)
    p90 = _quantile_idx(90.0)
    return df.with_columns(
        [
            pl.Series(f"{out_prefix}p10", p10).cast(pl.Float32),
            pl.Series(f"{out_prefix}p25", p25).cast(pl.Float32),
            pl.Series(f"{out_prefix}p50", p50).cast(pl.Float32),
            pl.Series(f"{out_prefix}p75", p75).cast(pl.Float32),
            pl.Series(f"{out_prefix}p90", p90).cast(pl.Float32),
        ]
    )


def _build_joined(deals_path: Path, verified_map_path: Path) -> pl.DataFrame:
    t0 = time.perf_counter()
    print(f"[T1] start={_utc_now()}")
    print(f"[T1] deals={deals_path}")
    print(f"[T1] verified_map={verified_map_path}")

    if not deals_path.exists():
        raise FileNotFoundError(f"Deals parquet not found: {deals_path}")
    if not verified_map_path.exists():
        raise FileNotFoundError(f"Verified map parquet not found: {verified_map_path}")

    deals_schema = pl.scan_parquet(str(deals_path)).collect_schema().names()
    _ensure_required_columns(deals_schema, DD_REQUIRED_COLS, "DD trick")
    _ensure_required_columns(deals_schema, list(HAND_REQUIRED_COLS), "hand")
    _ensure_required_columns(deals_schema, ["Dealer", "Vul"], "context")

    deal_cols = ["Dealer", "Vul"] + DD_REQUIRED_COLS + list(HAND_REQUIRED_COLS)
    deals_df = pl.read_parquet(str(deals_path), columns=deal_cols).with_row_index("deal_idx")
    map_df = (
        pl.read_parquet(str(verified_map_path), columns=["deal_idx", "Matched_BT_Indices"])
        .explode("Matched_BT_Indices")
        .rename({"Matched_BT_Indices": "bt_index"})
        .drop_nulls(subset=["bt_index"])
    )
    joined = map_df.join(deals_df, on="deal_idx", how="inner")
    joined = joined.with_columns(
        [
            pl.col("bt_index").cast(pl.UInt32),
            pl.col("deal_idx").cast(pl.UInt32),
            pl.col("Dealer").cast(pl.Utf8),
            pl.col("Vul").cast(pl.Utf8),
        ]
    )
    print(
        f"[T1] done rows={joined.height:,}, cols={joined.width}, "
        f"elapsed={_elapsed_ms(t0)/1000.0:.1f}s"
    )
    return joined


def _merge_t2_parts(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    if left.is_empty():
        return right
    if right.is_empty():
        return left
    return (
        pl.concat([left, right], how="vertical")
        .group_by(["bt_index", "seat", "strain"])
        .agg(
            [pl.col("n").sum().cast(pl.UInt32).alias("n")]
            + [pl.col(f"hist_{i}").sum().cast(pl.UInt32).alias(f"hist_{i}") for i in range(14)]
        )
    )


def _empty_t2_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "bt_index": pl.UInt32,
            "seat": pl.UInt8,
            "strain": pl.UInt8,
            "n": pl.UInt32,
            **{f"hist_{i}": pl.UInt32 for i in range(14)},
        }
    )


def _t2_gpu_part(
    sub: pl.DataFrame,
    seat_idx: int,
    strain_idx: int,
    dd_col: str,
    gpu_chunk_rows: int,
) -> pl.DataFrame:
    torch = _get_torch_cuda()
    device = torch.device("cuda")
    out = _empty_t2_frame()
    cols = ["bt_index", dd_col]
    total = sub.height
    for start in range(0, total, gpu_chunk_rows):
        chunk = sub.slice(start, gpu_chunk_rows).select(cols)
        if chunk.is_empty():
            continue
        bt_np = chunk["bt_index"].to_numpy().astype(np.uint32, copy=False)
        dd_np = np.clip(chunk[dd_col].to_numpy().astype(np.int64, copy=False), 0, 13)

        bt_t = torch.from_numpy(bt_np.astype(np.int64, copy=False)).to(device=device)
        dd_t = torch.from_numpy(dd_np).to(device=device)
        uniq_bt, inv = torch.unique(bt_t, sorted=False, return_inverse=True)
        keys = inv.to(torch.int64) * 14 + dd_t.to(torch.int64)
        counts = torch.bincount(keys, minlength=int(uniq_bt.numel()) * 14).view(-1, 14)
        n_t = counts.sum(dim=1)

        bt_out = uniq_bt.to("cpu", non_blocking=False).numpy().astype(np.uint32, copy=False)
        n_out = n_t.to("cpu", non_blocking=False).numpy().astype(np.uint32, copy=False)
        hist_out = counts.to("cpu", non_blocking=False).numpy().astype(np.uint32, copy=False)
        chunk_df = pl.DataFrame(
            {
                "bt_index": bt_out,
                "seat": np.full(bt_out.shape[0], seat_idx, dtype=np.uint8),
                "strain": np.full(bt_out.shape[0], strain_idx, dtype=np.uint8),
                "n": n_out,
                **{f"hist_{i}": hist_out[:, i] for i in range(14)},
            }
        )
        out = _merge_t2_parts(out, chunk_df)
    torch.cuda.empty_cache()
    return out


def _build_trick_dist(
    joined: pl.DataFrame,
    ckpt: CheckpointManager,
    resume: bool,
    backend: str,
    gpu_chunk_rows: int,
) -> pl.DataFrame:
    t0 = time.perf_counter()
    print(f"[T2] start={_utc_now()}")
    if resume and ckpt.exists("t2_dist_final"):
        out = ckpt.read("t2_dist_final")
        print(f"[T2] resume hit: loaded final checkpoint ({out.height:,} rows)")
        return out

    if backend == "gpu":
        _ = _get_torch_cuda()

    out: pl.DataFrame
    if resume and ckpt.exists("t2_dist_base"):
        out = ckpt.read("t2_dist_base")
        print(f"[T2] resume hit: loaded base checkpoint ({out.height:,} rows)")
    else:
        out = _empty_t2_frame()
        total_parts = len(DEALER_DIRS) * 4 * len(STRAINS)
        part_idx = 0

        for dealer in DEALER_DIRS:
            sub = joined.filter(pl.col("Dealer") == dealer)
            if sub.is_empty():
                continue
            print(f"[T2] dealer={dealer}, rows={sub.height:,}")
            seat_dirs = _rotate_dirs_from_dealer(dealer)
            for seat_idx, decl_dir in enumerate(seat_dirs, start=1):
                for strain_idx, strain in enumerate(STRAINS):
                    part_idx += 1
                    part_key = f"t2_part_{dealer}_s{seat_idx}_st{strain_idx}"
                    if resume and ckpt.exists(part_key):
                        part = ckpt.read(part_key)
                        print(f"[T2] part {part_idx}/{total_parts} resume hit: {part_key}")
                    else:
                        dd_col = f"DD_{decl_dir}_{strain}"
                        if backend == "gpu":
                            part = _t2_gpu_part(
                                sub=sub,
                                seat_idx=seat_idx,
                                strain_idx=strain_idx,
                                dd_col=dd_col,
                                gpu_chunk_rows=gpu_chunk_rows,
                            )
                        else:
                            hist_exprs = [
                                (pl.col(dd_col) == i).sum().cast(pl.UInt32).alias(f"hist_{i}")
                                for i in range(14)
                            ]
                            part = (
                                sub.lazy()
                                .group_by("bt_index")
                                .agg([pl.len().cast(pl.UInt32).alias("n"), *hist_exprs])
                                .with_columns(
                                    [
                                        pl.lit(seat_idx).cast(pl.UInt8).alias("seat"),
                                        pl.lit(strain_idx).cast(pl.UInt8).alias("strain"),
                                    ]
                                )
                                .select(["bt_index", "seat", "strain", "n"] + [f"hist_{i}" for i in range(14)])
                                .collect(engine="streaming")
                            )
                        ckpt.write(part_key, part)
                        print(f"[T2] part {part_idx}/{total_parts} saved: {part_key} ({part.height:,} rows)")
                    out = _merge_t2_parts(out, part)
                    del part
                    gc.collect()

        if out.is_empty():
            raise ValueError("No trick histogram partials produced from joined data")
        ckpt.write("t2_dist_base", out)
        print(f"[T2] base checkpoint saved ({out.height:,} rows)")

    # Derive mean from histogram so merged partials remain exact.
    out = out.with_columns(
        (
            pl.sum_horizontal([pl.col(f"hist_{i}") * i for i in range(14)])
            / pl.col("n").clip(lower_bound=1)
        )
        .cast(pl.Float32)
        .alias("mean_tricks")
    )
    out = out.sort(["bt_index", "seat", "strain"])
    out = _compute_hist_quantiles(out, [f"hist_{i}" for i in range(14)], out_prefix="")
    out = out.rename(
        {
            "p10": "p10_tricks",
            "p25": "p25_tricks",
            "p50": "p50_tricks",
            "p75": "p75_tricks",
            "p90": "p90_tricks",
        }
    )
    ckpt.write("t2_dist_final", out)
    print(f"[T2] done rows={out.height:,}, elapsed={_elapsed_ms(t0)/1000.0:.1f}s")
    return out


def _merge_t3_parts(left: pl.DataFrame, right: pl.DataFrame, fields: list[str]) -> pl.DataFrame:
    if left.is_empty():
        return right
    if right.is_empty():
        return left
    agg_exprs: list[pl.Expr] = [pl.col("n").sum().cast(pl.UInt32).alias("n")]
    for f in fields:
        agg_exprs.append(pl.col(f"{f}_min").min().cast(pl.UInt8).alias(f"{f}_min"))
        agg_exprs.append(pl.col(f"{f}_max").max().cast(pl.UInt8).alias(f"{f}_max"))
        bins = 38 if f == "HCP" else (51 if f == "Total_Points" else 14)
        agg_exprs.extend(
            [pl.col(f"{f}_hist_{i}").sum().cast(pl.UInt32).alias(f"{f}_hist_{i}") for i in range(bins)]
        )
    return pl.concat([left, right], how="vertical").group_by(["bt_index", "seat"]).agg(agg_exprs)


def _empty_t3_frame(field_bins: dict[str, int]) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {"bt_index": pl.UInt32, "seat": pl.UInt8, "n": pl.UInt32}
    for field, bins in field_bins.items():
        schema[f"{field}_min"] = pl.UInt8
        schema[f"{field}_max"] = pl.UInt8
        for i in range(bins):
            schema[f"{field}_hist_{i}"] = pl.UInt32
    return pl.DataFrame(schema=schema)


def _hist_minmax_from_counts(counts_t, bins: int):
    torch = _get_torch_cuda()
    mask = counts_t > 0
    min_idx = torch.argmax(mask.to(torch.int8), dim=1).to(torch.int64)
    rev_idx = torch.argmax(torch.flip(mask, dims=[1]).to(torch.int8), dim=1).to(torch.int64)
    max_idx = (bins - 1 - rev_idx).to(torch.int64)
    return min_idx, max_idx


def _t3_gpu_part(
    sub: pl.DataFrame,
    seat_idx: int,
    decl_dir: str,
    field_bins: dict[str, int],
    gpu_chunk_rows: int,
) -> pl.DataFrame:
    torch = _get_torch_cuda()
    device = torch.device("cuda")
    cols = [
        "bt_index",
        f"HCP_{decl_dir}",
        f"Total_Points_{decl_dir}",
        f"SL_{decl_dir}_S",
        f"SL_{decl_dir}_H",
        f"SL_{decl_dir}_D",
        f"SL_{decl_dir}_C",
    ]
    rename = {
        f"HCP_{decl_dir}": "HCP",
        f"Total_Points_{decl_dir}": "Total_Points",
        f"SL_{decl_dir}_S": "SL_S",
        f"SL_{decl_dir}_H": "SL_H",
        f"SL_{decl_dir}_D": "SL_D",
        f"SL_{decl_dir}_C": "SL_C",
    }
    fields = list(field_bins.keys())
    out = _empty_t3_frame(field_bins)
    total = sub.height
    for start in range(0, total, gpu_chunk_rows):
        seat_df = sub.slice(start, gpu_chunk_rows).select(cols).rename(rename)
        if seat_df.is_empty():
            continue
        bt_np = seat_df["bt_index"].to_numpy().astype(np.uint32, copy=False)
        bt_t = torch.from_numpy(bt_np.astype(np.int64, copy=False)).to(device=device)
        uniq_bt, inv = torch.unique(bt_t, sorted=False, return_inverse=True)
        uniq_n = int(uniq_bt.numel())

        n_t = torch.bincount(inv.to(torch.int64), minlength=uniq_n)
        out_cols: dict[str, Any] = {
            "bt_index": uniq_bt.to("cpu", non_blocking=False).numpy().astype(np.uint32, copy=False),
            "seat": np.full(uniq_n, seat_idx, dtype=np.uint8),
            "n": n_t.to("cpu", non_blocking=False).numpy().astype(np.uint32, copy=False),
        }
        for field, bins in field_bins.items():
            vals_np = seat_df[field].to_numpy().astype(np.int64, copy=False)
            vals_np = np.clip(vals_np, 0, bins - 1)
            vals_t = torch.from_numpy(vals_np).to(device=device)
            keys = inv.to(torch.int64) * bins + vals_t.to(torch.int64)
            counts_t = torch.bincount(keys, minlength=uniq_n * bins).view(-1, bins)
            min_idx, max_idx = _hist_minmax_from_counts(counts_t, bins)
            out_cols[f"{field}_min"] = min_idx.to("cpu", non_blocking=False).numpy().astype(np.uint8, copy=False)
            out_cols[f"{field}_max"] = max_idx.to("cpu", non_blocking=False).numpy().astype(np.uint8, copy=False)
            counts_np = counts_t.to("cpu", non_blocking=False).numpy().astype(np.uint32, copy=False)
            for i in range(bins):
                out_cols[f"{field}_hist_{i}"] = counts_np[:, i]

        chunk_df = pl.DataFrame(out_cols)
        out = _merge_t3_parts(out, chunk_df, fields)
        del chunk_df
        gc.collect()
    torch.cuda.empty_cache()
    return out


def _build_hand_profile(
    joined: pl.DataFrame,
    ckpt: CheckpointManager,
    resume: bool,
    backend: str,
    gpu_chunk_rows: int,
) -> pl.DataFrame:
    t0 = time.perf_counter()
    print(f"[T3] start={_utc_now()}")
    if resume and ckpt.exists("t3_hand_final"):
        out = ckpt.read("t3_hand_final")
        print(f"[T3] resume hit: loaded final checkpoint ({out.height:,} rows)")
        return out

    field_bins: dict[str, int] = {
        "HCP": 38,
        "Total_Points": 51,
        "SL_S": 14,
        "SL_H": 14,
        "SL_D": 14,
        "SL_C": 14,
    }
    fields = list(field_bins.keys())
    if backend == "gpu":
        _ = _get_torch_cuda()

    if resume and ckpt.exists("t3_hand_base"):
        out = ckpt.read("t3_hand_base")
        print(f"[T3] resume hit: loaded base checkpoint ({out.height:,} rows)")
    else:
        out = _empty_t3_frame(field_bins)
        for dealer in DEALER_DIRS:
            dealer_t0 = time.perf_counter()
            sub = joined.filter(pl.col("Dealer") == dealer)
            if sub.is_empty():
                print(f"[T3] dealer={dealer}, rows=0 (skipped)")
                continue
            print(f"[T3] dealer={dealer}, rows={sub.height:,}")
            seat_dirs = _rotate_dirs_from_dealer(dealer)
            for seat_idx, decl_dir in enumerate(seat_dirs, start=1):
                part_key = f"t3_part_{dealer}_s{seat_idx}"
                if resume and ckpt.exists(part_key):
                    part = ckpt.read(part_key)
                    print(f"[T3] resume hit: {part_key}")
                else:
                    seat_t0 = time.perf_counter()
                    if backend == "gpu":
                        part = _t3_gpu_part(
                            sub=sub,
                            seat_idx=seat_idx,
                            decl_dir=decl_dir,
                            field_bins=field_bins,
                            gpu_chunk_rows=gpu_chunk_rows,
                        )
                    else:
                        seat_df = sub.select(
                            [
                                pl.col("bt_index"),
                                pl.lit(seat_idx).cast(pl.UInt8).alias("seat"),
                                pl.col(f"HCP_{decl_dir}").cast(pl.UInt8).alias("HCP"),
                                pl.col(f"Total_Points_{decl_dir}").cast(pl.UInt8).alias("Total_Points"),
                                pl.col(f"SL_{decl_dir}_S").cast(pl.UInt8).alias("SL_S"),
                                pl.col(f"SL_{decl_dir}_H").cast(pl.UInt8).alias("SL_H"),
                                pl.col(f"SL_{decl_dir}_D").cast(pl.UInt8).alias("SL_D"),
                                pl.col(f"SL_{decl_dir}_C").cast(pl.UInt8).alias("SL_C"),
                            ]
                        )
                        agg_exprs: list[pl.Expr] = [pl.len().cast(pl.UInt32).alias("n")]
                        for field, bins in field_bins.items():
                            agg_exprs.append(pl.col(field).min().cast(pl.UInt8).alias(f"{field}_min"))
                            agg_exprs.append(pl.col(field).max().cast(pl.UInt8).alias(f"{field}_max"))
                            for i in range(bins):
                                agg_exprs.append(
                                    (pl.col(field) == i).sum().cast(pl.UInt32).alias(f"{field}_hist_{i}")
                                )
                        part = seat_df.group_by(["bt_index", "seat"]).agg(agg_exprs)
                    ckpt.write(part_key, part)
                    print(
                        f"[T3] dealer={dealer}, seat={seat_idx}, dir={decl_dir}: "
                        f"part saved ({part.height:,} rows) in {_elapsed_ms(seat_t0)/1000.0:.1f}s"
                    )
                out = _merge_t3_parts(out, part, fields)
                del part
                gc.collect()
            print(f"[T3] dealer={dealer} complete in {_elapsed_ms(dealer_t0)/1000.0:.1f}s")
        out = out.sort(["bt_index", "seat"])
        ckpt.write("t3_hand_base", out)
        print(f"[T3] base checkpoint saved ({out.height:,} rows)")

    # Derive p10/p50/p90 from histograms for each field.
    if resume and ckpt.exists("t3_hand_qwork"):
        out = ckpt.read("t3_hand_qwork")
        print(f"[T3] resume hit: loaded quantile work checkpoint ({out.height:,} rows)")

    for field, bins in field_bins.items():
        if f"{field}_p90" in out.columns:
            print(f"[T3] quantiles {field} already present (resume skip)")
            continue
        q_t0 = time.perf_counter()
        hist_cols = [f"{field}_hist_{i}" for i in range(bins)]
        q_df = _compute_hist_quantiles(out.select(["bt_index", "seat", "n"] + hist_cols), hist_cols, out_prefix="")
        out = out.join(
            q_df.select(
                [
                    "bt_index",
                    "seat",
                    pl.col("p10").cast(pl.Float32).alias(f"{field}_p10"),
                    pl.col("p50").cast(pl.Float32).alias(f"{field}_p50"),
                    pl.col("p90").cast(pl.Float32).alias(f"{field}_p90"),
                ]
            ),
            on=["bt_index", "seat"],
            how="left",
        )
        ckpt.write("t3_hand_qwork", out)
        print(f"[T3] quantiles {field} done in {_elapsed_ms(q_t0)/1000.0:.1f}s")

    ckpt.write("t3_hand_final", out)
    print(f"[T3] done rows={out.height:,}, elapsed={_elapsed_ms(t0)/1000.0:.1f}s")
    return out


def _build_trick_quality(dist_df: pl.DataFrame, ckpt: CheckpointManager, resume: bool) -> pl.DataFrame:
    t0 = time.perf_counter()
    print(f"[T5] start={_utc_now()}")
    if resume and ckpt.exists("t5_trick_quality_final"):
        out = ckpt.read("t5_trick_quality_final")
        print(f"[T5] resume hit: loaded final checkpoint ({out.height:,} rows)")
        return out
    hist_cols = [f"hist_{i}" for i in range(14)]
    rows: list[pl.DataFrame] = []

    def _safe_sum_horizontal(cols: list[str]) -> pl.Expr:
        if not cols:
            return pl.lit(0, dtype=pl.UInt32)
        return pl.sum_horizontal([pl.col(c) for c in cols])

    for level in range(1, 8):
        book = level + 6
        make_cols = hist_cols[book:]
        plus1_cols = hist_cols[min(book + 1, 14):]
        p_make_expr = (_safe_sum_horizontal(make_cols) / pl.col("n").clip(lower_bound=1)).cast(pl.Float32)
        p_plus1_expr = (_safe_sum_horizontal(plus1_cols) / pl.col("n").clip(lower_bound=1)).cast(pl.Float32)
        base = dist_df.select(
            [
                "bt_index",
                "seat",
                "strain",
                "n",
                "mean_tricks",
                pl.lit(level).cast(pl.UInt8).alias("level"),
                pl.lit(book).cast(pl.UInt8).alias("book"),
                (pl.col("mean_tricks") - float(book)).cast(pl.Float32).alias("exp_extra"),
                p_make_expr.alias("p_make"),
                p_plus1_expr.alias("p_plus1"),
            ]
        )
        rows.append(base)

    level_df = pl.concat(rows, how="vertical")
    vul_nv = level_df.with_columns(pl.lit(0).cast(pl.UInt8).alias("vul"))
    vul_v = level_df.with_columns(pl.lit(1).cast(pl.UInt8).alias("vul"))
    out = pl.concat([vul_nv, vul_v], how="vertical").sort(["bt_index", "seat", "strain", "level", "vul"])
    ckpt.write("t5_trick_quality_final", out)
    print(f"[T5] done rows={out.height:,}, elapsed={_elapsed_ms(t0)/1000.0:.1f}s")
    return out


def _write(df: pl.DataFrame, path: Path, name: str) -> None:
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path), compression="zstd")
    print(f"[write] {name}: {path} ({df.height:,} rows, {df.width} cols) in {_elapsed_ms(t0)/1000.0:.1f}s")


def _sample_joined_for_validation(joined: pl.DataFrame, sample_rows: int, seed: int) -> pl.DataFrame:
    if sample_rows <= 0 or joined.height <= sample_rows:
        return joined
    t0 = time.perf_counter()
    sampled = joined.sample(n=sample_rows, with_replacement=False, shuffle=True, seed=seed)
    print(
        f"[validate] sampled joined rows={sampled.height:,}/{joined.height:,} "
        f"(seed={seed}) in {_elapsed_ms(t0)/1000.0:.1f}s"
    )
    return sampled


def _compare_outputs(
    name: str,
    cpu_df: pl.DataFrame,
    gpu_df: pl.DataFrame,
    key_cols: list[str],
    float_cols: set[str],
    atol: float = 1e-6,
) -> None:
    if set(cpu_df.columns) != set(gpu_df.columns):
        cpu_only = sorted(set(cpu_df.columns) - set(gpu_df.columns))
        gpu_only = sorted(set(gpu_df.columns) - set(cpu_df.columns))
        raise ValueError(f"[validate] {name}: column mismatch cpu_only={cpu_only} gpu_only={gpu_only}")
    cols = sorted(cpu_df.columns)
    cpu = cpu_df.select(cols).sort(key_cols)
    gpu = gpu_df.select(cols).sort(key_cols)
    if cpu.height != gpu.height:
        raise ValueError(f"[validate] {name}: row count mismatch cpu={cpu.height:,} gpu={gpu.height:,}")
    if cpu.is_empty() and gpu.is_empty():
        print(f"[validate] {name}: both outputs empty")
        return

    for c in cols:
        cpu_np = cpu[c].to_numpy()
        gpu_np = gpu[c].to_numpy()
        if c in float_cols:
            if not np.allclose(cpu_np, gpu_np, atol=atol, rtol=0.0, equal_nan=True):
                diff = np.nanmax(np.abs(cpu_np.astype(np.float64) - gpu_np.astype(np.float64)))
                raise ValueError(f"[validate] {name}.{c}: float mismatch (max_abs_diff={diff})")
        else:
            if not np.array_equal(cpu_np, gpu_np):
                raise ValueError(f"[validate] {name}.{c}: exact mismatch")
    print(f"[validate] {name}: CPU and GPU outputs match ({cpu.height:,} rows)")


def _validate_gpu_vs_cpu(
    joined: pl.DataFrame,
    ckpt_root: Path,
    deals_path: Path,
    verified_map_path: Path,
    bt_metadata_path: Path,
    gpu_chunk_rows: int,
    sample_rows: int,
    seed: int,
) -> None:
    print("[validate] mode=GPU-vs-CPU started")
    sampled = _sample_joined_for_validation(joined, sample_rows=sample_rows, seed=seed)

    cpu_ckpt = CheckpointManager(
        checkpoint_dir=ckpt_root / "validate_cpu",
        resume=False,
        deals_path=deals_path,
        verified_map_path=verified_map_path,
        bt_metadata_path=bt_metadata_path,
    )
    gpu_ckpt = CheckpointManager(
        checkpoint_dir=ckpt_root / "validate_gpu",
        resume=False,
        deals_path=deals_path,
        verified_map_path=verified_map_path,
        bt_metadata_path=bt_metadata_path,
    )

    t2_cpu = _build_trick_dist(sampled, ckpt=cpu_ckpt, resume=False, backend="cpu", gpu_chunk_rows=gpu_chunk_rows)
    t3_cpu = _build_hand_profile(sampled, ckpt=cpu_ckpt, resume=False, backend="cpu", gpu_chunk_rows=gpu_chunk_rows)
    t2_gpu = _build_trick_dist(sampled, ckpt=gpu_ckpt, resume=False, backend="gpu", gpu_chunk_rows=gpu_chunk_rows)
    t3_gpu = _build_hand_profile(sampled, ckpt=gpu_ckpt, resume=False, backend="gpu", gpu_chunk_rows=gpu_chunk_rows)

    _compare_outputs(
        name="T2",
        cpu_df=t2_cpu,
        gpu_df=t2_gpu,
        key_cols=["bt_index", "seat", "strain"],
        float_cols={"mean_tricks", "p10_tricks", "p25_tricks", "p50_tricks", "p75_tricks", "p90_tricks"},
    )
    _compare_outputs(
        name="T3",
        cpu_df=t3_cpu,
        gpu_df=t3_gpu,
        key_cols=["bt_index", "seat"],
        float_cols={c for c in t3_cpu.columns if c.endswith("_p10") or c.endswith("_p50") or c.endswith("_p90")},
    )
    print("[validate] mode=GPU-vs-CPU passed")


def run(
    deals_path: Path,
    verified_map_path: Path,
    bt_metadata_path: Path,
    out_trick_dist: Path,
    out_hand_profile: Path,
    out_trick_quality: Path,
    checkpoint_dir: Path,
    resume: bool,
    backend: str,
    gpu_chunk_rows: int,
    validate_gpu_vs_cpu: bool,
    validate_sample_rows: int,
    validate_seed: int,
) -> None:
    start = time.perf_counter()
    start_dt = _utc_now()
    print(f"[run] start={start_dt}")
    print(f"[run] bt_metadata={bt_metadata_path}")
    print(f"[run] git_sha={_get_git_sha()}")
    print(f"[run] checkpoint_dir={checkpoint_dir}")
    print(f"[run] resume={resume}")
    print(f"[run] backend={backend}")
    print(f"[run] gpu_chunk_rows={gpu_chunk_rows:,}")
    print(f"[run] validate_gpu_vs_cpu={validate_gpu_vs_cpu}")
    if backend == "gpu":
        torch = _get_torch_cuda()
        print(f"[run] cuda_device={torch.cuda.get_device_name(0)}")

    ckpt = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        deals_path=deals_path,
        verified_map_path=verified_map_path,
        bt_metadata_path=bt_metadata_path,
    )

    joined = _build_joined(deals_path=deals_path, verified_map_path=verified_map_path)
    if validate_gpu_vs_cpu:
        _validate_gpu_vs_cpu(
            joined=joined,
            ckpt_root=checkpoint_dir,
            deals_path=deals_path,
            verified_map_path=verified_map_path,
            bt_metadata_path=bt_metadata_path,
            gpu_chunk_rows=gpu_chunk_rows,
            sample_rows=validate_sample_rows,
            seed=validate_seed,
        )
        del joined
        gc.collect()
        end_dt = _utc_now()
        elapsed = time.perf_counter() - start
        print(f"[run] end={end_dt}")
        print(f"[run] elapsed={elapsed:.1f}s")
        return

    dist_df = _build_trick_dist(
        joined,
        ckpt=ckpt,
        resume=resume,
        backend=backend,
        gpu_chunk_rows=gpu_chunk_rows,
    )
    hand_df = _build_hand_profile(
        joined,
        ckpt=ckpt,
        resume=resume,
        backend=backend,
        gpu_chunk_rows=gpu_chunk_rows,
    )
    del joined
    gc.collect()
    tq_df = _build_trick_quality(dist_df, ckpt=ckpt, resume=resume)

    build_time = _utc_now()
    source_files = f"{deals_path};{verified_map_path};{bt_metadata_path}"
    git_sha = _get_git_sha()
    dist_df = dist_df.with_columns(
        [
            pl.lit(build_time).alias("build_time"),
            pl.lit(source_files).alias("source_files"),
            pl.lit(git_sha).alias("git_sha"),
        ]
    )
    hand_df = hand_df.with_columns(
        [
            pl.lit(build_time).alias("build_time"),
            pl.lit(source_files).alias("source_files"),
            pl.lit(git_sha).alias("git_sha"),
        ]
    )
    tq_df = tq_df.with_columns(
        [
            pl.lit(build_time).alias("build_time"),
            pl.lit(source_files).alias("source_files"),
            pl.lit(git_sha).alias("git_sha"),
        ]
    )

    _write(dist_df, out_trick_dist, "bt_trick_dist")
    _write(hand_df, out_hand_profile, "bt_hand_profile")
    _write(tq_df, out_trick_quality, "bt_trick_quality")

    end_dt = _utc_now()
    elapsed = time.perf_counter() - start
    print(f"[run] end={end_dt}")
    print(f"[run] elapsed={elapsed:.1f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trick-quality artifacts from deals + verified BT mapping.")
    parser.add_argument("--deals", type=Path, default=Path("E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet"))
    parser.add_argument("--verified-map", type=Path, default=Path("E:/bridge/data/bbo/bidding/bbo_deal_to_bt_verified.parquet"))
    parser.add_argument("--bt-metadata", type=Path, default=Path("E:/bridge/data/bbo/bidding/bbo_bt_seat1.parquet"))
    parser.add_argument("--out-trick-dist", type=Path, default=Path("E:/bridge/data/bbo/bidding/bt_trick_dist_gpu_v1.parquet"))
    parser.add_argument("--out-hand-profile", type=Path, default=Path("E:/bridge/data/bbo/bidding/bt_hand_profile_gpu_v1.parquet"))
    parser.add_argument("--out-trick-quality", type=Path, default=Path("E:/bridge/data/bbo/bidding/bt_trick_quality_gpu_v1.parquet"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("E:/bridge/data/bbo/bidding/checkpoints_trick_quality_gpu"),
        help="Checkpoint directory for restartable multi-hour runs",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu", help="Aggregation backend")
    parser.add_argument(
        "--gpu-chunk-rows",
        type=int,
        default=2_000_000,
        help="Rows per GPU chunk for histogram accumulation",
    )
    parser.add_argument(
        "--validate-gpu-vs-cpu",
        action="store_true",
        help="Run T2/T3 CPU vs GPU validation on a sampled subset and exit",
    )
    parser.add_argument(
        "--validate-sample-rows",
        type=int,
        default=2_000_000,
        help="Joined-row sample size for --validate-gpu-vs-cpu (<=0 means full joined set)",
    )
    parser.add_argument("--validate-seed", type=int, default=42, help="Random seed for validation sampling")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        deals_path=args.deals,
        verified_map_path=args.verified_map,
        bt_metadata_path=args.bt_metadata,
        out_trick_dist=args.out_trick_dist,
        out_hand_profile=args.out_hand_profile,
        out_trick_quality=args.out_trick_quality,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        backend=args.backend,
        gpu_chunk_rows=args.gpu_chunk_rows,
        validate_gpu_vs_cpu=args.validate_gpu_vs_cpu,
        validate_sample_rows=args.validate_sample_rows,
        validate_seed=args.validate_seed,
    )
