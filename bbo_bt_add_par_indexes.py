#!/usr/bin/env python3
"""
Build per-deal Par-matching BT index lists (offline, precomputed).

Takes 70s.

Goal:
- Create a parquet dataset mapping:
    deal_idx (row index in bbo_mldf_augmented.parquet) -> Par_Indexes (List[UInt32])
  where Par_Indexes is a subset of the GPU-verified deal→BT matches (Matched_BT_Indices),
  filtered to only those completed BT auctions whose *final contract signature* matches
  the deal's ParContracts (same semantics as Streamlit "Matching BT Auctions (by Par)").

Why this exists:
- The on-the-fly endpoint /find-bt-auctions-by-contracts currently scans completed auctions.
  For existing deals we can make it near-instant via O(log n) lookup of Par_Indexes.

Inputs (no fallback logic; fail-fast):
- deals parquet: E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet
    required cols: Dealer, ParContracts
    NOTE: deal_idx is the row index.
 - verified deal→BT parquet: E:/bridge/data/bbo/bidding/bbo_deal_to_bt_verified.parquet
    required cols: deal_idx, Matched_BT_Indices
 - completed BT auctions parquet (small): E:/bridge/data/bbo/bidding/bbo_bt_completed_agg_expr.parquet
    required cols: bt_index, Auction

Output:
- A single parquet file (written incrementally) at:
    E:/bridge/data/bbo/bidding/bbo_deal_to_bt_par_verified.parquet
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import polars as pl
import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from bbo_bidding_queries_lib import parse_contract_from_auction, get_declarer_for_auction


DEFAULT_DEALS_FILE = Path("E:/bridge/data/bbo/data/bbo_mldf_augmented.parquet")
DEFAULT_VERIFIED_FILE = Path("E:/bridge/data/bbo/bidding/bbo_deal_to_bt_verified.parquet")
DEFAULT_BT_COMPLETED_AGG_FILE = Path("E:/bridge/data/bbo/bidding/bbo_bt_completed_agg_expr.parquet")
DEFAULT_OUTPUT = Path("E:/bridge/data/bbo/bidding/bbo_deal_to_bt_par_verified.parquet")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


@dataclass(frozen=True)
class ContractKeyCodec:
    # level: 1..7
    # strain: C/D/H/S/N -> 0..4
    # dbl: ""/X/XX -> 0..2
    # pair_is_opener: 0/1
    #
    # key = (((level*5 + strain_code)*3 + dbl_code)*2 + pair_bit)  (fits UInt16)
    pass

    @staticmethod
    def strain_code(strain: str) -> int | None:
        s = (strain or "").strip().upper()
        if s == "C":
            return 0
        if s == "D":
            return 1
        if s == "H":
            return 2
        if s == "S":
            return 3
        if s in ("N", "NT"):
            return 4
        return None

    @staticmethod
    def dbl_code(dbl: str) -> int:
        d = (dbl or "").strip().upper()
        if d == "X":
            return 1
        if d == "XX":
            return 2
        return 0

    @staticmethod
    def encode(level: int, strain: str, dbl: str, pair_is_opener: bool) -> int | None:
        try:
            l = int(level)
        except Exception:
            return None
        if l < 1 or l > 7:
            return None
        sc = ContractKeyCodec.strain_code(strain)
        if sc is None:
            return None
        dc = ContractKeyCodec.dbl_code(dbl)
        bit = 1 if bool(pair_is_opener) else 0
        return (((l * 5 + int(sc)) * 3 + int(dc)) * 2 + int(bit))


def _bt_contract_key_from_auction(auction: str) -> tuple[int, str, str, bool] | None:
    """Return (level, strain, dbl, pair_is_opener) for a completed BT auction in seat-1 view."""
    auc = str(auction or "").strip().upper()
    if not auc:
        return None

    parsed = parse_contract_from_auction(auc)
    if not parsed:
        return None
    level, strain, dbl_count = parsed
    dbl = "XX" if int(dbl_count) == 2 else ("X" if int(dbl_count) == 1 else "")

    # Declarer in seat-1 view: opener is North ("N")
    decl = get_declarer_for_auction(auc, "N")
    if not decl:
        return None
    pair_is_opener = decl in ("N", "S")
    return int(level), str(strain).upper(), dbl, bool(pair_is_opener)


def _require_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def _build_bt_key_df(bt_completed_agg_file: Path) -> pl.DataFrame:
    _log(f"Loading completed BT auctions: {bt_completed_agg_file}")
    bt_df = pl.read_parquet(bt_completed_agg_file, columns=["bt_index", "Auction"])
    if "bt_index" not in bt_df.columns or "Auction" not in bt_df.columns:
        raise ValueError("BT completed agg file must contain columns: bt_index, Auction")

    t0 = time.perf_counter()
    bt_index = bt_df.get_column("bt_index").to_list()
    auctions = bt_df.get_column("Auction").to_list()

    keys: list[int | None] = [None] * len(auctions)
    n_ok = 0
    for i, auc in enumerate(auctions):
        info = _bt_contract_key_from_auction(str(auc or ""))
        if not info:
            continue
        level, strain, dbl, pair_is_opener = info
        k = ContractKeyCodec.encode(int(level), str(strain), str(dbl), bool(pair_is_opener))
        if k is None:
            continue
        keys[i] = int(k)
        n_ok += 1

    out = (
        pl.DataFrame(
            {
                "bt_index": pl.Series("bt_index", bt_index).cast(pl.UInt32),
                "bt_sig_key": pl.Series("bt_sig_key", keys, dtype=pl.UInt16),
            }
        )
        .drop_nulls(subset=["bt_index", "bt_sig_key"])
        .unique(subset=["bt_index"])
    )
    elapsed = time.perf_counter() - t0
    _log(f"BT signature keys built for {n_ok:,}/{len(auctions):,} auctions in {elapsed:.1f}s")
    return out


def _iter_ranges(n: int, chunk: int) -> Iterable[tuple[int, int]]:
    if chunk <= 0:
        raise ValueError("chunk must be > 0")
    i = 0
    while i < n:
        j = min(n, i + chunk)
        yield i, j
        i = j


def run(
    deals_file: Path,
    verified_file: Path,
    bt_completed_agg_file: Path,
    output_path: Path,
    chunk_size: int,
    max_deals: int | None,
) -> None:
    t0_all = time.perf_counter()
    _log("=" * 70)
    _log("BUILD DEAL→PAR_INDEXES (offline)")
    _log("=" * 70)
    _log(f"Start: {_now()}")

    _require_exists(deals_file, "deals parquet")
    _require_exists(verified_file, "verified deal→BT parquet")
    _require_exists(bt_completed_agg_file, "completed BT auctions parquet")

    if output_path.exists():
        _log(f"Removing existing output: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Step 1: BT signature map (bt_index -> bt_sig_key)
    # ---------------------------------------------------------------------
    t0 = time.perf_counter()
    bt_key_df = _build_bt_key_df(bt_completed_agg_file)
    bt_key_df = bt_key_df.sort("bt_index")
    _log(f"[1/3] BT signature map ready in {(time.perf_counter() - t0):.1f}s ({bt_key_df.height:,} rows)")

    # ---------------------------------------------------------------------
    # Step 2: Figure out deal count
    # ---------------------------------------------------------------------
    t0 = time.perf_counter()
    n_deals_total = pl.scan_parquet(deals_file).select(pl.len()).collect().item()
    if max_deals is not None:
        n_deals_total = min(int(n_deals_total), int(max_deals))
    _log(f"[2/3] Deals: {n_deals_total:,} (counted in {(time.perf_counter() - t0):.1f}s)")

    # ---------------------------------------------------------------------
    # Step 3: Chunked build of deal_idx -> Par_Indexes
    # ---------------------------------------------------------------------
    _log(f"[3/3] Computing Par_Indexes in chunks of {chunk_size:,} deals")
    t_step = time.perf_counter()

    # These are scans; we slice them per chunk to keep memory bounded.
    deals_scan = pl.scan_parquet(deals_file).with_row_index("deal_idx")
    verified_scan = pl.scan_parquet(verified_file).select(["deal_idx", "Matched_BT_Indices"])

    # Basic schema validation up-front (fail-fast).
    deals_schema = deals_scan.collect_schema()
    for col in ("Dealer", "ParContracts"):
        if col not in deals_schema.names():
            raise ValueError(f"deals parquet missing required column: {col}")
    verified_schema = verified_scan.collect_schema()
    for col in ("deal_idx", "Matched_BT_Indices"):
        if col not in verified_schema.names():
            raise ValueError(f"verified parquet missing required column: {col}")

    # Optional progress bar
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        pbar = tqdm(total=n_deals_total, desc="Par_Indexes", unit="deals")
    except Exception:
        pbar = None

    out_schema = pa.schema(
        [
            pa.field("deal_idx", pa.uint32(), nullable=False),
            pa.field("Par_Indexes", pa.list_(pa.uint32()), nullable=False),
        ]
    )
    writer: pq.ParquetWriter | None = None
    try:
        for i0, i1 in _iter_ranges(n_deals_total, chunk_size):
            # Use deal_idx range filtering for verified index (do not assume row-aligned slice).
            deal_lo = int(i0)
            deal_hi = int(i1)

            # Pull only what we need for this chunk.
            deals_chunk = (
                deals_scan.slice(i0, i1 - i0)
                .select(["deal_idx", "Dealer", "ParContracts"])
                .collect()
            )
            verified_chunk = (
                verified_scan.filter((pl.col("deal_idx") >= deal_lo) & (pl.col("deal_idx") < deal_hi))
                .collect()
            )

            # Join to ensure we only compute for deals present in verified index.
            base = deals_chunk.join(verified_chunk, on="deal_idx", how="left")

            # Build Par signature keys per deal, vectorized via explode + group_by.
            # opener_pair: NS if dealer is N/S, else EW
            exploded = (
                base.select(["deal_idx", "Dealer", "ParContracts", "Matched_BT_Indices"])
                .with_columns(
                    pl.when(pl.col("Dealer").cast(pl.Utf8).str.to_uppercase().is_in(["N", "S"]))
                    .then(pl.lit("NS"))
                    .otherwise(pl.lit("EW"))
                    .alias("_opener_pair")
                )
                .explode("ParContracts")
            )

            # If ParContracts is null for a deal, explode() drops it; we handle missing keys later.
            pc = pl.col("ParContracts")
            # Some datasets may store ParContracts already formatted as strings; that is not usable here.
            if exploded.schema.get("ParContracts") in (pl.Utf8, pl.String):
                raise ValueError(
                    "deals.ParContracts appears to be a string column; expected List[Struct] from bbo_mldf_augmented.parquet"
                )

            # ParContracts field names can vary slightly across datasets; only access fields that exist.
            pc_dtype = exploded.schema.get("ParContracts")
            pc_fields: set[str] = set()
            try:
                # Struct type exposes .fields (list of Field objects)
                for f in getattr(pc_dtype, "fields", []) or []:
                    nm = getattr(f, "name", None)
                    if nm:
                        pc_fields.add(str(nm))
            except Exception:
                pc_fields = set()

            required = ["Level", "Strain", "Pair_Direction"]
            missing = [x for x in required if x not in pc_fields]
            if missing:
                raise ValueError(f"ParContracts is missing required fields: {missing}. Present: {sorted(pc_fields)}")

            dbl_exprs: list[pl.Expr] = []
            # Most common: "Doubled" is present. Some datasets may use "Double"; only use if present.
            if "Doubled" in pc_fields:
                dbl_exprs.append(pc.struct.field("Doubled").cast(pl.Utf8))
            if "Double" in pc_fields:
                dbl_exprs.append(pc.struct.field("Double").cast(pl.Utf8))
            dbl_exprs.append(pl.lit(""))

            sigs = (
                exploded.with_columns(
                    [
                        pc.struct.field("Level").cast(pl.Int16).alias("_lvl"),
                        pc.struct.field("Strain").cast(pl.Utf8).str.to_uppercase().alias("_strain"),
                        pc.struct.field("Pair_Direction").cast(pl.Utf8).str.to_uppercase().alias("_pair"),
                        pl.coalesce(dbl_exprs).str.to_uppercase().alias("_dbl"),
                    ]
                )
                .with_columns(
                    [
                        pl.when(pl.col("_strain") == "C")
                        .then(pl.lit(0))
                        .when(pl.col("_strain") == "D")
                        .then(pl.lit(1))
                        .when(pl.col("_strain") == "H")
                        .then(pl.lit(2))
                        .when(pl.col("_strain") == "S")
                        .then(pl.lit(3))
                        .when(pl.col("_strain").is_in(["N", "NT"]))
                        .then(pl.lit(4))
                        .otherwise(None)
                        .cast(pl.UInt16)
                        .alias("_sc"),
                        pl.when(pl.col("_dbl") == "X")
                        .then(pl.lit(1))
                        .when(pl.col("_dbl") == "XX")
                        .then(pl.lit(2))
                        .otherwise(pl.lit(0))
                        .cast(pl.UInt16)
                        .alias("_dc"),
                        (pl.col("_pair") == pl.col("_opener_pair")).cast(pl.UInt16).alias("_pb"),
                    ]
                )
                .with_columns(
                    (
                        (((pl.col("_lvl").cast(pl.UInt16) * 5 + pl.col("_sc")) * 3 + pl.col("_dc")) * 2)
                        + pl.col("_pb")
                    )
                    .alias("par_sig_key")
                    .cast(pl.UInt16)
                )
                .drop_nulls(subset=["par_sig_key"])
                .group_by("deal_idx")
                .agg(pl.col("par_sig_key").unique().alias("Par_Sig_Keys"))
            )

            # Now filter Matched_BT_Indices to those whose bt_sig_key is in Par_Sig_Keys.
            # Explode matched indices, join bt signature key, then membership test.
            matched_expl = (
                base.select(["deal_idx", "Matched_BT_Indices"])
                .with_columns(
                    pl.when(pl.col("Matched_BT_Indices").is_null())
                    .then(pl.lit([]).cast(pl.List(pl.UInt32)))
                    .otherwise(pl.col("Matched_BT_Indices").cast(pl.List(pl.UInt32)))
                    .alias("Matched_BT_Indices")
                )
                .explode("Matched_BT_Indices")
                .rename({"Matched_BT_Indices": "bt_index"})
            )

            joined = (
                matched_expl.join(bt_key_df, on="bt_index", how="inner")
                .join(sigs, on="deal_idx", how="left")
                .filter(pl.col("Par_Sig_Keys").list.contains(pl.col("bt_sig_key")))
                .group_by("deal_idx")
                .agg(pl.col("bt_index").cast(pl.UInt32).unique().alias("Par_Indexes"))
            )

            # Ensure all deals in this chunk are present with empty list for no matches.
            out_chunk = (
                base.select(["deal_idx"])
                .join(joined, on="deal_idx", how="left")
                .with_columns(
                    pl.when(pl.col("Par_Indexes").is_null())
                    .then(pl.lit([]).cast(pl.List(pl.UInt32)))
                    .otherwise(pl.col("Par_Indexes"))
                    .alias("Par_Indexes")
                )
                .sort("deal_idx")
            )

            # Enforce stable dtypes before writing.
            out_chunk = out_chunk.with_columns(
                [
                    pl.col("deal_idx").cast(pl.UInt32),
                    pl.col("Par_Indexes").cast(pl.List(pl.UInt32)),
                ]
            )

            tbl = out_chunk.to_arrow()
            try:
                tbl = tbl.cast(out_schema)
            except Exception:
                raise ValueError(f"Output schema mismatch; got: {tbl.schema}, expected: {out_schema}")

            if writer is None:
                writer = pq.ParquetWriter(
                    where=str(output_path),
                    schema=out_schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(tbl, row_group_size=min(int(chunk_size), tbl.num_rows))

            if pbar:
                pbar.update(i1 - i0)
    finally:
        if pbar:
            pbar.close()
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - t_step
    _log(f"Computed Par_Indexes in {elapsed:.1f}s")
    try:
        size_mb = output_path.stat().st_size / 1e6
        _log(f"Output file size: {size_mb:.1f} MB")
    except Exception:
        pass

    total_elapsed = time.perf_counter() - t0_all
    _log("=" * 70)
    _log("DONE")
    _log(f"End: {_now()}")
    _log(f"Elapsed: {total_elapsed:.1f}s")
    _log(f"Output file: {output_path}")
    _log("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deal→Par_Indexes parquet dataset.")
    parser.add_argument("--deals-file", type=Path, default=DEFAULT_DEALS_FILE)
    parser.add_argument("--verified-file", type=Path, default=DEFAULT_VERIFIED_FILE)
    parser.add_argument("--bt-completed-file", type=Path, default=DEFAULT_BT_COMPLETED_AGG_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-deals", type=int, default=None)
    args = parser.parse_args()

    run(
        deals_file=args.deals_file,
        verified_file=args.verified_file,
        bt_completed_agg_file=args.bt_completed_file,
        output_path=args.output,
        chunk_size=int(args.chunk_size),
        max_deals=args.max_deals,
    )


if __name__ == "__main__":
    main()

