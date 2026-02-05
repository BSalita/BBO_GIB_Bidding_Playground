"""
Phase-1 bid ranking library (library-first core).

This module centralizes the "Phase 1" mechanics described in `BIDDING_MODEL_IMPLEMENTATION.md`:
  - enumerate next bids (provided by caller)
  - attach precomputed Avg_Par / Avg_EV statistics (NV/V or aggregate fallback)
  - rank candidates (fast, deterministic)

It is intentionally free of FastAPI/Streamlit concerns so both CLI and API handlers can reuse it.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import polars as pl


def get_avg_ev_par_precomputed_nv_v(
    precomputed: Mapping[str, Any] | None,
    *,
    seat: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Extract (avg_ev_nv, avg_ev_v, avg_par_nv, avg_par_v) from a precomputed stats row.

    Supports both schemas:
      - new: Avg_EV_S{seat}_NV / Avg_EV_S{seat}_V and Avg_Par_S{seat}_NV / Avg_Par_S{seat}_V
      - old: Avg_EV_S{seat} and Avg_Par_S{seat} (used for both NV and V)
    """
    if precomputed is None:
        precomputed = {}

    avg_ev_nv = precomputed.get(f"Avg_EV_S{seat}_NV")
    avg_ev_v = precomputed.get(f"Avg_EV_S{seat}_V")
    avg_par_nv = precomputed.get(f"Avg_Par_S{seat}_NV")
    avg_par_v = precomputed.get(f"Avg_Par_S{seat}_V")

    # Fall back to aggregate if NV/V not available.
    if avg_ev_nv is None:
        aggregate_ev = precomputed.get(f"Avg_EV_S{seat}")
        avg_ev_nv = aggregate_ev
        avg_ev_v = aggregate_ev
    if avg_par_nv is None:
        aggregate_par = precomputed.get(f"Avg_Par_S{seat}")
        avg_par_nv = aggregate_par
        avg_par_v = aggregate_par

    return (
        float(avg_ev_nv) if avg_ev_nv is not None else None,
        float(avg_ev_v) if avg_ev_v is not None else None,
        float(avg_par_nv) if avg_par_nv is not None else None,
        float(avg_par_v) if avg_par_v is not None else None,
    )


def preload_precomputed_ev_par_stats(
    bt_ev_stats_df: pl.DataFrame,
    *,
    bt_indices: list[int],
) -> dict[int, dict[str, Any]]:
    """Preload NV/V split Avg_EV/Avg_Par (with aggregate fallback) for many bt_index values.

    This is used by both:
    - `/list-next-bids` (to attach instantaneous per-seat precomputed EV/Par)
    - `/rank-bids-by-ev` (to attach precomputed EV/Par alongside computed values)
    """
    if bt_ev_stats_df is None or bt_ev_stats_df.is_empty() or not bt_indices:
        return {}

    bt_indices_i = [int(x) for x in bt_indices if x is not None]
    if not bt_indices_i:
        return {}

    ev_subset = bt_ev_stats_df.filter(pl.col("bt_index").is_in(bt_indices_i))
    out: dict[int, dict[str, Any]] = {}
    for ev_row in ev_subset.iter_rows(named=True):
        ev_data: dict[str, Any] = {}
        for s in range(1, 5):
            # Try new NV/V split columns first
            nv_col = f"Avg_EV_S{s}_NV"
            v_col = f"Avg_EV_S{s}_V"
            nv_par_col = f"Avg_Par_S{s}_NV"
            v_par_col = f"Avg_Par_S{s}_V"
            if nv_col in ev_row:
                ev_data[nv_col] = ev_row.get(nv_col)
                ev_data[v_col] = ev_row.get(v_col)
                ev_data[nv_par_col] = ev_row.get(nv_par_col)
                ev_data[v_par_col] = ev_row.get(v_par_col)
            else:
                # Fall back to old aggregate column (for backwards compatibility)
                ev_data[f"Avg_EV_S{s}"] = ev_row.get(f"Avg_EV_S{s}")
                ev_data[f"Avg_Par_S{s}"] = ev_row.get(f"Avg_Par_S{s}")
        try:
            out[int(ev_row["bt_index"])] = ev_data
        except Exception:
            continue
    return out


def pick_nv_v(value_nv: float | None, value_v: float | None, *, seat_vul: str | None) -> float | None:
    """Pick NV or V value based on `seat_vul` in {'NV','V'}; otherwise None."""
    if seat_vul not in ("NV", "V"):
        return None
    return value_v if seat_vul == "V" else value_nv


def _seat_direction_map(seat: int) -> dict[str, str]:
    """Return mapping Dealer -> direction for a dealer-relative seat number (1-4)."""
    seat_i = max(1, min(4, int(seat)))
    dirs = ["N", "E", "S", "W"]
    out: dict[str, str] = {}
    for i, d in enumerate(dirs):
        out[d] = dirs[(i + seat_i - 1) % 4]
    return out


def compute_vul_split_par_ev_at_bid(
    matched_df: pl.DataFrame,
    *,
    next_seat: int,
) -> dict[str, Any]:
    """Compute NV/V split counts + ParScore + EV-at-bid (mean/std), bidder-side oriented.

    Mirrors the logic in the API handler:
    - Vulnerability is computed relative to the bidder's side (bidder direction derived from Dealer + seat).
    - ParScore is NS-oriented in the dataset; we flip sign for EW bidder so it's from bidder's side.
    - EV-at-bid is computed from `Score` from bidder's side (NS score flipped for EW bidder).
    """
    if matched_df is None or matched_df.is_empty():
        return {
            "nv_count": 0,
            "v_count": 0,
            "avg_par_nv": None,
            "avg_par_v": None,
            "ev_score_nv": None,
            "ev_score_v": None,
            "ev_std_nv": None,
            "ev_std_v": None,
        }

    required = {"Vul", "Dealer"}
    if not required.issubset(set(matched_df.columns)):
        # If Vul/Dealer are missing, we cannot split by NV/V; return unknown splits.
        return {
            "nv_count": 0,
            "v_count": 0,
            "avg_par_nv": None,
            "avg_par_v": None,
            "ev_score_nv": None,
            "ev_score_v": None,
            "ev_std_nv": None,
            "ev_std_v": None,
        }

    dealer_to_bidder = _seat_direction_map(int(next_seat))
    bidder_expr = pl.col("Dealer").replace(dealer_to_bidder).alias("_Bidder")

    is_bidder_ns = pl.col("_Bidder").is_in(["N", "S"])
    is_vul_expr = (
        pl.when(is_bidder_ns)
        .then(pl.col("Vul").is_in(["N_S", "Both"]))
        .otherwise(pl.col("Vul").is_in(["E_W", "Both"]))
    )

    df0 = matched_df.with_columns(bidder_expr)
    nv_df = df0.filter(~is_vul_expr)
    v_df = df0.filter(is_vul_expr)

    def _par_sum_for_bidder(df: pl.DataFrame) -> float:
        if "ParScore" not in df.columns or df.is_empty():
            return 0.0
        par_for_bidder = (
            df.with_columns(
                pl.when(pl.col("_Bidder").is_in(["N", "S"]))
                .then(pl.col("ParScore"))
                .otherwise(-pl.col("ParScore"))
                .alias("_ParForBidder")
            )["_ParForBidder"]
            .drop_nulls()
        )
        if par_for_bidder.len() == 0:
            return 0.0
        s = par_for_bidder.sum()
        return float(s) if s is not None else 0.0

    def _score_sums_for_bidder(df: pl.DataFrame) -> tuple[float, float, int]:
        if "Score" not in df.columns or df.is_empty():
            return (0.0, 0.0, 0)
        score_for_bidder = (
            df.with_columns(
                pl.when(pl.col("_Bidder").is_in(["N", "S"]))
                .then(pl.col("Score").cast(pl.Float64, strict=False))
                .otherwise(-pl.col("Score").cast(pl.Float64, strict=False))
                .alias("_ScoreForBidder")
            )["_ScoreForBidder"]
            .drop_nulls()
        )
        n = int(score_for_bidder.len())
        if n == 0:
            return (0.0, 0.0, 0)
        s = float(score_for_bidder.sum() or 0.0)
        ss = float(((score_for_bidder * score_for_bidder).sum()) or 0.0)
        return (s, ss, n)

    nv_count = int(nv_df.height)
    v_count = int(v_df.height)

    nv_par_sum = _par_sum_for_bidder(nv_df)
    v_par_sum = _par_sum_for_bidder(v_df)

    ev_nv_sum, ev_nv_sum_sq, ev_nv_n = _score_sums_for_bidder(nv_df)
    ev_v_sum, ev_v_sum_sq, ev_v_n = _score_sums_for_bidder(v_df)

    avg_par_nv = round(nv_par_sum / nv_count, 0) if nv_count > 0 else None
    avg_par_v = round(v_par_sum / v_count, 0) if v_count > 0 else None

    ev_score_nv = round(ev_nv_sum / ev_nv_n, 1) if ev_nv_n > 0 else None
    if ev_nv_n > 1:
        var = (ev_nv_sum_sq - (ev_nv_sum * ev_nv_sum) / ev_nv_n) / (ev_nv_n - 1)
        ev_std_nv = round(var**0.5, 1) if var >= 0 else None
    else:
        ev_std_nv = None

    ev_score_v = round(ev_v_sum / ev_v_n, 1) if ev_v_n > 0 else None
    if ev_v_n > 1:
        var = (ev_v_sum_sq - (ev_v_sum * ev_v_sum) / ev_v_n) / (ev_v_n - 1)
        ev_std_v = round(var**0.5, 1) if var >= 0 else None
    else:
        ev_std_v = None

    return {
        "nv_count": nv_count,
        "v_count": v_count,
        "avg_par_nv": avg_par_nv,
        "avg_par_v": avg_par_v,
        "ev_score_nv": ev_score_nv,
        "ev_score_v": ev_score_v,
        "ev_std_nv": ev_std_nv,
        "ev_std_v": ev_std_v,
    }


def compute_ev_all_combos_for_matched_deals(matched_df: pl.DataFrame) -> dict[str, Optional[float]]:
    """Compute EV and Makes% for all (level, strain, vul_state, seat) combinations.

    Output schema is stable: returns 560 keys, initialized to None when unavailable.

    Naming convention matches existing API/UI:
      - EV_Score_{level}{strain}_{vul_state}_S{seat}
      - Makes_Pct_{level}{strain}_{vul_state}_S{seat}

    Notes:
    - S1..S4 are seats relative to dealer (Seat 1 = Dealer), not fixed N/E/S/W.
    - Declarer direction is derived per-deal from (Dealer + Seat).
    - EV columns are expected in the wide deal DF as EV_{pair}_{decl}_{strain}_{level}_{vul_state}
    - Makes% is computed from DD_Score_{level}{strain}_{decl} >= 0 (made contract).
    """
    out: dict[str, Optional[float]] = {}
    for level in range(1, 8):
        for strain in ["C", "D", "H", "S", "N"]:
            for vul_state in ["NV", "V"]:
                for seat in [1, 2, 3, 4]:
                    out[f"EV_Score_{level}{strain}_{vul_state}_S{seat}"] = None
                    out[f"Makes_Pct_{level}{strain}_{vul_state}_S{seat}"] = None

    if matched_df is None or matched_df.is_empty():
        return out

    if "Dealer" not in matched_df.columns or "Vul" not in matched_df.columns:
        return out

    for seat in [1, 2, 3, 4]:
        dealer_to_decl = _seat_direction_map(seat)
        seat_df0 = matched_df.with_columns(pl.col("Dealer").replace(dealer_to_decl).alias("_DeclDir"))

        is_decl_ns = pl.col("_DeclDir").is_in(["N", "S"])
        is_vul_expr = (
            pl.when(is_decl_ns)
            .then(pl.col("Vul").is_in(["N_S", "Both"]))
            .otherwise(pl.col("Vul").is_in(["E_W", "Both"]))
        )

        for vul_state in ["NV", "V"]:
            seat_df = seat_df0.filter(is_vul_expr if vul_state == "V" else ~is_vul_expr)
            if seat_df.is_empty():
                continue

            cols = set(seat_df.columns)
            exprs: list[pl.Expr] = []

            for level in range(1, 8):
                for strain in ["C", "D", "H", "S", "N"]:
                    ev_key = f"EV_Score_{level}{strain}_{vul_state}_S{seat}"
                    makes_key = f"Makes_Pct_{level}{strain}_{vul_state}_S{seat}"

                    # EV: pick correct EV column per-deal based on _DeclDir, then mean
                    w_ev = None
                    for d in ["N", "E", "S", "W"]:
                        pair = "NS" if d in ["N", "S"] else "EW"
                        col = f"EV_{pair}_{d}_{strain}_{level}_{vul_state}"
                        if col in cols:
                            if w_ev is None:
                                w_ev = pl.when(pl.col("_DeclDir") == d).then(pl.col(col))
                            else:
                                w_ev = w_ev.when(pl.col("_DeclDir") == d).then(pl.col(col))

                    # Makes %: pick correct DD_Score per-deal based on _DeclDir, then mean(made)*100
                    w_dd = None
                    for d in ["N", "E", "S", "W"]:
                        col = f"DD_Score_{level}{strain}_{d}"
                        if col in cols:
                            if w_dd is None:
                                w_dd = pl.when(pl.col("_DeclDir") == d).then(pl.col(col))
                            else:
                                w_dd = w_dd.when(pl.col("_DeclDir") == d).then(pl.col(col))
                    if w_dd is not None:
                        dd_sel = w_dd.otherwise(None).cast(pl.Float64, strict=False)
                        # If wide EV columns are not available, use mean(DD_Score) as a proxy EV score
                        # so the Streamlit "Contract EV Rankings" table still renders.
                        if w_ev is not None:
                            exprs.append(w_ev.otherwise(None).cast(pl.Float64, strict=False).mean().alias(ev_key))
                        else:
                            exprs.append(dd_sel.mean().alias(ev_key))
                        exprs.append(((dd_sel >= 0).mean() * 100).alias(makes_key))
                    else:
                        # EV columns may exist even when DD columns are missing.
                        if w_ev is not None:
                            exprs.append(w_ev.otherwise(None).cast(pl.Float64, strict=False).mean().alias(ev_key))

            if not exprs:
                continue

            row = seat_df.select(exprs).row(0, named=True)
            for k, v in row.items():
                if v is None:
                    continue
                try:
                    out[k] = round(float(v), 1)
                except Exception:
                    continue

    return out


def phase1_rank_candidates_from_edges(
    edges: list[tuple[str, int]],
    ev_df: pl.DataFrame,
    *,
    count_col: str,
    avg_par_col: str,
    avg_ev_col: str,
) -> pl.DataFrame:
    """
    Build and rank a Phase-1 candidate table from next-bid edges + a precomputed EV/Par stats DF.

    Args:
      edges: list of (bid, child_bt_index)
      ev_df: dataframe containing bt_index + precomputed stats columns
      count_col: column name like Count_S{seat}_{NV|V}
      avg_par_col: column name like Avg_Par_S{seat}_{NV|V}
      avg_ev_col: column name like Avg_EV_S{seat}_{NV|V}
    """
    for col in ("bt_index", count_col, avg_par_col, avg_ev_col):
        if col not in ev_df.columns:
            raise ValueError(
                f"EV/Par stats missing required column: {col}\n"
                f"Available columns include: {ev_df.columns[:40]}"
            )

    candidates_df = pl.DataFrame({"bid": [b for (b, _) in edges], "child_bt_index": [c for (_, c) in edges]})

    stats_sel = ev_df.select(
        [
            pl.col("bt_index").alias("child_bt_index"),
            pl.col(count_col).alias("deal_count"),
            pl.col(avg_par_col).alias("avg_par"),
            pl.col(avg_ev_col).alias("avg_ev"),
        ]
    )
    candidates_df = candidates_df.join(stats_sel, on="child_bt_index", how="left").with_columns(
        pl.when(pl.col("avg_par").is_null()).then(pl.lit(True)).otherwise(pl.lit(False)).alias("missing_stats")
    )
    return candidates_df.sort(by=[pl.col("missing_stats"), pl.col("avg_par")], descending=[False, True], nulls_last=True)

