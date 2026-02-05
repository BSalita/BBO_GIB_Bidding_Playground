from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import polars as pl


_CONTRACT_RE = re.compile(
    r"^\s*(?P<level>[1-7])\s*(?P<strain>N|NT|S|H|D|C)\s*(?P<decl>[NESW])\s*$",
    flags=re.IGNORECASE,
)
_PAIR_RE = re.compile(r"^\s*(NS|EW)\s*$", flags=re.IGNORECASE)
_HAND_RANKS_RE = re.compile(r"^[AKQJT98765432]*$")


def _entropy_nats(probs: Iterable[float]) -> float:
    h = 0.0
    for p in probs:
        if p <= 0:
            continue
        h -= p * math.log(p)
    return float(h)


def _canonical_contract_key(x: Any) -> str:
    """Canonicalize a ParContracts label into a stable key.

    Supported formats in this repo:
    - string: "3N E", "4S N" (declarer direction)
    - dict-like: {"Level": "3", "Strain": "C", "Pair_Direction": "EW", ...} (pair direction)
    - string containing a Python dict repr of the above

    Output key forms:
    - declarer known: "<level><strain> <decl>" (e.g., "3N E")
    - declarer unknown but pair known: "<level><strain> <pair>" (e.g., "3C EW")
    """
    if isinstance(x, dict):
        d = x
    else:
        s0 = str(x).strip()
        if not s0:
            raise ValueError(f"Unrecognized ParContracts label: {x!r}")

        # Parse python dict repr (common when stored as Utf8)
        if s0.startswith("{") and s0.endswith("}"):
            try:
                parsed = ast.literal_eval(s0)
            except Exception:
                parsed = None
            d = parsed if isinstance(parsed, dict) else None
        else:
            d = None

        if d is None:
            s = s0.upper().replace("NT", "N")
            m = _CONTRACT_RE.match(s)
            if not m:
                raise ValueError(f"Unrecognized ParContracts label: {x!r}")
            level = m.group("level")
            strain = m.group("strain").upper().replace("NT", "N")
            decl = m.group("decl").upper()
            return f"{level}{strain} {decl}"

    level_raw = d.get("Level") if isinstance(d, dict) else None
    strain_raw = d.get("Strain") if isinstance(d, dict) else None
    decl_raw = d.get("Declarer_Direction") or d.get("Declarer") if isinstance(d, dict) else None
    pair_raw = d.get("Pair_Direction") if isinstance(d, dict) else None

    if level_raw is None or strain_raw is None:
        raise ValueError(f"Unrecognized ParContracts label: {x!r}")

    level = int(str(level_raw).strip())
    strain = str(strain_raw).strip().upper().replace("NT", "N")

    decl: str | None = None
    if decl_raw is not None:
        cand = str(decl_raw).strip().upper()
        if cand in {"N", "E", "S", "W"}:
            decl = cand
    if decl is None and pair_raw is not None:
        cand = str(pair_raw).strip().upper()
        if _PAIR_RE.match(cand):
            decl = cand  # store as NS/EW when declarer direction is not known

    if decl is None:
        raise ValueError(f"Unrecognized ParContracts label (no decl/pair): {x!r}")

    return f"{level}{strain} {decl}"


def _flatten_par_contracts(val: Any) -> list[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return [x for x in val if x is not None and str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, dict):
        return [val]
    raise TypeError(f"Unsupported ParContracts value type: {type(val)} ({val!r})")


def _hist_from_ints(values: list[int], *, min_v: int, max_v: int) -> dict[str, int]:
    out = {str(i): 0 for i in range(min_v, max_v + 1)}
    for v in values:
        if v < min_v or v > max_v:
            continue
        out[str(int(v))] += 1
    return out


def _to_int_list(series: pl.Series) -> list[int]:
    return [int(x) for x in series.to_list() if x is not None]


def _to_float_list(series: pl.Series) -> list[float]:
    return [float(x) for x in series.to_list() if x is not None]


def _percentile_rank_0_100(sample: list[float], value: float) -> float:
    """Mid-rank percentile in [0, 100]."""
    if not sample:
        return float("nan")
    lt = sum(1 for x in sample if x < value)
    eq = sum(1 for x in sample if x == value)
    return 100.0 * (lt + 0.5 * eq) / len(sample)


def _quantile(sorted_x: list[float], q: float) -> float:
    if not sorted_x:
        return float("nan")
    if q <= 0:
        return float(sorted_x[0])
    if q >= 1:
        return float(sorted_x[-1])
    pos = (len(sorted_x) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_x[lo])
    w = pos - lo
    return float(sorted_x[lo] * (1 - w) + sorted_x[hi] * w)


def _parse_hand_suits(hand: str) -> dict[str, str]:
    """Parse a hand string like 'AKQ2.KQ2.AK2.AK2' into suit->ranks."""
    parts = str(hand).strip().upper().split(".")
    if len(parts) != 4:
        raise ValueError(f"Bad hand string (expected 4 suits): {hand!r}")
    suits = {"S": parts[0], "H": parts[1], "D": parts[2], "C": parts[3]}
    for k, v in list(suits.items()):
        if v in {"", "-"}:
            suits[k] = ""
        else:
            if not _HAND_RANKS_RE.match(v):
                raise ValueError(f"Bad ranks for suit {k} in hand {hand!r}: {v!r}")
    return suits


def _hand_contains_card(hand: str, suit: str, rank: str) -> bool:
    suits = _parse_hand_suits(hand)
    return rank.upper() in suits[suit.upper()]


def _count_aces(hand: str) -> int:
    suits = _parse_hand_suits(hand)
    return int(sum(1 for su in ["S", "H", "D", "C"] if "A" in suits[su]))


def _count_keycards(hand: str, trump_suit: str) -> int:
    """Keycards = 4 aces + trump king (0..5) for the given trump suit."""
    suits = _parse_hand_suits(hand)
    aces = _count_aces(hand)
    trump_k = 1 if ("K" in suits[str(trump_suit).upper()]) else 0
    return int(aces + trump_k)


def _dir_for_seat(dealer: str, seat: int) -> str:
    """Return direction for a dealer-relative seat number (1-4)."""
    dirs = ["N", "E", "S", "W"]
    d = str(dealer or "N").strip().upper()
    try:
        di = dirs.index(d)
    except ValueError:
        di = 0
    s = max(1, min(4, int(seat)))
    return dirs[(di + s - 1) % 4]


def _role_dirs_for_deal(dealer: str, next_seat: int) -> dict[str, str]:
    """Compute role directions (SELF/PARTNER/LHO/RHO) for a deal row."""
    s = int(next_seat)
    # Dealer-relative seat numbers
    seat_self = s
    seat_lho = (s % 4) + 1
    seat_partner = ((s + 1) % 4) + 1
    seat_rho = ((s + 2) % 4) + 1
    return {
        "SELF": _dir_for_seat(dealer, seat_self),
        "LHO": _dir_for_seat(dealer, seat_lho),
        "PARTNER": _dir_for_seat(dealer, seat_partner),
        "RHO": _dir_for_seat(dealer, seat_rho),
    }


@dataclass(frozen=True)
class BidDetailsConfig:
    topk: int = 10
    include_phase2a: bool = True
    include_honor_location: bool = False


def compute_bid_details_from_sample(
    deals_sample_df: pl.DataFrame,
    *,
    seat_dir: str,
    pinned_row: Optional[dict[str, Any]] = None,
    cfg: BidDetailsConfig = BidDetailsConfig(),
) -> dict[str, Any]:
    """Compute stable selected-bid details from a sampled matched-deals DataFrame.

    Assumptions:
    - `deals_sample_df` has already had the pinned deal excluded if `pinned_row` is provided.
    - Par contract distribution is computed from `ParContracts` with 1/len(labels) mass split per deal.
    - Percentiles are computed vs the sampled matched set (after pinned exclusion).
    """
    seat_dir_u = str(seat_dir or "").strip().upper()
    if seat_dir_u not in {"N", "E", "S", "W"}:
        raise ValueError(f"seat_dir must be one of N/E/S/W, got: {seat_dir!r}")

    required = {"index", "ParScore", "ParContracts"}
    missing = [c for c in required if c not in deals_sample_df.columns]
    if missing:
        raise ValueError(f"Matched deals missing required columns: {missing}")

    par_scores = _to_float_list(deals_sample_df.get_column("ParScore"))
    if not par_scores:
        raise ValueError("ParScore contains no numeric values in the matched sample.")

    par_scores_sorted = sorted(par_scores)
    n = len(par_scores_sorted)
    mean = float(sum(par_scores_sorted) / n)
    median = _quantile(par_scores_sorted, 0.5)
    p10 = _quantile(par_scores_sorted, 0.10)
    p25 = _quantile(par_scores_sorted, 0.25)
    p75 = _quantile(par_scores_sorted, 0.75)
    p90 = _quantile(par_scores_sorted, 0.90)
    mn = float(par_scores_sorted[0])
    mx = float(par_scores_sorted[-1])
    # Sample std (ddof=0)
    var = float(sum((x - mean) ** 2 for x in par_scores_sorted) / max(1, n))
    std = float(math.sqrt(var))
    p_par_ge_0 = float(sum(1 for x in par_scores_sorted if x >= 0.0) / n)
    neg = [x for x in par_scores_sorted if x < 0.0]
    tail_risk_mean_neg = float((-sum(neg) / len(neg))) if neg else 0.0  # magnitude of negative tail

    # Par contract distribution (weighted)
    contract_counts: Counter[str] = Counter()
    contract_par_score_mass: dict[str, float] = {}
    contract_weight_mass: dict[str, float] = {}
    distinct_pairs: set[str] = set()

    par_contracts_col = deals_sample_df.get_column("ParContracts").to_list()
    par_score_col = deals_sample_df.get_column("ParScore").to_list()

    for raw_contracts, ps in zip(par_contracts_col, par_score_col, strict=False):
        labels = _flatten_par_contracts(raw_contracts)
        if not labels:
            continue
        try:
            ps_f = float(ps) if ps is not None else None
        except Exception:
            ps_f = None
        w = 1.0 / len(labels)
        for lab in labels:
            key = _canonical_contract_key(lab)
            contract_counts[key] += w  # type: ignore[arg-type]
            if ps_f is not None:
                contract_par_score_mass[key] = contract_par_score_mass.get(key, 0.0) + (w * ps_f)
                contract_weight_mass[key] = contract_weight_mass.get(key, 0.0) + w
            # Track pair-level entropy distincts
            try:
                side = key.split(" ", 1)[1].strip().upper()
                if side in {"NS", "EW"}:
                    distinct_pairs.add(f"{key.split(' ', 1)[0]} {side}")
                elif side in {"N", "E", "S", "W"}:
                    distinct_pairs.add(f"{key.split(' ', 1)[0]} {'NS' if side in {'N','S'} else 'EW'}")
            except Exception:
                pass

    total_mass = float(sum(contract_counts.values()))
    if total_mass <= 0:
        raise ValueError("ParContracts produced no usable labels in the matched sample.")

    probs_full = [float(w / total_mass) for w in contract_counts.values()]
    entropy_full = _entropy_nats(probs_full)

    # Pair-level distribution
    pair_counts: Counter[str] = Counter()
    for k, w in contract_counts.items():
        try:
            lhs, rhs = k.split(" ", 1)
            rhs_u = rhs.strip().upper()
            if rhs_u in {"NS", "EW"}:
                pair_key = f"{lhs} {rhs_u}"
            else:
                pair_key = f"{lhs} {'NS' if rhs_u in {'N','S'} else 'EW'}"
            pair_counts[pair_key] += float(w)
        except Exception:
            continue
    pair_total = float(sum(pair_counts.values())) or 1.0
    probs_pair = [float(w / pair_total) for w in pair_counts.values()]
    entropy_pair = _entropy_nats(probs_pair)

    top = contract_counts.most_common(int(cfg.topk))
    topk_out: list[dict[str, Any]] = []
    for key, w in top:
        lhs, rhs = key.split(" ", 1)
        rhs_u = rhs.strip().upper()
        pair = rhs_u if rhs_u in {"NS", "EW"} else ("NS" if rhs_u in {"N", "S"} else "EW")
        declarer = rhs_u if rhs_u in {"N", "E", "S", "W"} else None
        avg_par_score = None
        if key in contract_par_score_mass and contract_weight_mass.get(key, 0.0) > 0:
            avg_par_score = float(contract_par_score_mass[key] / contract_weight_mass[key])
        topk_out.append(
            {
                "contract": lhs.replace("N", "NT") if lhs.endswith("N") else lhs,
                "pair": pair,
                "declarer": declarer,
                "prob": float(w / total_mass),
                "mass": float(w),
                "avg_par_score": avg_par_score,
            }
        )

    out: dict[str, Any] = {
        "par_score": {
            "mean": mean,
            "median": median,
            "std": std,
            "p10": p10,
            "p25": p25,
            "p75": p75,
            "p90": p90,
            "iqr": float(p75 - p25),
            "min": mn,
            "max": mx,
            "p_par_ge_0": p_par_ge_0,
            "tail_risk_mean_neg": tail_risk_mean_neg,
        },
        "par_contracts": {
            "topk": topk_out,
            "contract_entropy_full_nats": float(entropy_full),
            "contract_entropy_pair_nats": float(entropy_pair),
            "distinct_contracts": int(len(contract_counts)),
            "distinct_pairs": int(len(distinct_pairs)),
        },
    }

    # NOTE: Range percentiles and auction-conditioned phase2a posteriors are computed
    # in `compute_phase2a_auction_conditioned_posteriors()`, because they must be
    # rotated into SELF/PARTNER/LHO/RHO roles per deal (dealer varies across matches).

    if cfg.include_phase2a:
        # Keep legacy top-level histograms (absolute dirs) for backwards-compat,
        # but callers SHOULD prefer the auction-conditioned posteriors.
        try:
            need_cols = []
            for d in ["N", "E", "S", "W"]:
                need_cols.append(f"HCP_{d}")
                for su in ["S", "H", "D", "C"]:
                    need_cols.append(f"SL_{d}_{su}")
            missing2 = [c for c in need_cols if c not in deals_sample_df.columns]
            if not missing2:
                sl_hist: dict[str, Any] = {}
                for d in ["N", "E", "S", "W"]:
                    sl_hist[d] = {}
                    for su in ["S", "H", "D", "C"]:
                        col = f"SL_{d}_{su}"
                        sl_hist[d][su] = _hist_from_ints(_to_int_list(deals_sample_df.get_column(col)), min_v=0, max_v=13)

                fit_hist: dict[str, Any] = {"NS": {}, "EW": {}}
                for su in ["S", "H", "D", "C"]:
                    ns_vals = deals_sample_df.select((pl.col(f"SL_N_{su}") + pl.col(f"SL_S_{su}")).alias("x"))["x"]
                    ew_vals = deals_sample_df.select((pl.col(f"SL_E_{su}") + pl.col(f"SL_W_{su}")).alias("x"))["x"]
                    fit_hist["NS"][su] = _hist_from_ints(_to_int_list(ns_vals), min_v=0, max_v=26)
                    fit_hist["EW"][su] = _hist_from_ints(_to_int_list(ew_vals), min_v=0, max_v=26)

                hcp_hist: dict[str, Any] = {}
                for d in ["N", "E", "S", "W"]:
                    col = f"HCP_{d}"
                    hcp_hist[d] = _hist_from_ints(_to_int_list(deals_sample_df.get_column(col)), min_v=0, max_v=37)

                out.update(
                    {
                        "hcp_histograms": hcp_hist,
                        "suit_length_histograms": sl_hist,
                        "fit_histograms": fit_hist,
                    }
                )
        except Exception:
            # Caller can still request auction-conditioned posteriors separately.
            pass

    return out


def compute_phase2a_auction_conditioned_posteriors(
    deals_sample_df: pl.DataFrame,
    *,
    next_seat: int,
    pinned_deal_index: Optional[int] = None,
    include_keycards: bool = True,
    include_onside: bool = True,
) -> dict[str, Any]:
    """Phase 2a: auction-conditioned posteriors in SELF/PARTNER/LHO/RHO roles.

    This rotates each deal using its own Dealer value, so the posteriors are truly
    conditioned on "who is to act" at this auction step (dealer-relative seat).
    """
    if "Dealer" not in deals_sample_df.columns:
        raise ValueError("Phase2a requires Dealer column")

    # Required numeric columns
    need_cols = []
    for d in ["N", "E", "S", "W"]:
        need_cols.append(f"HCP_{d}")
        need_cols.append(f"Total_Points_{d}")
        for su in ["S", "H", "D", "C"]:
            need_cols.append(f"SL_{d}_{su}")

    missing = [c for c in need_cols if c not in deals_sample_df.columns]
    if missing:
        raise ValueError(f"Phase2a missing required columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    include_hands = include_keycards or include_onside
    hand_cols = [f"Hand_{d}" for d in ["N", "E", "S", "W"]] if include_hands else []
    missing_h = [c for c in hand_cols if c not in deals_sample_df.columns]
    if missing_h:
        raise ValueError(f"Phase2a honor/keycard summaries require hand columns: {missing_h}")

    # Collect per-role samples
    roles = ["SELF", "PARTNER", "LHO", "RHO"]
    hcp_vals: dict[str, list[int]] = {r: [] for r in roles}
    tp_vals: dict[str, list[int]] = {r: [] for r in roles}
    sl_vals: dict[str, dict[str, list[int]]] = {r: {su: [] for su in ["S", "H", "D", "C"]} for r in roles}

    # Fit distributions for OUR side (SELF+PARTNER) and THEM (LHO+RHO)
    fit_us: dict[str, list[int]] = {su: [] for su in ["S", "H", "D", "C"]}
    fit_them: dict[str, list[int]] = {su: [] for su in ["S", "H", "D", "C"]}

    # Keycards distributions per suit
    keycards_self: dict[str, Counter[int]] = {su: Counter() for su in ["S", "H", "D", "C"]}
    keycards_us: dict[str, Counter[int]] = {su: Counter() for su in ["S", "H", "D", "C"]}

    # Onside: honor location probabilities for LHO/RHO (relative to SELF)
    honor_counts: dict[str, Counter[str]] = {}  # key like "QS" -> Counter(role)
    tracked_honors = [("A", "S"), ("K", "S"), ("Q", "S"),
                      ("A", "H"), ("K", "H"), ("Q", "H"),
                      ("A", "D"), ("K", "D"), ("Q", "D"),
                      ("A", "C"), ("K", "C"), ("Q", "C")]
    if include_onside:
        for rnk, su in tracked_honors:
            honor_counts[f"{rnk}{su}"] = Counter()

    pinned_role_row: dict[str, Any] | None = None

    # Iterate rows (N up to ~5k default; this is OK and keeps logic explicit/robust).
    for row in deals_sample_df.iter_rows(named=True):
        dealer = str(row.get("Dealer", "N")).upper()
        rdirs = _role_dirs_for_deal(dealer, int(next_seat))
        # Collect role numeric samples
        for r in roles:
            d = rdirs[r]
            try:
                hcp_vals[r].append(int(row.get(f"HCP_{d}")))
            except Exception:
                pass
            try:
                tp_vals[r].append(int(row.get(f"Total_Points_{d}")))
            except Exception:
                pass
            for su in ["S", "H", "D", "C"]:
                try:
                    sl_vals[r][su].append(int(row.get(f"SL_{d}_{su}")))
                except Exception:
                    pass

        # Fit lengths
        try:
            sd = rdirs["SELF"]
            pd = rdirs["PARTNER"]
            ld = rdirs["LHO"]
            rd = rdirs["RHO"]
            for su in ["S", "H", "D", "C"]:
                fit_us[su].append(int(row.get(f"SL_{sd}_{su}") or 0) + int(row.get(f"SL_{pd}_{su}") or 0))
                fit_them[su].append(int(row.get(f"SL_{ld}_{su}") or 0) + int(row.get(f"SL_{rd}_{su}") or 0))
        except Exception:
            pass

        if pinned_deal_index is not None:
            try:
                if int(row.get("index")) == int(pinned_deal_index):
                    pinned_role_row = row
            except Exception:
                pass

        if include_hands:
            try:
                hands = {d: str(row.get(f"Hand_{d}")) for d in ["N", "E", "S", "W"]}
            except Exception:
                hands = {}

        if include_keycards and include_hands and hands:
            try:
                h_self = hands.get(rdirs["SELF"])
                h_partner = hands.get(rdirs["PARTNER"])
                if h_self and h_partner:
                    for su in ["S", "H", "D", "C"]:
                        kc_self = _count_keycards(h_self, su)
                        kc_us = _count_keycards(h_self, su) + _count_keycards(h_partner, su)
                        # Clamp US keycards to 0..10 for safety (should be 0..10)
                        kc_us = max(0, min(10, kc_us))
                        keycards_self[su][kc_self] += 1
                        keycards_us[su][kc_us] += 1
            except Exception:
                pass

        if include_onside and include_hands and hands:
            try:
                # Map hands into roles (SELF/PARTNER/LHO/RHO)
                role_to_hand = {
                    "SELF": hands.get(rdirs["SELF"]),
                    "PARTNER": hands.get(rdirs["PARTNER"]),
                    "LHO": hands.get(rdirs["LHO"]),
                    "RHO": hands.get(rdirs["RHO"]),
                }
                for rnk, su in tracked_honors:
                    card_key = f"{rnk}{su}"
                    found_role = None
                    for role_name, h in role_to_hand.items():
                        if not h:
                            continue
                        if _hand_contains_card(h, su, rnk):
                            found_role = role_name
                            break
                    if found_role is not None:
                        honor_counts[card_key][found_role] += 1
            except Exception:
                pass

    # Build histograms (role-conditioned)
    out_roles: dict[str, Any] = {}
    for r in roles:
        out_roles[r.lower()] = {
            "hcp_hist": _hist_from_ints(hcp_vals[r], min_v=0, max_v=37),
            "total_points_hist": _hist_from_ints(tp_vals[r], min_v=0, max_v=40),
            "sl_hist": {su: _hist_from_ints(sl_vals[r][su], min_v=0, max_v=13) for su in ["S", "H", "D", "C"]},
        }

    out_fit = {
        "us": {su: _hist_from_ints(fit_us[su], min_v=0, max_v=26) for su in ["S", "H", "D", "C"]},
        "them": {su: _hist_from_ints(fit_them[su], min_v=0, max_v=26) for su in ["S", "H", "D", "C"]},
    }

    def _prob_ge(hist: dict[str, int], k: int) -> float:
        try:
            total = float(sum(hist.values()))
            if total <= 0:
                return float("nan")
            num = float(sum(v for kk, v in hist.items() if int(kk) >= int(k)))
            return float(num / total)
        except Exception:
            return float("nan")

    def _exp_from_hist(hist: dict[str, int]) -> float:
        try:
            total = float(sum(hist.values()))
            if total <= 0:
                return float("nan")
            s = 0.0
            for kk, v in hist.items():
                s += float(int(kk)) * float(v)
            return float(s / total)
        except Exception:
            return float("nan")

    threat: dict[str, Any] = {}
    for su in ["S", "H", "D", "C"]:
        h_us = out_fit["us"][su]
        h_them = out_fit["them"][su]
        threat[su] = {
            "p_us_8plus": _prob_ge(h_us, 8),
            "p_them_5plus": _prob_ge(h_them, 5),
            "p_them_6plus": _prob_ge(h_them, 6),
            "p_them_7plus": _prob_ge(h_them, 7),
            "e_fit_us": _exp_from_hist(h_us),
            "e_fit_them": _exp_from_hist(h_them),
        }

    keycards_out = None
    if include_keycards:
        keycards_out = {
            "self": {su: {str(k): int(v) for k, v in keycards_self[su].items()} for su in ["S", "H", "D", "C"]},
            "us": {su: {str(k): int(v) for k, v in keycards_us[su].items()} for su in ["S", "H", "D", "C"]},
        }

    onside_out = None
    if include_onside:
        onside_out = {}
        for su in ["S", "H", "D", "C"]:
            onside_out[su] = {}
            for rnk in ["A", "K", "Q"]:
                key = f"{rnk}{su}"
                c = honor_counts.get(key, Counter())
                tot = float(sum(c.values()))
                if tot <= 0:
                    onside_out[su][rnk] = {"p_lho": float("nan"), "p_rho": float("nan"), "p_partner": float("nan"), "p_self": float("nan")}
                else:
                    onside_out[su][rnk] = {
                        "p_lho": float(c.get("LHO", 0) / tot),
                        "p_rho": float(c.get("RHO", 0) / tot),
                        "p_partner": float(c.get("PARTNER", 0) / tot),
                        "p_self": float(c.get("SELF", 0) / tot),
                        # Convention: "onside" = with LHO, "offside" = with RHO (relative to SELF).
                        "p_onside": float(c.get("LHO", 0) / tot),
                        "p_offside": float(c.get("RHO", 0) / tot),
                    }

    range_percentiles = None
    if pinned_deal_index is not None and pinned_role_row is not None:
        # Percentiles are vs SELF distribution (auction-conditioned) for consistency.
        try:
            self_hcp = out_roles["self"]["hcp_hist"]
            # Reconstruct samples cheaply from stored lists (we still have hcp_vals/ sl_vals).
            sample_hcp = [float(x) for x in hcp_vals["SELF"]]
            sample_tp = [float(x) for x in tp_vals["SELF"]]
            dealer = str(pinned_role_row.get("Dealer", "N")).upper()
            rdirs = _role_dirs_for_deal(dealer, int(next_seat))
            self_dir = rdirs["SELF"]
            hcp_v = float(pinned_role_row.get(f"HCP_{self_dir}"))
            tp_v = float(pinned_role_row.get(f"Total_Points_{self_dir}"))
            sl_pct: dict[str, Any] = {}
            for su in ["S", "H", "D", "C"]:
                sample_sl = [float(x) for x in sl_vals["SELF"][su]]
                v = float(pinned_role_row.get(f"SL_{self_dir}_{su}"))
                sl_pct[su] = {"value": v, "percentile": _percentile_rank_0_100(sample_sl, v)}
            range_percentiles = {
                "deal_index": int(pinned_deal_index),
                "role": "self",
                "hcp": {"value": hcp_v, "percentile": _percentile_rank_0_100(sample_hcp, hcp_v)},
                "total_points": {"value": tp_v, "percentile": _percentile_rank_0_100(sample_tp, tp_v)},
                "suit_lengths": sl_pct,
            }
        except Exception:
            range_percentiles = None

    return {
        "roles": out_roles,
        "fit": out_fit,
        "threats": threat,
        "keycards": keycards_out,
        "onside": onside_out,
        "range_percentiles": range_percentiles,
    }

