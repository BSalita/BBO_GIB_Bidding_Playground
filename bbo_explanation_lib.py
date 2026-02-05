from __future__ import annotations

import re
from typing import Any, Optional


def compute_eeo_from_bid_details(bid_details: dict[str, Any]) -> dict[str, Any]:
    """Compute an EEO (evidence-backed explanation object) from bid-details evidence.

    This is intentionally "no-RAG": it only uses fields already computed in the bid-details response.
    """
    par_score = bid_details.get("par_score") or {}
    par_contracts = bid_details.get("par_contracts") or {}

    mean = par_score.get("mean")
    p_par_ge_0 = par_score.get("p_par_ge_0")
    iqr = par_score.get("iqr")
    tail_risk = par_score.get("tail_risk_mean_neg")

    entropy_full = par_contracts.get("contract_entropy_full_nats")
    entropy_pair = par_contracts.get("contract_entropy_pair_nats")
    distinct_contracts = par_contracts.get("distinct_contracts")
    topk = par_contracts.get("topk") or []

    # A simple "utility" proxy for MVP: prefer higher mean ParScore, penalize tail risk + ambiguity.
    # (Weights can be adjusted at the API layer later if needed.)
    lam = 0.5
    mu = 0.25
    utility = None
    try:
        utility = float(mean) - lam * float(tail_risk or 0.0) - mu * float(entropy_full or 0.0)
    except Exception:
        utility = None

    return {
        "pred_par_mean": mean,
        "par_iqr": iqr,
        "p_par_ge_0": p_par_ge_0,
        "tail_risk_mean_neg": tail_risk,
        "ambiguity_entropy_full_nats": entropy_full,
        "ambiguity_entropy_pair_nats": entropy_pair,
        "distinct_contracts": distinct_contracts,
        "par_contract_topk": topk,
        "utility": utility,
        "utility_components": {"lambda_tail_risk": lam, "mu_ambiguity": mu, "info_value": 0.0},
    }


_SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}


def contract_rank(contract: Any) -> Optional[int]:
    """Comparable rank for contracts like '4H' or '3NT'.

    Returns None if unparseable.
    """
    try:
        s = str(contract or "").strip().upper()
        if not s:
            return None
        # Accept e.g. "4H", "3NT", "3N", "4HX", "4HXX" (ignore X/XX).
        m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
        if not m:
            return None
        level = int(m.group(1))
        strain_raw = str(m.group(2)).upper()
        strain = "N" if strain_raw in ("N", "NT") else strain_raw
        o = _SUIT_ORDER.get(strain)
        if o is None:
            return None
        return (level - 1) * 5 + int(o)
    except Exception:
        return None


def min_contract_from_auction(auction_text: Any) -> tuple[Optional[int], Optional[str]]:
    """Return (rank, label) for the last contract seen in an auction string."""
    try:
        toks = [t.strip().upper() for t in str(auction_text or "").split("-") if t and str(t).strip()]
    except Exception:
        toks = []
    last = None
    last_rank = None
    for t in toks:
        r = contract_rank(t)
        if r is None:
            continue
        last = t
        last_rank = r
    return last_rank, last


def render_recommendation_explanation(
    *,
    bid: str,
    eeo: dict[str, Any],
    min_contract_rank: Optional[int] = None,
    min_contract_label: Optional[str] = None,
) -> dict[str, Any]:
    """Template A: basic recommendation explanation (computed evidence only)."""
    mean = eeo.get("pred_par_mean")
    p_ok = eeo.get("p_par_ge_0")
    iqr = eeo.get("par_iqr")
    tail = eeo.get("tail_risk_mean_neg")
    ent = eeo.get("ambiguity_entropy_full_nats")
    topk = eeo.get("par_contract_topk") or []

    topk_str = ""
    try:
        if topk:
            # Filter displayed par contracts to those not below the current auction contract (if provided).
            filtered = []
            if min_contract_rank is None:
                filtered = list(topk)
            else:
                for x in topk:
                    try:
                        c0 = x.get("contract")
                    except Exception:
                        c0 = None
                    r0 = contract_rank(c0)
                    if r0 is None:
                        continue
                    if r0 >= int(min_contract_rank):
                        filtered.append(x)

            parts = []
            for x in filtered[:5]:
                c = x.get("contract")
                prob = x.get("prob")
                if c is None or prob is None:
                    continue
                parts.append(f"{c} ({float(prob)*100:.1f}%)")
            if parts:
                suffix = ""
                if min_contract_rank is not None and min_contract_label:
                    suffix = f" (filtered to ≥ {min_contract_label})"
                topk_str = " Top par contracts" + suffix + ": " + ", ".join(parts) + "."
    except Exception:
        topk_str = ""

    # Keep text stable and short (UI-friendly).
    text = (
        f"Recommend {bid}: mean ParScore={_fmt(mean)}, "
        f"P(ParScore≥0)={_fmt_pct(p_ok)}, IQR={_fmt(iqr)}, "
        f"tail-risk(neg-mean)={_fmt(tail)}, ambiguity(entropy)={_fmt(ent)}."
        f"{topk_str}"
    )

    return {
        "template_id": "A_recommendation_v1",
        "text": text,
        "evidence_fields": [
            "par_score.mean",
            "par_score.p_par_ge_0",
            "par_score.iqr",
            "par_score.tail_risk_mean_neg",
            "par_contracts.contract_entropy_full_nats",
            "par_contracts.topk",
        ],
    }


def render_counterfactual_why_not(
    *,
    bid: str,
    alt_bid: str,
    eeo: dict[str, Any],
    alt_eeo: dict[str, Any],
) -> dict[str, Any]:
    """Template B: 'Why not X?' counterfactual, backed by deltas."""
    dm = _delta(eeo.get("pred_par_mean"), alt_eeo.get("pred_par_mean"))
    dtail = _delta(eeo.get("tail_risk_mean_neg"), alt_eeo.get("tail_risk_mean_neg"))
    dent = _delta(eeo.get("ambiguity_entropy_full_nats"), alt_eeo.get("ambiguity_entropy_full_nats"))
    dpok = _delta(eeo.get("p_par_ge_0"), alt_eeo.get("p_par_ge_0"))

    parts: list[str] = [f"Why {bid} over {alt_bid}:"]

    if dm is not None:
        parts.append(f"mean ParScore Δ={_fmt(dm)}")
    if dpok is not None:
        parts.append(f"P(ParScore≥0) Δ={_fmt_pct(dpok, signed=True)}")
    if dtail is not None:
        # Positive dtail means bid has *higher* tail risk; that's worse.
        parts.append(f"tail-risk Δ={_fmt(dtail)} (lower is better)")
    if dent is not None:
        parts.append(f"ambiguity(entropy) Δ={_fmt(dent)} (lower is better)")

    text = "; ".join(parts) + "."

    return {
        "template_id": "B_why_not_v1",
        "text": text,
        "deltas": {
            "pred_par_mean": dm,
            "p_par_ge_0": dpok,
            "tail_risk_mean_neg": dtail,
            "ambiguity_entropy_full_nats": dent,
        },
        "evidence_fields": [
            "par_score.mean",
            "par_score.p_par_ge_0",
            "par_score.tail_risk_mean_neg",
            "par_contracts.contract_entropy_full_nats",
        ],
    }


def _fmt(x: Any) -> str:
    try:
        if x is None:
            return "NA"
        return f"{float(x):.1f}"
    except Exception:
        return "NA"


def _fmt_pct(x: Any, *, signed: bool = False) -> str:
    try:
        if x is None:
            return "NA"
        v = float(x) * 100.0
        return f"{v:+.1f}%" if signed else f"{v:.1f}%"
    except Exception:
        return "NA"


def _delta(a: Any, b: Any) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        return float(a) - float(b)
    except Exception:
        return None

