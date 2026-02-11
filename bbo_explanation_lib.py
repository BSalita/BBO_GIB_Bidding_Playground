from __future__ import annotations

import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Guardrail: common-sense penalty for bids whose level is inconsistent with
# the available evidence (par contracts, partnership total points, etc.).
#
# Each check produces a (penalty, reason_tag, reason_text) triple.
# Penalties are additive, and the final result is scaled by a master weight.
#
# Guardrail reason tags (exhaustive list):
#   OVERBID_VS_PAR      - Bid level exceeds most likely par contract level
#   TP_SHORTFALL         - Partnership TP below standard minimum for bid level
#   NEG_PAR_HIGH_LEVEL   - Expected acting-par is negative at level 4+
#   STRAIN_INEFFICIENCY  - Minor game (5m = 11 tricks) when 3NT (9 tricks)
#                          is the par contract
#   TRICKS_SHORTFALL     - Estimated tricks below required for contract level
#   UNDERBID_VS_PAR      - Bid level is below par contract owned by our side
#   TP_SURPLUS_UNDERBID  - Partnership TP exceeds threshold for a higher level
#                          that matches par
#   SACRIFICE_DISCOUNT   - (reduction) Overbid penalties reduced because par
#                          contract belongs to opponents (sacrifice context)
# ---------------------------------------------------------------------------

# Minimum partnership Total_Points typically required to make a contract at
# each level.  These are *approximate* bridge heuristics (adapted from the
# Rule of 25 / Losing Trick Count consensus).
_TP_THRESHOLDS: dict[int, float] = {
    1: 20.0,
    2: 23.0,
    3: 25.0,
    4: 26.0,
    5: 29.0,
    6: 33.0,
    7: 37.0,
}

# Tricks required per level (level + 6).
_TRICKS_REQUIRED: dict[int, int] = {i: i + 6 for i in range(1, 8)}


def _parse_bid_level(bid: Any) -> int | None:
    """Return the numeric level (1-7) from a bid string, or None for Pass/X/XX."""
    try:
        s = str(bid or "").strip().upper()
        m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _parse_bid_strain(bid: Any) -> str | None:
    """Return strain from bid string: 'C', 'D', 'H', 'S', or 'NT'.  None for Pass/X/XX."""
    try:
        s = str(bid or "").strip().upper()
        m = re.match(r"^[1-7]\s*(NT|N|[CDHS])", s)
        if m:
            strain = m.group(1).upper()
            return "NT" if strain in ("N", "NT") else strain
    except Exception:
        pass
    return None


def expected_from_hist(hist: dict[str, Any] | None) -> float | None:
    """Compute expected value from a {bucket: count} histogram."""
    if not hist or not isinstance(hist, dict):
        return None
    try:
        total = 0.0
        weighted = 0.0
        for k, v in hist.items():
            cnt = float(v)
            total += cnt
            weighted += float(int(k)) * cnt
        if total <= 0:
            return None
        return weighted / total
    except Exception:
        return None


def compute_guardrail_penalty(
    bid: str,
    par_contracts_topk: list[dict[str, Any]] | None,
    self_total_points: float | None,
    partner_tp_hist: dict[str, Any] | None,
    par_score_mean: float | None,
    acting_sign: float,
    *,
    w_overbid_level: float = 80.0,
    w_tp_shortfall: float = 15.0,
    w_neg_par_high: float = 0.5,
    w_underbid_level: float = 40.0,
    w_tp_surplus: float = 8.0,
    w_strain_ineff: float = 60.0,
    sacrifice_discount: float = 0.7,
    est_tricks: float | None = None,
    w_tricks_shortfall: float = 30.0,
) -> tuple[float, list[str]]:
    """Compute a non-negative guardrail penalty to subtract from the bid score.

    Seven checks plus a sacrifice exemption (each contributes additively):

    **Overbid checks** (bid level is too high for the hand):

    1. **OVERBID_VS_PAR** – bid level exceeds the most likely par contract
       level.  Catches bids like 5C when par says 3NT.
    2. **TP_SHORTFALL** – combined SELF + E[PARTNER] total points fall below
       the standard minimum for the bid level.
    3. **NEG_PAR_HIGH_LEVEL** – at the 4-level or above, the acting-side par
       mean is negative (contract is expected to go down on average).
    4. **STRAIN_INEFFICIENCY** – bidding 5C/5D (11 tricks) when the par
       contract is 3NT (9 tricks for comparable score).
    5. **TRICKS_SHORTFALL** – estimated partnership tricks fall below the
       number required for the contract level (level + 6).

    **Underbid checks** (bid level is too low for the hand):

    6. **UNDERBID_VS_PAR** – bid level is *below* the most likely par contract
       that *our side* owns.  Catches stopping at 2H when par says 4H.
    7. **TP_SURPLUS_UNDERBID** – partnership TP exceeds the threshold for a
       higher level consistent with par (game bonus being left on the table).

    **Sacrifice exemption**:

    When the most likely par contract belongs to the *opponents*, overbid
    checks (1-3) are discounted by ``sacrifice_discount`` (default 70%),
    because the bid may be a deliberate sacrifice — going down for less than
    opponents would score.  The sacrifice discount does NOT apply to underbid
    checks or to strain-inefficiency.

    Parameters
    ----------
    bid : str
        The candidate bid string (e.g. "5C", "3NT", "P").
    par_contracts_topk : list of dict, optional
        Top-K par contracts from ``/bid-details`` response.
        Each dict has keys: contract, pair, prob, avg_par_score, etc.
    self_total_points : float, optional
        Acting player's Total_Points (HCP + distribution points).
    partner_tp_hist : dict, optional
        Partner's Total_Points histogram from Phase2a posteriors
        (``phase2a.roles.partner.total_points_hist``).
    par_score_mean : float, optional
        Mean par score from ``/bid-details`` (NS-positive).
    acting_sign : float
        +1.0 for NS acting, -1.0 for EW acting.
    w_overbid_level : float
        Penalty per level the bid exceeds par contract (default 80).
    w_tp_shortfall : float
        Penalty per total-point shortfall below standard minimum (default 15).
    w_neg_par_high : float
        Extra penalty multiplier for negative acting-par at level 4+ (default 0.5).
    w_underbid_level : float
        Penalty per level the bid is below par contract *owned by our side*
        (default 40).
    w_tp_surplus : float
        Penalty per total-point surplus above threshold for the next achievable
        level (default 8).
    w_strain_ineff : float
        Flat penalty for bidding 5m when par says 3NT (default 60).
    sacrifice_discount : float
        Fraction by which overbid penalties are reduced when the par contract
        belongs to opponents (default 0.7 = 70% discount).
    est_tricks : float, optional
        Estimated partnership tricks from the heuristic trick model.
        When provided, enables the TRICKS_SHORTFALL check.
    w_tricks_shortfall : float
        Penalty per estimated trick below the required tricks for the
        contract level (default 30).

    Returns
    -------
    (penalty, reasons) where penalty >= 0.0 and reasons is a list of
    human-readable strings explaining each guardrail that fired.
    """
    penalty = 0.0
    reasons: list[str] = []

    bid_level = _parse_bid_level(bid)
    if bid_level is None:
        return 0.0, []  # Pass, X, XX – no guardrail applicable

    bid_strain = _parse_bid_strain(bid)
    acting_pair = "NS" if acting_sign > 0 else "EW"
    partner_expected_tp = expected_from_hist(partner_tp_hist)

    # ------------------------------------------------------------------
    # Determine the reference par contract and whether sacrifice context
    # ------------------------------------------------------------------
    top_contract: str | None = None
    top_level: int | None = None
    top_strain: str | None = None
    top_pair: str | None = None
    top_prob: float | None = None
    par_is_opponents: bool = False

    if par_contracts_topk:
        top = par_contracts_topk[0]
        top_contract = str(top.get("contract", "") or "")
        top_level = _parse_bid_level(top_contract)
        top_strain = _parse_bid_strain(top_contract)
        top_pair = str(top.get("pair", "") or "").upper()
        try:
            top_prob = float(top.get("prob", 0))
        except Exception:
            top_prob = None
        par_is_opponents = bool(top_pair and top_pair != acting_pair)

    # Sacrifice discount factor: 1.0 = full penalty, 0.0 = no penalty.
    # When opponents own the par, overbid penalties are reduced.
    sac_factor = 1.0
    if par_is_opponents and sacrifice_discount > 0:
        sac_factor = max(0.0, 1.0 - float(sacrifice_discount))

    # Combined partnership TP (used by multiple checks).
    combined_tp: float | None = None
    if self_total_points is not None and partner_expected_tp is not None:
        combined_tp = float(self_total_points) + float(partner_expected_tp)

    # ==================================================================
    # OVERBID CHECKS
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. OVERBID_VS_PAR – bid level exceeds par contract level
    # ------------------------------------------------------------------
    if top_level is not None and bid_level > top_level:
        overbid_levels = bid_level - top_level
        raw_p = float(overbid_levels) * float(w_overbid_level)
        p = raw_p * sac_factor
        penalty += p
        tag = "OVERBID_VS_PAR"
        sac_note = ""
        if par_is_opponents:
            sac_note = (
                f" [sacrifice context: par {top_contract} belongs to {top_pair}; "
                f"penalty reduced {sacrifice_discount*100:.0f}%]"
            )
        prob_s = f"{top_prob*100:.0f}%" if top_prob is not None else "?"
        reasons.append(
            f"{tag}: bid {bid} is {overbid_levels} level(s) above par "
            f"{top_contract} ({prob_s} likely); "
            f"{_TRICKS_REQUIRED.get(bid_level, '?')} tricks needed vs "
            f"{_TRICKS_REQUIRED.get(top_level, '?')} for par "
            f"(-{p:.0f}){sac_note}"
        )

    # ------------------------------------------------------------------
    # 2. TP_SHORTFALL – partnership TP below standard minimum for level
    # ------------------------------------------------------------------
    if combined_tp is not None:
        required_tp = _TP_THRESHOLDS.get(bid_level, 20.0)
        shortfall = required_tp - combined_tp
        if shortfall > 0:
            raw_p = float(shortfall) * float(w_tp_shortfall)
            p = raw_p * sac_factor
            penalty += p
            tag = "TP_SHORTFALL"
            sac_note = ""
            if par_is_opponents:
                sac_note = f" [sacrifice discount applied]"
            reasons.append(
                f"{tag}: need ~{required_tp:.0f} TP for level {bid_level}, "
                f"have ~{combined_tp:.0f} "
                f"(self={self_total_points:.0f} + partner~{partner_expected_tp:.0f}); "
                f"short by {shortfall:.0f} "
                f"(-{p:.0f}){sac_note}"
            )

    # ------------------------------------------------------------------
    # 3. NEG_PAR_HIGH_LEVEL – negative acting-par at level 4+
    # ------------------------------------------------------------------
    if bid_level >= 4 and par_score_mean is not None:
        acting_par = float(acting_sign) * float(par_score_mean)
        if acting_par < 0:
            raw_p = abs(acting_par) * float(w_neg_par_high)
            p = raw_p * sac_factor
            penalty += p
            tag = "NEG_PAR_HIGH_LEVEL"
            sac_note = ""
            if par_is_opponents:
                sac_note = f" [sacrifice discount applied]"
            reasons.append(
                f"{tag}: acting-side expected par is {acting_par:.0f} "
                f"(going down on average) at level {bid_level} "
                f"(-{p:.0f}){sac_note}"
            )

    # ------------------------------------------------------------------
    # 4. STRAIN_INEFFICIENCY – 5m when par is 3NT (same score, 2 extra
    #    tricks required)
    # ------------------------------------------------------------------
    if (
        bid_level == 5
        and bid_strain in ("C", "D")
        and top_level is not None
        and top_strain == "NT"
        and top_level == 3
        and not par_is_opponents
    ):
        p = float(w_strain_ineff)
        penalty += p
        reasons.append(
            f"STRAIN_INEFFICIENCY: {bid} needs {_TRICKS_REQUIRED[5]} tricks; "
            f"par {top_contract} needs only {_TRICKS_REQUIRED[3]} tricks "
            f"for comparable score (-{p:.0f})"
        )

    # ------------------------------------------------------------------
    # 5. TRICKS_SHORTFALL – estimated tricks below required for contract
    # ------------------------------------------------------------------
    if est_tricks is not None and bid_level is not None:
        required = _TRICKS_REQUIRED.get(bid_level, bid_level + 6)
        if est_tricks < required:
            shortfall = float(required) - float(est_tricks)
            raw_p = shortfall * float(w_tricks_shortfall)
            p = raw_p * sac_factor
            penalty += p
            sac_note = f" [sacrifice-discounted from {raw_p:.0f}]" if sac_factor < 1.0 else ""
            reasons.append(
                f"TRICKS_SHORTFALL: estimated {est_tricks:.1f} tricks "
                f"but {bid} requires {required}; "
                f"shortfall {shortfall:.1f} (-{p:.0f}){sac_note}"
            )

    # ==================================================================
    # UNDERBID CHECKS (only when par belongs to our side)
    # ==================================================================

    if not par_is_opponents and top_level is not None:
        # ------------------------------------------------------------------
        # 6. UNDERBID_VS_PAR – bid level below par that our side owns
        # ------------------------------------------------------------------
        if bid_level < top_level:
            underbid_levels = top_level - bid_level
            p = float(underbid_levels) * float(w_underbid_level)
            penalty += p
            prob_s = f"{top_prob*100:.0f}%" if top_prob is not None else "?"
            reasons.append(
                f"UNDERBID_VS_PAR: bid {bid} is {underbid_levels} level(s) "
                f"below par {top_contract} ({prob_s} likely) "
                f"owned by {top_pair}; may leave game/slam bonus on table "
                f"(-{p:.0f})"
            )

        # ------------------------------------------------------------------
        # 7. TP_SURPLUS_UNDERBID – partnership TP exceeds threshold for par
        #    level but we're bidding below it
        # ------------------------------------------------------------------
        if combined_tp is not None and bid_level < top_level:
            par_required_tp = _TP_THRESHOLDS.get(top_level, 20.0)
            surplus = combined_tp - par_required_tp
            if surplus > 0:
                p = float(surplus) * float(w_tp_surplus)
                penalty += p
                reasons.append(
                    f"TP_SURPLUS_UNDERBID: partnership has ~{combined_tp:.0f} TP, "
                    f"exceeding the ~{par_required_tp:.0f} needed for par level "
                    f"{top_level} by {surplus:.0f} "
                    f"(-{p:.0f})"
                )

    return penalty, reasons


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

