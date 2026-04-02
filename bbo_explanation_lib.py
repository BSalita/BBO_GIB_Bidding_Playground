from __future__ import annotations

import math
import re
from typing import Any, Optional

from bbo_hand_eval_lib import estimate_partnership_tricks


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
#   WEAK_HAND_OVERBID    - Acting player's own TP far too low for bid level
#   NEG_PAR_HIGH_LEVEL   - Expected acting-par is negative at level 4+
#   STRAIN_INEFFICIENCY  - Minor 4m/5m when 3NT is par (extra tricks for
#                          comparable or worse score, scaled by extra tricks)
#   TRICKS_SHORTFALL     - Estimated tricks below required for contract level
#   INSUFFICIENT_FIRST_ROUND_CONTROLS - Bid level exceeds 3 + estimated
#                          partnership first-round controls (aces + helpful
#                          voids in suit contracts).  Opponents can cash
#                          enough aces to set the contract.
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

# Minimum *individual* (self) Total_Points to justify voluntarily bidding at
# a level.  These are very conservative floors — even a raise of partner's
# suit generally requires at least this much.  Catches cases where a very
# weak hand (e.g. 5 HCP / 6 TP) makes a high-level bid purely because the
# BT node happens to exist and has inflated aggregate statistics.
_SELF_TP_FLOORS: dict[int, float] = {
    3: 6.0,
    4: 10.0,
    5: 12.0,
    6: 14.0,
    7: 16.0,
}

# Tricks required per level (level + 6).
_TRICKS_REQUIRED: dict[int, int] = {i: i + 6 for i in range(1, 8)}

# Partnership game thresholds used by the "forced non-pass" guardrails.
# These are intentionally simple bridge heuristics:
# - NT game: 25 combined HCP
# - Major game: 26 combined Total_Points
# - Minor game: 29 combined Total_Points
#
# These guardrails currently cover:
# - responder's second-pass shape: `partner opening - pass - ?`
# - opener rebid gap shape: `opening - pass - response - pass - ?`
#
# In both cases, pass is hard-blocked whenever the auction shape or explicit
# pass-node caps make signoff impossible while some legal non-pass continuation
# still exists.
_FORCED_NON_PASS_GAME_HCP_NT = 25.0
_FORCED_NON_PASS_GAME_TP_MAJOR = 26.0
_FORCED_NON_PASS_GAME_TP_MINOR = 29.0


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


_SUITS_ORDER = ["S", "H", "D", "C"]  # dot-notation order


def hand_controls(hand_str: str) -> tuple[int, int]:
    """Count aces and helpful voids from a dot-notation hand string.

    Parameters
    ----------
    hand_str : str
        Dot-notation hand string, e.g. "AT97.A98.AKQJ.T8" (S.H.D.C).

    Returns
    -------
    (aces, helpful_voids) : tuple[int, int]
        *aces* – number of aces in the hand.
        *helpful_voids* – voids in suits where the player does NOT hold the
        ace.  A helpful void prevents opponents from cashing an ace in that
        suit (declarer ruffs the first round).
    """
    parts = str(hand_str or "").strip().split(".")
    if len(parts) != 4:
        return (0, 0)
    aces = 0
    helpful_voids = 0
    for p in parts:
        has_ace = "A" in p
        is_void = len(p) == 0
        if has_ace:
            aces += 1
        if is_void and not has_ace:
            helpful_voids += 1
    return (aces, helpful_voids)


def hand_has_trump_king(hand_str: str, strain: str | None) -> bool:
    """Return True when the hand holds the king of the candidate trump suit."""
    try:
        s = str(strain or "").strip().upper()
        if s not in ("S", "H", "D", "C"):
            return False
        parts = str(hand_str or "").strip().split(".")
        if len(parts) != 4:
            return False
        idx = {"S": 0, "H": 1, "D": 2, "C": 3}[s]
        return "K" in str(parts[idx] or "")
    except Exception:
        return False


def infer_blackwood_partner_control_range(
    response_bid: str | None,
    *,
    contract_strain: str | None = None,
    asker_aces: int | None = None,
    asker_trump_king: bool | None = None,
) -> dict[str, Any] | None:
    """Decode Blackwood follow-up control knowledge from the partner response.

    Convention model:
    - In NT contexts, interpret responses as aces only.
    - In suit contexts, interpret responses as 0314-style keycards
      (aces plus the trump king).

    Response mapping:
    - `5C` -> 0 or 3
    - `5D` -> 1 or 4
    - `5H` -> 2 without the trump queen
    - `5S` -> 2 with the trump queen

    The optional ``asker_aces`` / ``asker_trump_king`` inputs clip impossible
    partner counts based on controls already known in the asker's hand.
    """
    bid = str(response_bid or "").strip().upper()
    strain = str(contract_strain or "").strip().upper()
    is_suit_context = strain in ("C", "D", "H", "S")
    raw_counts_map: dict[str, tuple[int, ...]] = {
        "5C": (0, 3),
        "5D": (1, 4),
        "5H": (2,),
        "5S": (2,),
    }
    raw_counts = raw_counts_map.get(bid)
    if raw_counts is None:
        return None

    max_controls_total = 5 if is_suit_context else 4
    asker_known_controls = 0
    if asker_aces is not None:
        try:
            asker_known_controls += max(0, min(4, int(asker_aces)))
        except Exception:
            asker_known_controls += 0
    if is_suit_context and asker_trump_king:
        asker_known_controls += 1
    asker_known_controls = max(0, min(max_controls_total, asker_known_controls))
    max_possible = max(0, min(max_controls_total, max_controls_total - asker_known_controls))

    possible_counts = sorted({int(v) for v in raw_counts if int(v) <= int(max_possible)})
    control_kind = "keycards" if is_suit_context else "aces"
    convention_name = "rkcb_0314_style" if is_suit_context else "ace_response_0314_style"
    if not possible_counts:
        return {
            "response_bid": bid,
            "possible": False,
            "possible_counts": [],
            "control_kind": control_kind,
            "trump_queen_known": bool(is_suit_context and bid in ("5H", "5S")),
            "partner_has_trump_queen": (False if is_suit_context and bid == "5H" else True if is_suit_context and bid == "5S" else None),
            "min_controls": None,
            "max_controls": None,
            "expected_controls": None,
            "slam_positive_expected_controls": None,
            "exact_controls": None,
            "min_aces": None,
            "max_aces": None,
            "expected_aces": None,
            "slam_positive_expected_aces": None,
            "exact_aces": None,
            "source": "blackwood_response",
            "convention": convention_name,
        }

    min_controls = int(min(possible_counts))
    max_controls = int(max(possible_counts))
    exact_controls = int(possible_counts[0]) if len(possible_counts) == 1 else None
    expected_controls = float(sum(possible_counts)) / float(len(possible_counts))
    slam_positive_expected_controls = expected_controls
    if len(possible_counts) >= 2:
        # Once the asker is considering continuing past the response, weight the
        # upper branch more heavily than a flat midpoint. Low-branch responses
        # usually lead to signoff rather than constructive slam action.
        _lo = float(min_controls)
        _hi = float(max_controls)
        slam_positive_expected_controls = (0.25 * _lo) + (0.75 * _hi)
    return {
        "response_bid": bid,
        "possible": True,
        "possible_counts": possible_counts,
        "control_kind": control_kind,
        "trump_queen_known": bool(is_suit_context and bid in ("5H", "5S")),
        "partner_has_trump_queen": (False if is_suit_context and bid == "5H" else True if is_suit_context and bid == "5S" else None),
        "min_controls": min_controls,
        "max_controls": max_controls,
        "expected_controls": expected_controls,
        "slam_positive_expected_controls": slam_positive_expected_controls,
        "exact_controls": exact_controls,
        # Legacy aliases kept so older call sites do not break.
        "min_aces": min_controls,
        "max_aces": max_controls,
        "expected_aces": expected_controls,
        "slam_positive_expected_aces": slam_positive_expected_controls,
        "exact_aces": exact_controls,
        "source": "blackwood_response",
        "convention": convention_name,
    }


def infer_blackwood_partner_ace_range(
    response_bid: str | None,
    *,
    asker_aces: int | None = None,
) -> dict[str, Any] | None:
    """Backward-compatible ace-only Blackwood decoder."""
    return infer_blackwood_partner_control_range(
        response_bid,
        contract_strain="NT",
        asker_aces=asker_aces,
        asker_trump_king=False,
    )


def should_relax_v2_slam_cap_for_blackwood(
    *,
    bid_level: int | None,
    bid_strain: str | None,
    game_level: int | None,
    prev_same_side_same_strain_level: int | None,
    level_cap_exceeded: bool,
    ceiling_qualified: bool,
    partner_known_ace_floor: float | None,
    partner_known_ace_ceiling: float | None,
    partner_known_ace_source: str | None,
) -> dict[str, Any]:
    """Return whether Blackwood follow-up evidence should relax the V2 slam cap.

    This is intentionally narrow:
    - only same-strain bids above game,
    - only when the side is continuing from its own game-level contract,
    - only when the normal floor-based V2 cap is the blocker,
    - only when the ceiling still qualifies the slam,
    - only when partner ace knowledge came from explicit Blackwood follow-up.
    """
    out = {
        "apply": False,
        "reason": None,
    }
    try:
        if bid_level is None or game_level is None:
            return out
        strain = str(bid_strain or "").upper()
        source = str(partner_known_ace_source or "")
        if strain not in ("C", "D", "H", "S", "N", "NT"):
            return out
        if int(bid_level) <= int(game_level):
            return out
        if prev_same_side_same_strain_level is None or int(prev_same_side_same_strain_level) != int(game_level):
            return out
        if not bool(level_cap_exceeded):
            return out
        if not bool(ceiling_qualified):
            return out
        if partner_known_ace_floor is None or partner_known_ace_ceiling is None:
            return out
        if not source.upper().startswith("BLACKWOOD "):
            return out
        out["apply"] = True
        out["reason"] = (
            "BLACKWOOD_SAME_STRAIN_SLAM_FLOOR_RELAXATION: explicit Blackwood follow-up "
            "evidence allows same-strain slam exploration when the ceiling qualifies "
            "even though the floor-based V2 cap does not"
        )
        return out
    except Exception:
        return out


def should_relax_slam_gate_for_blackwood_ask(
    *,
    bid_text: str | None,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    par_contracts_topk: list[dict[str, Any]] | None,
    p_slam_tp_ge_33: float | None = None,
    combined_tp_mean: float | None = None,
) -> dict[str, Any]:
    """Return whether a 4NT Blackwood ask should bypass the hard slam gate.

    This is intentionally narrow: only when our side already has a major-game
    contract on the table and aggregated top-k evidence already shows plausible
    slam ownership, even if EV/range-derived slam inputs are missing.
    """
    out = {
        "apply": False,
        "reason": None,
    }
    try:
        b = str(bid_text or "").strip().upper()
        if b not in ("4N", "4NT"):
            return out

        act = str(acting_direction or "").strip().upper()
        if act not in ("N", "E", "S", "W"):
            return out

        directions = ["N", "E", "S", "W"]
        d = str(dealer_actual or "N").upper()
        dealer_idx = directions.index(d) if d in directions else 0

        def _token_bidder_dir(token_idx: int) -> str:
            return directions[(dealer_idx + int(token_idx)) % 4]

        def _strain_rank(st: str) -> int:
            return {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4, "NT": 4}.get(str(st or "").upper(), -1)

        acting_side = "NS" if act in ("N", "S") else "EW"
        best_same_side: dict[str, Any] | None = None
        for i, tk in enumerate(list(auction_tokens or [])):
            lvl = _parse_bid_level(tk)
            st = _parse_bid_strain(tk)
            if lvl is None or st is None:
                continue
            bidder_dir = _token_bidder_dir(i)
            bidder_side = "NS" if bidder_dir in ("N", "S") else "EW"
            if bidder_side != acting_side:
                continue
            if (
                best_same_side is None
                or int(lvl) > int(best_same_side["level"])
                or (
                    int(lvl) == int(best_same_side["level"])
                    and _strain_rank(str(st)) > _strain_rank(str(best_same_side["strain"]))
                )
            ):
                best_same_side = {"level": int(lvl), "strain": str(st), "token_idx": int(i)}

        if best_same_side is None:
            return out
        if str(best_same_side["strain"]) not in ("H", "S"):
            return out
        if int(best_same_side["level"]) < 4:
            return out

        strong_structural_support = False
        support_parts: list[str] = []
        if p_slam_tp_ge_33 is not None and float(p_slam_tp_ge_33) >= 0.45:
            strong_structural_support = True
            support_parts.append(f"p33={float(p_slam_tp_ge_33):.2f}")
        if combined_tp_mean is not None and float(combined_tp_mean) >= 32.0:
            strong_structural_support = True
            support_parts.append(f"combinedTP={float(combined_tp_mean):.1f}")

        topk_slam_support = False
        if isinstance(par_contracts_topk, list):
            target_major = str(best_same_side["strain"])
            for row in par_contracts_topk:
                if not isinstance(row, dict):
                    continue
                contract = str(row.get("contract") or "").strip().upper()
                lvl = _parse_bid_level(contract)
                st = _parse_bid_strain(contract)
                if lvl is None or st is None or int(lvl) < 6:
                    continue
                if st not in (target_major, "NT"):
                    continue
                try:
                    prob = float(row.get("prob") or 0.0)
                    avg_par_score = float(row.get("avg_par_score") or 0.0)
                except Exception:
                    prob = 0.0
                    avg_par_score = 0.0
                if prob >= 0.10 and avg_par_score > 0:
                    topk_slam_support = True
                    support_parts.append(f"topk {contract} prob={prob:.2f}")
                    break

        if not topk_slam_support:
            return out
        if not strong_structural_support and not topk_slam_support:
            return out

        out["apply"] = True
        out["reason"] = (
            "BLACKWOOD_ASK_TOPK_BYPASS: agreed major game is already on the table and "
            f"aggregated slam evidence supports a 4NT ask ({', '.join(support_parts)})"
        )
        return out
    except Exception:
        return out


def should_relax_slam_gate_for_blackwood_followup(
    *,
    bid_text: str | None,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    par_contracts_topk: list[dict[str, Any]] | None,
    p_slam_tp_ge_33: float | None = None,
    combined_tp_mean: float | None = None,
) -> dict[str, Any]:
    """Return whether a post-Blackwood slam follow-up should bypass the slam gate.

    Narrow scope:
    - original 4NT asker is acting after partner's 5x response
    - same-strain continuation above game
    - only for actual slam-level continuation (level 6+)
    - exact candidate contract already has meaningful positive top-k support
    """
    out = {
        "apply": False,
        "reason": None,
    }
    try:
        ctx = extract_blackwood_same_strain_continuation_context(
            auction_tokens=list(auction_tokens or []),
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
            bid_text=bid_text,
        )
        if ctx is None or not bool(ctx.get("is_post_game_same_strain", False)):
            return out
        bid_level = int(ctx.get("bid_level"))
        bid_strain = str(ctx.get("bid_strain"))
        if bid_level < 6:
            return out

        support_parts: list[str] = []
        exact_topk_support = False
        if isinstance(par_contracts_topk, list):
            for row in par_contracts_topk:
                if not isinstance(row, dict):
                    continue
                contract = str(row.get("contract") or "").strip().upper()
                lvl = _parse_bid_level(contract)
                st = _parse_bid_strain(contract)
                if lvl is None or st is None:
                    continue
                if int(lvl) != int(bid_level) or str(st) != bid_strain:
                    continue
                try:
                    prob = float(row.get("prob") or 0.0)
                    avg_par_score = float(row.get("avg_par_score") or 0.0)
                except Exception:
                    prob = 0.0
                    avg_par_score = 0.0
                if prob >= 0.10 and avg_par_score > 0:
                    exact_topk_support = True
                    support_parts.append(f"topk {contract} prob={prob:.2f}")
                    break
        if not exact_topk_support:
            return out

        if p_slam_tp_ge_33 is not None and float(p_slam_tp_ge_33) >= 0.45:
            support_parts.append(f"p33={float(p_slam_tp_ge_33):.2f}")
        if combined_tp_mean is not None and float(combined_tp_mean) >= 32.0:
            support_parts.append(f"combinedTP={float(combined_tp_mean):.1f}")

        out["apply"] = True
        out["reason"] = (
            "BLACKWOOD_FOLLOWUP_TOPK_BYPASS: post-Blackwood same-strain slam continuation "
            f"is supported by aggregated evidence ({', '.join(support_parts)})"
        )
        return out
    except Exception:
        return out


def compute_blackwood_same_strain_signoff_penalty(
    *,
    bid_text: str | None,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    par_contracts_topk: list[dict[str, Any]] | None,
    partner_known_control_expected: float | None = None,
    partner_known_control_floor: float | None = None,
    p_slam_tp_ge_33: float | None = None,
    combined_tp_mean: float | None = None,
) -> dict[str, Any]:
    """Penalize post-Blackwood same-strain signoff when slam evidence is strong.

    This is intentionally narrow and only targets the "sign off at 5M after
    asking Blackwood over an already established game" pattern. It should not
    interfere with weak-response signoffs.
    """
    out = {
        "penalty": 0.0,
        "reason": None,
    }
    try:
        b = str(bid_text or "").strip().upper()
        m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", b)
        if not m:
            return out
        bid_level = int(m.group(1))
        bid_strain = "NT" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
        if bid_level != 5 or bid_strain not in ("H", "S", "D", "C"):
            return out

        ctx = extract_blackwood_same_strain_continuation_context(
            auction_tokens=list(auction_tokens or []),
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
            bid_text=b,
        )
        if ctx is None or not bool(ctx.get("is_post_game_same_strain", False)):
            return out
        if int(ctx.get("prev_same_side_same_strain_level") or 0) != 4:
            return out

        slam_contract = f"6{bid_strain}"
        exact_topk_support = False
        slam_prob = 0.0
        if isinstance(par_contracts_topk, list):
            for row in par_contracts_topk:
                if not isinstance(row, dict):
                    continue
                contract = str(row.get("contract") or "").strip().upper()
                if contract != slam_contract:
                    continue
                try:
                    prob = float(row.get("prob") or 0.0)
                    avg_par_score = float(row.get("avg_par_score") or 0.0)
                except Exception:
                    prob = 0.0
                    avg_par_score = 0.0
                if prob >= 0.10 and avg_par_score > 0:
                    exact_topk_support = True
                    slam_prob = prob
                    break
        if not exact_topk_support:
            return out

        control_expected = None
        if isinstance(partner_known_control_expected, (int, float)):
            control_expected = float(partner_known_control_expected)
        elif isinstance(partner_known_control_floor, (int, float)):
            control_expected = float(partner_known_control_floor)
        control_positive = bool(control_expected is not None and control_expected >= 2.0)
        point_positive = bool(
            isinstance(combined_tp_mean, (int, float))
            and float(combined_tp_mean) >= 32.0
        )
        prob_positive = bool(
            isinstance(p_slam_tp_ge_33, (int, float))
            and float(p_slam_tp_ge_33) >= 0.45
        )
        if not control_positive:
            return out
        if not (point_positive or prob_positive):
            return out

        penalty = 140.0
        support_parts = [f"topk {slam_contract} prob={slam_prob:.2f}"]
        if control_expected is not None:
            support_parts.append(f"partnerControls~{control_expected:.2f}")
        if prob_positive and p_slam_tp_ge_33 is not None:
            support_parts.append(f"p33={float(p_slam_tp_ge_33):.2f}")
        if point_positive and combined_tp_mean is not None:
            support_parts.append(f"combinedTP={float(combined_tp_mean):.1f}")
        out["penalty"] = penalty
        out["reason"] = (
            "BLACKWOOD_SAME_STRAIN_SIGNOFF_PENALTY: post-Blackwood same-strain signoff "
            f"{b} is too conservative when slam evidence is strong ({', '.join(support_parts)}) "
            f"(-{penalty:.0f})"
        )
        return out
    except Exception:
        return out


def compute_non_rebiddable_suit_rebid_penalty(
    *,
    bid_text: str,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    bt_acting_criteria: list[str] | None,
    self_suit_lengths: dict[str, int] | None = None,
) -> tuple[float, str | None, bool]:
    """Penalty for suit rebids lacking rebiddable support.

    Returns `(penalty, reason, hard_block)` where `hard_block` is reserved for
    clearly impossible rebids based on the known acting hand shape.
    """
    s = str(bid_text or "").strip().upper()
    m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
    if not m:
        return 0.0, None, False
    bid_level = int(m.group(1))
    bid_strain = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
    if bid_strain not in ("S", "H", "D", "C"):
        return 0.0, None, False
    if not acting_direction:
        return 0.0, None, False

    # Determine which direction bid each prior token from dealer + token index.
    directions = ["N", "E", "S", "W"]
    d = str(dealer_actual or "N").upper()
    dealer_idx = directions.index(d) if d in directions else 0

    def _token_bidder_dir(token_idx: int) -> str:
        return directions[(dealer_idx + int(token_idx)) % 4]

    # Jacoby transfer acceptance map: opener's forced reply to partner's
    # transfer bid is not a natural suit bid and should be excluded from
    # "previous bids in this strain" tracking.
    _TRANSFER_ACCEPT = {
        ("1N", "2D"): "2H", ("1N", "2H"): "2S",
        ("2N", "3D"): "3H", ("2N", "3H"): "3S",
    }
    _transfer_accept_indices: set[int] = set()
    for i, tk in enumerate(auction_tokens):
        tk_u = str(tk or "").strip().upper()
        if _token_bidder_dir(i) != acting_direction:
            continue
        # Look back for the (opener_nt, partner_transfer) pair
        if i >= 2:
            opener_nt = str(auction_tokens[i - 2] or "").strip().upper()
            partner_xfer = str(auction_tokens[i - 1] or "").strip().upper()
            if _TRANSFER_ACCEPT.get((opener_nt, partner_xfer)) == tk_u:
                _transfer_accept_indices.add(i)

    # Has this same player already bid this same strain?
    prev_levels: list[int] = []
    for i, tk in enumerate(auction_tokens):
        if _token_bidder_dir(i) != acting_direction:
            continue
        if i in _transfer_accept_indices:
            continue
        p = re.match(r"^([1-7])\s*(NT|N|[CDHS])", str(tk or "").strip().upper())
        if not p:
            continue
        pl = int(p.group(1))
        ps = "N" if p.group(2).upper() in ("N", "NT") else p.group(2).upper()
        if ps == bid_strain:
            prev_levels.append(pl)
    if not prev_levels:
        return 0.0, None, False

    partner_direction = {"N": "S", "S": "N", "E": "W", "W": "E"}.get(str(acting_direction), "")
    partner_supported_strain = False
    _TRANSFER_MAP = {"2D": "H", "2H": "S", "3D": "H", "3H": "S"}
    if partner_direction:
        for i, tk in enumerate(auction_tokens):
            if _token_bidder_dir(i) != partner_direction:
                continue
            tk_u = str(tk or "").strip().upper()
            p = re.match(r"^([1-7])\s*(NT|N|[CDHS])", tk_u)
            if not p:
                continue
            ps = "N" if p.group(2).upper() in ("N", "NT") else p.group(2).upper()
            if ps == bid_strain:
                partner_supported_strain = True
                break
            if _TRANSFER_MAP.get(tk_u) == bid_strain:
                partner_supported_strain = True
                break

    # Once partner has explicitly supported the same major, a direct jump to game
    # is a fit-based game commitment, not a pure "long-suit rebid" that should
    # require REBIDDABLE/TWICE_REBIDDABLE evidence.
    if bid_strain in ("H", "S") and bid_level == 4 and partner_supported_strain:
        return 0.0, None, False

    # After partner answers Blackwood, same-strain continuations above an
    # already-established game contract are convention-driven signoff/slam
    # decisions rather than natural "I am rebidding my long suit" actions.
    try:
        _bw_ctx = extract_blackwood_same_strain_continuation_context(
            auction_tokens=list(auction_tokens or []),
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
            bid_text=s,
        )
        if _bw_ctx is not None and bool(_bw_ctx.get("is_post_game_same_strain", False)):
            return 0.0, None, False
    except Exception:
        pass

    crit_norm = {str(c or "").strip().upper().strip("()") for c in (bt_acting_criteria or []) if str(c or "").strip()}
    has_reb = f"REBIDDABLE_{bid_strain}" in crit_norm
    has_twice = f"TWICE_REBIDDABLE_{bid_strain}" in crit_norm

    # 3-level+ rebids should generally be twice-rebiddable; lower rebids need rebiddable.
    needed = "TWICE_REBIDDABLE" if bid_level >= 3 else "REBIDDABLE"
    acceptable = has_twice if bid_level >= 3 else (has_reb or has_twice)
    actual_len = None
    try:
        if isinstance(self_suit_lengths, dict):
            _raw_len = self_suit_lengths.get(bid_strain)
            if _raw_len is not None:
                actual_len = int(_raw_len)
    except Exception:
        actual_len = None

    # When we know the actual hand, separate truly impossible rebids from
    # softer "criteria tag missing" cases.
    if actual_len is not None:
        if actual_len < 5:
            penalty = 260.0 if bid_level >= 3 else 170.0
            reason = (
                f"IMPOSSIBLE_SUIT_REBID: rebid {s} in {bid_strain} with actual length "
                f"{actual_len}; natural rebid needs at least 5 cards (-{penalty:.0f})"
            )
            return float(penalty), reason, True
        if actual_len >= 6:
            return 0.0, None, False
    if acceptable:
        return 0.0, None, False

    penalty = 170.0 if bid_level >= 3 else 120.0
    if actual_len is not None:
        reason = (
            f"NON_REBIDDABLE_SUIT_REBID: rebid {s} in {bid_strain} with actual length "
            f"{actual_len} but without required {needed}_{bid_strain} criteria support "
            f"(-{penalty:.0f})"
        )
    else:
        reason = (
            f"NON_REBIDDABLE_SUIT_REBID: rebid {s} in {bid_strain} "
            f"without required {needed}_{bid_strain} criteria support "
            f"(-{penalty:.0f})"
        )
    return float(penalty), reason, False


def compute_rebiddable_major_game_bonus(
    *,
    bid_text: str,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    bt_acting_criteria: list[str] | None,
) -> tuple[float, str | None]:
    """Bonus for committing to game in a strongly rebiddable major.

    Purpose: when a player has shown a long/strong major and now has a chance to
    bid game in that same major, prefer direct game over side-suit wandering.
    """
    s = str(bid_text or "").strip().upper()
    m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
    if not m:
        return 0.0, None
    bid_level = int(m.group(1))
    bid_strain = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
    if bid_strain not in ("H", "S"):
        return 0.0, None
    if bid_level < 4:
        return 0.0, None
    if not acting_direction:
        return 0.0, None

    directions = ["N", "E", "S", "W"]
    d = str(dealer_actual or "N").upper()
    dealer_idx = directions.index(d) if d in directions else 0

    def _token_bidder_dir(token_idx: int) -> str:
        return directions[(dealer_idx + int(token_idx)) % 4]

    # Must be a rebid (same player has bid this major earlier).
    showed_major = False
    for i, tk in enumerate(auction_tokens):
        if _token_bidder_dir(i) != acting_direction:
            continue
        p = re.match(r"^([1-7])\s*(NT|N|[CDHS])", str(tk or "").strip().upper())
        if not p:
            continue
        ps = "N" if p.group(2).upper() in ("N", "NT") else p.group(2).upper()
        if ps == bid_strain:
            showed_major = True
            break
    if not showed_major:
        return 0.0, None

    crit_norm = {str(c or "").strip().upper().strip("()") for c in (bt_acting_criteria or []) if str(c or "").strip()}
    has_reb = f"REBIDDABLE_{bid_strain}" in crit_norm
    has_twice = f"TWICE_REBIDDABLE_{bid_strain}" in crit_norm
    if not (has_reb or has_twice):
        return 0.0, None

    # Prefer "doubly rebiddable" as stronger evidence than plain rebiddable.
    bonus = 120.0 if has_twice else 70.0
    reason = (
        f"REBIDDABLE_MAJOR_GAME: committing to {bid_text} with "
        f"{'TWICE_REBIDDABLE' if has_twice else 'REBIDDABLE'}_{bid_strain} "
        f"criteria support (+{bonus:.0f})"
    )
    return float(bonus), reason


def compute_partner_major_game_commit_adjustment(
    *,
    bid_text: str,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    bt_acting_criteria: list[str] | None = None,
) -> tuple[float, float, str | None]:
    """Context adjustment for partner-major game commitment decisions.

    Returns (bonus, penalty, reason).
    - Bonus: when raising partner's repeatedly shown major directly to game.
    - Penalty: when detouring into side-suit (esp. 4m) instead of that game call.
    """
    s = str(bid_text or "").strip().upper()
    m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
    if not m:
        return 0.0, 0.0, None
    bid_level = int(m.group(1))
    bid_strain = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
    if not acting_direction:
        return 0.0, 0.0, None

    directions = ["N", "E", "S", "W"]
    d = str(dealer_actual or "N").upper()
    dealer_idx = directions.index(d) if d in directions else 0

    def _token_bidder_dir(token_idx: int) -> str:
        return directions[(dealer_idx + int(token_idx)) % 4]

    partner = {"N": "S", "S": "N", "E": "W", "W": "E"}.get(str(acting_direction), "")
    if not partner:
        return 0.0, 0.0, None

    # Partner suit-showing history.
    partner_major_counts = {"H": 0, "S": 0}
    partner_last_major: str | None = None
    for i, tk in enumerate(auction_tokens):
        if _token_bidder_dir(i) != partner:
            continue
        p = re.match(r"^([1-7])\s*(NT|N|[CDHS])", str(tk or "").strip().upper())
        if not p:
            continue
        ps = "N" if p.group(2).upper() in ("N", "NT") else p.group(2).upper()
        if ps in ("H", "S"):
            partner_major_counts[ps] += 1
            partner_last_major = ps

    if not partner_last_major:
        return 0.0, 0.0, None

    crit_norm = {str(c or "").strip().upper().strip("()") for c in (bt_acting_criteria or []) if str(c or "").strip()}
    fit_context = (
        "SUPPORTSHOWING" in crit_norm
        or "FITESTABLISHED" in crit_norm
        or "RAISE" in crit_norm
    )

    # Stayman acceptance is a softer but still meaningful major-fit signal:
    # after 1NT-2C-2M, responder should lean toward the shown major rather than
    # defaulting to 3NT. Keep this preference moderate because some hands still
    # belong in NT.
    stayman_ctx = _stayman_major_response_context(
        auction_tokens,
        acting_direction=acting_direction,
        dealer_actual=dealer_actual,
    )
    shown_major = str((stayman_ctx or {}).get("shown_major") or "").upper()
    if shown_major in ("H", "S"):
        if bid_strain == shown_major and bid_level >= 4:
            bonus = 120.0 + (20.0 if fit_context else 0.0)
            reason = (
                f"STAYMAN_MAJOR_FIT_PREFERENCE: partner showed {shown_major} via Stayman; "
                f"prefer committing to {bid_text} over notrump when game values are present "
                f"(+{bonus:.0f})"
            )
            return float(bonus), 0.0, reason
        if bid_strain == "N" and bid_level >= 3:
            penalty = 80.0 + (20.0 if fit_context else 0.0)
            reason = (
                f"STAYMAN_MAJOR_NT_DETOUR: partner showed {shown_major} via Stayman; "
                f"discount {bid_text} because the major-suit fit is still the default target "
                f"(-{penalty:.0f})"
            )
            return 0.0, float(penalty), reason

    opener_rebid_ctx = extract_opener_rebid_pass_context(
        auction_tokens=list(auction_tokens or []),
        acting_direction=acting_direction,
        dealer_actual=dealer_actual,
    )
    if opener_rebid_ctx is not None:
        opening_strain = str(_parse_bid_strain(opener_rebid_ctx.get("opening_bid")) or "").upper()
        response_strain = str(opener_rebid_ctx.get("response_strain") or "").upper()
        if opening_strain in ("C", "D") and response_strain in ("H", "S"):
            support_floor = None
            tp_floor = None
            hcp_floor = None
            for expr in crit_norm:
                m_support = re.fullmatch(rf"SL_{response_strain}\s*(>=|>|==)\s*(\d+)", expr)
                if m_support is not None:
                    support_val = int(m_support.group(2))
                    if m_support.group(1) == ">":
                        support_val += 1
                    support_floor = support_val if support_floor is None else max(support_floor, support_val)
                    continue
                m_tp = re.fullmatch(r"TOTAL_POINTS\s*(>=|>|==)\s*(-?\d+(?:\.\d+)?)", expr)
                if m_tp is not None:
                    tp_val = float(m_tp.group(2))
                    if m_tp.group(1) == ">":
                        tp_val += 1.0
                    tp_floor = tp_val if tp_floor is None else max(tp_floor, tp_val)
                    continue
                m_hcp = re.fullmatch(r"HCP\s*(>=|>|==)\s*(-?\d+(?:\.\d+)?)", expr)
                if m_hcp is not None:
                    hcp_val = float(m_hcp.group(2))
                    if m_hcp.group(1) == ">":
                        hcp_val += 1.0
                    hcp_floor = hcp_val if hcp_floor is None else max(hcp_floor, hcp_val)
            strong_major_raise = (
                support_floor is not None
                and support_floor >= 4
                and (
                    (tp_floor is not None and tp_floor >= 18.0)
                    or (hcp_floor is not None and hcp_floor >= 15.0)
                    or fit_context
                )
            )
            if strong_major_raise and bid_strain == response_strain and bid_level == 4:
                bonus = 95.0 + (20.0 if fit_context else 0.0)
                reason = (
                    "OPENER_MINOR_RESPONSE_MAJOR_GAME_COMMIT: partner responded "
                    f"1{response_strain} to opener's 1{opening_strain} and {bid_text} shows 4+ support "
                    "with extras; prefer direct major-game commitment "
                    f"(+{bonus:.0f})"
                )
                return float(bonus), 0.0, reason

    # "Doubly rebiddable by history": partner bid the same major at least twice.
    partner_major_rebid_count = int(partner_major_counts.get(partner_last_major, 0))
    if partner_major_rebid_count < 2:
        return 0.0, 0.0, None

    major_game_level = 4 if partner_last_major in ("H", "S") else 5

    # Prefer direct game in partner's repeated major, but do not reward
    # past-game continuations as if they were ordinary game signoffs.
    if bid_strain == partner_last_major and bid_level == major_game_level:
        bonus = 110.0 + (20.0 if fit_context else 0.0)
        reason = (
            f"PARTNER_MAJOR_GAME_COMMIT: partner has repeatedly shown {partner_last_major} "
            f"({partner_major_rebid_count} calls); prefer direct {bid_text} game commitment "
            f"(+{bonus:.0f})"
        )
        return float(bonus), 0.0, reason

    # Penalize side-suit detours when partner's major game is available.
    if bid_level >= 4 and bid_strain in ("C", "D", "H", "S", "N") and bid_strain != partner_last_major:
        # Stronger penalty for 4m detour.
        detour_pen = 120.0 if bid_strain in ("C", "D") else 90.0
        detour_pen += 20.0 if fit_context else 0.0
        reason = (
            f"PARTNER_MAJOR_GAME_DETOUR: partner repeatedly showed {partner_last_major}; "
            f"side-suit/non-fit game detour {bid_text} without committing that major (-{detour_pen:.0f})"
        )
        return 0.0, float(detour_pen), reason

    return 0.0, 0.0, None


def compute_pass_signoff_bonus(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    self_total_points: float | None = None,
    self_hcp: float | None = None,
    pass_agg_expr: list[Any] | None = None,
) -> tuple[float, str | None]:
    """Bonus for Pass when partner committed to game or invited and self is minimum."""
    if not acting_direction:
        return 0.0, None
    directions = ["N", "E", "S", "W"]
    d = str(dealer_actual or "N").upper()
    dealer_idx = directions.index(d) if d in directions else 0

    def _token_bidder_dir(token_idx: int) -> str:
        return directions[(dealer_idx + int(token_idx)) % 4]

    partner = {"N": "S", "S": "N", "E": "W", "W": "E"}.get(str(acting_direction), "")
    if not partner:
        return 0.0, None

    last_non_pass_idx = None
    for i in range(len(auction_tokens) - 1, -1, -1):
        tk = str(auction_tokens[i] or "").strip().upper()
        if tk and tk not in ("P", "PASS", "X", "XX"):
            last_non_pass_idx = i
            break
    if last_non_pass_idx is None:
        return 0.0, None

    last_bid = str(auction_tokens[last_non_pass_idx] or "").strip().upper()
    m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", last_bid)
    if not m:
        return 0.0, None
    lvl = int(m.group(1))
    st = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
    game_min = 3 if st == "N" else (4 if st in ("H", "S") else 5)
    last_bidder = _token_bidder_dir(int(last_non_pass_idx))
    if last_bidder != partner:
        return 0.0, None
    if lvl < game_min:
        tp_actual = float(self_total_points) if isinstance(self_total_points, (int, float)) else None
        hcp_actual = float(self_hcp) if isinstance(self_hcp, (int, float)) else None
        caps = extract_pass_range_caps(pass_agg_expr)
        tp_cap = caps.get("tp_cap")
        hcp_cap = caps.get("hcp_cap")
        has_minimum_cap = False
        if tp_cap is not None and tp_actual is not None and tp_actual <= float(tp_cap):
            has_minimum_cap = True
        if hcp_cap is not None and hcp_actual is not None and hcp_actual <= float(hcp_cap):
            has_minimum_cap = True
        if lvl == game_min - 1 and has_minimum_cap:
            bonus = 400.0 if st == "N" else 360.0
            tp_txt = f"{tp_actual:.0f}" if tp_actual is not None else "?"
            hcp_txt = f"{hcp_actual:.0f}" if hcp_actual is not None else "?"
            reason = (
                f"PASS_INVITE_DECLINE_MINIMUM: partner last bid {last_bid} inviting game; "
                f"self fits pass minimum range ({tp_txt} TP / {hcp_txt} HCP), so reward declining with Pass (+{bonus:.0f})"
            )
            return bonus, reason
        return 0.0, None

    bonus = 80.0
    reason = f"PASS_SIGNOFF_BONUS: partner last bid {last_bid} (game+); reward closing action (+{bonus:.0f})"
    return bonus, reason


def compute_constructive_pass_penalty(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    has_non_pass_choice: bool,
    self_total_points: float | None,
    self_hcp: float | None,
    pass_agg_expr: list[Any] | None,
) -> tuple[float, str | None]:
    """Penalize pass when the hand/action profile is too strong to sign off.

    Two cases are covered:
    1. The BT pass node itself has explicit upper bounds (for example
       ``Total_Points <= 9``) that the actual hand violates.
    2. In a one-sided constructive auction below game, partner made the last
       contract bid, constructive continuations still exist, and the acting
       hand has clear extras. In that case pass should not win merely by
       inheriting current-contract EV.
    """
    if not acting_direction or not has_non_pass_choice:
        return 0.0, None

    penalty = 0.0
    reasons: list[str] = []
    tp_actual = float(self_total_points) if isinstance(self_total_points, (int, float)) else None
    hcp_actual = float(self_hcp) if isinstance(self_hcp, (int, float)) else None

    tp_excess = _upper_bound_excess_from_exprs("TOTAL_POINTS", tp_actual, pass_agg_expr)
    if tp_excess is not None:
        tp_cap, tp_over = tp_excess
        tp_pen = min(220.0, 30.0 * float(tp_over))
        penalty += tp_pen
        reasons.append(
            "PASS_RANGE_TOO_STRONG: pass criteria cap Total_Points at "
            f"{tp_cap:.0f}; self has {tp_actual:.0f} (-{tp_pen:.0f})"
        )

    hcp_excess = _upper_bound_excess_from_exprs("HCP", hcp_actual, pass_agg_expr)
    if hcp_excess is not None:
        hcp_cap, hcp_over = hcp_excess
        hcp_pen = min(160.0, 20.0 * float(hcp_over))
        penalty += hcp_pen
        reasons.append(
            "PASS_RANGE_TOO_STRONG: pass criteria cap HCP at "
            f"{hcp_cap:.0f}; self has {hcp_actual:.0f} (-{hcp_pen:.0f})"
        )

    # Generic auction-level sanity check: below game, partner has made the last
    # constructive bid, and this hand has extras, so pass should not coast on
    # current-contract EV while better continuations are still available.
    try:
        toks = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
        directions = ["N", "E", "S", "W"]
        d = str(dealer_actual or "N").upper()
        dealer_idx = directions.index(d) if d in directions else 0

        def _token_bidder_dir(token_idx: int) -> str:
            return directions[(dealer_idx + int(token_idx)) % 4]

        contract_positions: list[int] = []
        contract_sides: set[str] = set()
        last_contract_idx: int | None = None
        last_contract_bid: str | None = None
        for idx, tk in enumerate(toks):
            lvl = _parse_bid_level(tk)
            st = _parse_bid_strain(tk)
            if lvl is None or st is None:
                continue
            contract_positions.append(idx)
            bidder_dir = _token_bidder_dir(idx)
            contract_sides.add("NS" if bidder_dir in ("N", "S") else "EW")
            last_contract_idx = idx
            last_contract_bid = tk

        if last_contract_idx is not None and last_contract_bid is not None and len(contract_sides) == 1:
            lvl = _parse_bid_level(last_contract_bid)
            st = _parse_bid_strain(last_contract_bid)
            if lvl is not None and st is not None:
                game_min = 3 if st == "NT" else (4 if st in ("H", "S") else 5)
                partner = {"N": "S", "S": "N", "E": "W", "W": "E"}.get(str(acting_direction).upper(), "")
                last_bidder = _token_bidder_dir(last_contract_idx)
                has_constructive_extras = (
                    (tp_actual is not None and tp_actual >= 14.0)
                    or (hcp_actual is not None and hcp_actual >= 13.0)
                )
                if (
                    partner
                    and last_bidder == partner
                    and int(lvl) < int(game_min)
                    and len(contract_positions) >= 2
                    and has_constructive_extras
                ):
                    extras_strength = max(
                        (tp_actual - 14.0) if tp_actual is not None else 0.0,
                        (hcp_actual - 13.0) if hcp_actual is not None else 0.0,
                        0.0,
                    )
                    extras_pen = 140.0 + min(80.0, 20.0 * extras_strength)
                    penalty += extras_pen
                    reasons.append(
                        "CONSTRUCTIVE_BELOW_GAME_PASS_PENALTY: partner made the last below-game contract bid, "
                        "constructive continuations exist, and self has extras "
                        f"({tp_actual:.0f} Total_Points / {hcp_actual:.0f} HCP, -{extras_pen:.0f})"
                    )
    except Exception:
        pass

    return penalty, "; ".join(reasons) if reasons else None


def compute_forced_non_pass_policy(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    has_non_pass_choice: bool,
    self_total_points: float | None,
    self_hcp: float | None,
    self_hand: str | None = None,
    pass_agg_expr: list[str] | None = None,
) -> dict[str, Any]:
    """Return hard-block policy for impossible early signoffs.

    This is the shared "can't pass" rule for early uncontested shapes:
    - `partner opening - pass - ?`
    - `opening - pass - response - pass - ?`

    Policy:
    - If partner opened `1NT` or `2NT`, treat that as at least 15 / 20 HCP.
      Hard-block responder pass when partnership HCP floor reaches 25.
    - If partner opened a one-level suit, treat that as at least 13 Total_Points.
      Hard-block responder pass when partnership TP floor reaches:
      - 26 for a major opening
      - 29 for a minor opening
    - If point floors do not reach game, a second hard-block still applies when
      responder can infer game-level tricks from the opening plus hand pattern.
      This uses a synthetic opener profile derived from the opening bid and the
      same `estimate_partnership_tricks()` heuristic used elsewhere in the
      scorer. The intent is to catch distributional game hands that should not
      be allowed to die with a second pass.
    - If opener is on rebid after an uncontested response and the current Pass
      node's explicit HCP / Total_Points caps are violated, hard-block pass.
      This catches BT range gaps where opener is "too strong to sign off" but
      the natural continuation buckets are too narrow.

    The block only applies when a non-pass continuation exists. If the BT truly
    offers no non-pass call, pass remains available so the auction can continue
    with the least-bad legal action elsewhere in the scorer.

    Additionally, if the current Pass node itself still carries a forcing flag
    like `Forcing_One_Round`, treat that as an active partnership obligation and
    hard-block pass while a legal non-pass continuation exists.
    """
    out: dict[str, Any] = {
        "hard_block": False,
        "reason": None,
        "context": None,
    }
    if not acting_direction or not has_non_pass_choice:
        return out

    try:
        forcing_tokens = {
            str(tok).strip().upper()
            for tok in list(pass_agg_expr or [])
            if str(tok).strip()
        }
        blackwood_followup_ctx = extract_blackwood_asker_followup_pass_context(
            auction_tokens=list(auction_tokens or []),
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        if blackwood_followup_ctx is not None:
            out["hard_block"] = True
            out["reason"] = (
                "FORCED_NON_PASS_BLACKWOOD_FOLLOWUP: original 4NT asker cannot pass after "
                f"partner's {blackwood_followup_ctx['response_bid']} ace response while legal "
                "non-pass continuations exist"
            )
            out["context"] = {
                "metric": "blackwood_followup",
                "ask_bid": str(blackwood_followup_ctx.get("ask_bid") or ""),
                "response_bid": str(blackwood_followup_ctx.get("response_bid") or ""),
                "acting_direction": str(blackwood_followup_ctx.get("acting_direction") or ""),
            }
            return out
        if "FORCING_ONE_ROUND" in forcing_tokens:
            out["hard_block"] = True
            out["reason"] = (
                "FORCED_NON_PASS_FORCING_ONE_ROUND: current Pass node still carries "
                "Forcing_One_Round, so pass cannot override a live forcing auction "
                "while legal non-pass continuations exist"
            )
            out["context"] = {
                "metric": "forcing_flag",
                "forcing_flag": "Forcing_One_Round",
                "pass_agg_expr": [str(tok) for tok in list(pass_agg_expr or []) if str(tok).strip()],
            }
            return out

        opener_minor_clarification_ctx = extract_opener_minor_clarification_pass_context(
            auction_tokens=list(auction_tokens or []),
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        if opener_minor_clarification_ctx is not None:
            out["hard_block"] = True
            out["reason"] = (
                "FORCED_NON_PASS_OPENER_MINOR_CLARIFICATION: opener cannot pass after "
                f"{opener_minor_clarification_ctx['opening_bid']}-P-"
                f"{opener_minor_clarification_ctx['response_bid']}-P-"
                f"{opener_minor_clarification_ctx['opener_rebid_bid']}-P-"
                f"{opener_minor_clarification_ctx['partner_bid']}-P while legal "
                "non-pass continuations exist"
            )
            out["context"] = {
                "metric": "opener_minor_clarification",
                "opening_bid": str(opener_minor_clarification_ctx.get("opening_bid") or ""),
                "response_bid": str(opener_minor_clarification_ctx.get("response_bid") or ""),
                "opener_rebid_bid": str(opener_minor_clarification_ctx.get("opener_rebid_bid") or ""),
                "partner_bid": str(opener_minor_clarification_ctx.get("partner_bid") or ""),
                "acting_direction": str(opener_minor_clarification_ctx.get("acting_direction") or ""),
            }
            return out

        second_pass_ctx = extract_second_pass_opening_context(
            auction_tokens=list(auction_tokens or []),
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        if second_pass_ctx is None:
            opener_rebid_ctx = extract_opener_rebid_pass_context(
                auction_tokens=list(auction_tokens or []),
                acting_direction=acting_direction,
                dealer_actual=dealer_actual,
            )
            if opener_rebid_ctx is None:
                return out

            tp_actual = float(self_total_points) if isinstance(self_total_points, (int, float)) else None
            hcp_actual = float(self_hcp) if isinstance(self_hcp, (int, float)) else None
            tp_excess = _upper_bound_excess_from_exprs("TOTAL_POINTS", tp_actual, pass_agg_expr)
            hcp_excess = _upper_bound_excess_from_exprs("HCP", hcp_actual, pass_agg_expr)
            if tp_excess is None and hcp_excess is None:
                return out

            reason_bits: list[str] = []
            context: dict[str, Any] = {
                "opening_bid": str(opener_rebid_ctx.get("opening_bid") or ""),
                "response_bid": str(opener_rebid_ctx.get("response_bid") or ""),
                "metric": "rebid_gap",
            }
            if tp_excess is not None and tp_actual is not None:
                tp_cap, tp_over = tp_excess
                reason_bits.append(
                    f"Pass node caps Total_Points at {tp_cap:.0f} but self has {tp_actual:.0f}"
                )
                context["self_total_points"] = float(tp_actual)
                context["tp_cap"] = float(tp_cap)
                context["tp_over"] = float(tp_over)
            if hcp_excess is not None and hcp_actual is not None:
                hcp_cap, hcp_over = hcp_excess
                reason_bits.append(
                    f"Pass node caps HCP at {hcp_cap:.0f} but self has {hcp_actual:.0f}"
                )
                context["self_hcp"] = float(hcp_actual)
                context["hcp_cap"] = float(hcp_cap)
                context["hcp_over"] = float(hcp_over)

            out["hard_block"] = True
            out["reason"] = (
                "FORCED_NON_PASS_REBID_GAP: opener cannot sign off after "
                f"{context['opening_bid']}-P-{context['response_bid']}-P when "
                + "; ".join(reason_bits)
                + " and legal non-pass continuations exist"
            )
            out["context"] = context
            return out

        opening_bid = str(second_pass_ctx.get("opening_bid") or "")
        opening_level = _parse_bid_level(opening_bid)
        opening_strain = _parse_bid_strain(opening_bid)
        if opening_level is None or opening_strain is None:
            return out

        if opening_level in (1, 2) and opening_strain == "NT":
            opener_hcp_floor = 15.0 if opening_level == 1 else 20.0
            if self_hcp is None:
                combined_hcp_floor = None
            else:
                combined_hcp_floor = float(self_hcp) + float(opener_hcp_floor)
                if combined_hcp_floor >= _FORCED_NON_PASS_GAME_HCP_NT:
                    out["hard_block"] = True
                    out["reason"] = (
                        "FORCED_NON_PASS_GAME_VALUES: responder cannot make the second pass after "
                        f"partner's {opening_bid} opening when partnership HCP floor "
                        f"{combined_hcp_floor:.0f} reaches NT game threshold "
                        f"{_FORCED_NON_PASS_GAME_HCP_NT:.0f}"
                    )
                    out["context"] = {
                        "opening_bid": opening_bid,
                        "metric": "HCP",
                        "self_hcp": float(self_hcp),
                        "partner_floor": float(opener_hcp_floor),
                        "combined_floor": float(combined_hcp_floor),
                        "required": float(_FORCED_NON_PASS_GAME_HCP_NT),
                    }
                    return out

            if self_hand:
                _nt_hcp_hist = (
                    {"15": 3, "16": 4, "17": 3}
                    if opening_level == 1
                    else {"20": 3, "21": 4, "22": 3}
                )
                _balanced_sl = {"2": 2, "3": 6, "4": 2}
                _nt_tricks = estimate_partnership_tricks(
                    self_hand=self_hand,
                    partner_hcp_hist=_nt_hcp_hist,
                    partner_sl_hists={s: dict(_balanced_sl) for s in ("S", "H", "D", "C")},
                    fit_us_hists={},
                    strain="NT",
                )
                _est_nt = _nt_tricks.get("est_tricks")
                if _est_nt is not None and float(_est_nt) >= 8.5:
                    out["hard_block"] = True
                    out["reason"] = (
                        "FORCED_NON_PASS_GAME_TRICKS: responder cannot make the second pass after "
                        f"partner's {opening_bid} opening when inferred NT tricks "
                        f"{float(_est_nt):.1f} reach game range"
                    )
                    out["context"] = {
                        "opening_bid": opening_bid,
                        "metric": "tricks",
                        "strain": "NT",
                        "est_tricks": float(_est_nt),
                        "required": 9.0,
                        "combined_hcp_floor": float(combined_hcp_floor) if combined_hcp_floor is not None else None,
                    }
                    return out
            return out

        if opening_level == 1 and opening_strain in ("S", "H", "D", "C"):
            opener_tp_floor = 13.0
            combined_tp_floor = (
                float(self_total_points) + float(opener_tp_floor)
                if self_total_points is not None
                else None
            )
            required_tp = (
                _FORCED_NON_PASS_GAME_TP_MAJOR
                if opening_strain in ("S", "H")
                else _FORCED_NON_PASS_GAME_TP_MINOR
            )
            if combined_tp_floor is not None and combined_tp_floor >= required_tp:
                game_label = "major" if opening_strain in ("S", "H") else "minor"
                out["hard_block"] = True
                out["reason"] = (
                    "FORCED_NON_PASS_GAME_VALUES: responder cannot make the second pass after "
                    f"partner's {opening_bid} opening when partnership Total_Points floor "
                    f"{combined_tp_floor:.0f} reaches {game_label} game threshold {required_tp:.0f}"
                )
                out["context"] = {
                    "opening_bid": opening_bid,
                    "metric": "Total_Points",
                    "self_total_points": float(self_total_points) if self_total_points is not None else None,
                    "partner_floor": float(opener_tp_floor),
                    "combined_floor": float(combined_tp_floor),
                    "required": float(required_tp),
                }
                return out

            if self_hand:
                _hand_parts = str(self_hand or "").strip().split(".")
                if len(_hand_parts) == 4:
                    _self_lengths = {
                        "S": len(str(_hand_parts[0] or "")),
                        "H": len(str(_hand_parts[1] or "")),
                        "D": len(str(_hand_parts[2] or "")),
                        "C": len(str(_hand_parts[3] or "")),
                    }
                else:
                    _self_lengths = {}
                _support_len = int(_self_lengths.get(opening_strain, 0))
                _need_support = 4 if opening_strain in ("S", "H") else 5
                if _support_len >= _need_support:
                    _partner_hcp_hist = {"13": 4, "14": 3, "15": 2, "16": 1}
                    _partner_len_floor = (
                        5 if opening_strain in ("S", "H")
                        else (4 if opening_strain == "D" else 3)
                    )
                    _partner_sl_hists = {
                        "S": {"2": 2, "3": 4, "4": 2},
                        "H": {"2": 2, "3": 4, "4": 2},
                        "D": {"2": 2, "3": 4, "4": 2},
                        "C": {"2": 2, "3": 4, "4": 2},
                    }
                    _partner_sl_hists[opening_strain] = {
                        str(_partner_len_floor): 4,
                        str(_partner_len_floor + 1): 3,
                        str(_partner_len_floor + 2): 1,
                    }
                    _fit_floor = _support_len + _partner_len_floor
                    _fit_hist = {
                        str(_fit_floor): 4,
                        str(_fit_floor + 1): 2,
                        str(_fit_floor + 2): 1,
                    }
                    _trick_res = estimate_partnership_tricks(
                        self_hand=self_hand,
                        partner_hcp_hist=_partner_hcp_hist,
                        partner_sl_hists=_partner_sl_hists,
                        fit_us_hists={opening_strain: _fit_hist},
                        strain=opening_strain,
                    )
                    _est_suit_tricks = _trick_res.get("est_tricks")
                    _required_tricks = 10.0 if opening_strain in ("S", "H") else 11.0
                    if _est_suit_tricks is not None and float(_est_suit_tricks) >= _required_tricks - 0.5:
                        out["hard_block"] = True
                        out["reason"] = (
                            "FORCED_NON_PASS_GAME_TRICKS: responder cannot make the second pass after "
                            f"partner's {opening_bid} opening when inferred {opening_strain} tricks "
                            f"{float(_est_suit_tricks):.1f} reach game range"
                        )
                        out["context"] = {
                            "opening_bid": opening_bid,
                            "metric": "tricks",
                            "strain": opening_strain,
                            "support_len": _support_len,
                            "partner_len_floor": _partner_len_floor,
                            "est_tricks": float(_est_suit_tricks),
                            "required": float(_required_tricks),
                            "combined_tp_floor": float(combined_tp_floor) if combined_tp_floor is not None else None,
                        }
                        return out
    except Exception:
        return out

    return out


def compute_post_game_slam_gate_adjustment(
    *,
    bid_text: str,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    self_total_points: float | None,
    partner_tp_hist: dict[str, Any] | None,
) -> dict[str, Any]:
    """Penalty for slam-commit / slam-try continuation without sufficient evidence.

    Originally this gate only activated after our side already had a game on
    the table. We now also gate pre-game slam-oriented commitments:
    - 5M (5H/5S) when our side has already shown that same major
    - 5NT (slam-try / choice-of-slams style action)
    - Any direct 6/7-level commitment before game sign-off
    """
    out: dict[str, Any] = {
        "penalty": 0.0,
        "reason": None,
        "game_contract_on_table": False,
        "p_slam_tp_ge_33": None,
        "combined_tp_mean": None,
    }
    if not acting_direction:
        return out

    b = str(bid_text or "").strip().upper()
    m_bid = re.match(r"^([1-7])\s*(NT|N|[CDHS])", b)
    if not m_bid:
        return out
    b_lvl = int(m_bid.group(1))
    b_st = "N" if m_bid.group(2).upper() in ("N", "NT") else m_bid.group(2).upper()
    is_slam_explore = b in ("4N", "4NT", "5N", "5NT") or b_lvl >= 5
    if not is_slam_explore:
        return out

    directions = ["N", "E", "S", "W"]
    d = str(dealer_actual or "N").upper()
    dealer_idx = directions.index(d) if d in directions else 0

    def _token_bidder_dir(token_idx: int) -> str:
        return directions[(dealer_idx + int(token_idx)) % 4]

    def _strain_rank(st: str) -> int:
        return {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}.get(str(st or "").upper(), -1)

    # Highest contract on table + bidder side
    best: dict[str, Any] | None = None
    for i, tk in enumerate(auction_tokens):
        s = str(tk or "").strip().upper()
        m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
        if not m:
            continue
        lvl = int(m.group(1))
        st = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
        if best is None or lvl > int(best["level"]) or (lvl == int(best["level"]) and _strain_rank(st) > _strain_rank(str(best["strain"]))):
            best = {"level": lvl, "strain": st, "token_idx": i}
    if best is None:
        return out

    bidder_dir = _token_bidder_dir(int(best["token_idx"]))
    bidder_side = "NS" if bidder_dir in ("N", "S") else "EW"
    acting_side = "NS" if acting_direction in ("N", "S") else "EW"
    if bidder_side != acting_side:
        return out

    top_level = int(best["level"])
    top_strain = str(best["strain"])
    game_min = 3 if top_strain == "N" else (4 if top_strain in ("H", "S") else 5)
    game_contract_on_table = top_level >= game_min
    out["game_contract_on_table"] = bool(game_contract_on_table)
    # Backstops for pre-game slam-oriented actions before game is explicitly
    # on the table.
    is_pre_game_major_slam_try = False
    is_pre_game_nt_slam_try = False
    is_pre_game_direct_slam_commit = False
    pre_game_gate_tag: str | None = None
    if not game_contract_on_table and b_lvl == 5 and b_st in ("H", "S"):
        try:
            showed_same_major = False
            for i, tk in enumerate(auction_tokens):
                s = str(tk or "").strip().upper()
                m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
                if not m:
                    continue
                token_lvl = int(m.group(1))
                token_st = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
                token_dir = _token_bidder_dir(i)
                token_side = "NS" if token_dir in ("N", "S") else "EW"
                if token_side == acting_side and token_st == b_st and token_lvl <= 4:
                    showed_same_major = True
                    break
            is_pre_game_major_slam_try = bool(showed_same_major)
        except Exception:
            is_pre_game_major_slam_try = False
    if not game_contract_on_table and b_lvl == 5 and b_st == "N":
        is_pre_game_nt_slam_try = True
    if not game_contract_on_table and b_lvl >= 6:
        is_pre_game_direct_slam_commit = True

    if is_pre_game_major_slam_try:
        pre_game_gate_tag = "PRE_GAME_5M_SLAM_TRY_GATE"
    elif is_pre_game_nt_slam_try:
        pre_game_gate_tag = "PRE_GAME_5NT_SLAM_TRY_GATE"
    elif is_pre_game_direct_slam_commit:
        pre_game_gate_tag = "PRE_GAME_DIRECT_SLAM_COMMIT_GATE"

    if not game_contract_on_table and pre_game_gate_tag is None:
        return out

    partner_mean_tp = expected_from_hist(partner_tp_hist) if partner_tp_hist is not None else None
    combined_tp_mean = None
    if self_total_points is not None and partner_mean_tp is not None:
        combined_tp_mean = float(self_total_points) + float(partner_mean_tp)
    out["combined_tp_mean"] = combined_tp_mean

    p_slam_tp_ge_33 = None
    if self_total_points is not None and isinstance(partner_tp_hist, dict):
        needed_partner_tp = max(0, int(math.ceil(33.0 - float(self_total_points))))
        tot = 0.0
        ok = 0.0
        for k, v in partner_tp_hist.items():
            try:
                tp = int(k)
                cnt = float(v)
            except Exception:
                continue
            if cnt <= 0:
                continue
            tot += cnt
            if tp >= needed_partner_tp:
                ok += cnt
        if tot > 0:
            p_slam_tp_ge_33 = ok / tot
    out["p_slam_tp_ge_33"] = p_slam_tp_ge_33

    p_min = 0.45
    penalty = 0.0
    if p_slam_tp_ge_33 is None:
        penalty += 140.0
    elif float(p_slam_tp_ge_33) < p_min:
        penalty += (p_min - float(p_slam_tp_ge_33)) * 500.0
    if combined_tp_mean is not None and float(combined_tp_mean) < 32.0:
        penalty += (32.0 - float(combined_tp_mean)) * 25.0
    if is_pre_game_major_slam_try:
        # Extra friction for direct 5M commitments before game sign-off.
        penalty += 100.0
    if is_pre_game_nt_slam_try:
        # Extra friction for pre-game 5NT slam-oriented jump.
        penalty += 100.0
    if is_pre_game_direct_slam_commit:
        # Stronger friction for direct 6/7 commitment before game sign-off.
        penalty += 140.0
    out["penalty"] = float(penalty)
    if penalty > 0:
        p_txt = f"{p_slam_tp_ge_33:.2f}" if p_slam_tp_ge_33 is not None else "NA"
        c_txt = f"{combined_tp_mean:.1f}" if combined_tp_mean is not None else "NA"
        gate_tag = pre_game_gate_tag or "POST_GAME_SLAM_GATE"
        out["reason"] = (
            f"{gate_tag}: slam explore {b} without enough evidence "
            f"(p33={p_txt}, combinedTP={c_txt}) (-{penalty:.0f})"
        )
    return out


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
    self_aces: int | None = None,
    self_helpful_voids: int | None = None,
    self_trump_king: bool | None = None,
    partner_known_control_floor: float | None = None,
    partner_known_control_ceiling: float | None = None,
    partner_known_control_expected: float | None = None,
    partner_known_control_source: str | None = None,
    partner_known_control_kind: str | None = None,
    partner_known_ace_floor: float | None = None,
    partner_known_ace_ceiling: float | None = None,
    partner_known_ace_expected: float | None = None,
    partner_known_ace_source: str | None = None,
    same_strain_blackwood_slam_bypass: bool = False,
    same_strain_blackwood_slam_reason: str | None = None,
    self_suit_lengths: dict[str, int] | None = None,
    fit_us_hist: dict[str, Any] | None = None,
    bt_acting_criteria: list[str] | None = None,
    is_reopening: bool = False,
    w_first_round_control: float = 75.0,
    w_weak_hand: float = 40.0,
    w_natural_suit_shortfall: float = 45.0,
    w_underspecified_strain: float = 35.0,
    w_reopen_jump: float = 80.0,
    w_reopen_low_fit: float = 100.0,
    w_non_sac_overbid_hard: float = 350.0,
    enable_tp_shortfall_check: bool = True,
    enable_tricks_shortfall_check: bool = True,
    enable_underbid_checks: bool = True,
    is_raise_of_partner_suit: bool = False,
    opp_shown_strains: set[str] | None = None,
    w_opp_suit_trespass: float = 200.0,
    debug_equivalence_bypass: bool = False,
    same_strain_prev_level: int | None = None,
    same_strain_support_level_floor: int | None = None,
    same_strain_support_level_ceiling: int | None = None,
    same_strain_point_type: str | None = None,
    is_transfer_acceptance: bool = False,
    is_transfer_response: bool = False,
) -> tuple[float, list[str]]:
    """Compute a non-negative guardrail penalty to subtract from the bid score.

    Additional semantic checks can be enabled by providing suit lengths,
    fit histograms, acting criteria, and reopening context.

    **Overbid checks** (bid level is too high for the hand):

    1. **OVERBID_VS_PAR** – bid level exceeds the most likely par contract
       level.  Catches bids like 5C when par says 3NT.
    2. **TP_SHORTFALL** – combined SELF + E[PARTNER] total points fall below
       the standard minimum for the bid level.
    3. **WEAK_HAND_OVERBID** – the acting player's own TP is below a
       minimum floor for the bid level.  Catches weak hands (e.g. 6 TP)
       making high-level bids that only exist in the BT because of
       strong distributional hands in the population.
    4. **NEG_PAR_HIGH_LEVEL** – at the 4-level or above, the acting-side par
       mean is negative (contract is expected to go down on average).
    5. **STRAIN_INEFFICIENCY** – bidding 4m/5m when the par contract is 3NT.
       3NT is almost always preferred.  4m is strictly worse (partscore vs
       game + extra trick): 1.5× base.  5m needs 2 extra tricks: 1.0× base.
    6. **TRICKS_SHORTFALL** – estimated partnership tricks fall below the
       number required for the contract level (level + 6).
    7. **INSUFFICIENT_FIRST_ROUND_CONTROLS** – bid level exceeds
       3 + estimated partnership first-round controls.  A first-round
       control is an ace or (in suit contracts) a void in a suit where
       the player doesn't hold the ace.  Without enough controls,
       opponents can cash aces to set the contract.  Rule:
       max safe level = 3 + first_round_controls.  Applies at level 4+.
       Sacrifice-discounted when par belongs to opponents.

    **Underbid checks** (bid level is too low for the hand):

    8. **UNDERBID_VS_PAR** – bid level is *below* the most likely par contract
       that *our side* owns.  Catches stopping at 2H when par says 4H.
    9. **TP_SURPLUS_UNDERBID** – partnership TP exceeds the threshold for a
       higher level consistent with par (game bonus being left on the table).

    **Sacrifice exemption**:

    When the most likely par contract belongs to the *opponents*, overbid
    checks (1-4, 6-7) are discounted by ``sacrifice_discount`` (default 70%),
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
        Base penalty for minor suit when par is 3NT (default 60).
        4m (partscore vs game) = ×1.5; 5m (2 extra tricks) = ×1.0.
    sacrifice_discount : float
        Fraction by which overbid penalties are reduced when the par contract
        belongs to opponents (default 0.7 = 70% discount).
    est_tricks : float, optional
        Estimated partnership tricks from the heuristic trick model.
        When provided, enables the TRICKS_SHORTFALL check.
    w_tricks_shortfall : float
        Penalty per estimated trick below the required tricks for the
        contract level (default 30).
    self_aces : int, optional
        Number of aces in the acting player's hand.  Enables the
        INSUFFICIENT_FIRST_ROUND_CONTROLS check.
    self_helpful_voids : int, optional
        Number of voids in suits where the player does NOT hold the ace.
        In suit contracts (non-NT), each helpful void is a first-round
        control (ruff prevents opponents from cashing that ace).
    self_trump_king : bool, optional
        Whether the acting player holds the king of the candidate trump suit.
        Relevant only for suit slams, where the trump king is the 5th control.
    partner_known_control_floor : float, optional
        Explicit lower bound on partner's controls, typically inferred from a
        Blackwood or RKCB response. In NT this is aces; in suit contracts this
        can be keycards (aces plus trump king).
    partner_known_control_ceiling : float, optional
        Explicit upper bound on partner's controls, typically inferred from a
        Blackwood or RKCB response.
    partner_known_control_expected : float, optional
        Preferred expected partner control count when convention evidence should
        be treated as more slam-positive than a flat midpoint.
    partner_known_control_source : str, optional
        Human-readable source label for explicit partner control knowledge.
    partner_known_control_kind : str, optional
        Label for the explicit partner control unit, e.g. ``"aces"`` or
        ``"keycards"``.
    partner_known_ace_floor : float, optional
        Legacy alias for explicit partner ace knowledge.
    partner_known_ace_ceiling : float, optional
        Legacy alias for explicit partner ace knowledge.
    partner_known_ace_expected : float, optional
        Legacy alias for explicit partner ace knowledge.
    same_strain_blackwood_slam_bypass : bool
        When True, treat same-strain post-Blackwood slam continuations as
        convention-supported value pushes rather than generic overbids.
    w_first_round_control : float
        Penalty per level the bid exceeds the max safe level (3 + controls).
        Default 75.
    w_weak_hand : float
        Penalty per self-TP below the floor for the bid level (default 40).
        Catches very weak hands making high-level bids.  Sacrifice-
        discounted.
    debug_equivalence_bypass : bool
        If True, append a debug-only reason when 3NT <-> 4M equivalence
        suppresses OVERBID/UNDERBID penalties.

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
    def _contract_rank_from_level_strain(level: int | None, strain: str | None) -> int | None:
        if level is None or strain is None:
            return None
        order = {"C": 0, "D": 1, "H": 2, "S": 3, "NT": 4}
        o = order.get(strain)
        if o is None:
            return None
        return (int(level) - 1) * 5 + int(o)

    def _is_nt_major_game_equivalent(
        level_a: int | None, strain_a: str | None, level_b: int | None, strain_b: str | None
    ) -> bool:
        # 3NT and 4M are close game targets; avoid penalizing this tradeoff as a
        # generic level error in OVERBID/UNDERBID checks.
        a_is_3nt = level_a == 3 and strain_a == "NT"
        b_is_3nt = level_b == 3 and strain_b == "NT"
        a_is_4m = level_a == 4 and strain_a in ("H", "S")
        b_is_4m = level_b == 4 and strain_b in ("H", "S")
        return (a_is_3nt and b_is_4m) or (b_is_3nt and a_is_4m)

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
        if top_pair in {"OPP", "OPPONENT", "THEM"}:
            par_is_opponents = True
        elif top_pair in {"SELF", "US", "OURS"}:
            par_is_opponents = False
        else:
            par_is_opponents = bool(top_pair and top_pair != acting_pair)

    bid_rank = _contract_rank_from_level_strain(bid_level, bid_strain)
    top_rank = _contract_rank_from_level_strain(top_level, top_strain)
    nt_major_game_equiv = _is_nt_major_game_equivalent(bid_level, bid_strain, top_level, top_strain)
    if debug_equivalence_bypass and nt_major_game_equiv and not par_is_opponents:
        reasons.append(
            "EQUIV_BYPASS_3NT_4M: treating 3NT and 4M as equivalent game targets; "
            "skipping generic OVERBID/UNDERBID level penalties (0)"
        )

    # Sacrifice discount factor: 1.0 = full penalty, 0.0 = no penalty.
    # When opponents own the par, overbid penalties are reduced.
    sac_factor = 1.0
    if par_is_opponents and sacrifice_discount > 0:
        sac_factor = max(0.0, 1.0 - float(sacrifice_discount))

    # Combined partnership TP (used by multiple checks).
    combined_tp: float | None = None
    if self_total_points is not None and partner_expected_tp is not None:
        combined_tp = float(self_total_points) + float(partner_expected_tp)

    def _prob_fit_8plus_for_strain(fit_hist: dict[str, Any] | None, strain: str | None) -> float | None:
        if fit_hist is None or strain is None:
            return None
        try:
            s = str(strain).upper()
            if s == "NT":
                return None
            h = fit_hist.get(s)
            if not isinstance(h, dict):
                return None
            total = 0.0
            good = 0.0
            for k, v in h.items():
                try:
                    fit_len = int(k)
                    cnt = float(v)
                except Exception:
                    continue
                if cnt <= 0:
                    continue
                total += cnt
                if fit_len >= 8:
                    good += cnt
            if total <= 0:
                return None
            return good / total
        except Exception:
            return None

    def _has_strain_length_criterion(criteria: list[str] | None, strain: str | None) -> bool:
        if not criteria or strain is None:
            return False
        try:
            s = str(strain).upper()
            if s == "NT":
                return False
            pat = re.compile(rf"\bSL_{re.escape(s)}\b", re.IGNORECASE)
            for c in criteria:
                if pat.search(str(c or "")):
                    return True
            return False
        except Exception:
            return False

    def _is_artificial_major_probe(criteria: list[str] | None, bid_text: str) -> bool:
        """Detect artificial minor bids that should not inherit natural-suit heuristics.

        This covers:
        - Stayman-like club asks that show majors rather than clubs.
        - Strong forcing 2C openings (`Forcing_To_2N`) that are explicitly
          conventional and should never be vetoed for lacking club length.
        """
        try:
            b = str(bid_text or "").strip().upper()
            if b not in ("2C", "3C"):
                return False
            crits = [str(c or "").upper() for c in list(criteria or [])]
            if not crits:
                return False
            joined = " ".join(crits)

            def _has_positive_major_marker(suit: str) -> bool:
                return bool(
                    re.search(rf"\bSL_{suit}\s*(>=|==)\s*[34]\b", joined)
                    or re.search(rf"\bSL_{suit}\s*>\s*3\b", joined)
                )

            # Stayman-like asks positively probe for majors and do not constrain clubs.
            has_major_probe = _has_positive_major_marker("H") and _has_positive_major_marker("S")
            has_club_constraint = bool(re.search(r"\bSL_C\b", joined))
            if has_major_probe and not has_club_constraint:
                return True

            # Strong artificial 2C openings are forcing and not natural clubs.
            if b == "2C" and "FORCING_TO_2N" in joined and not has_club_constraint:
                return True

            return False
        except Exception:
            return False

    def _is_new_minor_forcing_like(criteria: list[str] | None, bid_text: str) -> bool:
        """Detect minor-suit convention bids that show a major, not the minor.

        This covers explicit forcing/checkback-style BT rows like `3D` with
        criteria such as `SL_H >= 5`, `Forcing_To_3N`, and no `SL_D`
        constraint. Those bids behave like artificial/new-minor style asks and
        should not inherit the generic "natural level-3 suit bid means 6+
        cards in the bid suit" heuristic.

        Important: major-showing minor rows are *not* automatically artificial.
        Some stray BT rows mention a major but are really bogus natural minor
        continuations with missing strain constraints. Those should still be
        penalized by the natural-suit and underspecified-strain guardrails.
        """
        try:
            b = str(bid_text or "").strip().upper()
            if b not in ("2C", "2D", "3C", "3D"):
                return False
            crits = [str(c or "").upper() for c in list(criteria or [])]
            if not crits:
                return False
            joined = " ".join(crits)
            bid_strain = _parse_bid_strain(b)
            if bid_strain not in ("C", "D"):
                return False

            shows_major = bool(re.search(r"\bSL_[HS]\b", joined))
            constrains_bid_minor = bool(re.search(rf"\bSL_{re.escape(bid_strain)}\b", joined))
            has_forcing_or_probe_marker = bool(
                re.search(r"\bFORCING(?:_[A-Z0-9]+)*\b", joined)
                or re.search(r"\b(?:CHECKBACK|NEW_MINOR|ARTIFICIAL)\b", joined)
            )
            return bool(
                shows_major
                and not constrains_bid_minor
                and has_forcing_or_probe_marker
            )
        except Exception:
            return False

    def _tp_hist_confidence(tp_hist: dict[str, Any] | None) -> float:
        """Estimate how well partner points are constrained (0..1)."""
        if not isinstance(tp_hist, dict) or not tp_hist:
            return 0.0
        total = 0.0
        probs: list[tuple[float, float]] = []
        for k, v in tp_hist.items():
            try:
                tp = float(k)
                cnt = float(v)
            except Exception:
                continue
            if cnt <= 0:
                continue
            total += cnt
            probs.append((tp, cnt))
        if total <= 0:
            return 0.0
        mean = sum(tp * cnt for tp, cnt in probs) / total
        var = sum(((tp - mean) ** 2) * cnt for tp, cnt in probs) / total
        std = math.sqrt(max(0.0, var))
        pmax = max(cnt / total for _, cnt in probs)
        std_score = max(0.0, min(1.0, (6.0 - std) / 6.0))
        pmax_score = max(0.0, min(1.0, (pmax - 0.08) / 0.32))
        return max(0.0, min(1.0, 0.5 * std_score + 0.5 * pmax_score))

    # Hard policy with confidence-gated exception:
    # no non-sacrifice overbids unless partner is sufficiently constrained and
    # value-push evidence supports the target level.
    support_major_game_equiv = bool(
        is_raise_of_partner_suit and bid_strain in ("H", "S") and nt_major_game_equiv and not par_is_opponents
    )
    same_strain_floor_supports_bid = bool(
        same_strain_prev_level is not None
        and same_strain_support_level_floor is not None
        and bid_level is not None
        and int(bid_level) > int(same_strain_prev_level)
        and int(bid_level) <= int(same_strain_support_level_floor)
    )
    same_strain_ceiling_supports_bid = bool(
        same_strain_prev_level is not None
        and same_strain_support_level_ceiling is not None
        and bid_level is not None
        and int(bid_level) > int(same_strain_prev_level)
        and int(bid_level) <= int(same_strain_support_level_ceiling)
    )
    same_strain_confident_value_push = bool(
        same_strain_floor_supports_bid and same_strain_ceiling_supports_bid
    )
    same_strain_supported_overbid = bool(
        same_strain_confident_value_push
        and top_strain is not None
        and bid_strain == top_strain
        and not par_is_opponents
    )
    blackwood_same_strain_supported_overbid = bool(
        same_strain_blackwood_slam_bypass
        and top_strain is not None
        and bid_strain == top_strain
        and not par_is_opponents
    )

    if top_rank is not None and bid_rank is not None and bid_rank > top_rank and not par_is_opponents:
        rank_gap = int(bid_rank) - int(top_rank)
        tp_conf = _tp_hist_confidence(partner_tp_hist)
        fit_conf = None
        p_fit8 = _prob_fit_8plus_for_strain(fit_us_hist, bid_strain)
        if p_fit8 is not None:
            fit_conf = max(0.0, min(1.0, (float(p_fit8) - 0.35) / 0.45))
        partner_confidence = float(tp_conf) if fit_conf is None else float(0.6 * tp_conf + 0.4 * fit_conf)

        required_tp_for_bid = _TP_THRESHOLDS.get(bid_level, 20.0)
        required_tricks_for_bid = _TRICKS_REQUIRED.get(bid_level, bid_level + 6)
        tp_supports_level = bool(combined_tp is not None and float(combined_tp) >= float(required_tp_for_bid) - 1.0)
        tricks_support_level = bool(est_tricks is not None and float(est_tricks) >= float(required_tricks_for_bid) - 0.5)
        value_push_evidence = tp_supports_level or tricks_support_level or same_strain_confident_value_push
        allow_confident_value_push = bool(partner_confidence >= 0.75 and value_push_evidence)

        if support_major_game_equiv:
            reasons.append(
                "NO_NON_SAC_OVERBID_BYPASS: partner-major support context treats 3NT and 4M "
                "as equivalent game targets"
            )
        elif blackwood_same_strain_supported_overbid:
            reasons.append(
                f"NO_NON_SAC_OVERBID_BYPASS: {same_strain_blackwood_slam_reason or 'explicit Blackwood same-strain slam evidence'}"
            )
        elif same_strain_supported_overbid:
            point_type = str(same_strain_point_type or "points")
            reasons.append(
                f"SAME_STRAIN_RANGE_BYPASS: same-strain {point_type} floor/ceiling already "
                f"support {bid}; skipping hard non-sacrifice overbid block"
            )
        elif not allow_confident_value_push:
            # 1.0 at low confidence, down to 0.15 at very high confidence.
            hard_scale = max(0.15, 1.0 - 0.85 * float(partner_confidence))
            hard_p = max(0.0, float(rank_gap) * float(w_non_sac_overbid_hard) * hard_scale)
            if hard_p > 0:
                penalty += hard_p
                prob_s = f"{top_prob*100:.0f}%" if top_prob is not None else "?"
                reasons.append(
                    f"NO_NON_SAC_OVERBID: bid {bid} outranks par {top_contract} ({prob_s} likely) "
                    f"owned by {top_pair}; partner_conf={partner_confidence:.2f}, "
                    f"value_push_evidence={'yes' if value_push_evidence else 'no'} "
                    f"(rank gap {rank_gap}, -{hard_p:.0f})"
                )
        else:
            reasons.append(
                f"NO_NON_SAC_OVERBID_BYPASS: confident partner model (conf={partner_confidence:.2f}) "
                f"with value-push evidence allows controlled overbid"
            )

    # ==================================================================
    # OVERBID CHECKS
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. OVERBID_VS_PAR – bid level exceeds par contract level
    # ------------------------------------------------------------------
    if top_rank is not None and bid_rank is not None and bid_rank > top_rank and not (
        nt_major_game_equiv and not par_is_opponents
    ) and not same_strain_supported_overbid and not blackwood_same_strain_supported_overbid:
        top_level_i = int(top_level) if top_level is not None else None
        if top_level_i is None:
            overbid_levels = 0
        else:
            overbid_levels = int(bid_level) - top_level_i
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
        par_tricks = _TRICKS_REQUIRED.get(top_level_i) if top_level_i is not None else None
        reasons.append(
            f"{tag}: bid {bid} is {overbid_levels} level(s) above par "
            f"{top_contract} ({prob_s} likely); "
            f"{_TRICKS_REQUIRED.get(bid_level, '?')} tricks needed vs "
            f"{par_tricks if par_tricks is not None else '?'} for par "
            f"(-{p:.0f}){sac_note}"
        )
    elif top_rank is not None and bid_rank is not None and bid_rank > top_rank and (
        same_strain_supported_overbid or blackwood_same_strain_supported_overbid
    ):
        if blackwood_same_strain_supported_overbid:
            reasons.append(
                f"BLACKWOOD_SAME_STRAIN_OVERBID_BYPASS: {same_strain_blackwood_slam_reason or 'explicit Blackwood same-strain slam evidence'}"
            )
        else:
            point_type = str(same_strain_point_type or "points")
            reasons.append(
                f"SAME_STRAIN_RANGE_OVERBID_BYPASS: same-strain {point_type} floor/ceiling "
                f"support {bid}; skipping generic par-level overbid penalty"
            )

    # ------------------------------------------------------------------
    # 2. TP_SHORTFALL – partnership TP below standard minimum for level
    # ------------------------------------------------------------------
    if enable_tp_shortfall_check and combined_tp is not None:
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
    # 3. WEAK_HAND_OVERBID – acting player's own TP far too low for
    #    the bid level.  Even a raise of partner's suit requires some
    #    minimum contribution.  Catches weak hands (e.g. 5-6 TP) bidding
    #    at the 4+ level because a BT node with inflated statistics
    #    happened to match their hand pattern.
    # ------------------------------------------------------------------
    if self_total_points is not None and bid_level is not None:
        _self_tp_floor = _SELF_TP_FLOORS.get(bid_level)
        if _self_tp_floor is not None:
            _self_tp_gap = _self_tp_floor - float(self_total_points)
            if _self_tp_gap > 0:
                raw_p = _self_tp_gap * float(w_weak_hand)
                p = raw_p * sac_factor
                penalty += p
                sac_note = f" [sacrifice discount applied]" if sac_factor < 1.0 else ""
                reasons.append(
                    f"WEAK_HAND_OVERBID: self TP {self_total_points:.0f} is below "
                    f"floor {_self_tp_floor:.0f} for level {bid_level}; "
                    f"deficit {_self_tp_gap:.0f} (-{p:.0f}){sac_note}"
                )

    # ------------------------------------------------------------------
    # 4. NEG_PAR_HIGH_LEVEL – negative acting-par at level 4+
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
    # 5. STRAIN_INEFFICIENCY – 4m/5m when par is 3NT
    #
    #    Bridge reality: 3NT is almost always preferred over minor suits.
    #    - 4m is strictly worse: partscore (130 NV) vs game (400 NV),
    #      AND needs 1 more trick.  Greatly penalised (1.5× base).
    #    - 5m is usually worse: same game score but needs 2 extra tricks.
    #      Only correct when a suit is unstoppable AND hand has 2 aces
    #      (or a void covering the missing ace).  1.0× base.
    #    Future: reduce penalty when Phase2a shows no stopper.
    # ------------------------------------------------------------------
    if (
        bid_level is not None
        and bid_level >= 4
        and bid_strain in ("C", "D")
        and top_level is not None
        and top_strain == "NT"
        and top_level == 3
        and not par_is_opponents
    ):
        # Level 4: partscore vs game — strictly dominated by 3NT → 1.5×
        # Level 5: both game but 2 extra tricks needed         → 1.0×
        _strain_scale = 1.5 if bid_level == 4 else 1.0
        p = float(w_strain_ineff) * _strain_scale
        if p > 0:
            _req_tricks = _TRICKS_REQUIRED.get(bid_level, bid_level + 6)
            penalty += p
            reasons.append(
                f"STRAIN_INEFFICIENCY: {bid} needs {_req_tricks} tricks; "
                f"par {top_contract} needs only {_TRICKS_REQUIRED[3]} tricks "
                f"for comparable score (-{p:.0f})"
            )

    # ------------------------------------------------------------------
    # 6. TRICKS_SHORTFALL – estimated tricks below required for contract
    # ------------------------------------------------------------------
    if enable_tricks_shortfall_check and est_tricks is not None and bid_level is not None:
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

    # ------------------------------------------------------------------
    # 7. CONTROL / SLAM SAFETY CHECKS
    #
    #    Two related policies:
    #    - Levels 4-5 keep the older first-round-control heuristic:
    #      max safe level = 3 + first-round controls.
    #    - Slam bids (6/7) use explicit bridge-control targets:
    #      * 6NT needs 3 of 4 aces
    #      * 7NT needs 4 of 4 aces
    #      * suit small slam needs 4 of 5 controls
    #      * suit grand slam needs 5 of 5 controls
    #        where the 5th control is the king of trumps.
    #
    #    Helpful voids count as ace-equivalent controls only in suit contracts.
    # ------------------------------------------------------------------
    if (
        bid_level is not None
        and bid_level >= 4
        and self_aces is not None
    ):
        _hv = int(self_helpful_voids or 0)
        _is_suit = bid_strain in ("C", "D", "H", "S")

        # Self first-round controls: aces + helpful voids (suit only)
        _self_controls = int(self_aces) + (_hv if _is_suit else 0)

        _known_partner_floor = None
        _known_partner_ceiling = None
        _known_partner_expected = None
        try:
            _raw_partner_floor = (
                partner_known_control_floor
                if partner_known_control_floor is not None
                else partner_known_ace_floor
            )
            _raw_partner_ceiling = (
                partner_known_control_ceiling
                if partner_known_control_ceiling is not None
                else partner_known_ace_ceiling
            )
            _raw_partner_expected = (
                partner_known_control_expected
                if partner_known_control_expected is not None
                else partner_known_ace_expected
            )
            _partner_control_cap = 5.0 if _is_suit else 4.0
            if _raw_partner_floor is not None or _raw_partner_ceiling is not None:
                _fallback_floor = (
                    _raw_partner_ceiling if _raw_partner_ceiling is not None else 0.0
                )
                _fallback_ceiling = (
                    _raw_partner_floor if _raw_partner_floor is not None else 0.0
                )
                _lo = (
                    float(_raw_partner_floor)
                    if _raw_partner_floor is not None
                    else float(_fallback_floor)
                )
                _hi = (
                    float(_raw_partner_ceiling)
                    if _raw_partner_ceiling is not None
                    else float(_fallback_ceiling)
                )
                _known_partner_floor = max(0.0, min(_partner_control_cap, min(_lo, _hi)))
                _known_partner_ceiling = max(0.0, min(_partner_control_cap, max(_lo, _hi)))
                if _raw_partner_expected is not None:
                    _known_partner_expected = max(0.0, min(_partner_control_cap, float(_raw_partner_expected)))
        except Exception:
            _known_partner_floor = None
            _known_partner_ceiling = None
            _known_partner_expected = None

        # Estimate partner controls from TP histogram unless explicit convention
        # context already provides a control-count range.
        # Heuristic fallback: 4 aces = 16 HCP of 40 → ~1 ace per 10 HCP.
        # TP includes ~1 distribution point on average, so subtract 1.
        _est_partner_aces = 1.0  # default if no histogram or explicit range
        if _known_partner_floor is not None and _known_partner_ceiling is not None:
            if _known_partner_expected is not None:
                _est_partner_aces = float(_known_partner_expected)
            else:
                _est_partner_aces = 0.5 * (_known_partner_floor + _known_partner_ceiling)
        elif partner_tp_hist is not None:
            _e_tp = expected_from_hist(partner_tp_hist)
            if _e_tp is not None:
                _est_partner_hcp = max(0.0, float(_e_tp) - 1.0)
                _est_partner_aces = _est_partner_hcp / 10.0

        # Partner's aces are in suits where self doesn't have the ace.
        # Of (4 - self_aces) such suits, self may already control some
        # via voids (suit contracts).  Partner aces in void-controlled
        # suits are redundant.  Distribute partner's aces uniformly
        # among the (4 - self_aces) suits and count only those in the
        # uncontrolled remainder.
        _suits_without_self_ace = max(1, 4 - int(self_aces))
        _uncontrolled = max(0, _suits_without_self_ace - (_hv if _is_suit else 0))
        _partner_new = _est_partner_aces * (_uncontrolled / _suits_without_self_ace)

        _ctrl_parts: list[str] = []
        _ctrl_parts.append(f"self {self_aces} ace(s)")
        if _is_suit and _hv > 0:
            _ctrl_parts.append(f"{_hv} helpful void(s)")
        if _known_partner_floor is not None and _known_partner_ceiling is not None:
            _src = str(partner_known_control_source or partner_known_ace_source or "Blackwood")
            _partner_unit = str(
                partner_known_control_kind
                or ("controls" if _is_suit else "aces")
            ).strip().lower()
            if abs(_known_partner_floor - _known_partner_ceiling) < 1e-9:
                _ctrl_parts.append(
                    f"partner shown {_known_partner_floor:.0f} {_partner_unit} via {_src}"
                )
            else:
                _expectation_label = (
                    "slam-positive expected"
                    if _known_partner_expected is not None
                    else "midpoint"
                )
                _ctrl_parts.append(
                    f"partner shown {_known_partner_floor:.0f}-{_known_partner_ceiling:.0f} "
                    f"{_partner_unit} via {_src} ({_expectation_label} {_est_partner_aces:.1f})"
                )
        else:
            _ctrl_parts.append(f"est. partner ~{_est_partner_aces:.1f} aces")

        if bid_level >= 6:
            _self_trump = 1.0 if bool(self_trump_king) and _is_suit else 0.0
            if _is_suit and _self_trump > 0:
                _ctrl_parts.append("self trump king")
            _slam_controls = float(self_aces) + (_hv if _is_suit else 0.0) + float(_partner_new) + _self_trump
            _required_controls = 4.0 if int(bid_level) == 6 and _is_suit else 5.0
            if not _is_suit:
                _required_controls = 3.0 if int(bid_level) == 6 else 4.0
            _control_shortfall = float(_required_controls) - float(_slam_controls)
            if _control_shortfall > 0:
                raw_p = _control_shortfall * float(w_first_round_control) * 2.0
                p = raw_p * sac_factor
                sac_note = f" [sacrifice-discounted from {raw_p:.0f}]" if sac_factor < 1.0 else ""
                _kind = "SLAM_CONTROL_SHORTFALL" if _is_suit else "NT_SLAM_ACE_SHORTFALL"
                _ctrl_parts.append(f"total ~{_slam_controls:.1f} controls")
                _ctrl_parts.append(f"need {_required_controls:.1f} for level {bid_level}")
                penalty += p
                reasons.append(
                    f"{_kind}: {bid} needs {_required_controls:.1f} controls but has only "
                    f"~{_slam_controls:.1f}; {', '.join(_ctrl_parts)}; short by {_control_shortfall:.1f} "
                    f"(-{p:.0f}){sac_note}"
                )
        else:
            _total_controls = min(4.0, float(_self_controls) + _partner_new)
            _max_safe = 3.0 + _total_controls
            _excess = float(bid_level) - _max_safe

            if _excess > 0:
                raw_p = _excess * float(w_first_round_control)
                p = raw_p * sac_factor
                sac_note = f" [sacrifice-discounted from {raw_p:.0f}]" if sac_factor < 1.0 else ""
                _ctrl_parts.append(f"total ~{_total_controls:.1f} controls")
                _ctrl_parts.append(f"max safe level {_max_safe:.1f}")
                penalty += p
                reasons.append(
                    f"INSUFFICIENT_FIRST_ROUND_CONTROLS: level {bid_level} "
                    f"exceeds max safe level {_max_safe:.1f} "
                    f"(= 3 + {_total_controls:.1f} controls); "
                    f"{', '.join(_ctrl_parts)}; "
                    f"excess {_excess:.1f} level(s) (-{p:.0f}){sac_note}"
                )

    # ==================================================================
    # SEMANTIC / REOPENING CHECKS
    # ==================================================================
    if bid_level is not None and bid_strain in ("C", "D", "H", "S"):
        is_artificial_probe = _is_artificial_major_probe(bt_acting_criteria, bid)
        is_new_minor_forcing_like = _is_new_minor_forcing_like(bt_acting_criteria, bid)

        # 8. NATURAL_SUIT_LENGTH_SHORTFALL
        #    Natural suit bids should usually show real length in that suit.
        #    Level 1-2: require ~5 cards. Level 3+: require ~6 cards.
        #    Exception: major-suit raises of partner can be correct with 3-card support.
        required_len = 6 if int(bid_level) >= 3 else 5
        if is_raise_of_partner_suit and bid_strain in ("H", "S"):
            required_len = 3
        self_len = None
        if isinstance(self_suit_lengths, dict):
            try:
                self_len = int(self_suit_lengths.get(str(bid_strain), -1))
            except Exception:
                self_len = None
        if (
            (not is_artificial_probe)
            and (not is_new_minor_forcing_like)
            and (not is_transfer_acceptance)
            and (not is_transfer_response)
            and self_len is not None
            and self_len >= 0
            and self_len < required_len
        ):
            shortfall = int(required_len) - int(self_len)
            p = float(shortfall) * float(w_natural_suit_shortfall)
            if is_reopening:
                p *= 1.25  # stricter in pass-out/reopening seat
            penalty += p
            reopen_note = " [reopening context]" if is_reopening else ""
            reasons.append(
                f"NATURAL_SUIT_LENGTH_SHORTFALL: bid {bid} suggests {required_len}+ "
                f"{bid_strain}; self has {self_len} (shortfall {shortfall}) "
                f"(-{p:.0f}){reopen_note}"
            )

        # 9. UNDERSPECIFIED_STRAIN_CRITERIA
        #    Backstop: if the bid's acting criteria don't constrain the bid strain
        #    length at all, demote as underspecified (especially problematic in
        #    reopening where partner can infer a lead/suit preference).
        if (
            (not is_artificial_probe)
            and (not is_new_minor_forcing_like)
            and (not is_transfer_response)
            and not _has_strain_length_criterion(bt_acting_criteria, bid_strain)
        ):
            p = float(w_underspecified_strain) * (1.5 if is_reopening else 1.0)
            penalty += p
            reopen_note = " [reopening context]" if is_reopening else ""
            reasons.append(
                f"UNDERSPECIFIED_STRAIN_CRITERIA: criteria missing SL_{bid_strain} "
                f"constraint for bid {bid} (-{p:.0f}){reopen_note}"
            )

        # 10. REOPEN_JUMP_SANITY
        #     In reopening seat, jumping to level 3+ in a suit should require
        #     something special (very long suit, high tricks, or strong hand).
        if is_reopening and int(bid_level) >= 3:
            strong_by_len = bool(self_len is not None and self_len >= 7)
            strong_by_tricks = bool(est_tricks is not None and float(est_tricks) >= 8.5)
            strong_by_points = bool(self_total_points is not None and float(self_total_points) >= 15.0)
            if not (strong_by_len or strong_by_tricks or strong_by_points):
                p = float(w_reopen_jump)
                penalty += p
                reasons.append(
                    f"REOPEN_JUMP_SANITY: reopening jump to {bid} without exceptional "
                    f"evidence (len>=7, est_tricks>=8.5, or self TP>=15) (-{p:.0f})"
                )

        # 11. REOPEN_LOW_US_FIT_FOR_BID_STRAIN
        #     Reopening in a strain with weak expected partnership support is risky.
        if is_reopening:
            p_fit8 = _prob_fit_8plus_for_strain(fit_us_hist, bid_strain)
            if p_fit8 is not None:
                min_fit_p = 0.35
                if float(p_fit8) < min_fit_p:
                    deficit = float(min_fit_p) - float(p_fit8)
                    p = deficit * float(w_reopen_low_fit)
                    penalty += p
                    reasons.append(
                        f"REOPEN_LOW_US_FIT_FOR_BID_STRAIN: P(us fit 8+) for {bid_strain} "
                        f"is {p_fit8:.2f} (< {min_fit_p:.2f}) in reopening seat "
                        f"(-{p:.0f})"
                    )

        # 12. OPP_SUIT_TRESPASS
        #     Bidding a suit that an opponent has naturally shown is almost
        #     always wrong — they announced length/strength there and will
        #     sit behind declarer.  Exempt only when the bidder holds a
        #     self-sufficient (6+ cards) or rebiddable suit per BT criteria.
        if opp_shown_strains and bid_strain in opp_shown_strains and not is_artificial_probe:
            _has_rebiddable = False
            if bt_acting_criteria:
                _crit_upper = " ".join(str(c) for c in bt_acting_criteria).upper()
                _has_rebiddable = (
                    f"TWICE_REBIDDABLE_{bid_strain}" in _crit_upper
                    or f"REBIDDABLE_{bid_strain}" in _crit_upper
                )
            _self_sufficient = bool(self_len is not None and self_len >= 6)
            if not (_has_rebiddable or _self_sufficient):
                _len_note = f"; self has {self_len}" if self_len is not None else ""
                p = float(w_opp_suit_trespass)
                penalty += p
                reasons.append(
                    f"OPP_SUIT_TRESPASS: bid {bid} in {bid_strain} which opponent "
                    f"has shown{_len_note}; need 6+ cards or rebiddable criteria "
                    f"to compete in opponent's suit (-{p:.0f})"
                )

    # ==================================================================
    # UNDERBID CHECKS (only when par belongs to our side)
    # ==================================================================

    if enable_underbid_checks and not par_is_opponents and top_level is not None:
        # ------------------------------------------------------------------
        # 8. UNDERBID_VS_PAR – bid level below par that our side owns
        # ------------------------------------------------------------------
        if bid_level < top_level and not nt_major_game_equiv:
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
        # 9. TP_SURPLUS_UNDERBID – partnership TP exceeds threshold for par
        #    level but we're bidding below it
        # ------------------------------------------------------------------
        if combined_tp is not None and bid_level < top_level and not nt_major_game_equiv:
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

        supported_level = None
        if (
            same_strain_prev_level is not None
            and same_strain_support_level_floor is not None
            and same_strain_support_level_ceiling is not None
        ):
            supported_level = min(int(same_strain_support_level_floor), int(same_strain_support_level_ceiling))
        game_level_for_bid = 3 if bid_strain == "NT" else (4 if bid_strain in ("H", "S") else 5)
        if (
            supported_level is not None
            and bid_level is not None
            and same_strain_prev_level is not None
            and bid_level >= game_level_for_bid
            and bid_level > int(same_strain_prev_level)
            and supported_level > int(bid_level)
        ):
            shortfall_levels = int(supported_level) - int(bid_level)
            p = float(shortfall_levels) * float(w_underbid_level) * 2.0
            penalty += p
            point_type = str(same_strain_point_type or "points")
            reasons.append(
                f"SAME_STRAIN_RANGE_UNDERBID: same-strain {point_type} floor/ceiling support "
                f"at least level {supported_level}{bid_strain}; bid {bid} stops {shortfall_levels} "
                f"level(s) short after prior level {same_strain_prev_level}{bid_strain} (-{p:.0f})"
            )

    return penalty, reasons


def _dir_side(direction: str | None) -> str:
    d = str(direction or "").strip().upper()
    return "NS" if d in ("N", "S") else "EW"


def _token_bidder_dir(token_idx: int, dealer_actual: str | None) -> str:
    directions = ["N", "E", "S", "W"]
    d = str(dealer_actual or "N").upper()
    base = directions.index(d) if d in directions else 0
    return directions[(base + int(token_idx)) % 4]


def _partner_dir(direction: str | None) -> str:
    d = str(direction or "").strip().upper()
    return {"N": "S", "S": "N", "E": "W", "W": "E"}.get(d, "N")


def _upper_bound_excess_from_exprs(
    metric_name: str,
    actual_value: float | None,
    exprs: list[Any] | None,
) -> tuple[float, float] | None:
    """Return (effective_cap, excess) for the tightest upper bound in `exprs`."""
    if actual_value is None:
        return None
    best_cap: float | None = None
    for raw_expr in list(exprs or []):
        try:
            s = str(raw_expr or "").strip().upper()
        except Exception:
            continue
        if not s:
            continue
        m = re.fullmatch(rf"{metric_name}\s*(<=|<)\s*(-?\d+(?:\.\d+)?)", s)
        if not m:
            continue
        op = str(m.group(1))
        cap = float(m.group(2))
        eff_cap = cap if op == "<=" else (cap - 1.0)
        best_cap = eff_cap if best_cap is None else min(best_cap, eff_cap)
    if best_cap is None or actual_value <= best_cap:
        return None
    return best_cap, float(actual_value - best_cap)


def extract_pass_range_caps(pass_agg_expr: list[Any] | None) -> dict[str, float | None]:
    """Extract the tightest explicit Pass caps for Total_Points and HCP."""
    tp_cap: float | None = None
    hcp_cap: float | None = None
    for raw_expr in list(pass_agg_expr or []):
        try:
            expr = str(raw_expr or "").strip().upper()
        except Exception:
            continue
        if not expr:
            continue
        tp_match = re.fullmatch(r"TOTAL_POINTS\s*(<=|<)\s*(-?\d+(?:\.\d+)?)", expr)
        if tp_match is not None:
            cap = float(tp_match.group(2))
            if str(tp_match.group(1)) == "<":
                cap -= 1.0
            tp_cap = cap if tp_cap is None else min(tp_cap, cap)
            continue
        hcp_match = re.fullmatch(r"HCP\s*(<=|<)\s*(-?\d+(?:\.\d+)?)", expr)
        if hcp_match is not None:
            cap = float(hcp_match.group(2))
            if str(hcp_match.group(1)) == "<":
                cap -= 1.0
            hcp_cap = cap if hcp_cap is None else min(hcp_cap, cap)
    return {
        "tp_cap": tp_cap,
        "hcp_cap": hcp_cap,
    }


def extract_second_pass_opening_context(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
) -> dict[str, Any] | None:
    """Normalize `partner opening - pass - responder to act` contexts.

    Leading opening passes are allowed, so `P-1S-P` is treated the same as
    `1S-P` for responder-side heuristics.
    """
    act = str(acting_direction or "").strip().upper()
    if act not in ("N", "E", "S", "W"):
        return None

    toks = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
    if not toks:
        return None

    opening_idx: int | None = None
    opening_bid = ""
    opening_level: int | None = None
    opening_strain: str | None = None
    for i, tk in enumerate(toks):
        lvl = _parse_bid_level(tk)
        st = _parse_bid_strain(tk)
        if lvl is None or st is None:
            continue
        opening_idx = i
        opening_bid = tk
        opening_level = lvl
        opening_strain = st
        break

    if (
        opening_idx is None
        or opening_level is None
        or opening_strain is None
        or (opening_idx + 1) >= len(toks)
        or toks[opening_idx + 1] not in ("P", "PASS")
        or len(toks) != (opening_idx + 2)
    ):
        return None

    opener_direction = _token_bidder_dir(int(opening_idx), dealer_actual)
    partner_direction = _partner_dir(act)
    if opener_direction != partner_direction:
        return None

    next_to_act = _token_bidder_dir(len(toks), dealer_actual)
    if next_to_act != act:
        return None

    return {
        "tokens": toks,
        "opening_idx": int(opening_idx),
        "leading_passes": int(opening_idx),
        "opening_bid": opening_bid,
        "opening_level": int(opening_level),
        "opening_strain": str(opening_strain),
        "pass_idx": int(opening_idx + 1),
        "opener_direction": opener_direction,
        "partner_direction": partner_direction,
        "acting_direction": act,
    }


def extract_opener_rebid_pass_context(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
) -> dict[str, Any] | None:
    """Normalize uncontested opener-rebid pass shapes: `opening-P-response-P-?`.

    Leading opening passes are allowed, so `P-1S-P-1N-P` is treated the same as
    `1S-P-1N-P` for opener-side heuristics.
    """
    act = str(acting_direction or "").strip().upper()
    if act not in ("N", "E", "S", "W"):
        return None

    toks = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
    if len(toks) < 4:
        return None

    opening_idx: int | None = None
    opening_bid = ""
    for i, tk in enumerate(toks):
        lvl = _parse_bid_level(tk)
        st = _parse_bid_strain(tk)
        if lvl is None or st is None:
            continue
        opening_idx = i
        opening_bid = tk
        break

    if opening_idx is None or len(toks) != (opening_idx + 4):
        return None
    if toks[opening_idx + 1] not in ("P", "PASS") or toks[opening_idx + 3] not in ("P", "PASS"):
        return None

    response_bid = toks[opening_idx + 2]
    response_level = _parse_bid_level(response_bid)
    response_strain = _parse_bid_strain(response_bid)
    if response_level is None or response_strain is None:
        return None

    opener_direction = _token_bidder_dir(int(opening_idx), dealer_actual)
    if opener_direction != act:
        return None
    responder_direction = _token_bidder_dir(int(opening_idx + 2), dealer_actual)
    if responder_direction != _partner_dir(act):
        return None

    next_to_act = _token_bidder_dir(len(toks), dealer_actual)
    if next_to_act != act:
        return None

    return {
        "tokens": toks,
        "opening_idx": int(opening_idx),
        "leading_passes": int(opening_idx),
        "opening_bid": opening_bid,
        "response_bid": response_bid,
        "response_level": int(response_level),
        "response_strain": str(response_strain),
        "opener_direction": opener_direction,
        "responder_direction": responder_direction,
        "acting_direction": act,
    }


def extract_opener_minor_clarification_pass_context(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
) -> dict[str, Any] | None:
    """Normalize `1m-P-1M-P-1oM-P-2m-P-?` opener clarification contexts."""
    act = str(acting_direction or "").strip().upper()
    if act not in ("N", "E", "S", "W"):
        return None

    toks = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
    if len(toks) < 8:
        return None

    opening_idx: int | None = None
    opening_bid = ""
    opening_level: int | None = None
    opening_strain: str | None = None
    for i, tk in enumerate(toks):
        lvl = _parse_bid_level(tk)
        st = _parse_bid_strain(tk)
        if lvl is None or st is None:
            continue
        opening_idx = i
        opening_bid = tk
        opening_level = lvl
        opening_strain = st
        break

    if (
        opening_idx is None
        or opening_level != 1
        or opening_strain not in ("C", "D")
        or len(toks) != (opening_idx + 8)
    ):
        return None

    if (
        toks[opening_idx + 1] not in ("P", "PASS")
        or toks[opening_idx + 3] not in ("P", "PASS")
        or toks[opening_idx + 5] not in ("P", "PASS")
        or toks[opening_idx + 7] not in ("P", "PASS")
    ):
        return None

    response_bid = toks[opening_idx + 2]
    response_level = _parse_bid_level(response_bid)
    response_strain = _parse_bid_strain(response_bid)
    if response_level != 1 or response_strain not in ("H", "S"):
        return None

    opener_rebid_bid = toks[opening_idx + 4]
    opener_rebid_level = _parse_bid_level(opener_rebid_bid)
    opener_rebid_strain = _parse_bid_strain(opener_rebid_bid)
    if (
        opener_rebid_level != 1
        or opener_rebid_strain not in ("H", "S")
        or opener_rebid_strain == response_strain
    ):
        return None

    partner_bid = toks[opening_idx + 6]
    partner_level = _parse_bid_level(partner_bid)
    partner_strain = _parse_bid_strain(partner_bid)
    if partner_level != 2 or partner_strain != opening_strain:
        return None

    partner_direction = _partner_dir(act)
    if not partner_direction:
        return None
    if _token_bidder_dir(int(opening_idx), dealer_actual) != act:
        return None
    if _token_bidder_dir(int(opening_idx + 2), dealer_actual) != partner_direction:
        return None
    if _token_bidder_dir(int(opening_idx + 4), dealer_actual) != act:
        return None
    if _token_bidder_dir(int(opening_idx + 6), dealer_actual) != partner_direction:
        return None

    next_to_act = _token_bidder_dir(len(toks), dealer_actual)
    if next_to_act != act:
        return None

    return {
        "tokens": toks,
        "opening_idx": int(opening_idx),
        "leading_passes": int(opening_idx),
        "opening_bid": opening_bid,
        "opening_strain": str(opening_strain),
        "response_bid": response_bid,
        "response_strain": str(response_strain),
        "opener_rebid_bid": opener_rebid_bid,
        "opener_rebid_strain": str(opener_rebid_strain),
        "partner_bid": partner_bid,
        "partner_direction": partner_direction,
        "acting_direction": act,
    }


def extract_blackwood_asker_followup_pass_context(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
) -> dict[str, Any] | None:
    """Normalize `4NT-P-5x-P-asker to act` Blackwood follow-up contexts."""
    act = str(acting_direction or "").strip().upper()
    if act not in ("N", "E", "S", "W"):
        return None

    toks = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
    if len(toks) < 4:
        return None

    last_non_pass_idx: int | None = None
    for i in range(len(toks) - 1, -1, -1):
        tk = toks[i]
        if tk not in ("P", "PASS", "X", "XX"):
            last_non_pass_idx = i
            break
    if last_non_pass_idx is None or last_non_pass_idx < 2:
        return None

    response_bid = toks[last_non_pass_idx]
    if response_bid not in ("5C", "5D", "5H", "5S"):
        return None
    if toks[last_non_pass_idx - 1] not in ("P", "PASS"):
        return None
    ask_bid = toks[last_non_pass_idx - 2]
    if ask_bid not in ("4N", "4NT"):
        return None

    trailing = toks[last_non_pass_idx + 1 :]
    if len(trailing) != 1 or trailing[0] not in ("P", "PASS"):
        return None

    asker_direction = _token_bidder_dir(int(last_non_pass_idx - 2), dealer_actual)
    responder_direction = _token_bidder_dir(int(last_non_pass_idx), dealer_actual)
    if responder_direction != _partner_dir(asker_direction):
        return None

    next_to_act = _token_bidder_dir(len(toks), dealer_actual)
    if next_to_act != act or asker_direction != act:
        return None

    return {
        "tokens": toks,
        "ask_bid": ask_bid,
        "ask_idx": int(last_non_pass_idx - 2),
        "response_bid": response_bid,
        "response_idx": int(last_non_pass_idx),
        "asker_direction": asker_direction,
        "responder_direction": responder_direction,
        "acting_direction": act,
    }


def extract_blackwood_same_strain_continuation_context(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    bid_text: str | None,
) -> dict[str, Any] | None:
    """Detect same-strain post-Blackwood continuations by the original asker."""
    followup_ctx = extract_blackwood_asker_followup_pass_context(
        auction_tokens=list(auction_tokens or []),
        acting_direction=acting_direction,
        dealer_actual=dealer_actual,
    )
    if followup_ctx is None:
        return None

    bid_level = _parse_bid_level(bid_text)
    bid_strain = _parse_bid_strain(bid_text)
    if bid_level is None or bid_strain not in ("C", "D", "H", "S", "NT"):
        return None

    game_level = 3 if bid_strain == "NT" else (4 if bid_strain in ("H", "S") else 5)
    act = str(acting_direction or "").strip().upper()
    side_map = {"N": "NS", "S": "NS", "E": "EW", "W": "EW"}
    acting_side = side_map.get(act)
    if acting_side is None:
        return None

    prev_same_side_same_strain_level: int | None = None
    for idx, tk in enumerate(list(followup_ctx.get("tokens") or [])):
        lvl = _parse_bid_level(tk)
        st = _parse_bid_strain(tk)
        if lvl is None or st is None:
            continue
        tk_dir = _token_bidder_dir(int(idx), dealer_actual)
        if side_map.get(tk_dir) != acting_side:
            continue
        if st == bid_strain:
            prev_same_side_same_strain_level = int(lvl)

    if prev_same_side_same_strain_level is None:
        return None

    is_raise = int(bid_level) > int(prev_same_side_same_strain_level)
    is_post_game_same_strain = bool(
        is_raise and int(prev_same_side_same_strain_level) >= int(game_level)
    )

    return {
        "ask_bid": followup_ctx.get("ask_bid"),
        "response_bid": followup_ctx.get("response_bid"),
        "acting_direction": act,
        "bid_text": str(bid_text or "").strip().upper(),
        "bid_level": int(bid_level),
        "bid_strain": str(bid_strain),
        "game_level": int(game_level),
        "prev_same_side_same_strain_level": int(prev_same_side_same_strain_level),
        "is_raise": bool(is_raise),
        "is_post_game_same_strain": bool(is_post_game_same_strain),
    }


def extract_early_constructive_range_gap_context(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
) -> dict[str, Any] | None:
    """Return a normalized early constructive pass-gap context.

    Supported shapes:
    - responder second pass: `partner opening - pass - ?`
    - opener rebid gap: `opening - pass - response - pass - ?`
    """
    second_pass_ctx = extract_second_pass_opening_context(
        auction_tokens=list(auction_tokens or []),
        acting_direction=acting_direction,
        dealer_actual=dealer_actual,
    )
    if second_pass_ctx is not None:
        return {
            **second_pass_ctx,
            "kind": "response_new_suit",
            "partner_bid": str(second_pass_ctx.get("opening_bid") or ""),
            "reference_strain": str(second_pass_ctx.get("opening_strain") or ""),
            "partner_direction": str(second_pass_ctx.get("partner_direction") or ""),
        }

    opener_rebid_ctx = extract_opener_rebid_pass_context(
        auction_tokens=list(auction_tokens or []),
        acting_direction=acting_direction,
        dealer_actual=dealer_actual,
    )
    if opener_rebid_ctx is not None:
        return {
            **opener_rebid_ctx,
            "kind": "opener_rebid",
            "partner_bid": str(opener_rebid_ctx.get("response_bid") or ""),
            "reference_strain": str(_parse_bid_strain(opener_rebid_ctx.get("opening_bid")) or ""),
            "partner_direction": str(opener_rebid_ctx.get("responder_direction") or ""),
        }

    return None


def opponent_shown_natural_strains(
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer: str | None,
) -> set[str]:
    """Return the set of natural suit strains (C/D/H/S) bid by opponents.

    Only includes natural suit bids (not NT, Pass, Double, Redouble).
    Doubles and artificial bids are excluded because we can't reliably
    distinguish them here; only explicit suit calls count.
    """
    result: set[str] = set()
    if not auction_tokens or not acting_direction:
        return result
    act = str(acting_direction or "").strip().upper()
    opp_dirs = {"N": {"E", "W"}, "E": {"N", "S"},
                "S": {"E", "W"}, "W": {"N", "S"}}.get(act, set())
    if not opp_dirs:
        return result
    for i, tk in enumerate(auction_tokens):
        tk_s = str(tk or "").strip().upper()
        m = re.match(r"^[1-7]\s*([CDHS])", tk_s)
        if not m:
            continue
        bidder = _token_bidder_dir(i, dealer)
        if bidder in opp_dirs:
            result.add(m.group(1).upper())
    return result


def _parse_contract_bid_text(bid_text: str) -> tuple[int, str] | None:
    s = str(bid_text or "").strip().upper()
    m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
    if not m:
        return None
    lvl = int(m.group(1))
    st = m.group(2).upper()
    return lvl, ("N" if st in ("N", "NT") else st)


def _pass_would_end_auction(auction_tokens: list[str]) -> bool:
    toks = [str(t or "").strip().upper() for t in list(auction_tokens or [])]
    if not toks:
        return False
    has_non_pass = any(t not in ("P", "PASS") for t in toks)
    trailing = 0
    for t in reversed(toks):
        if t in ("P", "PASS"):
            trailing += 1
        else:
            break
    if not has_non_pass:
        return len(toks) >= 3
    return trailing >= 2


def _prob_fit_8plus(fit_hist: dict[str, Any] | None, suit: str) -> float | None:
    if not isinstance(fit_hist, dict):
        return None
    h = fit_hist.get(str(suit))
    if not isinstance(h, dict):
        return None
    tot = 0.0
    good = 0.0
    for k, v in h.items():
        try:
            n = int(k)
            cnt = float(v)
        except Exception:
            continue
        if cnt <= 0:
            continue
        tot += cnt
        if n >= 8:
            good += cnt
    return (good / tot) if tot > 0 else None


def _last_opponent_contract(
    auction_tokens: list[str],
    *,
    acting_direction: str | None,
    dealer_actual: str | None,
) -> tuple[int, str] | None:
    side = _dir_side(acting_direction)
    for i in range(len(list(auction_tokens or [])) - 1, -1, -1):
        tk = str((auction_tokens or [])[int(i)] or "").strip().upper()
        c = _parse_contract_bid_text(tk)
        if c is None:
            continue
        if _dir_side(_token_bidder_dir(int(i), dealer_actual)) == side:
            continue
        return int(c[0]), str(c[1])
    return None


def is_nt_transfer_acceptance_call(
    auction_tokens: list[str],
    token_idx: int,
    *,
    dealer_actual: str | None,
) -> bool:
    """Return True when the indexed call is opener's simple Jacoby transfer acceptance."""
    if int(token_idx) < 0:
        return False
    non_pass_calls: list[tuple[int, str, str]] = []
    for i, tk in enumerate(list(auction_tokens or [])):
        bid_u = str(tk or "").strip().upper()
        if bid_u in ("", "P", "PASS", "X", "XX", "DOUBLE", "REDOUBLE"):
            continue
        parsed = _parse_contract_bid_text(bid_u)
        if parsed is None:
            continue
        bidder = _token_bidder_dir(i, dealer_actual)
        if bidder not in ("N", "E", "S", "W"):
            continue
        bid_txt = f"{int(parsed[0])}{str(parsed[1]).upper()}"
        non_pass_calls.append((int(i), bidder, bid_txt))
    if len(non_pass_calls) < 3:
        return False

    accept_idx, accept_dir, accept_bid = non_pass_calls[-1]
    if accept_idx != int(token_idx):
        return False
    opener_idx, opener_dir, opener_bid = non_pass_calls[-3]
    transfer_idx, transfer_dir, transfer_bid = non_pass_calls[-2]
    if opener_idx >= transfer_idx or transfer_idx >= accept_idx:
        return False
    if accept_dir != opener_dir:
        return False
    if transfer_dir != _partner_dir(accept_dir):
        return False

    expected_transfer_bid = {
        ("1N", "2H"): "2D",
        ("1N", "2S"): "2H",
        ("1N", "3C"): "2S",
        ("1N", "3D"): "2N",
        ("2N", "3H"): "3D",
        ("2N", "3S"): "3H",
        ("2N", "4C"): "3S",
    }.get((opener_bid, accept_bid))
    return bool(expected_transfer_bid and transfer_bid == expected_transfer_bid)


def _nt_transfer_shown_major(
    auction_tokens: list[str],
    token_idx: int,
    *,
    dealer_actual: str | None,
) -> str | None:
    """Return the actual major shown by a Jacoby transfer request, if any."""
    if int(token_idx) < 0:
        return None
    non_pass_calls: list[tuple[int, str, str]] = []
    for i, tk in enumerate(list(auction_tokens or [])):
        bid_u = str(tk or "").strip().upper()
        if bid_u in ("", "P", "PASS", "X", "XX", "DOUBLE", "REDOUBLE"):
            continue
        parsed = _parse_contract_bid_text(bid_u)
        if parsed is None:
            continue
        bidder = _token_bidder_dir(i, dealer_actual)
        if bidder not in ("N", "E", "S", "W"):
            continue
        bid_txt = f"{int(parsed[0])}{str(parsed[1]).upper()}"
        non_pass_calls.append((int(i), bidder, bid_txt))
    call_pos = next(
        (idx for idx, (orig_idx, _bidder, _bid) in enumerate(non_pass_calls) if orig_idx == int(token_idx)),
        None,
    )
    if call_pos is None or call_pos <= 0:
        return None
    transfer_idx, transfer_dir, transfer_bid = non_pass_calls[call_pos]
    opener_idx, opener_dir, opener_bid = non_pass_calls[call_pos - 1]
    if opener_idx >= transfer_idx:
        return None
    if transfer_dir != _partner_dir(opener_dir):
        return None
    return {
        ("1N", "2D"): "H",
        ("1N", "2H"): "S",
        ("2N", "3D"): "H",
        ("2N", "3H"): "S",
    }.get((opener_bid, transfer_bid))


def _partner_shown_majors(
    auction_tokens: list[str],
    *,
    acting_direction: str | None,
    dealer_actual: str | None,
) -> set[str]:
    out: set[str] = set()
    p = _partner_dir(acting_direction)
    for i, tk in enumerate(list(auction_tokens or [])):
        if _token_bidder_dir(i, dealer_actual) != p:
            continue
        c = _parse_contract_bid_text(str(tk or "").strip().upper())
        if c is None:
            continue
        st = str(c[1])
        if is_nt_transfer_acceptance_call(
            auction_tokens,
            i,
            dealer_actual=dealer_actual,
        ):
            continue
        transfer_major = _nt_transfer_shown_major(
            auction_tokens,
            i,
            dealer_actual=dealer_actual,
        )
        if transfer_major in ("H", "S"):
            out.add(transfer_major)
            continue
        if st in ("H", "S"):
            out.add(st)
    return out


def _partner_takeout_double_shown(
    auction_tokens: list[str],
    *,
    acting_direction: str | None,
    dealer_actual: str | None,
) -> bool:
    p = _partner_dir(acting_direction)
    for i, tk in enumerate(list(auction_tokens or [])):
        if _token_bidder_dir(i, dealer_actual) != p:
            continue
        b = str(tk or "").strip().upper()
        if b in ("D", "X", "DOUBLE"):
            return True
    return False


def _stayman_major_response_context(
    auction_tokens: list[str],
    *,
    acting_direction: str | None,
    dealer_actual: str | None,
) -> dict[str, Any] | None:
    """Detect responder's rebid after opener answers Stayman with a major."""
    act = str(acting_direction or "").strip().upper()
    if act not in ("N", "E", "S", "W"):
        return None
    partner = _partner_dir(act)
    if not partner:
        return None
    side = _dir_side(act)
    if not side:
        return None

    own_contracts: list[tuple[str, str]] = []
    for i, tk in enumerate(list(auction_tokens or [])):
        bid = str(tk or "").strip().upper()
        if bid in ("", "P", "PASS", "D", "X", "DOUBLE", "XX", "REDOUBLE"):
            continue
        bidder = _token_bidder_dir(i, dealer_actual)
        if _dir_side(bidder) != side:
            continue
        own_contracts.append((str(bidder), bid))

    if len(own_contracts) < 3:
        return None

    first_bidder, first_bid = own_contracts[0]
    second_bidder, second_bid = own_contracts[1]
    third_bidder, third_bid = own_contracts[2]
    if first_bidder != partner or first_bid not in ("1N", "1NT"):
        return None
    if second_bidder != act or second_bid != "2C":
        return None
    if third_bidder != partner or third_bid not in ("2H", "2S"):
        return None

    return {
        "opener_direction": partner,
        "responder_direction": act,
        "shown_major": third_bid[1],
    }


def _major_opening_nt_rebid_response_context(
    auction_tokens: list[str],
    *,
    acting_direction: str | None,
    dealer_actual: str | None,
) -> dict[str, Any] | None:
    """Detect responder's rebid after `1M-P-1oM-P-2NT-P`.

    This is the classic opener-major / responder-other-major / opener-2NT shape
    where responder may need to place the contract back in opener's major rather
    than let generic pass/NT logic dominate.
    """
    act = str(acting_direction or "").strip().upper()
    if act not in ("N", "E", "S", "W"):
        return None

    toks = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
    if len(toks) < 6:
        return None

    opening_idx: int | None = None
    opening_bid = ""
    for i, tk in enumerate(toks):
        lvl = _parse_bid_level(tk)
        st = _parse_bid_strain(tk)
        if lvl is None or st is None:
            continue
        opening_idx = i
        opening_bid = tk
        break

    if opening_idx is None or len(toks) != (opening_idx + 6):
        return None
    if (
        toks[opening_idx + 1] not in ("P", "PASS")
        or toks[opening_idx + 3] not in ("P", "PASS")
        or toks[opening_idx + 5] not in ("P", "PASS")
    ):
        return None

    opening_level = _parse_bid_level(opening_bid)
    opening_strain = _parse_bid_strain(opening_bid)
    if opening_level != 1 or opening_strain not in ("H", "S"):
        return None

    response_bid = toks[opening_idx + 2]
    response_level = _parse_bid_level(response_bid)
    response_strain = _parse_bid_strain(response_bid)
    if (
        response_level != 1
        or response_strain not in ("H", "S")
        or response_strain == opening_strain
    ):
        return None

    opener_rebid_bid = toks[opening_idx + 4]
    opener_rebid_level = _parse_bid_level(opener_rebid_bid)
    opener_rebid_strain = _parse_bid_strain(opener_rebid_bid)
    if opener_rebid_level != 2 or opener_rebid_strain != "NT":
        return None

    opener_direction = _token_bidder_dir(int(opening_idx), dealer_actual)
    partner_direction = _partner_dir(act)
    if not partner_direction or opener_direction != partner_direction:
        return None
    if _token_bidder_dir(int(opening_idx + 2), dealer_actual) != act:
        return None
    if _token_bidder_dir(int(opening_idx + 4), dealer_actual) != partner_direction:
        return None

    next_to_act = _token_bidder_dir(len(toks), dealer_actual)
    if next_to_act != act:
        return None

    return {
        "tokens": toks,
        "opening_idx": int(opening_idx),
        "leading_passes": int(opening_idx),
        "opening_bid": opening_bid,
        "opening_strain": str(opening_strain),
        "response_bid": response_bid,
        "response_strain": str(response_strain),
        "opener_rebid_bid": opener_rebid_bid,
        "opener_direction": opener_direction,
        "responder_direction": act,
        "acting_direction": act,
    }


def compute_common_sense_adjustments(
    *,
    bid_text: str,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    self_total_points: float | None,
    partner_total_points_expected: float | None,
    self_suit_lengths: dict[str, int] | None,
    fit_us_hist: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute global common-sense score adjustments for a candidate bid."""
    b = str(bid_text or "").strip().upper()
    c = _parse_contract_bid_text(b)
    lvl = int(c[0]) if c is not None else None
    st = str(c[1]) if c is not None else None
    tp_self = float(self_total_points) if isinstance(self_total_points, (int, float)) else None
    tp_partner = float(partner_total_points_expected) if isinstance(partner_total_points_expected, (int, float)) else None
    tp_combined = (tp_self + tp_partner) if (tp_self is not None and tp_partner is not None) else None
    sl = dict(self_suit_lengths or {})

    bonus = 0.0
    penalty = 0.0
    reason_codes: list[str] = []
    evidence: dict[str, Any] = {
        "self_total_points": tp_self,
        "partner_total_points_expected": tp_partner,
        "combined_total_points_expected": tp_combined,
    }

    try:
        # Policy 1: non-pass after partner takeout double with values.
        partner_dbl = _partner_takeout_double_shown(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["partner_takeout_double_shown"] = bool(partner_dbl)
        if partner_dbl and tp_self is not None and tp_self >= 11.0:
            if b in ("P", "PASS"):
                penalty += 220.0
                reason_codes.append("non_pass_after_partner_takeout_double_with_values")
            else:
                bonus += 35.0

        # Policy 2: major-fit realism.
        majors = _partner_shown_majors(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["partner_shown_majors"] = sorted(list(majors))
        if majors:
            best_major = max(list(majors), key=lambda s: int(sl.get(s, 0)))
            support_len = int(sl.get(best_major, 0))
            p_fit = _prob_fit_8plus(fit_us_hist, best_major)
            evidence["major_fit_candidate"] = best_major
            evidence["self_support_len_in_partner_major"] = support_len
            evidence["p_fit8plus_partner_major"] = p_fit

            if st == best_major and lvl is not None and lvl >= 3 and support_len >= 4:
                bonus += 110.0
                reason_codes.append("major_fit_raise_after_partner_shows_heart_or_spade_fit")
            if st == "N" and lvl is not None and lvl <= 3:
                likely_fit = bool(support_len >= 4 or (p_fit is not None and float(p_fit) >= 0.45))
                if likely_fit:
                    penalty += 120.0
                    reason_codes.append("major_fit_nt_detour_with_likely_fit")

        # Policy 3: defensive realism vs opponent NT partscore.
        opp_last = _last_opponent_contract(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["last_opponent_contract"] = opp_last
        if opp_last is not None:
            opp_lvl, opp_st = opp_last
            if opp_st == "N" and 1 <= int(opp_lvl) <= 3:
                if tp_self is not None and tp_self >= 10.0 and (tp_combined is None or tp_combined >= 22.0):
                    if b in ("D", "X", "DOUBLE"):
                        bonus += 140.0
                        reason_codes.append("double_over_nt_partscore")
                    elif b in ("P", "PASS"):
                        penalty += 80.0
                        reason_codes.append("pass_vs_attackable_nt_partscore")

        # Policy 4: auction coherence near pass-out.
        if b in ("P", "PASS") and _pass_would_end_auction(auction_tokens):
            if tp_combined is not None and tp_combined >= 22.0:
                penalty += 120.0
                reason_codes.append("pass_out_with_values_when_constructive_actions_exist")
    except Exception:
        pass

    return {
        "bonus": float(bonus),
        "penalty": float(penalty),
        "reason_codes": list(reason_codes),
        "evidence": evidence,
    }


def compute_common_sense_hard_override(
    *,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    self_total_points: float | None,
    partner_total_points_expected: float | None,
    self_suit_lengths: dict[str, int] | None,
    current_best_bid: str | None,
    legal_non_pass_candidates: list[dict[str, Any]] | None,
    blocked_candidates: list[dict[str, Any]] | None,
    criteria_failed_candidates: list[dict[str, Any]] | None = None,
    scored_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return controlled hard-override decision for whitelisted no-brainer actions."""
    tp_self = float(self_total_points) if isinstance(self_total_points, (int, float)) else None
    tp_partner = float(partner_total_points_expected) if isinstance(partner_total_points_expected, (int, float)) else None
    tp_combined = (tp_self + tp_partner) if (tp_self is not None and tp_partner is not None) else None
    sl = dict(self_suit_lengths or {})
    best_bid = str(current_best_bid or "").strip().upper()
    legal = list(legal_non_pass_candidates or [])
    blocked = list(blocked_candidates or [])
    criteria_failed = list(criteria_failed_candidates or [])
    scored = list(scored_candidates or [])

    def _pick_bid(rows: list[dict[str, Any]], target_bid: str) -> dict[str, Any] | None:
        tgt = str(target_bid or "").strip().upper()
        for row in rows:
            bid = str((row or {}).get("bid", "") or "").strip().upper()
            if bid == tgt:
                return row
        return None

    def _is_strong_forcing_two_club_row(row: dict[str, Any] | None) -> bool:
        if not isinstance(row, dict):
            return False
        bid = str((row or {}).get("bid", "") or "").strip().upper()
        if bid != "2C":
            return False
        crits = {
            str(expr or "").strip().upper().strip("()")
            for expr in list((row or {}).get("agg_expr") or [])
            if str(expr or "").strip()
        }
        return "FORCING_TO_2N" in crits

    def _opening_round_has_strong_two_club_available() -> bool:
        try:
            for tk in list(auction_tokens or []):
                if _parse_contract_bid_text(str(tk or "").strip().upper()) is not None:
                    return False
            rows = list(scored or []) + list(legal or []) + list(blocked or []) + list(criteria_failed or [])
            return any(_is_strong_forcing_two_club_row(row) for row in rows)
        except Exception:
            return False

    def _is_direct_overcall_gap_rescued_3n(rows: list[dict[str, Any]]) -> bool:
        row_3n = _pick_bid(rows, "3N") or _pick_bid(rows, "3NT")
        if row_3n is None:
            return False
        if bool((row_3n or {}).get("_direct_overcall_gap_rescue")):
            return True
        rescue_reason = str((row_3n or {}).get("_direct_overcall_gap_rescue_reason") or "").strip().upper()
        return "DIRECT_OVERCALL_GAP_RESCUE" in rescue_reason

    def _priority(row: dict[str, Any], opp_last_strain: str | None) -> float:
        b = str((row or {}).get("bid", "") or "").strip().upper()
        c = _parse_contract_bid_text(b)
        lvl = int(c[0]) if c is not None else 0
        st = str(c[1]) if c is not None else None
        p = 0.0
        if b in ("2N", "2NT"):
            p += 400.0
        if opp_last_strain is not None and st == opp_last_strain and lvl >= 2:
            p += 350.0
        if st in ("H", "S") and lvl >= 2:
            p += 300.0 + float(lvl)
        if b in ("D", "X", "DOUBLE"):
            p += 280.0
        try:
            p += min(20.0, max(0.0, float((row or {}).get("matching_deal_count") or 0.0) / 50000.0))
        except Exception:
            pass
        return p

    evidence: dict[str, Any] = {
        "self_total_points": tp_self,
        "partner_total_points_expected": tp_partner,
        "combined_total_points_expected": tp_combined,
        "current_best_bid": best_bid,
    }

    def _forced_stayman_reply_bid_here(tokens_now: list[str]) -> str | None:
        try:
            toks = [str(t or "").strip().upper() for t in list(tokens_now or []) if str(t or "").strip()]
            while toks and toks[0] in ("P", "PASS"):
                toks = toks[1:]
            tail4 = toks[-4:] if len(toks) >= 4 else toks
            if tail4 == ["1N", "P", "2C", "P"]:
                return "2D"
            if tail4 == ["2N", "P", "3C", "P"]:
                return "3D"
        except Exception:
            return None
        return None

    forced_stayman_reply_bid = _forced_stayman_reply_bid_here(auction_tokens)

    def _forced_transfer_acceptance_bid_here(tokens_now: list[str]) -> str | None:
        try:
            toks = [str(t or "").strip().upper() for t in list(tokens_now or []) if str(t or "").strip()]
            while toks and toks[0] in ("P", "PASS"):
                toks = toks[1:]
            tail4 = toks[-4:] if len(toks) >= 4 else toks
            _TRANSFER_MAP = {
                ("1N", "2D"): "2H",
                ("1N", "2H"): "2S",
                ("2N", "3D"): "3H",
                ("2N", "3H"): "3S",
            }
            # Only treat this as a transfer acceptance when the stripped auction
            # is exactly the NT opening plus transfer response sequence.
            if len(toks) == 4 and len(tail4) == 4 and tail4[1] == "P" and tail4[3] == "P":
                result = _TRANSFER_MAP.get((tail4[0], tail4[2]))
                if result is not None:
                    return result
        except Exception:
            return None
        return None

    forced_transfer_bid = _forced_transfer_acceptance_bid_here(auction_tokens)

    # Whitelist -0 (highest priority): Jacoby transfer acceptance is mandatory.
    if forced_transfer_bid is not None:
        picked_row = (
            _pick_bid(scored, forced_transfer_bid)
            or _pick_bid(legal, forced_transfer_bid)
            or _pick_bid(blocked, forced_transfer_bid)
            or _pick_bid(criteria_failed, forced_transfer_bid)
        )
        if picked_row is not None:
            source = (
                "scored_candidate" if _pick_bid(scored, forced_transfer_bid) is not None
                else "legal" if _pick_bid(legal, forced_transfer_bid) is not None
                else "blocked_cannot_complete" if _pick_bid(blocked, forced_transfer_bid) is not None
                else "criteria_fail"
            )
            return {
                "apply": True,
                "selected_bid": forced_transfer_bid,
                "reason_codes": ["jacoby_transfer_acceptance"],
                "reason": (
                    f"COMMON_SENSE_HARD_OVERRIDE: opener must complete Jacoby transfer "
                    f"by bidding {forced_transfer_bid}"
                ),
                "evidence": {
                    **evidence,
                    "source": source,
                    "selected_bid": forced_transfer_bid,
                    "forced_transfer_bid": forced_transfer_bid,
                },
            }

    # Whitelist -0.1: after transfer acceptance + partner's 3N, opener corrects
    # to 4-major with 3+ card support for partner's known 5+ card suit.
    # This must be a hard override because 5+ guardrails (IMPOSSIBLE_SUIT_REBID,
    # PULL_NT_GAME_PENALTY, NO_NON_SAC_OVERBID, NATURAL_SUIT_LENGTH_SHORTFALL,
    # TRICKS_SHORTFALL) all treat 4H as a natural rebid rather than a transfer
    # correction and hard-block it with ~1400+ penalty points.
    def _transfer_3n_correction_bid_here(tokens_now: list[str]) -> str | None:
        try:
            toks = [str(t or "").strip().upper() for t in list(tokens_now or []) if str(t or "").strip()]
            while toks and toks[0] in ("P", "PASS"):
                toks = toks[1:]
            tail8 = toks[-8:] if len(toks) >= 8 else toks
            _CORRECTION_SEQS: dict[tuple[str, ...], tuple[str, str]] = {
                ("1N", "P", "2D", "P", "2H", "P", "3N", "P"): ("H", "4H"),
                ("1N", "P", "2H", "P", "2S", "P", "3N", "P"): ("S", "4S"),
                ("2N", "P", "3D", "P", "3H", "P", "3N", "P"): ("H", "4H"),
                ("2N", "P", "3H", "P", "3S", "P", "3N", "P"): ("S", "4S"),
            }
            match = _CORRECTION_SEQS.get(tuple(tail8))
            if match is not None:
                suit, correction_bid = match
                if sl.get(suit, 0) >= 3:
                    return correction_bid
        except Exception:
            return None
        return None

    transfer_3n_correction = _transfer_3n_correction_bid_here(auction_tokens)
    if transfer_3n_correction is not None:
        picked_row = (
            _pick_bid(scored, transfer_3n_correction)
            or _pick_bid(legal, transfer_3n_correction)
            or _pick_bid(blocked, transfer_3n_correction)
            or _pick_bid(criteria_failed, transfer_3n_correction)
        )
        if picked_row is not None:
            source = (
                "scored_candidate" if _pick_bid(scored, transfer_3n_correction) is not None
                else "legal" if _pick_bid(legal, transfer_3n_correction) is not None
                else "blocked_cannot_complete" if _pick_bid(blocked, transfer_3n_correction) is not None
                else "criteria_fail"
            )
            suit = transfer_3n_correction[1]
            return {
                "apply": True,
                "selected_bid": transfer_3n_correction,
                "reason_codes": ["transfer_3n_major_correction"],
                "reason": (
                    f"COMMON_SENSE_HARD_OVERRIDE: after transfer + 3N, opener corrects to "
                    f"{transfer_3n_correction} with {sl.get(suit, 0)} cards in {suit} "
                    f"(partner's transfer guarantees 5+)"
                ),
                "evidence": {
                    **evidence,
                    "source": source,
                    "selected_bid": transfer_3n_correction,
                    "self_suit_length": sl.get(suit, 0),
                    "transfer_suit": suit,
                },
            }

    def _strain_rank(strain: str | None) -> int:
        order = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}
        return int(order.get(str(strain or "").upper(), -1))

    def _last_shown_index(strain: str | None) -> int:
        target = str(strain or "").strip().upper()
        if target not in ("C", "D", "H", "S"):
            return -1
        directions = ("N", "E", "S", "W")
        dealer_u = str(dealer_actual or "N").strip().upper()
        if dealer_u not in directions:
            dealer_u = "N"
        dealer_idx = directions.index(dealer_u)
        last_idx = -1
        for i, tk in enumerate(list(auction_tokens or [])):
            bidder_dir = directions[(dealer_idx + int(i)) % 4]
            if bidder_dir != acting_direction:
                continue
            parsed = _parse_contract_bid_text(str(tk or "").strip().upper())
            if parsed is None:
                continue
            if str(parsed[1]) == target:
                last_idx = i
        return last_idx

    def _actor_shown_suits() -> list[str]:
        directions = ("N", "E", "S", "W")
        dealer_u = str(dealer_actual or "N").strip().upper()
        if dealer_u not in directions:
            dealer_u = "N"
        dealer_idx = directions.index(dealer_u)
        shown: list[str] = []
        seen: set[str] = set()
        for i, tk in enumerate(list(auction_tokens or [])):
            bidder_dir = directions[(dealer_idx + int(i)) % 4]
            if bidder_dir != acting_direction:
                continue
            parsed = _parse_contract_bid_text(str(tk or "").strip().upper())
            if parsed is None:
                continue
            strain = str(parsed[1] or "").upper()
            if strain not in ("C", "D", "H", "S"):
                continue
            if strain not in seen:
                shown.append(strain)
                seen.add(strain)
        return shown

    def _actor_has_bid_contract() -> bool:
        directions = ("N", "E", "S", "W")
        dealer_u = str(dealer_actual or "N").strip().upper()
        if dealer_u not in directions:
            dealer_u = "N"
        dealer_idx = directions.index(dealer_u)
        for i, tk in enumerate(list(auction_tokens or [])):
            bidder_dir = directions[(dealer_idx + int(i)) % 4]
            if bidder_dir != acting_direction:
                continue
            if _parse_contract_bid_text(str(tk or "").strip().upper()) is not None:
                return True
        return False

    def _pure_direct_overcall_one_level_opening() -> tuple[int, str] | None:
        directions = ("N", "E", "S", "W")
        acting_dir_u = str(acting_direction or "").strip().upper()
        dealer_u = str(dealer_actual or "N").strip().upper()
        if acting_dir_u not in directions or dealer_u not in directions:
            return None
        dealer_idx = directions.index(dealer_u)
        acting_has_bid = False
        same_side_contracts = 0
        opp_contracts: list[tuple[int, str]] = []
        contract_count = 0
        for i, tk in enumerate(list(auction_tokens or [])):
            bidder_dir = directions[(dealer_idx + int(i)) % 4]
            parsed = _parse_contract_bid_text(str(tk or "").strip().upper())
            if parsed is None:
                continue
            contract_count += 1
            lvl, st = int(parsed[0]), str(parsed[1])
            if bidder_dir == acting_dir_u:
                acting_has_bid = True
            elif _dir_side(bidder_dir) == _dir_side(acting_dir_u):
                same_side_contracts += 1
            else:
                opp_contracts.append((lvl, st))
        if (
            (not acting_has_bid)
            and same_side_contracts == 0
            and contract_count == 1
            and len(opp_contracts) == 1
            and opp_contracts[0][0] == 1
            and opp_contracts[0][1] in ("C", "D", "H", "S")
        ):
            return opp_contracts[0]
        return None

    def _qualifies_second_suit_rebid_semantics(bid_text: str) -> tuple[bool, dict[str, Any]]:
        parsed = _parse_contract_bid_text(str(bid_text or "").strip().upper())
        if parsed is None:
            return False, {}
        bid_suit = str(parsed[1] or "").upper()
        if bid_suit not in ("C", "D", "H", "S"):
            return False, {}
        shown = _actor_shown_suits()
        if len(shown) < 2 or bid_suit != shown[1]:
            return False, {"shown_suits": shown}
        first_suit = shown[0]
        first_len = sl.get(first_suit)
        second_len = sl.get(bid_suit)
        if first_len is None or second_len is None:
            return False, {"shown_suits": shown}
        ok = bool(int(second_len) >= 5 and int(first_len) >= int(second_len))
        return ok, {
            "shown_suits": shown,
            "first_suit": first_suit,
            "first_len": int(first_len),
            "second_suit": bid_suit,
            "second_len": int(second_len),
        }

    def _bid_row_with_source(target_bid: str) -> tuple[dict[str, Any] | None, str | None]:
        bid_u = str(target_bid or "").strip().upper()
        for source_name, rows in (
            ("scored_candidate", scored),
            ("legal", legal),
            ("blocked_cannot_complete", blocked),
            ("criteria_fail", criteria_failed),
        ):
            row = _pick_bid(list(rows or []), bid_u)
            if row is not None:
                return row, source_name
        return None, None

    def _candidate_rows_for_major(target_major: str) -> list[tuple[int, dict[str, Any], str]]:
        rows_out: list[tuple[int, dict[str, Any], str]] = []
        seen: set[tuple[str, str]] = set()
        major_u = str(target_major or "").strip().upper()
        for source_name, rows in (
            ("scored_candidate", scored),
            ("legal", legal),
            ("blocked_cannot_complete", blocked),
            ("criteria_fail", criteria_failed),
        ):
            for row in list(rows or []):
                bid_u = str((row or {}).get("bid", "") or "").strip().upper()
                parsed = _parse_contract_bid_text(bid_u)
                if parsed is None:
                    continue
                lvl, strain = int(parsed[0]), str(parsed[1]).upper()
                if strain != major_u:
                    continue
                key = (bid_u, source_name)
                if key in seen:
                    continue
                seen.add(key)
                rows_out.append((lvl, row, source_name))
        return rows_out

    def _major_row_has_rebiddable_signal(target_major: str, rows_now: list[tuple[int, dict[str, Any], str]]) -> bool:
        target_u = str(target_major or "").strip().upper()
        wanted = {
            f"REBIDDABLE_{target_u}",
            f"STRONG_REBIDDABLE_{target_u}",
            f"TWICE_REBIDDABLE_{target_u}",
        }
        for _lvl, row, _source in list(rows_now or []):
            agg_expr = list((row or {}).get("agg_expr") or [])
            norm = {
                str(expr or "").strip().upper().strip("()")
                for expr in agg_expr
                if str(expr or "").strip()
            }
            if norm & wanted:
                return True
        return False

    def _major_row_est_tricks(row: dict[str, Any] | None) -> float | None:
        if not isinstance(row, dict):
            return None
        val = row.get("est_tricks")
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _major_row_supports_game(target_major: str, row: dict[str, Any] | None) -> bool:
        if not isinstance(row, dict):
            return False
        major_u = str(target_major or "").strip().upper()
        likely_final_contract = str((row or {}).get("likely_final_contract") or "").strip().upper()
        if likely_final_contract.startswith(f"4{major_u}") and "SELF" in likely_final_contract:
            return True
        guard_inputs = dict((row or {}).get("guard_inputs") or {})
        topk = list(guard_inputs.get("par_contracts_topk") or [])
        for entry in topk:
            if not isinstance(entry, dict):
                continue
            contract = str(entry.get("contract") or "").strip().upper()
            pair = str(entry.get("pair") or "").strip().upper()
            if contract != f"4{major_u}" or pair != "SELF":
                continue
            try:
                prob = float(entry.get("prob") or 0.0)
            except Exception:
                prob = 0.0
            try:
                avg_par = float(entry.get("avg_par_score") or 0.0)
            except Exception:
                avg_par = 0.0
            if prob >= 0.10 and avg_par > 0.0:
                return True
        return False

    def _row_has_forcing_probe_marker(row: dict[str, Any] | None) -> bool:
        if not isinstance(row, dict):
            return False
        norm = {
            str(expr or "").strip().upper().strip("()")
            for expr in list((row or {}).get("agg_expr") or [])
            if str(expr or "").strip()
        }
        if "FORCING_ONE_ROUND" in norm:
            return True
        return any(expr.startswith("FORCING_TO_") for expr in norm)

    def _major_slam_probe_rows(target_major: str) -> list[tuple[int, int, int, dict[str, Any], str]]:
        rows_out: list[tuple[int, int, int, dict[str, Any], str]] = []
        seen: set[tuple[str, str]] = set()
        major_u = str(target_major or "").strip().upper()
        for source_name, rows in (
            ("scored_candidate", scored),
            ("legal", legal),
            ("blocked_cannot_complete", blocked),
            ("criteria_fail", criteria_failed),
        ):
            for row in list(rows or []):
                bid_u = str((row or {}).get("bid", "") or "").strip().upper()
                parsed = _parse_contract_bid_text(bid_u)
                if parsed is None:
                    continue
                lvl, strain = int(parsed[0]), str(parsed[1]).upper()
                if strain not in ("C", "D", "H", "S") or strain == major_u or lvl >= 4:
                    continue
                suit_len = int(sl.get(strain, 0) or 0)
                if suit_len < 4 or not _row_has_forcing_probe_marker(row):
                    continue
                key = (bid_u, source_name)
                if key in seen:
                    continue
                seen.add(key)
                rows_out.append((suit_len, lvl, _strain_rank(strain), row, source_name))
        rows_out.sort(
            key=lambda item: (
                -int(item[0]),
                int(item[1]),
                -int(item[2]),
                0 if item[4] == "scored_candidate" else 1 if item[4] == "legal" else 2 if item[4] == "blocked_cannot_complete" else 3,
                -float((item[3] or {}).get("score") or 0.0),
                -float((item[3] or {}).get("matching_deal_count") or 0.0),
            )
        )
        return rows_out

    # Whitelist -3: responder should not default to a generic blocked continuation
    # over the more specific `1M-P-1oM-P-2NT-P -> 4M` contract-placement rule.
    try:
        major_nt_ctx = _major_opening_nt_rebid_response_context(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["major_opening_nt_rebid_response"] = dict(major_nt_ctx or {})
        if major_nt_ctx is not None and best_bid in ("P", "PASS", "3N", "3NT"):
            opening_major = str(major_nt_ctx.get("opening_strain") or "")
            response_major = str(major_nt_ctx.get("response_strain") or "")
            opener_support = int(sl.get(opening_major, 0) or 0)
            responder_major_len = int(sl.get(response_major, 0) or 0)
            evidence["major_opening_nt_rebid_opening_major"] = opening_major
            evidence["major_opening_nt_rebid_support_len"] = opener_support
            evidence["major_opening_nt_rebid_response_len"] = responder_major_len
            if (
                opening_major in ("H", "S")
                and opener_support >= 3
                and responder_major_len >= 4
                and tp_self is not None
                and 6.0 <= tp_self <= 10.0
            ):
                target_bid = f"4{opening_major}"
                legal_target = _pick_bid(legal, target_bid)
                blocked_target = _pick_bid(blocked, target_bid)
                failed_target = _pick_bid(criteria_failed, target_bid)
                scored_target = _pick_bid(scored, target_bid)
                picked_row = legal_target or blocked_target or failed_target or scored_target
                if picked_row is not None:
                    return {
                        "apply": True,
                        "selected_bid": target_bid,
                        "reason_codes": ["major_opening_nt_rebid_place_in_openers_major"],
                        "reason": (
                            "COMMON_SENSE_HARD_OVERRIDE: responder should place the contract in "
                            f"opener's {opening_major} after {major_nt_ctx.get('opening_bid')}-P-"
                            f"{major_nt_ctx.get('response_bid')}-P-{major_nt_ctx.get('opener_rebid_bid')}-P"
                        ),
                        "evidence": {
                            **evidence,
                            "source": (
                                "legal"
                                if legal_target is not None
                                else "blocked_cannot_complete"
                                if blocked_target is not None
                                else "criteria_fail"
                                if failed_target is not None
                                else "scored_candidate"
                            ),
                            "selected_bid": target_bid,
                        },
                    }
    except Exception:
        pass

    # Whitelist -2: when the engine would otherwise pass with only blocked
    # non-pass calls available, rescue the most natural actual-shape continuation.
    if best_bid in ("", "P", "PASS") and isinstance(sl, dict) and sl:
        if _pure_direct_overcall_one_level_opening() is None:
            natural_rescue_pool: list[tuple[float, dict[str, Any], str, int]] = []
            for source_name, rows in (
                ("legal", legal),
                ("blocked_cannot_complete", blocked),
                ("criteria_fail", criteria_failed),
            ):
                for row in rows:
                    row_bid = str((row or {}).get("bid", "") or "").strip().upper()
                    if not row_bid or row_bid in ("P", "PASS"):
                        continue
                    row_contract = _parse_contract_bid_text(row_bid)
                    if row_contract is None:
                        continue
                    row_level, row_strain = int(row_contract[0]), str(row_contract[1])
                    pref = 0.0
                    shown_idx = _last_shown_index(row_strain)
                    if row_strain in ("C", "D", "H", "S"):
                        row_len = sl.get(row_strain)
                        if row_len is None or int(row_len) < 5:
                            continue
                        pref += float(int(row_len) * 20)
                        pref -= float(int(row_level) * 50)
                        pref += float(max(shown_idx, -1) * 5)
                        if source_name == "legal":
                            pref += 25.0
                        elif source_name == "blocked_cannot_complete":
                            pref += 15.0
                    elif row_strain == "N":
                        pref += 20.0
                        pref -= float(int(row_level) * 40)
                        if source_name == "legal":
                            pref += 20.0
                    else:
                        continue
                    natural_rescue_pool.append((pref, row, source_name, shown_idx))
            if natural_rescue_pool:
                natural_rescue_pool.sort(
                    key=lambda item: (
                        -float(item[0]),
                        -int(item[3]),
                        str((item[1] or {}).get("bid", "") or "").strip().upper(),
                    )
                )
                chosen_row = natural_rescue_pool[0][1]
                chosen_bid = str((chosen_row or {}).get("bid", "") or "").strip().upper()
                if chosen_bid:
                    return {
                        "apply": True,
                        "selected_bid": chosen_bid,
                        "reason_codes": ["blocked_natural_continuation_rescue"],
                        "reason": (
                            "COMMON_SENSE_HARD_OVERRIDE: replace forced pass with a natural blocked continuation "
                            f"{chosen_bid}"
                        ),
                        "evidence": {
                            **evidence,
                            "selected_source": natural_rescue_pool[0][2],
                            "selected_last_shown_index": natural_rescue_pool[0][3],
                        },
                    }

    # Whitelist -1: if the current best bid is a non-rebiddable same-suit rebid,
    # rescue a more natural same-level continuation from the legal/blocked pools.
    try:
        best_contract = _parse_contract_bid_text(best_bid)
        if best_contract is not None and isinstance(sl, dict) and sl:
            best_level, best_strain = int(best_contract[0]), str(best_contract[1])
            if best_strain in ("C", "D", "H", "S"):
                nr_pen, nr_reason, nr_hard = compute_non_rebiddable_suit_rebid_penalty(
                    bid_text=best_bid,
                    auction_tokens=list(auction_tokens or []),
                    acting_direction=acting_direction,
                    dealer_actual=dealer_actual,
                    bt_acting_criteria=None,
                    self_suit_lengths=sl,
                )
                if nr_pen > 0.0:
                    candidate_pool: list[tuple[float, dict[str, Any], str]] = []
                    for source_name, rows in (
                        ("legal", legal),
                        ("blocked_cannot_complete", blocked),
                        ("criteria_fail", criteria_failed),
                    ):
                        for row in rows:
                            row_bid = str((row or {}).get("bid", "") or "").strip().upper()
                            if not row_bid or row_bid in ("P", "PASS", best_bid):
                                continue
                            row_contract = _parse_contract_bid_text(row_bid)
                            if row_contract is None:
                                continue
                            row_level, row_strain = int(row_contract[0]), str(row_contract[1])
                            if row_level != best_level:
                                continue
                            pref = 0.0
                            if row_strain in ("C", "D", "H", "S"):
                                row_len = sl.get(row_strain)
                                if row_len is None or int(row_len) < 5:
                                    continue
                                cand_pen, _cand_reason, cand_hard = compute_non_rebiddable_suit_rebid_penalty(
                                    bid_text=row_bid,
                                    auction_tokens=list(auction_tokens or []),
                                    acting_direction=acting_direction,
                                    dealer_actual=dealer_actual,
                                    bt_acting_criteria=list((row or {}).get("agg_expr") or []),
                                    self_suit_lengths=sl,
                                )
                                if cand_hard:
                                    continue
                                pref += float(int(row_len) * 20)
                                if cand_pen <= 0.0:
                                    pref += 180.0
                                if row_strain != best_strain:
                                    pref += 40.0
                                if source_name == "legal":
                                    pref += 25.0
                                elif source_name == "blocked_cannot_complete":
                                    pref += 15.0
                            elif row_strain == "N":
                                pref += 60.0
                                if source_name == "legal":
                                    pref += 20.0
                            else:
                                continue
                            candidate_pool.append((pref, row, source_name))
                    if candidate_pool:
                        candidate_pool.sort(
                            key=lambda item: (
                                -float(item[0]),
                                str((item[1] or {}).get("bid", "") or "").strip().upper(),
                            )
                        )
                        chosen_row = candidate_pool[0][1]
                        chosen_bid = str((chosen_row or {}).get("bid", "") or "").strip().upper()
                        if chosen_bid:
                            return {
                                "apply": True,
                                "selected_bid": chosen_bid,
                                "reason_codes": ["natural_rebid_rescue_from_non_rebiddable_choice"],
                                "reason": (
                                    "COMMON_SENSE_HARD_OVERRIDE: replace non-rebiddable rebid "
                                    f"{best_bid} with more natural same-level continuation {chosen_bid}"
                                ),
                                "evidence": {
                                    **evidence,
                                    "current_best_non_rebiddable_reason": nr_reason,
                                    "current_best_hard_blocked": bool(nr_hard),
                                    "selected_source": candidate_pool[0][2],
                                },
                            }
    except Exception:
        pass

    # Whitelist -0.5: if the current best action is NT, allow a blocked
    # second-suit rebid that matches the actor's previously shown two-suit shape.
    try:
        best_contract = _parse_contract_bid_text(best_bid)
        if best_contract is not None and str(best_contract[1]) == "N" and isinstance(sl, dict) and sl:
            second_suit_blocked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for row in blocked:
                row_bid = str((row or {}).get("bid", "") or "").strip().upper()
                if not row_bid:
                    continue
                ok, detail = _qualifies_second_suit_rebid_semantics(row_bid)
                if not ok:
                    continue
                row_contract = _parse_contract_bid_text(row_bid)
                if row_contract is None:
                    continue
                row_level, row_strain = int(row_contract[0]), str(row_contract[1])
                pref = 0.0
                pref += float(int(sl.get(row_strain, 0)) * 25)
                pref -= float(int(row_level) * 20)
                pref += float(max(_last_shown_index(row_strain), -1) * 5)
                second_suit_blocked.append((pref, row, detail))
            if second_suit_blocked:
                second_suit_blocked.sort(
                    key=lambda item: (
                        -float(item[0]),
                        str((item[1] or {}).get("bid", "") or "").strip().upper(),
                    )
                )
                chosen_row = second_suit_blocked[0][1]
                chosen_bid = str((chosen_row or {}).get("bid", "") or "").strip().upper()
                if chosen_bid:
                    return {
                        "apply": True,
                        "selected_bid": chosen_bid,
                        "reason_codes": ["blocked_second_suit_rebid_rescue"],
                        "reason": (
                            "COMMON_SENSE_HARD_OVERRIDE: prefer valid second-suit rebid "
                            f"{chosen_bid} over NT when BT marks it cannot_complete"
                        ),
                        "evidence": {
                            **evidence,
                            "selected_source": "blocked_cannot_complete",
                            "second_suit_rebid_detail": second_suit_blocked[0][2],
                        },
                    }
    except Exception:
        pass

    # Whitelist -0.25: after landing on 3NT with clear slam values, prefer the
    # cheapest blocked NT continuation over an artificial signoff.
    try:
        best_contract = _parse_contract_bid_text(best_bid)
        if (
            forced_stayman_reply_bid is None
            and forced_transfer_bid is None
            and best_contract is not None
            and int(best_contract[0]) == 3
            and str(best_contract[1]) == "N"
        ):
            if (
                _is_direct_overcall_gap_rescued_3n(legal)
                or _is_direct_overcall_gap_rescued_3n(blocked)
                or _is_direct_overcall_gap_rescued_3n(criteria_failed)
            ):
                return {"apply": False}
            if tp_combined is not None and tp_combined >= 34.0:
                nt_slam_candidates: list[tuple[float, dict[str, Any], str, str]] = []
                for source_name, rows in (
                    ("blocked_cannot_complete", blocked),
                    ("criteria_fail", criteria_failed),
                    ("legal", legal),
                ):
                    row_4n = _pick_bid(rows, "4N")
                    if row_4n is not None:
                        pref = 260.0
                        if source_name == "blocked_cannot_complete":
                            pref += 20.0
                        elif source_name == "criteria_fail":
                            pref += 10.0
                        try:
                            pref += min(20.0, max(0.0, float((row_4n or {}).get("matching_deal_count") or 0.0) / 50000.0))
                        except Exception:
                            pass
                        nt_slam_candidates.append((pref, row_4n, source_name, "4N"))

                    if tp_combined >= 36.0:
                        row_6n = _pick_bid(rows, "6N")
                        if row_6n is not None:
                            pref = 210.0
                            if source_name == "blocked_cannot_complete":
                                pref += 20.0
                            elif source_name == "criteria_fail":
                                pref += 10.0
                            try:
                                pref += min(20.0, max(0.0, float((row_6n or {}).get("matching_deal_count") or 0.0) / 50000.0))
                            except Exception:
                                pass
                            nt_slam_candidates.append((pref, row_6n, source_name, "6N"))

                if nt_slam_candidates:
                    nt_slam_candidates.sort(
                        key=lambda item: (
                            -float(item[0]),
                            str(item[3]),
                        )
                    )
                    chosen_row = nt_slam_candidates[0][1]
                    chosen_bid = str((chosen_row or {}).get("bid", "") or "").strip().upper()
                    if chosen_bid:
                        return {
                            "apply": True,
                            "selected_bid": chosen_bid,
                            "reason_codes": ["blocked_nt_slam_followup_rescue"],
                            "reason": (
                                "COMMON_SENSE_HARD_OVERRIDE: prefer blocked NT slam continuation "
                                f"{chosen_bid} over 3NT signoff when combined values are slam-positive"
                            ),
                            "evidence": {
                                **evidence,
                                "selected_source": nt_slam_candidates[0][2],
                                "slam_value_threshold_met": True,
                                "selected_nt_followup": chosen_bid,
                            },
                        }
    except Exception:
        pass

    # Whitelist 0: with equal 5-5 or 6-6 suits, prefer the higher-ranked suit.
    try:
        best_contract = _parse_contract_bid_text(best_bid)
        if best_contract is not None and isinstance(sl, dict) and sl:
            best_level, best_strain = int(best_contract[0]), str(best_contract[1])
            if best_strain in ("C", "D", "H", "S"):
                same_level_suits: list[tuple[str, int]] = []
                for row in legal:
                    row_bid = str((row or {}).get("bid", "") or "").strip().upper()
                    row_contract = _parse_contract_bid_text(row_bid)
                    if row_contract is None:
                        continue
                    row_level, row_strain = int(row_contract[0]), str(row_contract[1])
                    if row_level != best_level or row_strain not in ("C", "D", "H", "S"):
                        continue
                    row_len = sl.get(row_strain)
                    if row_len is None:
                        continue
                    same_level_suits.append((row_strain, int(row_len)))

                best_len = sl.get(best_strain)
                if best_len is not None and int(best_len) in (5, 6):
                    same_len_suits = sorted(
                        {
                            str(row_strain)
                            for row_strain, row_len in same_level_suits
                            if int(row_len) == int(best_len)
                        },
                        key=_strain_rank,
                    )
                    if len(same_len_suits) >= 2:
                        preferred_strain = same_len_suits[-1]
                        if _strain_rank(preferred_strain) > _strain_rank(best_strain):
                            selected_bid = f"{best_level}{preferred_strain}"
                            if any(
                                str((row or {}).get("bid", "") or "").strip().upper() == selected_bid
                                for row in legal
                            ):
                                return {
                                    "apply": True,
                                    "selected_bid": selected_bid,
                                    "reason_codes": ["equal_length_higher_suit_preference"],
                                    "reason": "COMMON_SENSE_HARD_OVERRIDE: prefer higher-ranked suit with equal 5-5 or 6-6 shape",
                                    "evidence": {
                                        **evidence,
                                        "equal_length": int(best_len),
                                        "same_level_suits": same_len_suits,
                                        "preferred_strain": preferred_strain,
                                    },
                                }
    except Exception:
        pass

    # Whitelist 0.25: in a pure direct-overcall seat over a lone opponent
    # one-level opening, do not allow pass to beat a natural legal suit overcall
    # when the actor has real one-suited values.
    try:
        if best_bid in ("P", "PASS") and isinstance(sl, dict) and sl:
            direct_overcall_ctx = _pure_direct_overcall_one_level_opening()
            if direct_overcall_ctx is not None:
                opp_open_lvl, opp_open_st = direct_overcall_ctx
                direct_overcall_rows: list[tuple[int, int, int, dict[str, Any]]] = []
                for row in legal:
                    row_bid = str((row or {}).get("bid", "") or "").strip().upper()
                    row_contract = _parse_contract_bid_text(row_bid)
                    if row_contract is None:
                        continue
                    row_level, row_strain = int(row_contract[0]), str(row_contract[1])
                    if row_strain not in ("C", "D", "H", "S"):
                        continue
                    if row_strain == opp_open_st:
                        continue
                    row_len = sl.get(row_strain)
                    if row_len is None:
                        continue
                    row_len_i = int(row_len)
                    strong_enough = bool(
                        (row_len_i >= 6 and tp_self is not None and tp_self >= 10.0)
                        or (row_len_i >= 5 and tp_self is not None and tp_self >= 13.0)
                    )
                    if not strong_enough:
                        continue
                    direct_overcall_rows.append((row_len_i, row_level, _strain_rank(row_strain), row))

                if direct_overcall_rows:
                    direct_overcall_rows.sort(
                        key=lambda item: (
                            -int(item[0]),
                            int(item[1]),
                            -int(item[2]),
                            -float((item[3] or {}).get("matching_deal_count") or 0.0),
                            str((item[3] or {}).get("bid", "") or "").strip().upper(),
                        )
                    )
                    picked = str((direct_overcall_rows[0][3] or {}).get("bid", "") or "").strip().upper()
                    if picked:
                        return {
                            "apply": True,
                            "selected_bid": picked,
                            "reason_codes": ["direct_natural_overcall_over_one_level_opening"],
                            "reason": (
                                "COMMON_SENSE_HARD_OVERRIDE: prefer natural suit overcall "
                                "over passing opponent's one-level opening"
                            ),
                            "evidence": {
                                **evidence,
                                "opponent_opening": f"{opp_open_lvl}{opp_open_st}",
                                "selected_overcall_bid": picked,
                                "selected_overcall_len": int(direct_overcall_rows[0][0]),
                                "current_best_bid": best_bid,
                            },
                        }
    except Exception:
        pass

    # Whitelist 0.5: uncontested responder should show a real 5+ card major
    # over a one-level minor response after partner opens 1m.
    try:
        best_contract = _parse_contract_bid_text(best_bid)
        if best_contract is not None and isinstance(sl, dict) and sl:
            best_level, best_strain = int(best_contract[0]), str(best_contract[1])
            directions = ("N", "E", "S", "W")
            acting_dir_u = str(acting_direction or "").strip().upper()
            dealer_u = str(dealer_actual or "N").strip().upper()
            if acting_dir_u in directions and dealer_u in directions:
                dealer_idx = directions.index(dealer_u)
                partner_dir = _partner_dir(acting_dir_u)
                partner_contracts: list[tuple[int, int, str]] = []
                opp_contracts: list[tuple[int, int, str]] = []
                acting_has_bid = False
                for i, tk in enumerate(list(auction_tokens or [])):
                    bidder_dir = directions[(dealer_idx + int(i)) % 4]
                    parsed = _parse_contract_bid_text(str(tk or "").strip().upper())
                    if bidder_dir == acting_dir_u and parsed is not None:
                        acting_has_bid = True
                    if parsed is None:
                        continue
                    lvl, st = int(parsed[0]), str(parsed[1])
                    if bidder_dir == partner_dir:
                        partner_contracts.append((i, lvl, st))
                    elif _dir_side(bidder_dir) != _dir_side(acting_dir_u):
                        opp_contracts.append((i, lvl, st))

                if (
                    not acting_has_bid
                    and len(partner_contracts) == 1
                    and len(opp_contracts) == 0
                    and partner_contracts[0][1] == 1
                    and partner_contracts[0][2] in ("C", "D")
                    and best_level == 1
                    and best_strain in ("C", "D")
                ):
                    legal_major_rows: list[tuple[int, int, dict[str, Any]]] = []
                    for row in legal:
                        row_bid = str((row or {}).get("bid", "") or "").strip().upper()
                        row_contract = _parse_contract_bid_text(row_bid)
                        if row_contract is None:
                            continue
                        row_level, row_strain = int(row_contract[0]), str(row_contract[1])
                        if row_level != 1 or row_strain not in ("H", "S"):
                            continue
                        row_len = sl.get(row_strain)
                        if row_len is None or int(row_len) < 5:
                            continue
                        legal_major_rows.append((int(row_len), _strain_rank(row_strain), row))

                    if legal_major_rows:
                        legal_major_rows.sort(
                            key=lambda item: (
                                -int(item[0]),
                                -int(item[1]),
                                str((item[2] or {}).get("bid", "") or "").strip().upper(),
                            )
                        )
                        picked = str((legal_major_rows[0][2] or {}).get("bid", "")).strip().upper()
                        if picked and picked != best_bid:
                            return {
                                "apply": True,
                                "selected_bid": picked,
                                "reason_codes": ["major_over_minor_response_after_minor_opening"],
                                "reason": "COMMON_SENSE_HARD_OVERRIDE: prefer 5+ card major response over one-level minor response",
                                "evidence": {
                                    **evidence,
                                    "partner_opening": f"{partner_contracts[0][1]}{partner_contracts[0][2]}",
                                    "selected_major_len": int(legal_major_rows[0][0]),
                                    "selected_major_bid": picked,
                                    "current_best_bid": best_bid,
                                },
                            }
    except Exception:
        pass

    # Whitelist 0.75: partner's forcing 3H bid should not die in pass-out.
    try:
        if best_bid in ("P", "PASS") and isinstance(sl, dict) and int(sl.get("S", 0) or 0) >= 3:
            toks_u = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
            partner_dir = _partner_dir(acting_direction)
            partner_last_bid = None
            if partner_dir:
                for idx in range(len(toks_u) - 1, -1, -1):
                    bidder_dir = _token_bidder_dir(idx, dealer_actual)
                    tk_u = str(toks_u[idx] or "").strip().upper()
                    if bidder_dir == partner_dir and tk_u not in ("P", "PASS", "X", "XX"):
                        partner_last_bid = tk_u
                        break
            if partner_last_bid == "3H":
                spade_rows: list[tuple[int, dict[str, Any], str]] = []
                for source_name, rows in (
                    ("scored_candidate", scored),
                    ("legal", legal),
                    ("blocked_cannot_complete", blocked),
                    ("criteria_fail", criteria_failed),
                ):
                    for row in list(rows or []):
                        bid_u = str((row or {}).get("bid", "") or "").strip().upper()
                        parsed = _parse_contract_bid_text(bid_u)
                        if parsed is None:
                            continue
                        lvl, strain = int(parsed[0]), str(parsed[1]).upper()
                        if strain != "S" or lvl < 3:
                            continue
                        spade_rows.append((lvl, row, source_name))
                if spade_rows:
                    spade_rows.sort(
                        key=lambda item: (
                            int(item[0]),
                            0 if item[2] == "scored_candidate" else 1 if item[2] == "legal" else 2 if item[2] == "blocked_cannot_complete" else 3,
                            -float((item[1] or {}).get("matching_deal_count") or 0.0),
                            str((item[1] or {}).get("bid", "") or "").strip().upper(),
                        )
                    )
                    picked = str((spade_rows[0][1] or {}).get("bid", "") or "").strip().upper()
                    if picked:
                        return {
                            "apply": True,
                            "selected_bid": picked,
                            "reason_codes": ["forcing_heart_spade_acceptance"],
                            "reason": (
                                "COMMON_SENSE_HARD_OVERRIDE: partner's forcing 3H bid requires "
                                f"a spade continuation, not pass; choose {picked}"
                            ),
                            "evidence": {
                                **evidence,
                                "partner_last_bid": partner_last_bid,
                                "source": spade_rows[0][2],
                                "spade_len": int(sl.get("S", 0) or 0),
                            },
                        }
    except Exception:
        pass

    # Whitelist B.5: self-sufficient long major progression / game commitment.
    try:
        if isinstance(sl, dict) and sl:
            best_bid_parsed = _parse_contract_bid_text(best_bid)
            best_bid_row = _pick_bid(scored, best_bid)
            for target_major in ("H", "S"):
                major_len = int(sl.get(target_major, 0) or 0)
                if major_len < 6:
                    continue
                major_rows = _candidate_rows_for_major(target_major)
                if not major_rows:
                    continue

                game_bid = f"4{target_major}"
                game_row, game_source = _bid_row_with_source(game_bid)
                lower_rows = [
                    (lvl, row, source_name)
                    for lvl, row, source_name in list(major_rows or [])
                    if int(lvl) < 4
                ]
                if not lower_rows and game_row is None:
                    continue

                best_same_major_lower = bool(
                    best_bid_parsed is not None
                    and str(best_bid_parsed[1]).upper() == target_major
                    and int(best_bid_parsed[0]) < 4
                )
                major_rebiddable = _major_row_has_rebiddable_signal(target_major, major_rows)

                trick_sources: list[dict[str, Any]] = []
                if isinstance(best_bid_row, dict):
                    trick_sources.append(best_bid_row)
                trick_sources.extend([row for _lvl, row, _src in lower_rows if isinstance(row, dict)])
                if isinstance(game_row, dict):
                    trick_sources.append(game_row)

                est_tricks_vals = [
                    float(v)
                    for v in (_major_row_est_tricks(row) for row in trick_sources)
                    if isinstance(v, (int, float))
                ]
                max_est_tricks = max(est_tricks_vals) if est_tricks_vals else None

                game_support = any(
                    _major_row_supports_game(target_major, row)
                    for row in trick_sources
                )

                evidence_score = 0
                if major_len >= 7:
                    evidence_score += 1
                if tp_self is not None and tp_self >= 17.0:
                    evidence_score += 1
                if major_rebiddable:
                    evidence_score += 1
                if max_est_tricks is not None and max_est_tricks >= 8.5:
                    evidence_score += 1
                if max_est_tricks is not None and max_est_tricks >= 9.0:
                    evidence_score += 1
                if game_support:
                    evidence_score += 2

                if (
                    game_row is not None
                    and best_bid != game_bid
                    and major_len >= 7
                    and tp_self is not None
                    and tp_self >= 17.0
                    and major_rebiddable
                    and evidence_score >= 5
                    and (
                        game_support
                        or (max_est_tricks is not None and max_est_tricks >= 9.0)
                    )
                ):
                    return {
                        "apply": True,
                        "selected_bid": game_bid,
                        "reason_codes": ["self_sufficient_long_major_game_commit"],
                        "reason": (
                            "COMMON_SENSE_HARD_OVERRIDE: long self-sufficient major should "
                            f"commit directly to {game_bid}"
                        ),
                        "evidence": {
                            **evidence,
                            "source": game_source,
                            "target_major": target_major,
                            "major_len": major_len,
                            "self_sufficient_major_evidence_score": evidence_score,
                            "self_sufficient_major_rebiddable": major_rebiddable,
                            "self_sufficient_major_est_tricks": max_est_tricks,
                            "self_sufficient_major_game_support": bool(game_support),
                        },
                    }

                slam_probe_rows = _major_slam_probe_rows(target_major)
                slam_probe_bid = (
                    str((slam_probe_rows[0][3] or {}).get("bid", "") or "").strip().upper()
                    if slam_probe_rows
                    else ""
                )
                slam_probe_source = slam_probe_rows[0][4] if slam_probe_rows else None
                slam_probe_suit = ""
                slam_probe_len = 0
                if slam_probe_rows:
                    slam_probe_parsed = _parse_contract_bid_text(slam_probe_bid)
                    slam_probe_suit = str(slam_probe_parsed[1]).upper() if slam_probe_parsed is not None else ""
                    slam_probe_len = int(slam_probe_rows[0][0])

                if (
                    slam_probe_bid
                    and slam_probe_bid != best_bid
                    and evidence_score >= 5
                    and game_support
                    and tp_self is not None
                    and tp_self >= 18.0
                    and max_est_tricks is not None
                    and max_est_tricks >= 9.0
                ):
                    return {
                        "apply": True,
                        "selected_bid": slam_probe_bid,
                        "reason_codes": ["self_sufficient_long_major_slam_probe"],
                        "reason": (
                            "COMMON_SENSE_HARD_OVERRIDE: long self-sufficient major with game in hand "
                            f"should probe below game with {slam_probe_bid} before signing off"
                        ),
                        "evidence": {
                            **evidence,
                            "source": slam_probe_source,
                            "target_major": target_major,
                            "major_len": major_len,
                            "probe_suit": slam_probe_suit,
                            "probe_len": slam_probe_len,
                            "self_sufficient_major_evidence_score": evidence_score,
                            "self_sufficient_major_rebiddable": major_rebiddable,
                            "self_sufficient_major_est_tricks": max_est_tricks,
                            "self_sufficient_major_game_support": bool(game_support),
                        },
                    }

                if best_same_major_lower:
                    continue

                progression_rows = sorted(
                    list(lower_rows or []),
                    key=lambda item: (
                        int(item[0]),
                        0 if item[2] == "legal" else 1 if item[2] == "scored_candidate" else 2 if item[2] == "blocked_cannot_complete" else 3,
                        -float((item[1] or {}).get("matching_deal_count") or 0.0),
                    ),
                )
                progression_bid = (
                    str((progression_rows[0][1] or {}).get("bid", "") or "").strip().upper()
                    if progression_rows
                    else ""
                )
                progression_source = progression_rows[0][2] if progression_rows else None

                if (
                    progression_bid
                    and progression_bid != best_bid
                    and evidence_score >= 3
                    and (
                        major_rebiddable
                        or (tp_self is not None and tp_self >= 16.0)
                        or (max_est_tricks is not None and max_est_tricks >= 8.5)
                    )
                ):
                    if (
                        best_bid == "2C"
                        and _opening_round_has_strong_two_club_available()
                    ):
                        continue
                    return {
                        "apply": True,
                        "selected_bid": progression_bid,
                        "reason_codes": ["self_sufficient_long_major_progression"],
                        "reason": (
                            "COMMON_SENSE_HARD_OVERRIDE: long major should keep the natural "
                            f"{target_major} progression alive with {progression_bid}"
                        ),
                        "evidence": {
                            **evidence,
                            "source": progression_source,
                            "target_major": target_major,
                            "major_len": major_len,
                            "self_sufficient_major_evidence_score": evidence_score,
                            "self_sufficient_major_rebiddable": major_rebiddable,
                            "self_sufficient_major_est_tricks": max_est_tricks,
                            "self_sufficient_major_game_support": bool(game_support),
                        },
                    }
    except Exception:
        pass

    # Whitelist A: double over opponent NT partscore.
    try:
        opp_last = _last_opponent_contract(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["last_opponent_contract"] = opp_last
        has_dbl_legal = next((r for r in legal if str((r or {}).get("bid", "")).strip().upper() in ("D", "X", "DOUBLE")), None)
        has_dbl_blocked = next((r for r in blocked if str((r or {}).get("bid", "")).strip().upper() in ("D", "X", "DOUBLE")), None)
        if (
            opp_last is not None
            and opp_last[1] == "N"
            and 1 <= int(opp_last[0]) <= 3
            and tp_self is not None
            and tp_self >= 10.0
            and (tp_combined is None or tp_combined >= 22.0)
            and (has_dbl_legal is not None or has_dbl_blocked is not None)
            and best_bid not in ("D", "X", "DOUBLE")
        ):
            return {
                "apply": True,
                "selected_bid": "D",
                "reason_codes": ["double_over_nt_partscore"],
                "reason": "COMMON_SENSE_HARD_OVERRIDE: double over opponent NT partscore",
                "evidence": {
                    **evidence,
                    "source": "legal" if has_dbl_legal is not None else "blocked_cannot_complete",
                },
            }
    except Exception:
        pass

    # Whitelist B: major-fit raise after partner major show.
    try:
        majors = _partner_shown_majors(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["partner_shown_majors"] = sorted(list(majors))
        if majors:
            target = max(list(majors), key=lambda s: int(sl.get(s, 0)))
            support = int(sl.get(target, 0))
            acting_has_bid = _actor_has_bid_contract()
            partner_last_bid = None
            partner_dir = _partner_dir(acting_direction)
            if partner_dir:
                for idx in range(len(auction_tokens) - 1, -1, -1):
                    bidder_dir = _token_bidder_dir(idx, dealer_actual)
                    tk_u = str(auction_tokens[idx] or "").strip().upper()
                    if bidder_dir == partner_dir and tk_u not in ("", "P", "PASS", "X", "XX"):
                        partner_last_bid = tk_u
                        break
            forcing_spade_accept_ctx = bool(
                best_bid in ("P", "PASS")
                and target == "S"
                and support >= 3
                and partner_last_bid == "3H"
            )
            if (support >= 4 and tp_self is not None and tp_self >= 10.0) or forcing_spade_accept_ctx:
                best_bid_parsed = _parse_contract_bid_text(best_bid)
                protected_temporizing = False
                if (
                    not acting_has_bid
                    and tp_self >= 13.0
                    and best_bid_parsed is not None
                ):
                    best_bid_level, best_bid_strain = int(best_bid_parsed[0]), str(best_bid_parsed[1])
                    protected_temporizing = (
                        best_bid in ("2N", "2NT")
                        or (
                            best_bid_level == 2
                            and best_bid_strain in ("C", "D", "H", "S")
                            and best_bid_strain != target
                        )
                    )
                if protected_temporizing:
                    evidence["major_fit_temporizing_bid_protected"] = best_bid
                    evidence["major_fit_target"] = target
                    evidence["major_fit_support_len"] = support
                    return {}
                raises: list[dict[str, Any]] = []
                raise_sources = (
                    (("legal", legal), ("blocked_cannot_complete", blocked), ("criteria_fail", criteria_failed))
                    if forcing_spade_accept_ctx
                    else (("legal", legal),)
                )
                raises_with_source: list[tuple[dict[str, Any], str]] = []
                for source_name, rows in raise_sources:
                    for r in list(rows or []):
                        b = str((r or {}).get("bid", "") or "").strip().upper()
                        c = _parse_contract_bid_text(b)
                        if c is None:
                            continue
                        lvl, st = int(c[0]), str(c[1])
                        if st == target and lvl >= 3:
                            raises.append(r)
                            raises_with_source.append((r, source_name))
                if raises:
                    if forcing_spade_accept_ctx:
                        raises_with_source.sort(
                            key=lambda item: (
                                int((_parse_contract_bid_text(str((item[0] or {}).get("bid", "") or "").strip().upper()) or (9, "S"))[0]),
                                0 if item[1] == "legal" else 1 if item[1] == "blocked_cannot_complete" else 2,
                                -float((item[0] or {}).get("matching_deal_count") or 0.0),
                            )
                        )
                        picked = str((raises_with_source[0][0] or {}).get("bid", "")).strip().upper()
                        if picked and picked != best_bid:
                            return {
                                "apply": True,
                                "selected_bid": picked,
                                "reason_codes": ["major_fit_raise_after_partner_shows_heart_or_spade_fit"],
                                "reason": (
                                    "COMMON_SENSE_HARD_OVERRIDE: partner's forcing 3H bid "
                                    f"requires spade acceptance with {picked}"
                                ),
                                "evidence": {
                                    **evidence,
                                    "major": target,
                                    "support_len": support,
                                    "forcing_spade_acceptance": True,
                                    "partner_last_bid": partner_last_bid,
                                    "source": raises_with_source[0][1],
                                },
                            }
                    major_game_level = 4 if target in ("H", "S") else 5
                    sane_raises = []
                    for rr in raises:
                        _bid = str((rr or {}).get("bid", "")).strip().upper()
                        _parsed = _parse_contract_bid_text(_bid)
                        if _parsed is None:
                            continue
                        _lvl, _st = int(_parsed[0]), str(_parsed[1])
                        if _st == target and _lvl <= major_game_level:
                            sane_raises.append(rr)
                    if sane_raises:
                        raises = sane_raises
                    else:
                        # Do not force a major-fit "raise" into slam just because stale BT
                        # buckets left only wild high-level calls legal.
                        return {}
                    best_bid_same_major_below_game = (
                        best_bid_parsed is not None
                        and str(best_bid_parsed[1]) == target
                        and int(best_bid_parsed[0]) < major_game_level
                    )
                    prefer_game_raise = (
                        (tp_combined is not None and tp_combined >= 25.0)
                        or (
                            tp_self is not None
                            and tp_self >= 18.0
                            and support >= 4
                            and any(
                                (
                                    (_rr_parsed := _parse_contract_bid_text(str((_rr or {}).get("bid", "")).strip().upper()))
                                    is not None
                                    and int(_rr_parsed[0]) == major_game_level
                                    and str(_rr_parsed[1]) == target
                                )
                                for _rr in raises
                            )
                        )
                    )
                    if best_bid_same_major_below_game and not prefer_game_raise:
                        capped_raises = []
                        best_bid_level = int(best_bid_parsed[0])
                        for rr in raises:
                            _bid = str((rr or {}).get("bid", "")).strip().upper()
                            _parsed = _parse_contract_bid_text(_bid)
                            if _parsed is None:
                                continue
                            _lvl, _st = int(_parsed[0]), str(_parsed[1])
                            if _st == target and _lvl <= best_bid_level:
                                capped_raises.append(rr)
                        if capped_raises:
                            raises = capped_raises
                    if acting_has_bid and best_bid_same_major_below_game:
                        evidence["major_fit_existing_raise_kept"] = best_bid
                        evidence["major_fit_target"] = target
                        evidence["major_fit_support_len"] = support
                        return {}
                    raises.sort(
                        key=lambda rr: (
                            (
                                -1
                                if prefer_game_raise
                                else 1
                            ) * int((_parse_contract_bid_text(str((rr or {}).get("bid", "") or "").strip().upper()) or (0, "C"))[0]),
                            -float((rr or {}).get("matching_deal_count") or 0.0),
                        )
                    )
                    picked = str((raises[0] or {}).get("bid", "")).strip().upper()
                    if picked and picked != best_bid:
                        return {
                            "apply": True,
                            "selected_bid": picked,
                            "reason_codes": ["major_fit_raise_after_partner_shows_heart_or_spade_fit"],
                            "reason": "COMMON_SENSE_HARD_OVERRIDE: force major-fit raise",
                            "evidence": {
                                **evidence,
                                "major": target,
                                "support_len": support,
                                "prefer_game_raise": bool(prefer_game_raise),
                            },
                        }
    except Exception:
        pass

    # Whitelist C: non-pass after partner takeout double with values.
    try:
        partner_dbl = _partner_takeout_double_shown(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["partner_takeout_double_shown"] = bool(partner_dbl)
        if partner_dbl and tp_self is not None and tp_self >= 11.0 and best_bid in ("P", "PASS") and legal:
            opp_last = _last_opponent_contract(
                auction_tokens,
                acting_direction=acting_direction,
                dealer_actual=dealer_actual,
            )
            opp_st = str(opp_last[1]) if opp_last is not None else None
            ranked = sorted(list(legal), key=lambda r: -float(_priority(r, opp_st)))
            picked = str((ranked[0] or {}).get("bid", "")).strip().upper() if ranked else ""
            if picked:
                return {
                    "apply": True,
                    "selected_bid": picked,
                    "reason_codes": ["non_pass_after_partner_takeout_double_with_values"],
                    "reason": "COMMON_SENSE_HARD_OVERRIDE: disallow pass after partner takeout double",
                    "evidence": evidence,
                }
    except Exception:
        pass

    # Whitelist D: responder should not pass opener's Stayman major reply with
    # obvious misfit when a constructive continuation is available.
    try:
        stayman_ctx = _stayman_major_response_context(
            auction_tokens,
            acting_direction=acting_direction,
            dealer_actual=dealer_actual,
        )
        evidence["stayman_major_response"] = dict(stayman_ctx or {})
        if stayman_ctx is not None and best_bid in ("P", "PASS"):
            shown_major = str(stayman_ctx.get("shown_major") or "")
            support = int(sl.get(shown_major, 0))
            evidence["stayman_shown_major"] = shown_major
            evidence["stayman_support_len"] = support
            if shown_major in ("H", "S") and support <= 2 and tp_self is not None and tp_self >= 8.0:
                nt_legal: list[dict[str, Any]] = []
                other_legal: list[dict[str, Any]] = []
                for row in legal:
                    bid = str((row or {}).get("bid", "") or "").strip().upper()
                    if bid in ("2N", "2NT", "3N", "3NT"):
                        nt_legal.append(row)
                    elif bid not in ("", "P", "PASS"):
                        other_legal.append(row)
                evidence["stayman_legal_nt_candidates"] = [
                    str((row or {}).get("bid", "") or "").strip().upper()
                    for row in nt_legal
                ]
                evidence["stayman_other_legal_candidates"] = [
                    str((row or {}).get("bid", "") or "").strip().upper()
                    for row in other_legal
                ]
                preferred_nts = ("3N", "3NT", "2N", "2NT") if tp_self >= 10.0 else ("2N", "2NT", "3N", "3NT")
                picked = ""
                for pref in preferred_nts:
                    if any(str((row or {}).get("bid", "") or "").strip().upper() == pref for row in nt_legal):
                        picked = pref
                        break
                if not picked and other_legal:
                    picked = str((other_legal[0] or {}).get("bid", "") or "").strip().upper()
                if picked:
                    return {
                        "apply": True,
                        "selected_bid": picked,
                        "reason_codes": ["stayman_major_misfit_non_pass"],
                        "reason": "COMMON_SENSE_HARD_OVERRIDE: avoid passing partner's Stayman major with a doubleton fit",
                        "evidence": evidence,
                    }
    except Exception:
        pass

    # Whitelist E: after ...3H-P-3S-P, opener with a long heart suit and
    # game-going values should not sign off below game just because `4H`
    # failed stale BT range caps.
    try:
        toks = [str(t or "").strip().upper() for t in list(auction_tokens or []) if str(t or "").strip()]
        evidence["forced_major_game_tokens_tail"] = toks[-4:] if len(toks) >= 4 else list(toks)
        if len(toks) >= 4 and toks[-4:] == ["3H", "P", "3S", "P"] and best_bid in ("P", "PASS"):
            directions = ["N", "E", "S", "W"]
            d = str(dealer_actual or "N").upper()
            dealer_idx = directions.index(d) if d in directions else 0

            def _token_bidder_dir(token_idx: int) -> str:
                return directions[(dealer_idx + int(token_idx)) % 4]

            opener_dir = _token_bidder_dir(len(toks) - 4)
            heart_len = int(sl.get("H", 0) or 0)
            evidence["forced_major_game_opener_dir"] = opener_dir
            evidence["forced_major_game_heart_len"] = heart_len
            if (
                str(acting_direction or "").upper() == opener_dir
                and tp_self is not None
                and tp_self >= 17.0
                and heart_len >= 6
            ):
                legal_4h = _pick_bid(legal, "4H")
                blocked_4h = _pick_bid(criteria_failed, "4H")
                picked_row = legal_4h or blocked_4h
                if picked_row is not None:
                    return {
                        "apply": True,
                        "selected_bid": "4H",
                        "reason_codes": ["forced_major_rebid_long_hearts_commit_game"],
                        "reason": "COMMON_SENSE_HARD_OVERRIDE: opener must commit to 4H after forced major-game acceptance",
                        "evidence": {
                            **evidence,
                            "source": "legal" if legal_4h is not None else "criteria_fail",
                            "self_total_points": tp_self,
                            "heart_len": heart_len,
                        },
                    }
    except Exception:
        pass

    return {
        "apply": False,
        "selected_bid": None,
        "reason_codes": [],
        "reason": None,
        "evidence": evidence,
    }


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
        mean_f = float(mean) if isinstance(mean, (int, float)) else None
        tail_f = float(tail_risk) if isinstance(tail_risk, (int, float)) else 0.0
        entropy_f = float(entropy_full) if isinstance(entropy_full, (int, float)) else 0.0
        if mean_f is not None:
            utility = mean_f - lam * tail_f - mu * entropy_f
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
        parts.append(f"mean ParScore delta={_fmt(dm)}")
    if dpok is not None:
        parts.append(f"P(ParScore>=0) delta={_fmt_pct(dpok, signed=True)}")
    if dtail is not None:
        # Positive dtail means bid has *higher* tail risk; that's worse.
        parts.append(f"tail-risk delta={_fmt(dtail)} (lower is better)")
    if dent is not None:
        parts.append(f"ambiguity(entropy) delta={_fmt(dent)} (lower is better)")

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

