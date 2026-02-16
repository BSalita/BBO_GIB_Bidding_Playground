from __future__ import annotations

import math
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


def compute_non_rebiddable_suit_rebid_penalty(
    *,
    bid_text: str,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    bt_acting_criteria: list[str] | None,
) -> tuple[float, str | None]:
    """Penalty for suit rebids lacking rebiddable criteria support."""
    s = str(bid_text or "").strip().upper()
    m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", s)
    if not m:
        return 0.0, None
    bid_level = int(m.group(1))
    bid_strain = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
    if bid_strain not in ("S", "H", "D", "C"):
        return 0.0, None
    if not acting_direction:
        return 0.0, None

    # Determine which direction bid each prior token from dealer + token index.
    directions = ["N", "E", "S", "W"]
    d = str(dealer_actual or "N").upper()
    dealer_idx = directions.index(d) if d in directions else 0

    def _token_bidder_dir(token_idx: int) -> str:
        return directions[(dealer_idx + int(token_idx)) % 4]

    # Has this same player already bid this same strain?
    prev_levels: list[int] = []
    for i, tk in enumerate(auction_tokens):
        if _token_bidder_dir(i) != acting_direction:
            continue
        p = re.match(r"^([1-7])\s*(NT|N|[CDHS])", str(tk or "").strip().upper())
        if not p:
            continue
        pl = int(p.group(1))
        ps = "N" if p.group(2).upper() in ("N", "NT") else p.group(2).upper()
        if ps == bid_strain:
            prev_levels.append(pl)
    if not prev_levels:
        return 0.0, None

    crit_norm = {str(c or "").strip().upper().strip("()") for c in (bt_acting_criteria or []) if str(c or "").strip()}
    has_reb = f"REBIDDABLE_{bid_strain}" in crit_norm
    has_twice = f"TWICE_REBIDDABLE_{bid_strain}" in crit_norm

    # 3-level+ rebids should generally be twice-rebiddable; lower rebids need rebiddable.
    needed = "TWICE_REBIDDABLE" if bid_level >= 3 else "REBIDDABLE"
    acceptable = has_twice if bid_level >= 3 else (has_reb or has_twice)
    if acceptable:
        return 0.0, None

    penalty = 170.0 if bid_level >= 3 else 120.0
    reason = (
        f"NON_REBIDDABLE_SUIT_REBID: rebid {s} in {bid_strain} "
        f"without required {needed}_{bid_strain} criteria support "
        f"(-{penalty:.0f})"
    )
    return float(penalty), reason


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

    # "Doubly rebiddable by history": partner bid the same major at least twice.
    partner_major_rebid_count = int(partner_major_counts.get(partner_last_major, 0))
    if partner_major_rebid_count < 2:
        return 0.0, 0.0, None

    crit_norm = {str(c or "").strip().upper().strip("()") for c in (bt_acting_criteria or []) if str(c or "").strip()}
    fit_context = (
        "SUPPORTSHOWING" in crit_norm
        or "FITESTABLISHED" in crit_norm
        or "RAISE" in crit_norm
    )

    # Prefer direct game in partner's repeated major.
    if bid_strain == partner_last_major and bid_level >= 4:
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
) -> tuple[float, str | None]:
    """Bonus for Pass when partner already committed to game."""
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
        return 0.0, None

    bonus = 80.0
    reason = f"PASS_SIGNOFF_BONUS: partner last bid {last_bid} (game+); reward closing action (+{bonus:.0f})"
    return bonus, reason


def compute_post_game_slam_gate_adjustment(
    *,
    bid_text: str,
    auction_tokens: list[str],
    acting_direction: str | None,
    dealer_actual: str | None,
    self_total_points: float | None,
    partner_tp_hist: dict[str, Any] | None,
) -> dict[str, Any]:
    """Penalty for slam-explore continuation without sufficient evidence."""
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
    if not game_contract_on_table:
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
    out["penalty"] = float(penalty)
    if penalty > 0:
        p_txt = f"{p_slam_tp_ge_33:.2f}" if p_slam_tp_ge_33 is not None else "NA"
        c_txt = f"{combined_tp_mean:.1f}" if combined_tp_mean is not None else "NA"
        out["reason"] = (
            f"POST_GAME_SLAM_GATE: slam explore {b} without enough evidence "
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
    w_first_round_control: float = 75.0,
    w_weak_hand: float = 40.0,
    debug_equivalence_bypass: bool = False,
) -> tuple[float, list[str]]:
    """Compute a non-negative guardrail penalty to subtract from the bid score.

    Ten checks plus a sacrifice exemption (each contributes additively):

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

    # ==================================================================
    # OVERBID CHECKS
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. OVERBID_VS_PAR – bid level exceeds par contract level
    # ------------------------------------------------------------------
    if top_rank is not None and bid_rank is not None and bid_rank > top_rank and not (
        nt_major_game_equiv and not par_is_opponents
    ):
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

    # ------------------------------------------------------------------
    # 7. INSUFFICIENT_FIRST_ROUND_CONTROLS
    #
    #    Universal bridge principle: max safe level = 3 + first-round
    #    controls.  Without enough controls opponents can cash aces to
    #    set the contract before declarer gains the lead.
    #
    #    First-round controls per suit:
    #      - An ace (either partner)
    #      - A void in a suit contract (ruff), provided it doesn't
    #        double-count with partner's ace in the same suit.
    #    In NT, only aces count (can't ruff).
    #
    #    Self controls are known.  Partner aces are estimated from the
    #    TP histogram (E[partner_aces] ≈ E[partner_HCP] / 10).
    #    Partner's aces live in suits where self has no ace, so we
    #    only count them toward UNCONTROLLED suits.
    #
    #    Examples:
    #      0 controls → max level 3  (opponents cash 4 aces)
    #      1 control  → max level 4  (opponents cash 3 aces)
    #      2 controls → max level 5  (opponents cash 2 aces)
    #      3 controls → max level 6  (opponents cash 1 ace)
    #      4 controls → max level 7  (no aces to cash)
    #
    #    Sacrifice-discounted: if par belongs to opponents, this is
    #    expected to go down and the penalty is reduced.
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

        # Estimate partner aces from TP histogram.
        # Heuristic: 4 aces = 16 HCP of 40 → ~1 ace per 10 HCP.
        # TP includes ~1 distribution point on average, so subtract 1.
        _est_partner_aces = 1.0  # default if no histogram
        if partner_tp_hist is not None:
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

        _total_controls = min(4.0, float(_self_controls) + _partner_new)
        _max_safe = 3.0 + _total_controls
        _excess = float(bid_level) - _max_safe

        if _excess > 0:
            raw_p = _excess * float(w_first_round_control)
            p = raw_p * sac_factor
            sac_note = f" [sacrifice-discounted from {raw_p:.0f}]" if sac_factor < 1.0 else ""
            _ctrl_parts: list[str] = []
            _ctrl_parts.append(f"self {self_aces} ace(s)")
            if _is_suit and _hv > 0:
                _ctrl_parts.append(f"{_hv} helpful void(s)")
            _ctrl_parts.append(f"est. partner ~{_est_partner_aces:.1f} aces")
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
    # UNDERBID CHECKS (only when par belongs to our side)
    # ==================================================================

    if not par_is_opponents and top_level is not None:
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

