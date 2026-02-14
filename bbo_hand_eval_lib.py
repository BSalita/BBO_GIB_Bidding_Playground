"""Hand evaluation heuristics for trick estimation and quick loser counting.

Pure-function library — no external dependencies beyond the standard library
(except pivot_bt_seat_stats which requires Polars).
These models use the pinned deal's exact hand plus Phase2a partner posteriors
to estimate partnership playing strength per strain.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Hand parsing
# ---------------------------------------------------------------------------

_HONOR_RANK = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}
_HCP_VALUES = {"A": 4, "K": 3, "Q": 2, "J": 1}
_SUITS = ("S", "H", "D", "C")


def parse_hand_cards(hand_str: str) -> dict[str, list[str]]:
    """Parse a dot-notation hand string into per-suit card lists.

    Example:
        "AKQ2.J98.T75.643" -> {"S": ["A","K","Q","2"], "H": ["J","9","8"],
                                "D": ["T","7","5"], "C": ["6","4","3"]}

    Cards within each suit are sorted high-to-low by rank.
    """
    parts = str(hand_str or "").strip().split(".")
    if len(parts) != 4:
        return {}
    out: dict[str, list[str]] = {}
    for suit, cards_str in zip(_SUITS, parts):
        cards = [c.upper() for c in cards_str if c.strip()]
        # Sort high-to-low
        cards.sort(key=lambda c: _HONOR_RANK.get(c, int(c) if c.isdigit() else 0), reverse=True)
        out[suit] = cards
    return out


# ---------------------------------------------------------------------------
# Quick Tricks (per suit)
# ---------------------------------------------------------------------------

def _suit_quick_tricks(cards: list[str]) -> float:
    """Count quick tricks in a single suit.

    AK = 2, AQ = 1.5, A = 1, KQ = 1, Kx(+) = 0.5
    """
    n = len(cards)
    if n == 0:
        return 0.0

    has_a = "A" in cards
    has_k = "K" in cards
    has_q = "Q" in cards

    if has_a and has_k:
        return 2.0
    if has_a and has_q and n >= 2:
        return 1.5
    if has_a:
        return 1.0
    if has_k and has_q and n >= 2:
        return 1.0
    if has_k and n >= 2:
        return 0.5
    return 0.0


def count_quick_tricks(cards: dict[str, list[str]]) -> tuple[float, dict[str, float]]:
    """Count quick tricks for an entire hand.

    Returns (total, per_suit_breakdown).
    """
    per_suit: dict[str, float] = {}
    total = 0.0
    for suit in _SUITS:
        qt = _suit_quick_tricks(cards.get(suit, []))
        per_suit[suit] = qt
        total += qt
    return total, per_suit


# ---------------------------------------------------------------------------
# Losing Trick Count (LTC)
# ---------------------------------------------------------------------------

def count_suit_ltc(suit_cards: list[str]) -> int:
    """Losing Trick Count for a single suit (0-3 losers).

    Rules:
    - Void: 0 losers
    - Singleton: 0 if A, else 1
    - Doubleton: count missing A, K (max 2)
    - 3+ cards: count missing A, K, Q (max 3)
    """
    n = len(suit_cards)
    if n == 0:
        return 0
    if n == 1:
        return 0 if "A" in suit_cards else 1
    if n == 2:
        losers = 0
        if "A" not in suit_cards:
            losers += 1
        if "K" not in suit_cards:
            losers += 1
        return losers
    # 3+ cards
    losers = 0
    if "A" not in suit_cards:
        losers += 1
    if "K" not in suit_cards:
        losers += 1
    if "Q" not in suit_cards:
        losers += 1
    return losers


def count_hand_ltc(cards: dict[str, list[str]]) -> tuple[int, dict[str, int]]:
    """Losing Trick Count for an entire hand.

    Returns (total_ltc, per_suit_breakdown).
    Standard LTC range: 0-12 (max 3 per suit).
    """
    per_suit: dict[str, int] = {}
    total = 0
    for suit in _SUITS:
        ltc = count_suit_ltc(cards.get(suit, []))
        per_suit[suit] = ltc
        total += ltc
    return total, per_suit


# ---------------------------------------------------------------------------
# Quick Losers (opponents can cash immediately)
# ---------------------------------------------------------------------------

def _suit_quick_losers_trump(
    suit_cards: list[str],
    is_trump: bool,
    partner_expected_length: float | None,
) -> float:
    """Estimate quick losers in a single suit for a trump contract.

    Quick losers = tricks opponents can cash off the top before declarer
    gains control.

    In the trump suit: 0 quick losers (declarer controls timing).

    In side suits:
    - Void: 0
    - Singleton: 1 if missing A, else 0
    - Doubleton: count missing A, K (0-2)
    - 3+ cards: count missing A, K (0-2) — opponents can cash these immediately

    Partner adjustment:
    - If partner is short (expected length <= 1) and we have 2 quick losers,
      partner can ruff the second round -> reduce by 1.
    """
    if is_trump:
        return 0.0

    n = len(suit_cards)
    if n == 0:
        return 0.0

    has_a = "A" in suit_cards
    has_k = "K" in suit_cards

    if n == 1:
        losers = 0.0 if has_a else 1.0
    elif n == 2:
        losers = 0.0
        if not has_a:
            losers += 1.0
        if not has_k:
            losers += 1.0
    else:
        # 3+ cards: opponents can cash A and/or K if we don't have them
        losers = 0.0
        if not has_a:
            losers += 1.0
        if not has_k:
            losers += 1.0

    # Partner ruffing adjustment
    if partner_expected_length is not None and losers >= 2.0:
        if partner_expected_length <= 1.0:
            losers -= 1.0  # Partner can ruff the second round

    return losers


def _suit_quick_losers_nt(
    suit_cards: list[str],
    combined_expected_length: float | None,
) -> float:
    """Estimate quick losers in a single suit for a NT contract.

    In NT, opponents can run a suit if they have enough length and we lack
    a stopper.  Quick losers are higher in suits where we're short and
    missing the A.

    Heuristic:
    - Combined length < 7 and missing A: 2 quick losers (opponents have
      length + top card)
    - Combined length < 7 and have A but missing K: 1 quick loser
    - Combined length >= 7 or have AK: 0 quick losers (suit is controlled)
    """
    n = len(suit_cards)
    has_a = "A" in suit_cards
    has_k = "K" in suit_cards

    short_combined = (
        combined_expected_length is not None and combined_expected_length < 7.0
    )

    if short_combined or (combined_expected_length is None and n <= 3):
        if not has_a:
            return 2.0
        if not has_k:
            return 1.0
    return 0.0


def estimate_quick_losers(
    self_hand: str,
    partner_sl_hists: dict[str, dict[str, Any]] | None,
    fit_us_hists: dict[str, dict[str, Any]] | None,
    strain: str,
) -> dict[str, Any]:
    """Estimate quick losers for the partnership.

    Parameters
    ----------
    self_hand : str
        Dot-notation hand string for the acting player.
    partner_sl_hists : dict, optional
        Partner suit-length histograms from Phase2a
        (``phase2a.roles.partner.sl_hist``).
    fit_us_hists : dict, optional
        Partnership combined fit histograms (``phase2a.fit.us``).
    strain : str
        "C", "D", "H", "S", or "NT".

    Returns
    -------
    dict with keys: quick_losers (float), per_suit (dict), notes (list).
    """
    cards = parse_hand_cards(self_hand)
    if not cards:
        return {"quick_losers": None, "per_suit": {}, "notes": ["unparseable hand"]}

    strain_u = str(strain or "").strip().upper()
    if strain_u in ("N", "NOTRUMP", "NOTRUMPS"):
        strain_u = "NT"

    is_nt = strain_u == "NT"
    trump_suit: str | None = strain_u if strain_u in _SUITS else None

    per_suit: dict[str, float] = {}
    notes: list[str] = []
    total = 0.0

    for suit in _SUITS:
        suit_cards = cards.get(suit, [])

        # Get partner expected length
        partner_len: float | None = None
        if partner_sl_hists and suit in partner_sl_hists:
            partner_len = _expected_from_hist(partner_sl_hists[suit])

        if is_nt:
            # Combined expected length for NT evaluation
            combined_len: float | None = None
            if fit_us_hists and suit in fit_us_hists:
                combined_len = _expected_from_hist(fit_us_hists[suit])
            elif partner_len is not None:
                combined_len = float(len(suit_cards)) + partner_len
            ql = _suit_quick_losers_nt(suit_cards, combined_len)
        else:
            is_trump = (suit == trump_suit)
            ql = _suit_quick_losers_trump(suit_cards, is_trump, partner_len)

        per_suit[suit] = ql
        total += ql

    return {
        "quick_losers": round(total, 1),
        "per_suit": per_suit,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Estimated Partnership Trick Count
# ---------------------------------------------------------------------------

def _expected_from_hist(hist: dict[str, Any] | None) -> float | None:
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


def _self_length_tricks(
    cards: dict[str, list[str]],
    trump_suit: str | None,
) -> tuple[float, dict[str, float]]:
    """Estimate length tricks from self's hand.

    In the trump suit: cards beyond length 3 contribute 0.5 tricks each
    (need entries and timing).
    In side suits: cards beyond length 4 contribute 0.5 tricks each
    (need to establish and cash).
    """
    per_suit: dict[str, float] = {}
    total = 0.0
    for suit in _SUITS:
        n = len(cards.get(suit, []))
        if suit == trump_suit:
            extra = max(0, n - 3)
        else:
            extra = max(0, n - 4)
        lt = float(extra) * 0.5
        per_suit[suit] = lt
        total += lt
    return total, per_suit


def _partner_length_tricks(
    partner_sl_hists: dict[str, dict[str, Any]] | None,
    trump_suit: str | None,
) -> float:
    """Estimate partner's length tricks from posterior suit-length distributions.

    Side suits only: max(0, E[length] - 4) * 0.4 (discounted — uncertain).
    """
    if not partner_sl_hists:
        return 0.0
    total = 0.0
    for suit in _SUITS:
        if suit == trump_suit:
            continue  # Trump length handled via fit bonus
        hist = partner_sl_hists.get(suit)
        e_len = _expected_from_hist(hist)
        if e_len is not None:
            total += max(0.0, e_len - 4.0) * 0.4
    return total


def estimate_partnership_tricks(
    self_hand: str,
    partner_hcp_hist: dict[str, Any] | None,
    partner_sl_hists: dict[str, dict[str, Any]] | None,
    fit_us_hists: dict[str, dict[str, Any]] | None,
    strain: str,
) -> dict[str, Any]:
    """Estimate how many tricks the partnership can take in a given strain.

    Parameters
    ----------
    self_hand : str
        Dot-notation hand string for the acting player.
    partner_hcp_hist : dict, optional
        Partner HCP histogram from Phase2a (``phase2a.roles.partner.hcp_hist``).
    partner_sl_hists : dict, optional
        Partner suit-length histograms (``phase2a.roles.partner.sl_hist``).
    fit_us_hists : dict, optional
        Partnership combined fit histograms (``phase2a.fit.us``).
    strain : str
        "C", "D", "H", "S", or "NT".

    Returns
    -------
    dict with keys:
        est_tricks (float | None), self_qt (float), partner_qt (float),
        self_length (float), partner_length (float), fit_bonus (float),
        self_ltc (int), breakdown (dict).
    """
    cards = parse_hand_cards(self_hand)
    if not cards:
        return {
            "est_tricks": None,
            "self_qt": None,
            "partner_qt": None,
            "self_length": None,
            "partner_length": None,
            "fit_bonus": None,
            "self_ltc": None,
            "breakdown": {},
        }

    strain_u = str(strain or "").strip().upper()
    if strain_u in ("N", "NOTRUMP", "NOTRUMPS"):
        strain_u = "NT"
    trump_suit: str | None = strain_u if strain_u in _SUITS else None

    # Self quick tricks
    self_qt, qt_breakdown = count_quick_tricks(cards)

    # Self length tricks
    self_lt, lt_breakdown = _self_length_tricks(cards, trump_suit)

    # Self LTC
    self_ltc, ltc_breakdown = count_hand_ltc(cards)

    # Partner quick tricks (estimated from HCP)
    partner_qt = 0.0
    partner_hcp_e = _expected_from_hist(partner_hcp_hist)
    if partner_hcp_e is not None:
        partner_qt = partner_hcp_e / 3.5  # Empirical: ~1 QT per 3.5 HCP

    # Partner length tricks
    partner_lt = _partner_length_tricks(partner_sl_hists, trump_suit)

    # Fit bonus (trump contracts only)
    fit_bonus = 0.0
    if trump_suit is not None and fit_us_hists:
        fit_hist = fit_us_hists.get(trump_suit)
        e_fit = _expected_from_hist(fit_hist)
        if e_fit is not None:
            fit_bonus = max(0.0, e_fit - 7.0) * 0.5

    # Combined estimate
    est = self_qt + self_lt + partner_qt + partner_lt + fit_bonus
    est = max(0.0, min(13.0, est))

    return {
        "est_tricks": round(est, 1),
        "self_qt": round(self_qt, 1),
        "partner_qt": round(partner_qt, 1),
        "self_length": round(self_lt, 1),
        "partner_length": round(partner_lt, 1),
        "fit_bonus": round(fit_bonus, 1),
        "self_ltc": self_ltc,
        "breakdown": {
            "qt_per_suit": qt_breakdown,
            "lt_per_suit": lt_breakdown,
            "ltc_per_suit": ltc_breakdown,
        },
    }


# ---------------------------------------------------------------------------
# Pivot precomputed bt_stats_df seat-suffixed columns into rows
# ---------------------------------------------------------------------------

_DIRECTIONS = ["N", "E", "S", "W"]

# Hand-stat columns to include (base name -> display name)
_HAND_STAT_COLS: list[tuple[str, str, str]] = [
    # (base_prefix, agg_suffix, display_name)
    ("HCP", "mean", "HCP"),
    ("HCP", "std", "HCP_std"),
    ("SL_C", "mean", "SL_C"),
    ("SL_D", "mean", "SL_D"),
    ("SL_H", "mean", "SL_H"),
    ("SL_S", "mean", "SL_S"),
    ("Total_Points", "mean", "Total_Points"),
    ("Total_Points", "std", "TP_std"),
]

# DD trick-mean columns: declarer x strain
# v3 columns: DD_S{seat}_{strain}_mean_S{seat} (seat-relative)
# v2 fallback: DD_{compass}_{strain}_mean_S1 (absolute, diluted)
_DD_STRAINS = ["C", "D", "H", "S", "N"]
_DD_FRIENDLY_STRAIN = {"C": "C", "D": "D", "H": "H", "S": "S", "N": "NT"}


def _seat_to_direction(seat_1based: int, dealer: str | None) -> str:
    """Map seat number (1-4) to compass direction given dealer.

    Seat 1 = dealer, seat 2 = next clockwise, etc.
    """
    if dealer is None:
        return f"S{seat_1based}"
    try:
        dealer_idx = _DIRECTIONS.index(str(dealer).upper())
    except ValueError:
        return f"S{seat_1based}"
    return _DIRECTIONS[(dealer_idx + seat_1based - 1) % 4]


def pivot_bt_seat_stats(
    bt_stats_df: Any,  # pl.DataFrame
    bt_indices: list[int],
    dealer: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Pivot seat-suffixed columns from bt_stats_df into per-seat rows.

    Parameters
    ----------
    bt_stats_df : pl.DataFrame
        The precomputed stats DataFrame (974K rows x 205 cols) with columns
        like ``HCP_mean_S1``, ``DD_S1_C_mean_S1`` (v3) or ``DD_N_C_mean_S1`` (v2).
    bt_indices : list[int]
        bt_index values to look up.
    dealer : str, optional
        Compass direction of dealer (N/E/S/W). When provided, seat numbers
        are mapped to directions; otherwise labelled S1-S4.

    Returns
    -------
    dict keyed by bt_index, each value has:
        ``hand_stats`` – list of 4 dicts (one per seat/direction)
        ``dd_means``   – list of 4 dicts (one per seat/direction)
    Only bt_indices that exist in bt_stats_df are included.
    """
    import polars as pl

    if bt_stats_df is None or not bt_indices:
        return {}

    # Filter to requested bt_indices
    idx_series = pl.Series("bt_index", [int(x) for x in bt_indices], dtype=pl.Int64)
    matched = bt_stats_df.filter(pl.col("bt_index").is_in(idx_series))
    if matched.height == 0:
        return {}

    # Build direction-to-seat mapping so we can output rows in N, E, S, W order
    dir_to_seat: dict[str, int] = {}
    for s in range(1, 5):
        d = _seat_to_direction(s, dealer)
        dir_to_seat[d] = s

    result: dict[int, dict[str, Any]] = {}
    for row in matched.iter_rows(named=True):
        bt_idx = int(row["bt_index"])

        hand_rows: list[dict[str, Any]] = []
        dd_rows: list[dict[str, Any]] = []

        # Always iterate in canonical N, E, S, W order.
        # Hand stats are SEAT-RELATIVE: HCP_mean_S1 = mean HCP of the player
        # in seat 1 (whoever that is).  We map direction→seat so the value
        # lands on the correct compass row.
        for direction in _DIRECTIONS:
            seat = dir_to_seat.get(direction, 1)

            # -- Hand stats (seat-relative columns) --
            hs: dict[str, Any] = {"Direction": direction}
            for base, agg, friendly in _HAND_STAT_COLS:
                col_name = f"{base}_{agg}_S{seat}"
                val = row.get(col_name)
                if val is not None:
                    try:
                        val = round(float(val), 1)
                    except (TypeError, ValueError):
                        pass
                hs[friendly] = val
            hand_rows.append(hs)

        # -- DD trick means: 4 rows (declarer) × 5 cols (strain) --
        # v3 columns are seat-relative: DD_S{decl_seat}_{strain}_mean_S{pop_seat}.
        # For each compass direction, look up which seat that direction maps to,
        # then read DD_S{that_seat}_{strain}_mean_S{that_seat}.
        # Falls back to v2 absolute columns (DD_{compass}_{strain}_mean_S1) if v3 not found.
        for direction in _DIRECTIONS:
            decl_seat = dir_to_seat.get(direction, 1)
            dd: dict[str, Any] = {"Declarer": direction}
            for strain in _DD_STRAINS:
                # Try v3 seat-relative column first
                col_name = f"DD_S{decl_seat}_{strain}_mean_S{decl_seat}"
                val = row.get(col_name)
                if val is None:
                    # Fall back to v2 absolute column
                    col_name = f"DD_{direction}_{strain}_mean_S1"
                    val = row.get(col_name)
                if val is not None:
                    try:
                        val = round(float(val), 1)
                    except (TypeError, ValueError):
                        pass
                dd[strain] = val
            dd_rows.append(dd)

        result[bt_idx] = {
            "hand_stats": hand_rows,
            "dd_means": dd_rows,
        }

    return result
