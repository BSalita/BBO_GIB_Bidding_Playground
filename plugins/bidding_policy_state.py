from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


SUITS = ("S", "H", "D", "C")
HONOR_HCP = {"A": 4, "K": 3, "Q": 2, "J": 1}


def parse_hand_suit_lengths(hand_pbn: str | None) -> dict[str, int]:
    hand = str(hand_pbn or "").strip()
    if not hand or "." not in hand:
        return {}
    parts = hand.split(".")
    if len(parts) != 4:
        return {}
    return {
        "S": len(parts[0]),
        "H": len(parts[1]),
        "D": len(parts[2]),
        "C": len(parts[3]),
    }


def parse_hand_suit_hcp(hand_pbn: str | None) -> dict[str, int]:
    hand = str(hand_pbn or "").strip()
    if not hand or "." not in hand:
        return {}
    parts = hand.split(".")
    if len(parts) != 4:
        return {}
    return {
        "S": sum(HONOR_HCP.get(card.upper(), 0) for card in parts[0]),
        "H": sum(HONOR_HCP.get(card.upper(), 0) for card in parts[1]),
        "D": sum(HONOR_HCP.get(card.upper(), 0) for card in parts[2]),
        "C": sum(HONOR_HCP.get(card.upper(), 0) for card in parts[3]),
    }


@dataclass(frozen=True)
class ActorVisibleHand:
    direction: str
    hand_pbn: str
    hcp: Optional[float]
    total_points: Optional[float]
    suit_lengths: Mapping[str, int] = field(default_factory=dict)
    suit_hcp: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BidderViewState:
    auction_tokens: tuple[str, ...]
    dealer_actual: str
    board_vul: Any
    acting_direction: str
    next_seat: int
    visible_hand: ActorVisibleHand
    convention_state: Mapping[str, Any] = field(default_factory=dict)
    belief_summary: Mapping[str, Any] = field(default_factory=dict)


def build_actor_visible_hand(
    *,
    deal_row: Mapping[str, Any],
    acting_direction: str,
) -> ActorVisibleHand:
    acting_dir = str(acting_direction or "").upper()
    hand_pbn = str(deal_row.get(f"Hand_{acting_dir}", "") or "").strip()
    hcp_val = deal_row.get(f"HCP_{acting_dir}")
    tp_val = deal_row.get(f"Total_Points_{acting_dir}")
    try:
        hcp = float(hcp_val) if hcp_val is not None else None
    except Exception:
        hcp = None
    try:
        total_points = float(tp_val) if tp_val is not None else None
    except Exception:
        total_points = None
    suit_lengths = parse_hand_suit_lengths(hand_pbn)
    suit_hcp = parse_hand_suit_hcp(hand_pbn)
    return ActorVisibleHand(
        direction=acting_dir,
        hand_pbn=hand_pbn,
        hcp=hcp,
        total_points=total_points,
        suit_lengths=suit_lengths,
        suit_hcp=suit_hcp,
    )


def build_bidder_view_state(
    *,
    auction_tokens: Sequence[str],
    dealer_actual: str,
    board_vul: Any,
    acting_direction: str,
    next_seat: int,
    deal_row: Mapping[str, Any],
    convention_state: Optional[Mapping[str, Any]] = None,
    belief_summary: Optional[Mapping[str, Any]] = None,
) -> BidderViewState:
    return BidderViewState(
        auction_tokens=tuple(str(t or "").strip().upper() for t in list(auction_tokens or [])),
        dealer_actual=str(dealer_actual or "N").upper(),
        board_vul=board_vul,
        acting_direction=str(acting_direction or "").upper(),
        next_seat=int(next_seat),
        visible_hand=build_actor_visible_hand(deal_row=deal_row, acting_direction=acting_direction),
        convention_state=dict(convention_state or {}),
        belief_summary=dict(belief_summary or {}),
    )


def has_stopper_in_suit(view_state: BidderViewState, suit: str) -> bool:
    suit_u = str(suit or "").upper()
    if suit_u not in SUITS:
        return False
    sl = int(view_state.visible_hand.suit_lengths.get(suit_u, 0) or 0)
    suit_hcp = int(view_state.visible_hand.suit_hcp.get(suit_u, 0) or 0)
    return suit_hcp >= 4 or (sl + suit_hcp) >= 6
