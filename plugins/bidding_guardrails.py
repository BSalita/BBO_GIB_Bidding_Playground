from __future__ import annotations

from enum import Enum


class InformationClass(str, Enum):
    PUBLIC_AUCTION = "public_auction"
    SELF_HAND = "self_hand"
    POSTERIOR = "posterior"
    ORACLE = "oracle"
    DEBUG_ONLY = "debug_only"


def delayed_support_target_level(partner_strain: str, self_total_points: float | None) -> int:
    strain = str(partner_strain or "").upper()
    tp = float(self_total_points) if self_total_points is not None else 0.0
    is_major = strain in ("H", "S")
    if is_major:
        if tp >= 13:
            return 4
        if tp >= 10:
            return 3
        return 0
    if tp >= 15:
        return 5
    if tp >= 12:
        return 4
    if tp >= 10:
        return 3
    return 0


def self_only_level_cap(self_points: float | None, *, max_cap: int = 3) -> int:
    if self_points is None:
        return 0
    return max(0, min(int(max_cap), int((float(self_points) - 4.0) // 3.0)))
