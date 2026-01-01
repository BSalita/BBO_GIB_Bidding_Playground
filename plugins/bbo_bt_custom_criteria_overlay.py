from __future__ import annotations

from typing import Any

from bbo_bidding_queries_lib import pattern_matches


def normalize_auction_for_overlay(auction: str) -> str:
    """Normalize auction for overlay prefix matching: UPPERCASE and strip leading passes.
    
    This uses the canonical uppercase form for consistency across the codebase.
    """
    a = (auction or "").strip().upper()
    # Strip leading 'P-' prefixes (seat-1 view matching)
    while a.startswith("P-"):
        a = a[2:]
    return a


def apply_custom_criteria_overlay_to_bt_row(
    bt_row: dict[str, Any],
    overlay: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Apply overlay criteria rules to a BT row dict in-place (returns the dict).

    Overlay rule format: {"partial": str, "seat": int, "criteria": list[str]}

    This helper is intentionally lightweight and safe:
    - If overlay is empty, returns the row unchanged.
    - If Auction is missing, returns unchanged.
    - De-dupes criteria while preserving order.
    """
    if not overlay:
        return bt_row

    auction = bt_row.get("Auction")
    if not auction:
        return bt_row

    auction_norm = normalize_auction_for_overlay(str(auction))
    if not auction_norm:
        return bt_row

    for rule in overlay:
        partial = str(rule.get("partial") or "")
        if not partial:
            continue
        # Hybrid matching: literal prefix for simple patterns, regex for complex ones
        if not pattern_matches(partial, auction_norm):
            continue

        try:
            seat = int(rule.get("seat") or 0)
        except Exception:
            continue
        if seat < 1 or seat > 4:
            continue

        crit_to_add = rule.get("criteria") or []
        if not crit_to_add:
            continue

        col = f"Agg_Expr_Seat_{seat}"
        existing = bt_row.get(col) or []
        combined = list(existing)
        for c in crit_to_add:
            if c not in combined:
                combined.append(c)
        bt_row[col] = combined

    return bt_row

