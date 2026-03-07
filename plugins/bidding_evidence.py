from __future__ import annotations

from typing import Any, Dict, Tuple


def split_bid_details_for_live_policy(details: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = dict(details or {})
    debug_annotations: Dict[str, Any] = {}

    live_details = dict(base)
    range_percentiles = live_details.pop("range_percentiles", None)
    if range_percentiles is not None:
        debug_annotations["range_percentiles"] = range_percentiles

    phase2a = live_details.get("phase2a")
    if isinstance(phase2a, dict):
        phase2a_live = dict(phase2a)
        phase2a_range_percentiles = phase2a_live.pop("range_percentiles", None)
        if phase2a_range_percentiles is not None:
            debug_annotations["phase2a_range_percentiles"] = phase2a_range_percentiles
        live_details["phase2a"] = phase2a_live

    return live_details, debug_annotations


def attach_bid_details_views(details: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(details or {})
    live_details, debug_annotations = split_bid_details_for_live_policy(out)
    out["live_policy_evidence"] = live_details
    out["debug_annotations"] = debug_annotations
    return out


def get_live_policy_details(details: Dict[str, Any]) -> Dict[str, Any]:
    live_details = (details or {}).get("live_policy_evidence")
    if isinstance(live_details, dict):
        return live_details
    attached = attach_bid_details_views(details or {})
    return dict(attached.get("live_policy_evidence") or {})
