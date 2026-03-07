from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

import bbo_bidding_queries_api as legacy_api
from bbo_api_surface_builder import LIVE_PATHS, attach_hot_reload_info, build_surface_app, prepare_handler_call
from plugins.bidding_evidence import get_live_policy_details


app = build_surface_app(
    title="BBO Live Bidding API",
    allowed_paths=LIVE_PATHS,
)


@app.post("/bid-details")
def bid_details(req: legacy_api.BidDetailsRequest) -> Dict[str, Any]:
    """Live-safe selected-bid details for scoring and bidder-visible UI."""
    state, reload_info, handler_module = prepare_handler_call()
    try:
        resp = handler_module.handle_bid_details(
            state=state,
            auction=req.auction,
            bid=req.bid,
            max_deals=req.max_deals,
            seed=req.seed,
            vul_filter=req.vul_filter,
            deal_index=req.deal_index,
            topk=req.topk,
            include_phase2a=req.include_phase2a,
        )
        live_resp = get_live_policy_details(resp)
        return attach_hot_reload_info(live_resp, reload_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        legacy_api._log_and_raise("live bid-details", e)
