from __future__ import annotations

from typing import Any

from bbo_bidding_queries_lib import (
    compute_par_score as _compute_par_score,
    get_dd_score_for_auction as _get_dd_score_for_auction,
    get_ev_for_auction as _get_ev_for_auction,
    get_ev_for_auction_pre as _get_ev_for_auction_pre,
)


def compute_par_score(pbn: str, dealer: str, vul: str) -> Any:
    return _compute_par_score(pbn, dealer, vul)


def get_dd_score_for_auction(auction: str, dealer: str, deal_row: Any) -> Any:
    return _get_dd_score_for_auction(auction, dealer, deal_row)


def get_ev_for_auction(auction: str, dealer: str, deal_row: Any) -> Any:
    return _get_ev_for_auction(auction, dealer, deal_row)


def get_ev_for_auction_pre(*args: Any, **kwargs: Any) -> Any:
    return _get_ev_for_auction_pre(*args, **kwargs)
