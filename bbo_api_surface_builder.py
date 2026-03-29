from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute

import bbo_bidding_queries_api as legacy_api


LIVE_PATHS = {
    "/status",
    "/code-sync-status",
    "/ai-model-advanced-path/start",
    "/ai-model-advanced-path/status/{job_id}",
    "/belief-snapshot",
    "/belief-trace",
    "/list-next-bids",
}

ORACLE_PATHS = {
    "/status",
    "/code-sync-status",
    "/auction-dd-analysis",
    "/rank-bids-by-ev",
    "/contract-ev-deals",
    "/bid-details",
    "/explain-bid",
    "/best-auctions-lookahead",
    "/best-auctions-lookahead/start",
    "/best-auctions-lookahead/status/{job_id}",
    "/critical-mistake-analysis",
    "/greedy-model-path",
    "/bidding-arena",
}

SHARED_PATHS = {
    "/status",
    "/code-sync-status",
    "/init",
    "/new-rules-lookup",
    "/custom-criteria-info",
    "/custom-criteria-rules",
    "/custom-criteria-validate",
    "/custom-criteria-preview",
    "/custom-criteria-reload",
    "/custom-category-overrides-info",
    "/custom-category-overrides-reload",
    "/openings-by-deal-index",
    "/random-auction-sequences",
    "/auction-sequences-matching",
    "/auction-sequences-by-index",
    "/deal-criteria-eval-batch",
    "/deal-criteria-pass-batch",
    "/deals-by-index",
    "/bt-categories-by-index",
    "/resolve-auction-path",
    "/bt-dd-mean-tricks",
    "/find-bt-auctions-by-contracts",
    "/deals-matching-auction",
    "/sample-deals-by-auction-pattern",
    "/auction-pattern-counts",
    "/bidding-table-statistics",
    "/process-pbn",
    "/find-matching-auctions",
    "/pbn-sample",
    "/pbn-random",
    "/pbn-lookup",
    "/group-by-bid",
    "/bt-seat-stats",
    "/bt-seat-stats-pivot",
    "/sample-deals-for-bt-indices",
    "/criteria-stats-by-bt-indices",
    "/bt-trick-quality",
    "/bt-hand-profile",
    "/execute-sql",
    "/wrong-bid-stats",
    "/failed-criteria-summary",
    "/wrong-bid-leaderboard",
    "/bidding-models",
    "/deals-matching-next-bid-criteria",
    "/custom-criteria-impact",
    "/deal-matched-bt-sample",
}


def build_surface_app(
    *,
    title: str,
    allowed_paths: Iterable[str],
    excluded_paths: Iterable[str] | None = None,
) -> FastAPI:
    app = FastAPI(title=title, lifespan=legacy_api.lifespan)
    include_paths = set(allowed_paths)
    skip_paths = set(excluded_paths or [])

    for route in legacy_api.app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path not in include_paths or route.path in skip_paths:
            continue
        app.router.add_api_route(
            route.path,
            route.endpoint,
            methods=list(route.methods or []),
            response_model=route.response_model,
            status_code=route.status_code,
            tags=list(route.tags or []),
            dependencies=list(route.dependencies or []),
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=dict(route.responses or {}),
            deprecated=route.deprecated,
            operation_id=route.operation_id,
            response_class=route.response_class,
            name=route.name,
            callbacks=list(route.callbacks or []),
            openapi_extra=dict(route.openapi_extra or {}) if route.openapi_extra else None,
            generate_unique_id_function=route.generate_unique_id_function,
            include_in_schema=route.include_in_schema,
        )
    return app


def prepare_handler_call() -> tuple[dict[str, Any], dict[str, object], Any]:
    return legacy_api._prepare_handler_call()


def attach_hot_reload_info(resp: dict[str, Any], reload_info: dict[str, object]) -> dict[str, Any]:
    return legacy_api._attach_hot_reload_info(resp, reload_info)
