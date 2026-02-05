#!/usr/bin/env python3
"""
Best Auctions Search (API-backed)
--------------------------------

This is a lightweight CLI client for the FastAPI server (`bbo_bidding_queries_api.py`).

Purpose:
- Exercise the server-side "best auctions lookahead" DFS without loading local parquet files.
- Make it easy to iterate on API contracts and "best bids" logic with minimal noise.

Typical usage:
    python bbo_best_auctions_search_api.py --deal-index 12345 --metric DD
    python bbo_best_auctions_search_api.py --deal-index 12345 --metric EV --auction-prefix "1N-P-3N"
    python bbo_best_auctions_search_api.py --deal-index 12345 --deadline-s 1000 --max-depth 20 --max-results 25

Notes:
- Requires the API server running locally (default: http://127.0.0.1:8000).
- Uses the async job endpoints to avoid client timeouts:
    POST /best-auctions-lookahead/start
    GET  /best-auctions-lookahead/status/{job_id}
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import requests


DEFAULT_API_BASE = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class SearchRequest:
    deal_index: int
    auction_prefix: str
    metric: Literal["DD", "EV"]
    max_depth: int
    max_results: int
    deadline_s: float
    beam_width: int
    max_nodes: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_elapsed_s(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60.0:.1f}m"


def api_get(api_base: str, path: str, *, timeout_s: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = requests.get(f"{api_base}{path}", timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"{e}\nServer detail: {detail}", response=resp) from e
    data = resp.json()
    if isinstance(data, dict):
        data["_client_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return data


def api_post(api_base: str, path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = requests.post(f"{api_base}{path}", json=payload, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"{e}\nServer detail: {detail}", response=resp) from e
    data = resp.json()
    if isinstance(data, dict):
        data["_client_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return data


def resolve_deal_row_idx(api_base: str, *, deal_index: int, timeout_s: float) -> tuple[int, dict[str, Any]]:
    """Resolve user-facing deal `index` to server-side `_row_idx` (fast-path if monotonic)."""
    payload = {
        "indices": [int(deal_index)],
        "max_rows": 1,
        # Request minimal columns; server always includes _row_idx in output.
        "columns": ["index", "Dealer", "Vul", "ParScore", "Par_Score"],
    }
    data = api_post(api_base, "/deals-by-index", payload, timeout_s=timeout_s)
    rows = data.get("rows") or []
    if not rows:
        raise ValueError(f"Deal index {deal_index} not found (rows empty).")
    row0 = dict(rows[0])
    if "_row_idx" not in row0:
        raise ValueError("API /deals-by-index did not return _row_idx (unexpected).")
    return int(row0["_row_idx"]), row0


def start_best_auctions_job(api_base: str, req: SearchRequest, *, deal_row_idx: int, timeout_s: float) -> str:
    payload = {
        "deal_row_idx": int(deal_row_idx),
        "auction_prefix": str(req.auction_prefix or ""),
        "metric": str(req.metric),
        "max_depth": int(req.max_depth),
        "max_results": int(req.max_results),
        "deadline_s": float(req.deadline_s),
        "max_nodes": int(req.max_nodes),
        "beam_width": int(req.beam_width),
    }
    data = api_post(api_base, "/best-auctions-lookahead/start", payload, timeout_s=timeout_s)
    job_id = data.get("job_id")
    if not job_id:
        raise ValueError(f"Missing job_id in response: {data}")
    return str(job_id)


def poll_best_auctions_job(
    api_base: str,
    job_id: str,
    *,
    poll_interval_s: float,
    overall_timeout_s: float,
    status_timeout_s: float,
) -> dict[str, Any]:
    """Poll until completed/failed or overall timeout."""
    t0 = time.perf_counter()
    last_print = 0.0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > overall_timeout_s:
            raise TimeoutError(
                f"Timed out waiting for job {job_id} after {_format_elapsed_s(elapsed)} "
                f"(overall_timeout_s={overall_timeout_s})."
            )

        status = api_get(api_base, f"/best-auctions-lookahead/status/{job_id}", timeout_s=status_timeout_s)
        st = str(status.get("status") or "")

        # Progress indicator for long waits (>30s), but also print early once.
        if elapsed - last_print >= 5.0 or last_print == 0.0:
            last_print = elapsed
            print(f"[poll] status={st:<9} elapsed={_format_elapsed_s(elapsed)}")

        if st in ("completed", "failed"):
            return status

        time.sleep(max(0.1, float(poll_interval_s)))


def _print_results(result: dict[str, Any]) -> None:
    auctions = result.get("auctions") or []
    metric = str(result.get("metric") or "")
    par_score = result.get("par_score")
    elapsed_ms = result.get("elapsed_ms")

    print()
    print(f"[result] metric={metric} par_score={par_score} elapsed_ms={elapsed_ms}")
    if not auctions:
        print("No auctions returned.")
        return

    # Simple fixed-width table
    header = f"{'score':>8}  {'DD':>6}  {'EV':>8}  {'Par':>3}  {'Contract':<10}  Auction"
    print(header)
    print("-" * len(header))
    for r in auctions:
        auc = str(r.get("auction") or "")
        contract = str(r.get("contract") or "")
        dd = r.get("dd_score")
        ev = r.get("ev")
        is_par = bool(r.get("is_par"))
        score = dd if metric == "DD" else ev
        score_s = f"{score:>8}" if score is not None else f"{'—':>8}"
        dd_s = f"{dd:>6}" if dd is not None else f"{'—':>6}"
        ev_s = f"{ev:>8}" if ev is not None else f"{'—':>8}"
        par_s = "Y" if is_par else ""
        print(f"{score_s}  {dd_s}  {ev_s}  {par_s:>3}  {contract:<10}  {auc}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="API-backed best auction search")
    p.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    p.add_argument("--deal-index", type=int, required=True, help="User-facing deal index (deal_df['index']).")
    p.add_argument("--auction-prefix", type=str, default="", help="Auction prefix (e.g. '1N-P-3N').")
    p.add_argument("--metric", choices=["DD", "EV"], default="DD")
    p.add_argument("--max-depth", type=int, default=20)
    p.add_argument("--max-results", type=int, default=25)
    p.add_argument("--deadline-s", type=float, default=1000.0, help="Server search time budget (seconds).")
    p.add_argument("--beam-width", type=int, default=50)
    p.add_argument("--max-nodes", type=int, default=200000)
    p.add_argument("--poll-interval-s", type=float, default=1.0)
    p.add_argument("--overall-timeout-s", type=float, default=1200.0)
    p.add_argument("--http-timeout-s", type=float, default=10.0, help="Per-request HTTP timeout.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    start_dt = _utc_now()
    args = parse_args(argv)

    req = SearchRequest(
        deal_index=int(args.deal_index),
        auction_prefix=str(args.auction_prefix or ""),
        metric=str(args.metric),  # type: ignore[arg-type]
        max_depth=int(args.max_depth),
        max_results=int(args.max_results),
        deadline_s=float(args.deadline_s),
        beam_width=int(args.beam_width),
        max_nodes=int(args.max_nodes),
    )

    print(f"[{start_dt.isoformat()}] Starting API best-auctions search...")
    print(f"  api_base      = {args.api_base}")
    print(f"  deal_index    = {req.deal_index}")
    print(f"  auction_prefix= {req.auction_prefix!r}")
    print(f"  metric        = {req.metric}")
    print(f"  max_depth     = {req.max_depth}")
    print(f"  max_results   = {req.max_results}")
    print(f"  deadline_s    = {req.deadline_s}")
    print(f"  beam_width    = {req.beam_width}")
    print(f"  max_nodes     = {req.max_nodes}")

    # Resolve deal index -> row idx
    deal_row_idx, deal_min = resolve_deal_row_idx(args.api_base, deal_index=req.deal_index, timeout_s=args.http_timeout_s)
    print(f"[deal] resolved _row_idx={deal_row_idx} Dealer={deal_min.get('Dealer')} Vul={deal_min.get('Vul')}")

    # Start async job (fast)
    job_id = start_best_auctions_job(args.api_base, req, deal_row_idx=deal_row_idx, timeout_s=args.http_timeout_s)
    print(f"[start] job_id={job_id}")

    # Poll status/results
    status = poll_best_auctions_job(
        args.api_base,
        job_id,
        poll_interval_s=float(args.poll_interval_s),
        overall_timeout_s=float(args.overall_timeout_s),
        status_timeout_s=args.http_timeout_s,
    )
    if str(status.get("status")) == "failed":
        print(f"[failed] error={status.get('error')}")
        return 2

    result = status.get("result") or {}
    _print_results(result)

    end_dt = _utc_now()
    total_s = (end_dt - start_dt).total_seconds()
    print()
    print(f"[{end_dt.isoformat()}] Done. total_elapsed={_format_elapsed_s(total_s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

