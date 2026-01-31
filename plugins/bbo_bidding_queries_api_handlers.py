"""
Hot-reloadable API handlers for BBO Bidding Queries.

This module contains all endpoint implementations. It is imported by
bbo_bidding_queries_api.py and reloaded (via importlib.reload) when
the file's mtime changes, allowing rapid iteration without restarting
the server or reloading data.

Handler functions receive a `state` dict containing loaded DataFrames
and precomputed structures.

Refactored: Common helpers and constants are in handlers_common.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import pathlib
import random
import re
import statistics
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import duckdb  # pyright: ignore[reportMissingImports]
import numpy as np
import polars as pl

# These imports are needed for the handler logic
from endplay.types import Deal, Vul, Player
from endplay.dds import calc_dd_table, par

from bbo_bidding_queries_lib import (
    calculate_imp,
    normalize_auction_pattern,
    normalize_auction_input,
    normalize_auction_user_text,
    get_ai_contract,
    get_declarer_for_auction,
    get_dd_score_for_auction,
    get_ev_for_auction,
    compute_hand_features,
    compute_par_score,
    parse_pbn_deal,
    parse_contract_from_auction,
    build_distribution_sql_for_bt,
    build_distribution_sql_for_deals,
    add_suit_length_columns,
)

from mlBridge.mlBridgeBiddingLib import DIRECTIONS
import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib

# Import from common module (eliminates code duplication)
from plugins.bbo_handlers_common import (
    # Constants
    SEAT_RANGE,
    MAX_SAMPLE_SIZE,
    DEFAULT_SEED,
    SUIT_IDX,
    DIRECTIONS_LIST,
    agg_expr_col,
    wrong_bid_col,
    invalid_criteria_col,
    # Model constants
    MODEL_RULES,
    MODEL_RULES_BASE,
    MODEL_RULES_LEARNED,
    MODEL_ACTUAL,
    MODELS_NO_OVERLAY,
    # Canonical casing (single source of truth)
    normalize_auction_case,
    # Elapsed time formatting
    format_elapsed,
    # Typed state
    HandlerState,
    # Auction helpers
    display_auction_with_seat_prefix as _display_auction_with_seat_prefix,
    normalize_to_seat1 as _normalize_to_seat1,
    expand_row_to_all_seats as _expand_row_to_all_seats,
    # Rule application (overlay only - merged rules pre-compiled in BT)
    apply_overlay_and_dedupe as _apply_overlay_and_dedupe,
    apply_all_rules_to_bt_row as _apply_all_rules_to_bt_row,  # backwards compat alias
    apply_rules_by_model as _apply_rules_by_model,
    apply_custom_criteria_overlay_to_bt_row as _apply_overlay_only,
    dedupe_criteria_all_seats,
    # Criteria deduplication
    dedupe_criteria_least_restrictive,
    # DataFrame helpers
    take_rows_by_index as _take_rows_by_index,
    effective_seed as _effective_seed,
    safe_float as _safe_float,
    # Bid/Contract helpers
    bid_value_to_str as _bid_value_to_str,
    count_leading_passes as _count_leading_passes,
    extract_bid_at_seat as _extract_bid_at_seat,
    seat_direction_map as _seat_direction_map,
    # Par contract helpers
    par_contract_signature as _par_contract_signature,
    dedup_par_contracts as _dedup_par_contracts,
    format_par_contracts as _format_par_contracts,
    ev_list_for_par_contracts as _ev_list_for_par_contracts,
    # BT Lookup (extracted from duplicate pattern)
    lookup_bt_row as _lookup_bt_row,
    get_bt_info_from_match as _get_bt_info_from_match,
    # Wrong bid checking
    check_deal_criteria_conformance_bitmap as _check_deal_criteria_conformance_bitmap,
    # Suit Length evaluation
    evaluate_sl_criterion,
    # Best Auction Selection
    choose_best_auction_match,
    # Vectorized join helpers (performance optimization)
    prepare_deals_with_bid_str,
    prepare_bt_for_join,
    join_deals_with_bt,
    batch_check_wrong_bids,
    join_deals_with_bt_on_demand,
    join_deals_with_bt_via_index,
    # On-demand Agg_Expr loading (DuckDB-based for efficiency)
    load_agg_expr_for_bt_indices as _load_agg_expr_for_bt_indices,
    # SL (Suit Length) evaluation helpers
    seat_to_direction,
    format_seat_notation as _format_seat_notation,
    hand_suit_length,
    parse_sl_comparison_relative,
    parse_sl_comparison_numeric,
    eval_comparison,
    annotate_criterion_with_value,
)


# ---------------------------------------------------------------------------
# V3 global cache (distinct from V1/V2)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tuning constants (avoid magic numbers scattered through handlers)
# ---------------------------------------------------------------------------
_CANDIDATE_POOL_LIMIT = 5000
_DEBUG_CANDIDATE_SAMPLE = 25
_DEBUG_CRITERIA_PREVIEW = 25
_RULES_CRITERIA_PREVIEW = 50
_NEED_RULES_MATCHES_LIST_THRESHOLD = 50


# ---------------------------------------------------------------------------
# BT Traversal Helpers (avoid full-table Auction lookups on 461M-row BT)
# ---------------------------------------------------------------------------

def _get_local_bid_vocab():
    """Consistent bid encoding for CSR traversal."""
    code_to_bid = [""]
    bid_to_code = {"": 0}
    def add(b):
        b = b.upper()
        if b not in bid_to_code:
            bid_to_code[b] = len(code_to_bid); code_to_bid.append(b)
    for b in ["P", "X", "XX", "D", "R"]: add(b)
    for l in range(1, 8):
        for s in ["C", "D", "H", "S", "N"]: add(f"{l}{s}")
    # Add lowercase mappings for robustness
    for b in ["p", "x", "xx", "d", "r"]:
        if b.upper() in bid_to_code:
            bid_to_code[b] = bid_to_code[b.upper()]
    return bid_to_code, code_to_bid

BID_TO_CODE, CODE_TO_BID = _get_local_bid_vocab()

_NEXT_BID_INDICES_CACHE: dict[int, list[int]] = {}
_NEXT_CHILDREN_MIN_CACHE: dict[int, dict[str, int]] = {}
_CACHE_MAX_PARENTS = 50_000


def _cache_put(d: dict, key: Any, val: Any, max_items: int) -> None:
    """Simple size-capped cache (insertion-order eviction)."""
    d[key] = val
    if len(d) > max_items:
        try:
            oldest = next(iter(d.keys()))
            d.pop(oldest, None)
        except Exception:
            d.clear()


def _bt_file_path_for_sql(state: Dict[str, Any]) -> str:
    bt_parquet_file = state.get("bt_seat1_file")
    if bt_parquet_file is None:
        raise ValueError("bt_seat1_file missing from state; cannot query BT")
    return str(bt_parquet_file).replace("\\", "/")


def _get_next_bid_indices_for_parent(state: Dict[str, Any], parent_bt_index: int) -> list[int]:
    """Fetch next_bid_indices for a BT node by bt_index, with caching."""
    p = int(parent_bt_index)
    
    # Try Gemini-3.2 CSR index first
    g3_index = state.get("g3_index")
    if g3_index:
        if p >= len(g3_index.offsets) - 1: return []
        start, end = g3_index.offsets[p], g3_index.offsets[p + 1]
        # children are uint32, need to convert back to list of int
        return [int(x) for x in g3_index.children[start:end]]

    # Fallback to cache/DuckDB
    cached = _NEXT_BID_INDICES_CACHE.get(p)
    if cached is not None:
        return cached

    file_path = _bt_file_path_for_sql(state)
    conn = duckdb.connect(":memory:")
    try:
        row = conn.execute(
            f"SELECT next_bid_indices FROM read_parquet('{file_path}') WHERE bt_index = {p} LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    out: list[int] = []
    if row and row[0]:
        out = [int(x) for x in (row[0] or []) if x is not None]

    _cache_put(_NEXT_BID_INDICES_CACHE, p, out, _CACHE_MAX_PARENTS)
    return out


def _get_child_map_for_parent(state: Dict[str, Any], parent_bt_index: int) -> dict[str, int]:
    """Return dict: candidate_bid (UPPER) -> child bt_index for a parent node."""
    p = int(parent_bt_index)

    # Try Gemini-3.2 CSR index first
    g3_index = state.get("g3_index")
    if g3_index:
        if p >= len(g3_index.offsets) - 1: return {}
        start, end = g3_index.offsets[p], g3_index.offsets[p + 1]
        children = g3_index.children[start:end]
        bidcodes = g3_index.bidcodes[start:end]
        
        # We need BID_TO_CODE/CODE_TO_BID here. They are defined in the api.py, 
        # but the handlers module might not have them.
        # Luckily, BID_TO_CODE is basically constant, but we should import it or get it from state.
        # Since this is a plugin, we'll try to get it from state or define a local copy if needed.
        # Actually, we can just use the CODE_TO_BID from the api module if we can import it,
        # but it's better to pass it via state or use the vocab helper.
        
        # For now, let's assume we can get it from the api module or just re-define it here
        # to avoid complex circular imports.
        _, code_to_bid = _get_local_bid_vocab()
        
        out = {}
        for i in range(len(children)):
            code = bidcodes[i]
            if code > 0 and code < len(code_to_bid):
                bid_str = code_to_bid[code]
                out[bid_str] = int(children[i])
        return out

    cached = _NEXT_CHILDREN_MIN_CACHE.get(p)
    if cached is not None:
        return cached

    next_indices = _get_next_bid_indices_for_parent(state, p)
    if not next_indices:
        out: dict[str, int] = {}
        _cache_put(_NEXT_CHILDREN_MIN_CACHE, p, out, _CACHE_MAX_PARENTS)
        return out

    file_path = _bt_file_path_for_sql(state)
    in_list = ", ".join(str(int(x)) for x in next_indices if x is not None)
    if not in_list:
        out = {}
        _cache_put(_NEXT_CHILDREN_MIN_CACHE, p, out, _CACHE_MAX_PARENTS)
        return out

    conn = duckdb.connect(":memory:")
    try:
        rows = conn.execute(
            f"SELECT bt_index, candidate_bid FROM read_parquet('{file_path}') WHERE bt_index IN ({in_list})"
        ).fetchall()
    finally:
        conn.close()

    out = {}
    for bt_idx, cand in rows:
        if cand is None or bt_idx is None:
            continue
        out[str(cand).upper()] = int(bt_idx)

    _cache_put(_NEXT_CHILDREN_MIN_CACHE, p, out, _CACHE_MAX_PARENTS)
    return out


def _resolve_bt_index_by_traversal(state: Dict[str, Any], auction: str) -> int | None:
    """Resolve an auction prefix to a bt_index using the Gemini-3.2 CSR index.
    
    This replaces the old manual traversal and DuckDB lookups.
    """
    g3_index = state.get("g3_index")
    if g3_index:
        # Gemini-3.2: O(1) direct address traversal
        return g3_index.walk(auction)

    # Fallback to old slow logic (only if G3 not built)
    auction_input = normalize_auction_input(auction)
    auction_norm = re.sub(r"(?i)^(P-)+", "", auction_input).rstrip("-").upper()
    tokens = [t for t in auction_norm.split("-") if t]
    if not tokens:
        return None

    bt_openings_df = state.get("bt_openings_df")
    if not isinstance(bt_openings_df, pl.DataFrame) or bt_openings_df.is_empty():
        return None # Silent fail, caller handles

    seat1_open = bt_openings_df.filter(pl.col("seat") == 1) if "seat" in bt_openings_df.columns else bt_openings_df
    first_tok = tokens[0].upper()
    first_match = seat1_open.filter(pl.col("Auction") == first_tok)
    if first_match.height == 0:
        return None
    bt_idx = first_match.row(0, named=True).get("bt_index")
    if bt_idx is None:
        return None
    parent_bt_index = int(bt_idx)

    for tok in tokens[1:]:
        child_map = _get_child_map_for_parent(state, parent_bt_index)
        nxt = child_map.get(str(tok).upper())
        if nxt is None:
            return None
        parent_bt_index = int(nxt)

    return parent_bt_index


from plugins.bbo_bt_custom_criteria_overlay import apply_custom_criteria_overlay_to_bt_row as _apply_custom_criteria_overlay_to_bt_row


def _compute_cumulative_deal_mask(
    state: dict[str, Any],
    bt_row: dict[str, Any],
    up_to_seat: int,
) -> "pl.Series | None":
    """Build cumulative mask for all seats 1..up_to_seat using bt_row's Agg_Expr_Seat_N columns."""
    deal_df = state.get("deal_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    
    if deal_df is None:
        return None

    dealer_series = deal_df["Dealer"]
    global_mask: pl.Series | None = None
    
    # For each seat 1..up_to_seat, AND in the criteria
    for seat_i in range(1, up_to_seat + 1):
        criteria_list = bt_row.get(f"Agg_Expr_Seat_{seat_i}") or []
        if not criteria_list:
            continue
            
        seat_criteria_for_seat = deal_criteria_by_seat_dfs.get(seat_i, {})
        if not seat_criteria_for_seat:
            continue

        # Find valid criteria
        sample_criteria_df = None
        for dealer in DIRECTIONS:
            sample_criteria_df = seat_criteria_for_seat.get(dealer)
            if sample_criteria_df is not None and not sample_criteria_df.is_empty():
                break
        
        if sample_criteria_df is None:
            continue
            
        available_cols = set(sample_criteria_df.columns)
        valid_criteria = [c for c in criteria_list if c in available_cols]
        
        if not valid_criteria:
            continue

        # Build per-dealer masks for this seat
        seat_mask: pl.Series | None = None
        for dealer in DIRECTIONS:
            dealer_mask = dealer_series == dealer
            if not dealer_mask.any():
                continue

            seat_criteria_df = seat_criteria_for_seat.get(dealer)
            if seat_criteria_df is None or seat_criteria_df.is_empty():
                continue

            combined = dealer_mask
            for crit in valid_criteria:
                combined = combined & seat_criteria_df[crit]
                if not combined.any():
                    combined = None
                    break

            if combined is not None:
                seat_mask = combined if seat_mask is None else (seat_mask | combined)
        
        # AND this seat's mask into global mask
        if seat_mask is not None:
            global_mask = seat_mask if global_mask is None else (global_mask & seat_mask)

    return global_mask


def _compute_deal_count_with_base_mask(
    state: dict[str, Any],
    base_mask: "pl.Series | None",
    bt_row: dict[str, Any],
    seat: int,
) -> int:
    """Compute deal count by intersecting base_mask with new seat's criteria."""
    deal_df = state.get("deal_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    
    if deal_df is None:
        return 0

    seat_i = max(1, min(4, int(seat)))
    criteria_list = bt_row.get(f"Agg_Expr_Seat_{seat_i}") or []
    
    # If no new criteria, just count the base mask
    if not criteria_list:
        return int(base_mask.sum()) if base_mask is not None else 0
    
    seat_criteria_for_seat = deal_criteria_by_seat_dfs.get(seat_i, {})
    if not seat_criteria_for_seat:
        return int(base_mask.sum()) if base_mask is not None else 0

    # Find valid criteria
    sample_criteria_df = None
    for dealer in DIRECTIONS:
        sample_criteria_df = seat_criteria_for_seat.get(dealer)
        if sample_criteria_df is not None and not sample_criteria_df.is_empty():
            break
    
    if sample_criteria_df is None:
        return int(base_mask.sum()) if base_mask is not None else 0
        
    available_cols = set(sample_criteria_df.columns)
    valid_criteria = [c for c in criteria_list if c in available_cols]
    
    if not valid_criteria:
        return int(base_mask.sum()) if base_mask is not None else 0

    dealer_series = deal_df["Dealer"]
    new_seat_mask: pl.Series | None = None
    
    for dealer in DIRECTIONS:
        dealer_mask = dealer_series == dealer
        if not dealer_mask.any():
            continue

        seat_criteria_df = seat_criteria_for_seat.get(dealer)
        if seat_criteria_df is None or seat_criteria_df.is_empty():
            continue

        combined = dealer_mask
        for crit in valid_criteria:
            combined = combined & seat_criteria_df[crit]
            if not combined.any():
                combined = None
                break

        if combined is not None:
            new_seat_mask = combined if new_seat_mask is None else (new_seat_mask | combined)

    if new_seat_mask is None:
        return 0
    
    # Intersect with base mask
    if base_mask is not None:
        final_mask = base_mask & new_seat_mask
    else:
        final_mask = new_seat_mask
    
    return int(final_mask.sum())


def _compute_seat_stats_for_bt_row(
    state: dict[str, Any],
    bt_row: dict[str, Any],
    seat: int,
) -> dict[str, Any]:
    """Compute on-the-fly stats for a single bt_seat1 row and seat using deal_df + bitmaps.
    
    Returns a dict with:
        - "matching_deal_count": int
        - "stats": nested dict of metric -> {min,max,mean,std} | None
        - "expr": list of criteria expression names (Agg_Expr_Seat_{seat}) used for this seat
        - "invalid_criteria": list of criteria names from Agg_Expr that have no bitmap column
    """
    seat_i = max(1, min(4, int(seat)))
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state["deal_criteria_by_seat_dfs"]

    criteria_col = f"Agg_Expr_Seat_{seat_i}"
    criteria_list = bt_row.get(criteria_col) or []
    if not criteria_list:
        # No criteria for this seat: zero deals, no stats, but return empty expr list so
        # callers can distinguish "no criteria" from "criteria with no matching deals".
        return {"matching_deal_count": 0, "stats": None, "expr": [], "invalid_criteria": []}

    # Build a global mask over deals using the precomputed bitmaps for this seat.
    dealer_series = deal_df["Dealer"]
    global_mask: pl.Series | None = None
    
    # Track criteria that don't have bitmap columns (invalid/missing criteria)
    invalid_criteria: list[str] = []
    valid_criteria: list[str] = []

    seat_criteria_for_seat = deal_criteria_by_seat_dfs.get(seat_i, {})
    if not seat_criteria_for_seat:
        return {"matching_deal_count": 0, "stats": None, "expr": criteria_list, "invalid_criteria": list(criteria_list)}

    # Check which criteria have bitmap columns (use first available dealer's df)
    sample_criteria_df = None
    for dealer in DIRECTIONS:
        sample_criteria_df = seat_criteria_for_seat.get(dealer)
        if sample_criteria_df is not None and not sample_criteria_df.is_empty():
            break
    
    if sample_criteria_df is not None:
        available_cols = set(sample_criteria_df.columns)
        for crit in criteria_list:
            if crit in available_cols:
                valid_criteria.append(crit)
            else:
                invalid_criteria.append(crit)
    else:
        invalid_criteria = list(criteria_list)

    for dealer in DIRECTIONS:
        dealer_mask = dealer_series == dealer
        if not dealer_mask.any():
            continue

        seat_criteria_df = seat_criteria_for_seat.get(dealer)
        if seat_criteria_df is None or seat_criteria_df.is_empty():
            continue

        combined = dealer_mask
        for crit in valid_criteria:
            combined = combined & seat_criteria_df[crit]
            if not combined.any():
                combined = None
                break

        if combined is not None:
            global_mask = combined if global_mask is None else (global_mask | combined)

    if global_mask is None or not global_mask.any():
        return {"matching_deal_count": 0, "stats": None, "expr": criteria_list, "invalid_criteria": invalid_criteria}

    matched_df = deal_df.filter(global_mask)
    if matched_df.is_empty():
        return {"matching_deal_count": 0, "stats": None, "expr": criteria_list, "invalid_criteria": invalid_criteria}

    matching_deal_count = matched_df.height
    # ------------------------------------------------------------------
    # Build seat-relative metrics per row (HCP / suit lengths / total points)
    # directly from the actual hand string for this seat (derived from
    # Hand_N/E/S/W). We keep everything expression-based to avoid referencing
    # intermediate columns that don't yet exist.
    # ------------------------------------------------------------------

    dir_map = _seat_direction_map(seat_i)

    # Base expression for this seat's hand: PBN format "S.H.D.C".
    hand_expr = (
        pl.when(pl.col("Dealer") == DIRECTIONS[0])
        .then(pl.col(f"Hand_{dir_map[DIRECTIONS[0]]}"))
        .when(pl.col("Dealer") == DIRECTIONS[1])
        .then(pl.col(f"Hand_{dir_map[DIRECTIONS[1]]}"))
        .when(pl.col("Dealer") == DIRECTIONS[2])
        .then(pl.col(f"Hand_{dir_map[DIRECTIONS[2]]}"))
        .otherwise(pl.col(f"Hand_{dir_map[DIRECTIONS[3]]}"))
    )

    # Suit lengths from hand_expr (PBN format "S.H.D.C").
    split_col = hand_expr.str.split(".")
    sl_s_expr = split_col.list.get(0).str.len_chars().alias("SL_S_seat")
    sl_h_expr = split_col.list.get(1).str.len_chars().alias("SL_H_seat")
    sl_d_expr = split_col.list.get(2).str.len_chars().alias("SL_D_seat")
    sl_c_expr = split_col.list.get(3).str.len_chars().alias("SL_C_seat")

    # HCP from hand_expr: 4*A + 3*K + 2*Q + 1*J
    hcp_expr = (
        hand_expr.str.count_matches("A") * 4
        + hand_expr.str.count_matches("K") * 3
        + hand_expr.str.count_matches("Q") * 2
        + hand_expr.str.count_matches("J")
    ).alias("HCP_seat")

    # Distribution points: void=3, singleton=2, doubleton=1 per suit.
    def _dp_for(col: str) -> pl.Expr:
        c = pl.col(col)
        return (
            pl.when(c == 0)
            .then(3)
            .when(c == 1)
            .then(2)
            .when(c == 2)
            .then(1)
            .otherwise(0)
        )

    # First add suit-length and HCP columns
    matched_df = matched_df.with_columns(
        [
            sl_s_expr,
            sl_h_expr,
            sl_d_expr,
            sl_c_expr,
            hcp_expr,
        ]
    )

    # Then compute distribution points and total points based on those columns.
    # NOTE: We must add DP_seat in a separate with_columns call before referencing it
    # in Total_Points_seat, otherwise some Polars versions will fail to resolve it.
    dp_expr = (
        (_dp_for("SL_S_seat") + _dp_for("SL_H_seat") + _dp_for("SL_D_seat") + _dp_for("SL_C_seat"))
        .alias("DP_seat")
    )
    matched_df = matched_df.with_columns(dp_expr)

    total_points_expr = (pl.col("HCP_seat") + pl.col("DP_seat")).alias("Total_Points_seat")
    matched_df = matched_df.with_columns(total_points_expr)

    metric_col_map: dict[str, str] = {
        "HCP": "HCP_seat",
        "SL_S": "SL_S_seat",
        "SL_H": "SL_H_seat",
        "SL_D": "SL_D_seat",
        "SL_C": "SL_C_seat",
        "Total_Points": "Total_Points_seat",
    }

    # Filter to metrics whose columns actually exist (defensive).
    available_metrics = {
        m: c for m, c in metric_col_map.items() if c in matched_df.columns
    }
    if not available_metrics:
        return {"matching_deal_count": matching_deal_count, "stats": None, "expr": criteria_list, "invalid_criteria": invalid_criteria}

    # Aggregate min/max/mean/std for each metric in a single pass.
    agg_exprs: list[pl.Expr] = [pl.len().alias("matching_deal_count")]
    for metric_key, col_name in available_metrics.items():
        col = pl.col(col_name)
        agg_exprs.extend(
            [
                col.min().alias(f"{metric_key}_min"),
                col.max().alias(f"{metric_key}_max"),
                col.mean().alias(f"{metric_key}_mean"),
                col.std().alias(f"{metric_key}_std"),
            ]
        )

    agg_df = matched_df.select(agg_exprs)
    agg_row = agg_df.row(0, named=True)

    # Build nested stats dict.
    stats: dict[str, dict[str, float | None]] = {}
    for metric_key in available_metrics.keys():
        stats[metric_key] = {
            "min": _safe_float(agg_row.get(f"{metric_key}_min")),
            "max": _safe_float(agg_row.get(f"{metric_key}_max")),
            "mean": _safe_float(agg_row.get(f"{metric_key}_mean")),
            "std": _safe_float(agg_row.get(f"{metric_key}_std")),
        }

    # Prefer the exact aggregated count (in case some rows had nulls).
    mcount = agg_row.get("matching_deal_count")
    if mcount is not None:
        matching_deal_count = int(mcount)

    return {"matching_deal_count": matching_deal_count, "stats": stats, "expr": criteria_list, "invalid_criteria": invalid_criteria}


# ---------------------------------------------------------------------------
# Handler: /openings-by-deal-index
# ---------------------------------------------------------------------------


def handle_openings_by_deal_index(
    state: Dict[str, Any],
    sample_size: int,
    seats: Optional[List[int]],
    directions: Optional[List[str]],
    opening_directions: Optional[List[str]],
    seed: Optional[int],
    load_auction_criteria_fn,
    filter_auctions_by_hand_criteria_fn,
) -> Dict[str, Any]:
    """Handle /openings-by-deal-index endpoint."""
    t0 = time.perf_counter()
    
    deal_df = state["deal_df"]
    bt_openings_df = state.get("bt_openings_df")
    if bt_openings_df is None:
        raise ValueError("bt_openings_df not loaded (pipeline error)")
    results = state["results"]

    dir_to_idx = {d: i for i, d in enumerate(DIRECTIONS)}

    # Load custom auction criteria once per request
    criteria_list = load_auction_criteria_fn()
    criteria_loaded = len(criteria_list)

    seats_to_process = seats if seats is not None else [1, 2, 3, 4]
    directions_to_process = directions if directions is not None else list(DIRECTIONS)
    opening_dirs_filter = opening_directions

    # Add original row position BEFORE filtering so we can track it after sampling
    # This is needed because results[] is keyed by row positions in deal_df
    deal_df_with_pos = deal_df.with_row_index("_original_row_pos")
    
    # Filter by dealer then sample exactly sample_size deals
    target_rows = deal_df_with_pos.filter(pl.col("Dealer").is_in(directions_to_process))
    if target_rows.height > sample_size:
        effective_seed = _effective_seed(seed)
        target_rows = target_rows.sample(n=sample_size, seed=effective_seed)
    
    out_deals: List[Dict[str, Any]] = []

    for row_idx in range(target_rows.height):
        current_row = target_rows.row(row_idx, named=True)
        dealer = current_row["Dealer"]
        idx_val = int(current_row["index"])
        dealer_idx = dir_to_idx[dealer]
        # Use the tracked original row position, not the index column value
        original_pos = int(current_row["_original_row_pos"])

        # Compute Par once per deal (used to enrich each opening-bid row)
        par_score: Any = current_row.get("ParScore")
        par_contract: Any = current_row.get("ParContract")
        par_contracts_display = _format_par_contracts(current_row.get("ParContracts"))
        # Prefer the dataset's ParContracts formatting (matches EV_ParContracts list).
        if par_contracts_display:
            par_contract = par_contracts_display
        if par_score is None or par_contract is None:
            try:
                dirs = ["N", "E", "S", "W"]
                d_idx = dirs.index(dealer) if dealer in dirs else 0
                hands_in_dealer_order = []
                for i in range(4):
                    d = dirs[(d_idx + i) % 4]
                    hands_in_dealer_order.append(current_row.get(f"Hand_{d}", "") or "")
                pbn = f"{dealer}:" + " ".join(hands_in_dealer_order)
                vul = current_row.get("Vul", "None") or "None"
                par_res = compute_par_score(pbn, dealer, str(vul))
                par_score = par_res.get("Par_Score")
                par_contract = par_res.get("Par_Contract")
            except Exception:
                # If endplay isn't available or something fails, keep None
                par_score = par_score if par_score is not None else None
                par_contract = par_contract if par_contract is not None else None

        opening_bids: List[int] = []
        opening_seat_num: Optional[int] = None
        criteria_rejected_for_deal: list[dict] = []
        criteria_filtered_for_deal = 0

        for seat in seats_to_process:
            key = (dealer, seat)
            if key not in results:
                continue

            orig_indices = results[key]["original_indices"]
            pos = orig_indices.search_sorted(original_pos)

            if pos < len(orig_indices) and int(orig_indices[pos]) == original_pos:
                bids = results[key]["candidates"]["candidate_bids"][pos]
                if bids is not None and len(bids) > 0:
                    opener_for_seat = DIRECTIONS[(dealer_idx + seat - 1) % 4]
                    if opening_dirs_filter is None or opener_for_seat in opening_dirs_filter:
                        bids_list = [int(b) for b in bids]
                        if criteria_loaded > 0:
                            hand_col = f"Hand_{opener_for_seat}"
                            hand_str = current_row.get(hand_col)
                            hand_values = compute_hand_features(hand_str) if isinstance(hand_str, str) else {}

                            if hand_values:
                                bt_display_cols = ["index", "Auction", "seat", "Expr"]
                                available_bt_cols = [c for c in bt_display_cols if c in bt_openings_df.columns]
                                cand_bt = bt_openings_df.filter(pl.col("index").is_in(bids_list)).select(available_bt_cols)
                                pre = cand_bt.height
                                cand_bt, rejected = filter_auctions_by_hand_criteria_fn(
                                    cand_bt, hand_values=hand_values, seat=seat, auction_col="Auction"
                                )
                                post = cand_bt.height
                                if pre != post:
                                    criteria_filtered_for_deal += (pre - post)
                                if rejected:
                                    for r in rejected[:10]:
                                        r["Dealer"] = str(dealer)
                                        r["Opening_Direction"] = str(opener_for_seat)
                                    criteria_rejected_for_deal.extend(rejected[:10])

                                bids_list = [int(x) for x in cand_bt["index"].to_list()] if cand_bt.height else []

                        if bids_list:
                            if opening_seat_num is None:
                                opening_seat_num = seat
                            opening_bids.extend(bids_list)

        # Always include the deal, even if no opening bids found (empty df)
        opening_seat = DIRECTIONS[(dealer_idx + opening_seat_num - 1) % 4] if opening_seat_num else None

        opening_bids_unique = list(dict.fromkeys(opening_bids))

        opening_bids_df_rows: List[Dict[str, Any]] = []
        if opening_bids_unique:
            bt_display_cols = ["index", "Auction", "seat", "Expr"]
            available_bt_cols = [c for c in bt_display_cols if c in bt_openings_df.columns]
            filtered_bt = bt_openings_df.filter(pl.col("index").is_in(opening_bids_unique)).select(available_bt_cols)
            opening_bids_df_rows = filtered_bt.to_dicts()
            # Enrich each bid row with deal-level and derived contract info
            ev_par_contracts = _ev_list_for_par_contracts(current_row)
            actual_contract = current_row.get("Contract")
            bid_str = _bid_value_to_str(current_row.get("bid"))
            for r in opening_bids_df_rows:
                auction = r.get("Auction")
                auction_disp = _display_auction_with_seat_prefix(auction, r.get("seat"))
                r["Auction"] = auction_disp
                if "Rules_Auction" in r:
                    r["Rules_Auction"] = _display_auction_with_seat_prefix(r.get("Rules_Auction"), r.get("seat"))
                r["Dealer"] = dealer
                # IMPORTANT: contract/dd/ev computations require seat-relative prefixes.
                r["Rules_Contract"] = get_ai_contract(str(auction_disp), dealer) if auction_disp is not None else None
                r["Actual_Contract"] = actual_contract
                # 'Actual_Auction' is the stringized deal_df['bid'] column
                r["Actual_Auction"] = bid_str
                r["DD_Score_Declarer"] = (
                    get_dd_score_for_auction(str(auction_disp), dealer, current_row) if auction_disp is not None else None
                )
                r["EV_Score_Declarer"] = (
                    get_ev_for_auction(str(auction_disp), dealer, current_row) if auction_disp is not None else None
                )
                r["ParScore"] = par_score
                r["ParContract"] = par_contract
                r["EV_ParContracts"] = ev_par_contracts

        hands = {
            "Hand_N": current_row.get("Hand_N"),
            "Hand_E": current_row.get("Hand_E"),
            "Hand_S": current_row.get("Hand_S"),
            "Hand_W": current_row.get("Hand_W"),
        }

        contract = current_row.get("Contract")
        dd_score_declarer = current_row.get("DD_Score_Declarer")
        par_score = current_row.get("ParScore")
        par_contracts = current_row.get("ParContracts")
        par_contracts = _format_par_contracts(par_contracts)

        out_deals.append(
            {
                "index": idx_val,
                "dealer": dealer,
                "opening_seat": opening_seat,
                "opening_bid_indices": opening_bids_unique,
                "opening_bids_df": opening_bids_df_rows,
                "hands": hands,
                "Actual_Contract": contract,
                "DD_Score_Declarer": dd_score_declarer,
                "ParScore": par_score,
                "ParContracts": par_contracts,
                "auction_criteria_loaded": criteria_loaded,
                "auction_criteria_filtered": criteria_filtered_for_deal,
                "criteria_rejected": criteria_rejected_for_deal[:10] if criteria_rejected_for_deal else [],
                "sql_query": f"SELECT * FROM auctions WHERE bt_index IN ({', '.join(map(str, opening_bids_unique[:20]))}...)" if opening_bids_unique else "-- No opening bids found"
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[openings-by-deal-index] {format_elapsed(elapsed_ms)}, {len(out_deals)} deals")
    return {
        "deals": out_deals,
        "auction_criteria_loaded": criteria_loaded,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Handler: /random-auction-sequences
# ---------------------------------------------------------------------------


def handle_find_bt_auctions_by_contracts(
    state: Dict[str, Any],
    par_contracts: List[Dict[str, Any]],
    dealer: str,
    auction_prefix: str = "",
    deal_row_idx: int | None = None,
) -> Dict[str, Any]:
    """Find BT completed auctions matching a set of ParContracts.
    
    Uses bt_stats_df (completed auctions with aggregates) when available so we can
    return the same basic stats shown elsewhere in the UI (Deals, Avg_EV).
    If auction_prefix is provided, only returns auctions starting with that prefix.
    """
    t0 = time.perf_counter()
    
    bt_stats_df = state.get("bt_stats_df")
    bt_completed_agg_df = state.get("bt_completed_agg_df")
    bt_seat1_df = state.get("bt_seat1_df")

    # ---------------------------------------------------------------------
    # Fast path: existing deal (dataset-backed) with precomputed Par_Indexes.
    # ---------------------------------------------------------------------
    deal_to_bt_par_index_df: Optional[pl.DataFrame] = state.get("deal_to_bt_par_index_df")
    if deal_row_idx is not None and deal_to_bt_par_index_df is not None and deal_to_bt_par_index_df.height > 0:
        import numpy as np

        # Cache numpy materializations across requests for O(log n) lookup.
        global _DEAL_TO_BT_PAR_INDEX_CACHE  # type: ignore[declared-but-unused]
        try:
            _DEAL_TO_BT_PAR_INDEX_CACHE  # type: ignore[name-defined]
        except Exception:
            _DEAL_TO_BT_PAR_INDEX_CACHE = {}  # type: ignore[name-defined]

        cache = _DEAL_TO_BT_PAR_INDEX_CACHE  # type: ignore[name-defined]
        df_id = id(deal_to_bt_par_index_df)
        if cache.get("df_id") != df_id:
            cache["df_id"] = df_id
            cache["deal_idx_arr"] = deal_to_bt_par_index_df["deal_idx"].to_numpy()
            cache["par_series"] = deal_to_bt_par_index_df.get_column("Par_Indexes")

        deal_idx_arr = cache.get("deal_idx_arr")
        par_series = cache.get("par_series")
        if deal_idx_arr is None or par_series is None:
            raise ValueError("deal_to_bt_par_index_df cache not initialized")

        pos = np.searchsorted(deal_idx_arr, int(deal_row_idx))
        par_indices: list[int] = []
        if pos < len(deal_idx_arr) and int(deal_idx_arr[pos]) == int(deal_row_idx):
            try:
                m = par_series[int(pos)]
            except Exception:
                m = None
            if m is None:
                par_indices = []
            else:
                try:
                    if isinstance(m, pl.Series):
                        par_indices = [int(x) for x in m.to_list() if x is not None]
                    elif isinstance(m, (list, tuple)):
                        par_indices = [int(x) for x in m if x is not None]
                    else:
                        par_indices = [int(x) for x in list(m) if x is not None]
                except Exception:
                    par_indices = []

        if not par_indices:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {"auctions": [], "elapsed_ms": round(elapsed_ms, 1), "total_completed_searched": 0}

        # Fetch minimal completed-auction rows for these bt_index values.
        # Prefer bt_stats_df (has matching_deal_count), else bt_completed_agg_df, else seat1 (slow).
        if bt_stats_df is not None and bt_stats_df.height > 0:
            subset = bt_stats_df.filter(pl.col("bt_index").is_in(par_indices))
        elif bt_completed_agg_df is not None and bt_completed_agg_df.height > 0:
            subset = bt_completed_agg_df.filter(pl.col("bt_index").is_in(par_indices))
        else:
            if bt_seat1_df is None:
                raise ValueError("bt_seat1_df not loaded")
            subset = bt_seat1_df.filter(pl.col("bt_index").is_in(par_indices) & pl.col("is_completed_auction"))

        # Ensure Auction exists for prefix filtering + contract parsing.
        if "Auction" not in subset.columns:
            lookup: pl.DataFrame | None = None
            if bt_completed_agg_df is not None and bt_completed_agg_df.height > 0 and "Auction" in bt_completed_agg_df.columns:
                lookup = bt_completed_agg_df.filter(pl.col("bt_index").is_in(par_indices)).select(["bt_index", "Auction"])
            elif bt_seat1_df is not None and bt_seat1_df.height > 0 and "Auction" in bt_seat1_df.columns:
                lookup = bt_seat1_df.filter(pl.col("bt_index").is_in(par_indices)).select(["bt_index", "Auction"])
            if lookup is None:
                raise ValueError("Auction column missing and no bt_index→Auction lookup available")
            subset = subset.join(lookup, on="bt_index", how="left").drop_nulls(subset=["Auction"])

        # Apply prefix filter (small subset).
        auction_prefix = str(auction_prefix or "").strip().upper()
        if auction_prefix:
            prefix_with_sep = auction_prefix if auction_prefix.endswith("-") else auction_prefix + "-"
            subset = subset.filter(
                pl.col("Auction").str.to_uppercase().str.starts_with(prefix_with_sep)
                | pl.col("Auction").str.to_uppercase().eq(auction_prefix)
            )

        # Normalize dealer (same semantics as slow path).
        dealer = str(dealer).upper()
        if dealer not in DIRECTIONS:
            dealer = "N"

        # Build response rows by parsing only the returned auctions (near-instant).
        auctions = subset["Auction"].to_list()
        bt_indices = subset["bt_index"].to_list()
        deals_list = subset["matching_deal_count"].to_list() if "matching_deal_count" in subset.columns else None

        matches: list[dict[str, Any]] = []
        for i, auc in enumerate(auctions):
            parsed = parse_contract_from_auction(auc)
            if not parsed:
                continue
            l, s, d_count = parsed
            d_str = "XX" if d_count == 2 else ("X" if d_count == 1 else "")
            decl = get_declarer_for_auction(auc, "N")  # opener is North (seat-1 view)
            if not decl:
                continue

            is_opener_pair = decl in ("N", "S")
            if dealer in ("N", "S"):
                opener_pair = "NS"
                opponent_pair = "EW"
            else:
                opener_pair = "EW"
                opponent_pair = "NS"
            actual_pair = opener_pair if is_opener_pair else opponent_pair
            decl_seat = 1 if decl == "N" else (2 if decl == "E" else (3 if decl == "S" else (4 if decl == "W" else None)))

            bt_i = bt_indices[i]
            if bt_i is None:
                continue

            matches.append(
                {
                    "bt_index": int(bt_i),
                    "Auction": auc,
                    "Contract": f"{l}{s}{d_str}{decl}",
                    "Pair": actual_pair,
                    "Deals": int(deals_list[i]) if (deals_list is not None and deals_list[i] is not None) else None,
                    "decl": decl,
                    "decl_seat": decl_seat,
                }
            )

        # Attach avg_ev_nv/avg_ev_v from bt_ev_stats_df (same as slow path).
        bt_ev_stats_df = state.get("bt_ev_stats_df")
        if bt_ev_stats_df is not None and matches:
            try:
                bt_idx_list_fast: list[int] = []
                for mm in matches:
                    bt_val = mm.get("bt_index")
                    if bt_val is None:
                        continue
                    bt_idx_list_fast.append(int(bt_val))
                bt_idx_list_fast = list(dict.fromkeys(bt_idx_list_fast))
                if bt_idx_list_fast:
                    ev_subset = bt_ev_stats_df.filter(pl.col("bt_index").is_in(bt_idx_list_fast))
                    ev_map_fast: dict[int, dict[str, Any]] = {}
                    for ev_row in ev_subset.iter_rows(named=True):
                        try:
                            ev_map_fast[int(ev_row["bt_index"])] = dict(ev_row)
                        except Exception:
                            continue
                    for m in matches:
                        bt_i = m.get("bt_index")
                        seat_i = m.get("decl_seat")
                        if bt_i is None or seat_i is None:
                            m["avg_ev_nv"] = None
                            m["avg_ev_v"] = None
                            continue
                        row = ev_map_fast.get(int(bt_i), {}) or {}
                        nv_key = f"Avg_EV_S{int(seat_i)}_NV"
                        v_key = f"Avg_EV_S{int(seat_i)}_V"
                        if nv_key in row:
                            m["avg_ev_nv"] = row.get(nv_key)
                            m["avg_ev_v"] = row.get(v_key)
                        else:
                            aggregate = row.get(f"Avg_EV_S{int(seat_i)}")
                            m["avg_ev_nv"] = aggregate
                            m["avg_ev_v"] = aggregate
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"auctions": matches, "elapsed_ms": round(elapsed_ms, 1), "total_completed_searched": int(len(par_indices))}

    if bt_stats_df is not None and bt_stats_df.height > 0:
        completed_df = bt_stats_df
    elif bt_completed_agg_df is not None and bt_completed_agg_df.height > 0:
        # Fallback: bt_completed_agg_df (may not include Deals/Avg_EV columns)
        completed_df = bt_completed_agg_df
    else:
        # Slow fallback: filter bt_seat1_df for completed auctions
        if bt_seat1_df is None:
            raise ValueError("bt_seat1_df not loaded")
        if "is_completed_auction" not in bt_seat1_df.columns:
            raise ValueError("is_completed_auction column missing")
        completed_df = bt_seat1_df.filter(pl.col("is_completed_auction"))

    if completed_df.height == 0:
        return {"auctions": [], "elapsed_ms": 0}

    # bt_stats_df is keyed by bt_index but may not include the auction string.
    # For this endpoint we must have "Auction" available to:
    # - filter by prefix
    # - parse the final contract from the auction string
    if "Auction" not in completed_df.columns:
        auction_lookup: pl.DataFrame | None = None
        if bt_completed_agg_df is not None and bt_completed_agg_df.height > 0 and "Auction" in bt_completed_agg_df.columns:
            auction_lookup = bt_completed_agg_df.select(["bt_index", "Auction"])
        elif bt_seat1_df is not None and bt_seat1_df.height > 0 and "Auction" in bt_seat1_df.columns:
            # Slowest in-memory fallback (bt_seat1_df is large), but only used if needed.
            auction_lookup = bt_seat1_df.select(["bt_index", "Auction"])
        if auction_lookup is None:
            raise ValueError("Auction column missing and no bt_index→Auction lookup available")
        completed_df = completed_df.join(auction_lookup, on="bt_index", how="left")
        # If the join fails to recover auction strings, drop those rows (cannot match/parsing fails).
        if "Auction" in completed_df.columns:
            completed_df = completed_df.drop_nulls(subset=["Auction"])
    
    # Normalize and apply auction prefix filter if provided
    auction_prefix = str(auction_prefix or "").strip().upper()
    if auction_prefix:
        # Prefix match: auction must start with the given prefix
        # Handle cases where prefix may or may not end with a separator
        prefix_with_sep = auction_prefix if auction_prefix.endswith("-") else auction_prefix + "-"
        completed_df = completed_df.filter(
            pl.col("Auction").str.to_uppercase().str.starts_with(prefix_with_sep) |
            pl.col("Auction").str.to_uppercase().eq(auction_prefix)
        )

    # Normalize dealer
    dealer = str(dealer).upper()
    if dealer not in DIRECTIONS:
        dealer = "N"

    # Prepare target contracts for matching
    targets = []
    for c in par_contracts:
        level = c.get("Level")
        strain = c.get("Strain")
        pair = c.get("Pair_Direction")
        dbl = c.get("Doubled") or c.get("Double") or ""
        if level is not None and strain is not None and pair in ("NS", "EW"):
            targets.append({
                "level": int(level),
                "strain": str(strain).upper(),
                "pair": pair,
                "doubled": str(dbl).upper() if dbl else ""
            })

    if not targets:
        return {"auctions": [], "elapsed_ms": 0, "message": "No valid target contracts provided"}

    # Optimization: Filter by strain first if possible, but strain is inside the Auction string.
    # We'll iterate and filter. For 1M rows, this should be fast enough if we don't do too much per row.
    auctions = completed_df["Auction"].to_list()
    bt_indices = completed_df["bt_index"].to_list()
    # Optional stats columns (present in bt_stats_df)
    deals_list = completed_df["matching_deal_count"].to_list() if "matching_deal_count" in completed_df.columns else None
    
    matches = []
    
    # Pre-parse dealer indices for pair matching
    # Opener (Seat 1) in BT is 'dealer' in the pinned deal context.
    # NS pair in pinned deal: N, S
    # EW pair in pinned deal: E, W
    
    for i, auc in enumerate(auctions):
        # parse_contract_from_auction returns (level, strain, doubled_count)
        # doubled_count: 0=none, 1=X, 2=XX
        parsed = parse_contract_from_auction(auc)
        if not parsed:
            continue
            
        l, s, d_count = parsed
        d_str = ""
        if d_count == 1: d_str = "X"
        elif d_count == 2: d_str = "XX"
        
        # Check if contract (level, strain, doubled) matches any target
        potential_targets = [t for t in targets if t["level"] == l and t["strain"] == s and t["doubled"] == d_str]
        if not potential_targets:
            continue
            
        # Contract matches, now check pair
        # Declarer relative to opener as Seat 1
        decl = get_declarer_for_auction(auc, "N") # opener is North
        if not decl:
            continue
            
        # Map BT declarer to pinned deal pair
        # BT Seat 1/3 -> opener's pair in pinned deal
        # BT Seat 2/4 -> opponent's pair in pinned deal
        
        is_opener_pair = decl in ("N", "S")
        
        # Opener's pair in pinned deal:
        if dealer in ("N", "S"):
            opener_pair = "NS"
            opponent_pair = "EW"
        else:
            opener_pair = "EW"
            opponent_pair = "NS"
            
        actual_pair = opener_pair if is_opener_pair else opponent_pair
        
        # Final match check
        for t in potential_targets:
            if t["pair"] == actual_pair:
                # Map seat-1-view direction to seat number (N/E/S/W -> 1/2/3/4)
                decl_seat = 1 if decl == "N" else (2 if decl == "E" else (3 if decl == "S" else (4 if decl == "W" else None)))
                bt_i = bt_indices[i]
                if bt_i is None:
                    break
                matches.append({
                    "bt_index": int(bt_i),
                    "Auction": auc,
                    "Contract": f"{l}{s}{d_str}{decl}", # Relative to opener
                    "Pair": actual_pair,
                    # These align with the bid-options grids (criteria-based count + precomputed Avg_EV).
                    "Deals": int(deals_list[i]) if (deals_list is not None and deals_list[i] is not None) else None,
                    "decl": decl,
                    "decl_seat": decl_seat,
                })
                break

    # Attach precomputed avg_ev_nv/avg_ev_v from bt_ev_stats_df when available.
    # We return NV/V split so the Streamlit client can choose based on pinned-deal vul.
    bt_ev_stats_df = state.get("bt_ev_stats_df")
    if bt_ev_stats_df is not None and matches:
        try:
            bt_idx_list: list[int] = []
            for mm in matches:
                bt_val = mm.get("bt_index")
                if bt_val is None:
                    continue
                bt_idx_list.append(int(bt_val))
            bt_idx_list = list(dict.fromkeys(bt_idx_list))
            if bt_idx_list:
                ev_subset = bt_ev_stats_df.filter(pl.col("bt_index").is_in(bt_idx_list))
                ev_map: dict[int, dict[str, Any]] = {}
                for ev_row in ev_subset.iter_rows(named=True):
                    try:
                        ev_map[int(ev_row["bt_index"])] = dict(ev_row)
                    except Exception:
                        continue
                for m in matches:
                    bt_i = m.get("bt_index")
                    seat_i = m.get("decl_seat")
                    if bt_i is None or seat_i is None:
                        m["avg_ev_nv"] = None
                        m["avg_ev_v"] = None
                        continue
                    row = ev_map.get(int(bt_i), {}) or {}
                    nv_key = f"Avg_EV_S{int(seat_i)}_NV"
                    v_key = f"Avg_EV_S{int(seat_i)}_V"
                    if nv_key in row:
                        m["avg_ev_nv"] = row.get(nv_key)
                        m["avg_ev_v"] = row.get(v_key)
                    else:
                        # Backwards compatibility with old aggregate stats file
                        aggregate = row.get(f"Avg_EV_S{int(seat_i)}")
                        m["avg_ev_nv"] = aggregate
                        m["avg_ev_v"] = aggregate
        except Exception:
            # Best-effort: leave avg_ev fields unset on any failure
            pass
                
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "auctions": matches,
        "elapsed_ms": round(elapsed_ms, 1),
        "total_completed_searched": completed_df.height
    }


def handle_random_auction_sequences(
    state: Dict[str, Any],
    n_samples: int,
    seed: Optional[int],
    completed_only: bool = True,
    partial_only: bool = False,
) -> Dict[str, Any]:
    """Handle /random-auction-sequences endpoint.
    
    Requires bt_seat1_df (pipeline invariant).
    
    Args:
        completed_only: If True, only sample from completed auctions (default).
        partial_only: If True, only sample from partial (non-completed) auctions.
        If both False, sample from both (50/50 split).
    """
    t0 = time.perf_counter()
    
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")

    base_df = bt_seat1_df
    index_col = "bt_index"
    effective_seed = _effective_seed(seed)
    
    # Get completed auctions DataFrame
    bt_completed_agg_df = state.get("bt_completed_agg_df")
    if bt_completed_agg_df is not None and bt_completed_agg_df.height > 0:
        completed_df = bt_completed_agg_df
    else:
        # Fallback to filtering bt_seat1_df (slow)
        if "is_completed_auction" not in bt_seat1_df.columns:
            raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")
        completed_df = bt_seat1_df.filter(pl.col("is_completed_auction"))
    
    if completed_only:
        # Sample only from completed auctions
        sample_df = completed_df
        if sample_df.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[random-auction-sequences] {format_elapsed(elapsed_ms)} (empty)")
            return {"samples": [], "elapsed_ms": round(elapsed_ms, 1)}
        
        sample_n = min(n_samples, sample_df.height)
        sampled_df = sample_df.sample(n=sample_n, seed=effective_seed)
        sampled_df = sampled_df.with_columns(pl.lit(True).alias("_is_completed_sample"))
        sql_query = f"""SELECT * FROM auctions WHERE is_completed_auction = true ORDER BY RANDOM() LIMIT {sample_n}"""
    
    elif partial_only:
        # PERFORMANCE: Generate random bt_index values and fetch by index
        # 99.8% of bt_index values are partial auctions, so this is fast
        bt_parquet_file = state.get("bt_seat1_file")
        if bt_parquet_file is None:
            raise ValueError("bt_seat1_file not set in state")
        
        import duckdb
        import random
        file_path = str(bt_parquet_file).replace("\\", "/")
        
        # Generate random bt_index values (oversample to account for completed auction hits)
        rng = random.Random(effective_seed)
        max_bt_index = 461_000_000  # Approximate max bt_index
        # Generate 3x the needed samples to ensure we get enough partial auctions
        random_indices = [rng.randint(0, max_bt_index) for _ in range(n_samples * 3)]
        indices_str = ", ".join(str(i) for i in random_indices)
        
        query = f"""
            SELECT bt_index, Auction, previous_bid_indices, Expr
            FROM read_parquet('{file_path}')
            WHERE bt_index IN ({indices_str}) AND is_completed_auction = false
            LIMIT {n_samples}
        """
        conn = duckdb.connect(":memory:")
        try:
            sampled_df = conn.execute(query).pl()
        finally:
            conn.close()
        
        if sampled_df.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[random-auction-sequences] {format_elapsed(elapsed_ms)} (empty - no partial auctions)")
            return {"samples": [], "elapsed_ms": round(elapsed_ms, 1)}
        
        if sampled_df.height > 0:
            sampled_df = sampled_df.with_columns(pl.lit(False).alias("_is_completed_sample"))
        sample_n = sampled_df.height
        sql_query = f"""SELECT * FROM auctions WHERE is_completed_auction = false LIMIT {sample_n}"""
    
    else:
        # Sample from BOTH completed and partial auctions (n/2 each)
        n_completed = n_samples // 2
        n_partial = n_samples - n_completed
        
        # Sample completed auctions (from in-memory bt_completed_agg_df - fast)
        completed_samples = min(n_completed, completed_df.height)
        sampled_completed = completed_df.sample(n=completed_samples, seed=effective_seed) if completed_samples > 0 else pl.DataFrame()
        if sampled_completed.height > 0:
            sampled_completed = sampled_completed.with_columns(pl.lit(True).alias("_is_completed_sample"))
        
        # Sample partial auctions using random bt_index values (fast)
        bt_parquet_file = state.get("bt_seat1_file")
        if bt_parquet_file is not None and n_partial > 0:
            import duckdb
            import random
            file_path = str(bt_parquet_file).replace("\\", "/")
            partial_seed = (effective_seed + 1) if effective_seed is not None else None
            
            # Generate random bt_index values
            rng = random.Random(partial_seed)
            max_bt_index = 461_000_000
            random_indices = [rng.randint(0, max_bt_index) for _ in range(n_partial * 3)]
            indices_str = ", ".join(str(i) for i in random_indices)
            
            query = f"""
                SELECT bt_index, Auction, previous_bid_indices, Expr
                FROM read_parquet('{file_path}')
                WHERE bt_index IN ({indices_str}) AND is_completed_auction = false
                LIMIT {n_partial}
            """
            conn = duckdb.connect(":memory:")
            try:
                sampled_partial = conn.execute(query).pl()
            finally:
                conn.close()
        else:
            sampled_partial = pl.DataFrame()
        
        # Combine both samples
        if sampled_completed.height > 0 and sampled_partial.height > 0:
            # Keep all columns (Agg_Expr columns from completed rows)
            sampled_df = pl.concat([sampled_completed, sampled_partial], how="diagonal_relaxed")
        elif sampled_completed.height > 0:
            sampled_df = sampled_completed
        elif sampled_partial.height > 0:
            sampled_df = sampled_partial
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[random-auction-sequences] {format_elapsed(elapsed_ms)} (empty)")
            return {"samples": [], "elapsed_ms": round(elapsed_ms, 1)}
        
        sample_n = sampled_df.height
        sql_query = f"""SELECT * FROM auctions ORDER BY RANDOM() LIMIT {sample_n}"""
    
    if index_col not in sampled_df.columns:
        raise ValueError("REQUIRED column 'bt_index' missing from sampled data. Pipeline error.")

    agg_expr_cols = [f"Agg_Expr_Seat_{i}" for i in range(1, 5)]

    extra_cols = ["Expr"] + agg_expr_cols
    display_cols = [index_col, "Auction"]
    lookup_cols = display_cols + [c for c in extra_cols if c in base_df.columns]
    available_display_cols = [c for c in lookup_cols if c in base_df.columns]

    all_prev_indices = set()
    prev_idx_col = "previous_bid_indices"
    if prev_idx_col in sampled_df.columns:
        for prev_list in sampled_df[prev_idx_col].to_list():
            if prev_list:
                all_prev_indices.update(prev_list)

    if all_prev_indices:
        # Previous bid indices are intermediate (non-completed) auctions.
        # Pull the same display columns as the final row (where available) so
        # intermediate rows aren't empty in the UI.
        prev_cols = [c for c in available_display_cols if c in base_df.columns]
        if index_col not in prev_cols and index_col in base_df.columns:
            prev_cols.append(index_col)
        prev_rows_df = base_df.filter(pl.col(index_col).is_in(list(all_prev_indices))).select(prev_cols)
        prev_rows_lookup = {row[index_col]: row for row in prev_rows_df.iter_rows(named=True)}
    else:
        prev_rows_lookup = {}

    # Preload Agg_Expr for completed final rows only (previous rows use Expr fallback).
    agg_expr_by_bt_index: dict[int, dict[str, Any]] = {}
    try:
        sampled_bt_indices = set()
        if index_col in sampled_df.columns:
            if "_is_completed_sample" in sampled_df.columns:
                completed_only_df = sampled_df.filter(pl.col("_is_completed_sample") == True)  # noqa: E712
                sampled_bt_indices = {int(x) for x in completed_only_df[index_col].drop_nulls().to_list()}
            else:
                sampled_bt_indices = {int(x) for x in sampled_df[index_col].drop_nulls().to_list()}
        if sampled_bt_indices:
            bt_parquet_file = state.get("bt_seat1_file")
            if bt_parquet_file is not None:
                agg_expr_by_bt_index = _load_agg_expr_for_bt_indices(list(sampled_bt_indices), bt_parquet_file)
    except Exception:
        agg_expr_by_bt_index = {}

    out_samples: List[Dict[str, Any]] = []
    overlay = state.get("custom_criteria_overlay") or []

    def _with_rules_seat1_cols(raw_row: dict[str, Any], step_seat: int | None = None) -> dict[str, Any]:
        """Return the canonical BT row with overlay applied.

        Fallback: use Expr as Agg_Expr_Seat_{step_seat} when Agg_Expr is missing.
        This matches Auction Builder behavior for fast path enrichment.
        """
        base = dict(raw_row)
        bt_idx = base.get("bt_index")
        if bt_idx is not None:
            try:
                bt_idx_i = int(bt_idx)
            except Exception:
                bt_idx_i = None
            if bt_idx_i is not None and bt_idx_i in agg_expr_by_bt_index:
                base.update(agg_expr_by_bt_index[bt_idx_i])
        # If Agg_Expr columns are missing, use Expr for the step seat (fast fallback).
        if step_seat is not None and "Expr" in base:
            expr_val = base.get("Expr")
            if expr_val is not None:
                for s in range(1, 5):
                    col_s = f"Agg_Expr_Seat_{s}"
                    if col_s not in base or base.get(col_s) is None:
                        base[col_s] = expr_val if s == step_seat else []
        base_row = _apply_all_rules_to_bt_row(base, state)
        # Agg_Expr_Seat_1 now contains pre-compiled merged rules
        base_row["Merged_Rules_Seat_1"] = base_row.get(agg_expr_col(1)) or []
        base_row["Agg_Expr_Seat_1_Rules"] = base_row.get(agg_expr_col(1)) or []
        return base_row

    for row in sampled_df.iter_rows(named=True):
        prev_indices = row.get(prev_idx_col, [])

        sequence_data: List[Dict[str, Any]] = []
        if prev_indices:
            for idx in prev_indices:
                if idx in prev_rows_lookup:
                    # Enrich intermediate bids using Expr fallback per seat (fast).
                    step_seat = (len(sequence_data) % 4) + 1
                    sequence_data.append(_with_rules_seat1_cols(dict(prev_rows_lookup[idx]), step_seat=step_seat))
        # Enrich the final (completed/partial) auction.
        step_seat = (len(sequence_data) % 4) + 1
        sequence_data.append(_with_rules_seat1_cols(dict(row), step_seat=step_seat))

        # IMPORTANT: keep the sequence in chronological order (previous_bid_indices chain + final row).
        # Do NOT sort by bt_index here; bt_index is not a time/sequence key and sorting breaks is_match_row.
        seq_df = pl.DataFrame(sequence_data)
        
        # Mark the final row explicitly and use it as the sample title.
        if seq_df.height > 0:
            seq_df = seq_df.with_row_index("_seq_pos").with_columns(
                (pl.col("_seq_pos") == (pl.len() - 1)).alias("is_match_row")
            ).drop("_seq_pos")
            matched_auction = seq_df.select("Auction").tail(1).item()
        else:
            matched_auction = row.get("Auction")

        out_cols = [index_col, "Auction"] if index_col in seq_df.columns else ["Auction"]
        if "is_match_row" in seq_df.columns:
            out_cols.append("is_match_row")
        if "Expr" in seq_df.columns:
            out_cols.append("Expr")
        # Include all Agg_Expr_Seat columns
        for col in agg_expr_cols:
            if col in seq_df.columns:
                out_cols.append(col)
        for col in ("Merged_Rules_Seat_1", "Agg_Expr_Seat_1_Rules"):
            if col in seq_df.columns:
                out_cols.append(col)
        seq_df = seq_df.select([c for c in out_cols if c in seq_df.columns])
        
        # Rename bt_index to index for API consistency in the API payload
        if index_col == "bt_index" and "bt_index" in seq_df.columns:
            seq_df = seq_df.rename({"bt_index": "index"})

        # Build SQL preview using whatever index column is available
        idx_col_for_sql = "index" if "index" in seq_df.columns else index_col
        if idx_col_for_sql in seq_df.columns:
            idx_vals = seq_df[idx_col_for_sql].to_list()
            idx_str = ", ".join(map(str, idx_vals))
            sql_seq = f"SELECT * FROM auctions WHERE bt_index IN ({idx_str}) ORDER BY bt_index"
        else:
            sql_seq = "-- index column not available for sequence SQL preview"

        # -------------------------------------------------------------------
        # PERFORMANCE NOTE:
        # The Random Auction Samples endpoint is a UI convenience. It must stay fast.
        # Computing per-auction deal_count and wrong_bid_rate by scanning deal_df is
        # extremely expensive (can be 10+ seconds) and duplicates work done by other
        # endpoints. Instead:
        # - Use precomputed `matching_deal_count` from BT stats when available.
        # - Do NOT compute wrong_bid_rate here (set to 0.0).
        # -------------------------------------------------------------------
        deal_count = 0
        try:
            if "matching_deal_count" in base_df.columns and row.get("matching_deal_count") is not None:
                deal_count = int(row.get("matching_deal_count") or 0)
        except Exception:
            deal_count = 0
        wrong_bid_count = 0
        wrong_bid_rate = 0.0

        out_samples.append({
            "auction": matched_auction, 
            "sequence": seq_df.to_dicts(),
            "sql_query": sql_seq,
            "deal_count": deal_count,
            "wrong_bid_count": wrong_bid_count,
            "wrong_bid_rate": round(wrong_bid_rate, 4),
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[random-auction-sequences] {format_elapsed(elapsed_ms)} ({len(out_samples)} samples)")
    return {"samples": out_samples, "elapsed_ms": round(elapsed_ms, 1), "sql_query": sql_query}


# ---------------------------------------------------------------------------
# Handler: /auction-sequences-matching
# ---------------------------------------------------------------------------


def handle_auction_sequences_matching(
    state: Dict[str, Any],
    pattern: str,
    n_samples: int,
    seed: Optional[int],
    apply_auction_criteria_fn,
    allow_initial_passes: bool = True,
) -> Dict[str, Any]:
    """Handle /auction-sequences-matching endpoint.
    
    Uses bt_seat1_df which contains only seat 1 auctions (no p- prefix complications).
    Pattern matching is done directly on clean auction strings.
    """
    t0 = time.perf_counter()
    
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")

    # Hard fail if is_completed_auction is missing
    if "is_completed_auction" not in bt_seat1_df.columns:
        raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")

    base_df = bt_seat1_df
    # Filter to completed auctions
    base_df = base_df.filter(pl.col("is_completed_auction"))

    if allow_initial_passes:
        # Seat-agnostic matching: compare against seat-1 view (ignore leading passes in stored auctions).
        pattern = _normalize_to_seat1(pattern)
        auction_expr = pl.col("Auction").cast(pl.Utf8).str.replace(r"(?i)^(p-)+", "")
    else:
        # Literal matching against raw Auction string.
        pattern = normalize_auction_user_text(pattern)
        auction_expr = pl.col("Auction").cast(pl.Utf8)
    
    is_regex = pattern.startswith("^") or pattern.endswith("$")
    regex_pattern = f"(?i){pattern}"
    filtered_df = base_df.filter(auction_expr.str.contains(regex_pattern))

    # NOTE: Auction-sequence filtering here is BT-only (no dealer/hand context), so we do not
    # apply CSV rules as a filter. Instead, CSV rules are applied as an overlay onto Agg_Expr_Seat_*.
    rejected_df = None

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "pattern": pattern,
            "samples": [],
            "total_matching": 0,
            "rejected_count": 0,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    sample_n = min(n_samples, filtered_df.height)
    effective_seed = _effective_seed(seed)
    sampled_df = filtered_df.sample(n=sample_n, seed=effective_seed)

    # Use bt_index for lookups (from bt_seat1_df)
    if "bt_index" not in base_df.columns:
        raise ValueError("REQUIRED column 'bt_index' missing from bt_seat1_df. Pipeline error.")
    index_col = "bt_index"
    
    agg_expr_cols = [f"Agg_Expr_Seat_{i}" for i in range(1, 5)]
    extra_cols = ["Expr"] + agg_expr_cols
    display_cols = [index_col, "Auction"]
    lookup_cols = display_cols + [c for c in extra_cols if c in base_df.columns]
    available_display_cols = [c for c in lookup_cols if c in base_df.columns]

    # Get previous bid indices for sequence building
    prev_idx_col = "previous_bid_indices"
    if prev_idx_col in sampled_df.columns:
        all_prev_indices = set()
        for prev_list in sampled_df[prev_idx_col].to_list():
            if prev_list:
                all_prev_indices.update(prev_list)

        if all_prev_indices:
            prev_rows_df = bt_seat1_df.filter(pl.col(index_col).is_in(list(all_prev_indices))).select(
                [c for c in available_display_cols if c in bt_seat1_df.columns]
            )
            prev_rows_lookup = {row[index_col]: row for row in prev_rows_df.iter_rows(named=True)}
        else:
            prev_rows_lookup = {}
    else:
        prev_rows_lookup = {}

    out_samples: List[Dict[str, Any]] = []
    overlay = state.get("custom_criteria_overlay") or []

    def _with_rules_seat1_cols(raw_row: dict[str, Any]) -> dict[str, Any]:
        """Return the canonical BT row (on-demand enriched) with overlay applied.

        This MUST go through `_apply_all_rules_to_bt_row` so Agg_Expr_Seat_* are available
        even when bt_seat1_df is loaded in lightweight mode (no Agg_Expr columns).
        """
        base_row = _apply_all_rules_to_bt_row(dict(raw_row), state)
        base_row["Merged_Rules_Seat_1"] = base_row.get(agg_expr_col(1)) or []
        base_row["Agg_Expr_Seat_1_Rules"] = base_row.get(agg_expr_col(1)) or []
        return base_row

    for row in sampled_df.iter_rows(named=True):
        prev_indices = row.get(prev_idx_col, [])

        sequence_data: List[Dict[str, Any]] = []
        if prev_indices:
            for idx in prev_indices:
                if idx in prev_rows_lookup:
                    sequence_data.append(_with_rules_seat1_cols(dict(prev_rows_lookup[idx])))
        sequence_data.append(_with_rules_seat1_cols({c: row[c] for c in available_display_cols if c in row}))

        # IMPORTANT: keep the sequence in chronological order (previous_bid_indices chain + final row).
        # Do NOT sort by bt_index here; bt_index is not a time/sequence key and sorting breaks is_match_row.
        seq_df = pl.DataFrame(sequence_data)

        # Mark the final (matched) row explicitly so UI can't confuse it with prefix rows.
        # Also use the final row's Auction as the sample title.
        if seq_df.height > 0:
            seq_df = seq_df.with_row_index("_seq_pos").with_columns(
                (pl.col("_seq_pos") == (pl.len() - 1)).alias("is_match_row")
            ).drop("_seq_pos")
            matched_auction = seq_df.select("Auction").tail(1).item()
        else:
            matched_auction = row.get("Auction")

        # Select output columns - rename bt_index to index for API consistency
        out_cols = []
        if index_col in seq_df.columns:
            out_cols.append(index_col)
        if "Auction" in seq_df.columns:
            out_cols.append("Auction")
        if "is_match_row" in seq_df.columns:
            out_cols.append("is_match_row")
        if "Expr" in seq_df.columns:
            out_cols.append("Expr")
        # Include all Agg_Expr_Seat columns
        for col in agg_expr_cols:
            if col in seq_df.columns:
                out_cols.append(col)
        # Include rules pipeline variation columns for seat 1
        for col in ("Agg_Expr_S1_Base", "Agg_Expr_S1_Learned", "Agg_Expr_S1_Full",
                    "Merged_Rules_Seat_1", "Agg_Expr_Seat_1_Rules"):
            if col in seq_df.columns:
                out_cols.append(col)
        
        seq_df = seq_df.select([c for c in out_cols if c in seq_df.columns])
        if index_col == "bt_index":
            seq_df = seq_df.rename({"bt_index": "index"})

        out_samples.append({"auction": matched_auction, "sequence": seq_df.to_dicts()})

    # Expand each sample to 4 seat variants when allow_initial_passes is True
    total_matching = filtered_df.height
    if allow_initial_passes:
        expanded_samples = []
        for sample in out_samples:
            for num_passes in range(4):
                prefix = "p-" * num_passes
                opener_seat = num_passes + 1
                
                # Create expanded sample with prefixed auctions
                expanded_sample = {
                    "auction": prefix + sample["auction"] if sample["auction"] else sample["auction"],
                    "opener_seat": opener_seat,
                    "sequence": [],
                }
                
                # Prefix all auctions in the sequence and rotate Agg_Expr columns
                for seq_row in sample["sequence"]:
                    new_row = dict(seq_row)
                    if "Auction" in new_row and new_row["Auction"]:
                        new_row["Auction"] = prefix + str(new_row["Auction"])
                    
                    # Rotate Agg_Expr_Seat columns
                    for display_seat in range(1, 5):
                        original_seat = ((display_seat - 1 - num_passes) % 4) + 1
                        orig_col = f"Agg_Expr_Seat_{original_seat}"
                        display_col = f"Agg_Expr_Seat_{display_seat}"
                        if orig_col in seq_row:
                            new_row[display_col] = seq_row[orig_col]
                    
                    expanded_sample["sequence"].append(new_row)
                
                expanded_samples.append(expanded_sample)
        
        out_samples = expanded_samples
        total_matching = total_matching * 4

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[auction-sequences-matching] {format_elapsed(elapsed_ms)} ({len(out_samples)} samples)")
    return {
        "pattern": pattern,
        "samples": out_samples,
        "total_matching": total_matching,
        "rejected_count": 0,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_auction_sequences_by_index(
    state: Dict[str, Any],
    indices: List[int],
    allow_initial_passes: bool = True,
) -> Dict[str, Any]:
    """Handle /auction-sequences-by-index endpoint.

    Fetches sequences for the provided bt_index values (order-preserving) and returns
    the same output shape as /auction-sequences-matching.
    """
    t0 = time.perf_counter()

    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")

    # Hard fail if is_completed_auction is missing
    if "is_completed_auction" not in bt_seat1_df.columns:
        raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")

    if "bt_index" not in bt_seat1_df.columns:
        raise ValueError("REQUIRED column 'bt_index' missing from bt_seat1_df. Pipeline error.")

    index_col = "bt_index"
    prev_idx_col = "previous_bid_indices"
    if prev_idx_col not in bt_seat1_df.columns:
        raise ValueError("REQUIRED column 'previous_bid_indices' missing from bt_seat1_df. Pipeline error.")

    # Normalize/limit indices
    requested: list[int] = []
    seen: set[int] = set()
    for x in (indices or []):
        try:
            xi = int(x)
        except (ValueError, TypeError):
            continue
        if xi in seen:
            continue
        requested.append(xi)
        seen.add(xi)
    requested = requested[:200]

    if not requested:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "indices": [],
            "samples": [],
            "total_matching": 0,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    # Load matched rows and keep original order
    order_df = pl.DataFrame({index_col: requested, "_ord": list(range(len(requested)))})
    base_df = bt_seat1_df.filter(pl.col("is_completed_auction"))
    sampled_df = (
        base_df.join(order_df, on=index_col, how="inner")
        .sort("_ord")
        .drop("_ord")
    )

    if sampled_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "indices": requested,
            "missing_indices": requested,
            "samples": [],
            "total_matching": 0,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    found_indices = sampled_df.get_column(index_col).to_list() if index_col in sampled_df.columns else []
    found_set = set(int(x) for x in found_indices if x is not None)
    missing_indices = [x for x in requested if x not in found_set]

    agg_expr_cols = [f"Agg_Expr_Seat_{i}" for i in range(1, 5)]
    extra_cols = ["Expr"] + agg_expr_cols
    display_cols = [index_col, "Auction"]
    lookup_cols = display_cols + [c for c in extra_cols if c in base_df.columns]
    available_display_cols = [c for c in lookup_cols if c in base_df.columns]

    # Get previous bid indices for sequence building
    all_prev_indices = set()
    for prev_list in sampled_df[prev_idx_col].to_list():
        if prev_list:
            all_prev_indices.update(prev_list)

    if all_prev_indices:
        prev_rows_df = bt_seat1_df.filter(pl.col(index_col).is_in(list(all_prev_indices))).select(
            [c for c in available_display_cols if c in bt_seat1_df.columns]
        )
        prev_rows_lookup = {row[index_col]: row for row in prev_rows_df.iter_rows(named=True)}
    else:
        prev_rows_lookup = {}

    out_samples: List[Dict[str, Any]] = []
    overlay = state.get("custom_criteria_overlay") or []

    for row in sampled_df.iter_rows(named=True):
        prev_indices = row.get(prev_idx_col, [])

        sequence_data: List[Dict[str, Any]] = []
        if prev_indices:
            for idx in prev_indices:
                if idx in prev_rows_lookup:
                    sequence_data.append(_apply_all_rules_to_bt_row(dict(prev_rows_lookup[idx]), state))
        sequence_data.append(
            _apply_all_rules_to_bt_row({c: row[c] for c in available_display_cols if c in row}, state)
        )

        seq_df = pl.DataFrame(sequence_data)
        if index_col in seq_df.columns:
            seq_df = seq_df.sort(index_col)

        # Mark the final (matched) row explicitly so UI can't confuse it with prefix rows.
        if seq_df.height > 0:
            seq_df = seq_df.with_row_index("_seq_pos").with_columns(
                (pl.col("_seq_pos") == (pl.len() - 1)).alias("is_match_row")
            ).drop("_seq_pos")
            matched_auction = seq_df.select("Auction").tail(1).item()
        else:
            matched_auction = row.get("Auction")

        out_cols = []
        if index_col in seq_df.columns:
            out_cols.append(index_col)
        if "Auction" in seq_df.columns:
            out_cols.append("Auction")
        if "is_match_row" in seq_df.columns:
            out_cols.append("is_match_row")
        if "Expr" in seq_df.columns:
            out_cols.append("Expr")
        for col in agg_expr_cols:
            if col in seq_df.columns:
                out_cols.append(col)
        for col in ("Merged_Rules_Seat_1", "Agg_Expr_Seat_1_Rules"):
            if col in seq_df.columns:
                out_cols.append(col)

        seq_df = seq_df.select(out_cols)
        if index_col == "bt_index":
            seq_df = seq_df.rename({"bt_index": "index"})

        out_samples.append({"auction": matched_auction, "sequence": seq_df.to_dicts()})

    # Expand each sample to 4 seat variants when allow_initial_passes is True
    total_matching = len(out_samples)
    if allow_initial_passes:
        expanded_samples = []
        for sample in out_samples:
            for num_passes in range(4):
                prefix = "p-" * num_passes
                opener_seat = num_passes + 1

                expanded_sample = {
                    "auction": prefix + sample["auction"] if sample["auction"] else sample["auction"],
                    "opener_seat": opener_seat,
                    "sequence": [],
                }

                for seq_row in sample["sequence"]:
                    new_row = dict(seq_row)
                    if "Auction" in new_row and new_row["Auction"]:
                        new_row["Auction"] = prefix + str(new_row["Auction"])

                    for display_seat in range(1, 5):
                        original_seat = ((display_seat - 1 - num_passes) % 4) + 1
                        orig_col = f"Agg_Expr_Seat_{original_seat}"
                        display_col = f"Agg_Expr_Seat_{display_seat}"
                        if orig_col in seq_row:
                            new_row[display_col] = seq_row[orig_col]

                    expanded_sample["sequence"].append(new_row)

                expanded_samples.append(expanded_sample)

        out_samples = expanded_samples
        total_matching = total_matching * 4

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[auction-sequences-by-index] {format_elapsed(elapsed_ms)} ({len(out_samples)} samples)")
    return {
        "indices": requested,
        "missing_indices": missing_indices,
        "samples": out_samples,
        "total_matching": total_matching,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_deal_criteria_failures_batch(
    state: Dict[str, Any],
    deal_row_idx: int,
    dealer: str,
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Handle /deal-criteria-eval-batch.

    Uses bitmap DFs (deal_criteria_by_seat_dfs) to evaluate criteria per seat for a specific deal row index.
    Also supports dynamic suit-length comparisons like 'SL_S >= SL_H' by parsing Hand_{N/E/S/W}.
    """
    deal_df = state.get("deal_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    if deal_df is None:
        raise ValueError("deal_df not loaded")

    # Get deal row using positional index
    try:
        row = deal_df.row(int(deal_row_idx), named=True)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid deal_row_idx {deal_row_idx}: {e}")

    dealer = str(dealer or "N").upper()

    results: List[Dict[str, Any]] = []
    for chk in (checks or []):
        seat = int(chk.get("seat") or 0)
        criteria_list = chk.get("criteria") or []
        failed: List[str] = []
        untracked: List[str] = []

        if seat < 1 or seat > 4:
            results.append({"seat": seat, "passed": [], "failed": failed, "untracked": untracked})
            continue

        criteria_df = deal_criteria_by_seat_dfs.get(seat, {}).get(dealer)
        available_cols = set(criteria_df.columns) if criteria_df is not None else set()

        # Evaluate each criterion
        crit_norm: List[str] = []
        for crit in criteria_list:
            if crit is None:
                continue
            crit_s = str(crit)
            crit_norm.append(crit_s)
            
            # Try dynamic SL evaluation first (uses shared helpers)
            # Use fail_on_missing=False so "can't evaluate" becomes "untracked" for UI display
            sl_result = evaluate_sl_criterion(crit_s, dealer, seat, row, fail_on_missing=False)
            if sl_result is True:
                continue  # Passed
            elif sl_result is False:
                failed.append(crit_s)
                continue
            # sl_result is None - either not an SL criterion OR can't evaluate
            # Check if it was an SL criterion that couldn't be evaluated
            if parse_sl_comparison_relative(crit_s) is not None or parse_sl_comparison_numeric(crit_s) is not None:
                untracked.append(crit_s)
                continue
            
            # Fall back to bitmap lookup for other criteria
            if criteria_df is not None and crit_s in available_cols:
                try:
                    if not bool(criteria_df[crit_s][int(deal_row_idx)]):
                        failed.append(crit_s)
                except (IndexError, KeyError):
                    untracked.append(crit_s)
                continue

            # Unknown / untracked criterion
            untracked.append(crit_s)

        # Passed = evaluated successfully and did not fail
        failed_set = set(failed)
        untracked_set = set(untracked)
        passed = [c for c in crit_norm if c not in failed_set and c not in untracked_set]

        # Annotate failed criteria with actual values (e.g., 'HCP <= 11' -> 'HCP(10) <= 11')
        failed_annotated = [
            annotate_criterion_with_value(c, dealer, seat, row) for c in failed
        ]

        results.append({
            "seat": seat, 
            "seat_dir": seat_to_direction(dealer, seat),
            "seat_label": _format_seat_notation(dealer, seat, include_bt_seat=False),
            "passed": passed, 
            "failed": failed_annotated, 
            "untracked": untracked
        })

    return {
        "deal_row_idx": int(deal_row_idx), 
        "dealer": dealer,
        "results": results
    }


# ---------------------------------------------------------------------------
# Handler: /pbn-sample
# ---------------------------------------------------------------------------


def handle_pbn_sample(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle /pbn-sample endpoint."""
    import random
    t0 = time.perf_counter()
    deal_df = state["deal_df"]
    
    if deal_df.height == 0:
        raise ValueError("No deals found in dataset")
    
    first_row = deal_df.row(0, named=True)
    
    dealer = first_row.get('Dealer', 'N')
    directions = ['N', 'E', 'S', 'W']
    dealer_idx = directions.index(dealer) if dealer in directions else 0
    
    hands = []
    for i in range(4):
        d = directions[(dealer_idx + i) % 4]
        hand = first_row.get(f'Hand_{d}', '')
        hands.append(hand)
    
    pbn = f"{dealer}:" + " ".join(hands)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "pbn": pbn,
        "dealer": dealer,
        "elapsed_ms": round(elapsed_ms, 1),
        "sql_query": "SELECT * FROM deals LIMIT 1",
    }


# ---------------------------------------------------------------------------
# Handler: /pbn-random
# ---------------------------------------------------------------------------


def handle_pbn_random(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle /pbn-random endpoint."""
    import random
    t0 = time.perf_counter()
    deal_df = state["deal_df"]
    
    if deal_df.height == 0:
        raise ValueError("No deals found in dataset")
    
    random_idx = random.randint(0, deal_df.height - 1)
    random_row = deal_df.row(random_idx, named=True)
    
    dealer = random_row.get('Dealer', 'N')
    directions = ['N', 'E', 'S', 'W']
    dealer_idx = directions.index(dealer) if dealer in directions else 0
    
    hands = []
    for i in range(4):
        d = directions[(dealer_idx + i) % 4]
        hand = random_row.get(f'Hand_{d}', '')
        hands.append(hand)
    
    pbn = f"{dealer}:" + " ".join(hands)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "pbn": pbn,
        "dealer": dealer,
        "row_idx": random_idx,
        "elapsed_ms": round(elapsed_ms, 1),
        "sql_query": f"SELECT * FROM deals WHERE index = {random_idx}",
    }


# ---------------------------------------------------------------------------
# Handler: /pbn-lookup
# ---------------------------------------------------------------------------


def handle_pbn_lookup(
    state: Dict[str, Any],
    pbn: str,
    max_results: int,
) -> Dict[str, Any]:
    """Handle /pbn-lookup endpoint."""
    t0 = time.perf_counter()
    deal_df = state["deal_df"]
    bt_seat1_df = state.get("bt_seat1_df")
    
    pbn_input = pbn.strip()
    
    parsed = parse_pbn_deal(pbn_input)
    if not parsed:
        raise ValueError(f"Invalid PBN format: {pbn_input[:100]}")
    
    match_criteria = pl.lit(True)
    sql_parts = []
    for direction in 'NESW':
        hand_col = f'Hand_{direction}'
        if hand_col in parsed and hand_col in deal_df.columns:
            val = parsed[hand_col]
            match_criteria = match_criteria & (pl.col(hand_col) == val)
            sql_parts.append(f"{hand_col} = '{val}'")
    
    sql_query = f"SELECT * FROM deals WHERE {' AND '.join(sql_parts)} LIMIT {max_results}"

    matching = deal_df.filter(match_criteria)
    
    if matching.height > max_results:
        matching = matching.head(max_results)
    
    # Rename 'bid' to 'Actual_Auction' in output
    matches_list = matching.to_dicts()
    for m in matches_list:
        if "bid" in m:
            m["Actual_Auction"] = _bid_value_to_str(m.pop("bid"))
    
    # Find matched BT auctions for this PBN (on-the-fly)
    matched_bt_auctions: List[str] = []
    if bt_seat1_df is not None:
        try:
            from bbo_bidding_queries_lib import find_matching_bt_auctions_from_pbn
            # Get completed auctions
            if "is_completed_auction" in bt_seat1_df.columns:
                bt_completed = bt_seat1_df.filter(pl.col("is_completed_auction"))
            else:
                bt_completed = bt_seat1_df
            # Limit for performance
            if "matching_deal_count" in bt_completed.columns:
                bt_completed = bt_completed.sort("matching_deal_count", descending=True).head(2000)
            else:
                bt_completed = bt_completed.head(2000)
            
            matched_bt_auctions = find_matching_bt_auctions_from_pbn(
                pbn_input, bt_completed, max_matches=10
            )
        except Exception as e:
            # Non-fatal: this endpoint still returns deal matches even if we fail to suggest BT auctions.
            print(f"[pbn-lookup] Warning: Could not find matched BT auctions: {type(e).__name__}: {e}")
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[pbn-lookup] Found {matching.height} matches, {len(matched_bt_auctions)} BT auctions in {format_elapsed(elapsed_ms)}")
    
    return {
        "matches": matches_list,
        "count": matching.height,
        "total_in_df": deal_df.height,
        "pbn_searched": pbn_input,
        "matched_bt_auctions": matched_bt_auctions,
        "elapsed_ms": round(elapsed_ms, 1),
        "sql_query": sql_query,
    }


# ---------------------------------------------------------------------------
# Handler: /deals-matching-auction
# ---------------------------------------------------------------------------


def handle_deals_matching_auction(
    state: Dict[str, Any],
    pattern: str,
    n_auction_samples: int,
    n_deal_samples: int,
    seed: Optional[int],
    dist_pattern: Optional[str],
    sorted_shape: Optional[str],
    dist_direction: str,
    apply_auction_criteria_fn,
    allow_initial_passes: bool = True,
    wrong_bid_filter: str = "all",
) -> Dict[str, Any]:
    """Handle /deals-matching-auction endpoint.
    
    Uses bt_seat1_df which contains only seat 1 auctions (no p- prefix complications).
    
    Args:
        wrong_bid_filter: "all" (default), "no_wrong" (only conforming bids), 
                          "only_wrong" (only non-conforming bids)
    """
    t0 = time.perf_counter()
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state["deal_criteria_by_seat_dfs"]
    
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")

    # Hard fail if is_completed_auction is missing
    if "is_completed_auction" not in bt_seat1_df.columns:
        raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")

    base_df = bt_seat1_df
    base_df = base_df.filter(pl.col("is_completed_auction"))

    if allow_initial_passes:
        pattern = _normalize_to_seat1(pattern)
        auction_expr = pl.col("Auction").cast(pl.Utf8).str.replace(r"(?i)^(p-)+", "")
    else:
        pattern = normalize_auction_user_text(pattern)
        auction_expr = pl.col("Auction").cast(pl.Utf8)

    regex_pattern = f"(?i){pattern}"
    filtered_df = base_df.filter(auction_expr.str.contains(regex_pattern))

    # NOTE: This step is BT-only (no dealer/hand context), so we do not apply CSV rules as a filter.
    # CSV rules are applied as an overlay onto Agg_Expr_Seat_* and will affect criteria mask building below.
    rejected_df = None

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[deals-matching-auction] {format_elapsed(elapsed_ms)} (no matches)")
        return {"pattern": pattern, "auctions": [], "elapsed_ms": round(elapsed_ms, 1)}

    sample_n = min(n_auction_samples, filtered_df.height)
    effective_seed = _effective_seed(seed)
    sampled_auctions = filtered_df.sample(n=sample_n, seed=effective_seed)

    deal_display_cols = [
        "index", "Dealer", "Vul", "Actual_Auction", "Contract", "Hand_N", "Hand_E", "Hand_S", "Hand_W",
        "Declarer", "Result", "Tricks", "Score", "DD_Score_Declarer", "EV_Score_Declarer",
        "EV_Declarer", "ParScore", "ParContracts", "EV_ParContracts",
    ]

    out_auctions: List[Dict[str, Any]] = []
    overlay = state.get("custom_criteria_overlay") or []
    
    # Initialize loop variables for type checker
    auction: str = ""
    auction_info: Dict[str, Any] = {}
    combined_df: pl.DataFrame = pl.DataFrame()

    for auction_row_raw in sampled_auctions.iter_rows(named=True):
        # Canonical: enrich Agg_Expr if missing + apply overlay + dedupe
        auction_row = _apply_all_rules_to_bt_row(dict(auction_row_raw), state)

        auction = auction_row["Auction"]

        auction_info = {
            "bt_index": auction_row.get("bt_index"),
            "auction": auction,
            "expr": auction_row.get("Expr"),
            "criteria_by_seat": {},
            "deals": [],
            "criteria_debug": {},
            "sql_query": f"-- Logic for auction: {auction}\n-- 1. Get criteria from auctions table\n-- 2. SELECT * FROM deals WHERE [Criteria Matches]\n-- (Criteria matching uses pre-computed bitmaps for performance)",
        }

        for s in range(1, 5):
            agg_col = f"Agg_Expr_Seat_{s}"
            if agg_col in auction_row:
                crit_list = auction_row[agg_col]
                if crit_list:
                    auction_info["criteria_by_seat"][str(s)] = crit_list

        # We'll sample *across all dealers* (Dealer column), not per-dealer.
        # The prior implementation sampled up to n_deal_samples per dealer and then
        # truncated, which biased results toward the first dealer and could make
        # some metrics look "stuck" at specific sample sizes.
        candidate_indices: List[int] = []
        total_matching_count = 0
        criteria_found: Dict[str, List[str]] = {}
        criteria_missing: Dict[str, List[str]] = {}
        
        # Compute final seat from the auction length (not from "has criteria"), so
        # complete auctions like "1S-p-p-p" correctly report Seat_4 in debug.
        try:
            bids = [b for b in str(auction).split("-") if b != ""]
            actual_final_seat = ((len(bids) - 1) % 4) + 1 if bids else 1
        except (ValueError, AttributeError):
            actual_final_seat = 1
        
        # Build found/missing debug for all seats up to final seat (inclusive).
        for s in range(1, int(actual_final_seat) + 1):
            agg_col = f"Agg_Expr_Seat_{s}"
            seat_key = f"Seat_{s}"
            criteria_found[seat_key] = []
            criteria_missing[seat_key] = []
            
            if agg_col in auction_row:
                criteria_list = auction_row[agg_col]
                if criteria_list:
                    for dealer in DIRECTIONS:
                        if s in deal_criteria_by_seat_dfs and dealer in deal_criteria_by_seat_dfs[s]:
                            seat_criteria_df = deal_criteria_by_seat_dfs[s][dealer]
                            for criterion in criteria_list:
                                # Treat suit-length comparisons as "found" because we evaluate them dynamically.
                                c = str(criterion).strip().replace("≥", ">=").replace("≤", "<=")
                                is_dyn_sl = bool(re.match(r"^SL_[SHDC]\s*(>=|<=|>|<|==|!=)\s*SL_[SHDC]$", c))
                                if is_dyn_sl or (criterion in seat_criteria_df.columns):
                                    if criterion not in criteria_found[seat_key]:
                                        criteria_found[seat_key].append(criterion)
                                else:
                                    if criterion not in criteria_missing[seat_key]:
                                        criteria_missing[seat_key].append(criterion)
                            break

        # Pre-lookup criteria for the helper
        criteria_for_helper = {int(s): cl for s, cl in auction_info["criteria_by_seat"].items()}
        
        for dealer in DIRECTIONS:
            dealer_mask, _ = _build_criteria_mask_for_dealer(
                deal_df, dealer, criteria_for_helper, deal_criteria_by_seat_dfs
            )
            
            if dealer_mask is None:
                continue

            matching_idx = dealer_mask.arg_true()
            if len(matching_idx) > 0:
                total_matching_count += len(matching_idx)
                # Take up to n_deal_samples candidates from this dealer, then do a final
                # global sample across all dealers.
                if len(matching_idx) > n_deal_samples:
                    rng = random.Random(effective_seed)
                    sampled_indices = rng.sample(list(matching_idx), n_deal_samples)
                else:
                    sampled_indices = list(matching_idx)
                candidate_indices.extend(int(i) for i in sampled_indices)
        
        auction_info["criteria_debug"] = {
            "actual_final_seat": actual_final_seat,
            "found": criteria_found,
            "missing": {k: v for k, v in criteria_missing.items() if v},
        }
        auction_info["total_matching_deals"] = total_matching_count

        if candidate_indices:
            # Final sample across all dealers (stable if seed != 0, non-deterministic if seed==0)
            if len(candidate_indices) > n_deal_samples:
                rng = random.Random(effective_seed)
                final_indices = rng.sample(candidate_indices, n_deal_samples)
            else:
                final_indices = candidate_indices

            combined_df = _take_rows_by_index(deal_df, final_indices)
            # Add row indices for bitmap lookups (wrong_bid checking)
            combined_df = combined_df.with_columns(
                pl.Series("_row_idx", final_indices)
            )
        else:
            # Important: Reset combined_df for iterations where no candidates are found
            combined_df = pl.DataFrame()
            
            if "Dealer" in combined_df.columns:
                combined_df = combined_df.with_columns(pl.col("Dealer").alias("Opener_Direction"))
            
        if dist_pattern or sorted_shape:
            direction = dist_direction.upper()
            if direction in 'NESW':
                combined_df = add_suit_length_columns(combined_df, direction)
                dist_where = build_distribution_sql_for_deals(dist_pattern, sorted_shape, direction)
                if dist_where:
                    try:
                        dist_sql = f"SELECT * FROM combined_df WHERE {dist_where}"
                        conn = state.get("duckdb_conn") or duckdb
                        combined_df = conn.sql(dist_sql).pl()
                        auction_info["dist_sql_query"] = dist_sql
                    except Exception as e:
                        print(f"[deals-matching-auction] Distribution filter error: {e}")
                sl_cols = [f"SL_{s}_{direction}" for s in ['S', 'H', 'D', 'C']]
                combined_df = combined_df.drop([c for c in sl_cols if c in combined_df.columns])
        
        # Build bt_info from auction_row for wrong_bid checking
        bt_info_for_check: Dict[str, Any] = {}
        for s in range(1, 5):
            agg_col = f"Agg_Expr_Seat_{s}"
            if agg_col in auction_row and auction_row[agg_col]:
                bt_info_for_check[agg_col] = auction_row[agg_col]
        
        deals_list = combined_df.to_dicts()
        wrong_bid_count = 0
        for deal_row in deals_list:
            # Rename 'bid' column to 'Actual_Auction'
            if "bid" in deal_row:
                deal_row["Actual_Auction"] = _bid_value_to_str(deal_row.pop("bid"))
            
            dealer = deal_row.get("Dealer", "N")
            deal_row["Rules_Contract"] = get_ai_contract(auction, dealer)
            dd_score_rules = get_dd_score_for_auction(auction, dealer, deal_row)
            deal_row["DD_Score_Rules"] = dd_score_rules
            ev_rules = get_ev_for_auction(auction, dealer, deal_row)
            deal_row["EV_Rules"] = ev_rules
            
            # Check for wrong bids using bitmap lookups
            deal_idx = deal_row.get("_row_idx")
            if deal_idx is not None and bt_info_for_check:
                conformance = _check_deal_criteria_conformance_bitmap(
                    int(deal_idx), bt_info_for_check, dealer, deal_criteria_by_seat_dfs,
                    deal_row=deal_row,
                    auction=auction,
                )
                # Add per-seat wrong bid columns
                for seat in range(1, 5):
                    deal_row[f"Wrong_Bid_S{seat}"] = conformance[f"Wrong_Bid_S{seat}"]
                    deal_row[f"Invalid_Criteria_S{seat}"] = conformance[f"Invalid_Criteria_S{seat}"]
                deal_row["first_wrong_seat"] = conformance["first_wrong_seat"]
                if conformance["first_wrong_seat"] is not None:
                    wrong_bid_count += 1
            else:
                # No bitmap info available - set defaults
                for seat in range(1, 5):
                    deal_row[f"Wrong_Bid_S{seat}"] = False
                    deal_row[f"Invalid_Criteria_S{seat}"] = None
                deal_row["first_wrong_seat"] = None
            # Keep _row_idx in output so clients can use it for /deal-criteria-eval-batch
            
            # Add requested EV fields
            # - EV_Declarer: EV for the deal's actual declarer/contract (already precomputed in deal_df)
            deal_row["EV_Declarer"] = deal_row.get("EV_Score_Declarer")
            # - _ap: list aligned to (de-duped) ParContracts
            deal_row["EV_ParContracts"] = _ev_list_for_par_contracts(deal_row)
            
            dd_score_actual = deal_row.get("DD_Score_Declarer")
            if dd_score_actual is not None and dd_score_rules is not None:
                score_diff = int(dd_score_rules) - int(dd_score_actual)
                imp_diff = calculate_imp(abs(score_diff))
                deal_row["IMP_Rules_vs_Actual"] = imp_diff if score_diff >= 0 else -imp_diff
            else:
                deal_row["IMP_Rules_vs_Actual"] = None
            
            # Normalize ParContracts display and ensure it matches EV_ParContracts de-duped ordering
            par_contracts_raw = deal_row.get("ParContracts")
            deal_row["ParContracts"] = _format_par_contracts(par_contracts_raw)
        
        # Add wrong_bid_count to auction_info
        auction_info["wrong_bid_count"] = wrong_bid_count
        
        # Apply wrong_bid_filter
        if wrong_bid_filter == "no_wrong":
            deals_list = [d for d in deals_list if d.get("first_wrong_seat") is None]
        elif wrong_bid_filter == "only_wrong":
            deals_list = [d for d in deals_list if d.get("first_wrong_seat") is not None]
        # else: "all" - no filtering
        
        auction_info["wrong_bid_filter"] = wrong_bid_filter
        auction_info["filtered_deal_count"] = len(deals_list)
        
        deals_with_computed = pl.DataFrame(deals_list) if deals_list else pl.DataFrame()
        if "Contract" in deals_with_computed.columns and "IMP_Rules_vs_Actual" in deals_with_computed.columns:
            agg_exprs = [
                pl.len().alias("Count"),
                pl.col("IMP_Rules_vs_Actual").mean().alias("Avg_IMP_Rules"),
                (pl.col("DD_Score_Declarer").cast(pl.Int64, strict=False).ge(0).sum() * 100.0 / pl.len()).alias("Contract_Made%"),
                (pl.col("DD_Score_Rules").cast(pl.Int64, strict=False).ge(0).sum() * 100.0 / pl.len()).alias("Rules_Made%"),
                (pl.col("DD_Score_Declarer").cast(pl.Int64, strict=False).eq(pl.col("ParScore").cast(pl.Int64, strict=False)).sum() * 100.0 / pl.len()).alias("Contract_Par%"),
                (pl.col("DD_Score_Rules").cast(pl.Int64, strict=False).eq(pl.col("ParScore").cast(pl.Int64, strict=False)).sum() * 100.0 / pl.len()).alias("Rules_Par%"),
            ]
            
            has_ev_contract = "EV_Score_Declarer" in deals_with_computed.columns
            has_ev_rules = "EV_Rules" in deals_with_computed.columns
            
            if has_ev_contract:
                agg_exprs.append(pl.col("EV_Score_Declarer").cast(pl.Float32, strict=False).mean().alias("Avg_EV_Contract"))
            if has_ev_rules:
                agg_exprs.append(pl.col("EV_Rules").cast(pl.Float32, strict=False).mean().alias("Avg_EV_Rules"))
            if has_ev_contract and has_ev_rules:
                agg_exprs.append(
                    (pl.col("EV_Rules").cast(pl.Float32, strict=False) - pl.col("EV_Score_Declarer").cast(pl.Float32, strict=False))
                    .mean().alias("Avg_EV_Diff")
                )
            
            contract_summary = deals_with_computed.group_by("Contract").agg(agg_exprs).sort("Count", descending=True)
            round_cols = [
                pl.col("Avg_IMP_Rules").round(1), pl.col("Contract_Made%").round(1),
                pl.col("Rules_Made%").round(1), pl.col("Contract_Par%").round(1), pl.col("Rules_Par%").round(1),
            ]
            if has_ev_contract:
                round_cols.append(pl.col("Avg_EV_Contract").round(2))
            if has_ev_rules:
                round_cols.append(pl.col("Avg_EV_Rules").round(2))
            if has_ev_contract and has_ev_rules:
                round_cols.append(pl.col("Avg_EV_Diff").round(2))
            contract_summary = contract_summary.with_columns(round_cols)
            auction_info["contract_summary"] = contract_summary.to_dicts()
            
            total_deals = deals_with_computed.height
            auction_info["deals_used"] = int(total_deals)
            imp_values = deals_with_computed["IMP_Rules_vs_Actual"].cast(pl.Int64, strict=False)
            total_imp = imp_values.sum()
            auction_info["total_imp_rules"] = int(total_imp) if total_imp is not None else 0
            auction_info["total_deals"] = total_deals
            
            imp_rules_wins = imp_values.filter(imp_values > 0).sum()
            imp_actual_wins = (-imp_values.filter(imp_values < 0)).sum()
            auction_info["imp_rules_advantage"] = int(imp_rules_wins) if imp_rules_wins is not None else 0
            auction_info["imp_actual_advantage"] = int(imp_actual_wins) if imp_actual_wins is not None else 0
            
            dd_actual = deals_with_computed["DD_Score_Declarer"].cast(pl.Int64, strict=False)
            dd_rules = deals_with_computed["DD_Score_Rules"].cast(pl.Int64, strict=False)
            par_score_col = deals_with_computed["ParScore"].cast(pl.Int64, strict=False)
            
            # Polars stubs sometimes type these aggregations as Expr; cast via Any for type-checkers.
            auction_info["contract_makes_count"] = int(cast(Any, (dd_actual >= 0).sum()))
            auction_info["rules_makes_count"] = int(cast(Any, (dd_rules >= 0).sum()))
            auction_info["contract_par_count"] = int(cast(Any, (dd_actual == par_score_col).sum()))
            auction_info["rules_par_count"] = int(cast(Any, (dd_rules == par_score_col).sum()))
            
            if has_ev_contract:
                ev_contract_mean = deals_with_computed["EV_Score_Declarer"].cast(pl.Float32, strict=False).mean()
                ev_contract_f = _safe_float(ev_contract_mean)
                auction_info["avg_ev_contract"] = round(ev_contract_f, 2) if ev_contract_f is not None else None
            if has_ev_rules:
                ev_rules_mean = deals_with_computed["EV_Rules"].cast(pl.Float32, strict=False).mean()
                ev_rules_f = _safe_float(ev_rules_mean)
                auction_info["avg_ev_rules"] = round(ev_rules_f, 2) if ev_rules_f is not None else None
            if has_ev_contract and has_ev_rules:
                ev_diff_mean = (deals_with_computed["EV_Rules"].cast(pl.Float32, strict=False) - 
                                deals_with_computed["EV_Score_Declarer"].cast(pl.Float32, strict=False)).mean()
                ev_diff_f = _safe_float(ev_diff_mean)
                auction_info["avg_ev_diff"] = round(ev_diff_f, 2) if ev_diff_f is not None else None
        
        display_cols_set = set(deal_display_cols) | {"DD_Score_Rules", "EV_Rules", "Rules_Contract", "IMP_Rules_vs_Actual"}
        auction_info["deals"] = [{k: v for k, v in d.items() if k in display_cols_set} for d in deals_list]

        out_auctions.append(auction_info)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_deals_count = sum(len(a.get("deals", [])) for a in out_auctions)
    print(f"[deals-matching-auction] {format_elapsed(elapsed_ms)} ({len(out_auctions)} auctions, {total_deals_count} deals)")
    
    response: Dict[str, Any] = {"pattern": pattern, "auctions": out_auctions, "elapsed_ms": round(elapsed_ms, 1)}
    if dist_pattern or sorted_shape:
        response["dist_filter"] = {"dist_pattern": dist_pattern, "sorted_shape": sorted_shape, "direction": dist_direction}
    if wrong_bid_filter != "all":
        response["wrong_bid_filter"] = wrong_bid_filter
    return response


def handle_sample_deals_by_auction_pattern(
    state: Dict[str, Any],
    pattern: str,
    sample_size: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Return a small sample of deals whose *actual auction* matches regex `pattern`.

    This intentionally does NOT run any BT / Rules logic. It is used by Streamlit's Auction Builder
    to power "Show Matching Deals" quickly without the heavy /bidding-arena path.
    """
    t0 = time.perf_counter()

    deal_df = state.get("deal_df")
    if not isinstance(deal_df, pl.DataFrame) or deal_df.is_empty():
        raise ValueError("deal_df not loaded")

    # Pick an auction string column if present; otherwise compute _bid_str on demand.
    df = deal_df
    auction_col: str | None = None
    if "Actual_Auction" in df.columns:
        auction_col = "Actual_Auction"
    elif "_bid_str" in df.columns:
        auction_col = "_bid_str"

    if auction_col is None:
        df = prepare_deals_with_bid_str(df, include_auction_key=False)
        auction_col = "_bid_str"

    try:
        regex = f"(?i){pattern}"
        # Polars str.contains() with regex should respect ^ and $ anchors
        # For exact matches (^...$), str.contains() will match the full string
        # For prefix matches (^...), str.contains() will match from the start
        filtered = df.filter(pl.col(auction_col).cast(pl.Utf8).str.contains(regex))
    except Exception as e:
        raise ValueError(f"Invalid pattern: {e}")

    total_count = filtered.height
    if filtered.is_empty():
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[sample-deals-by-auction-pattern] {format_elapsed(elapsed_ms)} (0 deals)")
        return {"pattern": pattern, "deals": [], "total_count": 0, "elapsed_ms": round(elapsed_ms, 1)}

    n = int(sample_size) if sample_size is not None else 25
    n = 1 if n < 1 else (500 if n > 500 else n)
    n = min(n, total_count)

    sampled = filtered.sample(n=n, seed=_effective_seed(seed)) if n < total_count else filtered

    cols = [
        "index",
        "Dealer",
        "Vul",
        "Hand_N",
        "Hand_E",
        "Hand_S",
        "Hand_W",
        "Contract",
        "Declarer",
        "Result",
        "Tricks",
        "Score",
        "ParScore",
    ]
    if auction_col not in cols:
        cols.append(auction_col)
    cols = [c for c in cols if c in sampled.columns]
    out_rows = sampled.select(cols).to_dicts()
    for r in out_rows:
        if "Auction_Actual" not in r and auction_col in r:
            r["Auction_Actual"] = r.get(auction_col)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[sample-deals-by-auction-pattern] {format_elapsed(elapsed_ms)} ({len(out_rows)}/{total_count} deals)")
    return {"pattern": pattern, "deals": out_rows, "total_count": total_count, "elapsed_ms": round(elapsed_ms, 1)}


def handle_auction_pattern_counts(
    state: Dict[str, Any],
    patterns: List[str],
) -> Dict[str, Any]:
    """Return counts for multiple *actual auction* regex patterns.

    This is used by Streamlit Auction Summary to get per-step "Matches" counts in one HTTP call.
    """
    t0 = time.perf_counter()

    deal_df = state.get("deal_df")
    if not isinstance(deal_df, pl.DataFrame) or deal_df.is_empty():
        raise ValueError("deal_df not loaded")

    # Pick an auction string column if present; otherwise compute _bid_str on demand.
    df = deal_df
    auction_col: str | None = None
    if "Actual_Auction" in df.columns:
        auction_col = "Actual_Auction"
    elif "_bid_str" in df.columns:
        auction_col = "_bid_str"

    if auction_col is None:
        df = prepare_deals_with_bid_str(df, include_auction_key=False)
        auction_col = "_bid_str"

    # Deduplicate patterns but keep stable output mapping
    patterns_in = [str(p) for p in (patterns or []) if p is not None]
    unique_patterns: List[str] = list(dict.fromkeys(patterns_in))

    counts_by_pattern: Dict[str, int] = {}
    errors_by_pattern: Dict[str, str] = {}

    if not unique_patterns:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[auction-pattern-counts] {format_elapsed(elapsed_ms)} (0 patterns)")
        return {"counts": {}, "errors": {}, "elapsed_ms": round(elapsed_ms, 1)}

    # Compute counts in one Polars select (avoids N HTTP calls; still N regex evals)
    exprs: List[pl.Expr] = []
    for pat in unique_patterns:
        try:
            regex = f"(?i){pat}"
            exprs.append(
                pl.col(auction_col)
                .cast(pl.Utf8)
                .str.contains(regex)
                .cast(pl.Int32)
                .sum()
                .alias(pat)
            )
        except Exception as e:
            errors_by_pattern[pat] = str(e)

    if exprs:
        try:
            out = df.select(exprs).to_dicts()[0]
            for k, v in out.items():
                try:
                    counts_by_pattern[str(k)] = int(v) if v is not None else 0
                except Exception:
                    counts_by_pattern[str(k)] = 0
        except Exception as e:
            # Fall back to per-pattern filtering (more robust)
            for pat in unique_patterns:
                if pat in errors_by_pattern:
                    continue
                try:
                    regex = f"(?i){pat}"
                    counts_by_pattern[pat] = int(df.filter(pl.col(auction_col).cast(pl.Utf8).str.contains(regex)).height)
                except Exception as ee:
                    errors_by_pattern[pat] = str(ee)
                    counts_by_pattern[pat] = 0

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[auction-pattern-counts] {format_elapsed(elapsed_ms)} ({len(unique_patterns)} patterns)")
    return {"counts": counts_by_pattern, "errors": errors_by_pattern, "elapsed_ms": round(elapsed_ms, 1)}


def handle_resolve_auction_path(
    state: Dict[str, Any],
    auction: str,
) -> Dict[str, Any]:
    """Resolve an entire auction path into detailed step info in one call.

    Uses the DuckDB fallback implementation for reliability.
    """
    t0 = time.perf_counter()
    return _handle_resolve_auction_path_fallback(state, auction, t0)


# ---------------------------------------------------------------------------
# Handler: /bidding-table-statistics
# ---------------------------------------------------------------------------


def handle_bidding_table_statistics(
    state: Dict[str, Any],
    auction_pattern: str,
    sample_size: int,
    min_matches: int,
    seed: Optional[int],
    dist_pattern: Optional[str],
    sorted_shape: Optional[str],
    dist_seat: int,
    allow_initial_passes: bool = True,
    include_categories: bool = False,
) -> Dict[str, Any]:
    """Handle /bidding-table-statistics endpoint.
    
    Uses bt_seat1_df (seat-1 auctions, no p- prefix complications) plus bt_stats_df
    (completed-auction criteria/aggregates keyed by bt_index).
    """
    t0 = time.perf_counter()
    
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")
    bt_stats_df = state.get("bt_stats_df")
    bt_categories_df = state.get("bt_categories_df")
    bt_category_cols = state.get("bt_category_cols") or []

    # New architecture: bt_seat1_df has only core bidding columns; criteria/aggregates
    # for completed auctions live in bt_stats_df, keyed by bt_index.
    # Hard fail if is_completed_auction is missing
    if "is_completed_auction" not in bt_seat1_df.columns:
        raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")

    base_df = bt_seat1_df
    base_df = base_df.filter(pl.col("is_completed_auction"))
    # Use bt_index as the index column when present (REQUIRED)
    if "bt_index" not in base_df.columns:
        raise ValueError("REQUIRED column 'bt_index' missing from bt_seat1_df. Pipeline error.")
    index_col = "bt_index"
    base_df = base_df.with_columns(pl.col(index_col).alias("_idx"))

    # Criteria/aggregates come from bt_stats_df, not bt_seat1_df.
    has_criteria = bool(bt_stats_df is not None and any(c.endswith("_min_S1") for c in bt_stats_df.columns))
    has_aggregates = bool(bt_stats_df is not None and "matching_deal_count" in bt_stats_df.columns)
    
    try:
        if allow_initial_passes:
            auction_pattern = _normalize_to_seat1(auction_pattern)
            auction_expr = pl.col("Auction").cast(pl.Utf8).str.replace(r"(?i)^(p-)+", "")
        else:
            auction_pattern = normalize_auction_user_text(auction_pattern)
            auction_expr = pl.col("Auction").cast(pl.Utf8)
        regex_pattern = f"(?i){auction_pattern}"
        matched_df = base_df.filter(auction_expr.str.contains(regex_pattern))
    except Exception as e:
        raise ValueError(f"Invalid regex pattern: {type(e).__name__}: {e}")
    
    if matched_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "pattern": auction_pattern, "total_matches": 0, "rows": [],
            "has_criteria": has_criteria, "has_aggregates": has_aggregates,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    # Attach criteria/aggregates for all matched rows (completed auctions only) via bt_stats_df.
    if bt_stats_df is not None and index_col == "bt_index" and "bt_index" in matched_df.columns:
        matched_df = matched_df.join(bt_stats_df, on="bt_index", how="left")

    # Attach categories if requested
    if include_categories and bt_categories_df is not None:
        matched_df = matched_df.join(bt_categories_df, on="bt_index", how="left")

    # Apply min_matches filter using matching_deal_count from bt_stats_df (if available).
    if min_matches > 0:
        if "matching_deal_count" not in matched_df.columns:
            raise ValueError("min_matches requested but 'matching_deal_count' not present in bt_stats_df")
        matched_df = matched_df.filter(pl.col("matching_deal_count") >= min_matches)
        if matched_df.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "pattern": auction_pattern, "total_matches": 0, "rows": [],
                "has_criteria": has_criteria, "has_aggregates": has_aggregates,
                "message": f"No matching auctions have >= {min_matches} matching deals",
                "elapsed_ms": round(elapsed_ms, 1),
            }
    
    total_matches = matched_df.height
    sample_n = min(sample_size, total_matches)
    effective_seed = None if (seed is None or seed == 0) else seed
    sampled_df = matched_df.sample(n=sample_n, seed=effective_seed).sort("_idx")

    # Optional: attach bid-category flags (Phase 4) on bt_index.
    # Join AFTER sampling for performance (keeps join size small).
    if (
        include_categories
        and bt_categories_df is not None
        and "bt_index" in sampled_df.columns
        and isinstance(bt_category_cols, list)
        and bt_category_cols
    ):
        try:
            cat_cols_present = [c for c in bt_category_cols if c in bt_categories_df.columns]
            if cat_cols_present:
                sampled_df = sampled_df.join(
                    bt_categories_df.select(["bt_index"] + cat_cols_present),
                    on="bt_index",
                    how="left",
                )
        except Exception:
            # Categories are optional; never fail the endpoint due to category join issues.
            pass
    sampled_indices = sampled_df["_idx"].to_list()
    
    # Start result_df with index + Auction, then join all other sampled columns.
    result_df = sampled_df.select(
        [
            pl.col("_idx").alias("original_idx"),
            pl.col("Auction"),
        ]
    ).with_row_index("row_idx")
    
    # Add all remaining columns (core bt_seat1 + criteria/aggregates) via join on original_idx.
    result_cols = set(result_df.columns)
    extra_cols = [c for c in sampled_df.columns if c not in ["_idx", "Auction", "row_idx"] and c not in result_cols]
    if extra_cols:
        extra_df = sampled_df.select([pl.col("_idx").alias("original_idx")] + [pl.col(c) for c in extra_cols])
        result_df = result_df.join(extra_df, on="original_idx", how="left")
    
    dist_sql_query = None
    if (dist_pattern or sorted_shape):
        dist_where = build_distribution_sql_for_bt(dist_pattern, sorted_shape, dist_seat, result_df.columns)
        if dist_where:
            dist_sql_query = f"SELECT * FROM result_df WHERE {dist_where}"
            try:
                conn = state.get("duckdb_conn") or duckdb
                result_df = conn.sql(dist_sql_query).pl()
            except Exception as e:
                print(f"[bidding-table-statistics] Distribution filter error: {e}")
    
    result_rows = result_df.to_dicts()
    # Canonical: enrich Agg_Expr if missing + apply overlay + dedupe
    result_rows = [_apply_all_rules_to_bt_row(r, state) for r in result_rows]
    
    # Expand each matched row to 4 seat variants when allow_initial_passes is True
    if allow_initial_passes:
        expanded_rows = []
        for row in result_rows:
            expanded_rows.extend(_expand_row_to_all_seats(row, allow_initial_passes=True))
        result_rows = expanded_rows
        # Update total_matches to reflect expanded count
        total_matches = total_matches * 4
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[bidding-table-statistics] {format_elapsed(elapsed_ms)} ({len(result_rows)} rows from {total_matches} matches)")
    
    final_sql_query = dist_sql_query if dist_sql_query else f"SELECT * FROM auctions WHERE Auction ~* '{regex_pattern}' LIMIT {sample_n}"

    return {
        "pattern": auction_pattern, "total_matches": total_matches, "sample_size": len(result_rows),
        "rows": result_rows, "has_criteria": has_criteria, "has_aggregates": has_aggregates,
        "dist_sql_query": dist_sql_query, "elapsed_ms": round(elapsed_ms, 1),
        "sql_query": final_sql_query,
    }


# ---------------------------------------------------------------------------
# Handler: /find-matching-auctions
# ---------------------------------------------------------------------------


def handle_find_matching_auctions(
    state: Dict[str, Any],
    hcp: int, sl_s: int, sl_h: int, sl_d: int, sl_c: int,
    total_points: int, seat: int, max_results: int,
    load_auction_criteria_fn,
    filter_auctions_by_hand_criteria_fn,
) -> Dict[str, Any]:
    """Handle /find-matching-auctions endpoint.
    
    Uses bt_seat1_df which has inlined criteria for all seats.
    The 'seat' parameter determines which seat's criteria columns to use for matching.
    """
    t0 = time.perf_counter()
    
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")
    bt_stats_df = state.get("bt_stats_df")
    if bt_stats_df is None:
        raise ValueError("bt_stats_df not loaded (pipeline error): missing bbo_bt_criteria/bbo_bt_aggregate parquet)")
    
    criteria_list = load_auction_criteria_fn()
    criteria_loaded = len(criteria_list)
    
    # Work against the compact stats table (completed auctions only).
    base_df = bt_stats_df
    
    # Build SQL conditions for matching hand criteria.
    # Seat-specific columns are REQUIRED.
    def _min_col(name: str, seat_num: int) -> str:
        c1 = f"{name}_min_S{seat_num}"
        if c1 in base_df.columns:
            return c1
        raise ValueError(f"REQUIRED criteria column missing: '{c1}' in auction_stats/bt_stats_df. Pipeline error.")

    def _max_col(name: str, seat_num: int) -> str:
        c1 = f"{name}_max_S{seat_num}"
        if c1 in base_df.columns:
            return c1
        raise ValueError(f"REQUIRED criteria column missing: '{c1}' in auction_stats/bt_stats_df. Pipeline error.")

    conditions = [
        f'"{_min_col("HCP", seat)}" <= {hcp} AND "{_max_col("HCP", seat)}" >= {hcp}',
        f'"{_min_col("SL_S", seat)}" <= {sl_s} AND "{_max_col("SL_S", seat)}" >= {sl_s}',
        f'"{_min_col("SL_H", seat)}" <= {sl_h} AND "{_max_col("SL_H", seat)}" >= {sl_h}',
        f'"{_min_col("SL_D", seat)}" <= {sl_d} AND "{_max_col("SL_D", seat)}" >= {sl_d}',
        f'"{_min_col("SL_C", seat)}" <= {sl_c} AND "{_max_col("SL_C", seat)}" >= {sl_c}',
        f'"{_min_col("Total_Points", seat)}" <= {total_points} AND "{_max_col("Total_Points", seat)}" >= {total_points}',
    ]
    joined_df = base_df
    
    where_clause = " AND ".join(conditions)
    
    # Query the compact stats table in DuckDB; this is much smaller than bt_seat1_df.
    sql_query = f"SELECT * FROM auction_stats WHERE {where_clause} LIMIT {max_results * 3}"
    try:
        conn = state.get("duckdb_conn") or duckdb
        matching_df = conn.sql(sql_query).pl()
    except Exception as e:
        raise ValueError(f"SQL error: {e}")
    
    hand_values = {'HCP': hcp, 'SL_S': sl_s, 'SL_H': sl_h, 'SL_D': sl_d, 'SL_C': sl_c, 'Total_Points': total_points}
    pre_criteria_count = matching_df.height
    matching_df, rejected_auctions = filter_auctions_by_hand_criteria_fn(matching_df, hand_values, seat)
    post_criteria_count = matching_df.height
    
    if matching_df.height > max_results:
        matching_df = matching_df.head(max_results)
    
    # Join back to bt_seat1_df to recover auction strings and Agg_Expr_Seat_* lists.
    if "bt_index" not in matching_df.columns:
        raise ValueError("bt_stats_df / auction_stats is expected to contain 'bt_index' column")
    
    join_df = matching_df.join(bt_seat1_df, on="bt_index", how="left", suffix="_bt")
    
    # Keep bt_index so we can enrich Agg_Expr on-demand in lightweight mode
    result_cols = ["bt_index", "Auction"]
    if "matching_deal_count" in join_df.columns:
        result_cols.append("matching_deal_count")
    for s in range(1, 5):
        agg_expr_col = f"Agg_Expr_Seat_{s}"
        if agg_expr_col in join_df.columns:
            result_cols.append(agg_expr_col)
    for s in range(1, 5):
        for col in sorted(join_df.columns):
            if f"_S{s}" in col and col not in result_cols:
                result_cols.append(col)
    result_cols = [c for c in result_cols if c in join_df.columns]
    result_df = join_df.select(result_cols)
    result_rows = result_df.to_dicts()
    # Canonical: enrich Agg_Expr if missing + apply overlay + dedupe
    result_rows = [_apply_all_rules_to_bt_row(r, state) for r in result_rows]
    # Display: prepend leading passes for the requested seat.
    for r in result_rows:
        if "Auction" in r:
            r["Auction"] = _display_auction_with_seat_prefix(r.get("Auction"), seat)
        if "Rules_Auction" in r:
            r["Rules_Auction"] = _display_auction_with_seat_prefix(r.get("Rules_Auction"), seat)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    criteria_filtered = pre_criteria_count - post_criteria_count
    print(f"[find-matching-auctions] {format_elapsed(elapsed_ms)} ({len(result_rows)} matches, {criteria_filtered} filtered by CSV criteria)")
    
    response: Dict[str, Any] = {
        "sql_query": sql_query, "auctions": result_rows, "total_matches": len(result_rows),
        "seat": seat, "criteria": {"HCP": hcp, "SL_S": sl_s, "SL_H": sl_h, "SL_D": sl_d, "SL_C": sl_c, "Total_Points": total_points},
        "auction_criteria_loaded": criteria_loaded, "auction_criteria_filtered": criteria_filtered,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if rejected_auctions:
        response["criteria_rejected"] = rejected_auctions[:10]
    return response


# ---------------------------------------------------------------------------
# Handler: /bt-seat-stats (on-the-fly stats for a single bt row)
# ---------------------------------------------------------------------------


def _compute_wrong_bid_rate_for_bt_row(
    state: Dict[str, Any],
    bt_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute wrong bid rate for deals that have this auction.
    
    Returns stats about how many deals with this auction have wrong bids (criteria failures).
    """
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    
    auction = bt_row.get("Auction", "")
    if not auction:
        return {"analyzed_deals": 0, "wrong_bid_count": 0, "wrong_bid_rate": 0.0, "by_seat": {}}
    
    # Find deals with this auction (including those with leading passes stripped)
    auction_lower = auction.lower()
    auction_variations = [auction_lower]
    # Also match with leading passes
    for prefix in ["p-", "p-p-", "p-p-p-"]:
        auction_variations.append(prefix + auction_lower)
    
    # Get bid column as string
    bid_dtype = deal_df.schema.get("bid")
    if bid_dtype == pl.List(pl.Utf8):
        deal_df_with_str = deal_df.with_columns(pl.col("bid").list.join("-").alias("_bid_str"))
    elif bid_dtype == pl.Utf8:
        deal_df_with_str = deal_df.with_columns(pl.col("bid").fill_null("").alias("_bid_str"))
    else:
        deal_df_with_str = deal_df.with_columns(
            pl.col("bid").map_elements(
                lambda x: "-".join(map(str, x)) if isinstance(x, list) else (str(x) if x is not None else ""),
                return_dtype=pl.Utf8,
            ).alias("_bid_str")
        )
    
    # Add row index for bitmap lookups
    deal_df_with_str = deal_df_with_str.with_row_index("_row_idx")
    
    # Filter to deals matching this auction
    filtered_df = deal_df_with_str.filter(
        pl.col("_bid_str").str.to_lowercase().is_in(auction_variations)
    )
    
    if filtered_df.height == 0:
        return {"analyzed_deals": 0, "wrong_bid_count": 0, "wrong_bid_rate": 0.0, "by_seat": {}}
    
    # Limit sample for performance
    sample_size = min(1000, filtered_df.height)
    if sample_size < filtered_df.height:
        sample_df = filtered_df.sample(n=sample_size, seed=42)
    else:
        sample_df = filtered_df
    
    analyzed_deals = sample_df.height
    wrong_bid_count = 0
    wrong_by_seat = {1: 0, 2: 0, 3: 0, 4: 0}
    
    # Apply merged rules + overlay to get final criteria
    bt_row_with_rules = _apply_all_rules_to_bt_row(dict(bt_row), state)
    bt_info = {
        "Agg_Expr_Seat_1": bt_row_with_rules.get("Agg_Expr_Seat_1"),
        "Agg_Expr_Seat_2": bt_row_with_rules.get("Agg_Expr_Seat_2"),
        "Agg_Expr_Seat_3": bt_row_with_rules.get("Agg_Expr_Seat_3"),
        "Agg_Expr_Seat_4": bt_row_with_rules.get("Agg_Expr_Seat_4"),
    }
    
    for row in sample_df.iter_rows(named=True):
        deal_idx = row.get("_row_idx", 0)
        dealer = row.get("Dealer", "N")
        bid_str = row.get("_bid_str", "")
        
        conformance = _check_deal_criteria_conformance_bitmap(
            int(deal_idx), bt_info, dealer, deal_criteria_by_seat_dfs, deal_row=row, auction=bid_str
        )
        
        first_wrong = conformance["first_wrong_seat"]
        if first_wrong is not None:
            wrong_bid_count += 1
            wrong_by_seat[first_wrong] = wrong_by_seat.get(first_wrong, 0) + 1
    
    wrong_bid_rate = wrong_bid_count / analyzed_deals if analyzed_deals > 0 else 0.0
    
    seat_stats = {}
    for s in range(1, 5):
        seat_stats[f"seat_{s}"] = {
            "count": wrong_by_seat[s],
            "rate": round(wrong_by_seat[s] / analyzed_deals, 4) if analyzed_deals > 0 else 0.0,
        }
    
    return {
        "analyzed_deals": analyzed_deals,
        "wrong_bid_count": wrong_bid_count,
        "wrong_bid_rate": round(wrong_bid_rate, 4),
        "by_seat": seat_stats,
    }


def handle_bt_seat_stats(
    state: Dict[str, Any],
    bt_row: Dict[str, Any],
    seat: int,
    max_deals: int | None = None,  # reserved for future sampling/limits
) -> Dict[str, Any]:
    """Compute on-the-fly hand stats for one bt_seat1 row and seat (or all seats).
    
    Args:
        state: Shared API state (deal_df, deal_criteria_by_seat_dfs, etc.)
        bt_row: One row from bt_seat1_df (as dict)
        seat: 1-4 for specific seat, 0 for all seats.
        max_deals: Optional cap on deals to aggregate (currently unused; exact stats).
    """
    # Canonical: enrich Agg_Expr if missing + apply overlay + dedupe
    bt_row = _apply_all_rules_to_bt_row(dict(bt_row), state)

    if seat == 0:
        seats = [1, 2, 3, 4]
    else:
        if seat not in (1, 2, 3, 4):
            raise ValueError("seat must be 0 or an integer in 1..4")
        seats = [seat]

    seat_results: Dict[str, Any] = {}
    for s in seats:
        seat_results[str(s)] = _compute_seat_stats_for_bt_row(state, bt_row, s)
    
    # Calculate wrong bid rate for deals matching this auction
    wrong_bid_stats = _compute_wrong_bid_rate_for_bt_row(state, bt_row)

    return {
        "bt_index": int(bt_row.get("bt_index", -1)),
        "auction": bt_row.get("Auction"),
        "seat": seat,
        "seats": seat_results,
        "wrong_bid_stats": wrong_bid_stats,
    }


# ---------------------------------------------------------------------------
# Handler: /process-pbn
# ---------------------------------------------------------------------------


def handle_process_pbn(
    state: Dict[str, Any],
    pbn_input: str,
    include_par: bool,
    default_vul: str,
    parse_file_with_endplay_fn,
) -> Dict[str, Any]:
    """Handle /process-pbn endpoint."""
    import os
    import requests as http_requests
    
    t0 = time.perf_counter()
    deal_df = state["deal_df"]
    
    pbn_input = pbn_input.strip()
    pbn_deals: List[str] = []
    deal_vuls: Dict[int, str] = {}
    input_type = "unknown"
    input_source = ""
    
    if pbn_input.startswith('http://') or pbn_input.startswith('https://'):
        url = pbn_input
        if 'github.com' in url and '/blob/' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        try:
            response = http_requests.get(url, timeout=30)
            response.raise_for_status()
            file_content = response.text
            is_lin = url.lower().endswith('.lin') or 'md|' in file_content[:500]
            pbn_deals, deal_vuls = parse_file_with_endplay_fn(file_content, is_lin=is_lin)
            input_type = "LIN URL" if is_lin else "PBN URL"
            input_source = url
        except Exception as e:
            raise ValueError(f"Failed to fetch/parse URL: {e}")
    
    elif os.path.isfile(pbn_input) or (
        (pbn_input.lower().endswith('.pbn') or pbn_input.lower().endswith('.lin')) and 
        (pbn_input.startswith('/') or (len(pbn_input) > 2 and pbn_input[1] == ':'))
    ):
        try:
            file_path = pbn_input
            if not os.path.isfile(file_path):
                raise ValueError(f"File not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            is_lin = file_path.lower().endswith('.lin') or 'md|' in file_content[:500]
            pbn_deals, deal_vuls = parse_file_with_endplay_fn(file_content, is_lin=is_lin)
            input_type = "LIN file" if is_lin else "PBN file"
            input_source = file_path
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to read/parse file: {e}")
    
    elif 'md|' in pbn_input and '|' in pbn_input:
        pbn_deals, deal_vuls = parse_file_with_endplay_fn(pbn_input, is_lin=True)
        input_type = "LIN string"
        input_source = f"{len(pbn_input)} chars"
    
    else:
        pbn_deals = [pbn_input]
        input_type = "PBN string"
        input_source = f"{len(pbn_input)} chars"
    
    if not pbn_deals:
        raise ValueError("No valid PBN/LIN deals found")
    
    results = []
    for deal_idx, pbn_str in enumerate(pbn_deals):
        deal = parse_pbn_deal(pbn_str)
        if not deal:
            results.append({"error": f"Invalid PBN: {pbn_str[:50]}..."})
            continue

        deal.setdefault("pbn", pbn_str)
        deal.setdefault("Dealer", "N")
        
        for direction in 'NESW':
            hand_col = f'Hand_{direction}'
            if hand_col in deal:
                features = compute_hand_features(deal[hand_col])
                for key, value in features.items():
                    deal[f'{key}_{direction}'] = value
        
        if include_par:
            vul = deal_vuls.get(deal_idx, default_vul)
            deal['Vulnerability'] = vul
            par_info = compute_par_score(pbn_str, str(deal.get('Dealer', 'N')), vul)
            deal.update(par_info)
        
        try:
            match_criteria = pl.lit(True)
            for direction in 'NESW':
                hand_col = f'Hand_{direction}'
                if hand_col in deal and hand_col in deal_df.columns:
                    match_criteria = match_criteria & (pl.col(hand_col) == deal[hand_col])
            
            matching_deals = deal_df.filter(match_criteria)
            if matching_deals.height > 0:
                first_match = matching_deals.row(0, named=True)
                if 'Dealer' in first_match:
                    deal['Dealer'] = first_match['Dealer']
                if 'Vul' in first_match:
                    deal['Vulnerability'] = first_match['Vul']
                game_result_cols = ['bid', 'Declarer', 'Result', 'Tricks', 'Score', 'ParScore', 'DD_Tricks']
                for col in game_result_cols:
                    if col in first_match:
                        deal[col] = first_match[col]
                deal['matching_deals_in_db'] = matching_deals.height
        except Exception as e:
            print(f"[process-pbn] Deal lookup failed: {e}")
        
        results.append(deal)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[process-pbn] {format_elapsed(elapsed_ms)} ({len(results)} deals, type={input_type})")
    
    return {"deals": results, "count": len(results), "input_type": input_type, "input_source": input_source, "elapsed_ms": round(elapsed_ms, 1)}


# ---------------------------------------------------------------------------
# Handler: /group-by-bid
# ---------------------------------------------------------------------------


def handle_group_by_bid(
    state: Dict[str, Any],
    auction_pattern: str,
    n_auction_groups: int,
    n_deals_per_group: int,
    seed: Optional[int],
    min_deals: int,
) -> Dict[str, Any]:
    """Handle /group-by-bid endpoint.
    
    Uses bt_seat1_df for auction lookups if available.
    """
    t0 = time.perf_counter()
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")
    
    if 'bid' not in deal_df.columns:
        raise ValueError("Column 'bid' not found in deal_df")
    
    pattern = normalize_auction_user_text(auction_pattern)
    
    try:
        # Add row index to track original position for bitmap lookups
        deal_df_indexed = deal_df.with_row_index("_row_idx")
        
        bid_dtype = deal_df_indexed.schema.get("bid")
        if bid_dtype == pl.List(pl.Utf8):
            deal_df_with_str = deal_df_indexed.with_columns(pl.col("bid").list.join("-").alias("bid_str"))
        elif bid_dtype == pl.Utf8:
            deal_df_with_str = deal_df_indexed.with_columns(pl.col("bid").fill_null("").alias("bid_str"))
        else:
            bid_str_expr = pl.col("bid").map_elements(
                lambda x: "-".join(map(str, x)) if isinstance(x, list) else (str(x) if x is not None else ""),
                return_dtype=pl.Utf8,
            )
            deal_df_with_str = deal_df_indexed.with_columns(bid_str_expr.alias("bid_str"))
    except Exception as e:
        raise ValueError(f"Failed to build bid_str from 'bid' column: {e}")

    try:
        regex_pattern = f"(?i){pattern}"
        filtered_df = deal_df_with_str.filter(pl.col("bid_str").str.contains(regex_pattern))
    except Exception as e:
        raise ValueError(f"Invalid regex pattern: {e}")
    
    top_level_sql = f"""SELECT bid_str, count(*) as deal_count, AVG(HCP_N), AVG(HCP_S)... 
FROM deals 
WHERE bid ~* '{regex_pattern}' 
GROUP BY bid 
HAVING count(*) >= {min_deals} 
ORDER BY deal_count DESC 
LIMIT {n_auction_groups}"""

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"pattern": pattern, "auction_groups": [], "total_matching_deals": 0, "elapsed_ms": round(elapsed_ms, 1), "sql_query": top_level_sql}
    
    bid_counts = filtered_df.group_by("bid_str").agg(pl.len().alias("deal_count")).sort("deal_count", descending=True)
    if min_deals > 1:
        bid_counts = bid_counts.filter(pl.col("deal_count") >= min_deals)
    
    effective_seed = _effective_seed(seed)
    n_groups = min(n_auction_groups, bid_counts.height)
    if n_groups < bid_counts.height:
        sampled_bids = bid_counts.sample(n=n_groups, seed=effective_seed)
    else:
        sampled_bids = bid_counts
    
    deal_cols = ["_row_idx", "index", "Dealer", "Vul", "Declarer", "bid", "Result", "Tricks", "Score", "ParScore",
                 "Hand_N", "Hand_E", "Hand_S", "Hand_W", "HCP_N", "HCP_E", "HCP_S", "HCP_W",
                 "SL_S_N", "SL_H_N", "SL_D_N", "SL_C_N", "SL_S_E", "SL_H_E", "SL_D_E", "SL_C_E",
                 "SL_S_S", "SL_H_S", "SL_D_S", "SL_C_S", "SL_S_W", "SL_H_W", "SL_D_W", "SL_C_W",
                 "Total_Points_N", "Total_Points_E", "Total_Points_S", "Total_Points_W"]
    # Note: "bid" column will be renamed to "Actual_Auction" in the output
    available_deal_cols = [c for c in deal_cols if c in filtered_df.columns]
    
    bt_cols = ["Auction", "Expr", "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4"]
    
    bt_lookup_df = bt_seat1_df
    # Hard fail if is_completed_auction is missing
    if "is_completed_auction" not in bt_lookup_df.columns:
        raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")
        bt_lookup_df = bt_lookup_df.filter(pl.col("is_completed_auction"))
    # Do not gate on bt_lookup_df columns: in lightweight mode Agg_Expr columns are missing
    # but will be loaded on-demand via `_apply_all_rules_to_bt_row`.
    available_bt_cols = bt_cols
    
    auction_groups = []
    
    for row in sampled_bids.iter_rows(named=True):
        bid_auction = row["bid_str"]
        deal_count = row["deal_count"]
        
        # Use case-insensitive match to be consistent with the regex filter
        group_all_deals = filtered_df.filter(
            pl.col("bid_str").str.to_lowercase() == bid_auction.lower()
        ).select(available_deal_cols)
        n_samples = min(n_deals_per_group, group_all_deals.height)
        if n_samples < group_all_deals.height:
            group_deals = group_all_deals.sample(n=n_samples, seed=effective_seed)
        else:
            group_deals = group_all_deals
        
        bt_info = None
        bt_auction = None
        # Strip leading "P-" prefixes (use regex, not lstrip which removes individual chars)
        auction_for_search = re.sub(r"^(P-)+", "", (bid_auction.upper() if bid_auction else ""))

        # Fast exact match on Auction (pre-normalized)
        bt_match = bt_lookup_df.filter(pl.col("Auction") == auction_for_search)
        if bt_match.height == 0 and not auction_for_search.endswith("-P-P-P"):
            auction_with_passes = auction_for_search + "-P-P-P"
            bt_match = bt_lookup_df.filter(pl.col("Auction") == auction_with_passes)
        if bt_match.height > 0:
            # Canonical: enrich Agg_Expr if missing + apply overlay + dedupe
            bt_row = _apply_all_rules_to_bt_row(dict(bt_match.row(0, named=True)), state)
            bt_info = {c: bt_row.get(c) for c in available_bt_cols if c in bt_row}
            bt_auction = bt_row.get("Auction")
        
        if bt_auction:
            group_deals = group_deals.with_columns(pl.lit(bt_auction).alias("Auction"))
        
        # Use library-style vectorized operations for score deltas and outcome flags
        # Patterned after mlBridgeAugmentLib.create_score_diff_columns and add_trick_columns
        if "Score" in group_deals.columns and "ParScore" in group_deals.columns:
            group_deals = group_deals.with_columns([
                (pl.col("Score").cast(pl.Int64, strict=False) - pl.col("ParScore").cast(pl.Int64, strict=False)).alias("Score_Delta"),
                (pl.col("Result") > 0).alias("OverTricks"),
                (pl.col("Result") == 0).alias("JustMade"),
                (pl.col("Result") < 0).alias("UnderTricks"),
            ])
            group_deals = group_deals.with_columns(
                pl.col("Score_Delta").map_elements(
                    lambda x: calculate_imp(x) * (1 if x >= 0 else -1) if x is not None else None,
                    return_dtype=pl.Int64
                ).alias("Score_IMP")
            )
        
        stats: Dict[str, Any] = {}
        for direction in DIRECTIONS:
            hcp_col = f"HCP_{direction}"
            tp_col = f"Total_Points_{direction}"
            if hcp_col in group_deals.columns:
                hcp_mean = group_deals[hcp_col].mean()
                if hcp_mean is not None:
                    stats[f"HCP_{direction}_avg"] = round(hcp_mean, 1)
            if tp_col in group_deals.columns:
                tp_mean = group_deals[tp_col].mean()
                if tp_mean is not None:
                    stats[f"TP_{direction}_avg"] = round(tp_mean, 1)
        
        # Check each deal for criteria conformance ("wrong bids") using bitmap lookups
        deals_list = group_deals.to_dicts()
        wrong_bid_count = 0
        for deal_row in deals_list:
            # Rename 'bid' column to 'Actual_Auction'
            if "bid" in deal_row:
                deal_row["Actual_Auction"] = _bid_value_to_str(deal_row.pop("bid"))
            
            dealer = deal_row.get("Dealer", "N")
            deal_idx = deal_row.get("_row_idx")
            if deal_idx is not None and bt_info:
                conformance = _check_deal_criteria_conformance_bitmap(
                    int(deal_idx), bt_info, dealer, deal_criteria_by_seat_dfs,
                    deal_row=deal_row,
                    auction=bt_auction or bid_auction
                )
                # Add per-seat wrong bid columns
                for seat in range(1, 5):
                    deal_row[f"Wrong_Bid_S{seat}"] = conformance[f"Wrong_Bid_S{seat}"]
                    deal_row[f"Invalid_Criteria_S{seat}"] = conformance[f"Invalid_Criteria_S{seat}"]
                deal_row["first_wrong_seat"] = conformance["first_wrong_seat"]
                if conformance["first_wrong_seat"] is not None:
                    wrong_bid_count += 1
            else:
                # No bitmap info available - set defaults
                for seat in range(1, 5):
                    deal_row[f"Wrong_Bid_S{seat}"] = False
                    deal_row[f"Invalid_Criteria_S{seat}"] = None
                deal_row["first_wrong_seat"] = None
            # Keep _row_idx in output so clients can use it for /deal-criteria-eval-batch
        
        auction_groups.append({
            "auction": bid_auction, "bt_auction": bt_auction, "deal_count": deal_count,
            "sample_count": group_deals.height, "bt_info": bt_info, "stats": stats,
            "wrong_bid_count": wrong_bid_count,
            "deals": deals_list,
        })
    
    # Sort groups by number of leading passes (seat order: 1, 2, 3, 4)
    auction_groups.sort(key=lambda g: _count_leading_passes(g.get("auction", "")))
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_deals_out = sum(g["sample_count"] for g in auction_groups)
    print(f"[group-by-bid] {format_elapsed(elapsed_ms)} ({len(auction_groups)} groups, {total_deals_out} deals)")
    
    return {
        "pattern": pattern, "auction_groups": auction_groups, "total_matching_deals": filtered_df.height,
        "unique_auctions": bid_counts.height, "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Handler: /wrong-bid-stats
# ---------------------------------------------------------------------------


def handle_wrong_bid_stats(
    state: Dict[str, Any],
    auction_pattern: Optional[str],
    seat: Optional[int],
) -> Dict[str, Any]:
    """Handle /wrong-bid-stats endpoint.
    
    Provides aggregate statistics about wrong bids across the dataset.
    Uses vectorized join instead of per-row BT lookups for performance.
    """
    t0 = time.perf_counter()
    timings_ms: Dict[str, float] = {}
    
    def _mark(label: str) -> None:
        timings_ms[label] = round((time.perf_counter() - t0) * 1000, 1)
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    bt_stats_df = state.get("bt_stats_df")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index for bitmap lookups
    deal_df = deal_df.with_row_index("_row_idx")
    _mark("add_row_idx")
    
    # PERFORMANCE: Defer expensive string processing until after sampling if possible.
    # If filtering by pattern, we MUST compute _bid_str for all rows.
    # But we only need _auction_key (regex replaces) for the join on the small sample.
    if auction_pattern:
        deals_prepared = prepare_deals_with_bid_str(deal_df, include_auction_key=False)
        try:
            regex_pattern = f"(?i){normalize_auction_user_text(auction_pattern)}"
            deals_prepared = deals_prepared.filter(pl.col("_bid_str").str.contains(regex_pattern))
        except Exception as e:
            raise ValueError(f"Invalid auction pattern: {e}")
        _mark("filter_pattern")
    else:
        deals_prepared = deal_df

    total_deals = deals_prepared.height
    
    # Sample deals to analyze (for performance, limit to 10000)
    sample_size = min(10000, total_deals)
    if sample_size < total_deals:
        sample_df = deals_prepared.sample(n=sample_size, seed=42)
    else:
        sample_df = deals_prepared
    _mark("sample")
    
    # Fully prepare the small sample (including _auction_key for the join)
    sample_df = prepare_deals_with_bid_str(sample_df, include_auction_key=True)
    _mark("prepare_bid_str")

    analyzed_deals = sample_df.height
    
    # PERFORMANCE: Use pre-computed deal_to_bt_index_df for O(1) lookup
    # instead of expensive string join on _auction_key (~25s -> ~200ms)
    joined_df = join_deals_with_bt_via_index(sample_df, state)
    _mark("join_bt")
    
    unique_bt_count = joined_df.select(pl.col("bt_index").drop_nulls().n_unique()).item() if "bt_index" in joined_df.columns else 0
    _mark("count_unique_bt")
    
    # Batch check wrong bids (still loops but no per-row filter operations)
    result_df = batch_check_wrong_bids(joined_df, deal_criteria_by_seat_dfs, seat, state=state)
    _mark("batch_check")
    
    # Aggregate statistics from result_df
    wrong_rows = result_df.filter(pl.col("first_wrong_seat").is_not_null())
    deals_with_wrong_bid = wrong_rows.height
    
    # By seat
    wrong_bids_by_seat: Dict[int, int] = {s: 0 for s in range(1, 5)}
    for s in range(1, 5):
        wrong_bids_by_seat[s] = result_df.filter(pl.col(f"Wrong_Bid_S{s}")).height
    
    # By dealer
    wrong_bids_by_dealer = {"N": 0, "E": 0, "S": 0, "W": 0}
    if wrong_rows.height > 0 and "Dealer" in wrong_rows.columns:
        dealer_counts = wrong_rows.group_by("Dealer").agg(pl.len().alias("count")).to_dicts()
        for row in dealer_counts:
            d = row.get("Dealer")
            if d in wrong_bids_by_dealer:
                wrong_bids_by_dealer[d] = row.get("count", 0)
    
    # By vulnerability
    wrong_bids_by_vul = {"None": 0, "NS": 0, "EW": 0, "Both": 0}
    if wrong_rows.height > 0 and "Vul" in wrong_rows.columns:
        vul_counts = wrong_rows.group_by("Vul").agg(pl.len().alias("count")).to_dicts()
        for row in vul_counts:
            v = row.get("Vul")
            if v in wrong_bids_by_vul:
                wrong_bids_by_vul[v] = row.get("count", 0)

    # Unique auctions with wrong bids (for overall stats)
    unique_auctions_with_wrong_bids = 0
    if wrong_rows.height > 0:
        if "_auction_key" in wrong_rows.columns:
            unique_auctions_with_wrong_bids = wrong_rows.select(pl.col("_auction_key").n_unique()).item()
        elif "_bid_str" in wrong_rows.columns:
            unique_auctions_with_wrong_bids = wrong_rows.select(pl.col("_bid_str").n_unique()).item()
    
    # Calculate rates
    wrong_bid_rate = deals_with_wrong_bid / analyzed_deals if analyzed_deals > 0 else 0.0
    
    seat_rates: Dict[str, Dict[str, float]] = {}
    per_seat: Dict[str, float] = {}
    for s in range(1, 5):
        count_s = wrong_bids_by_seat[s]
        rate_s = count_s / analyzed_deals if analyzed_deals > 0 else 0.0
        seat_rates[f"seat_{s}"] = {
            "count": count_s,
            "rate": rate_s,
        }
        # Structure expected by Streamlit UI (seat_{i}_wrong_bids / seat_{i}_rate)
        per_seat[f"seat_{s}_wrong_bids"] = count_s
        per_seat[f"seat_{s}_rate"] = rate_s
    
    dealer_rates: Dict[str, Dict[str, float]] = {}
    for d in ["N", "E", "S", "W"]:
        count_d = wrong_bids_by_dealer[d]
        dealer_rates[d] = {
            "count": count_d,
            "rate": count_d / analyzed_deals if analyzed_deals > 0 else 0.0,
        }
    
    vul_rates: Dict[str, Dict[str, float]] = {}
    for v in ["None", "NS", "EW", "Both"]:
        count_v = wrong_bids_by_vul[v]
        vul_rates[v] = {
            "count": count_v,
            "rate": count_v / analyzed_deals if analyzed_deals > 0 else 0.0,
        }
    
    # Overall summary block expected by Streamlit
    overall_stats = {
        "total_deals": total_deals,
        "deals_with_wrong_bids": deals_with_wrong_bid,
        "wrong_bid_rate": round(wrong_bid_rate, 4),
        "unique_auctions_with_wrong_bids": unique_auctions_with_wrong_bids,
    }

    _mark("aggregate")
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[wrong-bid-stats] {format_elapsed(elapsed_ms)} ({analyzed_deals} analyzed, {deals_with_wrong_bid} wrong, {unique_bt_count} unique auctions)")
    try:
        parts = ", ".join(f"{k}={v}ms" for k, v in timings_ms.items())
        print(f"[wrong-bid-stats] TIMING: {parts}, total_ms={round(elapsed_ms, 1)}")
    except Exception:
        pass
    
    return {
        # Legacy flat fields
        "total_deals": total_deals,
        "analyzed_deals": analyzed_deals,
        "deals_with_wrong_bid": deals_with_wrong_bid,
        "wrong_bid_rate": round(wrong_bid_rate, 4),
        "by_seat": seat_rates,
        "by_dealer": dealer_rates,
        "by_vulnerability": vul_rates,
        "auction_pattern": auction_pattern,
        "seat_filter": seat,
        "elapsed_ms": round(elapsed_ms, 1),
        # New structured fields for Wrong Bid Analysis UI
        "overall_stats": overall_stats,
        "per_seat": per_seat,
    }


# ---------------------------------------------------------------------------
# Handler: /failed-criteria-summary
# ---------------------------------------------------------------------------


def handle_failed_criteria_summary(
    state: Dict[str, Any],
    auction_pattern: Optional[str],
    top_n: int,
    seat: Optional[int],
) -> Dict[str, Any]:
    """Handle /failed-criteria-summary endpoint.
    
    Analyzes which criteria fail most often across deals.
    Uses vectorized join instead of per-row BT lookups for performance.
    """
    t0 = time.perf_counter()
    timings_ms: dict[str, float] = {}
    _t_last = t0
    def _mark(name: str) -> None:
        nonlocal _t_last
        now = time.perf_counter()
        timings_ms[name] = round((now - _t_last) * 1000, 1)
        _t_last = now
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    bt_stats_df = state.get("bt_stats_df")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index for bitmap lookups
    deal_df = deal_df.with_row_index("_row_idx")
    _mark("add_row_index")
    
    # PERFORMANCE: Defer expensive string processing until after sampling.
    if auction_pattern:
        deals_prepared = prepare_deals_with_bid_str(deal_df, include_auction_key=False)
        try:
            regex_pattern = f"(?i){normalize_auction_user_text(auction_pattern)}"
            deals_prepared = deals_prepared.filter(pl.col("_bid_str").str.contains(regex_pattern))
        except Exception as e:
            raise ValueError(f"Invalid auction pattern: {e}")
    else:
        deals_prepared = deal_df
    _mark("pattern_filter")

    total_deals = deals_prepared.height
    
    # Sample for performance
    sample_size = min(10000, total_deals)
    if sample_size < total_deals:
        sample_df = deals_prepared.sample(n=sample_size, seed=42)
    else:
        sample_df = deals_prepared
    _mark("sample")
    
    # Fully prepare the small sample
    sample_df = prepare_deals_with_bid_str(sample_df, include_auction_key=True)
    _mark("prepare_sample")

    analyzed_deals = sample_df.height
    
    # PERFORMANCE: Use pre-computed deal_to_bt_index_df for O(1) lookup
    # instead of expensive string join on _auction_key (~25s -> ~200ms)
    joined_df = join_deals_with_bt_via_index(sample_df, state)
    _mark("join")

    # PERFORMANCE: Precompute (overlay+deduped) criteria lists per bt_index once.
    bt_criteria_by_bt_index: dict[int, dict[int, list[Any]]] = {}
    if "bt_index" in joined_df.columns:
        bt_cols = ["bt_index", "Auction"] + [agg_expr_col(s) for s in SEAT_RANGE if agg_expr_col(s) in joined_df.columns]
        try:
            uniq_bt = joined_df.select([c for c in bt_cols if c in joined_df.columns]).unique(subset=["bt_index"])
            for r in uniq_bt.iter_rows(named=True):
                try:
                    bt_idx_raw = r.get("bt_index")
                    if bt_idx_raw is None:
                        continue
                    bt_idx = int(bt_idx_raw)
                except Exception:
                    continue
                processed = _apply_all_rules_to_bt_row(dict(r), state)
                bt_criteria_by_bt_index[bt_idx] = {
                    s: (processed.get(agg_expr_col(s)) or [])
                    for s in SEAT_RANGE
                    if agg_expr_col(s) in processed
                }
        except Exception:
            bt_criteria_by_bt_index = {}
    _mark("precompute_bt_criteria")
    
    # Track criteria failures
    criteria_fail_counts: Dict[str, int] = {}
    criteria_check_counts: Dict[str, int] = {}
    criteria_by_seat: Dict[int, Dict[str, int]] = {1: {}, 2: {}, 3: {}, 4: {}}
    
    # Process each deal - now without per-row BT lookups (already joined)
    seats_to_check = [seat] if seat else list(range(1, 5))
    # Cache columns sets per seat+dealer once
    colset_cache: dict[tuple[int, str], set[str]] = {}

    t_loop0 = time.perf_counter()
    for row in joined_df.iter_rows(named=True):
        deal_idx_raw = row.get("_row_idx", 0)
        deal_idx = int(deal_idx_raw) if deal_idx_raw is not None else 0
        dealer = str(row.get("Dealer", "N") or "N").upper()
        bt_idx = None
        try:
            bt_idx_val = row.get("bt_index")
            bt_idx = int(bt_idx_val) if bt_idx_val is not None else None
        except Exception:
            bt_idx = None
        
        # Check each seat's criteria
        for s in seats_to_check:
            if s is None:
                continue
            s_i = int(s)
            # Pull from bt_index cache when possible (overlay+dedupe already applied)
            if bt_idx is not None and bt_idx in bt_criteria_by_bt_index:
                criteria_list = bt_criteria_by_bt_index[bt_idx].get(s_i)
            else:
                criteria_list = row.get(f"Agg_Expr_Seat_{s_i}")
            if not criteria_list:
                continue
            
            seat_dfs = deal_criteria_by_seat_dfs.get(s_i, {})
            criteria_df = seat_dfs.get(dealer)
            if criteria_df is None or criteria_df.is_empty():
                continue

            key = (s_i, dealer)
            cols = colset_cache.get(key)
            if cols is None:
                cols = set(criteria_df.columns)
                colset_cache[key] = cols
            
            for criterion in criteria_list:
                if str(criterion) not in cols:
                    continue
                
                # Track check count
                criteria_check_counts[criterion] = criteria_check_counts.get(criterion, 0) + 1
                
                try:
                    bitmap_value = criteria_df[str(criterion)][deal_idx]
                    if not bitmap_value:
                        criteria_fail_counts[criterion] = criteria_fail_counts.get(criterion, 0) + 1
                        criteria_by_seat[s_i][criterion] = criteria_by_seat[s_i].get(criterion, 0) + 1
                except (IndexError, KeyError):
                    continue
    timings_ms["loop"] = round((time.perf_counter() - t_loop0) * 1000, 1)
    _t_last = time.perf_counter()
    
    # Build results sorted by fail count
    criteria_results = []
    for criterion, fail_count in sorted(criteria_fail_counts.items(), key=lambda x: -x[1]):
        check_count = criteria_check_counts.get(criterion, 0)
        fail_rate = fail_count / check_count if check_count > 0 else 0.0
        criteria_results.append({
            "criterion": criterion,
            "failure_count": fail_count,
            "check_count": check_count,
            "affected_auctions": check_count,
            "fail_rate": round(fail_rate, 4),
        })
    
    # Top N
    top_criteria = criteria_results[:top_n]
    
    # By seat breakdown
    seat_breakdown = {}
    for s in range(1, 5):
        seat_top = sorted(criteria_by_seat[s].items(), key=lambda x: -x[1])[:10]
        seat_breakdown[f"seat_{s}"] = [{"criterion": c, "failure_count": cnt} for c, cnt in seat_top]
    _mark("build_results")
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[failed-criteria-summary] {format_elapsed(elapsed_ms)} ({analyzed_deals} analyzed)")
    try:
        parts = ", ".join(f"{k}={v}ms" for k, v in timings_ms.items())
        print(f"[failed-criteria-summary] TIMING: {parts}, total_ms={round(elapsed_ms,1)}")
    except Exception:
        pass
    
    return {
        "total_deals": total_deals,
        "analyzed_deals": analyzed_deals,
        "unique_criteria_failed": len(criteria_fail_counts),
        "top_failing_criteria": top_criteria,
        "by_seat": seat_breakdown,
        "auction_pattern": auction_pattern,
        "seat_filter": seat,
        "elapsed_ms": round(elapsed_ms, 1),
        # Field expected by Streamlit Wrong Bid Analysis UI
        "criteria": top_criteria,
    }


# ---------------------------------------------------------------------------
# Handler: /wrong-bid-leaderboard
# ---------------------------------------------------------------------------


def handle_wrong_bid_leaderboard(
    state: Dict[str, Any],
    top_n: int,
    seat: Optional[int],
) -> Dict[str, Any]:
    """Handle /wrong-bid-leaderboard endpoint.
    
    Returns leaderboard of bids with highest error rates.
    Uses vectorized join instead of per-row BT lookups for performance.
    """
    t0 = time.perf_counter()
    timings_ms: dict[str, float] = {}
    _t_last = t0
    def _mark(name: str) -> None:
        nonlocal _t_last
        now = time.perf_counter()
        timings_ms[name] = round((now - _t_last) * 1000, 1)
        _t_last = now
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    bt_stats_df = state.get("bt_stats_df")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index
    deal_df = deal_df.with_row_index("_row_idx")
    _mark("add_row_index")
    
    # PERFORMANCE: Defer expensive string processing until after sampling.
    # No auction pattern filter here, so we always defer.
    deals_prepared = deal_df
    _mark("pattern_filter")

    # Sample for performance
    total_deals = deals_prepared.height
    sample_size = min(10000, total_deals)
    if sample_size < total_deals:
        sample_df = deals_prepared.sample(n=sample_size, seed=42)
    else:
        sample_df = deals_prepared
    _mark("sample")
    
    # Fully prepare the small sample
    sample_df = prepare_deals_with_bid_str(sample_df, include_auction_key=True)
    _mark("prepare_sample")

    analyzed_deals = sample_df.height
    
    # PERFORMANCE: Use pre-computed deal_to_bt_index_df for O(1) lookup
    # instead of expensive string join on _auction_key (~25s -> ~200ms)
    joined_df = join_deals_with_bt_via_index(sample_df, state)
    _mark("join")

    # PERFORMANCE: Precompute (overlay+deduped) criteria lists per bt_index once.
    bt_criteria_by_bt_index: dict[int, dict[int, list[Any]]] = {}
    if "bt_index" in joined_df.columns:
        bt_cols = ["bt_index", "Auction"] + [agg_expr_col(s) for s in SEAT_RANGE if agg_expr_col(s) in joined_df.columns]
        try:
            uniq_bt = joined_df.select([c for c in bt_cols if c in joined_df.columns]).unique(subset=["bt_index"])
            for r in uniq_bt.iter_rows(named=True):
                try:
                    bt_idx_raw = r.get("bt_index")
                    if bt_idx_raw is None:
                        continue
                    bt_idx = int(bt_idx_raw)
                except Exception:
                    continue
                processed = _apply_all_rules_to_bt_row(dict(r), state)
                bt_criteria_by_bt_index[bt_idx] = {
                    s: (processed.get(agg_expr_col(s)) or [])
                    for s in SEAT_RANGE
                    if agg_expr_col(s) in processed
                }
        except Exception:
            bt_criteria_by_bt_index = {}
    _mark("precompute_bt_criteria")

    colset_cache: dict[tuple[int, str], set[str]] = {}
    
    # Track wrong bids by (bid, seat)
    bid_seat_wrong: Dict[Tuple[str, int], int] = {}
    bid_seat_total: Dict[Tuple[str, int], int] = {}
    bid_failed_criteria: Dict[Tuple[str, int], Dict[str, int]] = {}
    
    # Process each deal - now without per-row BT lookups (already joined)
    seats_to_check = [seat] if seat else list(range(1, 5))
    t_loop0 = time.perf_counter()
    for row in joined_df.iter_rows(named=True):
        deal_idx_raw = row.get("_row_idx", 0)
        deal_idx = int(deal_idx_raw) if deal_idx_raw is not None else 0
        dealer = str(row.get("Dealer", "N") or "N").upper()
        bid_str = row.get("_bid_str", "")
        bt_idx = None
        try:
            bt_idx_val = row.get("bt_index")
            bt_idx = int(bt_idx_val) if bt_idx_val is not None else None
        except Exception:
            bt_idx = None
        
        # For each seat, track the bid and whether it's wrong
        for s in seats_to_check:
            if s is None:
                continue
            s_i = int(s)
            bid_at_seat = _extract_bid_at_seat(bid_str, s_i)
            if not bid_at_seat:
                continue
            
            key = (bid_at_seat.upper(), s_i)
            bid_seat_total[key] = bid_seat_total.get(key, 0) + 1
            
            # Check this seat's criteria (from joined data)
            if bt_idx is not None and bt_idx in bt_criteria_by_bt_index:
                criteria_list = bt_criteria_by_bt_index[bt_idx].get(s_i)
            else:
                criteria_list = row.get(f"Agg_Expr_Seat_{s_i}")
            if not criteria_list:
                continue
            
            seat_dfs = deal_criteria_by_seat_dfs.get(s_i, {})
            criteria_df = seat_dfs.get(dealer)
            if criteria_df is None or criteria_df.is_empty():
                continue

            key_sd = (s_i, dealer)
            cols = colset_cache.get(key_sd)
            if cols is None:
                cols = set(criteria_df.columns)
                colset_cache[key_sd] = cols
            
            seat_failed = []
            for criterion in criteria_list:
                if str(criterion) not in cols:
                    continue
                try:
                    bitmap_value = criteria_df[str(criterion)][deal_idx]
                    if not bitmap_value:
                        seat_failed.append(criterion)
                except (IndexError, KeyError):
                    continue
            
            if seat_failed:
                bid_seat_wrong[key] = bid_seat_wrong.get(key, 0) + 1
                if key not in bid_failed_criteria:
                    bid_failed_criteria[key] = {}
                for c in seat_failed:
                    bid_failed_criteria[key][c] = bid_failed_criteria[key].get(c, 0) + 1
    timings_ms["loop"] = round((time.perf_counter() - t_loop0) * 1000, 1)
    _t_last = time.perf_counter()
    
    # Build leaderboard sorted by wrong rate (with minimum occurrences)
    min_occurrences = 10
    leaderboard = []
    for key, wrong_count in bid_seat_wrong.items():
        total = bid_seat_total.get(key, 0)
        if total < min_occurrences:
            continue
        
        bid, s = key
        wrong_rate = wrong_count / total if total > 0 else 0
        
        # Get top failing criteria for this bid
        criteria_counts = bid_failed_criteria.get(key, {})
        top_criteria = sorted(criteria_counts.items(), key=lambda x: -x[1])[:5]
        
        leaderboard.append({
            "bid": bid,
            "seat": s,
            "wrong_count": wrong_count,
            "total_count": total,
            "wrong_rate": round(wrong_rate, 4),
            "common_failures": [c for c, _ in top_criteria],
        })
    
    # Sort by wrong_rate descending
    leaderboard.sort(key=lambda x: -x["wrong_rate"])
    top_leaderboard = leaderboard[:top_n]
    _mark("build_results")
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[wrong-bid-leaderboard] {format_elapsed(elapsed_ms)} ({analyzed_deals} analyzed)")
    try:
        parts = ", ".join(f"{k}={v}ms" for k, v in timings_ms.items())
        print(f"[wrong-bid-leaderboard] TIMING: {parts}, total_ms={round(elapsed_ms,1)}")
    except Exception:
        pass
    
    return {
        "analyzed_deals": analyzed_deals,
        "unique_bid_seat_combos": len(bid_seat_total),
        "leaderboard": top_leaderboard,
        "min_occurrences": min_occurrences,
        "seat_filter": seat,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Handler: /bidding-arena (Bidding Arena)
# ---------------------------------------------------------------------------


def _parse_contract(contract: str | None) -> Dict[str, Any]:
    """Parse a contract string into components.
    
    Examples: "3NT", "4S", "2Hx", "6Cxx", "Pass"
    Returns: {"level": int, "strain": str, "doubled": bool, "redoubled": bool, "is_pass": bool}
    """
    if not contract or contract.lower() in ("pass", "passed", "p", ""):
        return {"level": 0, "strain": "", "doubled": False, "redoubled": False, "is_pass": True}
    
    contract = contract.strip().upper()
    
    # Check for doubles/redoubles
    redoubled = "XX" in contract
    doubled = "X" in contract and not redoubled
    contract = contract.replace("XX", "").replace("X", "")
    
    # Parse level and strain
    if not contract or not contract[0].isdigit():
        return {"level": 0, "strain": "", "doubled": False, "redoubled": False, "is_pass": True}
    
    level = int(contract[0])
    strain = contract[1:].strip() if len(contract) > 1 else ""
    
    # Normalize strain names
    strain_map = {"N": "NT", "S": "S", "H": "H", "D": "D", "C": "C", "NT": "NT", "NOTRUMP": "NT"}
    strain = strain_map.get(strain.upper(), strain.upper())
    
    return {
        "level": level,
        "strain": strain,
        "doubled": doubled,
        "redoubled": redoubled,
        "is_pass": False,
    }


def _is_slam(contract: str | None) -> bool:
    """Check if contract is a slam (6 or 7 level)."""
    parsed = _parse_contract(contract)
    return parsed["level"] >= 6


def _is_game(contract: str | None) -> bool:
    """Check if contract is a game (3NT, 4H, 4S, 5C, 5D, or higher)."""
    parsed = _parse_contract(contract)
    level = parsed["level"]
    strain = parsed["strain"]
    
    if level >= 6:  # Slams are also games
        return True
    if level == 5 and strain in ("C", "D"):
        return True
    if level >= 4 and strain in ("H", "S"):
        return True
    if level >= 3 and strain == "NT":
        return True
    return False


def _is_partscore(contract: str | None) -> bool:
    """Check if contract is a partscore (not game, not passed out)."""
    parsed = _parse_contract(contract)
    if parsed["is_pass"]:
        return False
    return not _is_game(contract)


def _hcp_range_label(hcp: int | None) -> str:
    """Categorize HCP into ranges."""
    if hcp is None:
        return "unknown"
    if hcp <= 10:
        return "0-10"
    elif hcp <= 12:
        return "11-12"
    elif hcp <= 14:
        return "13-14"
    elif hcp <= 17:
        return "15-17"
    elif hcp <= 19:
        return "18-19"
    elif hcp <= 21:
        return "20-21"
    else:
        return "22+"


def _load_deals_from_uri(uri: str) -> pl.DataFrame:
    """Load deals from a file path or URL.
    
    Supports:
    - Local file paths (.parquet, .csv)
    - HTTP/HTTPS URLs (.parquet, .csv)
    - GitHub blob URLs (auto-converted to raw)
    
    Args:
        uri: File path or URL to load deals from.
    
    Returns:
        Polars DataFrame with deal data.
    
    Raises:
        ValueError: If the file format is not supported or load fails.
    """
    import requests as http_requests
    import tempfile
    import os
    
    uri = uri.strip()
    
    # Determine if it's a URL or file path
    is_url = uri.startswith("http://") or uri.startswith("https://")
    
    if is_url:
        # Convert GitHub blob URLs to raw URLs
        if "github.com" in uri and "/blob/" in uri:
            uri = uri.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        
        # Download the file
        try:
            response = http_requests.get(uri, timeout=60)
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"Failed to download from URL: {e}")
        
        # Determine format from URL or content
        uri_lower = uri.lower()
        if uri_lower.endswith(".parquet"):
            # Save to temp file and read
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            try:
                df = pl.read_parquet(temp_path)
            finally:
                os.unlink(temp_path)
        elif uri_lower.endswith(".csv"):
            from io import StringIO
            df = pl.read_csv(StringIO(response.text))
        else:
            # Try parquet first, then CSV
            try:
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name
                try:
                    df = pl.read_parquet(temp_path)
                finally:
                    os.unlink(temp_path)
            except Exception:
                from io import StringIO
                try:
                    df = pl.read_csv(StringIO(response.text))
                except Exception as e:
                    raise ValueError(f"Could not parse as parquet or CSV: {e}")
    else:
        # Local file path
        if not os.path.exists(uri):
            raise ValueError(f"File not found: {uri}")
        
        uri_lower = uri.lower()
        if uri_lower.endswith(".parquet"):
            df = pl.read_parquet(uri)
        elif uri_lower.endswith(".csv"):
            df = pl.read_csv(uri)
        else:
            # Try parquet first, then CSV
            try:
                df = pl.read_parquet(uri)
            except Exception:
                try:
                    df = pl.read_csv(uri)
                except Exception as e:
                    raise ValueError(f"Could not parse as parquet or CSV: {e}")
    
    # Validate required columns
    required_cols = {"Dealer", "Vul"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Need at least one hand column
    hand_cols = [c for c in df.columns if c.startswith("Hand_")]
    if not hand_cols:
        raise ValueError("No Hand_* columns found (need Hand_N, Hand_E, Hand_S, Hand_W)")
    
    return df


def handle_bidding_arena(
    state: Dict[str, Any],
    model_a: str,
    model_b: str,
    auction_pattern: Optional[str],
    sample_size: int,
    seed: Optional[int],
    deals_uri: Optional[str] = None,
    deal_indices: Optional[List[int]] = None,
    search_all_bt_rows: bool = False,
    use_model_rules: bool = False,
    include_model_auction: bool = False,
) -> Dict[str, Any]:
    """Handle /bidding-arena endpoint (Bidding Arena).
    
    Provides comprehensive head-to-head comparison between two bidding models.
    
    Supported models:
    - "Actual": Human bids from the dataset
    - "Rules": Full pipeline (Base + Learned + CSV Overlay)
    - "Rules_Learned": Base + Learned (no CSV overlay)
    - "Rules_Base": Base only (original GIB criteria)
    
    Args:
        deals_uri: Optional file path or URL to load custom deals from.
                   Supports .parquet and .csv files. If None, uses deal_df.
    """
    t0 = time.perf_counter()
    debug = bool(state.get("debug_bidding_arena", False))
    # Console timings to pinpoint slow phases (also returned in payload as timings_ms).
    _t_last = t0
    timings_ms: dict[str, float] = {}

    def _mark(label: str) -> None:
        nonlocal _t_last
        now = time.perf_counter()
        timings_ms[label] = round((now - _t_last) * 1000, 1)
        _t_last = now
    
    # Load deals from URI or use default
    deals_source = "default"
    if deals_uri:
        try:
            deal_df = _load_deals_from_uri(deals_uri)
            deals_source = deals_uri
            if debug:
                print(f"[bidding-arena] Loaded {deal_df.height} deals from {deals_uri}")
        except Exception as e:
            raise ValueError(f"Failed to load deals from URI: {e}")
    else:
        deal_df = state["deal_df"]
    _mark("load_deals")
    
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    _mark("read_state_refs")

    # Model auction computation requires criteria bitmaps aligned to the in-memory deal_df.
    # If the caller provides a custom deals_uri, we can't safely evaluate criteria against it.
    if (use_model_rules or include_model_auction) and deals_uri:
        raise ValueError(
            "Model auction computation is not supported with custom deals_uri "
            "(criteria bitmaps are not available for external deals)."
        )
    
    # Validate models
    # Note: Merged rules are pre-compiled into BT, so all models are available
    valid_models = {MODEL_ACTUAL, MODEL_RULES_BASE, MODEL_RULES, MODEL_RULES_LEARNED}
    
    if model_a not in valid_models:
        raise ValueError(f"Invalid model_a: {model_a}. Valid: {valid_models}")
    if model_b not in valid_models:
        raise ValueError(f"Invalid model_b: {model_b}. Valid: {valid_models}")
    if model_a == model_b:
        raise ValueError("model_a and model_b must be different")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index
    deal_df = deal_df.with_row_index("_row_idx")
    _mark("add_row_index")
    
    # PERFORMANCE: Defer expensive string processing until after sampling if possible.
    # If filtering by pattern, we MUST compute _bid_str for all rows.
    # But we NEVER need _auction_key in the Bidding Arena (it's for joins).
    if auction_pattern:
        deals_prepared = prepare_deals_with_bid_str(deal_df, include_auction_key=False)
        try:
            regex_pattern = f"(?i){normalize_auction_user_text(auction_pattern)}"
            deals_prepared = deals_prepared.filter(pl.col("_bid_str").str.contains(regex_pattern))
        except Exception as e:
            raise ValueError(f"Invalid auction pattern: {e}")
    else:
        # If no pattern, we can sample FIRST, then prepare ONLY the samples.
        deals_prepared = deal_df
    _mark("prepare_and_filter_deals")

    total_deals = deals_prepared.height
    
    # Sample deals (optionally force-include pinned deal indexes)
    effective_seed = _effective_seed(seed)
    pinned_df: Optional[pl.DataFrame] = None
    pinned_set: set[int] = set()
    if deal_indices:
        # Normalize/validate indices
        pinned_list: list[int] = []
        for x in deal_indices:
            try:
                pinned_list.append(int(x))
            except (ValueError, TypeError):
                continue
        if pinned_list and "index" in deal_df.columns:
            pinned_set = set(pinned_list)
            # Optimization (Invariant B: monotonic `index`):
            # If the deals file preserves row order and `index` is monotonic, map
            # deal `index` -> row position via binary search, then take rows directly.
            # This avoids scanning/joining the full deal_df for pinned-only workflows.
            if not auction_pattern:
                try:
                    idx_arr = state.get("deal_index_arr")
                    is_mono = bool(state.get("deal_index_monotonic", False))
                    if is_mono and idx_arr is not None:
                        import numpy as np
                        row_positions: list[int] = []
                        for idx_val in pinned_list:
                            pos = int(np.searchsorted(idx_arr, int(idx_val)))
                            if 0 <= pos < len(idx_arr) and int(idx_arr[pos]) == int(idx_val):
                                row_positions.append(pos)
                        if row_positions:
                            # Take rows from the full deal_df (keeps order of pinned_list)
                            pinned_df = _take_rows_by_index(deal_df, row_positions)
                except Exception:
                    pinned_df = None

            # Fallback: join on index (works for pattern-filtered deals_prepared)
            if pinned_df is None and pinned_list and "index" in deals_prepared.columns:
                order_df = pl.DataFrame({"index": pinned_list, "_pin_rank": list(range(len(pinned_list)))})
                pinned_df = (
                    deals_prepared
                    .join(order_df, on="index", how="inner")
                    .sort("_pin_rank")
                    .drop("_pin_rank")
                )

    if pinned_df is not None and pinned_df.height > 0:
        remaining_n = max(0, int(sample_size) - int(pinned_df.height))
        if remaining_n <= 0:
            sample_df = pinned_df
        else:
            if auction_pattern:
                # Pattern-filtered mode: remaining_df is already a filtered view.
                remaining_df = deals_prepared.filter(~pl.col("index").is_in(list(pinned_set)))
                if remaining_df.height > 0:
                    remaining_n = min(remaining_n, remaining_df.height)
                    sampled_rest = remaining_df.sample(n=remaining_n, seed=effective_seed) if remaining_n < remaining_df.height else remaining_df
                    sample_df = pl.concat([pinned_df, sampled_rest], how="vertical")
                else:
                    sample_df = pinned_df
            else:
                # Fast path: avoid filtering the full deal_df. Sample a slightly larger batch,
                # then drop pinned indices in the *small* sample.
                oversample_n = min(total_deals, remaining_n + len(pinned_set))
                sampled = deal_df.sample(n=oversample_n, seed=effective_seed) if oversample_n < total_deals else deal_df
                sampled = sampled.filter(~pl.col("index").is_in(list(pinned_set)))
                if sampled.height > remaining_n:
                    sampled = sampled.head(remaining_n)
                sample_df = pl.concat([pinned_df, sampled], how="vertical")
    else:
        if sample_size < total_deals:
            sample_df = deals_prepared.sample(n=sample_size, seed=effective_seed)
        else:
            sample_df = deals_prepared
    
    # If we deferred preparation, do it now on the small sample_df.
    if not auction_pattern:
        sample_df = prepare_deals_with_bid_str(sample_df, include_auction_key=False)
    _mark("sample_and_prepare")

    analyzed_deals = sample_df.height
    
    # ---------------------------------------------------------------------------
    # OPTIMIZATION: Pinned-Only Mode Detection
    # ---------------------------------------------------------------------------
    # If we're only processing pinned deals (no random samples), we can dramatically
    # reduce the candidate pool by only loading BT rows for the specific auctions
    # in those deals. This avoids the expensive 5,000-row Agg_Expr load.
    pinned_only_mode = (
        pinned_df is not None 
        and pinned_df.height > 0 
        and pinned_df.height == sample_df.height  # No random samples added
    )
    pinned_auction_bt_indices: set[int] = set()
    
    if pinned_only_mode and "_bid_str" in sample_df.columns:
        # Extract unique auctions from pinned deals and resolve to bt_indices
        t_pinned_resolve = time.perf_counter()
        try:
            pinned_auctions = sample_df["_bid_str"].unique().to_list()
            for auc in pinned_auctions:
                if not auc:
                    continue
                try:
                    # Normalize to seat-1 view for lookup
                    auc_seat1 = re.sub(r"(?i)^(p-)+", "", normalize_auction_input(str(auc)))
                    bt_idx = _resolve_bt_index_by_traversal(state, auc_seat1)
                    if bt_idx is not None:
                        pinned_auction_bt_indices.add(int(bt_idx))
                except Exception:
                    pass
            elapsed_resolve = (time.perf_counter() - t_pinned_resolve) * 1000
            if debug:
                print(f"[bidding-arena] PINNED-ONLY MODE: {len(pinned_auctions)} auctions -> {len(pinned_auction_bt_indices)} bt_indices ({elapsed_resolve:.1f}ms)")
        except Exception as e:
            if debug:
                print(f"[bidding-arena] Pinned mode optimization failed: {e}, falling back to full candidate pool")
            pinned_only_mode = False
    _mark("pinned_only_resolve")
    
    # ---------------------------------------------------------------------------
    # Rules Auction Matching Strategy
    # ---------------------------------------------------------------------------
    # Use GPU-verified deal-to-BT index when available (bbo_deal_to_bt_verified.parquet).
    # This index is already bitmap-verified, so we can skip _deal_meets_all_seat_criteria.
    deal_to_bt_index_df: Optional[pl.DataFrame] = state.get("deal_to_bt_index_df")
    has_verified_index = deal_to_bt_index_df is not None and deal_to_bt_index_df.height > 0
    has_precomputed_matches = has_verified_index  # Only trust the verified index
    
    # Build fast O(log n) lookup using numpy arrays (DataFrame is pre-sorted by deal_idx)
    import numpy as np
    # IMPORTANT: do NOT rebuild numpy/list materializations per-request.
    # This was costing ~16-17s every Arena run. Cache in module globals.
    global _BIDDING_ARENA_VERIFIED_INDEX_CACHE  # type: ignore[declared-but-unused]
    try:
        _BIDDING_ARENA_VERIFIED_INDEX_CACHE  # type: ignore[name-defined]
    except Exception:
        _BIDDING_ARENA_VERIFIED_INDEX_CACHE = {}  # type: ignore[name-defined]

    _deal_idx_arr: Optional[np.ndarray] = None
    _matched_bt_series: Optional[pl.Series] = None
    if has_verified_index and deal_to_bt_index_df is not None:
        cache = _BIDDING_ARENA_VERIFIED_INDEX_CACHE  # type: ignore[name-defined]
        df_id = id(deal_to_bt_index_df)
        cached = cache.get("df_id")
        if cached != df_id:
            # Refresh cache (one-time per process / per new df object)
            cache["df_id"] = df_id
            cache["deal_idx_arr"] = deal_to_bt_index_df["deal_idx"].to_numpy()
            # Keep as Series to avoid huge Python list materialization.
            cache["matched_series"] = deal_to_bt_index_df.get_column("Matched_BT_Indices")
        _deal_idx_arr = cache.get("deal_idx_arr")
        _matched_bt_series = cache.get("matched_series")
    _mark("verified_index_setup")
    
    def _get_verified_matches(deal_idx: int) -> list[int]:
        """Get verified BT matches for a deal using O(log n) binary search."""
        if _deal_idx_arr is None or _matched_bt_series is None:
            return []
        idx = np.searchsorted(_deal_idx_arr, deal_idx)
        if idx < len(_deal_idx_arr) and _deal_idx_arr[idx] == deal_idx:
            # IMPORTANT: Polars Series indexing wants a Python int, not numpy.int64.
            try:
                idx_i = int(idx)
            except Exception:
                return []
            try:
                matches = _matched_bt_series[idx_i]
            except Exception:
                return []
            if matches is None:
                return []
            # Polars list element may be returned as a Python list or as a Series.
            try:
                if isinstance(matches, pl.Series):
                    return [int(x) for x in matches.to_list() if x is not None]
            except Exception:
                pass
            if isinstance(matches, (list, tuple)):
                return [int(x) for x in matches if x is not None]
            # Fallback: try to iterate
            try:
                return [int(x) for x in list(matches) if x is not None]
            except Exception:
                return []
        return []

    # Rules candidate source:
    # - Default: use all completed BT rows (merged rules are pre-compiled)
    # - Backstop: fall back to generic on-the-fly candidate pool (bt_seat1 completed auctions)
    rules_search_mode = "merged_default"
    rules_search_limit: int | None = None
    rules_search_limit_fallback: int | None = None
    # Branch-specific structures (initialized for type-checkers)
    bt_idx_to_row: dict[int, dict[str, Any]] = {}
    bt_idx_to_row_base: dict[int, dict[str, Any]] = {}  # Without overlay (for Rules model)
    bt_auction_to_row_base: dict[str, dict[str, Any]] = {}
    bt_auction_to_row_full_base: dict[str, dict[str, Any]] = {}
    bt_completed_rows: list[dict[str, Any]] = []
    bt_completed_rows_base: list[dict[str, Any]] = []  # Backstop pool (without overlay)
    bt_rules_default_rows_base: list[dict[str, Any]] = []  # Default pool (without overlay)
    RULES_MATCHES_MAX_RETURNED = 200  # UI payload guardrail
        
    def _auction_key_seat1(auction: Any) -> str:
        """Canonical seat-1 lookup key for auctions.
        
        - Normalizes tokens via normalize_auction_input (1nt/1n -> 1N, pass -> P)
        - Strips leading passes (seat-1 view)
        - UPPERCASE for canonical form (stable dict keys)
        """
        if auction is None:
            return ""
        try:
            s = normalize_auction_input(str(auction))
        except Exception:
            s = str(auction)
        s = re.sub(r"(?i)^(P-)+", "", s)
        return s.upper()
    
    def _first_failure_for_bt_row(
        deal_idx: int,
        dealer: str,
        bt_row: Dict[str, Any],
        deal_row: Dict[str, Any] | None = None,
    ) -> str | None:
        """Return a short 'first failure' string for debugging, or None if all criteria pass."""
        try:
            d = str(dealer or "N").upper()
        except Exception:
            d = "N"
        # Optional seat mapping metadata (when bt_row has been rotated for leading passes).
        try:
            lead_passes = int(bt_row.get("_lead_passes", 0) or 0)
        except Exception:
            lead_passes = 0
        for seat in SEAT_RANGE:
            criteria_list = bt_row.get(agg_expr_col(seat)) or []
            if not criteria_list:
                continue
            criteria_df = (deal_criteria_by_seat_dfs.get(seat, {}) or {}).get(d)
            if criteria_df is None or criteria_df.is_empty():
                seat_label = _format_seat_notation(d, seat, lead_passes=lead_passes, include_bt_seat=True)
                return f"{seat_label}: missing bitmap dealer={d}"
            for criterion in criteria_list:
                crit_s = str(criterion).strip()
                # Dynamic SL evaluation if possible
                if deal_row is not None:
                    sl_result = evaluate_sl_criterion(crit_s, d, seat, deal_row, fail_on_missing=False)
                    if sl_result is True:
                        continue
                    if sl_result is False:
                        seat_label = _format_seat_notation(d, seat, lead_passes=lead_passes, include_bt_seat=True)
                        annotated = annotate_criterion_with_value(crit_s, d, seat, deal_row)
                        return f"{seat_label}: failed {annotated}"
                # Bitmap evaluation
                if crit_s not in criteria_df.columns:
                    continue
                try:
                    if not bool(criteria_df[crit_s][deal_idx]):
                        seat_label = _format_seat_notation(d, seat, lead_passes=lead_passes, include_bt_seat=True)
                        # Annotate if deal_row is available
                        if deal_row is not None:
                            annotated = annotate_criterion_with_value(crit_s, d, seat, deal_row)
                        else:
                            annotated = crit_s
                        return f"{seat_label}: failed {annotated}"
                except (IndexError, KeyError, TypeError):
                    seat_label = _format_seat_notation(d, seat, lead_passes=lead_passes, include_bt_seat=True)
                    return f"{seat_label}: bitmap lookup error for {crit_s}"
        return None

    def _deal_meets_all_seat_criteria(deal_idx: int, dealer: str, bt_row: Dict[str, Any], deal_row: Dict[str, Any] | None = None) -> bool:
        """Check if a deal meets ALL seat criteria for a BT row.
        
        IMPORTANT: Unknown criteria (not in bitmap, typically from CSV overlay) 
        are treated as FAIL. This ensures CSV overlay can add blocking criteria.
        
        If deal_row is provided, SL (suit length) criteria are evaluated dynamically
        to ensure correct seat-direction mapping.
        """
        for seat in SEAT_RANGE:
            criteria_list = bt_row.get(agg_expr_col(seat))
            if not criteria_list:
                continue  # No criteria for this seat = passes
            
            seat_dfs = deal_criteria_by_seat_dfs.get(seat, {})
            criteria_df = seat_dfs.get(dealer)
            if criteria_df is None or criteria_df.is_empty():
                return False  # Can't verify = fail
            
            for criterion in criteria_list:
                crit_s = str(criterion).strip()
                
                # Try dynamic SL evaluation first if deal_row is provided
                if deal_row is not None:
                    sl_result = evaluate_sl_criterion(crit_s, dealer, seat, deal_row, fail_on_missing=False)
                    if sl_result is True:
                        continue  # SL criterion passed, skip bitmap lookup
                    elif sl_result is False:
                        return False  # SL criterion failed
                    # sl_result is None - not an SL criterion OR hand data missing, fall through to bitmap
                
                # Bitmap lookup for non-SL criteria
                if crit_s not in criteria_df.columns:
                    # Unknown criterion - skip it (can't verify, assume passes)
                    # Note: This means CSV overlay criteria not in bitmap will be IGNORED
                    # for matching purposes. Dynamic SL evaluation handles SL criteria.
                    continue
                try:
                    if not bool(criteria_df[crit_s][deal_idx]):
                        return False  # Failed this criterion
                except (IndexError, KeyError):
                    return False
        return True
        
    def _find_best_rules_match_precomputed(
        deal_idx: int,
        dealer: str,
        matched_indices: List[int] | None,
        deal_row: Dict[str, Any] | None = None,
        skip_criteria_check: bool = False,
    ) -> str | None:
        """Pick the best matching auction among precomputed candidates."""
        if not matched_indices:
            return None
        valid_matches = []
        for bt_idx in matched_indices:
            bt_idx_i = int(bt_idx)
            bt_row = bt_idx_to_row.get(bt_idx_i)
            if bt_row is None:
                continue
            # When using verified index, skip expensive criteria check
            if skip_criteria_check or _deal_meets_all_seat_criteria(deal_idx, dealer, bt_row, deal_row):
                valid_matches.append({"bt_index": bt_idx_i, "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        
        if not deal_row:
            return str(valid_matches[0]["auction"]) if valid_matches else None
            
        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row)

    def _find_best_rules_match_onthefly(deal_idx: int, dealer: str, deal_row: Dict[str, Any] | None = None) -> str | None:
        """Pick the best matching auction from the on-the-fly candidate pool."""
        valid_matches = []
        for bt_row in bt_completed_rows:
            if _deal_meets_all_seat_criteria(deal_idx, dealer, bt_row, deal_row):
                valid_matches.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        
        if not deal_row:
            return str(valid_matches[0]["auction"]) if valid_matches else None

        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row)

    if has_precomputed_matches:
        # Use precomputed candidate indices but re-check against (base + overlay) criteria so the CSV can override.
        overlay = state.get("custom_criteria_overlay") or []

        # Collect all needed bt_indices from verified index for the sampled deals
        needed_idxs: list[int] = []
        if has_verified_index:
            # Get deal indices from sample_df and look up in verified index
            try:
                sample_deal_idxs = sample_df["_row_idx"].to_list() if "_row_idx" in sample_df.columns else []
                for deal_idx in sample_deal_idxs:
                    needed_idxs.extend(_get_verified_matches(int(deal_idx)))
                needed_idxs = list(set(needed_idxs))  # Dedupe
            except Exception:
                needed_idxs = []
        else:
            # Fallback: try to get from sample_df column
            try:
                matched_series = sample_df.get_column("Matched_BT_Indices")
                needed_idxs = (
                    matched_series
                    .explode()
                    .drop_nulls()
                    .unique()
                    .to_list()
                )
            except Exception:
                needed_idxs = []

        bt_idx_to_auction: dict[int, str] = {}
        # Keyed by a seat-1 normalized, lowercased auction string (leading passes stripped).
        # (Initialized above for type-checkers; populated here.)
        if needed_idxs:
            try:
                cols = ["bt_index", "Auction", "matching_deal_count", agg_expr_col(1), agg_expr_col(2), agg_expr_col(3), agg_expr_col(4)]
                cols = [c for c in cols if c in bt_seat1_df.columns]
                lookup_df = bt_seat1_df.select(cols).filter(pl.col("bt_index").is_in(needed_idxs))
                lookup_rows_raw = lookup_df.to_dicts()
                
                # Apply Overlay + Dedup (merged rules are pre-compiled in BT)
                overlay = state.get("custom_criteria_overlay") or []
                
                lookup_rows = []
                for r in lookup_rows_raw:
                    p = dict(r)
                    if overlay:
                        p = _apply_overlay_only(p, overlay)
                    p = dedupe_criteria_all_seats(p)
                    lookup_rows.append(p)

                # Store pre-processed rows
                for r in lookup_rows:
                    bt_idx = r.get("bt_index")
                    if bt_idx is not None:
                        bt_idx_i = int(bt_idx)
                        bt_idx_to_row_base[bt_idx_i] = r # Base rows in this scope are already merged
                        bt_idx_to_row[bt_idx_i] = r
                        auc = r.get("Auction")
                        if auc:
                            bt_idx_to_auction[bt_idx_i] = str(auc)
                            k = _auction_key_seat1(auc)
                            bt_auction_to_row_base[k] = r
            except Exception:
                bt_idx_to_row = {}
                bt_idx_to_row_base = {}
                bt_idx_to_auction = {}

        def _find_rules_auction_precomputed(deal_idx: int, dealer: str, matched_indices: List[int] | None, skip_criteria_check: bool = False) -> Optional[str]:
            """Pick the first precomputed candidate that satisfies (base + overlay) criteria for this deal.
            
            If skip_criteria_check is True (verified index), we trust the precomputed matches.
            """
            return _find_best_rules_match_precomputed(deal_idx, dealer, matched_indices, skip_criteria_check=skip_criteria_check)
        _mark("candidate_pool_precomputed")
    else:
        # Candidate pool(s) for criteria matching (no deal-level Matched_BT_Indices)
        if "is_completed_auction" in bt_seat1_df.columns:
            bt_completed = bt_seat1_df.filter(pl.col("is_completed_auction"))
        else:
            bt_completed = bt_seat1_df

        # Rules candidate pool: use completed auctions (merged rules are pre-compiled in BT)
        bt_rules_default = bt_completed
        
        # -------------------------------------------------------------------
        # PINNED-ONLY OPTIMIZATION: Use minimal candidate pool
        # -------------------------------------------------------------------
        # When only processing pinned deals, we only need BT rows for those
        # specific auctions, not 5,000 random candidates.
        if pinned_only_mode and pinned_auction_bt_indices:
            t_pinned_pool = time.perf_counter()
            pinned_idx_list = list(pinned_auction_bt_indices)
            bt_rules_default = bt_completed.filter(pl.col("bt_index").is_in(pinned_idx_list))
            bt_completed = bt_rules_default  # Same pool for pinned-only mode
            elapsed_pool = (time.perf_counter() - t_pinned_pool) * 1000
            print(f"[bidding-arena] Pinned pool: {bt_rules_default.height} rows ({elapsed_pool:.1f}ms)")
        elif not search_all_bt_rows:
            # Standard path: Limit pools for performance (default + fallback)
            if "matching_deal_count" in bt_rules_default.columns:
                bt_rules_default = bt_rules_default.sort("matching_deal_count", descending=True).head(_CANDIDATE_POOL_LIMIT)
            elif "bt_index" in bt_rules_default.columns:
                bt_rules_default = bt_rules_default.sort("bt_index").head(_CANDIDATE_POOL_LIMIT)
            else:
                bt_rules_default = bt_rules_default.head(_CANDIDATE_POOL_LIMIT)

            if "matching_deal_count" in bt_completed.columns:
                bt_completed = bt_completed.sort("matching_deal_count", descending=True).head(_CANDIDATE_POOL_LIMIT)
            elif "bt_index" in bt_completed.columns:
                bt_completed = bt_completed.sort("bt_index").head(_CANDIDATE_POOL_LIMIT)
            else:
                bt_completed = bt_completed.head(_CANDIDATE_POOL_LIMIT)

        # -------------------------------------------------------------------
        # Memory optimization: bt_seat1_df is loaded without Agg_Expr_Seat_*.
        # The Bidding Arena NEEDS criteria to evaluate candidates, so load
        # Agg_Expr for JUST the candidate pools on-demand (DuckDB, fast).
        # -------------------------------------------------------------------
        if "Agg_Expr_Seat_1" not in bt_rules_default.columns or "Agg_Expr_Seat_1" not in bt_completed.columns:
            bt_parquet_file = state.get("bt_seat1_file")
            if bt_parquet_file is None:
                raise ValueError("bt_seat1_file missing from state; cannot load Agg_Expr on-demand for bidding-arena")
            if "bt_index" not in bt_rules_default.columns or "bt_index" not in bt_completed.columns:
                raise ValueError("bt_index missing from BT candidate pool; cannot load Agg_Expr on-demand for bidding-arena")

            # For pinned-only mode, we already have the minimal set of indices
            if pinned_only_mode and pinned_auction_bt_indices:
                needed_idxs = list(pinned_auction_bt_indices)
                print(f"[bidding-arena] Loading Agg_Expr for {len(needed_idxs)} pinned bt_indices (vs {_CANDIDATE_POOL_LIMIT} normally)")
            else:
                needed_idxs = (
                    pl.concat(
                        [
                            bt_rules_default.select(pl.col("bt_index").cast(pl.Int64)),
                            bt_completed.select(pl.col("bt_index").cast(pl.Int64)),
                        ],
                        how="vertical",
                    )
                    .unique()
                    .to_series()
                    .to_list()
                )
            needed_idxs = [int(x) for x in needed_idxs if x is not None]
            
            if needed_idxs:
                t_agg_load = time.perf_counter()
                agg_data = _load_agg_expr_for_bt_indices(needed_idxs, bt_parquet_file)
                elapsed_agg = (time.perf_counter() - t_agg_load) * 1000
                if debug:
                    print(f"[bidding-arena] Agg_Expr load: {len(needed_idxs)} indices in {elapsed_agg:.1f}ms")
                agg_rows: list[dict[str, Any]] = []
                for bt_idx, cols_dict in agg_data.items():
                    row: dict[str, Any] = {"bt_index": bt_idx}
                    for k, v in cols_dict.items():
                        row[k] = v
                    agg_rows.append(row)
                if agg_rows:
                    agg_df = pl.DataFrame(agg_rows)
                    bt_rules_default = bt_rules_default.join(agg_df, on="bt_index", how="left")
                    bt_completed = bt_completed.join(agg_df, on="bt_index", how="left")
        _mark("candidate_pool_load_agg_expr")
        
        # Convert pools to list of dicts for iteration
        bt_cols = ["Auction", "bt_index", "matching_deal_count", agg_expr_col(1), agg_expr_col(2), agg_expr_col(3), agg_expr_col(4)]
        bt_cols = [c for c in bt_cols if c in bt_completed.columns]

        # NOTE: We intentionally do NOT build a global auction->bt_index map here.
        # It was extremely expensive on the 461M-row BT and is not used by the Arena.

        # PRE-PROCESS CANDIDATE POOLS: Apply Overlay + Dedup (merged rules pre-compiled in BT)
        overlay = state.get("custom_criteria_overlay") or []
        
        def _fully_pre_process_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = []
            for r in rows:
                processed = dict(r)
                # Apply custom CSV overlay
                if overlay:
                    processed = _apply_overlay_only(processed, overlay)
                # Deduplicate criteria (keep least restrictive)
                processed = dedupe_criteria_all_seats(processed)
                out.append(processed)
            return out

        bt_rules_default_rows_base = _fully_pre_process_rows(
            bt_rules_default.select(bt_cols).to_dicts() if bt_rules_default is not None else []
        )
        bt_completed_rows_base = _fully_pre_process_rows(
            bt_completed.select(bt_cols).to_dicts()
        )
        _mark("candidate_pool_preprocess")

        # Build Auction->row caches for fast lookup.
        # - default: merged-rules pool only (preferred when searching candidates)
        # - full: generic completed pool (used ONLY for "is actual auction in BT?" checks)
        def _build_bt_auction_cache(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for r in rows:
                auc = r.get("Auction")
                if not auc:
                    continue
                k = _auction_key_seat1(auc)
                if not k:
                    continue
                if k not in out:
                    out[k] = r
                    continue
                try:
                    old_passes = _count_leading_passes(out[k].get("Auction"))
                except Exception:
                    old_passes = 99
                try:
                    new_passes = _count_leading_passes(auc)
                except Exception:
                    new_passes = 99
                if new_passes < old_passes:
                    out[k] = r
            return out

        bt_auction_to_row_base = _build_bt_auction_cache(bt_rules_default_rows_base)
        bt_auction_to_row_full_base = _build_bt_auction_cache(bt_completed_rows_base)
        rules_search_limit = len(bt_rules_default_rows_base)
        rules_search_limit_fallback = len(bt_completed_rows_base)
        
        # DIAGNOSTIC: Show first few auctions from the candidates list
        top_auctions = [r.get("Auction") for r in bt_rules_default_rows_base[:5]]
        if rules_search_mode == "merged_default":
            print(f"[bidding-arena] merged-default {len(bt_rules_default_rows_base)} candidates (top 5: {top_auctions})")
        else:
            print(f"[bidding-arena] {len(bt_completed_rows_base)} candidates (top 5: {top_auctions})")
        
        # IMPORTANT: These rows are now fully pre-processed.
        bt_completed_rows = bt_completed_rows_base
        
        def _find_rules_auction_onthefly(deal_idx: int, dealer: str) -> Optional[str]:
            """Find the first completed auction whose criteria ALL match this deal."""
            return _find_best_rules_match_onthefly(deal_idx, dealer)
    
    # ---------------------------------------------------------------------------
    # Optimization: Deal-Level Criteria Cache (LAZY)
    #
    # IMPORTANT: Agg_Expr criteria strings can have very high cardinality (thousands+).
    # Pre-evaluating "all unique criteria in the candidate pool" per deal can be slower
    # than the actual search. Instead, evaluate criteria lazily on first use and cache.
    # ---------------------------------------------------------------------------
    class DealCriteriaCache:
        def __init__(self, deal_idx: int, dealer: str, deal_row: Dict[str, Any]):
            self.deal_idx = int(deal_idx)
            self.dealer = str(dealer or "N").upper()
            self.deal_row = deal_row
            # Per-seat caches: criterion -> bool result
            self.results: Dict[int, Dict[str, bool]] = {s: {} for s in SEAT_RANGE}
            # Keep references to bitmap dfs + a set of column names for O(1) membership
            self.criteria_df_by_seat: Dict[int, pl.DataFrame] = {}
            self.criteria_cols_by_seat: Dict[int, set[str]] = {}
            for s in SEAT_RANGE:
                df = (deal_criteria_by_seat_dfs.get(s, {}) or {}).get(self.dealer)
                if df is None or df.is_empty():
                    continue
                self.criteria_df_by_seat[s] = df
                self.criteria_cols_by_seat[s] = set(df.columns)

        def check(self, seat: int, criterion: Any) -> bool:
            """Return whether this deal satisfies a single criterion for a given seat."""
            try:
                s = int(seat)
            except Exception:
                return False
            crit_s = str(criterion).strip()
            if not crit_s:
                return True
            cached = self.results[s].get(crit_s)
            if cached is not None:
                return cached

            # Dynamic SL/complex evaluation is expensive; only attempt for SL_* or logical expressions.
            crit_up = crit_s.upper()
            maybe_dynamic = (
                ("SL_" in crit_up)
                or ("&" in crit_s)
                or ("|" in crit_s)
                or ("(" in crit_s)
                or (")" in crit_s)
                or (" AND " in crit_up)
                or (" OR " in crit_up)
                or (" NOT " in crit_up)
            )
            if maybe_dynamic:
                sl_res = evaluate_sl_criterion(crit_s, self.dealer, s, self.deal_row, fail_on_missing=False)
                if sl_res is not None:
                    self.results[s][crit_s] = bool(sl_res)
                    return bool(sl_res)

            # Bitmap evaluation
            df = self.criteria_df_by_seat.get(s)
            if df is None:
                self.results[s][crit_s] = False
                return False
            if crit_s not in self.criteria_cols_by_seat.get(s, set()):
                # Unknown / untracked criterion => pass (consistent with current matching semantics)
                self.results[s][crit_s] = True
                return True
            try:
                ok = bool(df[crit_s][self.deal_idx])
            except Exception:
                ok = False
            self.results[s][crit_s] = ok
            return ok

        def meets_all(self, bt_row: Dict[str, Any]) -> bool:
            for s in SEAT_RANGE:
                col = f"Agg_Expr_Seat_{s}"
                crits = bt_row.get(col) or []
                if not crits:
                    continue
                for crit in crits:
                    if not self.check(int(s), crit):
                        return False
            return True

    def _deal_meets_all_seat_criteria_cached(cache: DealCriteriaCache, bt_row: Dict[str, Any]) -> bool:
        return cache.meets_all(bt_row)

    # ---------------------------------------------------------------------------
    # Rules Model Matching Setup (Learned Criteria)
    # ---------------------------------------------------------------------------
    # Rules model uses learned criteria pre-compiled into bbo_bt_compiled.parquet.
    # No runtime rule application needed - just use BT data directly.
    
    use_rules = (model_a in (MODEL_RULES, MODEL_RULES_LEARNED)) or (model_b in (MODEL_RULES, MODEL_RULES_LEARNED))
    
    def _apply_overlay_to_bt_row(bt_row: Dict[str, Any]) -> Dict[str, Any]:
        """Apply overlay to BT row. Merged rules are pre-compiled."""
        return _apply_overlay_and_dedupe(bt_row, state)
    
    def _deal_meets_merged_criteria(cache: DealCriteriaCache, bt_row: Dict[str, Any]) -> bool:
        """Check if a deal meets criteria. bt_row is already pre-processed."""
        return _deal_meets_all_seat_criteria_cached(cache, bt_row)

    def _deal_meets_learned_criteria(cache: DealCriteriaCache, bt_row: Dict[str, Any]) -> bool:
        """Check if a deal meets criteria. bt_row is already pre-processed."""
        return _deal_meets_all_seat_criteria_cached(cache, bt_row)

    def _find_best_learned_rules_match_precomputed(
        cache: DealCriteriaCache,
        matched_indices: List[int] | None,
        deal_row: dict[str, Any],
        dealer: str
    ) -> str | None:
        if not matched_indices:
            return None
        valid_matches = []
        for bt_idx in matched_indices:
            bt_idx_i = int(bt_idx)
            bt_row = bt_idx_to_row_base.get(bt_idx_i)
            if bt_row and _deal_meets_learned_criteria(cache, bt_row):
                valid_matches.append({"bt_index": bt_idx_i, "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row_base)

    def _find_best_learned_rules_match_default(
        cache: DealCriteriaCache,
        deal_row: dict[str, Any],
        dealer: str
    ) -> str | None:
        valid_matches = []
        for bt_row in bt_rules_default_rows_base:
            if _deal_meets_learned_criteria(cache, bt_row):
                valid_matches.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row_base)

    def _find_best_learned_rules_match_onthefly(
        cache: DealCriteriaCache,
        deal_row: dict[str, Any],
        dealer: str
    ) -> str | None:
        valid_matches = []
        for bt_row in bt_completed_rows_base:
            if _deal_meets_learned_criteria(cache, bt_row):
                valid_matches.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row_base)
    
    def _find_best_merged_rules_match_precomputed(
        cache: DealCriteriaCache,
        matched_indices: List[int] | None,
        deal_row: dict[str, Any],
        dealer: str
    ) -> str | None:
        if not matched_indices:
            return None
        
        # Collect all valid matches from the precomputed list
        valid_matches = []
        for bt_idx in matched_indices:
            bt_idx_i = int(bt_idx)
            bt_row = bt_idx_to_row_base.get(bt_idx_i)
            if bt_row and _deal_meets_merged_criteria(cache, bt_row):
                valid_matches.append({"bt_index": bt_idx_i, "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        
        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row_base)
    
    def _find_best_merged_rules_match_onthefly(
        cache: DealCriteriaCache,
        deal_row: dict[str, Any],
        dealer: str
    ) -> str | None:
        # Collect valid matches from completed rows
        valid_matches = []
        for bt_row in bt_completed_rows_base:
            if _deal_meets_merged_criteria(cache, bt_row):
                valid_matches.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        
        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row_base)

    def _find_best_merged_rules_match_default(
        cache: DealCriteriaCache,
        deal_row: dict[str, Any],
        dealer: str
    ) -> str | None:
        """Default path: search the merged-rules candidate pool, choose the best by EV/popularity."""
        valid_matches = []
        for bt_row in bt_rules_default_rows_base:
            if _deal_meets_merged_criteria(cache, bt_row):
                valid_matches.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction"), "matching_deal_count": bt_row.get("matching_deal_count", 0)})
                if len(valid_matches) >= RULES_MATCHES_MAX_RETURNED:
                    break
        
        return choose_best_auction_match(valid_matches, deal_row, dealer, bt_idx_to_row_lookup=bt_idx_to_row_base)
    
    def _find_merged_rules_matches_precomputed(
        cache: DealCriteriaCache,
        matched_indices: List[int] | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return all matching BT candidates using merged rules (up to max returned)."""
        if not matched_indices:
            return [], False
        out: list[dict[str, Any]] = []
        truncated = False
        for bt_idx in matched_indices:
            bt_idx_i = int(bt_idx)
            bt_row = bt_idx_to_row_base.get(bt_idx_i)
            if bt_row is None:
                continue
            if _deal_meets_merged_criteria(cache, bt_row):
                out.append({"bt_index": bt_idx_i, "auction": bt_row.get("Auction")})
                if len(out) >= RULES_MATCHES_MAX_RETURNED:
                    truncated = True
                    break
        return out, truncated

    def _find_merged_rules_matches_onthefly(cache: DealCriteriaCache) -> tuple[list[dict[str, Any]], bool]:
        """Return matching BT candidates using merged rules, default pool first (up to max returned)."""
        out: list[dict[str, Any]] = []
        truncated = False
        for bt_row in bt_rules_default_rows_base:
            if _deal_meets_merged_criteria(cache, bt_row):
                out.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction")})
                if len(out) >= RULES_MATCHES_MAX_RETURNED:
                    truncated = True
                    break
        # Backstop: generic completed rows (avoid duplicates)
        if not truncated and len(out) < RULES_MATCHES_MAX_RETURNED:
            seen = set((m.get("bt_index"), m.get("auction")) for m in out)
            for bt_row in bt_completed_rows_base:
                if _deal_meets_merged_criteria(cache, bt_row):
                    item = {"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction")}
                    key = (item.get("bt_index"), item.get("auction"))
                    if key in seen:
                        continue
                    out.append(item)
                    seen.add(key)
                    if len(out) >= RULES_MATCHES_MAX_RETURNED:
                        truncated = True
                        break
        return out, truncated
    
    # Metrics accumulators
    model_a_wins = 0
    model_b_wins = 0
    ties = 0
    total_imp_a = 0
    total_imp_b = 0
    imp_diffs: List[int] = []
    sum_dd_a = 0
    sum_dd_b = 0
    sample_deals_output: List[Dict[str, Any]] = []  # Output list for processed sample deals
    
    # Diagnostic counters for debugging match rate
    diag_no_matched_indices = 0
    diag_rules_none = 0
    diag_dd_a_none = 0
    diag_dd_b_none = 0
    diag_dd_b_examples: list[str] = []
    diag_first_failure_reasons: list[str] = []
    
    # Contract agreement
    same_contract = 0
    same_strain = 0
    same_level = 0
    same_declarer = 0
    contracts_compared = 0
    
    # Contract quality per model
    quality_a = {
        "makes": 0, "fails": 0, "par_matches": 0,
        "slams_bid": 0, "slams_make": 0, "slams_fail": 0, "slam_opportunities": 0,
        "games_bid": 0, "games_make": 0, "games_fail": 0, "game_opportunities": 0,
        "partscores_bid": 0, "partscores_make": 0, "partscores_fail": 0,
        "total_overtricks": 0, "contracts_with_result": 0,
    }
    quality_b = {
        "makes": 0, "fails": 0, "par_matches": 0,
        "slams_bid": 0, "slams_make": 0, "slams_fail": 0, "slam_opportunities": 0,
        "games_bid": 0, "games_make": 0, "games_fail": 0, "game_opportunities": 0,
        "partscores_bid": 0, "partscores_make": 0, "partscores_fail": 0,
        "total_overtricks": 0, "contracts_with_result": 0,
    }
    
    # Segmentation
    by_vulnerability = {
        "None": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
        "NS": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
        "EW": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
        "Both": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
    }
    by_hcp_range: Dict[str, Dict[str, int]] = {}
    by_dealer = {
        "N": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
        "E": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
        "S": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
        "W": {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0},
    }
    
    def _rejection_reason_for(model: str, auction_val: Any, dd_score_val: Any, dealer: str) -> str | None:
        """Return a short reason string if this model can't be compared for this deal."""
        if model == MODEL_ACTUAL:
            if dd_score_val is None:
                return f"{MODEL_ACTUAL}: missing DD_Score_Declarer"
            return None
        # Rules model variant
        model_name = model
        if auction_val is None:
            # Distinguish "no match" from "can't verify because criteria bitmaps are missing for this dealer".
            try:
                d = str(dealer or "N").upper()
                missing_seats = []
                for s in SEAT_RANGE:
                    df = deal_criteria_by_seat_dfs.get(s, {}).get(d)
                    if df is None or df.is_empty():
                        missing_seats.append(s)
                if missing_seats:
                    return f"{model_name}: no matching auction (missing/empty criteria bitmap for dealer={d}, seat(s)={missing_seats})"
            except Exception:
                pass
            if rules_search_mode == "onthefly" and rules_search_limit is not None:
                return f"{model_name}: no matching auction after criteria (searched top {rules_search_limit} completed auctions)"
            return f"{model_name}: no matching auction after criteria"
        if dd_score_val is None:
            return f"{model_name}: DD score unavailable for chosen auction"
        return None
    
    # Process each deal
    # Limit sample output to min(sample_size, 200) to avoid excessive memory usage
    max_sample_output = min(sample_size, 200)
    sample_deals = sample_df.to_dicts()
    _mark("materialize_sample_rows")
    t_loop0 = time.perf_counter()
    # Cap expensive "actual auction lookup" debug work; otherwise it can dominate runtime.
    _actual_lookup_budget = 2
    _actual_lookup_used = 0
    for row in sample_deals:
        deal_idx = row.get("_row_idx", 0)
        # Robustly handle Dealer (could be string or int)
        dealer_val = row.get("Dealer", "N")
        if isinstance(dealer_val, int):
            dealer = ["N", "E", "S", "W"][dealer_val % 4]
        elif isinstance(dealer_val, str):
            dealer = dealer_val.upper()
            if dealer not in ["N", "E", "S", "W"]:
                dealer = "N"
        else:
            dealer = "N"
        
        # Ensure dealer is a string for all subsequent calls
        dealer = str(dealer)
        
        # Create criteria cache once per deal for efficient repeated criteria checks
        deal_cache = DealCriteriaCache(int(deal_idx), dealer, row)
        
        vul = row.get("Vul", "None")
        bid_str = row.get("_bid_str", "")
        par_score = row.get("ParScore")
        
        # Get opener's HCP (seat 1)
        dir_map = _seat_direction_map(1)
        opener_dir = dir_map.get(dealer, "N")
        opener_hcp = row.get(f"HCP_{opener_dir}")
        hcp_label = _hcp_range_label(opener_hcp)
        if hcp_label not in by_hcp_range:
            by_hcp_range[hcp_label] = {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0}
        
        # ---------------------------------------------------------------------------
        # Get auction for each model
        # ---------------------------------------------------------------------------
        def _rules_no_match_debug(deal_idx_i: int, dealer_dir: str | int, matched_indices: List[int] | None) -> str:
            """Best-effort diagnostic string when Rules can't find any matching BT auction.

            Uses merged rules + CSV overlay (the Rules path), not Rules_Base.
            We only call this for a small number of sampled deals, so it can afford to check a few candidates.
            """
            try:
                if isinstance(dealer_dir, int):
                    d = ["N", "E", "S", "W"][dealer_dir % 4]
                else:
                    d = str(dealer_dir or "N").upper()
                    if d not in ["N", "E", "S", "W"]:
                        d = "N"
                # Decide candidate list based on matching strategy
                # Use base rows and apply merged rules + CSV overlay (same as Rules model)
                candidates: list[dict[str, Any]] = []
                if has_precomputed_matches:
                    if not matched_indices:
                        return "no candidates (Matched_BT_Indices empty)"
                    # Use up to N candidates from the precomputed list, apply overlay
                    for bt_idx in matched_indices[:_DEBUG_CANDIDATE_SAMPLE]:
                        bt_row = bt_idx_to_row_base.get(int(bt_idx))
                        if bt_row is not None:
                            # Apply overlay + dedupe (merged rules are pre-compiled)
                            candidates.append(_apply_overlay_to_bt_row(bt_row))
                else:
                    # On-the-fly path: use base rows, apply merged rules + overlay
                    # Default pool first, then fallback pool
                    for bt_row in bt_rules_default_rows_base[:_DEBUG_CANDIDATE_SAMPLE]:
                        candidates.append(_apply_overlay_to_bt_row(bt_row))
                    if len(candidates) < _DEBUG_CANDIDATE_SAMPLE:
                        for bt_row in bt_completed_rows_base[:_DEBUG_CANDIDATE_SAMPLE]:
                            candidates.append(_apply_overlay_to_bt_row(bt_row))

                if not candidates:
                    return "no candidates available"

                # Show variety of candidate auctions
                unique_auctions = list(set(str(c.get("Auction", "?")) for c in candidates[:10]))[:5]
                
                # Diagnostic: how many matched_indices are actually found in bt_idx_to_row_base?
                if matched_indices:
                    found_count = sum(1 for idx in matched_indices[:_DEBUG_CANDIDATE_SAMPLE] if bt_idx_to_row_base.get(int(idx)) is not None)
                    total_count = min(_DEBUG_CANDIDATE_SAMPLE, len(matched_indices))
                    if found_count < total_count:
                        return f"MISMATCH: only {found_count}/{total_count} matched_indices found in bt_idx_to_row_base (missing rows in bt_seat1_df?)"

                # Check each candidate and find WHY it failed
                # Collect failures for first few candidates to show diversity
                candidate_failures: list[str] = []
                for cand in candidates:
                    cand_failed = False
                    cand_failure_info = ""
                    for seat in SEAT_RANGE:
                        crits = cand.get(agg_expr_col(seat)) or []
                        if not crits:
                            continue
                        criteria_df = deal_criteria_by_seat_dfs.get(seat, {}).get(d)
                        if criteria_df is None or criteria_df.is_empty():
                            cand_failed = True
                            cand_failure_info = f"missing bitmap (dealer={d}, seat={seat})"
                            break
                        failed: list[str] = []
                        for c in crits:
                            if c not in criteria_df.columns:
                                continue  # Skip untracked (same as actual matching)
                            try:
                                if not bool(criteria_df[c][int(deal_idx_i)]):
                                    failed.append(str(c))
                            except Exception:
                                pass
                        if failed:
                            auc = cand.get("Auction")
                            seat_label = _format_seat_notation(d, seat, include_bt_seat=False)
                            cand_failure_info = f"{auc}: {seat_label} failed {failed[:3]}"
                            cand_failed = True
                            break
                    if cand_failed and len(candidate_failures) < 3:
                        candidate_failures.append(cand_failure_info)
                    elif not cand_failed:
                        # This candidate should have matched - unexpected!
                        auc = cand.get("Auction")
                        return f"unexpected: candidate {auc} passed all criteria but wasn't matched"
                
                if candidate_failures:
                    return f"checked {len(candidates)} candidates (auctions: {unique_auctions}); failures: {candidate_failures}"
                return f"checked {len(candidates)} candidates; no failures found (unexpected)"
            except Exception as e:
                return f"debug failed: {e}"

        need_rules_final = (model_a == MODEL_RULES) or (model_b == MODEL_RULES)  # Stage 3/3
        need_rules_learned = (model_a == MODEL_RULES_LEARNED) or (model_b == MODEL_RULES_LEARNED)  # Stage 2/3
        need_rules = need_rules_final or need_rules_learned
        need_rules_base = (model_a == MODEL_RULES_BASE) or (model_b == MODEL_RULES_BASE)  # Stage 1/3
        need_rules_matches_list = need_rules and (len(sample_deals) < _NEED_RULES_MATCHES_LIST_THRESHOLD)

        # Pick a single Rules auction (learned criteria) for scoring
        # Pass `row` as deal_row for dynamic SL evaluation
        # Use verified index if available (O(log n) lookup, already bitmap-verified)
        matched_indices = _get_verified_matches(int(deal_idx)) if has_precomputed_matches else None
        # Always define these for the sample-deals output schema (even if Rules model not requested).
        rules_actual_lead_passes: int | None = None
        rules_actual_opener_seat: int | None = None
        rules_learned_auction: str | None = None
        if need_rules_final:
            # Avoid the expensive "actual auction lookup" path in candidate-pool-only mode.
            # When caches are limited (top N candidates), attempting exact lookup can
            # devolve into per-deal traversal + per-deal parquet reads.
            allow_actual_lookup = bool(
                (pinned_only_mode or search_all_bt_rows)
                and (_actual_lookup_used < _actual_lookup_budget)
            )
            # PRIORITIZE: Try actual auction first if it's in BT
            rules_auction = None
            rules_actual_bt_lookup: str = ""
            rules_actual_criteria_ok: bool | None = None
            rules_actual_first_failure: str = ""
            rules_actual_bt_index: int | None = None
            rules_actual_seat1_criteria: str = ""
            rules_actual_base_criteria_by_seat: str = ""
            rules_actual_merged_criteria_by_seat: str = ""
            rules_actual_merged_only_criteria_by_seat: str = ""

            # Optional: Use step-by-step "model rules" (greedy path) instead of auction matching.
            # This mirrors the UI's "Model's Predicted Path" behavior (opening-only).
            if use_model_rules:
                try:
                    greedy = handle_greedy_model_path(
                        state=state,
                        auction_prefix="",
                        deal_row_idx=int(deal_idx),
                        seed=int(seed or DEFAULT_SEED),
                        max_depth=40,
                    )
                    rules_auction = str(greedy.get("greedy_path") or "") or None
                    rules_actual_bt_lookup = "model_rules"
                except Exception as e:
                    rules_auction = None
                    rules_actual_bt_lookup = f"model_rules_error:{e}"
            if (not use_model_rules) and bid_str and allow_actual_lookup:
                _actual_lookup_used += 1
                # Normalize the deal's auction to the BT's seat-1 view for lookup.
                bid_norm_full = normalize_auction_input(str(bid_str))
                bid_str_seat1 = re.sub(r"(?i)^(P-)+", "", bid_norm_full)
                bid_key = bid_str_seat1.upper()  # Canonical UPPERCASE
                lead_passes = _count_leading_passes(bid_norm_full)
                rules_actual_lead_passes = int(lead_passes)
                rules_actual_opener_seat = int((lead_passes % 4) + 1)

                def _rotate_bt_row_for_leading_passes(bt_row: Dict[str, Any], n_passes: int) -> Dict[str, Any]:
                    """Rotate a BT row's seat-indexed columns to match a deal with leading passes.
                    
                    The BT is stored in seat-1 view (no leading passes). For a deal auction like
                    'p-p-p-1N-p-p-p', the opener is seat 4 relative to dealer, so we must rotate
                    Agg_Expr_Seat_* (and other seat-indexed columns) accordingly before evaluating.
                    """
                    try:
                        n = int(n_passes)
                    except Exception:
                        n = 0
                    # Always attach metadata for debugging / seat mapping clarity.
                    if n <= 0:
                        out0 = dict(bt_row)
                        out0["_lead_passes"] = 0
                        out0["_opener_seat"] = 1
                        return out0
                    try:
                        expanded = _expand_row_to_all_seats(bt_row, allow_initial_passes=True)
                        if 0 <= n < len(expanded):
                            outn = dict(expanded[n])
                            outn["_lead_passes"] = n
                            # expand_row_to_all_seats already sets _opener_seat; keep it if present.
                            outn.setdefault("_opener_seat", n + 1)
                            return outn
                    except Exception:
                        pass
                    out_fallback = dict(bt_row)
                    out_fallback["_lead_passes"] = n
                    out_fallback["_opener_seat"] = (n % 4) + 1
                    return out_fallback

                def _criteria_by_seat_str(bt_row: Dict[str, Any], dealer_dir: str, lead_passes: int) -> str:
                    parts: list[str] = []
                    for s in SEAT_RANGE:
                        try:
                            crits = bt_row.get(agg_expr_col(s)) or []
                            if crits:
                                txt = "; ".join(str(x) for x in crits[:_DEBUG_CRITERIA_PREVIEW])
                            else:
                                txt = "(no criteria)"
                            seat_label = _format_seat_notation(
                                dealer_dir,
                                int(s),
                                lead_passes=int(lead_passes or 0),
                                include_bt_seat=True,
                            )
                            parts.append(f"{seat_label}: {txt}")
                        except Exception:
                            parts.append(f"S{s}: (error)")
                    return " | ".join(parts)

                def _copy_bt_row_without_overlay(bt_row: Dict[str, Any]) -> Dict[str, Any]:
                    """Return a copy of the BT row without applying CSV overlay.
                    
                    Used when we need the pre-compiled rules without overlay modifications.
                    """
                    return dict(bt_row)

                bt_row_actual = bt_auction_to_row_base.get(bid_key)
                if bt_row_actual is None:
                    # Handle common variants with/without the trailing "-P-P-P".
                    if bid_key.endswith("-P-P-P"):
                        bt_row_actual = bt_auction_to_row_base.get(bid_key[: -len("-P-P-P")])
                    else:
                        bt_row_actual = bt_auction_to_row_base.get(bid_key + "-P-P-P")
                if bt_row_actual is None:
                    # Fallback: O(1) lookup in the full completed-auctions cache (avoid per-deal Polars scans)
                    bt_row_actual = bt_auction_to_row_full_base.get(bid_key)
                    if bt_row_actual is None:
                        if bid_key.endswith("-P-P-P"):
                            bt_row_actual = bt_auction_to_row_full_base.get(bid_key[: -len("-P-P-P")])
                        else:
                            bt_row_actual = bt_auction_to_row_full_base.get(bid_key + "-P-P-P")
                    if bt_row_actual is not None:
                        rules_actual_bt_lookup = "found_bt_full"
                    else:
                        # As a last resort, resolve by traversal (no Auction scans).
                        try:
                            bt_idx = _resolve_bt_index_by_traversal(state, bid_str_seat1)
                            if bt_idx is not None:
                                file_path = _bt_file_path_for_sql(state)
                                conn = duckdb.connect(":memory:")
                                try:
                                    bt_df_row = conn.execute(
                                        f"""
                                        SELECT bt_index, Auction, {agg_expr_col(1)}, {agg_expr_col(2)}, {agg_expr_col(3)}, {agg_expr_col(4)}
                                        FROM read_parquet('{file_path}')
                                        WHERE bt_index = {int(bt_idx)}
                                        LIMIT 1
                                        """
                                    ).pl()
                                finally:
                                    conn.close()
                                if not bt_df_row.is_empty():
                                    bt_row_actual = dict(bt_df_row.row(0, named=True))
                                    rules_actual_bt_lookup = "found_bt_traversal"
                                    if len(diag_first_failure_reasons) < 3:
                                        diag_first_failure_reasons.append(f"FOUND by traversal: bt_index={bt_idx}")
                        except Exception as e:
                            rules_actual_bt_lookup = "lookup_error"
                            if len(diag_first_failure_reasons) < 3:
                                diag_first_failure_reasons.append(f"TRAVERSAL LOOKUP ERROR: {e}")
                
                if bt_row_actual:
                    if not rules_actual_bt_lookup:
                        rules_actual_bt_lookup = "found_bt_candidates"
                    try:
                        bt_idx_val = bt_row_actual.get("bt_index")
                        rules_actual_bt_index = int(bt_idx_val) if bt_idx_val is not None else None
                    except Exception:
                        rules_actual_bt_index = None
                    try:
                        # Apply merged rules in seat-1 view first, then rotate to the deal's leading-pass seat.
                        merged_bt_row_actual = _apply_overlay_to_bt_row(bt_row_actual)
                        merged_bt_row_actual = _rotate_bt_row_for_leading_passes(merged_bt_row_actual, lead_passes)
                    except Exception:
                        merged_bt_row_actual = _rotate_bt_row_for_leading_passes(bt_row_actual, lead_passes)

                    # Diagnostic: compare rotated base vs rotated merged+overlay criteria.
                    try:
                        base_rot = _rotate_bt_row_for_leading_passes(bt_row_actual, lead_passes)
                        rules_actual_base_criteria_by_seat = _criteria_by_seat_str(base_rot, dealer, lead_passes)
                    except Exception:
                        rules_actual_base_criteria_by_seat = ""
                    try:
                        merged_only = _copy_bt_row_without_overlay(bt_row_actual)
                        merged_only_rot = _rotate_bt_row_for_leading_passes(merged_only, lead_passes)
                        rules_actual_merged_only_criteria_by_seat = _criteria_by_seat_str(merged_only_rot, dealer, lead_passes)
                    except Exception:
                        rules_actual_merged_only_criteria_by_seat = ""
                    try:
                        rules_actual_merged_criteria_by_seat = _criteria_by_seat_str(merged_bt_row_actual, dealer, lead_passes)
                    except Exception:
                        rules_actual_merged_criteria_by_seat = ""

                    meets = _deal_meets_all_seat_criteria(int(deal_idx), dealer, merged_bt_row_actual, row)
                    rules_actual_criteria_ok = bool(meets)

                    # Capture "seat 1" criteria used for UI/debug (note: after rotation, seat 1 is dealer-relative).
                    try:
                        s1 = merged_bt_row_actual.get(agg_expr_col(1)) or []
                        rules_actual_seat1_criteria = "; ".join(str(x) for x in s1[:_RULES_CRITERIA_PREVIEW])
                    except Exception:
                        rules_actual_seat1_criteria = ""

                    if meets:
                        # Keep the deal's full auction (including leading passes) for correct declarer/scoring.
                        rules_auction = bid_norm_full
                    else:
                        try:
                            ff = _first_failure_for_bt_row(
                                int(deal_idx),
                                dealer,
                                merged_bt_row_actual,
                                row,
                            )
                            rules_actual_first_failure = ff or ""
                        except Exception:
                            rules_actual_first_failure = ""
                        if len(diag_first_failure_reasons) < 5:
                            diag_first_failure_reasons.append(f"CRITERIA FAILED [{bid_norm_full}]")
                        rules_auction = None
                else:
                    # Auction not found in BT at all
                    if not rules_actual_bt_lookup:
                        rules_actual_bt_lookup = "not_in_bt"
                    if len(diag_first_failure_reasons) < 5:
                        diag_first_failure_reasons.append(f"NOT IN BT [{bid_str}]")
            elif bid_str and not allow_actual_lookup:
                rules_actual_bt_lookup = "skipped_actual_lookup"
            
            # If actual auction didn't match (or isn't in BT), search other candidates:
            # 1) default: merged-rules candidate pool
            # 2) fallback: generic on-the-fly candidate pool
            if (not use_model_rules) and rules_auction is None:
                if has_precomputed_matches:
                    if not matched_indices:
                        diag_no_matched_indices += 1
                    rules_auction = _find_best_merged_rules_match_precomputed(deal_cache, matched_indices, row, dealer)
                else:
                    rules_auction = _find_best_merged_rules_match_default(deal_cache, row, dealer)
                    if rules_auction is None and rules_search_mode == "merged_default":
                        rules_auction = _find_best_merged_rules_match_onthefly(deal_cache, row, dealer)
            
            if rules_auction is None:
                diag_rules_none += 1
                # Capture first 5 failure reasons for debugging
                if len(diag_first_failure_reasons) < 5:
                    debug_reason = _rules_no_match_debug(int(deal_idx), dealer, matched_indices)
                    diag_first_failure_reasons.append(debug_reason)
        else:
            rules_auction = None
            rules_actual_bt_lookup = ""
            rules_actual_criteria_ok = None
            rules_actual_first_failure = ""
            rules_actual_bt_index = None
            rules_actual_seat1_criteria = ""
            rules_actual_base_criteria_by_seat = ""
            rules_actual_merged_criteria_by_seat = ""
            rules_actual_merged_only_criteria_by_seat = ""

        # Compute Rules_Learned auction (stage 2/3) only if requested.
        if need_rules_learned:
            if has_precomputed_matches:
                mi2 = _get_verified_matches(int(deal_idx))
                rules_learned_auction = _find_best_learned_rules_match_precomputed(deal_cache, mi2, row, dealer)
            else:
                rules_learned_auction = _find_best_learned_rules_match_default(deal_cache, row, dealer)
                if rules_learned_auction is None and rules_search_mode == "merged_default":
                    rules_learned_auction = _find_best_learned_rules_match_onthefly(deal_cache, row, dealer)
        
        # Pick Rules_Base auction (original BT criteria)
        if need_rules_base:
            if has_precomputed_matches:
                # Use verified index (O(1) lookup, already bitmap-verified)
                matched_indices = _get_verified_matches(int(deal_idx))
                rules_base_auction = _find_best_rules_match_precomputed(
                    int(deal_idx), dealer, matched_indices, row, skip_criteria_check=True
                )
            else:
                rules_base_auction = _find_best_rules_match_onthefly(int(deal_idx), dealer, row)
        else:
            rules_base_auction = None

        # Collect all matches only for deals that are surfaced in Sample Deal Comparisons.
        # Uses merged rules (the new Rules model) for consistency with rules_auction
        rules_matches: list[dict[str, Any]] = []
        rules_matches_truncated = False
        if need_rules_matches_list:
            if has_precomputed_matches:
                matched_indices = _get_verified_matches(int(deal_idx))
                rules_matches, rules_matches_truncated = _find_merged_rules_matches_precomputed(deal_cache, matched_indices)
            else:
                rules_matches, rules_matches_truncated = _find_merged_rules_matches_onthefly(deal_cache)
        
        # Helper to get auction result for a model
        def _get_model_result(model: str) -> Tuple[Any, Any, Any]:
            """Get (contract, auction, dd_score) for a model.
            
            All auction strings are normalized to canonical uppercase for consistent comparison.
            """
            if model == MODEL_ACTUAL:
                auction_out = normalize_auction_case(bid_str)
                return row.get("Contract", ""), auction_out, row.get("DD_Score_Declarer")
            elif model == MODEL_RULES:
                if rules_auction:
                    auction_out = normalize_auction_case(rules_auction)
                    return (
                        get_ai_contract(rules_auction, dealer),
                        auction_out,
                        get_dd_score_for_auction(rules_auction, dealer, row),
                    )
                return None, None, None
            elif model == MODEL_RULES_LEARNED:
                if rules_learned_auction:
                    auction_out = normalize_auction_case(rules_learned_auction)
                    return (
                        get_ai_contract(rules_learned_auction, dealer),
                        auction_out,
                        get_dd_score_for_auction(rules_learned_auction, dealer, row),
                    )
                return None, None, None
            elif model == MODEL_RULES_BASE:
                if rules_base_auction:
                    auction_out = normalize_auction_case(rules_base_auction)
                    return (
                        get_ai_contract(rules_base_auction, dealer),
                        auction_out,
                        get_dd_score_for_auction(rules_base_auction, dealer, row),
                    )
                return None, None, None
            return None, None, None
        
        # Model A
        contract_a, auction_a, dd_score_a = _get_model_result(model_a)
        
        # Model B
        contract_b, auction_b, dd_score_b = _get_model_result(model_b)

        # Optional: compute a third "Model" auction (greedy path) for display-only comparison.
        model_auction: str | None = None
        model_contract: str | None = None
        model_dd_score: int | None = None
        if include_model_auction:
            try:
                greedy = handle_greedy_model_path(
                    state=state,
                    auction_prefix="",
                    deal_row_idx=int(deal_idx),
                    seed=int(seed or DEFAULT_SEED),
                    max_depth=40,
                )
                model_auction = str(greedy.get("greedy_path") or "") or None
                if model_auction:
                    model_contract = get_ai_contract(model_auction, dealer)
                    ddv = get_dd_score_for_auction(model_auction, dealer, row)
                    model_dd_score = int(ddv) if ddv is not None else None
            except Exception:
                model_auction = None
                model_contract = None
                model_dd_score = None
        
        # Collect sample deals (include rejected deals, with reason)
        if len(sample_deals_output) < max_sample_output:
            reason_a = _rejection_reason_for(model_a, auction_a, dd_score_a, dealer)
            reason_b = _rejection_reason_for(model_b, auction_b, dd_score_b, dealer)
            rejection_reason = reason_a or reason_b

            # Add a small diagnostic for the common confusion case: Rules has no match.
            rules_debug = ""
            if (model_a != MODEL_ACTUAL and auction_a is None) or (model_b != MODEL_ACTUAL and auction_b is None):
                mi = _get_verified_matches(int(deal_idx)) if has_precomputed_matches else None
                rules_debug = _rules_no_match_debug(int(deal_idx), dealer, mi)

            auctions_match = (auction_a == auction_b) if auction_a and auction_b else False
            rules_matches_auctions_str = ""
            try:
                if rules_matches:
                    rules_matches_auctions_str = ", ".join(
                        normalize_auction_case(str(m.get("auction"))) for m in rules_matches if m and m.get("auction") is not None
                    )
            except Exception:
                rules_matches_auctions_str = ""
            # Normalize rules_auction to canonical case for display
            rules_auction_display = normalize_auction_case(rules_auction) if rules_auction else None
            sample_deals_output.append(
                {
                    "index": row.get("index"),
                    "_row_idx": row.get("_row_idx"),
                    "Dealer": dealer,
                    "Vul": vul,
                    "Hand_N": row.get("Hand_N"),
                    "Hand_E": row.get("Hand_E"),
                    "Hand_S": row.get("Hand_S"),
                    "Hand_W": row.get("Hand_W"),
                    f"Auction_{model_a}": auction_a,
                    f"Auction_{model_b}": auction_b,
                    "Auction_Model": (normalize_auction_case(model_auction) if model_auction else None),
                    "Model_Contract": model_contract,
                    "DD_Score_Model": model_dd_score,
                    "Rules_Actual_BT_Lookup": rules_actual_bt_lookup,
                    "Rules_Actual_BT_Index": rules_actual_bt_index,
                    "Rules_Actual_Lead_Passes": rules_actual_lead_passes,
                    "Rules_Actual_Opener_Seat": rules_actual_opener_seat,
                    "Rules_Actual_Criteria_OK": rules_actual_criteria_ok,
                    "Rules_Actual_First_Failure": rules_actual_first_failure,
                    "Rules_Actual_Seat1_Criteria": rules_actual_seat1_criteria,
                    "Rules_Actual_BT_Base_Criteria_By_Seat": rules_actual_base_criteria_by_seat,
                    "Rules_Actual_BT_MergedOnly_Criteria_By_Seat": rules_actual_merged_only_criteria_by_seat,
                    "Rules_Actual_BT_Merged_Criteria_By_Seat": rules_actual_merged_criteria_by_seat,
                    "Rules_Matches_Count": len(rules_matches),
                    "Rules_Matches_Truncated": bool(rules_matches_truncated),
                    # String form only (avoid List[Null] / List[str] inference problems in Polars/AgGrid)
                    "Rules_Matches_Auctions_Str": rules_matches_auctions_str,
                    "Auction_Rules_Selected": rules_auction_display,
                    f"DD_Score_{model_a}": (int(dd_score_a) if dd_score_a is not None else None),
                    f"DD_Score_{model_b}": (int(dd_score_b) if dd_score_b is not None else None),
                    "IMP_Diff": None,
                    "Auctions_Match": auctions_match,
                    "Rejection_Reason": rejection_reason or "",
                    "Rules_NoMatch_Debug": rules_debug,
                    # Additional fields for display
                    "Contract": row.get("Contract", ""),
                    "Result": row.get("Result", ""),
                    "Score": row.get("Score", ""),
                    "ParScore": row.get("ParScore", par_score),
                }
            )

        # Skip if we don't have scores for both models
        if dd_score_a is None or dd_score_b is None:
            if dd_score_a is None:
                diag_dd_a_none += 1
            if dd_score_b is None:
                diag_dd_b_none += 1
                if len(diag_dd_b_examples) < 3:
                    try:
                        dd_cols = [k for k in row.keys() if str(k).startswith("DD_Score_")]
                        diag_dd_b_examples.append(
                            f"rules_auction={rules_auction!r}, dealer={dealer!r}, dd_cols_preview={dd_cols[:5]}"
                        )
                    except Exception:
                        diag_dd_b_examples.append("dd_score_b None (failed to inspect row keys)")
            continue
        
        contracts_compared += 1
        
        # Head-to-head comparison (from A's perspective)
        score_a = int(dd_score_a)
        score_b = int(dd_score_b)
        score_diff = score_a - score_b
        imp_diff_mag = calculate_imp(abs(score_diff))
        if score_diff > 0:
            model_a_wins += 1
            total_imp_a += imp_diff_mag
            imp_signed = imp_diff_mag
        elif score_diff < 0:
            model_b_wins += 1
            total_imp_b += imp_diff_mag
            imp_signed = -imp_diff_mag
        else:
            ties += 1
            imp_signed = 0
        imp_diffs.append(imp_signed)

        # Sum DD scores for averages
        sum_dd_a += score_a
        sum_dd_b += score_b

        # Patch in IMP_Diff for the last appended sample row if it corresponds to this deal.
        # (We always append before the skip-check, so at this point the last row is this deal.)
        if sample_deals_output:
            sample_deals_output[-1]["IMP_Diff"] = imp_signed
            sample_deals_output[-1]["Rejection_Reason"] = ""
        
        # Segmentation updates
        if vul in by_vulnerability:
            by_vulnerability[vul]["count"] += 1
            if score_diff > 0:
                by_vulnerability[vul]["a_wins"] += 1
            elif score_diff < 0:
                by_vulnerability[vul]["b_wins"] += 1
            else:
                by_vulnerability[vul]["ties"] += 1
        
        if dealer in by_dealer:
            by_dealer[dealer]["count"] += 1
            if score_diff > 0:
                by_dealer[dealer]["a_wins"] += 1
            elif score_diff < 0:
                by_dealer[dealer]["b_wins"] += 1
            else:
                by_dealer[dealer]["ties"] += 1
        
        by_hcp_range[hcp_label]["count"] += 1
        if score_diff > 0:
            by_hcp_range[hcp_label]["a_wins"] += 1
        elif score_diff < 0:
            by_hcp_range[hcp_label]["b_wins"] += 1
        else:
            by_hcp_range[hcp_label]["ties"] += 1
        
        # Contract agreement
        if contract_a and contract_b:
            parsed_a = _parse_contract(contract_a)
            parsed_b = _parse_contract(contract_b)
            
            if contract_a.upper().replace("X", "") == contract_b.upper().replace("X", ""):
                same_contract += 1
            if parsed_a["strain"] == parsed_b["strain"]:
                same_strain += 1
            if parsed_a["level"] == parsed_b["level"]:
                same_level += 1
            # Same declarer side - check from auction
            # For now, approximate by checking if both are passed out or both aren't
            if parsed_a["is_pass"] == parsed_b["is_pass"]:
                same_declarer += 1
        
        # Contract quality - Model A
        if dd_score_a is not None:
            quality_a["contracts_with_result"] += 1
            if int(dd_score_a) >= 0:
                quality_a["makes"] += 1
            else:
                quality_a["fails"] += 1
            if par_score is not None and int(dd_score_a) == int(par_score):
                quality_a["par_matches"] += 1
            
            if contract_a:
                if _is_slam(contract_a):
                    quality_a["slams_bid"] += 1
                    if int(dd_score_a) >= 0:
                        quality_a["slams_make"] += 1
                    else:
                        quality_a["slams_fail"] += 1
                elif _is_game(contract_a):
                    quality_a["games_bid"] += 1
                    if int(dd_score_a) >= 0:
                        quality_a["games_make"] += 1
                    else:
                        quality_a["games_fail"] += 1
                elif _is_partscore(contract_a):
                    quality_a["partscores_bid"] += 1
                    if int(dd_score_a) >= 0:
                        quality_a["partscores_make"] += 1
                    else:
                        quality_a["partscores_fail"] += 1
        
        # Contract quality - Model B
        if dd_score_b is not None:
            quality_b["contracts_with_result"] += 1
            if int(dd_score_b) >= 0:
                quality_b["makes"] += 1
            else:
                quality_b["fails"] += 1
            if par_score is not None and int(dd_score_b) == int(par_score):
                quality_b["par_matches"] += 1
            
            if contract_b:
                if _is_slam(contract_b):
                    quality_b["slams_bid"] += 1
                    if int(dd_score_b) >= 0:
                        quality_b["slams_make"] += 1
                    else:
                        quality_b["slams_fail"] += 1
                elif _is_game(contract_b):
                    quality_b["games_bid"] += 1
                    if int(dd_score_b) >= 0:
                        quality_b["games_make"] += 1
                    else:
                        quality_b["games_fail"] += 1
                elif _is_partscore(contract_b):
                    quality_b["partscores_bid"] += 1
                    if int(dd_score_b) >= 0:
                        quality_b["partscores_make"] += 1
                    else:
                        quality_b["partscores_fail"] += 1
    
    timings_ms["deal_loop_ms"] = round((time.perf_counter() - t_loop0) * 1000, 1)

    # Calculate derived metrics
    def calc_rates(q: Dict[str, int]) -> Dict[str, Any]:
        total = q["contracts_with_result"]
        if total == 0:
            return {
                "make_rate": 0.0, "par_rate": 0.0,
                "slam_accuracy": None, "game_accuracy": None,
                "partscore_accuracy": None,
            }
        
        slam_total = q["slams_bid"]
        game_total = q["games_bid"]
        partscore_total = q["partscores_bid"]
        
        return {
            "make_rate": round(q["makes"] / total, 4) if total > 0 else 0.0,
            "par_rate": round(q["par_matches"] / total, 4) if total > 0 else 0.0,
            "slam_accuracy": round(q["slams_make"] / slam_total, 4) if slam_total > 0 else None,
            "slam_bid_count": slam_total,
            "game_accuracy": round(q["games_make"] / game_total, 4) if game_total > 0 else None,
            "game_bid_count": game_total,
            "partscore_accuracy": round(q["partscores_make"] / partscore_total, 4) if partscore_total > 0 else None,
            "partscore_bid_count": partscore_total,
        }
    
    quality_a_rates = calc_rates(quality_a)
    quality_b_rates = calc_rates(quality_b)
    
    # Swing analysis (deals with ≥5 IMP difference)
    swing_threshold = 5
    swings_for_a = sum(1 for d in imp_diffs if d >= swing_threshold)
    swings_for_b = sum(1 for d in imp_diffs if d <= -swing_threshold)
    
    # Build response-level aggregates
    avg_imp_diff = sum(imp_diffs) / len(imp_diffs) if imp_diffs else 0.0
    avg_dd_a = sum_dd_a / contracts_compared if contracts_compared > 0 else 0.0
    avg_dd_b = sum_dd_b / contracts_compared if contracts_compared > 0 else 0.0

    # Summary block expected by Streamlit UI
    summary: Dict[str, Any] = {
        "total_deals": contracts_compared,
        f"avg_dd_score_{model_a.lower()}": avg_dd_a,
        f"avg_dd_score_{model_b.lower()}": avg_dd_b,
        f"avg_imp_{model_a.lower()}_vs_{model_b.lower()}": avg_imp_diff,
    }

    # Head-to-head block with dynamic keys per model
    head_to_head: Dict[str, Any] = {
        f"{model_a.lower()}_wins": model_a_wins,
        f"{model_b.lower()}_wins": model_b_wins,
        "ties": ties,
        "model_a_wins": model_a_wins,
        "model_b_wins": model_b_wins,
        "model_a_win_rate": round(model_a_wins / contracts_compared, 4) if contracts_compared > 0 else 0.0,
        "model_b_win_rate": round(model_b_wins / contracts_compared, 4) if contracts_compared > 0 else 0.0,
        "avg_imp_diff": round(avg_imp_diff, 2),  # Positive favors A
        "total_imp_a": total_imp_a,
        "total_imp_b": total_imp_b,
        "net_imp_for_a": total_imp_a - total_imp_b,
        "swings_for_a": swings_for_a,
        "swings_for_b": swings_for_b,
        "swing_threshold": swing_threshold,
    }

    # Contract quality block expected by UI
    contract_quality: Dict[str, Dict[str, Any]] = {
        "Make Rate": {
            model_a.lower(): quality_a_rates.get("make_rate", 0.0),
            model_b.lower(): quality_b_rates.get("make_rate", 0.0),
        },
        "Par Rate": {
            model_a.lower(): quality_a_rates.get("par_rate", 0.0),
            model_b.lower(): quality_b_rates.get("par_rate", 0.0),
        },
        "Slam Accuracy": {
            model_a.lower(): quality_a_rates.get("slam_accuracy"),
            model_b.lower(): quality_b_rates.get("slam_accuracy"),
        },
        "Game Accuracy": {
            model_a.lower(): quality_a_rates.get("game_accuracy"),
            model_b.lower(): quality_b_rates.get("game_accuracy"),
        },
        "Partscore Accuracy": {
            model_a.lower(): quality_a_rates.get("partscore_accuracy"),
            model_b.lower(): quality_b_rates.get("partscore_accuracy"),
        },
    }

    # Segmentation block expected by UI
    segmentation: Dict[str, Any] = {
        "by_vulnerability": [
            {
                "Vul": vul,
                **{
                    "count": data["count"],
                    "a_win_rate": round(data["a_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                    "b_win_rate": round(data["b_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                    "tie_rate": round(data["ties"] / data["count"], 4) if data["count"] > 0 else 0.0,
                },
            }
            for vul, data in by_vulnerability.items()
            if data["count"] > 0
        ],
        "by_dealer": [
            {
                "Dealer": dealer,
                **{
                    "count": data["count"],
                    "a_win_rate": round(data["a_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                    "b_win_rate": round(data["b_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                },
            }
            for dealer, data in by_dealer.items()
            if data["count"] > 0
        ],
        "by_hcp_range": [
            {
                "HCP_Range": hcp,
                **{
                    "count": data["count"],
                    "a_win_rate": round(data["a_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                    "b_win_rate": round(data["b_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                },
            }
            for hcp, data in sorted(by_hcp_range.items())
            if data["count"] > 0
        ],
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    timings_ms["total_ms"] = round(elapsed_ms, 1)
    print(f"[bidding-arena] {format_elapsed(elapsed_ms)} ({contracts_compared}/{analyzed_deals} compared)")
    # Diagnostic spam is debug-only (set STATE['debug_bidding_arena']=True to enable)
    if debug:
        print(
            f"[bidding-arena] DIAG: no_matched_indices={diag_no_matched_indices}, rules_none={diag_rules_none}, "
            f"dd_a_none={diag_dd_a_none}, dd_b_none={diag_dd_b_none}"
        )
        if diag_dd_b_examples:
            for i, ex in enumerate(diag_dd_b_examples[:3]):
                print(f"[bidding-arena] DIAG dd_b_none[{i+1}]: {ex[:250]}")
        try:
            # One-liner breakdown for quick perf comparisons.
            parts = ", ".join(f"{k}={v}ms" for k, v in timings_ms.items())
            print(f"[bidding-arena] TIMING: {parts}")
        except Exception:
            pass
        # Show actual auctions from sampled deals
        actual_auctions_sample = [d.get("Auction_Actual", d.get("_bid_str", "?"))[:30] for d in sample_deals_output[:10]]
        print(f"[bidding-arena] DIAG actual auctions (first 10): {actual_auctions_sample}")
        # Show raw Matched_BT_Indices to check if data is correct
        if sample_deals_output and len(sample_deals_output) >= 3:
            for i in range(3):
                deal = sample_deals_output[i]
                matched = deal.get("Matched_BT_Indices", [])
                matched_preview = matched[:3] if matched else "NONE"
                auction = deal.get("Auction_Actual", "?")[:25]
                print(f"[bidding-arena] DIAG deal[{i}] auction='{auction}', Matched_BT_Indices[:3]={matched_preview}")
        if diag_first_failure_reasons:
            print(f"[bidding-arena] DIAG first failures:")
            for i, reason in enumerate(diag_first_failure_reasons[:3]):
                print(f"  [{i+1}] {reason[:200]}")
    
    return {
        "model_a": model_a,
        "model_b": model_b,
        "deals_source": deals_source,
        "rules_search": {
            "mode": rules_search_mode,
            "search_all_bt_rows": bool(search_all_bt_rows),
            "candidate_count": rules_search_limit,
            "fallback_candidate_count": rules_search_limit_fallback,
            "note": (
                "mode=merged_default means Rules first searches BT rows with learned criteria (pre-compiled in BT), "
                "then falls back to the generic on-the-fly pool if no match is found."
            ),
        },
        "total_deals": total_deals,
        "analyzed_deals": analyzed_deals,
        "deals_compared": contracts_compared,
        "auction_pattern": auction_pattern,
        "timings_ms": timings_ms,
        "summary": summary,
        "head_to_head": head_to_head,
        "contract_agreement": {
            "same_contract": same_contract,
            "same_contract_rate": round(same_contract / contracts_compared, 4) if contracts_compared > 0 else 0.0,
            "same_strain": same_strain,
            "same_strain_rate": round(same_strain / contracts_compared, 4) if contracts_compared > 0 else 0.0,
            "same_level": same_level,
            "same_level_rate": round(same_level / contracts_compared, 4) if contracts_compared > 0 else 0.0,
            "same_declarer": same_declarer,
            "same_declarer_rate": round(same_declarer / contracts_compared, 4) if contracts_compared > 0 else 0.0,
        },
        "quality_by_model": {
            model_a: quality_a_rates,
            model_b: quality_b_rates,
        },
        "contract_quality": contract_quality,
        "by_vulnerability": {
            vul: {
                "count": data["count"],
                "a_win_rate": round(data["a_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                "b_win_rate": round(data["b_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                "tie_rate": round(data["ties"] / data["count"], 4) if data["count"] > 0 else 0.0,
            }
            for vul, data in by_vulnerability.items() if data["count"] > 0
        },
        "by_dealer": {
            dealer: {
                "count": data["count"],
                "a_win_rate": round(data["a_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                "b_win_rate": round(data["b_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
            }
            for dealer, data in by_dealer.items() if data["count"] > 0
        },
        "by_hcp_range": {
            hcp: {
                "count": data["count"],
                "a_win_rate": round(data["a_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
                "b_win_rate": round(data["b_wins"] / data["count"], 4) if data["count"] > 0 else 0.0,
            }
            for hcp, data in sorted(by_hcp_range.items()) if data["count"] > 0
        },
        "segmentation": segmentation,
        "sample_deals": sample_deals_output,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Handler: /auction-dd-analysis
# ---------------------------------------------------------------------------

# Constants for DD analysis
_DD_ANALYSIS_STATS_THRESHOLD = _CANDIDATE_POOL_LIMIT  # Use full data if <= this many matches
_DD_ANALYSIS_MAKE_PCT_THRESHOLD = 0  # Show all contracts (0 = no threshold)
_DD_ANALYSIS_TOP_RECOMMENDATIONS = 100  # Max contract recommendations to show (high = show all)
_DD_ANALYSIS_SQL_COLS_LIMIT = 15  # Limit columns in SQL display

# Seat labels for display
_SEAT_LABELS = {
    1: "Seat 1 (Opener/Dealer)",
    2: "Seat 2 (LHO)", 
    3: "Seat 3 (Partner)",
    4: "Seat 4 (RHO)",
}

# Valid vulnerability values (data uses underscores: N_S, E_W)
_VALID_VUL_VALUES = {"None", "Both", "NS", "EW", "N_S", "E_W"}

# Map UI values to data values (UI uses "NS"/"EW", data uses "N_S"/"E_W")
_VUL_UI_TO_DATA = {
    "NS": "N_S",
    "EW": "E_W",
    "None": "None",
    "Both": "Both",
}


def _build_criteria_mask_for_dealer(
    deal_df: pl.DataFrame,
    dealer: str,
    criteria_by_seat: Dict[int, List[str]],
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, pl.DataFrame]],
) -> Tuple[Optional[pl.Series], List[str]]:
    """Build a mask matching criteria for a specific dealer.

    IMPORTANT (pipeline note):
    - Criteria of the form `SL_S >= SL_H` (and related suit-length relational comparisons) are currently
      evaluated dynamically here by parsing `Hand_{N/E/S/W}`.
    - For best performance and consistency with the "pre-computed bitmaps" architecture, the bitmap
      creator should add explicit bitmap columns for suit-length relational comparisons, e.g.:
        - `SL_S >= SL_H`, `SL_S >= SL_D`, `SL_S >= SL_C`
        - and the other suit pairs / operators that appear in BT criteria
      so these do not need to be computed on the fly.
    """
    def _seat_dir_for_dealer(dealer_dir: str, seat: int) -> str:
        """Seat 1 is dealer; seat 2 is LHO; seat 3 is partner; seat 4 is RHO."""
        try:
            dealer_i = DIRECTIONS.index(str(dealer_dir))
        except Exception:
            dealer_i = 0
        seat_i = max(1, min(4, int(seat)))
        return DIRECTIONS[(dealer_i + seat_i - 1) % 4]

    def _sl_len_expr(direction: str, suit: str) -> pl.Expr:
        """Polars expression to get suit length from hand column."""
        hand_col = f"Hand_{direction}"
        idx = SUIT_IDX[suit]
        # Hand format: "S.H.D.C" (e.g., "AKQJ.T98.765.32")
        return (
            pl.col(hand_col)
            .cast(pl.Utf8)
            .str.split(".")
            .list.get(idx)
            .str.len_chars()
        )

    dealer_mask = deal_df["Dealer"] == dealer
    if not dealer_mask.any():
        return None, []
        
    all_seats_mask = dealer_mask
    invalid_criteria: List[str] = []
    
    for seat, criteria_list in criteria_by_seat.items():
        seat_criteria_df = deal_criteria_by_seat_dfs.get(seat, {}).get(dealer)
        
        if seat_criteria_df is None or seat_criteria_df.is_empty():
            return None, []
        
        available_cols = set(seat_criteria_df.columns)
        
        for crit in criteria_list:
            if crit not in available_cols:
                # Try dynamic suit-length comparisons (not precomputed as bitmap columns).
                parsed = parse_sl_comparison_relative(str(crit))
                if parsed is None:
                    if crit not in invalid_criteria:
                        invalid_criteria.append(crit)
                    continue
                left_s, op, right_s = parsed
                seat_dir = _seat_dir_for_dealer(dealer, seat)
                left_e = _sl_len_expr(seat_dir, left_s)
                right_e = _sl_len_expr(seat_dir, right_s)
                if op == ">=":
                    dyn = (left_e >= right_e)
                elif op == "<=":
                    dyn = (left_e <= right_e)
                elif op == ">":
                    dyn = (left_e > right_e)
                elif op == "<":
                    dyn = (left_e < right_e)
                elif op == "==":
                    dyn = (left_e == right_e)
                elif op == "!=":
                    dyn = (left_e != right_e)
                else:
                    if crit not in invalid_criteria:
                        invalid_criteria.append(crit)
                    continue
                dyn_series = deal_df.select(dyn.fill_null(False).alias("_dyn")).get_column("_dyn")
                all_seats_mask = all_seats_mask & dyn_series
                if not all_seats_mask.any():
                    return None, invalid_criteria
                continue
            col = seat_criteria_df[crit]
            all_seats_mask = all_seats_mask & col
            
            # Early exit if this dealer's mask is already empty
            if not all_seats_mask.any():
                return None, invalid_criteria
                
    return all_seats_mask, invalid_criteria


def _build_criteria_mask_for_all_seats(
    deal_df: pl.DataFrame,
    bt_row: Dict[str, Any],
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, pl.DataFrame]],
) -> Tuple[Optional[pl.Series], List[str]]:
    """Build a mask matching ALL 4 seats' criteria against the deal DataFrame.
    
    Returns:
        (mask, invalid_criteria) - mask is None if no deals match
    """
    # Pre-lookup criteria lists to avoid repeated .get()
    criteria_by_seat = {}
    for seat in range(1, 5):
        criteria_list = bt_row.get(f"Agg_Expr_Seat_{seat}") or []
        if criteria_list:
            criteria_by_seat[seat] = criteria_list
            
    if not criteria_by_seat:
        # If no criteria at all, all deals match (or rather, no criteria to filter by)
        return None, []

    global_mask: pl.Series | None = None
    all_invalid_criteria: List[str] = []
    
    for dealer in DIRECTIONS:
        dealer_mask, invalid_criteria = _build_criteria_mask_for_dealer(
            deal_df, dealer, criteria_by_seat, deal_criteria_by_seat_dfs
        )
        
        if invalid_criteria:
            for c in invalid_criteria:
                if c not in all_invalid_criteria:
                    all_invalid_criteria.append(c)
                    
        if dealer_mask is not None:
            global_mask = dealer_mask if global_mask is None else (global_mask | dealer_mask)
    
    return global_mask, all_invalid_criteria


def _compute_contract_recommendations(
    stats_df: pl.DataFrame,
    dd_cols: List[str],
) -> List[Dict[str, Any]]:
    """Compute which contracts are most likely to make based on DD analysis.
    
    Works with either:
    - Raw trick columns: DD_{direction}_{strain} (if available)
    - DD Score columns: DD_Score_{level}{strain}_{direction} (fallback)
    
    For DD Score columns, a contract "makes" if the score is >= 0.
    """
    contract_recommendations: List[Dict[str, Any]] = []
    strain_names = {'N': 'NT', 'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}
    
    # Compute average ParScore once for the whole dataset
    avg_par_score = None
    if "ParScore" in stats_df.columns:
        par_series = stats_df["ParScore"].drop_nulls()
        if par_series.len() > 0:
            par_mean = par_series.mean()
            if par_mean is not None:
                avg_par_score = float(str(par_mean)) if not isinstance(par_mean, (int, float)) else float(par_mean)
    
    # Check if we have raw trick columns or DD_Score columns
    has_raw_tricks = any(col.startswith("DD_") and not col.startswith("DD_Score") for col in dd_cols)
    
    if has_raw_tricks:
        # Original logic for raw trick columns
        for direction in ['N', 'E', 'S', 'W']:
            for strain in ['N', 'S', 'H', 'D', 'C']:
                col = f"DD_{direction}_{strain}"
                if col not in stats_df.columns:
                    continue
                
                tricks_series = stats_df[col].drop_nulls()
                if tricks_series.len() == 0:
                    continue
                
                mean_tricks_val = tricks_series.mean()
                if mean_tricks_val is None:
                    continue
                mean_tricks = float(str(mean_tricks_val)) if not isinstance(mean_tricks_val, (int, float)) else float(mean_tricks_val)
                
                for level in range(1, 8):
                    tricks_needed = level + 6
                    makes_count = int(cast(Any, (tricks_series >= tricks_needed).sum()))
                    make_pct = (float(makes_count) / float(tricks_series.len()) * 100.0) if tricks_series.len() > 0 else 0.0
                    
                    if make_pct >= _DD_ANALYSIS_MAKE_PCT_THRESHOLD:
                        contract_recommendations.append({
                            "contract": f"{level}{strain_names[strain]}",
                            "declarer": direction,
                            "make_pct": round(make_pct, 1),
                            "avg_tricks": round(mean_tricks, 1),
                            "sample_size": int(tricks_series.len()),
                        })
    else:
        # Use DD_Score columns: contract makes if score >= 0
        for direction in ['N', 'E', 'S', 'W']:
            for strain in ['N', 'S', 'H', 'D', 'C']:
                for level in range(1, 8):
                    col = f"DD_Score_{level}{strain}_{direction}"
                    if col not in stats_df.columns:
                        continue
                    
                    score_series = stats_df[col].drop_nulls()
                    if score_series.len() == 0:
                        continue
                    
                    # Contract makes if DD score >= 0
                    makes_count = int(cast(Any, (score_series >= 0).sum()))
                    make_pct = (float(makes_count) / float(score_series.len()) * 100.0) if score_series.len() > 0 else 0.0
                    
                    mean_score_val = score_series.mean()
                    avg_score = float(str(mean_score_val)) if mean_score_val is not None and not isinstance(mean_score_val, (int, float)) else (float(mean_score_val) if mean_score_val is not None else 0.0)
                    
                    # Calculate EV (Expected Value) - separate ROWS for V and NV
                    # EV columns: EV_{pair}_{declarer}_{strain}_{level}_{vul}
                    pair = "NS" if direction in ["N", "S"] else "EW"
                    
                    ev_col_v = f"EV_{pair}_{direction}_{strain}_{level}_V"
                    ev_col_nv = f"EV_{pair}_{direction}_{strain}_{level}_NV"
                    
                    if make_pct >= _DD_ANALYSIS_MAKE_PCT_THRESHOLD:
                        # Define vulnerability subsets for THIS declarer
                        if direction in ["N", "S"]:
                            nv_vuls = ["None", "E_W"]
                            v_vuls = ["N_S", "Both"]
                        else:
                            # For E/W declarers, vulnerability is determined by whether E/W is vulnerable
                            # NV: None or N_S (E/W not vulnerable)
                            # V:  E_W or Both (E/W vulnerable)
                            nv_vuls = ["None", "N_S"]
                            v_vuls = ["E_W", "Both"]
                            
                        # Create separate row for Not Vulnerable
                        if ev_col_nv in stats_df.columns:
                            nv_subset = stats_df.filter(pl.col("Vul").is_in(nv_vuls))
                            ev_series_nv = nv_subset[ev_col_nv].drop_nulls()
                            if ev_series_nv.len() > 0:
                                ev_mean_nv = ev_series_nv.mean()
                                # Compute stats for this subset
                                nv_score_series = nv_subset[col].drop_nulls()
                                nv_makes_count = int(cast(Any, (nv_score_series >= 0).sum()))
                                nv_make_pct = (float(nv_makes_count) / float(nv_score_series.len()) * 100.0) if nv_score_series.len() > 0 else 0.0
                                
                                if ev_mean_nv is not None:
                                    ev_nv = float(str(ev_mean_nv)) if not isinstance(ev_mean_nv, (int, float)) else float(ev_mean_nv)
                                    contract_recommendations.append({
                                        "contract": f"{level}{strain_names[strain]}",
                                        "declarer": direction,
                                        "vul": "NV",
                                        "make_pct": round(nv_make_pct, 1),
                                        "ev": round(ev_nv, 0),
                                        "sample_size": int(ev_series_nv.len()),
                                    })
                        
                        # Create separate row for Vulnerable
                        if ev_col_v in stats_df.columns:
                            v_subset = stats_df.filter(pl.col("Vul").is_in(v_vuls))
                            ev_series_v = v_subset[ev_col_v].drop_nulls()
                            if ev_series_v.len() > 0:
                                ev_mean_v = ev_series_v.mean()
                                # Compute stats for this subset
                                v_score_series = v_subset[col].drop_nulls()
                                v_makes_count = int(cast(Any, (v_score_series >= 0).sum()))
                                v_make_pct = (float(v_makes_count) / float(v_score_series.len()) * 100.0) if v_score_series.len() > 0 else 0.0
                                
                                if ev_mean_v is not None:
                                    ev_v = float(str(ev_mean_v)) if not isinstance(ev_mean_v, (int, float)) else float(ev_mean_v)
                                    contract_recommendations.append({
                                        "contract": f"{level}{strain_names[strain]}",
                                        "declarer": direction,
                                        "vul": "V",
                                        "make_pct": round(v_make_pct, 1),
                                        "ev": round(ev_v, 0),
                                        "sample_size": int(ev_series_v.len()),
                                    })
    
    # Sort by EV descending
    contract_recommendations.sort(key=lambda x: -x.get("ev", float("-inf")))
    return contract_recommendations[:_DD_ANALYSIS_TOP_RECOMMENDATIONS]


def _compute_par_contract_stats(stats_df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Compute par contract statistics split by vulnerability.
    
    Returns a list of dicts with category, count, pct, avg scores, and EV by vulnerability.
    """
    par_contract_stats: List[Dict[str, Any]] = []
    
    if "ParContracts" not in stats_df.columns or "ParScore" not in stats_df.columns or "Vul" not in stats_df.columns:
        return par_contract_stats
    
    # Helper to compute avg score for a filtered series
    def _safe_mean(series: pl.Series) -> Optional[float]:
        if series.len() == 0:
            return None
        m = series.mean()
        if m is None:
            return None
        return float(str(m)) if not isinstance(m, (int, float)) else float(m)
    
    # Define vulnerability groups
    # NV for NS: None, E_W (NS not vulnerable)
    # V for NS: N_S, Both (NS vulnerable)
    ns_nv_mask = stats_df["Vul"].is_in(["None", "E_W"])
    ns_v_mask = stats_df["Vul"].is_in(["N_S", "Both"])
    
    par_scores = stats_df["ParScore"].drop_nulls()
    total = par_scores.len()
    
    if total == 0:
        return par_contract_stats
    
    # Score categories to analyze
    categories = [
        ("NS Makes (Par > 0)", lambda s: s > 0),
        ("EW Makes (Par < 0)", lambda s: s < 0),
        ("Pass Out (Par = 0)", lambda s: s == 0),
        ("Slam (1000+)", lambda s: s >= 1000),
        ("Game (300-999)", lambda s: (s >= 300) & (s <= 999)),
        ("Partscore (50-299)", lambda s: (s >= 50) & (s <= 299)),
        ("Small (-49 to 49)", lambda s: (s >= -49) & (s <= 49)),
        ("Set 1-2 (-50 to -299)", lambda s: (s >= -299) & (s <= -50)),
        ("Set 3+ (-300 to -999)", lambda s: (s >= -999) & (s <= -300)),
        ("Doubled Set (-1000-)", lambda s: s <= -1000),
    ]
    
    for cat_name, filter_fn in categories:
        # Get scores for this category
        cat_mask = filter_fn(stats_df["ParScore"])
        cat_count = int(cat_mask.sum())
        
        if cat_count == 0:
            continue
        
        # Get NV and V subsets
        nv_df = stats_df.filter(cat_mask & ns_nv_mask)
        v_df = stats_df.filter(cat_mask & ns_v_mask)
        
        nv_scores = nv_df["ParScore"].drop_nulls() if nv_df.height > 0 else pl.Series([])
        v_scores = v_df["ParScore"].drop_nulls() if v_df.height > 0 else pl.Series([])
        
        rec: Dict[str, Any] = {
            "category": cat_name,
            "count": cat_count,
            "pct": round(cat_count / total * 100, 1),
        }
        
        # Avg Score NV and V
        avg_nv = _safe_mean(nv_scores)
        avg_v = _safe_mean(v_scores)
        rec["avg_nv"] = round(avg_nv, 0) if avg_nv is not None else None
        rec["avg_v"] = round(avg_v, 0) if avg_v is not None else None
        rec["count_nv"] = nv_scores.len()
        rec["count_v"] = v_scores.len()
        
        # Compute EV for actual par contracts using the native ParContracts structure
        cat_df = stats_df.filter(cat_mask)
        ev_nv_vals: List[float] = []
        ev_v_vals: List[float] = []
        
        for row in cat_df.iter_rows(named=True):
            # Get EV for this deal's actual par contract
            ev_list = _ev_list_for_par_contracts(row)
            if ev_list:
                # Use the first (best) par contract EV
                ev_val = ev_list[0]
                if ev_val is not None:
                    # Determine if NS is vulnerable based on the deal's Vul
                    vul = row.get("Vul", "None")
                    if vul in ["N_S", "Both"]:
                        ev_v_vals.append(ev_val)
                    else:
                        ev_nv_vals.append(ev_val)
        
        if ev_nv_vals:
            rec["ev_nv"] = round(sum(ev_nv_vals) / len(ev_nv_vals), 0)
        if ev_v_vals:
            rec["ev_v"] = round(sum(ev_v_vals) / len(ev_v_vals), 0)
        
        par_contract_stats.append(rec)
    
    return par_contract_stats


def handle_auction_dd_analysis(
    state: Dict[str, Any],
    auction: str,
    max_deals: int,
    seed: Optional[int],
    vul_filter: Optional[str] = None,
    include_hands: bool = True,
    include_scores: bool = True,
) -> Dict[str, Any]:
    """Handle /auction-dd-analysis endpoint.
    
    Finds deals matching an auction's criteria and returns their DD columns.
    
    Returns:
        - bt_row: The matched BT row (Auction, Agg_Expr_Seat_1-4)
        - criteria_breakdown: Criteria for each seat
        - dd_data: List of dicts with DD columns, hands, scores
        - contract_recommendations: Best contracts based on DD analysis
        - par_comparison: How often auction reaches par
        - total_matches: Total number of deals matching all criteria
        - sql_query: Example SQL query for the result
    """
    t0 = time.perf_counter()
    
    deal_df = state["deal_df"]
    bt_seat1_df = state.get("bt_seat1_df")
    deal_criteria_by_seat_dfs = state["deal_criteria_by_seat_dfs"]
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Normalize auction input (supports '-', whitespace, or ',' separators), then strip leading passes to match seat-1 view
    auction_normalized = re.sub(r"(?i)^(p-)+", "", normalize_auction_input(auction))
    if not auction_normalized:
        raise ValueError(f"Invalid auction: '{auction}'")
    
    # Resolve BT row by traversal (no Auction scans on 461M-row BT).
    bt_idx = _resolve_bt_index_by_traversal(state, auction_normalized)
    if bt_idx is None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "auction_input": auction,
            "auction_normalized": auction_normalized,
            "bt_row": None,
            "criteria_breakdown": {},
            "dd_data": [],
            "total_matches": 0,
            "sql_query": f"-- No auction found in BT for: {auction_normalized}",
            "elapsed_ms": round(elapsed_ms, 1),
        }

    # Fetch the row by bt_index (DuckDB) and apply overlay/dedupe.
    file_path = _bt_file_path_for_sql(state)
    conn = duckdb.connect(":memory:")
    try:
        bt_df_row = conn.execute(
            f"""
            SELECT bt_index, Auction, candidate_bid, Expr,
                   Agg_Expr_Seat_1, Agg_Expr_Seat_2, Agg_Expr_Seat_3, Agg_Expr_Seat_4
            FROM read_parquet('{file_path}')
            WHERE bt_index = {int(bt_idx)}
            LIMIT 1
            """
        ).pl()
    finally:
        conn.close()

    if bt_df_row.is_empty():
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "auction_input": auction,
            "auction_normalized": auction_normalized,
            "bt_row": None,
            "criteria_breakdown": {},
            "dd_data": [],
            "total_matches": 0,
            "sql_query": f"-- No auction found in BT for: {auction_normalized}",
            "elapsed_ms": round(elapsed_ms, 1),
        }

    bt_row = _apply_all_rules_to_bt_row(dict(bt_df_row.row(0, named=True)), state)
    bt_row_display = {
        "Auction": bt_row.get("Auction"),
        "bt_index": bt_row.get("bt_index"),
    }
    
    # Build criteria breakdown for each seat
    criteria_breakdown: Dict[str, List[str]] = {}
    for s in range(1, 5):
        col = f"Agg_Expr_Seat_{s}"
        criteria_list = bt_row.get(col) or []
        bt_row_display[col] = criteria_list
        seat_label = _SEAT_LABELS[s]
        if criteria_list:
            criteria_breakdown[seat_label] = list(criteria_list)
        else:
            criteria_breakdown[seat_label] = ["(no criteria)"]
    
    # Build global mask matching ALL 4 seats' criteria
    global_mask, invalid_criteria = _build_criteria_mask_for_all_seats(
        deal_df, bt_row, deal_criteria_by_seat_dfs
    )
    
    if invalid_criteria:
        print(f"[auction-dd-analysis] Warning: {len(invalid_criteria)} criteria not found in bitmaps: {invalid_criteria[:5]}")
    
    if global_mask is None or not global_mask.any():
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "auction_input": auction,
            "auction_normalized": auction_normalized,
            "bt_row": bt_row_display,
            "criteria_breakdown": criteria_breakdown,
            "dd_data": [],
            "total_matches": 0,
            "sql_query": f"-- No deals matched the criteria for auction: {auction_normalized}",
            "elapsed_ms": round(elapsed_ms, 1),
        }
    
    matched_df = deal_df.filter(global_mask)
    
    # Debug: Log unique Vul values to diagnose mapping issues
    vul_breakdown: Dict[str, int] = {}
    if "Vul" in matched_df.columns:
        try:
            vul_counts = matched_df.group_by("Vul").len().to_dicts()
            for row in vul_counts:
                vul_val = str(row.get("Vul", "unknown"))
                vul_breakdown[vul_val] = row.get("len", 0)
            print(f"[auction-dd-analysis] Vul distribution before filter: {vul_breakdown}")
        except Exception as e:
            print(f"[auction-dd-analysis] Failed to get Vul breakdown: {e}")
    
    # Apply vulnerability filter if specified
    # Map UI values (NS, EW) to data values (N_S, E_W)
    if vul_filter and vul_filter != "all" and "Vul" in matched_df.columns:
        data_vul_value = _VUL_UI_TO_DATA.get(vul_filter, vul_filter)
        matched_df = matched_df.filter(pl.col("Vul") == data_vul_value)
    
    total_matches = matched_df.height
    
    # Sample if needed (but keep full df for statistics)
    full_matched_df = matched_df
    if total_matches > max_deals and max_deals > 0:
        effective_seed = _effective_seed(seed)
        matched_df = matched_df.sample(n=max_deals, seed=effective_seed)
    
    # Collect DD Score columns: DD_Score_{level}{strain}_{direction}
    # Note: Raw trick columns DD_[NESW]_[CDHSN] don't exist in deal_df.
    # We use DD_Score columns which contain the actual scores for each contract.
    dd_cols = []
    
    # First try raw trick columns (DD_N_C format) - may not exist
    for direction in ['N', 'E', 'S', 'W']:
        for strain in ['C', 'D', 'H', 'S', 'N']:
            col_name = f"DD_{direction}_{strain}"
            if col_name in matched_df.columns:
                dd_cols.append(col_name)
    
    # If no raw trick columns, use DD_Score columns for common contracts
    if not dd_cols:
        # Priority contracts for analysis: games and slams
        priority_contracts = [
            (1, 'N'), (2, 'N'), (3, 'N'),  # NT partials and game
            (4, 'H'), (4, 'S'),             # Major games
            (5, 'C'), (5, 'D'),             # Minor games
            (6, 'N'), (6, 'H'), (6, 'S'), (6, 'C'), (6, 'D'),  # Small slams
            (7, 'N'), (7, 'H'), (7, 'S'),   # Grand slams
        ]
        for level, strain in priority_contracts:
            for direction in ['N', 'E', 'S', 'W']:  # All declarers
                col_name = f"DD_Score_{level}{strain}_{direction}"
                if col_name in matched_df.columns:
                    dd_cols.append(col_name)
    
    if not dd_cols:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "auction_input": auction,
            "auction_normalized": auction_normalized,
            "bt_row": bt_row_display,
            "criteria_breakdown": criteria_breakdown,
            "dd_data": [],
            "total_matches": total_matches,
            "error": "No DD columns found in deal_df. Neither DD_[NESW]_[CDHSN] nor DD_Score_* columns are available.",
            "available_columns": [c for c in matched_df.columns if c.startswith("DD_")][:20],
            "sql_query": f"-- DD columns not available",
            "elapsed_ms": round(elapsed_ms, 1),
        }
    
    # Build select columns
    context_cols = ["index", "Dealer"]
    if vul_filter or "Vul" in matched_df.columns:
        context_cols.append("Vul")
    # Add Vul_NS and Vul_EW for detailed vulnerability info
    for vul_col in ["Vul_NS", "Vul_EW"]:
        if vul_col in matched_df.columns:
            context_cols.append(vul_col)
    
    select_cols = [c for c in context_cols if c in matched_df.columns]
    
    # Add hand columns if requested
    hand_cols = []
    if include_hands:
        for d in ['N', 'E', 'S', 'W']:
            col = f"Hand_{d}"
            if col in matched_df.columns:
                hand_cols.append(col)
        select_cols.extend(hand_cols)
    
    # Add DD score columns if requested
    # Note: dd_cols contains DD_Score columns (raw DD trick columns don't exist in our data)
    if include_scores:
        select_cols.extend(dd_cols)
    
    # Add ParScore if available
    if "ParScore" in matched_df.columns:
        select_cols.append("ParScore")
    if "ParContracts" in matched_df.columns:
        select_cols.append("ParContracts")
    
    # Use library-style vectorized operations for score deltas and outcome flags
    # Patterned after mlBridgeAugmentLib.create_score_diff_columns and add_trick_columns
    if "Score" in matched_df.columns and "ParScore" in matched_df.columns:
        matched_df = matched_df.with_columns([
            (pl.col("Score").cast(pl.Int64, strict=False) - pl.col("ParScore").cast(pl.Int64, strict=False)).alias("Score_Delta")
        ])
        if "Score_Delta" in matched_df.columns:
            select_cols.append("Score_Delta")
            
    if "Result" in matched_df.columns:
        matched_df = matched_df.with_columns([
            (pl.col("Result") > 0).alias("OverTricks"),
            (pl.col("Result") == 0).alias("JustMade"),
            (pl.col("Result") < 0).alias("UnderTricks"),
        ])
        select_cols.extend(["OverTricks", "JustMade", "UnderTricks"])

    result_df = matched_df.select([c for c in select_cols if c in matched_df.columns])
    dd_data = result_df.to_dicts()
    
    # Format ParContracts from nested list/struct to readable string
    if "ParContracts" in select_cols:
        for row in dd_data:
            if "ParContracts" in row:
                row["ParContracts"] = _format_par_contracts(row["ParContracts"])
    
    # -------------------------------------------------------------------------
    # Contract Recommendations: Analyze which contracts are most likely to make
    # -------------------------------------------------------------------------
    # Use the full matched set for statistics (not just sampled)
    stats_df = full_matched_df if total_matches <= _DD_ANALYSIS_STATS_THRESHOLD else matched_df
    
    contract_recommendations = _compute_contract_recommendations(stats_df, dd_cols)
    
    # -------------------------------------------------------------------------
    # Par Score Comparison
    # -------------------------------------------------------------------------
    par_comparison: Dict[str, Any] = {}
    if "ParScore" in stats_df.columns:
        par_series = stats_df["ParScore"].drop_nulls()
        if par_series.len() > 0:
            par_comparison["avg_par"] = round(par_series.mean(), 1) if par_series.mean() else 0
            par_comparison["min_par"] = int(par_series.min()) if par_series.min() else 0
            par_comparison["max_par"] = int(par_series.max()) if par_series.max() else 0
            par_comparison["sample_size"] = int(par_series.len())
    
    # -------------------------------------------------------------------------
    # Par Contract Analysis - Stats on par contracts, sacrifices, sets
    # Split by vulnerability: separate columns for NV and V
    # -------------------------------------------------------------------------
    par_contract_stats = _compute_par_contract_stats(stats_df)
    
    # -------------------------------------------------------------------------
    # DD Statistics Summary
    # -------------------------------------------------------------------------
    dd_stats: List[Dict[str, Any]] = []
    for col in dd_cols:
        col_data = stats_df[col].drop_nulls()
        if col_data.len() > 0:
            # Parse column name based on format
            # DD_N_C format: direction=N, strain=C
            # DD_Score_3N_N format: level=3, strain=N, direction=N
            parts = col.split("_")
            if col.startswith("DD_Score_"):
                # DD_Score_{level}{strain}_{direction} e.g. DD_Score_3N_N
                level_strain = parts[2]  # "3N"
                level = level_strain[0]
                strain = level_strain[1] if len(level_strain) > 1 else "?"
                direction = parts[3] if len(parts) > 3 else "?"
                stat_entry = {
                    "column": col,
                    "contract": f"{level}{strain}",
                    "direction": direction,
                    "mean": round(col_data.mean(), 0) if col_data.mean() is not None else None,
                    "min": int(col_data.min()) if col_data.min() is not None else None,
                    "max": int(col_data.max()) if col_data.max() is not None else None,
                    "makes%": round((col_data >= 0).mean() * 100, 1) if col_data.len() > 0 else None,
                }
            else:
                # DD_{direction}_{strain} format
                direction = parts[1] if len(parts) > 1 else "?"
                strain = parts[2] if len(parts) > 2 else "?"
                stat_entry = {
                    "column": col,
                    "direction": direction,
                    "strain": strain,
                    "mean": round(col_data.mean(), 2) if col_data.mean() is not None else None,
                    "min": int(col_data.min()) if col_data.min() is not None else None,
                    "max": int(col_data.max()) if col_data.max() is not None else None,
                    "std": round(col_data.std(), 2) if col_data.std() is not None else None,
                }
            dd_stats.append(stat_entry)
    
    # Build SQL query for display
    selected_cols_sql = ", ".join([f'"{c}"' for c in select_cols[:_DD_ANALYSIS_SQL_COLS_LIMIT]])
    vul_clause = f"\n  AND Vul = '{vul_filter}'" if vul_filter and vul_filter != "all" else ""
    sql_query = f"""SELECT {selected_cols_sql}
FROM deals
WHERE -- criteria matching for auction '{auction_normalized}'{vul_clause}
LIMIT {max_deals}"""
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[auction-dd-analysis] Found {total_matches} matches, returning {len(dd_data)} rows in {format_elapsed(elapsed_ms)}")
    
    return {
        "auction_input": auction,
        "auction_normalized": auction_normalized,
        "bt_row": bt_row_display,
        "criteria_breakdown": criteria_breakdown,
        "dd_data": dd_data,
        "dd_columns": dd_cols,
        "dd_stats": dd_stats,
        "contract_recommendations": contract_recommendations,
        "par_comparison": par_comparison,
        "par_contract_stats": par_contract_stats,  # Par contract analysis (sacrifices, sets, etc.)
        "total_matches": total_matches,
        "returned_count": len(dd_data),
        "vul_filter": vul_filter,
        "vul_breakdown": vul_breakdown,  # Show distribution of Vul values in matched deals
        "sql_query": sql_query,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# List Next Bids – Fast lookup using next_bid_indices (no EV computation)
# ---------------------------------------------------------------------------

def handle_list_next_bids(
    state: Dict[str, Any],
    auction: str,
) -> Dict[str, Any]:
    """Fast lookup of available next bids using Gemini-3.2 CSR index.
    
    Returns:
        - auction_input: The input auction
        - auction_normalized: Normalized form
        - next_bids: List of dicts with bid, bt_index, agg_expr, is_completed_auction
        - elapsed_ms: Processing time
    """
    t0 = time.perf_counter()
    
    # Use the optimized walk-fallback which now uses CSR + Polars
    # This covers both empty auctions and continuations.
    auction_input = normalize_auction_input(auction)
    auction_normalized = re.sub(r"(?i)^(p-)+", "", auction_input) if auction_input else ""
    
    resp = _handle_list_next_bids_walk_fallback(
        state,
        auction_input,
        auction_normalized,
        t0,
        include_deal_counts=True,
        include_ev_stats=True,
    )
    
    # Add extra metadata for the response if missing
    if "next_bids" not in resp and "bid_rankings" in resp:
        resp["next_bids"] = resp.pop("bid_rankings")
        
    return resp


# ---------------------------------------------------------------------------
# Rank Next Bids by EV – Rank next bids after an auction by Expected Value
# ---------------------------------------------------------------------------

def handle_rank_bids_by_ev(
    state: Dict[str, Any],
    auction: str,
    max_deals: int,
    seed: Optional[int],
    vul_filter: Optional[str] = None,
    include_hands: bool = True,
    include_scores: bool = True,
) -> Dict[str, Any]:
    """Handle /rank-bids-by-ev endpoint.
    
    Given an auction prefix (or empty for opening bids), finds all possible next bids
    using next_bid_indices and computes average EV for each bid across matching deals.
    
    Also computes DD analysis (contract recommendations, par stats) on the aggregated
    deals from all next bids.
    
    Returns:
        - auction_input: The input auction
        - auction_normalized: Normalized (leading p- removed)
        - parent_bt_row: The BT row for the input auction (None for opening bids)
        - bid_rankings: List of dicts with bid, count, avg_ev_nv, avg_ev_v, avg_par, etc.
        - total_next_bids: Total number of next bid options
        - contract_recommendations: Contracts ranked by EV (aggregated across all next bids)
        - par_contract_stats: Par contract breakdown
        - dd_data: Deal data for matched deals
        - elapsed_ms: Processing time
    """
    t0 = time.perf_counter()
    
    deal_df = state["deal_df"]
    bt_seat1_df = state.get("bt_seat1_df")
    deal_criteria_by_seat_dfs = state["deal_criteria_by_seat_dfs"]
    overlay = state.get("custom_criteria_overlay") or []
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Check if next_bid_indices is available
    if "next_bid_indices" not in bt_seat1_df.columns:
        raise ValueError("next_bid_indices column not loaded in bt_seat1_df. Restart API server to load it.")
    
    auction_input = normalize_auction_input(auction)
    
    # Count expected leading passes from auction prefix
    # E.g., "" -> 0 passes (dealer opens), "p-1N" -> 1 pass, "p-p-1N" -> 2 passes
    expected_passes = _count_leading_passes(auction_input)
    
    # Normalize: strip trailing dash if present (e.g., "p-" -> "p", "p-p-" -> "p-p")
    auction_for_lookup = auction_input.rstrip("-") if auction_input else ""
    # For display, strip leading passes from the non-pass part
    auction_normalized = re.sub(r"(?i)^(p-)+", "", auction_input) if auction_input else ""
    
    parent_bt_row: Optional[Dict[str, Any]] = None
    next_bid_rows: pl.DataFrame
    
    # Determine if this is purely passes (e.g., "", "p", "p-p", "p-p-p")
    is_passes_only = not auction_normalized or auction_normalized.lower() in ("p", "")

    # Seat (1-4) of the *next bidder* relative to the dealer (Seat 1 = dealer).
    # This is constant for all candidate next bids for a given auction prefix.
    call_tokens = [t for t in auction_for_lookup.split("-") if t] if auction_for_lookup else []
    next_seat = (len(call_tokens) % 4) + 1
    
    if not auction_input:
        # Truly empty auction: use precomputed opening-bids table if available.
        # This avoids touching Agg_Expr for the full 461M-row BT.
        bt_openings_df = state.get("bt_openings_df")
        if isinstance(bt_openings_df, pl.DataFrame) and "seat" in bt_openings_df.columns:
            next_bid_rows = bt_openings_df.filter(pl.col("seat") == 1)
            # Normalize shape to expected columns for downstream logic
            if "candidate_bid" not in next_bid_rows.columns and "Auction" in next_bid_rows.columns:
                next_bid_rows = next_bid_rows.with_columns(pl.col("Auction").alias("candidate_bid"))
        else:
            # Intentionally no fallback to scanning the full 461M-row bt_seat1_df.
            raise ValueError(
                "bt_openings_df is missing/empty; refusing to scan bt_seat1_df for opening bids. "
                "Restart the API server and ensure initialization completes successfully."
            )
    elif is_passes_only and expected_passes > 0:
        # Pass-only prefix. Seat-1 view does not support leading passes, but users can request them.
        # We treat leading passes as seat alignment metadata and traverse the non-pass part.
        # For pure pass sequences, there is no opening bid to anchor traversal; return empty.
        pass_auction = "-".join(["p"] * expected_passes)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "auction_input": auction_input,
            "auction_normalized": pass_auction,
            "parent_bt_row": None,
            "bid_rankings": [],
            "total_next_bids": 0,
            "message": f"Pass-only prefix '{pass_auction}' cannot be resolved without Auction scans (disabled).",
            "elapsed_ms": round(elapsed_ms, 1),
        }
    else:
        # Resolve parent bt_index via traversal (no Auction scans)
        parent_bt_index = _resolve_bt_index_by_traversal(state, auction_for_lookup)
        if parent_bt_index is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "auction_input": auction_input,
                "auction_normalized": auction_normalized,
                "parent_bt_row": None,
                "bid_rankings": [],
                "total_next_bids": 0,
                "error": f"Auction '{auction_for_lookup}' not found in BT (traversal failed)",
                "elapsed_ms": round(elapsed_ms, 1),
            }

        parent_bt_row = {"bt_index": parent_bt_index, "Auction": auction_normalized, "candidate_bid": None}
        next_indices = _get_next_bid_indices_for_parent(state, parent_bt_index)
        if not next_indices:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "auction_input": auction_input,
                "auction_normalized": auction_normalized,
                "parent_bt_row": parent_bt_row,
                "bid_rankings": [],
                "total_next_bids": 0,
                "message": "No next bids found (auction may be completed)",
                "elapsed_ms": round(elapsed_ms, 1),
            }

        # Fetch next-bid rows by bt_index IN (...) via DuckDB (avoid scanning 461M-row bt_seat1_df)
        file_path = _bt_file_path_for_sql(state)
        in_list = ", ".join(str(int(x)) for x in next_indices if x is not None)
        conn = duckdb.connect(":memory:")
        try:
            next_bid_rows = conn.execute(
                f"""
                SELECT bt_index, Auction, candidate_bid, is_completed_auction, Expr
                FROM read_parquet('{file_path}')
                WHERE bt_index IN ({in_list})
                """
            ).pl()
        finally:
            conn.close()
    
    total_next_bids = next_bid_rows.height
    
    if total_next_bids == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "auction_input": auction_input,
            "auction_normalized": auction_normalized,
            "parent_bt_row": parent_bt_row,
            "bid_rankings": [],
            "total_next_bids": 0,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    
    effective_seed = _effective_seed(seed)

    # Ensure Agg_Expr columns exist for next-bid rows (needed below for mask building).
    # In lightweight mode bt_seat1_df excludes Agg_Expr, so we load them on-demand for these bt_index values.
    # Uses DuckDB for efficient lookup (much faster than Polars scan for small IN lists).
    if total_next_bids > 0 and "Agg_Expr_Seat_1" not in next_bid_rows.columns:
        bt_parquet_file = state.get("bt_seat1_file")
        if bt_parquet_file is None:
            raise ValueError("bt_seat1_file missing from state; cannot load Agg_Expr on-demand")
        if "bt_index" in next_bid_rows.columns:
            needed = (
                next_bid_rows
                .select(pl.col("bt_index").drop_nulls().cast(pl.Int64).unique())
                .to_series()
                .to_list()
            )
            needed = [int(x) for x in needed if x is not None]
            if needed:
                # Use the shared DuckDB-based loader and convert result to DataFrame for join
                agg_data = _load_agg_expr_for_bt_indices(needed, bt_parquet_file)
                if agg_data:
                    # Convert dict to DataFrame for join
                    agg_rows: list[dict[str, Any]] = []
                    for bt_idx, cols_dict in agg_data.items():
                        row: dict[str, Any] = {"bt_index": bt_idx}
                        for k, v in cols_dict.items():
                            row[k] = v
                        agg_rows.append(row)
                    agg_df = pl.DataFrame(agg_rows)
                    next_bid_rows = next_bid_rows.join(agg_df, on="bt_index", how="left")
    
    # =========================================================================
    # PRE-LOAD: EV stats from GPU pipeline (optional optimization) - NV/V split
    # =========================================================================
    # If bt_ev_stats_df is available, pre-load Avg_EV and Avg_Par for all next bids.
    # This gives instant ranking without computing from deals.
    bt_ev_stats_df = state.get("bt_ev_stats_df")
    ev_stats_lookup: Dict[int, Dict[str, Any]] = {}
    if bt_ev_stats_df is not None and next_bid_rows.height > 0:
        bt_indices = next_bid_rows["bt_index"].unique().to_list()
        bt_indices = [int(x) for x in bt_indices if x is not None]
        if bt_indices:
            ev_subset = bt_ev_stats_df.filter(pl.col("bt_index").is_in(bt_indices))
            for ev_row in ev_subset.iter_rows(named=True):
                ev_data: Dict[str, Any] = {}
                for s in range(1, 5):
                    # Try NV/V split columns first (new format)
                    for vul in ["NV", "V"]:
                        ev_key = f"Avg_EV_S{s}_{vul}"
                        par_key = f"Avg_Par_S{s}_{vul}"
                        if ev_key in ev_row:
                            ev_data[ev_key] = ev_row.get(ev_key)
                            ev_data[par_key] = ev_row.get(par_key)
                    # Fall back to old aggregate columns
                    if f"Avg_EV_S{s}_NV" not in ev_data:
                        ev_data[f"Avg_EV_S{s}"] = ev_row.get(f"Avg_EV_S{s}")
                        ev_data[f"Avg_Par_S{s}"] = ev_row.get(f"Avg_Par_S{s}")
                ev_stats_lookup[int(ev_row["bt_index"])] = ev_data

    # =========================================================================
    # PRE-FILTER: Opening seat alignment
    # =========================================================================
    # All matched deals MUST match the expected opening seat (based on leading passes).
    # Pre-compute this mask ONCE to avoid map_elements in the loop.
    opening_seat_mask = None
    if "bid" in deal_df.columns:
        # Vectorized check for leading passes (normalize auction first)
        # BBO format: "p-p-1H", "1N", etc.
        prefix = "p-" * expected_passes
        if expected_passes == 0:
            # Must NOT start with 'p-'
            opening_seat_mask = ~deal_df["bid"].str.starts_with("p-")
        else:
            # Must start with exactly 'p-' repeated expected_passes times
            prefix = "p-" * expected_passes
            not_extra_pass = ~deal_df["bid"].str.starts_with("p-" * (expected_passes + 1))
            opening_seat_mask = deal_df["bid"].str.starts_with(prefix) & not_extra_pass

    # =========================================================================
    # OPTIMIZATION: Pre-calculate mask for constant seats (usually 1, 2, 3)
    # =========================================================================
    # Many next bids share the same parent auction, meaning seats 1-3 have the same criteria.
    # We can pre-compute a "parent mask" and only add the next seat's criteria in the loop.
    parent_mask_by_dealer: Dict[str, pl.Series] = {}
    
    # Identify which seats have constant criteria across all candidate bids
    # For next_bid_indices, usually seats 1 to (parent_len + 1) are constant, 
    # and only the seat of the next bid changes.
    # For simplicity, we'll check which Agg_Expr_Seat_N columns are identical across all candidates.
    constant_seats = []
    for seat in range(1, 5):
        unique_exprs = next_bid_rows[f"Agg_Expr_Seat_{seat}"].n_unique()
        if unique_exprs == 1:
            constant_seats.append(seat)
    
    if constant_seats and not next_bid_rows.is_empty():
        first_row = _apply_all_rules_to_bt_row(dict(next_bid_rows.row(0, named=True)), state)
        for dealer in DIRECTIONS:
            dealer_mask = deal_df["Dealer"] == dealer
            
            # Combine with opening seat mask if it exists
            if opening_seat_mask is not None:
                dealer_mask = dealer_mask & opening_seat_mask
                
            parent_seat_mask = None
            for seat in constant_seats:
                criteria_list = first_row.get(f"Agg_Expr_Seat_{seat}") or []
                if not criteria_list: continue
                
                seat_criteria_df = deal_criteria_by_seat_dfs.get(seat, {}).get(dealer)
                if seat_criteria_df is None or seat_criteria_df.is_empty():
                    parent_seat_mask = pl.Series([False] * deal_df.height)
                    break
                
                available_cols = set(seat_criteria_df.columns)
                for crit in criteria_list:
                    if crit in available_cols:
                        col = seat_criteria_df[crit]
                        parent_seat_mask = col if parent_seat_mask is None else (parent_seat_mask & col)
            
            if parent_seat_mask is not None:
                # Include dealer_mask in parent_mask to avoid repeating it in the loop
                parent_mask_by_dealer[dealer] = dealer_mask & parent_seat_mask
            else:
                parent_mask_by_dealer[dealer] = dealer_mask
    
    non_constant_seats = [s for s in range(1, 5) if s not in constant_seats]

    # =========================================================================
    # For each bid, match deals using criteria bitmaps and track by bid
    # =========================================================================
    bid_rankings: List[Dict[str, Any]] = []
    # Keep the large per-bid EV/Makes blob separate so we don't duplicate it when
    # emitting 2 rows per bid (NV/V) in bid_rankings.
    ev_all_combos_by_bid: Dict[str, Dict[str, Optional[float]]] = {}
    matched_by_bid: Dict[str, Set[int]] = {}  # Track which deals matched which bid
    matched_dfs_by_bid: Dict[str, pl.DataFrame] = {}  # Store matched DataFrames for per-bid sampling
    
    # Pre-lookup seat criteria DFs to avoid repeated .get() calls
    seat_dfs_prelookup = {
        s: {d: deal_criteria_by_seat_dfs.get(s, {}).get(d) for d in DIRECTIONS}
        for s in non_constant_seats
    }

    for i, row in enumerate(next_bid_rows.iter_rows(named=True)):
        bid_name = row.get("candidate_bid", "?")
        bt_index = row.get("bt_index")
        bid_auction = row.get("Auction", "")
        
        # Get pre-computed EV/Par from GPU pipeline (if available) - used in all exit paths
        # Now with NV/V splits matching the bid_rankings rows
        precomputed = ev_stats_lookup.get(bt_index, {}) if bt_index is not None else {}
        # Try NV/V split columns first
        avg_ev_precomputed_nv = precomputed.get(f"Avg_EV_S{next_seat}_NV")
        avg_ev_precomputed_v = precomputed.get(f"Avg_EV_S{next_seat}_V")
        avg_par_precomputed_nv = precomputed.get(f"Avg_Par_S{next_seat}_NV")
        avg_par_precomputed_v = precomputed.get(f"Avg_Par_S{next_seat}_V")
        # Fall back to aggregate if NV/V not available
        if avg_ev_precomputed_nv is None:
            aggregate_ev = precomputed.get(f"Avg_EV_S{next_seat}")
            avg_ev_precomputed_nv = aggregate_ev
            avg_ev_precomputed_v = aggregate_ev
        if avg_par_precomputed_nv is None:
            aggregate_par = precomputed.get(f"Avg_Par_S{next_seat}")
            avg_par_precomputed_nv = aggregate_par
            avg_par_precomputed_v = aggregate_par
        
        # Build criteria mask using optimized parent mask where possible
        global_mask = None
        
        # Optimization: Pre-fetch non-constant criteria lists
        # Apply overlay per candidate row before extracting criteria lists
        # Canonical: overlay + dedupe (Agg_Expr already loaded above, so this is cheap)
        row_overlayed = _apply_all_rules_to_bt_row(dict(row), state)
        seat_criteria_lists = {s: row_overlayed.get(f"Agg_Expr_Seat_{s}") or [] for s in non_constant_seats}
        
        for dealer in DIRECTIONS:
            # Use pre-computed parent mask (includes dealer_mask and opening_seat_mask)
            all_seats_mask = parent_mask_by_dealer.get(dealer)
            
            # If parent_mask is already all False for this dealer, skip
            if all_seats_mask is not None and not all_seats_mask.any():
                continue
                
            # Add non-constant seats
            current_dealer_mask = all_seats_mask
            for seat in non_constant_seats:
                criteria_list = seat_criteria_lists[seat]
                if not criteria_list: continue
                
                seat_criteria_df = seat_dfs_prelookup[seat][dealer]
                if seat_criteria_df is None or seat_criteria_df.is_empty():
                    current_dealer_mask = None
                    break
                
                available_cols = set(seat_criteria_df.columns)
                for crit in criteria_list:
                    if crit in available_cols:
                        col = seat_criteria_df[crit]
                        current_dealer_mask = col if current_dealer_mask is None else (current_dealer_mask & col)
            
            if current_dealer_mask is not None:
                global_mask = current_dealer_mask if global_mask is None else (global_mask | current_dealer_mask)
        
        if global_mask is None or not global_mask.any():
            bid_rankings.append({
                "bid": bid_name,
                "bt_index": bt_index,
                "auction": bid_auction,
                "match_count": 0,
                "nv_count": 0,
                "v_count": 0,
                "avg_par_nv": None,
                "avg_par_v": None,
                "avg_ev_precomputed_nv": round(avg_ev_precomputed_nv, 1) if avg_ev_precomputed_nv is not None else None,
                "avg_ev_precomputed_v": round(avg_ev_precomputed_v, 1) if avg_ev_precomputed_v is not None else None,
                "avg_par_precomputed_nv": round(avg_par_precomputed_nv, 1) if avg_par_precomputed_nv is not None else None,
                "avg_par_precomputed_v": round(avg_par_precomputed_v, 1) if avg_par_precomputed_v is not None else None,
            })
            continue
            
        # Combine with opening seat mask ONLY if parent_mask optimization was NOT used
        if not constant_seats and opening_seat_mask is not None:
            global_mask = global_mask & opening_seat_mask
            
        if not global_mask.any():
            bid_rankings.append({
                "bid": bid_name,
                "bt_index": bt_index,
                "auction": bid_auction,
                "match_count": 0,
                "nv_count": 0,
                "v_count": 0,
                "avg_par_nv": None,
                "avg_par_v": None,
                "avg_ev_precomputed_nv": round(avg_ev_precomputed_nv, 1) if avg_ev_precomputed_nv is not None else None,
                "avg_ev_precomputed_v": round(avg_ev_precomputed_v, 1) if avg_ev_precomputed_v is not None else None,
                "avg_par_precomputed_nv": round(avg_par_precomputed_nv, 1) if avg_par_precomputed_nv is not None else None,
                "avg_par_precomputed_v": round(avg_par_precomputed_v, 1) if avg_par_precomputed_v is not None else None,
            })
            continue
            
        # Combine with opening seat mask
        if opening_seat_mask is not None:
            global_mask = global_mask & opening_seat_mask
            
        if not global_mask.any():
            bid_rankings.append({
                "bid": bid_name,
                "bt_index": bt_index,
                "auction": bid_auction,
                "match_count": 0,
                "nv_count": 0,
                "v_count": 0,
                "avg_par_nv": None,
                "avg_par_v": None,
            })
            continue

        matched_df = deal_df.filter(global_mask)
        
        if matched_df.height == 0:
            # (already handled by mask.any() above, but kept for safety)
            bid_rankings.append({
                "bid": bid_name,
                "bt_index": bt_index,
                "auction": bid_auction,
                "match_count": 0,
                "nv_count": 0,
                "v_count": 0,
                "avg_par_nv": None,
                "avg_par_v": None,
                "avg_ev_precomputed_nv": round(avg_ev_precomputed_nv, 1) if avg_ev_precomputed_nv is not None else None,
                "avg_ev_precomputed_v": round(avg_ev_precomputed_v, 1) if avg_ev_precomputed_v is not None else None,
                "avg_par_precomputed_nv": round(avg_par_precomputed_nv, 1) if avg_par_precomputed_nv is not None else None,
                "avg_par_precomputed_v": round(avg_par_precomputed_v, 1) if avg_par_precomputed_v is not None else None,
            })
            continue
        
        # Collect indices for this bid - optimize by avoiding to_list() and using set() only if needed
        # Actually, for serialization, a list is fine.
        bid_matched_indices_list = []
        if "index" in matched_df.columns:
            # Using to_numpy().tolist() is often faster than to_list() for large Series
            bid_matched_indices_list = matched_df["index"].to_list()
        
        matched_by_bid[bid_name] = set(bid_matched_indices_list)
        match_count = len(bid_matched_indices_list)
        
        # Store the matched_df for this bid (for per-bid sampling later)
        matched_dfs_by_bid[bid_name] = matched_df
        
        # Parse bid to get level and strain for EV computation
        bid_level = bid_name[0] if bid_name and bid_name[0].isdigit() else None
        bid_strain = bid_name[1:].upper() if bid_name and len(bid_name) > 1 else None
        if bid_strain == "NT":
            bid_strain = "N"
        
        # Compute stats by vulnerability relative to the *next bidder seat* (seat-based, not direction-based)
        nv_count = 0
        v_count = 0
        nv_par_sum = 0.0
        v_par_sum = 0.0
        ev_nv_sum = 0.0
        ev_nv_sum_sq = 0.0
        ev_nv_n = 0
        ev_v_sum = 0.0
        ev_v_sum_sq = 0.0
        ev_v_n = 0
        
        # We can compute EV-at-bid from historical Score even if ParScore isn't loaded.
        # Par-related aggregates require ParScore, but EV/Std should not be blocked by it.
        if "Vul" in matched_df.columns and "Dealer" in matched_df.columns:
            # Determine the next bidder direction per deal using the seat mapping.
            dealer_to_bidder = _seat_direction_map(next_seat)
            bidder_expr = pl.col("Dealer").replace(dealer_to_bidder)

            # Vulnerability relative to bidder's side
            is_bidder_ns = bidder_expr.is_in(["N", "S"])
            is_vul_expr = pl.when(is_bidder_ns).then(pl.col("Vul").is_in(["N_S", "Both"])) \
                            .otherwise(pl.col("Vul").is_in(["E_W", "Both"]))
            nv_mask = ~is_vul_expr
            v_mask = is_vul_expr
            
            nv_df = matched_df.filter(nv_mask)
            v_df = matched_df.filter(v_mask)
            
            nv_count = nv_df.height
            v_count = v_df.height
            
            if nv_count > 0:
                # ParScore is NS-oriented; convert to "par from bidder's side" so it is
                # comparable to EV which is computed relative to the bidder's partnership.
                bidder_nv_df = nv_df.with_columns(bidder_expr.alias("_Bidder"))
                if "ParScore" in bidder_nv_df.columns:
                    nv_par = (
                        bidder_nv_df
                        .with_columns(
                            pl.when(pl.col("_Bidder").is_in(["N", "S"]))
                            .then(pl.col("ParScore"))
                            .otherwise(-pl.col("ParScore"))
                            .alias("_ParForBidder")
                        )["_ParForBidder"]
                        .drop_nulls()
                    )
                    if nv_par.len() > 0:
                        par_sum = nv_par.sum()
                        nv_par_sum = float(par_sum) if par_sum is not None else 0
                
                # EV-at-bid (and EV std) should be available even when per-contract EV columns
                # are not loaded into deal_df (they are very wide).
                #
                # Policy: compute "EV at Bid" from historical *Score* from the bidder's side
                # (NS score flipped for EW bidder). This matches the UI definition:
                # "average score achieved when that bid becomes the final contract".
                if "Score" in bidder_nv_df.columns:
                    score_for_bidder = (
                        bidder_nv_df
                        .with_columns(
                            pl.when(pl.col("_Bidder").is_in(["N", "S"]))
                            .then(pl.col("Score").cast(pl.Float64, strict=False))
                            .otherwise(-pl.col("Score").cast(pl.Float64, strict=False))
                            .alias("_ScoreForBidder")
                        )["_ScoreForBidder"]
                        .drop_nulls()
                    )
                    n = int(score_for_bidder.len())
                    if n > 0:
                        s = float(score_for_bidder.sum() or 0.0)
                        ss = float(((score_for_bidder * score_for_bidder).sum()) or 0.0)
                        ev_nv_sum += s
                        ev_nv_sum_sq += ss
                        ev_nv_n += n
            
            if v_count > 0:
                bidder_v_df = v_df.with_columns(bidder_expr.alias("_Bidder"))
                if "ParScore" in bidder_v_df.columns:
                    v_par = (
                        bidder_v_df
                        .with_columns(
                            pl.when(pl.col("_Bidder").is_in(["N", "S"]))
                            .then(pl.col("ParScore"))
                            .otherwise(-pl.col("ParScore"))
                            .alias("_ParForBidder")
                        )["_ParForBidder"]
                        .drop_nulls()
                    )
                    if v_par.len() > 0:
                        par_sum = v_par.sum()
                        v_par_sum = float(par_sum) if par_sum is not None else 0
                
                if "Score" in bidder_v_df.columns:
                    score_for_bidder = (
                        bidder_v_df
                        .with_columns(
                            pl.when(pl.col("_Bidder").is_in(["N", "S"]))
                            .then(pl.col("Score").cast(pl.Float64, strict=False))
                            .otherwise(-pl.col("Score").cast(pl.Float64, strict=False))
                            .alias("_ScoreForBidder")
                        )["_ScoreForBidder"]
                        .drop_nulls()
                    )
                    n = int(score_for_bidder.len())
                    if n > 0:
                        s = float(score_for_bidder.sum() or 0.0)
                        ss = float(((score_for_bidder * score_for_bidder).sum()) or 0.0)
                        ev_v_sum += s
                        ev_v_sum_sq += ss
                        ev_v_n += n
        
        # Compute averages
        avg_par_nv = round(nv_par_sum / nv_count, 0) if nv_count > 0 else None
        avg_par_v = round(v_par_sum / v_count, 0) if v_count > 0 else None
        
        # Compute EV mean and std (sample stddev)
        ev_score_nv = round(ev_nv_sum / ev_nv_n, 1) if ev_nv_n > 0 else None
        if ev_nv_n > 1:
            var = (ev_nv_sum_sq - (ev_nv_sum * ev_nv_sum) / ev_nv_n) / (ev_nv_n - 1)
            ev_std_nv = round(var ** 0.5, 1) if var >= 0 else None
        else:
            ev_std_nv = None

        ev_score_v = round(ev_v_sum / ev_v_n, 1) if ev_v_n > 0 else None
        if ev_v_n > 1:
            var = (ev_v_sum_sq - (ev_v_sum * ev_v_sum) / ev_v_n) / (ev_v_n - 1)
            ev_std_v = round(var ** 0.5, 1) if var >= 0 else None
        else:
            ev_std_v = None
        
        # Compute EV and Makes % for all level-strain-vul-seat combinations (560 columns)
        # IMPORTANT: S1..S4 are SEATS RELATIVE TO DEALER (Seat 1 = Dealer), not fixed N/E/S/W.
        # We must therefore compute the declarer direction per deal using (Dealer + Seat).
        # Use vectorized Polars expressions for performance.
        ev_all_combos: Dict[str, Optional[float]] = {}
        if matched_df.height > 0:
            # Initialize keys to None for stable schema
            for level in range(1, 8):
                for strain in ["C", "D", "H", "S", "N"]:
                    for vul_state in ["NV", "V"]:
                        for seat in [1, 2, 3, 4]:
                            ev_all_combos[f"EV_Score_{level}{strain}_{vul_state}_S{seat}"] = None
                            ev_all_combos[f"Makes_Pct_{level}{strain}_{vul_state}_S{seat}"] = None

            if "Dealer" in matched_df.columns and "Vul" in matched_df.columns:
                for seat in [1, 2, 3, 4]:
                    dealer_to_decl = _seat_direction_map(seat)
                    seat_df0 = matched_df.with_columns(
                        pl.col("Dealer").replace(dealer_to_decl).alias("_DeclDir")
                    )

                    is_decl_ns = pl.col("_DeclDir").is_in(["N", "S"])
                    is_vul_expr = pl.when(is_decl_ns).then(pl.col("Vul").is_in(["N_S", "Both"])) \
                                    .otherwise(pl.col("Vul").is_in(["E_W", "Both"]))

                    for vul_state in ["NV", "V"]:
                        seat_df = seat_df0.filter(is_vul_expr if vul_state == "V" else ~is_vul_expr)
                        if seat_df.height == 0:
                            continue

                        exprs: List[pl.Expr] = []
                        for level in range(1, 8):
                            for strain in ["C", "D", "H", "S", "N"]:
                                ev_key = f"EV_Score_{level}{strain}_{vul_state}_S{seat}"
                                makes_key = f"Makes_Pct_{level}{strain}_{vul_state}_S{seat}"

                                # EV: pick correct EV column per-deal based on _DeclDir, then mean
                                w_ev = None
                                for d in ["N", "E", "S", "W"]:
                                    pair = "NS" if d in ["N", "S"] else "EW"
                                    col = f"EV_{pair}_{d}_{strain}_{level}_{vul_state}"
                                    if col in seat_df.columns:
                                        if w_ev is None:
                                            w_ev = pl.when(pl.col("_DeclDir") == d).then(pl.col(col))
                                        else:
                                            w_ev = w_ev.when(pl.col("_DeclDir") == d).then(pl.col(col))
                                if w_ev is not None:
                                    exprs.append(w_ev.otherwise(None).mean().alias(ev_key))

                                # Makes %: pick correct DD_Score per-deal based on _DeclDir, then mean(made)*100
                                w_dd = None
                                for d in ["N", "E", "S", "W"]:
                                    col = f"DD_Score_{level}{strain}_{d}"
                                    if col in seat_df.columns:
                                        if w_dd is None:
                                            w_dd = pl.when(pl.col("_DeclDir") == d).then(pl.col(col))
                                        else:
                                            w_dd = w_dd.when(pl.col("_DeclDir") == d).then(pl.col(col))
                                if w_dd is not None:
                                    exprs.append(((w_dd.otherwise(None) >= 0).mean() * 100).alias(makes_key))

                        if exprs:
                            row = seat_df.select(exprs).row(0, named=True)
                            for k, v in row.items():
                                if v is not None:
                                    if k.startswith("EV_Score_"):
                                        ev_all_combos[k] = round(float(v), 1)
                                    else:
                                        ev_all_combos[k] = round(float(v), 1)
        
        # Store large EV/Makes blob once per bid (used by Streamlit "Contract EV Rankings")
        ev_all_combos_by_bid[bid_name] = ev_all_combos

        # Emit TWO rows per bid: one for NV, one for V (clearer UI, fewer columns)
        # Use matching NV/V precomputed values for each row
        bid_rankings.append(
            {
                "bid": bid_name,
                "bt_index": bt_index,
                "auction": bid_auction,
                "next_seat": next_seat,
                "vul": "NV",
                "match_count": nv_count,
                "match_total": match_count,
                "avg_par": avg_par_nv,
                "ev_score": ev_score_nv,
                "ev_std": ev_std_nv,
                "avg_ev_precomputed": round(avg_ev_precomputed_nv, 1) if avg_ev_precomputed_nv is not None else None,
                "avg_par_precomputed": round(avg_par_precomputed_nv, 1) if avg_par_precomputed_nv is not None else None,
            }
        )
        bid_rankings.append(
            {
                "bid": bid_name,
                "bt_index": bt_index,
                "auction": bid_auction,
                "next_seat": next_seat,
                "vul": "V",
                "match_count": v_count,
                "match_total": match_count,
                "avg_par": avg_par_v,
                "ev_score": ev_score_v,
                "ev_std": ev_std_v,
                "avg_ev_precomputed": round(avg_ev_precomputed_v, 1) if avg_ev_precomputed_v is not None else None,
                "avg_par_precomputed": round(avg_par_precomputed_v, 1) if avg_par_precomputed_v is not None else None,
            }
        )
    
    # Sort by EV-at-bid desc; then Avg Par desc; prefer NV rows; stabilize by bid name.
    def sort_key(r: Dict[str, Any]) -> Tuple[float, float, int, str]:
        ev = r.get("ev_score")
        ev_f = float(ev) if isinstance(ev, (int, float)) else float("-inf")
        ap = r.get("avg_par")
        ap_f = float(ap) if isinstance(ap, (int, float)) else float("-inf")
        vul_rank = 1 if r.get("vul") == "NV" else 0
        bid = str(r.get("bid") or "")
        return (ev_f, ap_f, vul_rank, bid)

    bid_rankings.sort(key=sort_key, reverse=True)
    
    # =========================================================================
    # STEP 3: Build deal data (bid tracking via matched_by_bid)
    # =========================================================================
    # Collect all matched indices (matched_by_bid already tracks bid -> deals)
    all_matched_indices: Set[int] = set()
    for indices in matched_by_bid.values():
        all_matched_indices.update(indices)
    
    # Serialization optimization: Base64 binary for matched_by_bid
    # Converting millions of integers to JSON lists is VERY slow in Python.
    # We send them as base64-encoded binary blobs of uint32 arrays.
    matched_by_bid_b64: Dict[str, str] = {}
    for bid, indices in matched_by_bid.items():
        if indices:
            # Convert set to sorted numpy uint32 array
            arr = np.array(sorted(list(indices)), dtype=np.uint32)
            # Encode to base64
            b64_str = base64.b64encode(arr.tobytes()).decode('utf-8')
            matched_by_bid_b64[bid] = b64_str
        else:
            matched_by_bid_b64[bid] = ""
    
    contract_recommendations: List[Dict[str, Any]] = []
    par_contract_stats: List[Dict[str, Any]] = []
    dd_data: List[Dict[str, Any]] = []
    dd_cols: List[str] = []
    total_matches = len(all_matched_indices)
    vul_breakdown: Dict[str, int] = {}
    
    if all_matched_indices:
        # Get the aggregated matched deals
        aggregated_df = deal_df.filter(pl.col("index").is_in(list(all_matched_indices)))
        
        # Apply vulnerability filter
        if vul_filter and vul_filter != "all":
            data_vul = _VUL_UI_TO_DATA.get(vul_filter, vul_filter)
            if "Vul" in aggregated_df.columns:
                aggregated_df = aggregated_df.filter(pl.col("Vul") == data_vul)
        
        total_matches = aggregated_df.height
        
        # Vul breakdown for debugging
        if "Vul" in aggregated_df.columns:
            vul_grouped = aggregated_df.group_by("Vul").len()
            vul_keys = vul_grouped["Vul"].to_list()
            vul_counts = vul_grouped["len"].to_list()
            vul_breakdown = {str(k): int(v) for k, v in zip(vul_keys, vul_counts)}
        
        # Sample for DD analysis
        stats_df = aggregated_df
        if aggregated_df.height > max_deals:
            stats_df = aggregated_df.sample(n=max_deals, seed=effective_seed)
        
        # Find DD columns
        dd_cols = [c for c in stats_df.columns if c.startswith("DD_")]
        
        # Compute contract recommendations
        if dd_cols:
            contract_recommendations = _compute_contract_recommendations(stats_df, dd_cols)
        
        # Compute par contract stats
        par_contract_stats = _compute_par_contract_stats(stats_df)
        
        # Build output columns
        select_cols = ["index", "Dealer", "Vul", "Vul_NS", "Vul_EW"]
        if include_hands:
            select_cols.extend(["Hand_N", "Hand_E", "Hand_S", "Hand_W"])
        select_cols.extend(["ParScore", "ParContracts"])
        if include_scores:
            select_cols.extend(dd_cols)
        
        # Filter to available columns
        select_cols = [c for c in select_cols if c in stats_df.columns]
        
        # Build deal data (bid tracking is via matched_by_bid, not per-deal)
        output_df = stats_df.select(select_cols)
        dd_data = output_df.to_dicts()
        
        # Format ParContracts
        for row in dd_data:
            if "ParContracts" in row:
                row["ParContracts"] = _format_par_contracts(row["ParContracts"])
    
    # =========================================================================
    # STEP 4: Build per-bid deal data (each bid gets up to max_deals)
    # =========================================================================
    dd_data_by_bid: Dict[str, List[Dict[str, Any]]] = {}
    
    for bid_name, bid_matched_df in matched_dfs_by_bid.items():
        if bid_matched_df.height == 0:
            dd_data_by_bid[bid_name] = []
            continue
        
        # Apply vulnerability filter
        filtered_bid_df = bid_matched_df
        if vul_filter and vul_filter != "all":
            data_vul = _VUL_UI_TO_DATA.get(vul_filter, vul_filter)
            if "Vul" in filtered_bid_df.columns:
                filtered_bid_df = filtered_bid_df.filter(pl.col("Vul") == data_vul)
        
        if filtered_bid_df.height == 0:
            dd_data_by_bid[bid_name] = []
            continue
        
        # Sample up to max_deals for this bid
        sampled_bid_df = filtered_bid_df
        if filtered_bid_df.height > max_deals:
            sampled_bid_df = filtered_bid_df.sample(n=max_deals, seed=effective_seed)
        
        # Parse bid to get level and strain for DD_Score/EV_Score computation
        # bid_name format: "1N", "2S", "3H", etc.
        bid_level = bid_name[0] if bid_name and bid_name[0].isdigit() else None
        bid_strain = bid_name[1:].upper() if bid_name and len(bid_name) > 1 else None
        # Normalize strain: "NT" -> "N", keep S/H/D/C as is
        if bid_strain == "NT":
            bid_strain = "N"
        
        # Build output columns - MINIMAL set for display (DD_Score/EV_Score computed separately)
        bid_select_cols = ["index", "Dealer", "Vul", "Vul_NS", "Vul_EW"]
        if include_hands:
            bid_select_cols.extend(["Hand_N", "Hand_E", "Hand_S", "Hand_W"])
        bid_select_cols.extend(["ParScore", "ParContracts"])
        
        # Only include the specific DD/EV columns needed for this bid (not all 280+)
        if bid_level and bid_strain:
            # Add DD_Score column for each declarer
            for decl in ["N", "E", "S", "W"]:
                dd_col = f"DD_Score_{bid_level}{bid_strain}_{decl}"
                if dd_col in sampled_bid_df.columns:
                    bid_select_cols.append(dd_col)
            # Add EV columns for this bid's level/strain only
            for pair in ["NS", "EW"]:
                for decl in ["N", "E", "S", "W"]:
                    for vul_st in ["NV", "V"]:
                        ev_col = f"EV_{pair}_{decl}_{bid_strain}_{bid_level}_{vul_st}"
                        if ev_col in sampled_bid_df.columns:
                            bid_select_cols.append(ev_col)
        
        
        # Use library-style vectorized column mapping for DD_Score and EV_Score
        # Patterned after mlBridgeAugmentLib.add_declarer_scores
        if bid_level and bid_strain:
            # 1. Add DD_Score for the matching bid (dependent on Dealer)
            # Build expression: "DD_Score_{level}{strain}_{Dealer}"
            dd_col_expr = pl.format("DD_Score_{}{}_{}", pl.lit(bid_level), pl.lit(bid_strain), pl.col("Dealer"))
            
            # 2. Add EV_Score for the matching bid (dependent on Dealer, Pair, and Vul)
            # Determine Pair and Vul_State side-by-side
            pair_expr = pl.when(pl.col("Dealer").is_in(["N", "S"])).then(pl.lit("NS")).otherwise(pl.lit("EW"))
            is_vul_expr = pl.when(pl.col("Dealer").is_in(["N", "S"])) \
                            .then(pl.col("Vul").is_in(["N_S", "Both"])) \
                            .otherwise(pl.col("Vul").is_in(["E_W", "Both"]))
            vul_state_expr = pl.when(is_vul_expr).then(pl.lit("V")).otherwise(pl.lit("NV"))
            
            # Build expression: "EV_{pair}_{declarer}_{strain}_{level}_{vul_state}"
            ev_col_expr = pl.format("EV_{}_{}_{}_{}_{}", 
                                    pair_expr, 
                                    pl.col("Dealer"), 
                                    pl.lit(bid_strain), 
                                    pl.lit(bid_level), 
                                    vul_state_expr)
            
            # Add alias columns using map_elements or dynamic selection
            # Since we can't easily use a dynamic column name in pl.col(), 
            # we'll use a trick: combine all possible columns into a struct and then pick one.
            
            # For DD_Score:
            dd_cols_for_bid = [f"DD_Score_{bid_level}{bid_strain}_{d}" for d in ["N", "E", "S", "W"]]
            existing_dd = [c for c in dd_cols_for_bid if c in sampled_bid_df.columns]
            if existing_dd:
                sampled_bid_df = sampled_bid_df.with_columns(
                    pl.struct(existing_dd).map_elements(
                        lambda r: r.get(f"DD_Score_{bid_level}{bid_strain}_{r.get('Dealer')}") if 'Dealer' in r else None,
                        return_dtype=pl.Int16
                    ).alias("DD_Score")
                )
            
            # For EV_Score:
            # (Similar logic but more complex, we'll keep the 8 columns for now but add the alias)
            existing_ev = [c for c in sampled_bid_df.columns if c.startswith(f"EV_") and f"_{bid_strain}_{bid_level}_" in c]
            if existing_ev:
                # We already added these to bid_select_cols, now pick the right one for each row
                def _get_ev_for_row(r):
                    d = r.get("Dealer")
                    v = "V" if (r.get("Vul") in ["N_S", "Both"] if d in ["N", "S"] else r.get("Vul") in ["E_W", "Both"]) else "NV"
                    p = "NS" if d in ["N", "S"] else "EW"
                    col = f"EV_{p}_{d}_{bid_strain}_{bid_level}_{v}"
                    return r.get(col)
                
                sampled_bid_df = sampled_bid_df.with_columns(
                    pl.struct(existing_ev + ["Dealer", "Vul"]).map_elements(
                        _get_ev_for_row, return_dtype=pl.Float64
                    ).cast(pl.Float32).alias("EV_Score")
                )

        bid_select_cols = [c for c in bid_select_cols if c in sampled_bid_df.columns]
        # Ensure our new alias columns are included
        for alias in ["DD_Score", "EV_Score"]:
            if alias in sampled_bid_df.columns:
                bid_select_cols.append(alias)
        
        bid_output_df = sampled_bid_df.select(bid_select_cols)
        bid_dd_data = bid_output_df.to_dicts()
        
        # Format ParContracts
        for row in bid_dd_data:
            if "ParContracts" in row:
                row["ParContracts"] = _format_par_contracts(row["ParContracts"])
        
        dd_data_by_bid[bid_name] = bid_dd_data
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[rank-bids-by-ev] {total_next_bids} bids analyzed, {total_matches} deals in {format_elapsed(elapsed_ms)}")
    
    # Describe the opening seat filter
    opening_seat = expected_passes + 1  # Seat 1-4
    opening_seat_desc = {0: "Dealer (Seat 1)", 1: "Seat 2 (after 1 pass)", 2: "Seat 3 (after 2 passes)", 3: "Seat 4 (after 3 passes)"}.get(expected_passes, f"Seat {opening_seat}")
    
    return {
        "auction_input": auction_input,
        "auction_normalized": auction_normalized,
        "parent_bt_row": parent_bt_row,
        "bid_rankings": bid_rankings,
        "total_next_bids": total_next_bids,
        "vul_filter": vul_filter,
        "expected_passes": expected_passes,
        "opening_seat": opening_seat_desc,
        # DD analysis results
        "contract_recommendations": contract_recommendations,
        "par_contract_stats": par_contract_stats,
        "dd_data": dd_data,
        "dd_data_by_bid": dd_data_by_bid,  # bid -> list of deal dicts (up to max_deals each)
        "ev_all_combos_by_bid": ev_all_combos_by_bid,  # bid -> {EV_Score_*, Makes_Pct_*}
        "dd_columns": dd_cols,
        "matched_by_bid_b64": matched_by_bid_b64,  # Optimized binary encoding
        "total_matches": total_matches,
        "returned_count": len(dd_data),
        "vul_breakdown": vul_breakdown,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_contract_ev_deals(
    state: Dict[str, Any],
    auction: str,
    next_bid: str,
    contract: str,
    declarer: str,
    seat: Optional[int],
    vul: str,
    max_deals: int,
    seed: Optional[int],
    include_hands: bool = True,
) -> Dict[str, Any]:
    """Return deal rows matching (auction prefix + selected next bid) AND a specific contract EV row.

    This is used to populate the Streamlit table under "Contract EV Rankings".
    It returns only the deal columns needed for display + the single EV/DD columns for the selected contract.
    """
    t0 = time.perf_counter()
    deal_df = state["deal_df"]
    bt_seat1_df = state.get("bt_seat1_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs") or {}

    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")

    auction_input = normalize_auction_input(auction)

    # Match the same BT lookup semantics as handle_rank_bids_by_ev
    expected_passes = _count_leading_passes(auction_input)
    auction_for_lookup = auction_input.rstrip("-") if auction_input else ""
    auction_normalized = re.sub(r"(?i)^(p-)+", "", auction_input) if auction_input else ""
    is_passes_only = not auction_normalized or auction_normalized.lower() in ("p", "")

    bt_row: Optional[Dict[str, Any]] = None

    if not auction_input:
        # Opening bids: MUST use bt_openings_df (fast). Do not scan bt_seat1_df (461M rows).
        bt_openings_df = state.get("bt_openings_df")
        if not isinstance(bt_openings_df, pl.DataFrame) or bt_openings_df.is_empty():
            raise ValueError(
                "bt_openings_df is missing/empty; refusing to scan bt_seat1_df for opening bids. "
                "Restart the API server and ensure initialization completes successfully."
            )
        seat1_df = bt_openings_df.filter(pl.col("seat") == 1) if "seat" in bt_openings_df.columns else bt_openings_df
        bid_norm = str(next_bid or "").strip().upper()
        if not bid_norm:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        bid_df = seat1_df.filter(pl.col("Auction") == bid_norm)
        if bid_df.height == 0:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        bt_row = dict(bid_df.row(0, named=True))
        # Normalize key expected by downstream logic
        if "candidate_bid" not in bt_row and bt_row.get("Auction"):
            bt_row["candidate_bid"] = bt_row["Auction"]
    else:
        # Resolve the parent BT node via traversal (no Auction scans), then fetch the selected next bid via bt_index lookups.
        if is_passes_only and expected_passes > 0:
            # Pass-only prefixes are not resolvable without Auction scans (disabled).
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        parent_bt_index = _resolve_bt_index_by_traversal(state, auction_for_lookup)
        if parent_bt_index is None:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        next_indices = _get_next_bid_indices_for_parent(state, parent_bt_index)
        if not next_indices:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        # Fetch only candidate rows in the next_indices set, then select the chosen bid.
        file_path = _bt_file_path_for_sql(state)
        in_list = ", ".join(str(int(x)) for x in next_indices if x is not None)
        bid_norm = str(next_bid or "").strip().upper()
        if not bid_norm:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        conn = duckdb.connect(":memory:")
        try:
            q = f"""
                SELECT bt_index, Auction, candidate_bid, is_completed_auction, Expr
                FROM read_parquet('{file_path}')
                WHERE bt_index IN ({in_list})
            """
            idx_df = conn.execute(q).pl()
        finally:
            conn.close()

        if idx_df.is_empty() or "candidate_bid" not in idx_df.columns:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        hit_df = idx_df.filter(pl.col("candidate_bid").cast(pl.Utf8).str.to_uppercase() == bid_norm)
        if hit_df.height == 0:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        bt_row = hit_df.row(0, named=True)

    if bt_row is None:
        return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    # Canonical: enrich Agg_Expr if missing + apply overlay + dedupe
    bt_row = _apply_all_rules_to_bt_row(dict(bt_row), state)
    opening_seat_mask = None
    if "bid" in deal_df.columns:
        if expected_passes == 0:
            opening_seat_mask = ~deal_df["bid"].str.starts_with("p-")
        else:
            prefix = "p-" * expected_passes
            not_extra_pass = ~deal_df["bid"].str.starts_with("p-" * (expected_passes + 1))
            opening_seat_mask = deal_df["bid"].str.starts_with(prefix) & not_extra_pass

    # Criteria mask for this bt_row
    global_mask, _invalid = _build_criteria_mask_for_all_seats(deal_df, bt_row, deal_criteria_by_seat_dfs)
    if global_mask is None or not global_mask.any():
        return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if opening_seat_mask is not None:
        global_mask = global_mask & opening_seat_mask
        if not global_mask.any():
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    matched_df = deal_df.filter(global_mask)
    if matched_df.height == 0:
        return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    # Parse contract (supports "4H" and "3N")
    contract = (contract or "").strip().upper()
    if len(contract) < 2 or not contract[0].isdigit():
        return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
    level = contract[0]
    strain = contract[1:]
    if strain == "NT":
        strain = "N"
    if strain not in ["C", "D", "H", "S", "N"]:
        # Standardize suit names to letters
        inv = {"C": "C", "D": "D", "H": "H", "S": "S"}
        strain = inv.get(strain, strain)

    declarer = (declarer or "").strip().upper()
    seat_i: Optional[int] = None
    try:
        if seat is not None:
            seat_i = max(1, min(4, int(seat)))
    except Exception:
        seat_i = None

    # For backwards compatibility, if seat is not provided, require declarer direction.
    if seat_i is None and declarer not in ["N", "E", "S", "W"]:
        return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    vul_state = (vul or "").strip().upper()
    if vul_state not in ["NV", "V"]:
        vul_state = "NV"

    # Filter by declarer-side vulnerability subset
    if "Vul" in matched_df.columns:
        if seat_i is None:
            # Legacy: declarer is fixed direction for all sampled deals
            if declarer in ["N", "S"]:
                nv_vuls = ["None", "E_W"]
                v_vuls = ["N_S", "Both"]
            else:
                nv_vuls = ["None", "N_S"]
                v_vuls = ["E_W", "Both"]
            target_vuls = nv_vuls if vul_state == "NV" else v_vuls
            matched_df = matched_df.filter(pl.col("Vul").is_in(target_vuls))
        else:
            # Seat-relative: declarer direction varies per deal based on Dealer + seat number
            dealer_to_decl = _seat_direction_map(seat_i)
            decl_expr = pl.col("Dealer").replace(dealer_to_decl).alias("_DeclDir")
            is_decl_ns = pl.col("_DeclDir").is_in(["N", "S"])
            is_vul_expr = pl.when(is_decl_ns).then(pl.col("Vul").is_in(["N_S", "Both"])) \
                            .otherwise(pl.col("Vul").is_in(["E_W", "Both"]))
            matched_df = matched_df.with_columns(decl_expr)
            matched_df = matched_df.filter(is_vul_expr if vul_state == "V" else ~is_vul_expr)

    total_matches = matched_df.height
    if total_matches == 0:
        return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    effective_seed = _effective_seed(seed)
    if max_deals and total_matches > max_deals:
        matched_df = matched_df.sample(n=max_deals, seed=effective_seed)

    # Select minimal output columns + contract-specific DD/EV
    out_cols = ["index", "Dealer", "Vul", "Vul_NS", "Vul_EW", "ParScore", "ParContracts"]
    if include_hands:
        out_cols.extend(["Hand_N", "Hand_E", "Hand_S", "Hand_W"])

    dd_col = f"DD_Score_{level}{strain}_{declarer}"
    pair = "NS" if declarer in ["N", "S"] else "EW"
    ev_col = f"EV_{pair}_{declarer}_{strain}_{level}_{vul_state}"

    # Seat-relative: compute DD_<contract> and EV_<contract> by selecting direction per deal.
    if seat_i is not None:
        # Ensure _DeclDir exists (we added it above when filtering by vul; add if not present)
        if "_DeclDir" not in matched_df.columns:
            matched_df = matched_df.with_columns(pl.col("Dealer").replace(_seat_direction_map(seat_i)).alias("_DeclDir"))

        # DD per deal: pick correct DD_Score_{contract}_{dir}
        dd_expr = None
        for d in ["N", "E", "S", "W"]:
            c = f"DD_Score_{level}{strain}_{d}"
            if c in matched_df.columns:
                e = pl.when(pl.col("_DeclDir") == d).then(pl.col(c))
                dd_expr = e if dd_expr is None else dd_expr.when(pl.col("_DeclDir") == d).then(pl.col(c))
        if dd_expr is not None:
            dd_expr = dd_expr.otherwise(None).alias(f"DD_{level}{strain}")
            matched_df = matched_df.with_columns(dd_expr)

        # EV per deal: pick correct EV_{pair}_{dir}_{strain}_{level}_{vul_state}
        ev_expr = None
        for d in ["N", "E", "S", "W"]:
            pair_d = "NS" if d in ["N", "S"] else "EW"
            c = f"EV_{pair_d}_{d}_{strain}_{level}_{vul_state}"
            if c in matched_df.columns:
                e = pl.when(pl.col("_DeclDir") == d).then(pl.col(c))
                ev_expr = e if ev_expr is None else ev_expr.when(pl.col("_DeclDir") == d).then(pl.col(c))
        if ev_expr is not None:
            ev_expr = ev_expr.otherwise(None).alias(f"EV_{level}{strain}")
            matched_df = matched_df.with_columns(ev_expr)

        # Add derived columns to output
        out_cols.extend([f"DD_{level}{strain}", f"EV_{level}{strain}"])
    else:
        # Legacy: include direction-specific raw columns and also DD_Score / EV_Score aliases
        if dd_col in matched_df.columns:
            out_cols.append(dd_col)
        if ev_col in matched_df.columns:
            out_cols.append(ev_col)

    out_df = matched_df.select([c for c in out_cols if c in matched_df.columns])
    deals = out_df.to_dicts()
    for row in deals:
        if "ParContracts" in row:
            row["ParContracts"] = _format_par_contracts(row["ParContracts"])
        # Keep legacy aliases for other UI surfaces
        row["DD_Score"] = row.get(dd_col)
        row["EV_Score"] = row.get(ev_col)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "deals": deals,
        "total_matches": int(total_matches),
        "returned_count": int(len(deals)),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_new_rules_lookup(
    state: Dict[str, Any],
    auction: str,
    bt_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Handle /new-rules-lookup endpoint."""
    t0 = time.perf_counter()
    new_rules_df = state.get("new_rules_df")
    
    if new_rules_df is None:
        return {"error": "New rules metrics (bbo_bt_new_rules.parquet) not loaded on server."}
    
    # Normalize auction for lookup
    auction_norm = auction.strip().upper()
    
    # Filter by auction
    # step_auction column contains the partial auction sequence
    # Fast exact match on step_auction
    filtered = new_rules_df.filter(pl.col("step_auction") == auction_norm)
    
    if filtered.height == 0:
        # Try finding by bt_index if provided
        if bt_index is not None:
            filtered = new_rules_df.filter(pl.col("bt_index") == bt_index)
    
    if filtered.height == 0:
        return {
            "found": False,
            "auction": auction,
            "bt_index": bt_index,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)
        }
    
    # Use first match
    row = filtered.row(0, named=True)
    
    # Extract criteria details
    # criteria_with_metrics: List(Struct) -> converted to List(Dict) by to_dicts()
    # base_rules: List(Utf8)
    # discovered_rules: List(Utf8)
    # New_Rules: List(Utf8)
    
    # In bbo_filter_new_rules.ipynb terminology:
    # Accepted_Criteria = subset of discovered_rules that are in New_Rules
    # Rejected_Criteria = discovered_rules NOT in New_Rules
    
    base_rules = list(row.get("base_rules") or [])
    discovered_rules = list(row.get("discovered_rules") or [])
    new_rules = list(row.get("New_Rules") or [])
    
    accepted = [c for c in discovered_rules if c in new_rules]
    rejected = [c for c in discovered_rules if c not in new_rules]
    
    # Deduplicate merged rules to keep least restrictive bounds
    merged_rules_deduped = dedupe_criteria_least_restrictive(new_rules)
    
    return {
        "found": True,
        "auction": row.get("step_auction"),
        "bt_index": row.get("bt_index"),
        "seat": row.get("seat"),
        "pos_count": row.get("pos_count"),
        "neg_count": row.get("neg_count"),
        "base_rules": base_rules,
        "accepted_criteria": accepted,
        "rejected_criteria": rejected,
        "merged_rules": new_rules,  # Raw merged rules (base + accepted)
        "merged_rules_deduped": merged_rules_deduped,  # Deduplicated with least restrictive bounds
        "criteria_details": row.get("criteria_with_metrics") or [],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)
    }


def _handle_list_next_bids_walk_fallback(
    state: Dict[str, Any],
    auction_input: str,
    auction_normalized: str,
    t0: float,
    *,
    include_deal_counts: bool = True,
    include_ev_stats: bool = True,
) -> Dict[str, Any]:
    """Gemini-3.2 optimized version: Uses CSR index and O(log n) Polars lookup.
    
    This replaces the slow DuckDB file scanning.
    """
    bt_seat1_df = state.get("bt_seat1_df")
    g3_index = state.get("g3_index")
    bt_index_arr = state.get("bt_index_arr")
    is_mono = state.get("bt_index_monotonic", False)
    bt_parquet_file = state.get("bt_seat1_file")
    bt_can_complete = state.get("bt_can_complete")

    # Handle empty auction (Opening Bids)
    if not auction_input:
        opening_indices = []
        if g3_index:
            opening_indices = list(g3_index.openings.values())
        else:
            bt_openings_df = state.get("bt_openings_df")
            if isinstance(bt_openings_df, pl.DataFrame):
                seat1_open = bt_openings_df.filter(pl.col("seat") == 1) if "seat" in bt_openings_df.columns else bt_openings_df
                opening_indices = [int(x) for x in seat1_open["bt_index"].to_list()]
        
        if not opening_indices:
            return {"auction_input": auction_input, "auction_normalized": auction_normalized, "next_bids": [], "elapsed_ms": 0.0}
        
        next_indices = opening_indices
        parent_bt_index = -1
    else:
        # Resolve parent bt_index
        parent_bt_index = _resolve_bt_index_by_traversal(state, auction_input)
        if parent_bt_index is None:
            return {"auction_input": auction_input, "auction_normalized": auction_normalized, "next_bids": [], "elapsed_ms": 0.0}

        # 2. Get next indices
        next_indices = _get_next_bid_indices_for_parent(state, parent_bt_index)
        if not next_indices:
            return {"auction_input": auction_input, "auction_normalized": auction_normalized, "next_bids": [], "elapsed_ms": 0.0}

    # 3. Fetch metadata for next-bid rows using Polars (in-memory)
    # Include next_bid_indices to detect dead ends
    next_bid_rows: pl.DataFrame
    select_cols = ["bt_index", "Auction", "candidate_bid", "is_completed_auction", "Expr", "matching_deal_count", "next_bid_indices"]
    if bt_seat1_df is not None and bt_index_arr is not None and is_mono:
        row_positions = np.searchsorted(bt_index_arr, next_indices)
        matched_positions = [pos for i, pos in enumerate(row_positions) if pos < len(bt_index_arr) and bt_index_arr[pos] == next_indices[i]]
        available_cols = [c for c in select_cols if c in bt_seat1_df.columns]
        if matched_positions:
            next_bid_rows = bt_seat1_df.select(available_cols).gather(matched_positions)
        else:
            next_bid_rows = bt_seat1_df.head(0).select(available_cols)
    else:
        # Fallback to DuckDB
        file_path = str(bt_parquet_file or "").replace("\\", "/")
        in_list = ", ".join(str(int(x)) for x in next_indices)
        import duckdb
        conn = duckdb.connect(":memory:")
        try:
            next_bid_rows = conn.execute(f"SELECT bt_index, Auction, candidate_bid, is_completed_auction, Expr, matching_deal_count, next_bid_indices FROM read_parquet('{file_path}') WHERE bt_index IN ({in_list})").pl()
        finally: conn.close()

    # 4. Enrichment with criteria (Prefer in-memory Expr to avoid heavy DuckDB hits)
    # The 'Expr' column in bt_seat1_df already contains the criteria for the candidate bid.
    # By pre-populating Agg_Expr_Seat_N, we prevent _apply_all_rules_to_bt_row from
    # triggering slow on-demand DuckDB hits for each candidate.
    next_seat = 1 if not auction_input else (len(auction_input.split("-")) % 4) + 1
    agg_col = f"Agg_Expr_Seat_{next_seat}"
    
    # 4b. Build base mask from parent's cumulative criteria (for accurate deal counts)
    base_mask: pl.Series | None = None
    if include_deal_counts and parent_bt_index >= 0:
        # Load parent row's Agg_Expr columns on-demand (they're not in bt_seat1_df)
        bt_parquet_file = state.get("bt_seat1_file")
        if bt_parquet_file:
            agg_data = _load_agg_expr_for_bt_indices([parent_bt_index], bt_parquet_file)
            if parent_bt_index in agg_data:
                parent_row: Dict[str, Any] = {"bt_index": parent_bt_index}
                parent_row.update(agg_data[parent_bt_index])
                parent_row = _apply_all_rules_to_bt_row(parent_row, state)
                # Build cumulative mask from ALL seats (criteria accumulate across rounds)
                # The parent row's Agg_Expr_Seat_N columns contain cumulative criteria
                base_mask = _compute_cumulative_deal_mask(state, parent_row, 4)

    # 5. Build EV stats lookup (if available) - now with NV/V splits
    # Pre-load EV stats for all bt_indices in one go (O(n) filter is faster than repeated lookups)
    bt_ev_stats_df = state.get("bt_ev_stats_df")
    ev_lookup: Dict[int, Dict[str, Any]] = {}
    if include_ev_stats and bt_ev_stats_df is not None and next_bid_rows.height > 0:
        bt_indices = next_bid_rows["bt_index"].unique().to_list()
        bt_indices = [int(x) for x in bt_indices if x is not None]
        if bt_indices:
            ev_subset = bt_ev_stats_df.filter(pl.col("bt_index").is_in(bt_indices))
            for ev_row in ev_subset.iter_rows(named=True):
                # Load NV/V split columns (new format) or fall back to aggregate (old format)
                ev_data: Dict[str, Any] = {}
                for s in range(1, 5):
                    # Try new NV/V split columns first
                    nv_col = f"Avg_EV_S{s}_NV"
                    v_col = f"Avg_EV_S{s}_V"
                    if nv_col in ev_row:
                        ev_data[nv_col] = ev_row.get(nv_col)
                        ev_data[v_col] = ev_row.get(v_col)
                    else:
                        # Fall back to old aggregate column (for backwards compatibility)
                        ev_data[f"Avg_EV_S{s}"] = ev_row.get(f"Avg_EV_S{s}")
                ev_lookup[int(ev_row["bt_index"])] = ev_data

    # 6. Build final response with on-demand deal counts
    next_bids = []
    for row in next_bid_rows.iter_rows(named=True):
        idx = row["bt_index"]
        row_dict = dict(row)
        
        # Fill Agg_Expr columns from in-memory 'Expr' to avoid DuckDB hits
        for s in range(1, 5):
            col_s = f"Agg_Expr_Seat_{s}"
            if col_s not in row_dict or row_dict[col_s] is None:
                row_dict[col_s] = row_dict.get("Expr") if s == next_seat else []
        
        # apply_overlay_and_dedupe handles CSV overlay and final cleanup
        row_with_rules = _apply_all_rules_to_bt_row(row_dict, state)
        crits = row_with_rules.get(agg_col) or []
        
        # Compute matching_deal_count on-demand by intersecting base_mask with new bid's criteria
        if include_deal_counts:
            deal_count = None
            try:
                deal_count = _compute_deal_count_with_base_mask(state, base_mask, row_with_rules, next_seat)
            except Exception:
                deal_count = row.get("matching_deal_count")  # Fall back to pre-computed
        else:
            # Cheap fallback: use pre-computed count (or None if missing)
            deal_count = row.get("matching_deal_count")
        
        # Detect dead end: not complete but has no children
        is_complete = bool(row.get("is_completed_auction", False))
        next_indices_list = row.get("next_bid_indices") or []
        is_dead_end = not is_complete and len(next_indices_list) == 0
        
        # Get Avg_EV for the next seat from precomputed stats (NV/V split)
        avg_ev_nv = None
        avg_ev_v = None
        if include_ev_stats and idx is not None and idx in ev_lookup:
            ev_data = ev_lookup[idx]
            # Try NV/V split columns first (new format)
            nv_key = f"Avg_EV_S{next_seat}_NV"
            v_key = f"Avg_EV_S{next_seat}_V"
            if nv_key in ev_data:
                avg_ev_nv = ev_data.get(nv_key)
                avg_ev_v = ev_data.get(v_key)
            else:
                # Fall back to aggregate (old format) - use for both NV and V
                aggregate = ev_data.get(f"Avg_EV_S{next_seat}")
                avg_ev_nv = aggregate
                avg_ev_v = aggregate
            
        next_bids.append({
            "bid": str(row.get("candidate_bid")).upper(),
            "bt_index": idx,
            # BT raw per-step criteria column (used as fallback to populate Agg_Expr_Seat_{next_seat})
            "expr": row.get("Expr") or [],
            "agg_expr": crits,
            "is_completed_auction": is_complete,
            "is_dead_end": is_dead_end,
            "matching_deal_count": deal_count,
            "avg_ev_nv": avg_ev_nv,
            "avg_ev_v": avg_ev_v,
            "can_complete": bool(bt_can_complete[int(idx)]) if (bt_can_complete is not None and idx is not None and int(idx) < len(bt_can_complete)) else None,
        })
    
    # Sort for UI consistency
    def sort_key(item: Dict[str, Any]) -> tuple:
        bid = item.get("bid", "")
        if bid == "P": return (2, 0, "")
        if bid in ("D", "R"): return (1, 0, bid)
        try:
            level = int(bid[0])
            suit = bid[1:] if len(bid) > 1 else ""
            suit_order = {"C": 1, "D": 2, "H": 3, "S": 4, "N": 5}.get(suit, 0)
            return (0, level, suit_order)
        except: return (0, 0, bid)
    
    next_bids.sort(key=sort_key)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"auction_input": auction_input, "next_bids": next_bids, "elapsed_ms": round(elapsed_ms, 1)}


def _handle_resolve_auction_path_fallback(
    state: Dict[str, Any],
    auction: str,
    t0: float,
) -> Dict[str, Any]:
    """Gemini-3.2 optimized version: Resolves an entire path in O(tokens) time."""
    g3_index = state.get("g3_index")
    if not g3_index: return {"path": [], "error": "G3 index not built", "elapsed_ms": 0.0}

    auction_input = normalize_auction_input(auction)
    
    # Extract leading passes before stripping for seat-1 view traversal
    leading_passes_match = re.match(r"(?i)^(P-)+", auction_input)
    leading_passes: List[str] = []
    if leading_passes_match:
        leading_passes = [t.upper() for t in leading_passes_match.group(0).rstrip("-").split("-") if t]
    
    # Strip leading passes for seat-1 view traversal (BT is indexed from first non-pass bid)
    auction_norm = re.sub(r"(?i)^(P-)+", "", auction_input).rstrip("-")
    tokens = [t.upper() for t in auction_norm.split("-") if t]
    
    # Handle pass-out auction (all passes, no actual bids)
    if not tokens:
        if leading_passes:
            # Return pass steps without BT info
            pass_path = [{"step": i + 1, "bid": "P", "bt_index": None, "agg_expr": [], "is_complete": len(leading_passes) >= 4, "categories": []} for i in range(len(leading_passes))]
            return {"path": pass_path, "elapsed_ms": (time.perf_counter() - t0) * 1000}
        return {"path": [], "elapsed_ms": 0.0}

    bt_seat1_df = state.get("bt_seat1_df")
    bt_index_arr = state.get("bt_index_arr")
    is_mono = state.get("bt_index_monotonic", False)
    bt_parquet_file = state.get("bt_seat1_file")

    path_info, curr_indices = [], []
    curr = g3_index.openings.get(tokens[0])
    if curr is not None:
        curr_indices.append(int(curr))
        for tok in tokens[1:]:
            code = BID_TO_CODE.get(tok, 0)
            start, end = g3_index.offsets[curr], g3_index.offsets[curr + 1]
            found = False
            for i in range(start, end):
                if g3_index.bidcodes[i] == code:
                    curr = int(g3_index.children[i]); found = True; break
            if not found: break
            curr_indices.append(curr)

    if not curr_indices: return {"path": [], "error": f"Path '{auction_norm}' not found", "elapsed_ms": 0.0}

    # Fetch metadata for all indices in one go
    if bt_seat1_df is not None and bt_index_arr is not None and is_mono:
        row_positions = np.searchsorted(bt_index_arr, curr_indices)
        matched_positions = [pos for i, pos in enumerate(row_positions) if pos < len(bt_index_arr) and bt_index_arr[pos] == curr_indices[i]]
        meta_df = bt_seat1_df.select(["bt_index", "Auction", "candidate_bid", "is_completed_auction", "Expr", "matching_deal_count"]).gather(matched_positions)
    else:
        in_list = ", ".join(str(idx) for idx in curr_indices)
        import duckdb
        conn = duckdb.connect(":memory:")
        try: meta_df = conn.execute(f"SELECT bt_index, Auction, candidate_bid, is_completed_auction, Expr, matching_deal_count FROM read_parquet('{bt_parquet_file or ''}') WHERE bt_index IN ({in_list})").pl()
        finally: conn.close()

    # Load Agg_Expr (Prefer in-memory Expr to avoid heavy DuckDB hits)
    meta_map = {row["bt_index"]: row for row in meta_df.to_dicts()}

    # Pre-load precomputed EV stats (NV/V split) for the whole path (optional)
    bt_ev_stats_df = state.get("bt_ev_stats_df")
    ev_map: Dict[int, Dict[str, Any]] = {}
    if bt_ev_stats_df is not None and curr_indices:
        try:
            bt_idx_list = [int(x) for x in curr_indices if x is not None]
            if bt_idx_list:
                ev_subset = bt_ev_stats_df.filter(pl.col("bt_index").is_in(bt_idx_list))
                for ev_row in ev_subset.iter_rows(named=True):
                    ev_map[int(ev_row["bt_index"])] = dict(ev_row)
        except Exception:
            ev_map = {}
    
    # Fetch categories for all indices in the path
    bt_categories_df = state.get("bt_categories_df")
    bt_category_cols = state.get("bt_category_cols") or []
    categories_by_idx: Dict[int, List[str]] = {}
    if bt_categories_df is not None and bt_category_cols:
        cat_idx_arr = bt_categories_df["bt_index"].to_numpy()
        for idx_val in curr_indices:
            pos = int(np.searchsorted(cat_idx_arr, int(idx_val)))
            if 0 <= pos < len(cat_idx_arr):
                try:
                    val_at_pos = cat_idx_arr[pos]
                    if not np.isnan(val_at_pos) and int(val_at_pos) == int(idx_val):
                        cats_true: List[str] = []
                        for c in bt_category_cols:
                            if bool(bt_categories_df[c][pos]):
                                cats_true.append(c[3:] if c.startswith("is_") else c)
                        categories_by_idx[int(idx_val)] = cats_true
                except (ValueError, TypeError):
                    continue

    # Prepend leading passes to path (no BT info for passes before opening)
    num_leading = len(leading_passes)
    for i in range(num_leading):
        path_info.append({
            "step": i + 1, "bid": "P", "bt_index": None, "agg_expr": [], 
            "is_complete": False, "matching_deal_count": None, "categories": [],
        })

    for i, idx in enumerate(curr_indices):
        row = meta_map.get(idx, {})
        row_dict = dict(row)
        # Seat calculation accounts for leading passes
        step_seat = ((num_leading + i) % 4) + 1
        
        # Populate Agg_Expr columns from in-memory 'Expr' to avoid DuckDB hits
        for s in range(1, 5):
            col_s = f"Agg_Expr_Seat_{s}"
            if col_s not in row_dict or row_dict[col_s] is None:
                row_dict[col_s] = row_dict.get("Expr") if s == step_seat else []

        # Apply CSV overlay logic if needed (via common helper)
        row_with_rules = _apply_all_rules_to_bt_row(row_dict, state)
        agg_col = f"Agg_Expr_Seat_{step_seat}"
        agg_list = row_with_rules.get(agg_col) or row_dict.get("Expr") or []

        # Attach precomputed Avg_EV (NV/V) if available for this bt_index/seat
        avg_ev_nv = None
        avg_ev_v = None
        ev_row = ev_map.get(int(idx), {})
        nv_key = f"Avg_EV_S{step_seat}_NV"
        v_key = f"Avg_EV_S{step_seat}_V"
        if nv_key in ev_row:
            avg_ev_nv = ev_row.get(nv_key)
            avg_ev_v = ev_row.get(v_key)
        else:
            # Backwards compatibility with old aggregate stats file
            aggregate = ev_row.get(f"Avg_EV_S{step_seat}")
            avg_ev_nv = aggregate
            avg_ev_v = aggregate
        
        path_info.append({
            "step": num_leading + i + 1, "bid": tokens[i], "bt_index": idx, 
            "agg_expr": agg_list, "is_complete": bool(row.get("is_completed_auction")),
            "matching_deal_count": row.get("matching_deal_count"),
            "avg_ev_nv": avg_ev_nv,
            "avg_ev_v": avg_ev_v,
            "categories": categories_by_idx.get(int(idx), []),
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"path": path_info, "elapsed_ms": round(elapsed_ms, 1)}


# ---------------------------------------------------------------------------
# Best Auctions Lookahead – Server-side DFS for DD/EV ranking
# ---------------------------------------------------------------------------

def handle_best_auctions_lookahead(
    state: Dict[str, Any],
    deal_row_idx: int,
    auction_prefix: str,
    metric: str,  # "DD" or "EV"
    max_depth: int = 20,
    max_results: int = 10,
    deadline_s: float = 5.0,
    max_nodes: int = 50000,  # max prefix expansions per request (safety cap)
    beam_width: int = 25,    # how many children to explore per node (lookahead + dfs)
) -> Dict[str, Any]:
    """Server-side DFS to find best completed auctions by DD or EV.
    
    Uses CSR index for O(1) next-bid traversal and bitmap DFs for O(1) criteria eval.
    Single request replaces dozens of client-side API calls.
    
    Returns:
        - auctions: List of {auction, contract, dd_score, ev, is_par}
        - par_score: Deal's par score
        - elapsed_ms: Processing time
    """
    t0 = time.perf_counter()
    # NOTE: This endpoint can be used both as a "quick lookahead" (short deadline)
    # and as a longer-running background job. Keep robust safety caps.
    try:
        deadline_s = float(deadline_s)
    except Exception:
        deadline_s = 5.0
    # Cap to avoid runaway server load (background jobs can increase this, but not unbounded).
    deadline_s = max(0.1, min(deadline_s, 3600.0))
    deadline_t = t0 + deadline_s

    try:
        max_nodes = int(max_nodes)
    except Exception:
        max_nodes = 50000
    max_nodes = max(0, min(max_nodes, 2_000_000))

    try:
        beam_width = int(beam_width)
    except Exception:
        beam_width = 25
    beam_width = max(1, min(beam_width, 200))

    expanded_nodes = 0
    
    deal_df = state.get("deal_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    g3_index = state.get("g3_index")
    
    if deal_df is None:
        raise ValueError("deal_df not loaded")
    
    # Get deal row
    try:
        deal_row = deal_df.row(int(deal_row_idx), named=True)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid deal_row_idx {deal_row_idx}: {e}")
    
    dealer_actual = str(deal_row.get("Dealer", "N")).upper()
    par_score = deal_row.get("ParScore", deal_row.get("Par_Score"))
    par_score_i: int | None = None
    try:
        par_score_i = int(par_score) if par_score is not None else None
    except Exception:
        par_score_i = None
    
    # Normalize auction prefix
    auction_norm = normalize_auction_input(auction_prefix).upper() if auction_prefix else ""
    # IMPORTANT: Auction Builder tends to operate in BT seat-1 view (leading passes stripped),
    # but DD/EV/par computations require seat-relative prefixes to align with the deal's dealer.
    # Ensure the search prefix includes the deal's opening passes so scoring/par matching works.
    deal_actual_auction_s = _bid_value_to_str(deal_row.get("bid") or deal_row.get("Actual_Auction") or "")
    deal_lp = _count_leading_passes(deal_actual_auction_s)
    prefix_lp = _count_leading_passes(auction_norm)
    if int(deal_lp) > int(prefix_lp):
        auction_norm = ("P-" * int(int(deal_lp) - int(prefix_lp))) + auction_norm
    
    # Custom CSV overlay (bbo_custom_auction_criteria.csv).
    # These rules are defined in BT "seat-1 view" (leading passes stripped) and apply to *any*
    # auction that starts with the rule's partial.
    overlay = state.get("custom_criteria_overlay") or []
    overlay_by_seat: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
    try:
        for r in list(overlay) if isinstance(overlay, list) else []:
            try:
                s = int(r.get("seat"))  # type: ignore[call-arg]
            except Exception:
                continue
            if 1 <= s <= 4:
                overlay_by_seat[s].append(dict(r))
    except Exception:
        overlay_by_seat = {1: [], 2: [], 3: [], 4: []}

    # Reuse a single DuckDB connection for the request and cache bt_index->metadata.
    # The previous implementation created a new DuckDB connection per node, which is extremely slow.
    file_path = _bt_file_path_for_sql(state)
    conn = duckdb.connect(":memory:")
    bt_meta_cache: Dict[int, Tuple[bool, List[str]]] = {}  # bt_index -> (is_completed, Expr list)
    bt_meta_loaded = 0

    def _hydrate_bt_meta(bt_indices: List[int]) -> None:
        """Populate bt_meta_cache for any missing bt_index values."""
        nonlocal bt_meta_loaded
        missing = [int(x) for x in bt_indices if int(x) not in bt_meta_cache]
        if not missing:
            return
        # Chunk to keep SQL manageable
        chunk_size = 1000
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i : i + chunk_size]
            if not chunk:
                continue
            in_list = ", ".join(str(int(x)) for x in chunk)
            # PERFORMANCE: Only load essential columns.
            # Agg_Expr_Seat_* are huge (lists of strings) and slow to load.
            # Use Expr directly - it's the criteria for this specific bid.
            rows = conn.execute(
                f"""
                SELECT bt_index, is_completed_auction, Expr
                FROM read_parquet('{file_path}')
                WHERE bt_index IN ({in_list})
                """
            ).fetchall()
            
            for bt_idx, is_comp, expr in rows:
                # Store Expr as the criteria for all seats (it's bid-specific)
                expr_list = list(expr) if expr else []
                bt_meta_cache[int(bt_idx)] = (
                    bool(is_comp),
                    expr_list,  # Single list, not per-seat
                )
            bt_meta_loaded += len(rows)
    
    # Helper: strip leading passes for BT lookup
    def strip_lp(auc: str) -> tuple[str, int]:
        toks = [t.strip().upper() for t in auc.split("-") if t.strip()] if auc else []
        n = 0
        for t in toks:
            if t == "P":
                n += 1
            else:
                break
        if n >= len(toks):
            return "", n
        return "-".join(toks[n:]), n
    
    # Helper: check if auction is complete
    def is_complete(auc: str) -> bool:
        bids = [b.strip().upper() for b in auc.split("-") if b.strip()]
        if len(bids) >= 4 and all(b == "P" for b in bids[:4]):
            return True
        last_c = -1
        for i, b in enumerate(bids):
            if b not in ("P", "X", "XX") and b and b[0].isdigit():
                last_c = i
        return last_c >= 0 and len(bids) >= last_c + 4 and all(b == "P" for b in bids[-3:])
    
    # Helper: get contract string
    def get_contract(auc: str) -> str:
        from bbo_bidding_queries_lib import parse_contract_from_auction
        c = parse_contract_from_auction(auc)
        if c:
            l, s, _ = c
            return f"{l}{'NT' if str(s).upper() == 'N' else str(s).upper()}"
        toks = [t.strip().upper() for t in auc.split("-") if t.strip()]
        return "Passed out" if toks and all(t == "P" for t in toks) else "?"
    
    # Helper: evaluate criteria for a deal (O(1) bitmap lookup)
    # STRICT POLICY: all criteria must be evaluatable and true.
    # - SL/complex criteria must be computable for this deal/seat (otherwise fail)
    # - Bitmap criteria must exist in the bitmap DF and be true (otherwise fail)
    def eval_criteria(criteria_list: List[str], bt_seat: int, dealer_rot: str) -> bool:
        if not criteria_list:
            return True  # Empty criteria = pass
        
        criteria_df = deal_criteria_by_seat_dfs.get(bt_seat, {}).get(dealer_rot)
        if criteria_df is None:
            return False
        available_cols = set(criteria_df.columns)
        
        for crit in criteria_list:
            if crit is None:
                continue
            crit_s = str(crit)
            
            # Try dynamic SL evaluation first
            sl_result = evaluate_sl_criterion(crit_s, dealer_rot, bt_seat, deal_row, fail_on_missing=False)
            if sl_result is True:
                continue
            elif sl_result is False:
                return False
            
            # If it's an SL criterion (or SL-style comparison) but can't be evaluated => FAIL (strict)
            if parse_sl_comparison_relative(crit_s) is not None or parse_sl_comparison_numeric(crit_s) is not None:
                return False
            
            # Bitmap lookup
            if crit_s not in available_cols:
                return False
            try:
                if not bool(criteria_df[crit_s][int(deal_row_idx)]):
                    return False
            except Exception:
                return False
        
        return True
    
    # Helper: get next bids from CSR index with criteria filtering
    def get_valid_next_bids(prefix_auc: str) -> List[Tuple[str, int, bool]]:
        """Returns list of (bid, bt_index, is_completed) for criteria-pass bids."""
        bt_prefix, lp = strip_lp(prefix_auc)
        
        # Resolve current bt_index
        if not bt_prefix:
            # Opening bids
            openings = g3_index.openings if g3_index else {}
            parent_idx = None
        else:
            parent_idx = _resolve_bt_index_by_traversal(state, bt_prefix)
            if parent_idx is None:
                return []
        
        # Get children
        if parent_idx is None:
            # Use openings dict for empty prefix
            if not g3_index or not g3_index.openings:
                return []
            child_map = dict(g3_index.openings)
        else:
            child_map = _get_child_map_for_parent(state, parent_idx)
        
        if not child_map:
            return []
        
        # Load metadata for children (Expr, is_completed_auction) with in-request caching.
        child_indices = list(child_map.values())
        t_hydrate = time.perf_counter()
        _hydrate_bt_meta(child_indices)
        _ = (time.perf_counter() - t_hydrate) * 1000  # keep timing local (no debug payload)
        
        # Seat/dealer for criteria evaluation
        toks = [t for t in prefix_auc.split("-") if t.strip()] if prefix_auc else []
        display_seat = (len(toks) % 4) + 1
        bt_seat = ((display_seat - 1 - lp) % 4) + 1
        dealer_rot = DIRECTIONS_LIST[(DIRECTIONS_LIST.index(dealer_actual) + lp) % 4]
        
        # Filter by criteria
        valid: List[Tuple[str, int, bool]] = []
        for bid, idx in child_map.items():
            m = bt_meta_cache.get(int(idx))
            if m is None:
                continue
            is_comp, expr = m
            # Expr is the criteria for this specific bid (candidate_bid at this BT row).
            # Evaluate it for the seat that is making this bid.
            if not eval_criteria(expr, bt_seat, dealer_rot):
                continue

            # ALSO enforce the custom CSV overlay rules (if any) at this exact BT seat.
            # The overlay is keyed in seat-1 view; `bt_prefix` is already stripped to that view.
            # The "partial" match should be checked against the seat-1 view auction string.
            rules_for_seat = overlay_by_seat.get(int(bt_seat), [])
            if rules_for_seat:
                bt_child_auc = f"{bt_prefix}-{bid}" if bt_prefix else str(bid)
                bt_child_auc = bt_child_auc.upper()
                from bbo_bidding_queries_lib import pattern_matches
                overlay_ok = True
                for rr in rules_for_seat:
                    partial = str(rr.get("partial") or "").strip().upper()
                    if not partial:
                        continue
                    if not pattern_matches(partial, bt_child_auc):
                        continue
                    criteria_list = rr.get("criteria") or []
                    # Must satisfy ALL criteria for this overlay rule.
                    try:
                        if not eval_criteria([str(x) for x in criteria_list if x is not None], bt_seat, dealer_rot):
                            overlay_ok = False
                            break
                    except Exception:
                        overlay_ok = False
                        break
                if not overlay_ok:
                    continue

            valid.append((bid, int(idx), bool(is_comp)))
        
        return valid
    
    # Caches for DFS
    dd_cache: Dict[str, int] = {}
    ev_cache: Dict[str, float] = {}
    ui_valid_cache: Dict[str, List[Tuple[str, int, bool]]] = {}
    best_dd_cache: Dict[Tuple[str, int], int] = {}     # (prefix, depth_remaining) -> best DD reachable
    best_ev_cache: Dict[Tuple[str, int], float] = {}   # (prefix, depth_remaining) -> best EV reachable
    
    def get_dd(auc: str) -> int:
        if auc in dd_cache:
            return dd_cache[auc]
        try:
            v = get_dd_score_for_auction(auc, dealer_actual, deal_row)
            out = int(v) if v is not None else -99999
        except Exception:
            out = -99999
        dd_cache[auc] = out
        return out
    
    def get_ev(auc: str) -> float:
        if auc in ev_cache:
            return ev_cache[auc]
        try:
            v = get_ev_for_auction(auc, dealer_actual, deal_row)
            out = float(v) if v is not None else float("-inf")
        except Exception:
            out = float("-inf")
        ev_cache[auc] = out
        return out
    
    # Results: (score, dd, ev, auction, contract)
    results: List[Tuple[float, int, float, str, str]] = []

    def _metric(dd_v: int, ev_v: float) -> float:
        return float(dd_v) if metric.upper() == "DD" else float(ev_v)

    def _best_reachable(prefix_auc: str, depth_remaining: int) -> float:
        """Return best reachable completed metric from this prefix within depth_remaining.
        
        This is the critical fix: we cannot prioritize by the *standing* contract's DD/EV
        for artificial bids. We need lookahead to know which branch can actually reach
        a good completed auction.
        """
        nonlocal expanded_nodes
        depth = max_depth - depth_remaining

        if depth_remaining <= 0:
            return float("-inf")
        if time.perf_counter() > deadline_t or expanded_nodes >= max_nodes:
            return float("-inf")

        key = (prefix_auc, depth_remaining)
        if metric.upper() == "DD":
            if key in best_dd_cache:
                return float(best_dd_cache[key])
        else:
            if key in best_ev_cache:
                return float(best_ev_cache[key])

        # If already complete, score it.
        if is_complete(prefix_auc):
            dd_v = get_dd(prefix_auc)
            ev_v = get_ev(prefix_auc)
            out = _metric(dd_v, ev_v)
            if metric.upper() == "DD":
                best_dd_cache[key] = int(out)
            else:
                best_ev_cache[key] = float(out)
            return out

        # Expand children (criteria-pass)
        valid = ui_valid_cache.get(prefix_auc)
        if valid is None:
            expanded_nodes += 1
            valid = get_valid_next_bids(prefix_auc)
            ui_valid_cache[prefix_auc] = valid
        
        if not valid:
            if metric.upper() == "DD":
                best_dd_cache[key] = -999999
            else:
                best_ev_cache[key] = float("-inf")
            return float("-inf")

        # Order children by quick heuristic, then compute best among top beam_width.
        heur: List[Tuple[float, str, bool]] = []
        for bid, _idx, is_comp in valid:
            child_auc = f"{prefix_auc}-{bid}" if prefix_auc else bid
            dd_v = get_dd(child_auc)
            ev_v = get_ev(child_auc)
            # Mildly prefer passing towards completion once a contract exists.
            bonus = 0.01 if bid == "P" else 0.0
            heur.append((_metric(dd_v, ev_v) + bonus, child_auc, bool(is_comp) or is_complete(child_auc)))
        heur.sort(key=lambda x: x[0], reverse=True)

        best = float("-inf")
        for _h, child_auc, child_complete in heur[:beam_width]:
            if time.perf_counter() > deadline_t or expanded_nodes >= max_nodes:
                break
            if child_complete:
                dd_v = get_dd(child_auc)
                ev_v = get_ev(child_auc)
                score = _metric(dd_v, ev_v)
                best = max(best, score)
            else:
                best = max(best, _best_reachable(child_auc, depth_remaining - 1))

        if metric.upper() == "DD":
            best_dd_cache[key] = int(best) if best != float("-inf") else -999999
        else:
            best_ev_cache[key] = float(best)
        return best
    
    def dfs(prefix_auc: str, depth: int) -> None:
        nonlocal expanded_nodes
        if time.perf_counter() > deadline_t:
            return
        if expanded_nodes >= max_nodes:
            return
        if depth >= max_depth:
            return
        if len(results) >= max_results * 50:
            return
        
        cache_key = prefix_auc
        if cache_key in ui_valid_cache:
            valid = ui_valid_cache[cache_key]
        else:
            expanded_nodes += 1
            valid = get_valid_next_bids(prefix_auc)
            ui_valid_cache[cache_key] = valid
        
        if not valid:
            return
        
        # Score and sort candidates by lookahead best reachable metric
        scored: List[Tuple[float, str, int, bool, str]] = []
        for bid, idx, is_comp in valid:
            child_auc = f"{prefix_auc}-{bid}" if prefix_auc else bid
            child_complete = bool(is_comp) or is_complete(child_auc)
            if child_complete:
                dd_v = get_dd(child_auc)
                ev_v = get_ev(child_auc)
                score = _metric(dd_v, ev_v)
                results.append((score, dd_v, ev_v, child_auc, get_contract(child_auc)))
                scored.append((score, bid, idx, True, child_auc))
            else:
                # Lookahead to avoid missing artificial-bid branches.
                score = _best_reachable(child_auc, max_depth - depth - 1)
                scored.append((score, bid, idx, False, child_auc))
        
        # Sort by lookahead score descending, explore top beam_width.
        scored.sort(key=lambda x: x[0], reverse=True)
        
        for score, bid, idx, is_comp, child_auc in scored[:beam_width]:
            if time.perf_counter() > deadline_t or expanded_nodes >= max_nodes:
                break
            if is_comp:
                continue  # Already added to results
            dfs(child_auc, depth + 1)
    
    # Start DFS
    try:
        dfs(auction_norm, 0)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    
    # Deduplicate and sort results
    results.sort(key=lambda x: x[0], reverse=True)
    seen: Set[str] = set()
    output: List[Dict[str, Any]] = []
    
    for score, dd_v, ev_v, auc, contract in results:
        if auc in seen:
            continue
        seen.add(auc)
        is_par = par_score_i is not None and dd_v == par_score_i
        output.append({
            "auction": auc,
            "contract": contract,
            "dd_score": dd_v if dd_v > -90000 else None,
            "ev": round(ev_v, 1) if ev_v != float("-inf") else None,
            "is_par": is_par,
        })
    final = output[:max_results]
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "auctions": final,
        "par_score": par_score_i,
        "metric": metric.upper(),
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Deal Matched BT Sample – sample BT rows for a pinned deal
# ---------------------------------------------------------------------------

def handle_deal_matched_bt_sample(
    state: Dict[str, Any],
    deal_row_idx: int,
    n_samples: int = 25,
    seed: int = 0,
    metric: str = "DD",
    permissive_pass: bool = True,
) -> Dict[str, Any]:
    """Return BT rows that match a specific deal (GPU-verified index).

    Uses `state["deal_to_bt_index_df"]` (loaded from `bbo_deal_to_bt_verified.parquet`) which maps:
      deal_idx (row position in deals file) -> Matched_BT_Indices (list of bt_index values)
    
    Args:
        permissive_pass: If True, Pass bids are always treated as valid even if criteria fails.
    """
    t0 = time.perf_counter()
    import numpy as np

    deal_df = state.get("deal_df")
    if deal_df is None:
        raise ValueError("deal_df not loaded")

    # Deal row is needed for scoring (DD/EV) + dealer/par
    try:
        deal_row = deal_df.row(int(deal_row_idx), named=True)
    except Exception as e:
        raise ValueError(f"Invalid deal_row_idx {deal_row_idx}: {e}")
    dealer_actual = str(deal_row.get("Dealer", "N")).upper()
    # Lead passes matter for mapping BT seat-1 view → dealer-relative seats.
    # BT seat-1 view strips opening passes; criteria are stored as Agg_Expr_Seat_1..4 in that view.
    # For conformance checks against deal bitmaps (dealer-relative), rotate criteria by the deal's leading passes.
    actual_auction_s = _bid_value_to_str(deal_row.get("bid") or deal_row.get("Actual_Auction") or "")
    lead_passes = _count_leading_passes(actual_auction_s)
    auction_prefix = ("P-" * int(lead_passes)) if int(lead_passes) > 0 else ""
    par_score = deal_row.get("ParScore", deal_row.get("Par_Score"))
    try:
        par_score_i = int(par_score) if par_score is not None else None
    except Exception:
        par_score_i = None

    deal_to_bt_index_df: Optional[pl.DataFrame] = state.get("deal_to_bt_index_df")
    if deal_to_bt_index_df is None or deal_to_bt_index_df.height <= 0:
        raise ValueError("deal_to_bt_index_df not loaded (run bbo_bt_filter_by_bitmap.py and restart API).")

    # Cache numpy materializations across requests for O(log n) lookup.
    global _DEAL_TO_BT_VERIFIED_INDEX_CACHE  # type: ignore[declared-but-unused]
    try:
        _DEAL_TO_BT_VERIFIED_INDEX_CACHE  # type: ignore[name-defined]
    except Exception:
        _DEAL_TO_BT_VERIFIED_INDEX_CACHE = {}  # type: ignore[name-defined]

    cache = _DEAL_TO_BT_VERIFIED_INDEX_CACHE  # type: ignore[name-defined]
    df_id = id(deal_to_bt_index_df)
    if cache.get("df_id") != df_id:
        cache["df_id"] = df_id
        cache["deal_idx_arr"] = deal_to_bt_index_df["deal_idx"].to_numpy()
        cache["matched_series"] = deal_to_bt_index_df.get_column("Matched_BT_Indices")

    deal_idx_arr = cache.get("deal_idx_arr")
    matched_series = cache.get("matched_series")
    if deal_idx_arr is None or matched_series is None:
        raise ValueError("Verified index cache not initialized")

    pos = np.searchsorted(deal_idx_arr, int(deal_row_idx))
    matches: list[int] = []
    if pos < len(deal_idx_arr) and int(deal_idx_arr[pos]) == int(deal_row_idx):
        try:
            m = matched_series[int(pos)]
        except Exception:
            m = None
        if m is None:
            matches = []
        else:
            try:
                if isinstance(m, pl.Series):
                    matches = [int(x) for x in m.to_list() if x is not None]
                elif isinstance(m, (list, tuple)):
                    matches = [int(x) for x in m if x is not None]
                else:
                    matches = [int(x) for x in list(m) if x is not None]
            except Exception:
                matches = []

    total_matches = len(matches)
    if total_matches == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "rows": [],
            "counts": {"total_matches": 0, "returned": 0},
            "metric": str(metric or "DD").upper(),
            "par_score": par_score_i,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    # Deterministic policy (per UI spec):
    # 1) Consider up to a fixed cap of matched BT indices (sorted by bt_index).
    # 2) Score all considered rows for the pinned deal.
    # 3) Sort by (DD/EV desc, bt_index asc).
    # 4) Return only the first `n_samples` rows for display.
    #
    # NOTE: `n_samples` is the UI "Max Best Auctions" (display cap).
    want = int(n_samples)
    want = max(1, min(want, 1000))  # display cap guardrail

    CONSIDER_CAP = 1000
    consider_n = min(int(CONSIDER_CAP), int(total_matches))
    matches_sorted = sorted(matches)
    consider_bt = matches_sorted[:consider_n]

    # Fetch BT metadata for considered indices via DuckDB (avoid scanning huge in-memory BT).
    file_path = _bt_file_path_for_sql(state)
    conn = duckdb.connect(":memory:")
    try:
        in_list = ", ".join(str(int(x)) for x in consider_bt)
        bt_df = conn.execute(
            f"""
            SELECT bt_index, Auction, is_completed_auction, matching_deal_count
            FROM read_parquet('{file_path}')
            WHERE bt_index IN ({in_list})
            """
        ).pl()
        # Keep output stable: preserve consider_bt order
        order_rank = {int(b): i for i, b in enumerate(consider_bt)}
        if "bt_index" in bt_df.columns:
            bt_df = bt_df.with_columns(
                pl.col("bt_index")
                .cast(pl.Int64)
                .map_elements(lambda x: order_rank.get(int(x), 10**12))
                .alias("_rank")
            ).sort("_rank").drop("_rank")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    from bbo_bidding_queries_lib import parse_contract_from_auction

    # Apply CSV overlay (custom_criteria_overlay) before scoring/display.
    # The deal→BT verified index is computed from the compiled BT; overlay can add criteria on top.
    # Filter out any BT rows that no longer match the pinned deal once overlay is applied.
    bt_rows_for_eval: list[dict[str, Any]] = []
    filtered_out = 0
    try:
        bt_seat1_file = state.get("bt_seat1_file")
        deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
        if bt_seat1_file is not None and isinstance(deal_criteria_by_seat_dfs, dict):
            bt_indices = [int(x) for x in (bt_df.get_column("bt_index").to_list() if "bt_index" in bt_df.columns else []) if x is not None]
            agg_map = _load_agg_expr_for_bt_indices(bt_indices, bt_seat1_file) if bt_indices else {}
            for row in bt_df.iter_rows(named=True):
                row_dict = dict(row)
                bt_index = row_dict.get("bt_index")
                if bt_index is not None:
                    try:
                        row_dict.update(agg_map.get(int(bt_index), {}))
                    except Exception:
                        pass
                # Apply overlay + dedupe (will NOT re-load Agg_Expr since we pre-filled it above).
                row_rules = _apply_overlay_and_dedupe(row_dict, state)
                # Check conformance for this deal_row_idx against (possibly augmented) criteria.
                # Use permissive_pass to allow 'P' bids even if their criteria fails.
                # This matches the client-side logic where 'P' is always a valid bid choice.
                rotated_rules = {
                    **row_rules,
                    # Rotate Agg_Expr_Seat_k from BT seat-1 view to dealer-relative seats.
                    # dealer_seat = (lead_passes + bt_seat - 1) % 4 + 1
                    **{
                        f"Agg_Expr_Seat_{((int(lead_passes) + (bt_seat - 1)) % 4) + 1}": row_rules.get(
                            f"Agg_Expr_Seat_{bt_seat}"
                        )
                        for bt_seat in range(1, 5)
                    },
                }
                auction_for_check = f"{auction_prefix}{str(row_rules.get('Auction') or '')}"
                conf = _check_deal_criteria_conformance_bitmap(
                    int(deal_row_idx),
                    rotated_rules,
                    dealer_actual,
                    deal_criteria_by_seat_dfs,
                    deal_row=deal_row,
                    auction=auction_for_check,
                    permissive_pass=permissive_pass,
                )
                if conf.get("first_wrong_seat") is None:
                    bt_rows_for_eval.append(row_rules)
                else:
                    filtered_out += 1
        else:
            bt_rows_for_eval = [dict(r) for r in bt_df.iter_rows(named=True)]
    except Exception:
        # If anything goes wrong, fall back to the raw BT rows (best-effort).
        bt_rows_for_eval = [dict(r) for r in bt_df.iter_rows(named=True)]

    # Fetch pattern-based counts (Matches) for all auctions using the same API as bid options
    auctions_list = [str(row.get("Auction") or "") for row in bt_rows_for_eval]
    pattern_counts: dict[str, int] = {}
    if auctions_list:
        # Build regex patterns for exact auction matches (completed auctions end with -P-P-P)
        # Use same pattern format as bid options: exact match for completed auctions
        patterns = []
        auc_to_pattern = {}
        for auc in auctions_list:
            auc_upper = auc.upper()
            # For completed auctions, use exact match pattern
            if auc_upper.endswith("-P-P-P"):
                pat = f"^{auc_upper}$"
            else:
                # For incomplete auctions, match any completion
                pat = f"^{auc_upper}.*-P-P-P$"
            patterns.append(pat)
            auc_to_pattern[auc] = pat
        unique_patterns = list(dict.fromkeys(patterns))
        try:
            pattern_counts_resp = handle_auction_pattern_counts(state, unique_patterns)
            counts_by_pattern = pattern_counts_resp.get("counts", {}) or {}
            # Map patterns back to auctions
            for auc in auctions_list:
                pat = auc_to_pattern[auc]
                pattern_counts[auc] = int(counts_by_pattern.get(pat, 0) or 0)
        except Exception:
            # Fallback: use 0 for all if pattern count fetch fails
            for auc in auctions_list:
                pattern_counts[auc] = 0

    out_rows: list[dict[str, Any]] = []
    metric_u = str(metric or "DD").upper()
    for row in bt_rows_for_eval:
        auc_raw = str(row.get("Auction") or "")
        # Ensure auction string includes the deal's opening passes for correct dealer-relative scoring.
        # (BT/criteria logic often uses seat-1 view; scoring/par matching must align to actual dealer.)
        auc_lp = _count_leading_passes(auc_raw)
        diff_lp = int(lead_passes) - int(auc_lp)
        auc = (("P-" * diff_lp) + auc_raw) if diff_lp > 0 else auc_raw
        bt_index = row.get("bt_index")
        is_completed = bool(row.get("is_completed_auction"))

        # Contract
        contract = "?"
        try:
            c = parse_contract_from_auction(auc)
            if c:
                lvl, strain, _ = c
                contract = f"{lvl}{'NT' if str(strain).upper() == 'N' else str(strain).upper()}"
            else:
                toks = [t for t in auc.split("-") if t.strip()]
                if toks and all(str(t).strip().upper() == "P" for t in toks):
                    contract = "Passed out"
        except Exception:
            contract = "?"

        dd_score: int | None = None
        ev: float | None = None
        try:
            dd_v = get_dd_score_for_auction(auc, dealer_actual, deal_row)
            dd_score = int(dd_v) if dd_v is not None else None
        except Exception:
            dd_score = None
        try:
            ev_v = get_ev_for_auction(auc, dealer_actual, deal_row)
            ev = round(float(ev_v), 1) if ev_v is not None else None
        except Exception:
            ev = None

        is_par = par_score_i is not None and dd_score is not None and int(dd_score) == int(par_score_i)
        matching_deal_count = row.get("matching_deal_count")
        # Matches: pattern-based count (same as bid options)
        matches_count = pattern_counts.get(auc, 0)
        # Deals: criteria-based count from BT row (same as bid options)
        deals_count = matching_deal_count if matching_deal_count is not None else ""
        out_rows.append(
            {
                "bt_index": bt_index,
                "Auction": auc,
                "Contract": contract,
                "DD_Score": dd_score,
                "EV": ev,
                "Par": "✅" if is_par else "",
                "Matches": matches_count,
                "Deals": deals_count,
                "is_completed_auction": is_completed,
                "Score": dd_score if metric_u == "DD" else ev,
            }
        )

    # Sort rows deterministically for UI:
    # - DD view: DD desc, bt_index asc
    # - EV view: EV desc, bt_index asc
    def _safe_bt(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 2**31 - 1

    def _safe_dd(x: Any) -> int:
        # None should sink to bottom for descending sort
        try:
            return int(x)
        except Exception:
            return -999_999

    def _safe_ev(x: Any) -> float:
        # None should sink to bottom for descending sort
        try:
            return float(x)
        except Exception:
            return float("-inf")

    if metric_u == "DD":
        out_rows.sort(key=lambda r: (-_safe_dd(r.get("DD_Score")), _safe_bt(r.get("bt_index"))))
    else:
        out_rows.sort(key=lambda r: (-_safe_ev(r.get("EV")), _safe_bt(r.get("bt_index"))))

    # Display cap (after scoring/sorting).
    out_rows = out_rows[:want]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "rows": out_rows,
        "counts": {"total_matches": total_matches, "considered": consider_n, "filtered_out": filtered_out, "returned": len(out_rows)},
        "metric": metric_u,
        "par_score": par_score_i,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_greedy_model_path(
    state: Dict[str, Any],
    auction_prefix: str,
    deal_row_idx: Optional[int] = None,
    seed: int = 42,
    max_depth: int = 40,
    permissive_pass: bool = True,
) -> Dict[str, Any]:
    """Compute the greedy 'model path' from a given prefix by picking the top bid at each step.
    
    Optimized version:
    1. Replicates 'Best Bids Ranked by Model' logic.
    2. Minimizes DuckDB overhead.
    3. Optimized criteria evaluation for single deal.
    """
    t0 = time.perf_counter()
    deal_df = state.get("deal_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    g3_index = state.get("g3_index")
    bt_can_complete = state.get("bt_can_complete")
    bt_openings_df = state.get("bt_openings_df")
    bt_ev_stats_df = state.get("bt_ev_stats_df")
    
    debug_info: Dict[str, Any] = {"steps_tried": 0, "break_reason": None, "forcing_pass_debug": []}
    
    if deal_df is None:
        return {"greedy_path": "", "steps": 0, "elapsed_ms": 0, "error": "deal_df not loaded", "debug": debug_info}
    
    deal_row = None
    if deal_row_idx is not None:
        try:
            deal_row = deal_df.row(int(deal_row_idx), named=True)
        except Exception as e:
            debug_info["deal_row_error"] = str(e)
            
    dealer_actual = str(deal_row.get("Dealer", "N")).upper() if deal_row else "N"
    
    current_auc_input = normalize_auction_input(auction_prefix).upper() if auction_prefix else ""
    path_bids = [t.strip() for t in current_auc_input.split("-") if t.strip()]
    # Track the chosen criteria for each bid in `path_bids` (same index).
    # For the initial prefix (if any), we don't have criteria here; leave empty.
    chosen_agg_expr_by_step: list[list[Any]] = [[] for _ in path_bids]
    chosen_expr_by_step: list[list[Any]] = [[] for _ in path_bids]
    chosen_bt_index_by_step: list[int | None] = [None for _ in path_bids]
    
    file_path = _bt_file_path_for_sql(state)
    conn = duckdb.connect(":memory:")
    
    # Pre-extract bid vocab
    _, code_to_bid = _get_local_bid_vocab()
    
    # Performance optimization: if deal_row_idx is provided, pre-fetch all criterion bits for this deal
    # to avoid repeated Series indexing in the loop.
    deal_bits_cache: Dict[Tuple[int, str], Dict[str, bool]] = {} # (bt_seat, dealer_rot) -> {crit_s: val}

    def _get_deal_bits(bt_seat: int, dealer_rot: str) -> Dict[str, bool]:
        key = (bt_seat, dealer_rot)
        if key in deal_bits_cache:
            return deal_bits_cache[key]
        
        bits = {}
        criteria_df = deal_criteria_by_seat_dfs.get(bt_seat, {}).get(dealer_rot)
        if criteria_df is not None and deal_row_idx is not None:
            # This is still a bit slow but we only do it once per (seat, dealer) combo
            idx = int(deal_row_idx)
            # Fetching the whole row as a dict is faster than fetching each column individually in a loop
            try:
                row = criteria_df.row(idx, named=True)
                bits = {k: bool(v) for k, v in row.items()}
            except:
                pass
        deal_bits_cache[key] = bits
        return bits

    def _eval_criteria_fast(criteria_list: List[str], bt_seat: int, dealer_rot: str) -> bool:
        if not criteria_list: return True
        if deal_row is None: return True
        
        bits = _get_deal_bits(bt_seat, dealer_rot)
        
        for crit in criteria_list:
            if crit is None: continue
            crit_s = str(crit)
            
            # SL evaluation
            sl_result = evaluate_sl_criterion(crit_s, dealer_rot, bt_seat, deal_row, fail_on_missing=False)
            if sl_result is True: continue
            elif sl_result is False: return False
            
            if parse_sl_comparison_relative(crit_s) is not None or parse_sl_comparison_numeric(crit_s) is not None:
                continue
            
            # Bitmap (cached)
            if crit_s in bits:
                if not bits[crit_s]:
                    return False
        return True

    def _iter_criteria_tokens(criteria_val: Any) -> list[str]:
        """Normalize Expr/Agg_Expr shapes into a list of upper-case tokens."""
        if criteria_val is None:
            return []
        # Common shapes:
        # - list[str]
        # - list[Any]
        # - single str
        if isinstance(criteria_val, str):
            s = criteria_val.strip()
            return [s.upper()] if s else []
        if isinstance(criteria_val, (list, tuple)):
            out: list[str] = []
            for x in criteria_val:
                try:
                    xs = str(x).strip()
                    if xs:
                        out.append(xs.upper())
                except Exception:
                    continue
            return out
        try:
            s = str(criteria_val).strip()
            return [s.upper()] if s else []
        except Exception:
            return []

    def _has_forcing_to_3n(expr_list: Any, agg_expr_list: Any) -> bool:
        # Forcing flag may be present either in raw Expr(s)/Criteria or in Agg_Expr.
        for tok in (_iter_criteria_tokens(expr_list) + _iter_criteria_tokens(agg_expr_list)):
            # Be permissive: sometimes a criterion can be embedded in a larger expression string.
            # Example: "Forcing_To_3N & Some_Other_Flag"
            if "FORCING_TO_3N" in tok:
                return True
        return False

    # NOTE: Do not fall back to scanning full BT rows for forcing detection.
    # Per project rule, use only partner's per-step Expr(s) and Agg_Expr(s) that came from list-next-bids.

    def _is_contract_below_3nt(auction: str) -> bool:
        """True iff the current contract (last bid) is strictly below 3NT."""
        try:
            toks0 = [t.strip().upper() for t in str(auction or "").split("-") if t.strip()]
        except Exception:
            toks0 = []
        last_level: int | None = None
        last_strain: str | None = None  # "C/D/H/S/N"
        for t in toks0:
            if len(t) == 2 and t[0] in "1234567" and t[1] in "CDHSN":
                try:
                    last_level = int(t[0])
                    last_strain = t[1]
                except Exception:
                    continue
        # No contract yet -> below 3NT
        if last_level is None or last_strain is None:
            return True
        suit_order = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}
        try:
            return (int(last_level), int(suit_order.get(str(last_strain).upper(), 0))) < (3, 4)
        except Exception:
            return True

    # Partnership-level Forcing_To_3N state for the greedy path.
    forcing_to_3n_active_by_side: dict[str, bool] = {"NS": False, "EW": False}
    forcing_to_3n_passes_below_3nt_by_side: dict[str, int] = {"NS": 0, "EW": 0}

    bt_seat1_file = state.get("bt_seat1_file") or _bt_file_path_for_sql(state)
    _agg_cache: dict[int, dict[str, Any]] = {}

    def _enrich_expr_and_agg_expr(
        bt_idx: int | None,
        bt_step_seat: int,
        *,
        fallback_expr: list[Any],
        fallback_agg: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Best-effort enrich chosen Expr/Agg_Expr from the BT row (on-demand).

        Rationale: list-next-bids may return only per-step Expr-derived criteria when bt_seat1_df
        is loaded in lightweight mode. Some aggregated flags (like Forcing_To_3N) can live only in
        Agg_Expr_Seat_* columns and must be loaded from the parquet on-demand.
        """
        if bt_idx is None or bt_seat1_file is None:
            return fallback_expr, fallback_agg
        try:
            bt_i = int(bt_idx)
        except Exception:
            return fallback_expr, fallback_agg
        try:
            row = _agg_cache.get(bt_i)
            if row is None:
                agg = _load_agg_expr_for_bt_indices([bt_i], bt_seat1_file)
                if not isinstance(agg, dict) or bt_i not in agg:
                    return fallback_expr, fallback_agg
                base_row: dict[str, Any] = {"bt_index": bt_i}
                base_row.update(agg.get(bt_i) or {})
                row = _apply_all_rules_to_bt_row(base_row, state)
                _agg_cache[bt_i] = row
            expr_val = row.get("Expr") or fallback_expr
            agg_val = row.get(f"Agg_Expr_Seat_{int(bt_step_seat)}") or fallback_agg
            return expr_val, agg_val
        except Exception:
            return fallback_expr, fallback_agg

    for step_i in range(max_depth):
        debug_info["steps_tried"] = step_i + 1
        
        if _is_auction_complete_list(path_bids):
            debug_info["break_reason"] = "complete"
            break
            
        prefix = "-".join(path_bids)
        # Correct LP logic
        toks = [t.strip().upper() for t in prefix.split("-") if t.strip()]
        n_lp = 0
        for t in toks:
            if t == "P": n_lp += 1
            else: break
        bt_prefix_str = "-".join(toks[n_lp:]) if n_lp < len(toks) else ""
        
        # Use the same next-bid materialization as the UI (/list-next-bids),
        # to avoid any mismatch in criteria selection/overlay/dedupe.
        try:
            auction_input = normalize_auction_input(bt_prefix_str or "")
            auction_normalized = re.sub(r"(?i)^(p-)+", "", auction_input) if auction_input else ""
            next_resp = _handle_list_next_bids_walk_fallback(
                state,
                auction_input,
                auction_normalized,
                time.perf_counter(),
                include_deal_counts=False,  # critical: avoid per-step bitmap scans
                include_ev_stats=False,
            )
            next_bids = next_resp.get("next_bids", []) or []
        except Exception as e:
            debug_info["break_reason"] = f"list_next_bids_error:{e}"
            break

        # UI behavior parity: at opening, ensure "P" is available
        if bt_prefix_str == "":
            has_p = any(str(b.get("bid", "")).upper() == "P" for b in next_bids)
            if not has_p:
                next_bids = list(next_bids) + [{
                    "bid": "P",
                    "bt_index": None,
                    "agg_expr": [],
                    "is_dead_end": False,
                    "can_complete": True,
                    "matching_deal_count": 0,
                    "is_completed_auction": False,
                }]

        if not next_bids:
            debug_info["break_reason"] = f"no_children:{bt_prefix_str or '(opening)'}"
            break
        
        # Seat/dealer
        display_seat = (len(path_bids) % 4) + 1
        bt_seat = ((display_seat - 1 - n_lp) % 4) + 1
        dealer_rot = DIRECTIONS_LIST[(DIRECTIONS_LIST.index(dealer_actual) + n_lp) % 4]
        
        # Materialize row_dicts / crits_list in the same shape as before
        row_dicts: list[dict[str, Any]] = []
        crits_list: list[list] = []
        for b in next_bids:
            bid = str(b.get("bid", "")).upper()
            row_dicts.append({
                "bt_index": b.get("bt_index"),
                "candidate_bid": bid,
                "is_completed_auction": bool(b.get("is_completed_auction", False) or b.get("is_completed", False)),
                "matching_deal_count": b.get("matching_deal_count", 0),
                # Prefer explicit flags from list-next-bids when present
                "is_dead_end": bool(b.get("is_dead_end", False)),
                "can_complete": b.get("can_complete"),
                # Optional (may be absent)
                "next_bid_indices": b.get("next_bid_indices") or [],
                # Optional: raw per-step criteria
                "expr": b.get("expr") or [],
            })
            crits_list.append(b.get("agg_expr") or [])

        # Batch criteria evaluation (matches UI logic)
        matches_pinned_by_row = [True] * len(row_dicts)
        if deal_row is not None and deal_row_idx is not None:
            checks: list[dict[str, Any]] = []
            check_idx: list[int | None] = [None] * len(row_dicts)
            for i, crits in enumerate(crits_list):
                if crits:
                    check_idx[i] = len(checks)
                    checks.append({"seat": bt_seat, "criteria": list(crits)})
            if checks:
                batch_resp = handle_deal_criteria_failures_batch(
                    state=state,
                    deal_row_idx=int(deal_row_idx),
                    dealer=dealer_rot,
                    checks=checks,
                )
                results = batch_resp.get("results", [])
                for i, idx in enumerate(check_idx):
                    if idx is None:
                        matches_pinned_by_row[i] = True
                    elif idx < len(results):
                        failed = results[idx].get("failed", []) or []
                        untracked = results[idx].get("untracked", []) or []
                        matches_pinned_by_row[i] = (len(failed) == 0 and len(untracked) == 0)
                    else:
                        matches_pinned_by_row[i] = True

        # Candidates for greedy choice
        candidates = []
        rejected_count = 0
        criteria_fail_count = 0
        total_children = len(row_dicts)

        # Forcing logic:
        # If a side has Forcing_To_3N active, then while the contract is still below 3NT:
        # - If the immediately previous call was an opponent Pass, the forcing side to act may NOT pass.
        #   (This blocks the common "partner forced → opponent passed → responder cannot pass" case.)
        # - Additionally, that side may make at most 1 Pass below 3NT; further passes below 3NT are rejected.
        bidder_idx = len(path_bids)
        bidder_dir = DIRECTIONS_LIST[(DIRECTIONS_LIST.index(dealer_actual) + bidder_idx) % 4]
        bidder_side = "NS" if bidder_dir in ("N", "S") else "EW"
        bidder_forcing_active_before = bool(forcing_to_3n_active_by_side.get(bidder_side, False))
        contract_below_3nt = _is_contract_below_3nt(prefix)
        prev_call = str(path_bids[-1]).strip().upper() if path_bids else ""
        prev_side: str | None = None
        try:
            if path_bids:
                prev_bidder_idx = len(path_bids) - 1
                prev_bidder_dir = DIRECTIONS_LIST[(DIRECTIONS_LIST.index(dealer_actual) + prev_bidder_idx) % 4]
                prev_side = "NS" if prev_bidder_dir in ("N", "S") else "EW"
        except Exception:
            prev_side = None

        block_pass_after_opponent_pass = (
            bidder_forcing_active_before
            and contract_below_3nt
            and prev_call in ("P", "PASS")
            and (prev_side is not None and prev_side != bidder_side)
        )
        block_pass_for_bidder = (
            block_pass_after_opponent_pass
            or (
                bidder_forcing_active_before
                and contract_below_3nt
                and int(forcing_to_3n_passes_below_3nt_by_side.get(bidder_side, 0) or 0) >= 1
            )
        )
        try:
            debug_info["forcing_pass_debug"].append({
                "step": step_i + 1,
                "prefix": prefix,
                "bidder_dir": bidder_dir,
                "bidder_side": bidder_side,
                "prev_call": prev_call,
                "prev_side": prev_side,
                "contract_below_3nt": contract_below_3nt,
                "forcing_active": dict(forcing_to_3n_active_by_side),
                "passes_below_3nt": dict(forcing_to_3n_passes_below_3nt_by_side),
                "block_pass_after_opponent_pass": bool(block_pass_after_opponent_pass),
                "block_pass_for_bidder": bool(block_pass_for_bidder),
            })
        except Exception:
            pass
        
        for i, row_dict in enumerate(row_dicts):
            bt_idx = row_dict["bt_index"]
            bid = row_dict["candidate_bid"]
            is_comp_auc = bool(row_dict.get("is_completed_auction"))
            matches_count = row_dict.get("matching_deal_count")
            next_indices_list = row_dict.get("next_bid_indices") or []
            crits = crits_list[i]
            exprs = row_dict.get("expr") or []
            
            # Determine if Rejected
            is_pass = str(bid).upper() in ("P", "PASS")
            # Enforce forcing sequences: once forcing is active, do not allow 2nd/3rd pass below 3NT.
            if block_pass_for_bidder and is_pass:
                rejected_count += 1
                continue
            # Prefer explicit dead-end flag from list-next-bids if present
            if "is_dead_end" in row_dict:
                is_dead_end = bool(row_dict.get("is_dead_end"))
            else:
                is_dead_end = (not is_comp_auc) and len(next_indices_list) == 0
            has_empty_criteria = not crits
            if row_dict.get("can_complete") is not None:
                can_complete = bool(row_dict.get("can_complete"))
            else:
                can_complete = bool(bt_can_complete[int(bt_idx)]) if (bt_can_complete is not None and bt_idx is not None and int(bt_idx) < len(bt_can_complete)) else True
            
            if deal_row:
                is_rejected = is_dead_end or has_empty_criteria
                if is_pass and not crits: is_rejected = False
            else:
                is_rejected = is_dead_end and not can_complete

            # UI parity: if permissive-pass is enabled, Pass is never rejected.
            if permissive_pass and is_pass:
                is_rejected = False
            
            # Check pinned deal match (batched evaluation)
            matches_pinned = matches_pinned_by_row[i]
            # UI parity: if permissive-pass is enabled, Pass always matches pinned deal.
            if permissive_pass and is_pass:
                matches_pinned = True
            
            if deal_row and matches_pinned and not can_complete:
                if not ((is_pass and not crits) or (permissive_pass and is_pass)):
                    is_rejected = True

            if is_rejected:
                rejected_count += 1
                continue
            if not matches_pinned:
                criteria_fail_count += 1
                continue
                
            # Compute scores for sorting
            dd_score = float("-inf")
            ev_score = 0.0
            
            next_auc = f"{prefix}-{bid}" if prefix else bid
            if deal_row:
                try:
                    # Side sign logic
                    bidder_idx = len(path_bids)
                    bidder_dir = DIRECTIONS_LIST[(DIRECTIONS_LIST.index(dealer_actual) + bidder_idx) % 4]
                    bidder_side = "NS" if bidder_dir in ("N", "S") else "EW"
                    
                    # DD Score (side-relative)
                    declarer = get_declarer_for_auction(next_auc, dealer_actual)
                    if declarer:
                        raw_dd = get_dd_score_for_auction(next_auc, dealer_actual, deal_row)
                        if raw_dd is not None:
                            declarer_side = "NS" if str(declarer).upper() in ("N", "S") else "EW"
                            side_sign = 1.0 if bidder_side == declarer_side else -1.0
                            dd_score = float(side_sign * float(raw_dd))
                    
                    # EV Score (side-relative)
                    raw_ev = get_ev_for_auction(next_auc, dealer_actual, deal_row)
                    if raw_ev is not None:
                        if declarer:
                            declarer_side = "NS" if str(declarer).upper() in ("N", "S") else "EW"
                            side_sign = 1.0 if bidder_side == declarer_side else -1.0
                            ev_score = float(side_sign * float(raw_ev))
                        else:
                            ev_score = float(raw_ev)
                except: pass

            # UI parity: if the auction so far is only passes (opening/lead-pass sequences),
            # treat Pass as neutral (DD=0, EV=0) so it doesn't get pushed below negative
            # DD-score actions just because DD is the first sort key.
            try:
                toks_now = [t.strip().upper() for t in str(prefix or "").split("-") if t.strip()]
                prefix_all_passes = (len(toks_now) == 0) or all(t == "P" for t in toks_now)
            except Exception:
                prefix_all_passes = False
            if prefix_all_passes and str(bid).upper() == "P":
                dd_score = 0.0
                ev_score = 0.0
            
            candidates.append({
                "bid": bid,
                "sort_key": (-dd_score, -ev_score, -float(matches_count or 0), str(bid).upper()),
                "_agg_expr": crits,
                "_expr": exprs,
                "_bt_index": bt_idx,
            })
        
        if not candidates:
            debug_info["break_reason"] = f"no_valid_candidates:total={total_children},rejected={rejected_count},criteria_fail={criteria_fail_count},at_step={step_i},prefix={prefix}"
            break
        
        candidates.sort(key=lambda x: x["sort_key"])
        best = candidates[0]
        best_bid = best["bid"]
        path_bids.append(best_bid)
        best_bt_idx = best.get("_bt_index")
        best_expr_raw = best.get("_expr") or []
        best_agg_raw = best.get("_agg_expr") or []
        # Enrich to get true aggregated criteria (Agg_Expr_Seat_{bt_seat}) when available.
        best_expr, best_agg = _enrich_expr_and_agg_expr(
            best_bt_idx,
            bt_seat,
            fallback_expr=list(best_expr_raw) if isinstance(best_expr_raw, list) else [],
            fallback_agg=list(best_agg_raw) if isinstance(best_agg_raw, list) else [],
        )
        chosen_agg_expr_by_step.append(best_agg or [])
        chosen_expr_by_step.append(best_expr or [])
        chosen_bt_index_by_step.append(best_bt_idx)

        # Update partnership forcing state and pass counter (based on the bid we just took).
        try:
            best_is_pass = str(best_bid).strip().upper() in ("P", "PASS")
            # Use enriched Expr/Agg_Expr for forcing detection.
            best_has_forcing = _has_forcing_to_3n(best_expr, best_agg)
            if best_has_forcing:
                forcing_to_3n_active_by_side[bidder_side] = True
            # Count passes only if forcing was already active at the time of the pass
            # and we are still below 3NT at that time.
            if best_is_pass and bidder_forcing_active_before and contract_below_3nt:
                forcing_to_3n_passes_below_3nt_by_side[bidder_side] = int(
                    forcing_to_3n_passes_below_3nt_by_side.get(bidder_side, 0) or 0
                ) + 1
            try:
                if debug_info.get("forcing_pass_debug"):
                    debug_info["forcing_pass_debug"][-1].update({
                        "chosen_bid": best_bid,
                        "chosen_is_pass": bool(best_is_pass),
                        "chosen_has_forcing": bool(best_has_forcing),
                        "chosen_bt_index": best_bt_idx,
                        "chosen_bt_seat": int(bt_seat),
                        "bt_seat1_file_present": bool(bt_seat1_file),
                        "chosen_expr_len": len(best_expr) if isinstance(best_expr, list) else None,
                        "chosen_agg_len": len(best_agg) if isinstance(best_agg, list) else None,
                        "chosen_expr_sample": [str(x) for x in (best_expr or [])[:6]],
                        "chosen_agg_sample": [str(x) for x in (best_agg or [])[:6]],
                    })
            except Exception:
                pass
        except Exception:
            pass
    
    conn.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "greedy_path": "-".join(path_bids),
        "steps": len(path_bids),
        "elapsed_ms": round(elapsed_ms, 1),
        "debug": debug_info,
    }

def _is_auction_complete_list(bids: List[str]) -> bool:
    if len(bids) >= 4 and all(b == "P" for b in bids[:4]):
        return True
    last_c = -1
    for i, b in enumerate(bids):
        if b not in ("P", "X", "XX") and b and b[0].isdigit():
            last_c = i
    return last_c >= 0 and len(bids) >= last_c + 4 and all(b == "P" for b in bids[-3:])
