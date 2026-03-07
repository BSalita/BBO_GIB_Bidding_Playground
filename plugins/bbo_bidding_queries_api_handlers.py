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
from collections import OrderedDict
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
    get_ev_for_auction_pre,
    compute_hand_features,
    compute_par_score,
    parse_pbn_deal,
    parse_contract_from_auction,
    build_distribution_sql_for_bt,
    build_distribution_sql_for_deals,
    add_suit_length_columns,
)

from bbo_bid_ranking_lib import (
    get_avg_ev_par_precomputed_nv_v,
    preload_precomputed_ev_par_stats,
    compute_vul_split_par_ev_at_bid,
    compute_ev_all_combos_for_matched_deals,
)

from bbo_bid_details_lib import (
    BidDetailsConfig,
    compute_bid_details_from_sample,
    compute_phase2a_auction_conditioned_posteriors,
)
from bbo_explanation_lib import (
    compute_common_sense_adjustments,
    compute_common_sense_hard_override,
    compute_forced_non_pass_policy,
    compute_partner_major_game_commit_adjustment,
    compute_pass_signoff_bonus,
    compute_post_game_slam_gate_adjustment,
    compute_rebiddable_major_game_bonus,
    compute_non_rebiddable_suit_rebid_penalty,
    compute_eeo_from_bid_details,
    compute_guardrail_penalty,
    extract_second_pass_opening_context,
    expected_from_hist,
    hand_controls,
    opponent_shown_natural_strains,
    render_counterfactual_why_not,
    render_recommendation_explanation,
)
from bbo_hand_eval_lib import estimate_partnership_tricks, pivot_bt_seat_stats, get_bt_dd_mean_tricks

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
    # Canonical board context helpers (dealer/vulnerability)
    normalize_dealer_strict,
    normalize_board_vulnerable,
    compute_seat_to_act,
    seat_vul_bucket,
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


_CACHE_MAX_BID_DETAILS = 2_000
_BID_DETAILS_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_BID_DETAILS_CACHE_LOCK = threading.Lock()

_CACHE_MAX_EXPLAIN_BID = 2_000
_EXPLAIN_BID_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_EXPLAIN_BID_CACHE_LOCK = threading.Lock()

# Cache for per-prefix "next BT rows" expansion.
# This is a major speed win for any logic that scores many candidate bids at the same auction prefix
# (e.g. AI Model greedy path): without this, each candidate triggers a DuckDB read_parquet query over
# the same small IN-list of bt_index values.
_CACHE_MAX_NEXT_BT_ROWS = 10_000
_NEXT_BT_ROWS_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_NEXT_BT_ROWS_CACHE_LOCK = threading.Lock()
_CACHE_MAX_GREEDY_EVAL = 100_000
_GREEDY_EVAL_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_GREEDY_EVAL_CACHE_LOCK = threading.Lock()


def _lru_get(cache: "OrderedDict[tuple[Any, ...], dict[str, Any]]", key: tuple[Any, ...], lock: threading.Lock) -> dict[str, Any] | None:
    try:
        with lock:
            v = cache.get(key)
            if v is None:
                return None
            # Move to end (most recently used)
            cache.move_to_end(key)
            return v
    except Exception:
        return None


def _lru_put(
    cache: "OrderedDict[tuple[Any, ...], dict[str, Any]]",
    key: tuple[Any, ...],
    val: dict[str, Any],
    max_items: int,
    lock: threading.Lock,
) -> None:
    try:
        with lock:
            cache[key] = val
            cache.move_to_end(key)
            while len(cache) > int(max_items):
                try:
                    cache.popitem(last=False)
                except Exception:
                    cache.clear()
                    break
    except Exception:
        # Fail-fast: caching is an optimization, never a correctness requirement.
        return


def _lookup_bid_feature_cache_row(state: Dict[str, Any], bt_index: Any) -> dict[str, Any] | None:
    """Lookup a bid-feature cache row by bt_index using startup-built sorted index arrays."""
    idx_obj = state.get("bid_feature_cache_index")
    df = state.get("bid_feature_cache_df")
    if idx_obj is None or df is None:
        return None
    try:
        bt_i = int(bt_index)
    except Exception:
        return None
    try:
        bt_sorted = idx_obj.get("bt_sorted")
        row_pos = idx_obj.get("row_pos")
        if bt_sorted is None or row_pos is None or len(bt_sorted) == 0:
            return None
        j = int(np.searchsorted(bt_sorted, bt_i))
        if j >= len(bt_sorted) or int(bt_sorted[j]) != bt_i:
            return None
        row_i = int(row_pos[j])
        return cast(dict[str, Any], df.row(row_i, named=True))
    except Exception:
        return None


def _get_next_bt_rows_for_parent(state: Dict[str, Any], parent_bt_index: int) -> dict[str, dict[str, Any]]:
    """
    Return a mapping: CANDIDATE_BID (upper) -> minimal BT row dict, for all children of `parent_bt_index`.

    This is intentionally cached because many callers (notably AI Model scoring) request multiple bids
    for the same prefix in a tight loop. Doing DuckDB read_parquet once per bid is extremely expensive.
    """
    key = (int(parent_bt_index),)
    cached = _lru_get(_NEXT_BT_ROWS_CACHE, key, _NEXT_BT_ROWS_CACHE_LOCK)
    if cached is not None:
        # cached is already a dict[str, dict[str, Any]]
        return cached  # type: ignore[return-value]

    next_indices = _get_next_bid_indices_for_parent(state, int(parent_bt_index))
    if not next_indices:
        out: dict[str, dict[str, Any]] = {}
        _lru_put(_NEXT_BT_ROWS_CACHE, key, out, _CACHE_MAX_NEXT_BT_ROWS, _NEXT_BT_ROWS_CACHE_LOCK)
        return out

    file_path = _bt_file_path_for_sql(state)
    in_list = ", ".join(str(int(x)) for x in next_indices if x is not None)

    # DuckDB is still used on cache miss (small IN-list query). We cache the expansion per parent.
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

    rows: dict[str, dict[str, Any]] = {}
    if not idx_df.is_empty():
        # Normalize keying by `candidate_bid` when present; fall back to `Auction` (opening rows).
        for r in idx_df.iter_rows(named=True):
            try:
                cand = r.get("candidate_bid")
                if cand is None or str(cand).strip() == "":
                    cand = r.get("Auction")
                b = str(cand).strip().upper() if cand is not None else ""
                if not b:
                    continue
                rows[b] = {
                    "bt_index": r.get("bt_index"),
                    "Auction": r.get("Auction"),
                    "candidate_bid": b,
                    "is_completed_auction": r.get("is_completed_auction"),
                    "Expr": r.get("Expr"),
                }
            except Exception:
                continue

    _lru_put(_NEXT_BT_ROWS_CACHE, key, rows, _CACHE_MAX_NEXT_BT_ROWS, _NEXT_BT_ROWS_CACHE_LOCK)
    return rows


def _resolve_deal_row_by_deal_index(state: Dict[str, Any], deal_index: int) -> dict[str, Any] | None:
    """Resolve a deal row by its user-facing `index` column value (not row position)."""
    deal_df = state.get("deal_df")
    if not isinstance(deal_df, pl.DataFrame) or deal_df.is_empty():
        return None
    if "index" not in deal_df.columns:
        return None

    idx = int(deal_index)
    # Fast path: monotonic index → row position via binary search + take rows.
    try:
        idx_arr = state.get("deal_index_arr")
        is_mono = bool(state.get("deal_index_monotonic", False))
        if is_mono and idx_arr is not None:
            pos = int(np.searchsorted(idx_arr, idx))
            if 0 <= pos < len(idx_arr) and int(idx_arr[pos]) == idx:
                one = _take_rows_by_index(deal_df, [pos])
                if isinstance(one, pl.DataFrame) and one.height == 1:
                    return dict(one.row(0, named=True))
    except Exception:
        pass

    # Fallback: filter scan (slow on huge deal_df; kept for correctness).
    try:
        one = deal_df.filter(pl.col("index") == idx).head(1)
        if one.height == 1:
            return dict(one.row(0, named=True))
    except Exception:
        return None
    return None


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


def _compute_deal_mask_with_base_mask(
    state: dict[str, Any],
    base_mask: "pl.Series | None",
    bt_row: dict[str, Any],
    seat: int,
) -> "pl.Series | None":
    """Compute deal mask by intersecting base_mask with new seat's criteria.

    This mirrors `_compute_deal_count_with_base_mask`, but returns the boolean mask so callers
    can sample the actual deals that comprise the "Deals" count shown in the UI.
    """
    deal_df = state.get("deal_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    if deal_df is None:
        return None

    seat_i = max(1, min(4, int(seat)))
    criteria_list = bt_row.get(f"Agg_Expr_Seat_{seat_i}") or []

    # If no new criteria, return the base mask (or all deals if no base mask).
    if not criteria_list:
        if base_mask is not None:
            return base_mask
        try:
            return pl.Series([True] * int(deal_df.height))
        except Exception:
            return None

    seat_criteria_for_seat = deal_criteria_by_seat_dfs.get(seat_i, {})
    if not seat_criteria_for_seat:
        return base_mask

    # Find valid criteria
    sample_criteria_df = None
    for dealer in DIRECTIONS:
        sample_criteria_df = seat_criteria_for_seat.get(dealer)
        if sample_criteria_df is not None and not sample_criteria_df.is_empty():
            break
    if sample_criteria_df is None:
        return base_mask

    available_cols = set(sample_criteria_df.columns)
    valid_criteria = [c for c in criteria_list if c in available_cols]
    if not valid_criteria:
        return base_mask

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
        return None
    return (base_mask & new_seat_mask) if base_mask is not None else new_seat_mask


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

        # If the precomputed Par_Indexes table doesn't contain this deal, do NOT return
        # "no matches" yet; fall through to the contract-based scan below.
        if par_indices:
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


def handle_deal_criteria_pass_batch(
    state: Dict[str, Any],
    deal_row_idx: int,
    dealer: str,
    checks: List[Dict[str, Any]],
    deal_row_dict: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fast pass/fail only version of /deal-criteria-eval-batch.

    This is optimized for callers that only need a boolean:
    - No annotation of failures
    - Short-circuits on first failure/untracked
    - Uses row-dict lookups for bitmap criteria (faster than column indexing)

    When deal_row_dict is provided (on-the-fly PBN/CSV deals not in the BBO database),
    criteria are evaluated dynamically against a 1-row Polars DataFrame built from
    deal_row_dict, using the stored pythonized expressions from state. This bypasses
    both the deal_df row lookup and the pre-computed bitmap index.
    """
    deal_df = state.get("deal_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    if deal_df is None and deal_row_dict is None:
        raise ValueError("deal_df not loaded")

    # Resolve deal_row: prefer supplied dict (on-the-fly), else fetch from DB by index.
    if deal_row_dict is not None:
        deal_row: Dict[str, Any] = deal_row_dict
    else:
        if deal_df is None:
            raise ValueError("deal_df not loaded")
        try:
            deal_row = deal_df.row(int(deal_row_idx), named=True)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid deal_row_idx {deal_row_idx}: {e}")

    dealer_n = str(dealer or "N").upper()

    # Determine if we can use the pre-computed bitmap (DB deals with valid row index).
    _use_bitmap = deal_row_dict is None and deal_row_idx >= 0

    # Cache per (seat,dealer) row dict for this deal to avoid repeated Polars overhead.
    row_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

    def _criteria_row_for(seat: int) -> Dict[str, Any] | None:
        key = (int(seat), dealer_n)
        if key in row_cache:
            return row_cache[key] or None
        df = (deal_criteria_by_seat_dfs.get(int(seat), {}) or {}).get(dealer_n)
        if df is None:
            row_cache[key] = {}
            return None
        try:
            r = df.row(int(deal_row_idx), named=True)
        except Exception:
            r = {}
        row_cache[key] = r
        return r

    # Dynamic expression evaluator for on-the-fly deals (no bitmap available).
    # Uses pythonized Polars expressions stored in state during server init.
    _one_row_df_cache: Dict[str, Any] = {}  # mutable dict used as a namespace for caching

    def _eval_criterion_dynamic(crit_s: str, seat: int) -> bool | None:
        """Evaluate a criterion expression against deal_row_dict dynamically.

        Returns True/False, or None if the expression is not found in the criteria map.
        """
        criteria_exprs = state.get("criteria_pythonized_exprs_by_direction") or {}
        direction = seat_to_direction(dealer_n, seat)
        dir_exprs = criteria_exprs.get(direction) or {}
        pythonized_expr = dir_exprs.get(crit_s)
        if pythonized_expr is None:
            return None

        # Build (and cache) a 1-row Polars DataFrame for this deal.
        if "df" not in _one_row_df_cache:
            try:
                import polars as pl_local
                _one_row_df_cache["df"] = pl_local.DataFrame([deal_row_dict])
                _one_row_df_cache["pl"] = pl_local
            except Exception:
                return None
        one_row_df = _one_row_df_cache["df"]
        pl_local = _one_row_df_cache["pl"]

        try:
            import re as _re
            eval_env = {col: pl_local.col(col) for col in one_row_df.columns}
            s = _re.sub(r"\bTrue\b", "pl.lit(True)", pythonized_expr)
            s = _re.sub(r"\bFalse\b", "pl.lit(False)", s)
            polars_expr = eval(s, {"pl": pl_local}, eval_env)
            return bool(one_row_df.select(polars_expr)[0, 0])
        except Exception:
            return None

    results: List[Dict[str, Any]] = []
    for chk in (checks or []):
        seat = int(chk.get("seat") or 0)
        criteria_list = chk.get("criteria") or []
        if seat < 1 or seat > 4:
            results.append({"seat": seat, "passes": False})
            continue

        crit_row = _criteria_row_for(seat) if _use_bitmap else None
        passes = True

        for crit in (criteria_list or []):
            if crit is None:
                continue
            crit_s = str(crit)
            crit_u = crit_s.strip().upper()

            # Convention/flow-control markers are metadata, not hand-shape predicates.
            # They should not block deal-level pass/fail in batch evaluation.
            if crit_u.startswith("FORCING_TO_") or crit_u in ("FORCING_ONE_ROUND",):
                continue

            # Dynamic SL first.
            sl_result = evaluate_sl_criterion(crit_s, dealer_n, seat, deal_row, fail_on_missing=False)
            if sl_result is True:
                continue
            if sl_result is False:
                passes = False
                break
            # SL criterion that can't be evaluated => untracked => fail (matches UI "passes = no failed and no untracked").
            if parse_sl_comparison_relative(crit_s) is not None or parse_sl_comparison_numeric(crit_s) is not None:
                passes = False
                break

            if _use_bitmap:
                # Pre-computed bitmap path (DB deals): fast row-dict lookup.
                if crit_row is None:
                    passes = False
                    break
                if crit_s not in crit_row:
                    passes = False
                    break
                try:
                    if not bool(crit_row.get(crit_s)):
                        passes = False
                        break
                except Exception:
                    passes = False
                    break
            else:
                # Dynamic evaluation path (on-the-fly PBN/CSV deals).
                dyn_result = _eval_criterion_dynamic(crit_s, seat)
                if dyn_result is None:
                    # Expression not found in criteria map — treat as untracked → fail.
                    passes = False
                    break
                if not dyn_result:
                    passes = False
                    break

        results.append(
            {
                "seat": seat,
                "seat_dir": seat_to_direction(dealer_n, seat),
                "passes": bool(passes),
            }
        )

    return {"deal_row_idx": int(deal_row_idx), "dealer": dealer_n, "results": results}


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

    # Add _row_idx BEFORE filtering so each match carries its original row position.
    # This is required for the AI model batch pipeline which needs deal_row_idx.
    deal_df_indexed = deal_df.with_row_index("_row_idx")
    matching = deal_df_indexed.filter(match_criteria)
    
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


def _compute_actual_auction_stats(
    deal_df: "pl.DataFrame",
    row_mask: "pl.Series",
) -> Dict[str, Any]:
    """Compute hand-stat means and DD trick means from *all* deals in ``row_mask``.

    Returns a dict with ``hand_stats`` (4 direction rows) and ``dd_means``
    (4 declarer rows × 5 strains), suitable for direct JSON serialisation.
    Runs over the full filtered set (not just the sample) for accuracy.
    """
    import numpy as np

    n_deals = int(row_mask.sum())
    if n_deals == 0:
        return {}

    deal_indices = np.where(row_mask.to_numpy())[0]

    def _mean(arr: np.ndarray, idx: np.ndarray):
        vals = arr[idx]
        if np.issubdtype(vals.dtype, np.floating):
            valid = vals[~np.isnan(vals)]
        else:
            valid = vals.astype(np.float64)
        return round(float(np.mean(valid)), 1) if len(valid) > 0 else None

    # -- Hand stats: 4 rows (direction) × stat columns --
    # Column naming: HCP_{dir}, Total_Points_{dir} but SL_{dir}_{suit}
    _DIRS = ["N", "E", "S", "W"]
    # (column_template, friendly_name, is_std)
    # {d} = direction placeholder
    _HS_SPECS: List[tuple] = [
        ("HCP_{d}", "HCP", False),
        ("HCP_{d}", "HCP_std", True),
        ("SL_{d}_C", "SL_C", False),
        ("SL_{d}_D", "SL_D", False),
        ("SL_{d}_H", "SL_H", False),
        ("SL_{d}_S", "SL_S", False),
        ("Total_Points_{d}", "Total_Points", False),
        ("Total_Points_{d}", "Total_Points_std", True),
    ]
    hand_rows: List[Dict[str, Any]] = []
    for direction in _DIRS:
        hs: Dict[str, Any] = {"Direction": direction}
        for col_tmpl, friendly, is_std in _HS_SPECS:
            col = col_tmpl.format(d=direction)
            if col in deal_df.columns:
                arr = deal_df[col].to_numpy(zero_copy_only=False)
                if is_std:
                    hs[friendly] = round(float(np.std(arr[deal_indices].astype(np.float64), ddof=0)), 1)
                else:
                    hs[friendly] = _mean(arr, deal_indices)
            else:
                hs[friendly] = None
        hand_rows.append(hs)

    # -- DD means: 4 rows (declarer) × 5 strain cols --
    _DECLARERS = ["N", "E", "S", "W"]
    _STRAINS = ["C", "D", "H", "S", "N"]
    dd_rows: List[Dict[str, Any]] = []
    for decl in _DECLARERS:
        row_dict: Dict[str, Any] = {"Declarer": decl}
        for strain in _STRAINS:
            col = f"DD_{decl}_{strain}"
            if col in deal_df.columns:
                arr = deal_df[col].to_numpy(zero_copy_only=False)
                row_dict[strain] = _mean(arr, deal_indices)
            else:
                row_dict[strain] = None
        dd_rows.append(row_dict)

    # Count how many hand-stat columns were found (for diagnostics)
    _hs_found = sum(1 for d in _DIRS for tmpl, _, _ in _HS_SPECS if tmpl.format(d=d) in deal_df.columns)
    _dd_found = sum(1 for d in _DECLARERS for s in _STRAINS if f"DD_{d}_{s}" in deal_df.columns)
    print(f"[actual-auction-stats] {n_deals} deals, "
          f"{_hs_found}/{len(_HS_SPECS)*4} HS cols, {_dd_found}/20 DD cols, "
          f"DD_S_H={dd_rows[2].get('H')}, DD_N_H={dd_rows[0].get('H')}")
    return {
        "hand_stats": hand_rows,
        "dd_means": dd_rows,
        "deal_count": n_deals,
    }


def _compute_hand_range_stats(
    deal_df: "pl.DataFrame",
    row_mask: "pl.Series",
) -> List[Dict[str, Any]]:
    """Compute observed hand-stat ranges (min/max + p10/p90) by direction."""
    import numpy as np

    n_deals = int(row_mask.sum())
    if n_deals == 0:
        return []

    deal_indices = np.where(row_mask.to_numpy())[0]

    def _series_vals(col: str) -> np.ndarray:
        if col not in deal_df.columns:
            return np.array([], dtype=np.float64)
        arr = deal_df[col].to_numpy(zero_copy_only=False)[deal_indices]
        vals = np.asarray(arr, dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        return vals

    def _q(vals: np.ndarray, pct: float) -> float | None:
        if len(vals) == 0:
            return None
        return round(float(np.percentile(vals, pct)), 1)

    def _mn(vals: np.ndarray) -> float | None:
        if len(vals) == 0:
            return None
        return round(float(np.min(vals)), 1)

    def _mx(vals: np.ndarray) -> float | None:
        if len(vals) == 0:
            return None
        return round(float(np.max(vals)), 1)

    rows: List[Dict[str, Any]] = []
    for d in ["N", "E", "S", "W"]:
        r: Dict[str, Any] = {"Direction": d, "Deals": n_deals}
        specs: List[tuple[str, str]] = [
            ("HCP", f"HCP_{d}"),
            ("Total_Points", f"Total_Points_{d}"),
            ("SL_C", f"SL_{d}_C"),
            ("SL_D", f"SL_{d}_D"),
            ("SL_H", f"SL_{d}_H"),
            ("SL_S", f"SL_{d}_S"),
        ]
        for metric, col in specs:
            vals = _series_vals(col)
            r[f"{metric}_min"] = _mn(vals)
            r[f"{metric}_max"] = _mx(vals)
            r[f"{metric}_p10"] = _q(vals, 10.0)
            r[f"{metric}_p90"] = _q(vals, 90.0)
        rows.append(r)
    return rows


def handle_sample_deals_by_auction_pattern(
    state: Dict[str, Any],
    pattern: str,
    sample_size: int,
    seed: Optional[int],
    include_stats: bool = False,
) -> Dict[str, Any]:
    """Return a small sample of deals whose *actual auction* matches regex `pattern`.

    This intentionally does NOT run any BT / Rules logic. It is used by Streamlit's Auction Builder
    to power "Show Matching Deals" quickly without the heavy /bidding-arena path.

    When ``include_stats`` is True, also computes aggregate hand-stat means and
    DD trick means from ALL filtered deals (not just the sample) and returns
    them in the ``"actual_auction_stats"`` key.
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
        mask = pl.col(auction_col).cast(pl.Utf8).str.contains(regex)
        filtered = df.filter(mask)
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

    result: Dict[str, Any] = {
        "pattern": pattern,
        "deals": out_rows,
        "total_count": total_count,
    }

    # Compute aggregate stats from ALL filtered deals (not just sample)
    if include_stats:
        # Build a boolean mask over the ORIGINAL deal_df (not the filtered subset)
        full_mask = df[auction_col].cast(pl.Utf8).str.contains(regex)
        result["actual_auction_stats"] = _compute_actual_auction_stats(df, full_mask)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    result["elapsed_ms"] = round(elapsed_ms, 1)
    print(f"[sample-deals-by-auction-pattern] {format_elapsed(elapsed_ms)} ({len(out_rows)}/{total_count} deals"
          f"{', +stats' if include_stats else ''})")
    return result


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


def handle_bt_dd_mean_tricks(
    state: Dict[str, Any],
    auctions: List[str],
    dealer: str,
) -> Dict[str, Any]:
    """Get BT node mean DD tricks for each auction's terminal contract.

    For each auction: resolve to BT path, parse the contract (declarer + strain),
    then look up the mean DD tricks from bt_stats_df for the terminal bt_index.
    """
    t0 = time.perf_counter()
    bt_stats_df = state.get("bt_stats_df")
    results: Dict[str, Any] = {}

    for auction in auctions:
        if not auction or not auction.strip():
            continue
        auc_key = str(auction).strip()
        toks = [t.strip().upper() for t in auc_key.split("-") if t.strip()]
        if not toks or all(t == "P" for t in toks):
            results[auc_key] = None
            continue

        # Resolve path to get terminal bt_index
        try:
            path_resp = _handle_resolve_auction_path_fallback(state, auc_key, time.perf_counter())
            path = path_resp.get("path") or []
        except Exception:
            results[auc_key] = None
            continue

        # Find terminal bt_index (last step with a bt_index)
        bt_index = None
        for step in reversed(path):
            if step.get("bt_index") is not None:
                bt_index = int(step["bt_index"])
                break

        if bt_index is None or bt_stats_df is None:
            results[auc_key] = None
            continue

        # Parse contract to get declarer + strain
        contract = parse_contract_from_auction(auc_key)
        if not contract:
            results[auc_key] = None
            continue
        _, strain, _ = contract
        declarer = get_declarer_for_auction(auc_key, dealer)
        if not declarer:
            results[auc_key] = None
            continue

        val = get_bt_dd_mean_tricks(bt_stats_df, bt_index, declarer, strain, dealer)
        results[auc_key] = val

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"results": results, "elapsed_ms": round(elapsed_ms, 1)}


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
# Handler: /bt-seat-stats-pivot  (precomputed v2 stats, batch)
# ---------------------------------------------------------------------------


def handle_bt_seat_stats_pivot(
    state: Dict[str, Any],
    bt_indices: List[int],
    dealer: str | None = None,
) -> Dict[str, Any]:
    """Return pivoted precomputed seat stats for a batch of bt_indices.

    Two stat layers per bt_index:
      1. **Computed hand stats** — from precomputed GPU EV v2 (seat-relative, correct)
      2. **Computed DD means**  — from precomputed GPU EV v2 (diluted, kept as-is)

    Actual-auction stats (reality-check baseline) are now served by
    ``/sample-deals-by-auction-pattern?include_stats=true`` instead.
    """
    from bbo_hand_eval_lib import pivot_bt_seat_stats

    t0 = time.perf_counter()
    bt_stats_df = state.get("bt_stats_df")
    if bt_stats_df is None:
        return {"stats": {}, "error": "bt_stats_df not loaded"}

    # Computed stats from precomputed data (hand stats correct, DD diluted)
    stats = pivot_bt_seat_stats(bt_stats_df, bt_indices, dealer=dealer)

    # Convert int keys to str for JSON serialisation
    stats_str = {str(k): v for k, v in stats.items()}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "stats": stats_str,
        "requested": len(bt_indices),
        "found": len(stats_str),
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Handler: /sample-deals-for-bt-indices
# ---------------------------------------------------------------------------


def handle_sample_deals_for_bt_indices(
    state: Dict[str, Any],
    bt_indices: List[int],
    sample_size: int = 100,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Return per-deal hand stats and DD tricks for deals matching given bt_indices.

    Uses the precomputed deal_to_bt_index_df (GPU-verified) to find deals,
    then extracts hand-stat and DD columns from deal_df.

    Returns ``deals_by_bt`` keyed by bt_index (str), each containing:
      - ``deals``: list of dicts with per-deal columns
      - ``total_count``: total matching deals (before sampling)
    """
    t0 = time.perf_counter()

    deal_df = state.get("deal_df")
    deal_to_bt_df = state.get("deal_to_bt_index_df")

    if not isinstance(deal_df, pl.DataFrame) or deal_df.is_empty():
        raise ValueError("deal_df not loaded")
    if not isinstance(deal_to_bt_df, pl.DataFrame) or deal_to_bt_df.is_empty():
        raise ValueError("deal_to_bt_index_df not loaded")

    # Columns to return per deal
    base_cols = ["index", "Dealer", "Vul"]
    hand_str_cols = [f"Hand_{d}" for d in "NESW"]
    hs_cols: List[str] = []
    for d in "NESW":
        hs_cols.extend([f"HCP_{d}", f"Total_Points_{d}"])
        for suit in "SHDC":
            hs_cols.append(f"SL_{suit}_{d}")
    dd_cols: List[str] = []
    for decl in "NESW":
        for strain in ["C", "D", "H", "S", "N"]:
            dd_cols.append(f"DD_{decl}_{strain}")

    all_cols = base_cols + hand_str_cols + hs_cols + dd_cols
    all_cols = [c for c in all_cols if c in deal_df.columns]

    effective_seed = _effective_seed(seed)
    rng = np.random.RandomState(effective_seed)

    n_deals = deal_df.height
    results: Dict[str, Any] = {}

    for bt_idx in bt_indices:
        # Find deal_idx values where bt_idx is in Matched_BT_Indices
        mask = deal_to_bt_df["Matched_BT_Indices"].list.contains(bt_idx)
        matching_deal_idxs = deal_to_bt_df.filter(mask)["deal_idx"].to_numpy()

        total_count = len(matching_deal_idxs)
        if total_count == 0:
            results[str(bt_idx)] = {"deals": [], "total_count": 0}
            continue

        # Clamp to valid range
        valid_idxs = matching_deal_idxs[matching_deal_idxs < n_deals]
        if len(valid_idxs) == 0:
            results[str(bt_idx)] = {"deals": [], "total_count": total_count}
            continue

        # Sample
        if len(valid_idxs) > sample_size:
            sampled_idxs = rng.choice(valid_idxs, size=sample_size, replace=False)
        else:
            sampled_idxs = valid_idxs

        # Sort for sequential access; use NumPy-style row indexing (OK for small N)
        sampled_sorted = sorted(int(x) for x in sampled_idxs)
        deal_rows_df = deal_df.select(all_cols)[sampled_sorted]

        # Compute suit lengths from Hand_ strings for any missing SL columns
        for d in "NESW":
            hand_col = f"Hand_{d}"
            if hand_col not in deal_rows_df.columns:
                continue
            for suit_idx, suit in enumerate("SHDC"):
                sl_col = f"SL_{suit}_{d}"
                if sl_col not in deal_rows_df.columns:
                    deal_rows_df = deal_rows_df.with_columns(
                        pl.col(hand_col)
                        .str.split(".")
                        .list.get(suit_idx)
                        .str.len_chars()
                        .alias(sl_col)
                    )

        results[str(bt_idx)] = {
            "deals": deal_rows_df.to_dicts(),
            "total_count": total_count,
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[sample-deals-for-bt-indices] {format_elapsed(elapsed_ms)} "
          f"({len(bt_indices)} bt_indices, {sum(len(v['deals']) for v in results.values())} deals)")
    return {
        "deals_by_bt": results,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Handler: /criteria-stats-by-bt-indices
# ---------------------------------------------------------------------------


def handle_criteria_stats_by_bt_indices(
    state: Dict[str, Any],
    bt_indices: List[int],
    dealer: str | None = None,
) -> Dict[str, Any]:
    """Return aggregate hand/DD stats for deals matched by BT criteria.

    This is a criteria-only aggregation path. For each ``bt_index`` we use
    ``deal_to_bt_index_df.Matched_BT_Indices`` to find matching deals, then
    compute the same aggregate output shape as ``actual_auction_stats``.
    """
    t0 = time.perf_counter()

    deal_df = state.get("deal_df")
    deal_to_bt_df = state.get("deal_to_bt_index_df")
    bt_stats_df = state.get("bt_stats_df")

    if not isinstance(deal_df, pl.DataFrame) or deal_df.is_empty():
        raise ValueError("deal_df not loaded")
    if not isinstance(deal_to_bt_df, pl.DataFrame) or deal_to_bt_df.is_empty():
        raise ValueError("deal_to_bt_index_df not loaded")

    if "Matched_BT_Indices" not in deal_to_bt_df.columns:
        raise ValueError("Required column 'Matched_BT_Indices' missing from deal_to_bt_index_df")

    # Phase 0 prerequisite: dealer is required for seat->compass mapping when
    # serving precomputed seat-relative stats.
    dealer_n = normalize_dealer_strict(dealer) if dealer is not None else None
    if dealer_n is None:
        raise ValueError("dealer is required for /criteria-stats-by-bt-indices precomputed mapping")

    # Fast path: use precomputed bt_stats_df (seat-relative) with explicit
    # seat-to-compass translation via pivot_bt_seat_stats.
    stats_by_bt: Dict[str, Any] = {}
    if isinstance(bt_stats_df, pl.DataFrame) and not bt_stats_df.is_empty():
        precomputed = pivot_bt_seat_stats(
            bt_stats_df=bt_stats_df,
            bt_indices=[int(x) for x in bt_indices],
            dealer=dealer_n,
        )
        for bt_idx in bt_indices:
            bt_i = int(bt_idx)
            row = precomputed.get(bt_i)
            if row is None:
                stats_by_bt[str(bt_i)] = {}
                continue
            # Keep response shape stable; hand_ranges now comes from /bt-hand-profile.
            stats_by_bt[str(bt_i)] = {
                "hand_stats": row.get("hand_stats") or [],
                "dd_means": row.get("dd_means") or [],
                "hand_ranges": [],
                "deal_count": None,
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[criteria-stats-by-bt-indices] {format_elapsed(elapsed_ms)} "
            f"({len(bt_indices)} bt_indices, precomputed={sum(1 for v in stats_by_bt.values() if v)})"
        )
        return {
            "stats_by_bt": stats_by_bt,
            "requested": len(bt_indices),
            "elapsed_ms": round(elapsed_ms, 1),
        }

    # Hard fail rather than silently regressing to heavy full-scan path.
    raise ValueError("bt_stats_df not loaded; cannot serve precomputed criteria stats")


def handle_bt_trick_quality(
    state: Dict[str, Any],
    bt_indices: List[int],
    seat: int | None = None,
    strain: int | None = None,
    level: int | None = None,
    vul: int | None = None,
) -> Dict[str, Any]:
    """Return precomputed trick quality rows for requested bt_indices."""
    t0 = time.perf_counter()
    df = state.get("bt_trick_quality_df")
    if not isinstance(df, pl.DataFrame) or df.is_empty():
        raise ValueError("bt_trick_quality_df not loaded")

    idxs = [int(x) for x in bt_indices]
    if len(idxs) == 0:
        return {"rows": [], "requested": 0, "elapsed_ms": 0.0}
    if len(idxs) > 100:
        raise ValueError("bt_indices batch limit exceeded (max 100)")

    q = df.filter(pl.col("bt_index").is_in(pl.Series("bt_index", idxs, dtype=pl.UInt32)))
    if seat is not None:
        if int(seat) not in (1, 2, 3, 4):
            raise ValueError("seat must be in 1..4")
        q = q.filter(pl.col("seat") == int(seat))
    if strain is not None:
        if int(strain) not in (0, 1, 2, 3, 4):
            raise ValueError("strain must be in 0..4")
        q = q.filter(pl.col("strain") == int(strain))
    if level is not None:
        if int(level) not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError("level must be in 1..7")
        q = q.filter(pl.col("level") == int(level))
    if vul is not None:
        if int(vul) not in (0, 1):
            raise ValueError("vul must be 0 (NV) or 1 (V)")
        q = q.filter(pl.col("vul") == int(vul))

    rows = q.sort(["bt_index", "seat", "strain", "level"]).to_dicts()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "rows": rows,
        "requested": len(idxs),
        "returned": len(rows),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_bt_hand_profile(
    state: Dict[str, Any],
    bt_indices: List[int],
    seat: int | None = None,
) -> Dict[str, Any]:
    """Return precomputed hand profile rows for requested bt_indices."""
    t0 = time.perf_counter()
    df = state.get("bt_hand_profile_df")
    if not isinstance(df, pl.DataFrame) or df.is_empty():
        raise ValueError("bt_hand_profile_df not loaded")

    idxs = [int(x) for x in bt_indices]
    if len(idxs) == 0:
        return {"rows": [], "requested": 0, "elapsed_ms": 0.0}
    if len(idxs) > 100:
        raise ValueError("bt_indices batch limit exceeded (max 100)")

    q = df.filter(pl.col("bt_index").is_in(pl.Series("bt_index", idxs, dtype=pl.UInt32)))
    if seat is not None:
        if int(seat) not in (1, 2, 3, 4):
            raise ValueError("seat must be in 1..4")
        q = q.filter(pl.col("seat") == int(seat))

    rows = q.sort(["bt_index", "seat"]).to_dicts()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "rows": rows,
        "requested": len(idxs),
        "returned": len(rows),
        "elapsed_ms": round(elapsed_ms, 1),
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
    deal_meta_by_idx: Dict[int, Dict[str, Any]] = {}
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
            pbn_deals, deal_vuls, deal_meta_by_idx = parse_file_with_endplay_fn(file_content, is_lin=is_lin)
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
            pbn_deals, deal_vuls, deal_meta_by_idx = parse_file_with_endplay_fn(file_content, is_lin=is_lin)
            input_type = "LIN file" if is_lin else "PBN file"
            input_source = file_path
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to read/parse file: {e}")
    
    elif 'md|' in pbn_input and '|' in pbn_input:
        pbn_deals, deal_vuls, deal_meta_by_idx = parse_file_with_endplay_fn(pbn_input, is_lin=True)
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
        # For PBN/LIN parsing, the parser-provided semantics are authoritative:
        # Tricks = total tricks taken; Result = Tricks - (level + 6).
        parsed_meta = deal_meta_by_idx.get(deal_idx) or {}
        if parsed_meta:
            deal.update({k: v for k, v in parsed_meta.items() if v is not None})
        
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
                        # Do not overwrite PBN/LIN parser semantics for Result/Tricks.
                        if col in ("Result", "Tricks") and col in parsed_meta:
                            continue
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
    *,
    dealer_for_criteria: str | None = None,
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

    dealer_crit = str(dealer_for_criteria or dealer).upper()
    if dealer_crit not in DIRECTIONS:
        dealer_crit = str(dealer).upper() if str(dealer).upper() in DIRECTIONS else "N"
    
    for seat, criteria_list in criteria_by_seat.items():
        seat_criteria_df = deal_criteria_by_seat_dfs.get(seat, {}).get(dealer_crit)
        
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
                seat_dir = _seat_dir_for_dealer(dealer_crit, seat)
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


def _rotate_dealer_dir(dealer: str, n_seats: int) -> str:
    """Rotate a dealer direction by `n_seats` (N->E->S->W)."""
    try:
        dirs = ["N", "E", "S", "W"]
        d = str(dealer or "N").strip().upper()
        i = dirs.index(d) if d in dirs else 0
        k = int(n_seats or 0) % 4
        return dirs[(i + k) % 4]
    except Exception:
        return str(dealer or "N").strip().upper()


def _build_criteria_mask_for_all_seats(
    deal_df: pl.DataFrame,
    bt_row: Dict[str, Any],
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, pl.DataFrame]],
    *,
    dealer_rotation: int = 0,
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
    
    rot = int(dealer_rotation or 0) % 4
    for dealer in DIRECTIONS:
        # Leading-pass support:
        # If the auction prefix has k leading passes, the "opener" is seat (k+1) relative to
        # the real dealer. BT criteria are seat-1 canonical, so we evaluate criteria under a
        # dealer that is rotated by k seats, while still restricting rows to the real dealer.
        crit_dealer = _rotate_dealer_dir(dealer, rot) if rot else dealer
        dealer_mask, invalid_criteria = _build_criteria_mask_for_dealer(
            deal_df,
            dealer,
            criteria_by_seat,
            deal_criteria_by_seat_dfs,
            dealer_for_criteria=crit_dealer,
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
    dealer: Optional[str] = None,
    vulnerable: Optional[str] = None,
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
        dealer=dealer,
        vulnerable=vulnerable,
    )
    
    # Add extra metadata for the response if missing
    if "next_bids" not in resp and "bid_rankings" in resp:
        resp["next_bids"] = resp.pop("bid_rankings")
        
    return resp


def handle_deals_matching_next_bid_criteria(
    state: Dict[str, Any],
    *,
    auction_prefix: str,
    bid: str,
    max_rows: int = 5000,
    seed: int = 0,
    columns: list[str] | None = None,
) -> Dict[str, Any]:
    """Return deals that comprise the BT criteria-based "Deals" count for a next bid.

    This mirrors the semantics used by `/list-next-bids` when `include_deal_counts=True`:
    - Build a base mask from the parent BT node's cumulative criteria (all seats).
    - AND the candidate next bid's acting-seat criteria mask.

    IMPORTANT:
    - This is NOT "deals whose actual auction includes this bid".
    - This is "deals where BT criteria say this bid is valid at this prefix".
    """
    t0 = time.perf_counter()
    deal_df = state.get("deal_df")
    bt_seat1_file = state.get("bt_seat1_file")
    if not isinstance(deal_df, pl.DataFrame) or deal_df.is_empty():
        raise ValueError("deal_df not loaded")
    if not bt_seat1_file:
        raise ValueError("bt_seat1_file not loaded")

    prefix = normalize_auction_input(auction_prefix or "").rstrip("-")
    bid_norm = str(bid or "").strip().upper()
    if not bid_norm:
        raise ValueError("bid is required")

    # Only BT-backed bids here (Pass/doubles have no BT row).
    if bid_norm in ("P", "PASS", "X", "D", "DBL", "DOUBLE", "XX", "R", "RDBL", "REDOUBLE"):
        return {
            "auction_prefix": prefix,
            "bid": bid_norm,
            "total_count": 0,
            "rows": [],
            "message": f"Bid {bid_norm!r} is not BT-backed for criteria deal sets.",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # Parent bt_index
    parent_bt_index = -1 if not prefix else _resolve_bt_index_by_traversal(state, prefix)
    if parent_bt_index is None:
        return {
            "auction_prefix": prefix,
            "bid": bid_norm,
            "total_count": 0,
            "rows": [],
            "error": f"Auction prefix {prefix!r} not found in BT (traversal failed)",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # Next seat (BT canonical, assumes prefix has no leading passes)
    next_seat = 1 if not prefix else (len(prefix.split("-")) % 4) + 1

    # Base mask from parent cumulative criteria (for accurate "Deals" semantics).
    base_mask: pl.Series | None = None
    if parent_bt_index >= 0:
        agg_data = _load_agg_expr_for_bt_indices([int(parent_bt_index)], bt_seat1_file)
        if int(parent_bt_index) in agg_data:
            parent_row: Dict[str, Any] = {"bt_index": int(parent_bt_index)}
            parent_row.update(agg_data[int(parent_bt_index)])
            parent_row = _apply_all_rules_to_bt_row(parent_row, state)
            base_mask = _compute_cumulative_deal_mask(state, parent_row, 4)

    # Resolve candidate bt_row for this next bid from parent's children.
    if parent_bt_index < 0:
        return {
            "auction_prefix": prefix,
            "bid": bid_norm,
            "total_count": 0,
            "rows": [],
            "error": "Opening prefix not supported for criteria-deals endpoint (use list-next-bids + bid-details instead).",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    rows_by_bid = _get_next_bt_rows_for_parent(state, int(parent_bt_index))
    if not rows_by_bid:
        return {
            "auction_prefix": prefix,
            "bid": bid_norm,
            "total_count": 0,
            "rows": [],
            "error": "BT lookup returned no candidate rows for this parent (cached expansion empty)",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    bt_row = rows_by_bid.get(bid_norm)
    if bt_row is None:
        return {
            "auction_prefix": prefix,
            "bid": bid_norm,
            "total_count": 0,
            "rows": [],
            "error": f"Bid {bid_norm!r} not found among next-bid candidates for this auction prefix",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # Ensure Agg_Expr columns exist (Expr is acting-seat per-step criteria).
    row_dict = dict(bt_row)
    for s in range(1, 5):
        col_s = f"Agg_Expr_Seat_{s}"
        if col_s not in row_dict or row_dict[col_s] is None:
            row_dict[col_s] = row_dict.get("Expr") if s == int(next_seat) else []
    row_with_rules = _apply_all_rules_to_bt_row(row_dict, state)

    final_mask = _compute_deal_mask_with_base_mask(state, base_mask, row_with_rules, int(next_seat))
    if final_mask is None or not final_mask.any():
        return {
            "auction_prefix": prefix,
            "bid": bid_norm,
            "total_count": 0,
            "rows": [],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    matched_df = deal_df.filter(final_mask)
    total_count = int(matched_df.height)

    # Column selection (keep payload bounded)
    default_cols = [
        "_row_idx",
        "index",
        "Dealer",
        "Vul",
        "Hand_N",
        "Hand_E",
        "Hand_S",
        "Hand_W",
        "ParScore",
        "Contract",
        "Result",
        "Score",
        "bid",
    ]
    cols = columns or default_cols
    # Ensure _row_idx exists
    if "_row_idx" not in matched_df.columns:
        matched_df = matched_df.with_row_index("_row_idx")
    cols_in = [c for c in cols if c in matched_df.columns]
    if cols_in:
        matched_df = matched_df.select(cols_in)

    max_rows_i = max(1, int(max_rows or 0))
    if max_rows_i and matched_df.height > max_rows_i:
        try:
            matched_df = matched_df.sample(n=max_rows_i, seed=int(seed or 0))
        except Exception:
            matched_df = matched_df.head(max_rows_i)

    rows_out = matched_df.to_dicts()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "auction_prefix": prefix,
        "bid": bid_norm,
        "total_count": total_count,
        "rows": rows_out,
        "elapsed_ms": round(elapsed_ms, 1),
    }


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
    # PRE-LOAD: EV stats from GPU pipeline (optional optimization) - NV/V split (library-first)
    # =========================================================================
    bt_ev_stats_df = state.get("bt_ev_stats_df")
    ev_stats_lookup: Dict[int, Dict[str, Any]] = {}
    if bt_ev_stats_df is not None and next_bid_rows.height > 0:
        bt_indices = next_bid_rows["bt_index"].unique().to_list()
        ev_stats_lookup = preload_precomputed_ev_par_stats(
            bt_ev_stats_df,
            bt_indices=[int(x) for x in bt_indices if x is not None],
        )

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
        # (supports NV/V split with aggregate fallback)
        precomputed = ev_stats_lookup.get(bt_index, {}) if bt_index is not None else {}
        avg_ev_precomputed_nv, avg_ev_precomputed_v, avg_par_precomputed_nv, avg_par_precomputed_v = (
            get_avg_ev_par_precomputed_nv_v(precomputed, seat=int(next_seat))
        )
        
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
        
        # Compute stats by vulnerability relative to the *next bidder seat* (library-first).
        split_stats = compute_vul_split_par_ev_at_bid(matched_df, next_seat=int(next_seat))
        nv_count = int(split_stats.get("nv_count") or 0)
        v_count = int(split_stats.get("v_count") or 0)
        avg_par_nv = split_stats.get("avg_par_nv")
        avg_par_v = split_stats.get("avg_par_v")
        ev_score_nv = split_stats.get("ev_score_nv")
        ev_score_v = split_stats.get("ev_score_v")
        ev_std_nv = split_stats.get("ev_std_nv")
        ev_std_v = split_stats.get("ev_std_v")
        
        # Compute EV and Makes % for all level-strain-vul-seat combinations (560 columns) (library-first).
        ev_all_combos = compute_ev_all_combos_for_matched_deals(matched_df)
        
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
    global_mask, untracked_criteria = _build_criteria_mask_for_all_seats(
        deal_df,
        bt_row,
        deal_criteria_by_seat_dfs,
        dealer_rotation=int(expected_passes or 0),
    )
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


# ---------------------------------------------------------------------------
# Bid Details – Stable schema (selected-bid drilldown)
# ---------------------------------------------------------------------------


def handle_bid_details(
    state: Dict[str, Any],
    auction: str,
    bid: str,
    max_deals: int,
    seed: Optional[int],
    vul_filter: Optional[str] = None,
    *,
    deal_index: Optional[int] = None,
    topk: int = 10,
    include_phase2a: bool = True,
    phase2a_include_keycards: bool = True,
    phase2a_include_onside: bool = True,
    include_timing: bool = False,
) -> Dict[str, Any]:
    """Return stable selected-bid details for (auction prefix + candidate bid).

    Includes:
    - Top-K par contracts + entropy
    - ParScore summary stats
    - Suit-length / fit / HCP histograms (phase2a)
    - Range percentiles for pinned deal (if deal_index provided)
    - Pinned-deal exclusion from aggregates (if deal_index is in the matched set)
    - Server-side caching (LRU)
    """
    t0 = time.perf_counter()
    t_mark = t0
    timing: dict[str, float] = {}

    bid_norm = str(bid or "").strip().upper()
    if not bid_norm:
        raise ValueError("bid is required")

    auction_input = normalize_auction_input(auction)
    expected_passes = _count_leading_passes(auction_input)
    auction_for_lookup = auction_input.rstrip("-") if auction_input else ""
    auction_normalized = re.sub(r"(?i)^(p-)+", "", auction_input) if auction_input else ""
    auction_for_traversal = auction_normalized.rstrip("-") if auction_normalized else ""

    # Seat of next bidder relative to dealer (Seat 1 = dealer), INCLUDING leading passes.
    call_tokens = [t for t in auction_for_lookup.split("-") if t] if auction_for_lookup else []
    next_seat = (len(call_tokens) % 4) + 1

    cache_key = (
        "bid-details-v1",
        auction_input,
        bid_norm,
        int(max_deals),
        int(seed or 0),
        str(vul_filter or ""),
        int(deal_index) if deal_index is not None else None,
        int(topk),
        bool(include_phase2a),
        bool(phase2a_include_keycards),
        bool(phase2a_include_onside),
    )
    cached = _lru_get(_BID_DETAILS_CACHE, cache_key, _BID_DETAILS_CACHE_LOCK)
    if cached is not None:
        out = dict(cached)
        out["cache"] = {"hit": True}
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if include_timing:
            out["timing"] = {"cached": 1.0}
        return out

    deal_df = state["deal_df"]
    bt_seat1_df = state.get("bt_seat1_df")
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs") or {}

    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")

    # Match BT row for (auction prefix + selected next bid)
    bt_row: Optional[Dict[str, Any]] = None
    child_bt_index: Optional[int] = None

    if not auction_input:
        # Opening bids: MUST use bt_openings_df (fast). Do not scan bt_seat1_df (461M rows).
        bt_openings_df = state.get("bt_openings_df")
        if not isinstance(bt_openings_df, pl.DataFrame) or bt_openings_df.is_empty():
            raise ValueError(
                "bt_openings_df is missing/empty; refusing to scan bt_seat1_df for opening bids. "
                "Restart the API server and ensure initialization completes successfully."
            )
        seat1_df = bt_openings_df.filter(pl.col("seat") == 1) if "seat" in bt_openings_df.columns else bt_openings_df
        bid_df = seat1_df.filter(pl.col("Auction") == bid_norm)
        if bid_df.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            out = {
                "auction_input": auction_input,
                "auction_normalized": auction_normalized,
                "bid": bid_norm,
                "child_bt_index": None,
                "matched_deals_total": 0,
                "matched_deals_sampled": 0,
                "pinned_deal_excluded": False,
                "error": f"Opening bid '{bid_norm}' not found in bt_openings_df",
                "elapsed_ms": round(elapsed_ms, 1),
                "cache": {"hit": False},
            }
            _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
            return out
        bt_row = dict(bid_df.row(0, named=True))
        if "candidate_bid" not in bt_row and bt_row.get("Auction"):
            bt_row["candidate_bid"] = bt_row["Auction"]
        child_bt_index = None
        try:
            v = bt_row.get("bt_index")
            child_bt_index = int(v) if v is not None else None
        except Exception:
            child_bt_index = None
    else:
        is_passes_only = not auction_normalized or auction_normalized.lower() in ("p", "")
        if is_passes_only and expected_passes > 0:
            # Leading-pass support (pass-only prefix):
            # We cannot traverse BT along opening passes (seat-1 canonical), so we treat the
            # prefix as "opening bids" while using expected_passes to:
            # - filter deals by exact leading-pass count (opening_seat_mask below)
            # - rotate criteria evaluation (dealer_rotation=expected_passes)
            bt_openings_df = state.get("bt_openings_df")
            if not isinstance(bt_openings_df, pl.DataFrame) or bt_openings_df.is_empty():
                raise ValueError(
                    "bt_openings_df is missing/empty; cannot resolve pass-only prefix without Auction scans. "
                    "Restart the API server and ensure initialization completes successfully."
                )
            seat1_df = bt_openings_df.filter(pl.col("seat") == 1) if "seat" in bt_openings_df.columns else bt_openings_df
            bid_df = seat1_df.filter(pl.col("Auction") == bid_norm)
            if bid_df.height == 0:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                out = {
                    "auction_input": auction_input,
                    "auction_normalized": auction_normalized,
                    "bid": bid_norm,
                    "child_bt_index": None,
                    "matched_deals_total": 0,
                    "matched_deals_sampled": 0,
                    "pinned_deal_excluded": False,
                    "error": f"Opening bid '{bid_norm}' not found in bt_openings_df (pass-only prefix mode)",
                    "elapsed_ms": round(elapsed_ms, 1),
                    "cache": {"hit": False},
                }
                _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
                return out
            bt_row = dict(bid_df.row(0, named=True))
            if "candidate_bid" not in bt_row and bt_row.get("Auction"):
                bt_row["candidate_bid"] = bt_row["Auction"]
            child_bt_index = None
            try:
                v = bt_row.get("bt_index")
                child_bt_index = int(v) if v is not None else None
            except Exception:
                child_bt_index = None
        # If we didn't resolve bt_row via the pass-only opening-bids fast-path above,
        # resolve it via traversal from the non-pass prefix.
        if bt_row is None:
            parent_bt_index = _resolve_bt_index_by_traversal(state, auction_for_traversal)
            if parent_bt_index is None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                out = {
                    "auction_input": auction_input,
                    "auction_normalized": auction_normalized,
                    "bid": bid_norm,
                    "child_bt_index": None,
                    "matched_deals_total": 0,
                    "matched_deals_sampled": 0,
                    "pinned_deal_excluded": False,
                    "error": f"Auction '{auction_for_traversal}' not found in BT (traversal failed)",
                    "elapsed_ms": round(elapsed_ms, 1),
                    "cache": {"hit": False},
                }
                _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
                return out

            next_indices = _get_next_bid_indices_for_parent(state, parent_bt_index)
            if not next_indices:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                out = {
                    "auction_input": auction_input,
                    "auction_normalized": auction_normalized,
                    "bid": bid_norm,
                    "child_bt_index": None,
                    "matched_deals_total": 0,
                    "matched_deals_sampled": 0,
                    "pinned_deal_excluded": False,
                    "message": "No next bids found (auction may be completed)",
                    "elapsed_ms": round(elapsed_ms, 1),
                    "cache": {"hit": False},
                }
                _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
                return out

            rows_by_bid = _get_next_bt_rows_for_parent(state, int(parent_bt_index))
            if not rows_by_bid:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                out = {
                    "auction_input": auction_input,
                    "auction_normalized": auction_normalized,
                    "bid": bid_norm,
                    "child_bt_index": None,
                    "matched_deals_total": 0,
                    "matched_deals_sampled": 0,
                    "pinned_deal_excluded": False,
                    "error": "BT lookup returned no candidate rows for this parent (cached expansion empty)",
                    "elapsed_ms": round(elapsed_ms, 1),
                    "cache": {"hit": False},
                }
                _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
                return out

            bt_row = rows_by_bid.get(bid_norm)
            if bt_row is None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                out = {
                    "auction_input": auction_input,
                    "auction_normalized": auction_normalized,
                    "bid": bid_norm,
                    "child_bt_index": None,
                    "matched_deals_total": 0,
                    "matched_deals_sampled": 0,
                    "pinned_deal_excluded": False,
                    "error": f"Bid '{bid_norm}' not found among next-bid candidates for this auction",
                    "elapsed_ms": round(elapsed_ms, 1),
                    "cache": {"hit": False},
                }
                _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
                return out
            bt_row = dict(bt_row)
            child_bt_index = None
            try:
                v = bt_row.get("bt_index")
                child_bt_index = int(v) if v is not None else None
            except Exception:
                child_bt_index = None

    if bt_row is None:
        raise ValueError("Internal error: bt_row resolution failed")

    bt_row = _apply_all_rules_to_bt_row(dict(bt_row), state)
    if include_timing:
        now = time.perf_counter()
        timing["bt_lookup_ms"] = (now - t_mark) * 1000
        t_mark = now

    # Opening-seat alignment filter (matches handle_rank_bids_by_ev semantics).
    opening_seat_mask = None
    if "bid" in deal_df.columns:
        if expected_passes == 0:
            opening_seat_mask = ~deal_df["bid"].str.starts_with("p-")
        else:
            prefix = "p-" * expected_passes
            not_extra_pass = ~deal_df["bid"].str.starts_with("p-" * (expected_passes + 1))
            opening_seat_mask = deal_df["bid"].str.starts_with(prefix) & not_extra_pass

    # Track criteria referenced by the BT row but not present in deal bitmaps.
    untracked_criteria: list[str] = []
    # BT-level annotations for this node ("Expr" column); not necessarily used for deal matching.
    bt_expr: list[str] = []
    # Acting-seat criteria at this node (seat relative to dealer for this prefix).
    bt_acting_criteria: list[str] = []

    # Extract BT metadata before masking (so it exists even on early returns).
    try:
        raw_expr = bt_row.get("Expr") if isinstance(bt_row, dict) else None
        if isinstance(raw_expr, list):
            bt_expr = [str(x) for x in raw_expr if x is not None and str(x).strip()]
        elif raw_expr is not None:
            bt_expr = [str(raw_expr)]
        bt_expr = bt_expr[:50]
    except Exception:
        bt_expr = []

    try:
        # BT criteria are seat-1 canonical (opener = seat 1). When the auction prefix has
        # leading passes, the real "next_seat" (dealer-relative) must be rotated back into
        # BT-canonical seat numbering to retrieve the intended criteria list.
        bt_seat = ((int(next_seat) - 1 - int(expected_passes or 0)) % 4) + 1
        k = f"Agg_Expr_Seat_{int(bt_seat)}"
        raw = bt_row.get(k) if isinstance(bt_row, dict) else None
        if isinstance(raw, list):
            bt_acting_criteria = [str(x) for x in raw if x is not None and str(x).strip()]
        elif raw is not None:
            bt_acting_criteria = [str(raw)]
        bt_acting_criteria = bt_acting_criteria[:50]
    except Exception:
        bt_acting_criteria = []

    # Leading-pass support: rotate dealer when evaluating criteria (seat alignment),
    # matching Auction Builder semantics.
    global_mask, untracked_criteria = _build_criteria_mask_for_all_seats(
        deal_df,
        bt_row,
        deal_criteria_by_seat_dfs,
        dealer_rotation=int(expected_passes or 0),
    )
    if include_timing:
        now = time.perf_counter()
        timing["criteria_mask_ms"] = (now - t_mark) * 1000
        t_mark = now

    # Normalize untracked criteria list (stable, UI-friendly)
    try:
        untracked_criteria = [str(x) for x in (untracked_criteria or []) if x is not None and str(x).strip()]
        untracked_criteria = list(dict.fromkeys(untracked_criteria))
    except Exception:
        untracked_criteria = []
    if global_mask is None or not global_mask.any():
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out = {
            "auction_input": auction_input,
            "auction_normalized": auction_normalized,
            "bid": bid_norm,
            "child_bt_index": child_bt_index,
            "matched_deals_total": 0,
            "matched_deals_sampled": 0,
            "pinned_deal_excluded": False,
            "message": "No matched deals for this bid (criteria mask empty)",
            "elapsed_ms": round(elapsed_ms, 1),
            "cache": {"hit": False},
        }
        if include_timing:
            out["timing"] = dict(timing)
        _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
        return out

    if opening_seat_mask is not None:
        global_mask = global_mask & opening_seat_mask
        if not global_mask.any():
            elapsed_ms = (time.perf_counter() - t0) * 1000
            out = {
                "auction_input": auction_input,
                "auction_normalized": auction_normalized,
                "bid": bid_norm,
                "child_bt_index": child_bt_index,
                "matched_deals_total": 0,
                "matched_deals_sampled": 0,
                "pinned_deal_excluded": False,
                "message": "No matched deals after opening-seat alignment filter",
                "elapsed_ms": round(elapsed_ms, 1),
                "cache": {"hit": False},
            }
            _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
            return out

    matched_df = deal_df.filter(global_mask)
    if include_timing:
        now = time.perf_counter()
        timing["filter_matches_ms"] = (now - t_mark) * 1000
        t_mark = now
    if matched_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out = {
            "auction_input": auction_input,
            "auction_normalized": auction_normalized,
            "bid": bid_norm,
            "child_bt_index": child_bt_index,
            "matched_deals_total": 0,
            "matched_deals_sampled": 0,
            "pinned_deal_excluded": False,
            "message": "No matched deals for this bid",
            "elapsed_ms": round(elapsed_ms, 1),
            "cache": {"hit": False},
        }
        _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
        return out

    # Optional vulnerability filter (exact data value: None/Both/N_S/E_W).
    if vul_filter and vul_filter != "all" and "Vul" in matched_df.columns:
        data_vul = _VUL_UI_TO_DATA.get(vul_filter, vul_filter)
        matched_df = matched_df.filter(pl.col("Vul") == data_vul)
        if matched_df.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            out = {
                "auction_input": auction_input,
                "auction_normalized": auction_normalized,
                "bid": bid_norm,
                "child_bt_index": child_bt_index,
                "matched_deals_total": 0,
                "matched_deals_sampled": 0,
                "pinned_deal_excluded": False,
                "message": f"No matched deals after vul_filter={vul_filter!r}",
                "elapsed_ms": round(elapsed_ms, 1),
                "cache": {"hit": False},
            }
            _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
            return out

    matched_total_pre_excl = int(matched_df.height)

    pinned_row: dict[str, Any] | None = None
    pinned_in_matches = False
    seat_dir: str | None = None
    degraded_mode: str | None = None
    degraded_reasons: list[str] = []

    if deal_index is not None:
        pinned_row = _resolve_deal_row_by_deal_index(state, int(deal_index))
        if pinned_row is not None:
            try:
                dealer = str(pinned_row.get("Dealer", "N")).upper()
                seat_dir = seat_to_direction(dealer, int(next_seat))
            except Exception:
                seat_dir = None
        # Exclude pinned from aggregates ONLY if it is present in the matched set.
        if "index" in matched_df.columns:
            try:
                pinned_in_matches = matched_df.filter(pl.col("index") == int(deal_index)).height > 0
            except Exception:
                pinned_in_matches = False
            if pinned_in_matches:
                matched_df = matched_df.filter(pl.col("index") != int(deal_index))

    matched_total_post_excl = int(matched_df.height)

    # Explicit degraded modes (schema-stable; caller can surface warnings in UI).
    # These are intentionally simple thresholds for MVP phase2a posteriors.
    if matched_total_post_excl < 50:
        degraded_mode = "very_sparse_matches"
        degraded_reasons.append(f"n_excluding_pinned<{50}")
    elif matched_total_post_excl < 200:
        degraded_mode = "sparse_matches"
        degraded_reasons.append(f"n_excluding_pinned<{200}")

    effective_seed = _effective_seed(seed)
    max_deals_i = max(1, int(max_deals))
    if matched_df.height > max_deals_i:
        matched_df = matched_df.sample(n=max_deals_i, seed=effective_seed)
    if include_timing:
        now = time.perf_counter()
        timing["sample_ms"] = (now - t_mark) * 1000
        t_mark = now

    # Select only required columns for the computation (keep memory bounded).
    cols: list[str] = ["index", "Dealer", "ParScore", "ParContracts"]
    # Phase2a posteriors (auction-conditioned) require full directional numeric columns and hands.
    if include_phase2a:
        for d in ["N", "E", "S", "W"]:
            cols.append(f"HCP_{d}")
            cols.append(f"Total_Points_{d}")
            for su in ["S", "H", "D", "C"]:
                cols.append(f"SL_{d}_{su}")
        # Hand columns are only required when keycards/onside summaries are requested.
        if bool(phase2a_include_keycards) or bool(phase2a_include_onside):
            for d in ["N", "E", "S", "W"]:
                cols.append(f"Hand_{d}")

    cols = [c for c in cols if c in matched_df.columns]
    deals_sample_df = matched_df.select(cols)

    cfg = BidDetailsConfig(topk=int(topk), include_phase2a=bool(include_phase2a), include_honor_location=False)
    computed = compute_bid_details_from_sample(
        deals_sample_df,
        next_seat=int(next_seat),
        seat_dir=seat_dir or "N",
        pinned_row=None,
        cfg=cfg,
    )
    if include_timing:
        now = time.perf_counter()
        timing["bid_details_compute_ms"] = (now - t_mark) * 1000
        t_mark = now

    # Phase2a: auction-conditioned posteriors (SELF/PARTNER/LHO/RHO) + threat/keycard/onside.
    phase2a = None
    if include_phase2a:
        try:
            # IMPORTANT:
            # - We exclude the pinned deal from aggregates (to avoid leaking it into means / posteriors).
            # - But Phase2a range percentiles need the pinned row's values to compare against the
            #   auction-conditioned SELF posterior.
            #
            # To support this without changing the library surface, we append the pinned row back
            # into the Phase2a input when available (even if it is not in the matched set after
            # opening-seat alignment). This matches Auction Builder semantics: the user is asking
            # "how typical is my pinned hand under this bid's posterior?" even for hypothetical
            # pass prefixes.
            phase2a_df = deals_sample_df
            if pinned_row is not None and deal_index is not None:
                try:
                    pinned_df = pl.DataFrame([pinned_row])
                    pinned_df = pinned_df.select([c for c in cols if c in pinned_df.columns])
                    if pinned_df.height == 1 and "index" in pinned_df.columns:
                        phase2a_df = pl.concat([phase2a_df, pinned_df], how="diagonal_relaxed")
                except Exception:
                    phase2a_df = deals_sample_df

            phase2a = compute_phase2a_auction_conditioned_posteriors(
                phase2a_df,
                next_seat=int(next_seat),
                pinned_deal_index=int(deal_index) if deal_index is not None else None,
                include_keycards=bool(phase2a_include_keycards),
                include_onside=bool(phase2a_include_onside),
            )
        except Exception as e:
            degraded_mode = degraded_mode or "phase2a_unavailable"
            degraded_reasons.append(f"phase2a_error:{e}")
            phase2a = None
    if include_timing:
        now = time.perf_counter()
        timing["phase2a_ms"] = (now - t_mark) * 1000
        t_mark = now

    elapsed_ms = (time.perf_counter() - t0) * 1000
    out = {
        "auction_input": auction_input,
        "auction_normalized": auction_normalized,
        "auction_for_traversal": auction_for_traversal,
        "bid": bid_norm,
        "next_seat": int(next_seat),
        "seat_dir": seat_dir,
        "child_bt_index": child_bt_index,
        # BT-level annotations for this node (not necessarily used for deal matching).
        "bt_expr": bt_expr,
        "bt_acting_criteria": bt_acting_criteria,
        "bt_acting_criteria_count": int(len(bt_acting_criteria)),
        "untracked_criteria": untracked_criteria,
        "untracked_criteria_count": int(len(untracked_criteria)),
        "matched_deals_total": int(matched_total_pre_excl),
        "matched_deals_total_excluding_pinned": int(matched_total_post_excl),
        "matched_deals_sampled": int(deals_sample_df.height),
        "pinned_deal_index": int(deal_index) if deal_index is not None else None,
        "pinned_deal_excluded": bool(pinned_in_matches),
        "degraded_mode": degraded_mode,
        "degraded_reasons": degraded_reasons,
        "cache": {"hit": False},
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if include_timing:
        timing["total_ms"] = elapsed_ms
        out["timing"] = dict(timing)
    out.update(computed)
    if phase2a is not None:
        # Keep top-level stable pointer for consumers; phase2a is the new preferred surface.
        out["phase2a"] = phase2a
        # Promote role-based percentiles to the stable top-level key when available.
        if isinstance(phase2a, dict) and phase2a.get("range_percentiles") is not None:
            out["range_percentiles"] = phase2a.get("range_percentiles")

    _lru_put(_BID_DETAILS_CACHE, cache_key, dict(out), _CACHE_MAX_BID_DETAILS, _BID_DETAILS_CACHE_LOCK)
    return out


# ---------------------------------------------------------------------------
# Explain Bid – EEO + templates + counterfactual (no RAG)
# ---------------------------------------------------------------------------


def handle_explain_bid(
    state: Dict[str, Any],
    auction: str,
    bid: str,
    max_deals: int,
    seed: Optional[int],
    vul_filter: Optional[str] = None,
    *,
    deal_index: Optional[int] = None,
    topk: int = 10,
    include_phase2a: bool = True,
    why_not_bid: Optional[str] = None,
) -> Dict[str, Any]:
    """Explain a selected bid using computed evidence only (no RAG yet)."""
    t0 = time.perf_counter()

    bid_norm = str(bid or "").strip().upper()
    if not bid_norm:
        raise ValueError("bid is required")
    alt_norm = str(why_not_bid or "").strip().upper() or None

    auction_input = normalize_auction_input(auction)
    cache_key = (
        "explain-bid-v1",
        auction_input,
        bid_norm,
        alt_norm,
        int(max_deals),
        int(seed or 0),
        str(vul_filter or ""),
        int(deal_index) if deal_index is not None else None,
        int(topk),
        bool(include_phase2a),
    )

    cached = _lru_get(_EXPLAIN_BID_CACHE, cache_key, _EXPLAIN_BID_CACHE_LOCK)
    if cached is not None:
        out = dict(cached)
        out["cache"] = {"hit": True}
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    # Reuse bid-details evidence (with its own cache + pinned exclusion).
    bid_details = handle_bid_details(
        state=state,
        auction=auction_input,
        bid=bid_norm,
        max_deals=max_deals,
        seed=seed,
        vul_filter=vul_filter,
        deal_index=deal_index,
        topk=topk,
        include_phase2a=include_phase2a,
    )
    eeo = compute_eeo_from_bid_details(bid_details)
    expl = render_recommendation_explanation(bid=bid_norm, eeo=eeo)

    counterfactual = None
    alt_details = None
    alt_eeo = None
    if alt_norm:
        alt_details = handle_bid_details(
            state=state,
            auction=auction_input,
            bid=alt_norm,
            max_deals=max_deals,
            seed=seed,
            vul_filter=vul_filter,
            deal_index=deal_index,
            topk=topk,
            include_phase2a=include_phase2a,
        )
        alt_eeo = compute_eeo_from_bid_details(alt_details)
        counterfactual = render_counterfactual_why_not(bid=bid_norm, alt_bid=alt_norm, eeo=eeo, alt_eeo=alt_eeo)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    out = {
        "auction_input": auction_input,
        "bid": bid_norm,
        "why_not_bid": alt_norm,
        "eeo": eeo,
        "explanation": expl,
        "counterfactual": counterfactual,
        "bid_details": bid_details,
        "why_not_bid_details": alt_details,
        "why_not_eeo": alt_eeo,
        "cache": {"hit": False},
        "elapsed_ms": round(elapsed_ms, 1),
    }

    _lru_put(_EXPLAIN_BID_CACHE, cache_key, dict(out), _CACHE_MAX_EXPLAIN_BID, _EXPLAIN_BID_CACHE_LOCK)
    return out


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
    dealer: Optional[str] = None,
    vulnerable: Optional[str] = None,
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

    # Optional board context: dealer + board vulnerability -> acting seat + seat-vul bucket.
    # This lets the API return a single "selected" Avg_Par/Avg_EV without making the client choose NV/V.
    context: dict[str, Any] | None = None
    try:
        if dealer is not None or vulnerable is not None:
            if dealer is None or vulnerable is None:
                raise ValueError("Both dealer and vulnerable must be provided together (or neither).")
            dealer_n = normalize_dealer_strict(dealer)
            board_vul = normalize_board_vulnerable(vulnerable)
            act = compute_seat_to_act(dealer=dealer_n, auction_full=auction_input)
            seat_to_act_dir = str(act["seat_to_act_dir"])
            seat_to_act_seat = int(act["seat_to_act_seat"])
            seat_vul = seat_vul_bucket(board_vul, seat_to_act_dir)
            context = {
                "dealer": dealer_n,
                "vulnerable": board_vul,
                "seat_to_act_dir": seat_to_act_dir,
                "seat_to_act_seat": seat_to_act_seat,
                "seat_vul": seat_vul,
            }
    except Exception as e:
        context = {
            "error": f"invalid_context:{e}",
            "dealer": dealer,
            "vulnerable": vulnerable,
        }

    # 5. Build EV/Par stats lookup (if available) - NV/V splits (library-first)
    bt_ev_stats_df = state.get("bt_ev_stats_df")
    ev_lookup: Dict[int, Dict[str, Any]] = {}
    if include_ev_stats and bt_ev_stats_df is not None and next_bid_rows.height > 0:
        bt_indices = next_bid_rows["bt_index"].unique().to_list()
        ev_lookup = preload_precomputed_ev_par_stats(bt_ev_stats_df, bt_indices=[int(x) for x in bt_indices if x is not None])

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
        
        # Get Avg_EV / Avg_Par for the next seat from precomputed stats (NV/V split with aggregate fallback)
        avg_ev_nv: float | None = None
        avg_ev_v: float | None = None
        avg_par_nv: float | None = None
        avg_par_v: float | None = None
        if include_ev_stats and idx is not None and idx in ev_lookup:
            ev_data = ev_lookup[idx]
            avg_ev_nv, avg_ev_v, avg_par_nv, avg_par_v = get_avg_ev_par_precomputed_nv_v(ev_data, seat=int(next_seat))

        # Convenience selection when context is provided (aligns with canonical doc).
        avg_ev = None
        avg_par = None
        if context and context.get("seat_vul") in ("NV", "V"):
            seat_vul = str(context["seat_vul"])
            avg_ev = avg_ev_v if seat_vul == "V" else avg_ev_nv
            avg_par = avg_par_v if seat_vul == "V" else avg_par_nv
            
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
            "avg_par_nv": avg_par_nv,
            "avg_par_v": avg_par_v,
            # Convenience columns (selected by board context when provided)
            "avg_ev": avg_ev,
            "avg_par": avg_par,
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
    resp: dict[str, Any] = {"auction_input": auction_input, "next_bids": next_bids, "elapsed_ms": round(elapsed_ms, 1)}
    if context is not None:
        resp["context"] = context
    return resp


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

        # Lazy-load bitmap table only if we encounter a bitmap-only criterion.
        # Dynamic criteria (SL/HCP/Total_Points expressions) should still be
        # evaluatable and pass even when bitmap tables are unavailable.
        criteria_df = None
        available_cols: set[str] | None = None
        
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
            if available_cols is None:
                criteria_df = deal_criteria_by_seat_dfs.get(bt_seat, {}).get(dealer_rot)
                if criteria_df is None:
                    # Strict policy for bitmap-only criteria: if we cannot evaluate
                    # against the bitmap table, this criterion fails.
                    return False
                available_cols = set(criteria_df.columns)
            if crit_s not in available_cols:
                return False
            try:
                assert criteria_df is not None
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
                    criteria_list = [str(x) for x in (rr.get("criteria") or []) if x is not None]
                    # If these rule criteria are already part of the candidate's agg_expr,
                    # they were already checked in the primary candidate filter pass.
                    # Re-checking them here can spuriously fail due context differences.
                    cand_expr = [str(x) for x in (expr or []) if x is not None]
                    if all(c in cand_expr for c in criteria_list):
                        continue
                    # Must satisfy ALL criteria for this overlay rule.
                    try:
                        if not eval_criteria(criteria_list, bt_seat, dealer_rot):
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

                    memo_key = (
                        int(deal_row_idx) if deal_row_idx is not None else -1,
                        str(dealer_actual).upper(),
                        str(next_auc).upper(),
                    )
                    memo_hit = _lru_get(_GREEDY_EVAL_CACHE, memo_key, _GREEDY_EVAL_CACHE_LOCK)
                    if memo_hit is not None:
                        dd_score = float(memo_hit.get("dd_score", float("-inf")))
                        ev_score = float(memo_hit.get("ev_score", 0.0))
                    else:
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
                        _lru_put(
                            _GREEDY_EVAL_CACHE,
                            memo_key,
                            {"dd_score": float(dd_score), "ev_score": float(ev_score)},
                            _CACHE_MAX_GREEDY_EVAL,
                            _GREEDY_EVAL_CACHE_LOCK,
                        )
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

def _trace_criteria_failures(
    criteria_list: List[str],
    dealer: str,
    seat: int,
    deal_row: Dict[str, Any],
    state: Dict[str, Any],
    deal_row_dict: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Evaluate each criterion individually and return pass/fail/value trace.

    Returns a list of dicts with keys: criterion, passed, annotated.
    Used to identify which specific criterion blocked a bid at a given step.
    """
    trace: List[Dict[str, Any]] = []
    row_source = deal_row_dict if deal_row_dict is not None else deal_row
    for crit in (criteria_list or []):
        crit_s = str(crit)
        passed: bool | None = None
        annotated = crit_s
        try:
            sl_result = evaluate_sl_criterion(crit_s, dealer, seat, row_source, fail_on_missing=False)
            if sl_result is not None:
                passed = bool(sl_result)
                annotated = annotate_criterion_with_value(crit_s, dealer, seat, row_source)
            else:
                # Try via criteria_pythonized_exprs_by_direction
                direction = seat_to_direction(dealer, seat)
                criteria_exprs = state.get("criteria_pythonized_exprs_by_direction") or {}
                dir_exprs = (criteria_exprs.get(direction) or {})
                pythonized_expr = dir_exprs.get(crit_s)
                if pythonized_expr is not None:
                    try:
                        one_row_df = pl.DataFrame([row_source])
                        eval_env = {col: pl.col(col) for col in one_row_df.columns}
                        safe_expr_str = re.sub(r"\bTrue\b", "pl.lit(True)", pythonized_expr)
                        safe_expr_str = re.sub(r"\bFalse\b", "pl.lit(False)", safe_expr_str)
                        polars_expr = eval(safe_expr_str, {"pl": pl}, eval_env)
                        passed = bool(one_row_df.select(polars_expr)[0, 0])
                    except Exception:
                        passed = None
                else:
                    # Evaluate as complex expression (last resort)
                    sl_result2 = evaluate_sl_criterion(crit_s, dealer, seat, row_source, fail_on_missing=True)
                    if sl_result2 is not None:
                        passed = bool(sl_result2)
                        annotated = annotate_criterion_with_value(crit_s, dealer, seat, row_source)
        except Exception:
            passed = None
        trace.append({"criterion": crit_s, "passed": passed, "annotated": annotated})
    return trace


def _is_auction_complete_list(bids: List[str]) -> bool:
    if len(bids) >= 4 and all(b == "P" for b in bids[:4]):
        return True
    last_c = -1
    for i, b in enumerate(bids):
        if b not in ("P", "X", "XX") and b and b[0].isdigit():
            last_c = i
    return last_c >= 0 and len(bids) >= last_c + 4 and all(b == "P" for b in bids[-3:])


def handle_ai_model_advanced_path(
    state: Dict[str, Any],
    *,
    auction_prefix: str = "",
    deal_row_idx: int = -1,
    deal_row_dict: Dict[str, Any] | None = None,
    seed: int = 0,
    max_steps: int = 40,
    top_n: int = 12,
    max_deals: int = 500,
    w_desc: float = 150.0,
    w_threat: float = 120.0,
    w_guard: float = 1.0,
    w_guard_overbid: float = 80.0,
    w_guard_tp: float = 15.0,
    w_guard_neg: float = 0.5,
    w_guard_underbid: float = 40.0,
    w_guard_tp_surplus: float = 8.0,
    w_guard_strain: float = 60.0,
    w_guard_sacrifice: float = 0.7,
    w_guard_tricks: float = 30.0,
    opening_base_mult: float = 0.25,
    early_uncontested_base_mult: float = 0.5,
    opening_pass_penalty: float = 30.0,
    opening_policy_bonus: float = 20.0,
    permissive_pass: bool = True,
    logic_mode: str = "all_logic",
    use_guardrails_v2: bool = False,
) -> Dict[str, Any]:
    """Compute the 'AI Model (Advanced top-1 greedy)' completed auction in-process.

    Semantics mirror the Streamlit AI Model path builder:
    - At each step, consider next bids from BT.
    - Keep only bids that match the pinned deal's criteria (boolean pass), and are not rejected.
    - If no bids match, choose Pass (if permissive_pass and Pass is available), otherwise stop.
    - If exactly one bid matches, take it without scoring.
    - Else score top-N using bid-details evidence (utility + typicality - threat), pick best.

    When deal_row_dict is provided (on-the-fly PBN/CSV deals), deal_row_idx is ignored for
    the deal row lookup; criteria are evaluated dynamically via handle_deal_criteria_pass_batch.

    Returns:
      - auction: completed (or best-effort) auction string
      - steps_detail: per-step timings + counts
    """
    import concurrent.futures

    t0 = time.perf_counter()
    deal_df = state.get("deal_df")
    if deal_df is None and deal_row_dict is None:
        return {"auction": "", "steps": 0, "elapsed_ms": 0, "error": "deal_df not loaded"}

    # Resolve deal_row: prefer supplied dict (on-the-fly), else fetch from DB by index.
    if deal_row_dict is not None:
        deal_row: Dict[str, Any] = deal_row_dict
    else:
        if deal_df is None:
            return {"auction": "", "steps": 0, "elapsed_ms": 0, "error": "deal_df not loaded"}
        try:
            deal_row = deal_df.row(int(deal_row_idx), named=True)
        except Exception as e:
            return {"auction": "", "steps": 0, "elapsed_ms": 0, "error": f"invalid_deal_row_idx:{e}"}

    dealer_actual = str(deal_row.get("Dealer", "N")).upper()
    board_vul = deal_row.get("Vul", deal_row.get("Vulnerability"))
    deal_index = deal_row.get("index")
    try:
        deal_index_i = int(deal_index) if deal_index is not None else None
    except Exception:
        deal_index_i = None

    logic_mode_norm = str(logic_mode or "all_logic").strip().lower()
    if logic_mode_norm not in {"all_logic", "ai_bt_only", "guardrails_only"}:
        logic_mode_norm = "all_logic"
    use_guardrails_v2 = bool(use_guardrails_v2)
    use_bt_only_scoring = logic_mode_norm == "ai_bt_only"
    # v2 mode intentionally disables common-sense adjustments and hard overrides.
    use_common_sense = (logic_mode_norm == "all_logic") and (not use_guardrails_v2)

    def _min_from_hist(hist: dict[str, Any] | None) -> float | None:
        """Return minimum observed bucket key from histogram-like dict."""
        if not isinstance(hist, dict) or not hist:
            return None
        mn = None
        for k, v in hist.items():
            try:
                cnt = float(v)
                if cnt <= 0:
                    continue
                kk = float(int(str(k)))
            except Exception:
                continue
            if mn is None or kk < mn:
                mn = kk
        return mn

    def _max_from_hist(hist: dict[str, Any] | None) -> float | None:
        """Return maximum observed bucket key from histogram-like dict."""
        if not isinstance(hist, dict) or not hist:
            return None
        mx = None
        for k, v in hist.items():
            try:
                cnt = float(v)
                if cnt <= 0:
                    continue
                kk = float(int(str(k)))
            except Exception:
                continue
            if mx is None or kk > mx:
                mx = kk
        return mx

    def _strip_leading_passes_tokens(tokens: List[str]) -> Tuple[List[str], int]:
        n = 0
        for t in tokens:
            if str(t).upper() == "P":
                n += 1
            else:
                break
        if n >= len(tokens):
            return [], int(n)
        return tokens[n:], int(n)

    def _next_display_seat_from_tokens(tokens: List[str]) -> int:
        return (len(tokens) % 4) + 1

    def _bt_seat_from_display_seat(seat_1_to_4: int, leading_passes: int) -> int:
        k = int(leading_passes or 0) % 4
        return ((int(seat_1_to_4) - 1 - k) % 4) + 1

    def _token_bidder_dir_for_dealer(token_idx: int, dealer: str) -> str:
        directions = ["N", "E", "S", "W"]
        d = str(dealer or "N").upper()
        dealer_idx = directions.index(d) if d in directions else 0
        return directions[(dealer_idx + int(token_idx)) % 4]

    def _current_bidder_dir_from_tokens(tokens_now: List[str], dealer: str) -> str:
        directions = ["N", "E", "S", "W"]
        d = str(dealer or "N").upper()
        dealer_idx = directions.index(d) if d in directions else 0
        return directions[(dealer_idx + len(tokens_now)) % 4]

    def _partner_dir(direction: str) -> str:
        return {"N": "S", "S": "N", "E": "W", "W": "E"}.get(str(direction or "").upper(), "")

    def _blackwood_aces_response_allowed(tokens_now: List[str], dealer: str) -> set[str] | None:
        """When partner is answering uncontested 4N/4NT, require ace-count response.

        We only enforce this in the canonical uncontested spot:
        - last non-pass call is 4N/4NT by partner,
        - exactly one trailing pass (RHO pass),
        - now it's partner's turn to respond.
        """
        if not tokens_now:
            return None
        last_non_pass_idx = None
        for i in range(len(tokens_now) - 1, -1, -1):
            tk = str(tokens_now[i] or "").strip().upper()
            if tk and tk not in ("P", "PASS", "X", "XX"):
                last_non_pass_idx = i
                break
        if last_non_pass_idx is None:
            return None
        last_call = str(tokens_now[last_non_pass_idx] or "").strip().upper()
        if last_call not in ("4N", "4NT"):
            return None
        trailing = [str(t or "").strip().upper() for t in tokens_now[last_non_pass_idx + 1 :]]
        if len(trailing) != 1 or trailing[0] not in ("P", "PASS"):
            return None
        asker_dir = _token_bidder_dir_for_dealer(last_non_pass_idx, dealer)
        acting_dir = _current_bidder_dir_from_tokens(tokens_now, dealer)
        if acting_dir != _partner_dir(asker_dir):
            return None
        return {"5C", "5D", "5H", "5S"}

    def _is_raise_of_partner_suit(
        tokens_now: List[str],
        bid_text: str,
        acting_direction: str | None,
        dealer: str,
    ) -> bool:
        """Return True when bid_text is a raise of partner's previously bid strain."""
        if not acting_direction:
            return False
        b = str(bid_text or "").strip().upper()
        m_bid = re.match(r"^([1-7])\s*(NT|N|[CDHS])", b)
        if not m_bid:
            return False
        bid_lvl = int(m_bid.group(1))
        bid_st = "N" if m_bid.group(2).upper() in ("N", "NT") else m_bid.group(2).upper()
        partner_dir = _partner_dir(str(acting_direction).upper())
        if not partner_dir:
            return False
        for i in range(len(tokens_now) - 1, -1, -1):
            tk = str(tokens_now[i] or "").strip().upper()
            m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", tk)
            if not m:
                continue
            tk_dir = _token_bidder_dir_for_dealer(i, dealer)
            if tk_dir != partner_dir:
                continue
            tk_lvl = int(m.group(1))
            tk_st = "N" if m.group(2).upper() in ("N", "NT") else m.group(2).upper()
            return bool(tk_st == bid_st and bid_lvl > tk_lvl)
        return False

    def _rotate_dealer_by(dealer: str, offset: int) -> str:
        try:
            directions = ["N", "E", "S", "W"]
            d = str(dealer or "N").upper()
            i = directions.index(d) if d in directions else 0
            k = int(offset or 0) % 4
            return directions[(i + k) % 4]
        except Exception:
            return str(dealer or "N").upper()

    def _is_underspecified_criteria(criteria_list: list[str]) -> bool:
        """Check if criteria list is underspecified (missing both HCP and Total_Points)."""
        if not criteria_list:
            return False
        for crit in criteria_list:
            crit_upper = str(crit).upper()
            if "HCP" in crit_upper or "TOTAL_POINTS" in crit_upper:
                return False
        return True

    def _is_strong_2c_forcing_active(tokens_now: List[str], side: str, dealer: str) -> bool:
        """Detect canonical strong-2C opening force (until side reaches at least 2NT)."""
        try:
            side_u = str(side or "").upper()
            if side_u not in ("NS", "EW"):
                return False
            first_side_contract_idx: int | None = None
            first_side_contract_bid: str | None = None
            # Any side contract at/above 2NT (or higher level) satisfies the force target.
            reached_2nt_or_higher = False

            for i, tk in enumerate(list(tokens_now or [])):
                tk_u = str(tk or "").strip().upper()
                m = re.match(r"^([1-7])\s*(NT|N|[CDHS])$", tk_u)
                if not m:
                    continue
                lvl = int(m.group(1))
                st = "N" if m.group(2).upper() in ("NT", "N") else m.group(2).upper()
                bidder_dir = _token_bidder_dir_for_dealer(i, dealer)
                bidder_side = "NS" if bidder_dir in ("N", "S") else "EW"
                if bidder_side != side_u:
                    continue
                if first_side_contract_idx is None:
                    first_side_contract_idx = i
                    first_side_contract_bid = tk_u
                if lvl > 2 or (lvl == 2 and st == "N"):
                    reached_2nt_or_higher = True
                    break

            if first_side_contract_idx is None:
                return False
            if first_side_contract_bid != "2C":
                return False
            # Opening-context requirement: only passes before the 2C call.
            for t in list(tokens_now or [])[: int(first_side_contract_idx)]:
                t_u = str(t or "").strip().upper()
                if t_u not in ("", "P", "PASS"):
                    return False
            return not reached_2nt_or_higher
        except Exception:
            return False

    def _stayman_forcing_fallback_bid(tokens_now: List[str]) -> str | None:
        """Return forced responder-opener reply bid for Stayman contexts, else None.

        Contexts:
        - 1N-P-2C-P -> opener must reply (2D/2H/2S); use 2D as fail-safe fallback.
        - 2N-P-3C-P -> opener must reply (3D/3H/3S); use 3D as fail-safe fallback.
        """
        try:
            bt_tokens_now, _lp_now = _strip_leading_passes_tokens(tokens_now)
            bt_prefix_now = "-".join(bt_tokens_now).upper() if bt_tokens_now else ""
            if re.match(r"^1N-P-2C-P$", bt_prefix_now):
                return "2D"
            if re.match(r"^2N-P-3C-P$", bt_prefix_now):
                return "3D"
        except Exception:
            return None
        return None

    def _partner_cue_bid_fit_context(
        tokens_now: List[str],
        dealer_now: str,
        acting_dir_now: str,
    ) -> tuple[bool, str | None]:
        """Detect 'second pass after partner cue-bid' context and a fit suit to return to.

        Active when:
        - Last call is Pass (RHO passed over partner's call).
        - Partner's immediately previous call is a suit bid in opponents' shown suit.
        """
        try:
            if len(tokens_now) < 2:
                return False, None
            if str(tokens_now[-1] or "").strip().upper() not in ("P", "PASS"):
                return False, None

            partner_dir_now = _partner_dir(acting_dir_now)
            acting_side_now = "NS" if acting_dir_now in ("N", "S") else "EW"

            cue_idx = len(tokens_now) - 2
            cue_bid = str(tokens_now[cue_idx] or "").strip().upper()
            m = re.match(r"^([1-7])\s*(NT|N|[CDHS])$", cue_bid)
            if not m:
                return False, None
            cue_suit = str(m.group(2) or "").upper()
            if cue_suit in ("N", "NT"):
                return False, None

            cue_bidder = _token_bidder_dir_for_dealer(cue_idx, dealer_now)
            if cue_bidder != partner_dir_now:
                return False, None

            # Cue-bid check: opponents must have shown this suit earlier.
            opp_showed_cue_suit = False
            for i, tk in enumerate(tokens_now[:cue_idx]):
                tku = str(tk or "").strip().upper()
                mm = re.match(r"^([1-7])\s*(NT|N|[CDHS])$", tku)
                if not mm:
                    continue
                st = str(mm.group(2) or "").upper()
                if st in ("N", "NT") or st != cue_suit:
                    continue
                bidder_i = _token_bidder_dir_for_dealer(i, dealer_now)
                bidder_side_i = "NS" if bidder_i in ("N", "S") else "EW"
                if bidder_side_i != acting_side_now:
                    opp_showed_cue_suit = True
                    break
            if not opp_showed_cue_suit:
                return False, None

            # Choose fit strain from our side's prior natural suit bids, preferring
            # repeated strain support and most recent action.
            suit_stats: dict[str, tuple[int, int]] = {}
            for i, tk in enumerate(tokens_now[:cue_idx]):
                tku = str(tk or "").strip().upper()
                mm = re.match(r"^([1-7])\s*(NT|N|[CDHS])$", tku)
                if not mm:
                    continue
                st = str(mm.group(2) or "").upper()
                if st in ("N", "NT"):
                    continue
                bidder_i = _token_bidder_dir_for_dealer(i, dealer_now)
                bidder_side_i = "NS" if bidder_i in ("N", "S") else "EW"
                if bidder_side_i != acting_side_now:
                    continue
                c_prev, _last_prev = suit_stats.get(st, (0, -1))
                suit_stats[st] = (c_prev + 1, i)

            if not suit_stats:
                return True, None

            # If cue suit somehow dominates, prefer a non-cue side suit when available.
            non_cue = {k: v for k, v in suit_stats.items() if k != cue_suit}
            pick_from = non_cue if non_cue else suit_stats
            fit_suit = max(pick_from.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
            return True, fit_suit
        except Exception:
            return False, None

    def _pick_return_to_fit_bid(
        fit_suit: str | None,
        options: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """Pick the cheapest playable non-pass bid in fit_suit (or any suit if None)."""
        try:
            best: Dict[str, Any] | None = None
            best_key: tuple[int, int] | None = None
            order = {"C": 0, "D": 1, "H": 2, "S": 3}
            for o in list(options or []):
                bid_txt = str((o or {}).get("bid", "") or "").strip().upper()
                if bid_txt in ("", "P", "PASS"):
                    continue
                m = re.match(r"^([1-7])\s*(NT|N|[CDHS])$", bid_txt)
                if not m:
                    continue
                lvl = int(m.group(1))
                st = str(m.group(2) or "").upper()
                if st in ("N", "NT"):
                    continue
                if fit_suit is not None and st != fit_suit:
                    continue
                if not bool((o or {}).get("can_complete", True)):
                    continue
                if bool((o or {}).get("is_dead_end", False)):
                    continue
                key = (lvl, order.get(st, 9))
                if best_key is None or key < best_key:
                    best = o
                    best_key = key
            return best
        except Exception:
            return None

    def _check_nt_injection_conditions(
        tokens: List[str],
        dealer_actual: str,
        acting_dir: str,
        deal_row: Dict[str, Any],
        bt_seat: int,
        dealer_rot: str,
        deal_row_idx: int,
        passed_opts: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """Check whether a synthetic NT bid should be injected.

        Returns None if no injection is warranted, or a dict with keys:
          bid, reason, evidence
        """
        try:
            acting_side = "NS" if acting_dir in ("N", "S") else "EW"
            partner_dir = _partner_dir(acting_dir)

            # --- Suits bid by each side ---
            our_suits: set[str] = set()
            partner_suits: set[str] = set()
            all_suits_bid: set[str] = set()
            current_level = 0
            for ti, tk in enumerate(tokens):
                tk_u = str(tk or "").strip().upper()
                m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", tk_u)
                if not m:
                    continue
                lvl = int(m.group(1))
                st = m.group(2).upper()
                if st in ("NT", "N"):
                    if lvl > current_level:
                        current_level = lvl
                    continue
                if st not in ("C", "D", "H", "S"):
                    continue
                all_suits_bid.add(st)
                bidder = _token_bidder_dir_for_dealer(ti, dealer_actual)
                bidder_side = "NS" if bidder in ("N", "S") else "EW"
                if bidder_side == acting_side:
                    our_suits.add(st)
                    if bidder == partner_dir:
                        partner_suits.add(st)
                if lvl > current_level:
                    current_level = lvl

            # --- Parse acting player's hand ---
            hand_pbn = str(deal_row.get(f"Hand_{acting_dir}", "") or "").strip()
            if not hand_pbn or "." not in hand_pbn:
                return None
            parts = hand_pbn.split(".")
            if len(parts) != 4:
                return None
            sl_map = {"S": len(parts[0]), "H": len(parts[1]), "D": len(parts[2]), "C": len(parts[3])}

            # --- Implicit major-fit detection ---
            for major in ("H", "S"):
                if major in partner_suits and sl_map.get(major, 0) >= 3:
                    return None

            # --- Stopper check for unbid suits ---
            unbid = {"C", "D", "H", "S"} - all_suits_bid
            if len(unbid) > 1:
                return None
            if len(unbid) == 1:
                unbid_suit = next(iter(unbid))
                has_stopper = _has_stopper_in_suit(
                    unbid_suit, acting_dir, deal_row, bt_seat, dealer_rot, deal_row_idx
                )
                if not has_stopper:
                    return None

            # --- Combined HCP ---
            self_hcp: float | None = None
            partner_hcp: float | None = None
            try:
                v = deal_row.get(f"HCP_{acting_dir}")
                if v is not None:
                    self_hcp = float(v)
            except Exception:
                pass
            try:
                v = deal_row.get(f"HCP_{partner_dir}")
                if v is not None:
                    partner_hcp = float(v)
            except Exception:
                pass
            if self_hcp is None or partner_hcp is None:
                return None
            combined_hcp = self_hcp + partner_hcp

            # --- NT level selection ---
            if combined_hcp >= 25:
                nt_bid = "3N"
                nt_level = 3
            elif combined_hcp >= 23:
                nt_bid = "2N"
                nt_level = 2
            else:
                return None

            if nt_level <= current_level:
                if combined_hcp >= 25 and current_level < 3:
                    nt_bid = "3N"
                    nt_level = 3
                else:
                    return None
            if nt_level <= current_level:
                return None

            # --- Already in passed_opts? ---
            for po in passed_opts:
                b = str((po or {}).get("bid", "") or "").strip().upper()
                if b in (nt_bid, nt_bid + "T"):
                    return None

            evidence = {
                "all_suits_bid": sorted(all_suits_bid),
                "our_suits": sorted(our_suits),
                "unbid_suits": sorted(unbid),
                "combined_hcp": combined_hcp,
                "self_hcp": self_hcp,
                "partner_hcp": partner_hcp,
                "current_level": current_level,
                "acting_dir": acting_dir,
                "partner_suits": sorted(partner_suits),
                "sl_map": sl_map,
            }
            reason = (
                f"NT_INJECTION_GATE: suits covered={sorted(all_suits_bid)}, "
                f"unbid={sorted(unbid)}, combined_hcp={combined_hcp}, "
                f"no major fit -> inject {nt_bid}"
            )
            return {"bid": nt_bid, "reason": reason, "evidence": evidence}

        except Exception:
            return None

    def _has_stopper_in_suit(
        suit: str,
        direction: str,
        deal_row: Dict[str, Any],
        bt_seat: int,
        dealer_rot: str,
        deal_row_idx: int,
    ) -> bool:
        """Check stopper via bitmap, falling back to dynamic hand computation."""
        crit_name = f"Stop_In_{suit}"
        try:
            _dcbsd = state.get("deal_criteria_by_seat_dfs", {})
            criteria_df = _dcbsd.get(bt_seat, {}).get(dealer_rot)
            if criteria_df is not None and crit_name in criteria_df.columns:
                return bool(criteria_df[crit_name][int(deal_row_idx)])
        except Exception:
            pass
        # Dynamic fallback: Stop_In = hcp >= 4 | (sl + hcp) >= 6
        hand_pbn = str(deal_row.get(f"Hand_{direction}", "") or "").strip()
        if not hand_pbn or "." not in hand_pbn:
            return False
        parts = hand_pbn.split(".")
        if len(parts) != 4:
            return False
        suit_idx = {"S": 0, "H": 1, "D": 2, "C": 3}.get(suit)
        if suit_idx is None:
            return False
        suit_cards = parts[suit_idx]
        sl = len(suit_cards)
        honors = {"A": 4, "K": 3, "Q": 2, "J": 1}
        suit_hcp = sum(honors.get(c.upper(), 0) for c in suit_cards)
        return suit_hcp >= 4 or (sl + suit_hcp) >= 6

    def _apply_direct_overcall_gap_rescue(
        *,
        passed_opts_now: List[Dict[str, Any]],
        filtered_bids_now: List[Dict[str, Any]],
        next_bids_now: List[Dict[str, Any]],
        tokens_now: List[str],
        acting_side_now: str,
        acting_dir_now: str,
        deal_row_now: Dict[str, Any],
        bt_seat_now: int,
        dealer_rot_now: str,
        dealer_actual_now: str,
        deal_row_idx_now: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str | None]:
        """Rescue plausible direct overcalls when BT gaps leave only Pass."""
        passed_non_pass_now = [
            o for o in list(passed_opts_now or [])
            if str((o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
        ]
        if passed_non_pass_now or not filtered_bids_now or not deal_row_now:
            return passed_opts_now, filtered_bids_now, None

        opp_natural: list[tuple[int, str, str]] = []
        our_non_pass = 0
        for i, tk in enumerate(list(tokens_now or [])):
            bid = str(tk or "").strip().upper()
            if bid in ("", "P", "PASS", "D", "X", "DOUBLE", "R", "XX", "REDOUBLE"):
                continue
            bidder_dir = _token_bidder_dir_for_dealer(i, dealer_actual_now)
            bidder_side = "NS" if bidder_dir in ("N", "S") else "EW"
            if len(bid) >= 2 and bid[0].isdigit() and bid[1:] in ("C", "D", "H", "S"):
                if bidder_side == acting_side_now:
                    our_non_pass += 1
                else:
                    opp_natural.append((int(bid[0]), bid[1:], bidder_dir))
            elif bidder_side == acting_side_now:
                our_non_pass += 1

        if our_non_pass != 0 or not opp_natural:
            return passed_opts_now, filtered_bids_now, None

        last_opp_level, last_opp_strain, _ = opp_natural[-1]
        hand_pbn = str(deal_row_now.get(f"Hand_{acting_dir_now}", "") or "").strip()
        sl_map: dict[str, int] = {}
        if hand_pbn and "." in hand_pbn:
            parts = hand_pbn.split(".")
            if len(parts) == 4:
                sl_map = {"S": len(parts[0]), "H": len(parts[1]), "D": len(parts[2]), "C": len(parts[3])}

        tp_raw = deal_row_now.get(f"Total_Points_{acting_dir_now}")
        hcp_raw = deal_row_now.get(f"HCP_{acting_dir_now}")
        tp_val = float(tp_raw) if tp_raw is not None else None
        hcp_val = float(hcp_raw) if hcp_raw is not None else None
        next_by_bid = {
            str((nb or {}).get("bid", "") or "").strip().upper(): nb
            for nb in (next_bids_now or [])
        }

        lowest_suit_bid: dict[str, str] = {}
        promoted: list[dict[str, Any]] = []
        promoted_names: set[str] = set()

        for entry in list(filtered_bids_now):
            if str(entry.get("filter_reason", "")).strip().lower() != "criteria_fail":
                continue
            bid = str(entry.get("bid", "") or "").strip().upper()
            if len(bid) < 2 or not bid[0].isdigit():
                continue
            level = int(bid[0])
            strain = bid[1:]
            nb = next_by_bid.get(bid)
            if nb is None:
                continue
            if (
                nb.get("bt_index") is None
                or bool((nb or {}).get("is_dead_end", False))
                or not bool((nb or {}).get("can_complete", True))
            ):
                continue

            if strain in ("C", "D", "H", "S"):
                if strain == last_opp_strain:
                    continue
                if int(sl_map.get(strain, 0) or 0) < 5:
                    continue
                if (tp_val is not None and tp_val < 12.0) and (hcp_val is not None and hcp_val < 10.0):
                    continue
                prev = lowest_suit_bid.get(strain)
                if prev is None or level < int(prev[0]):
                    lowest_suit_bid[strain] = bid
                continue

            if strain == "N":
                if level != 3:
                    continue
                if hcp_val is None or hcp_val < 18.0:
                    continue
                if not _has_stopper_in_suit(
                    last_opp_strain,
                    acting_dir_now,
                    deal_row_now,
                    int(bt_seat_now),
                    dealer_rot_now,
                    int(deal_row_idx_now),
                ):
                    continue
                nb_copy = dict(nb)
                nb_copy["_direct_overcall_gap_rescue"] = True
                nb_copy["_direct_overcall_gap_rescue_reason"] = (
                    f"DIRECT_OVERCALL_GAP_RESCUE: pass-only after opponent natural "
                    f"{last_opp_level}{last_opp_strain}; strong stopper-based 3N restored for scoring"
                )
                promoted.append(nb_copy)
                promoted_names.add(bid)

        for bid in lowest_suit_bid.values():
            nb = next_by_bid.get(bid)
            if nb is None:
                continue
            nb_copy = dict(nb)
            nb_copy["_direct_overcall_gap_rescue"] = True
            nb_copy["_direct_overcall_gap_rescue_reason"] = (
                f"DIRECT_OVERCALL_GAP_RESCUE: pass-only after opponent natural "
                f"{last_opp_level}{last_opp_strain}; restored cheapest natural 5-card "
                f"overcall {bid} for scoring"
            )
            promoted.append(nb_copy)
            promoted_names.add(bid)

        if not promoted:
            return passed_opts_now, filtered_bids_now, None

        new_passed_opts = list(passed_opts_now or []) + promoted
        new_filtered = [
            e for e in list(filtered_bids_now or [])
            if str(e.get("bid", "")).strip().upper() not in promoted_names
        ]
        gate_reason = None
        if (tp_val is not None and tp_val >= 18.0) or (hcp_val is not None and hcp_val >= 17.0):
            new_passed_opts = [
                o for o in list(new_passed_opts or [])
                if str((o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
            ]
            gate_reason = (
                f"DIRECT_OVERCALL_GAP_RESCUE: strong hand after opponent natural "
                f"{last_opp_level}{last_opp_strain}; pass blocked while restored non-pass actions exist"
            )
        return new_passed_opts, new_filtered, gate_reason

    tokens: List[str] = [t.strip().upper() for t in normalize_auction_input(auction_prefix).split("-") if t.strip()] if auction_prefix else []
    # Convention-aware memory across steps (direction -> flag).
    forcing_heart_shown_by_dir: Dict[str, bool] = {}
    takeout_double_shown_by_dir: Dict[str, bool] = {}
    steps_detail: List[Dict[str, Any]] = []

    for _step_i in range(int(max_steps)):
        if _is_auction_complete_list(tokens):
            break

        t_step0 = time.perf_counter()
        seat_display = _next_display_seat_from_tokens(tokens)
        bt_tokens, leading_passes = _strip_leading_passes_tokens(tokens)
        bt_prefix = "-".join(bt_tokens) if bt_tokens else ""
        dealer_rot = _rotate_dealer_by(dealer_actual, leading_passes)
        seat_bt = _bt_seat_from_display_seat(seat_display, leading_passes)

        # Next bids from BT (fast CSR).
        #
        # IMPORTANT performance: we do NOT need expensive per-bid deal counts here.
        # Those counts can cost seconds and dominate pass steps. We only need:
        # - bid, bt_index, agg_expr, dead-end/can-complete flags
        # - precomputed avg_ev/avg_par (cheap lookup)
        auction_input = normalize_auction_input(bt_prefix)
        auction_normalized = re.sub(r"(?i)^(p-)+", "", auction_input) if auction_input else ""
        resp = _handle_list_next_bids_walk_fallback(
            state,
            auction_input,
            auction_normalized,
            time.perf_counter(),
            include_deal_counts=False,
            include_ev_stats=True,
            dealer=dealer_rot,
            vulnerable=board_vul,
        )
        next_bids = resp.get("next_bids", []) or []

        # Ensure Pass exists when permissive.
        if permissive_pass:
            has_p = any(str(x.get("bid", "")).strip().upper() in ("P", "PASS") for x in next_bids)
            if not has_p:
                next_bids = list(next_bids) + [
                    {
                        "bid": "P",
                        "bt_index": None,
                        "expr": [],
                        "agg_expr": [],
                        "can_complete": True,
                        "is_completed_auction": False,
                        "is_dead_end": False,
                        "matching_deal_count": 0,
                        "avg_ev": None,
                        "avg_par": None,
                    }
                ]

        # Candidate filtering (structural only). Track why each bid is dropped.
        candidates: List[Dict[str, Any]] = []
        pass_opt: Dict[str, Any] | None = None
        _bids_struct_filtered: List[Dict[str, Any]] = []
        _blackwood_allowed = _blackwood_aces_response_allowed(tokens, dealer_actual)
        for opt in next_bids:
            bid0 = str(opt.get("bid", "") or "").strip().upper()
            if not bid0:
                continue
            if _blackwood_allowed is not None and bid0 not in _blackwood_allowed:
                _reason = "blackwood_aces_response_required"
                if bid0 in ("5N", "5NT"):
                    _reason = "invalid_blackwood_5nt_response"
                _bids_struct_filtered.append(
                    {
                        "bid": bid0,
                        "agg_expr": opt.get("agg_expr", []) or [],
                        "filter_reason": _reason,
                        "matching_deal_count": opt.get("matching_deal_count"),
                    }
                )
                continue
            # Handle Pass specially: store it for later fallback AND add to candidates to compete in scoring
            if bid0 in ("P", "PASS"):
                pass_opt = opt
                candidates.append(opt)
                continue
            bt_idx = opt.get("bt_index")
            crits = opt.get("agg_expr", []) or []
            is_dead_end = bool(opt.get("is_dead_end", False))
            has_empty_criteria = not crits
            if bt_idx is None:
                _bids_struct_filtered.append({"bid": bid0, "agg_expr": crits, "filter_reason": "no_bt_index"})
                continue
            # WORKAROUND: Invalidate bids with underspecified criteria (missing HCP and Total_Points)
            if _is_underspecified_criteria(crits):
                _bids_struct_filtered.append({"bid": bid0, "agg_expr": crits, "filter_reason": "underspecified", "matching_deal_count": opt.get("matching_deal_count")})
                continue
            # Do NOT reject on can_complete yet; only after pass.
            if is_dead_end:
                _bids_struct_filtered.append({"bid": bid0, "agg_expr": crits, "filter_reason": "dead_end", "matching_deal_count": opt.get("matching_deal_count")})
                continue
            if has_empty_criteria:
                _bids_struct_filtered.append({"bid": bid0, "agg_expr": crits, "filter_reason": "empty_criteria", "matching_deal_count": opt.get("matching_deal_count")})
                continue
            candidates.append(opt)

        step_rec: Dict[str, Any] = {
            "step": int(_step_i + 1),
            "seat": int(seat_display),
            "logic_mode": str(logic_mode_norm),
            "use_guardrails_v2": bool(use_guardrails_v2),
            "bt_prefix": str(bt_prefix),
            "leading_passes": int(leading_passes),
            "dealer_rot": str(dealer_rot),
            "seat_bt": int(seat_bt),
            "opts": int(len(next_bids)),
            "cand": int(len(candidates)),
        }

        # Perspective for this acting seat (used by per-step bidder_view debug fields).
        acting_sign_step = 1.0
        acting_dir_step = "N"
        try:
            directions = ["N", "E", "S", "W"]
            dealer_idx = directions.index(dealer_actual) if dealer_actual in directions else 0
            acting_dir_step = directions[(dealer_idx + int(seat_display) - 1) % 4]
            acting_sign_step = 1.0 if acting_dir_step in ("N", "S") else -1.0
        except Exception:
            acting_sign_step = 1.0
            acting_dir_step = "N"
        acting_side_step = "NS" if acting_dir_step in ("N", "S") else "EW"

        if not candidates:
            # Stayman is forcing for one round: opener must answer 2C/3C inquiry.
            try:
                _stayman_fallback = _stayman_forcing_fallback_bid(tokens)
                if _stayman_fallback is not None:
                    _fallback = None
                    for _o in list(next_bids or []):
                        _b = str((_o or {}).get("bid", "") or "").strip().upper()
                        if _b != _stayman_fallback:
                            continue
                        _bt_i = (_o or {}).get("bt_index")
                        _can_complete = bool((_o or {}).get("can_complete", True))
                        _is_dead_end = bool((_o or {}).get("is_dead_end", False))
                        if _bt_i is not None and _can_complete and (not _is_dead_end):
                            _fallback = _o
                            break
                    if _fallback is not None:
                        tokens.append(str(_stayman_fallback))
                        step_rec["choice"] = str(_stayman_fallback)
                        step_rec["hard_gate_reason"] = (
                            f"STAYMAN_FORCING_GATE: pass blocked; forcing fallback to {_stayman_fallback}"
                        )
                        step_rec["pass"] = 0
                        step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
                        steps_detail.append(step_rec)
                        continue
            except Exception:
                pass
            # Strong 2C opening force: do not allow pass below 2NT; force legal 2D fallback.
            try:
                if _is_strong_2c_forcing_active(tokens, acting_side_step, dealer_actual):
                    _fallback_2d = None
                    for _o in list(next_bids or []):
                        _b = str((_o or {}).get("bid", "") or "").strip().upper()
                        if _b != "2D":
                            continue
                        _bt_i = (_o or {}).get("bt_index")
                        _can_complete = bool((_o or {}).get("can_complete", True))
                        _is_dead_end = bool((_o or {}).get("is_dead_end", False))
                        if _bt_i is not None and _can_complete and (not _is_dead_end):
                            _fallback_2d = _o
                            break
                    if _fallback_2d is not None:
                        tokens.append("2D")
                        step_rec["choice"] = "2D"
                        step_rec["hard_gate_reason"] = (
                            "STRONG_2C_FORCING_TO_2N_GATE: pass blocked below 2NT; forcing fallback to 2D"
                        )
                        step_rec["pass"] = 0
                        step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
                        steps_detail.append(step_rec)
                        continue
            except Exception:
                pass
            # No BT-backed candidates: Pass if possible.
            if pass_opt is None:
                break
            tokens.append("P")
            step_rec["choice"] = "P"
            step_rec["pass"] = 0
            _avg_par_ns = None
            try:
                _ap0 = pass_opt.get("avg_par") if pass_opt is not None else None
                if _ap0 is not None:
                    _avg_par_ns = float(_ap0)
            except Exception:
                _avg_par_ns = None
            step_rec["bidder_view"] = {
                "acting_direction": acting_dir_step,
                "chosen_bid": "P",
                "likely_final_contract": None,
                "likely_par_score_ns": (
                    round(float(acting_sign_step) * float(_avg_par_ns), 2) if _avg_par_ns is not None else None
                ),
                "likely_par_score_acting": round(float(_avg_par_ns), 2) if _avg_par_ns is not None else None,
                "source": "pass_fallback_avg_par",
            }
            step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
            steps_detail.append(step_rec)
            continue

        # Seed ranking parity with standalone/Streamlit path: prefer higher Avg_EV,
        # then matching_deal_count.
        def _seed_key(o: Dict[str, Any]) -> Tuple[float, float, float]:
            def _f(x: Any) -> float:
                try:
                    if x is None or x == "":
                        return float("-inf")
                    return float(x)
                except Exception:
                    return float("-inf")
            return (_f(o.get("avg_ev")), _f(o.get("matching_deal_count")), _f(o.get("matching_deal_count")))

        candidates_sorted = sorted(candidates, key=_seed_key, reverse=True)

        # Pre-apply regex CSV overlay criteria to candidate agg_expr so the main
        # criteria-pass batch enforces them consistently.
        try:
            _overlay_rules = state.get("custom_criteria_overlay") or []
            _regex_overlay = [r for r in (_overlay_rules or []) if str(r.get("partial") or "").startswith("^")]
            if _regex_overlay and candidates_sorted:
                from bbo_bidding_queries_lib import pattern_matches as _pm
                for _opt in candidates_sorted:
                    _bid_o = str((_opt or {}).get("bid", "") or "").strip().upper()
                    if _bid_o in ("", "P", "PASS"):
                        continue
                    _child_auc = f"{bt_prefix}-{_bid_o}" if bt_prefix else _bid_o
                    _child_auc = _child_auc.upper()
                    _agg = [str(x) for x in ((_opt or {}).get("agg_expr") or []) if x is not None]
                    _changed = False
                    for _rr in _regex_overlay:
                        _rr_seat = 0
                        try:
                            _rr_seat = int(_rr.get("seat") or 0)
                        except Exception:
                            _rr_seat = 0
                        if _rr_seat and _rr_seat != int(seat_bt):
                            continue
                        _partial = str(_rr.get("partial") or "").strip().upper()
                        if not _partial or not _pm(_partial, _child_auc):
                            continue
                        _rr_flags = set(
                            str(x).strip().lower()
                            for x in (_rr.get("flags") or set())
                            if x is not None
                        )
                        _rr_crits = [str(x) for x in (_rr.get("criteria") or []) if x is not None]
                        if "replace_criteria" in _rr_flags:
                            _agg = list(_rr_crits)
                            _changed = True
                            continue
                        for _c in _rr_crits:
                            if _c not in _agg:
                                _agg.append(_c)
                                _changed = True
                    if _changed:
                        _opt["agg_expr"] = _agg
        except Exception:
            pass

        # Criteria pass batch for these candidates.
        # Pass always passes criteria check.
        checks = [
            {"seat": int(seat_bt), "criteria": list((o.get("agg_expr") or []))}
            for o in candidates_sorted
            if str(o.get("bid", "")).strip().upper() not in ("P", "PASS")
        ]
        
        passed_opts: List[Dict[str, Any]] = []
        _bids_criteria_filtered: List[Dict[str, Any]] = []
        try:
            pass_resp = handle_deal_criteria_pass_batch(state, int(deal_row_idx), dealer_rot, checks, deal_row_dict=deal_row_dict)
            results = pass_resp.get("results", []) or []

            # Re-merge results with candidates (Pass automatically passes)
            check_idx = 0
            for opt in candidates_sorted:
                bid0 = str(opt.get("bid", "")).strip().upper()
                if bid0 in ("P", "PASS"):
                    passed_opts.append(opt)
                    continue

                # Check result for non-pass
                if check_idx < len(results):
                    r0 = results[check_idx]
                    check_idx += 1
                    if bool((r0 or {}).get("passes", False)):
                        can_complete = opt.get("can_complete")
                        can_complete_b = bool(can_complete) if can_complete is not None else True
                        if can_complete_b:
                            passed_opts.append(opt)
                        else:
                            _bids_criteria_filtered.append({
                                "bid": bid0,
                                "agg_expr": list(opt.get("agg_expr") or []),
                                "filter_reason": "cannot_complete",
                                "matching_deal_count": opt.get("matching_deal_count"),
                            })
                    else:
                        crits_list = list(opt.get("agg_expr") or [])
                        _entry: Dict[str, Any] = {
                            "bid": bid0,
                            "agg_expr": crits_list,
                            "filter_reason": "criteria_fail",
                            "matching_deal_count": opt.get("matching_deal_count"),
                        }
                        _entry["criteria_trace"] = _trace_criteria_failures(
                            crits_list, dealer_rot, seat_bt, deal_row, state, deal_row_dict
                        )
                        _bids_criteria_filtered.append(_entry)
        except Exception:
            # Standalone parity: if criteria batch fails, optimistically keep all
            # candidates in seed-ranked order.
            passed_opts = list(candidates_sorted)
            _bids_criteria_filtered = []

        # Apply regex-pattern CSV overlay rules to passed_opts.
        # Only regex rules (partial starts with '^') act as blocking gates here;
        # non-regex rules are structural BT criteria modifications handled by
        # _apply_overlay_and_dedupe during BT row enrichment.
        try:
            _overlay_rules = state.get("custom_criteria_overlay") or []
            _regex_overlay = [r for r in (_overlay_rules or []) if str(r.get("partial") or "").startswith("^")]
            if _regex_overlay and passed_opts:
                from bbo_bidding_queries_lib import pattern_matches as _pm
                _overlay_kept: list[dict] = []
                for _o in list(passed_opts):
                    _bid_o = str((_o or {}).get("bid", "") or "").strip().upper()
                    if _bid_o in ("P", "PASS", ""):
                        _overlay_kept.append(_o)
                        continue
                    _child_auc = f"{bt_prefix}-{_bid_o}" if bt_prefix else _bid_o
                    _child_auc = _child_auc.upper()
                    _overlay_pass = True
                    for _rr in _regex_overlay:
                        _rr_seat = 0
                        try:
                            _rr_seat = int(_rr.get("seat") or 0)
                        except Exception:
                            _rr_seat = 0
                        if _rr_seat and _rr_seat != int(seat_bt):
                            continue
                        # Seat in overlay stats can drift for regex patterns with optional
                        # pass groups. We gate primarily by regex match on the concrete child
                        # auction and then evaluate criteria for the current acting seat.
                        _partial = str(_rr.get("partial") or "").strip().upper()
                        if not _partial:
                            continue
                        if not _pm(_partial, _child_auc):
                            continue
                        _rr_crits = [str(x) for x in (_rr.get("criteria") or []) if x is not None]
                        if not _rr_crits:
                            continue
                        try:
                            _overlay_check = handle_deal_criteria_pass_batch(
                                state,
                                int(deal_row_idx),
                                dealer_rot,
                                [{"seat": int(seat_bt), "criteria": list(_rr_crits)}],
                                deal_row_dict=deal_row_dict,
                            )
                            _overlay_results = _overlay_check.get("results", []) or []
                            if not _overlay_results or not bool((_overlay_results[0] or {}).get("passes", False)):
                                _overlay_pass = False
                                break
                        except Exception:
                            _overlay_pass = False
                            break
                    if _overlay_pass:
                        _overlay_kept.append(_o)
                    else:
                        _bids_criteria_filtered.append({
                            "bid": _bid_o,
                            "agg_expr": list((_o or {}).get("agg_expr") or []),
                            "filter_reason": "overlay_criteria_fail",
                            "matching_deal_count": (_o or {}).get("matching_deal_count"),
                        })
                passed_opts = _overlay_kept
        except Exception:
            pass

        # Defensive SL gate:
        # Re-check suit-length predicates directly on the pinned deal before scoring.
        # This prevents impossible natural suit bids (e.g., 4S without 4+ spades)
        # from surviving if an upstream batch check ever degrades or is bypassed.
        try:
            _sl_kept: list[dict] = []
            for _o in list(passed_opts or []):
                _bid_o = str((_o or {}).get("bid", "") or "").strip().upper()
                if _bid_o in ("", "P", "PASS"):
                    _sl_kept.append(_o)
                    continue
                _crits = [str(x) for x in ((_o or {}).get("agg_expr") or []) if x is not None]
                _sl_failed = False
                for _c in _crits:
                    _c_s = str(_c)
                    if parse_sl_comparison_relative(_c_s) is None and parse_sl_comparison_numeric(_c_s) is None:
                        continue
                    _sl_res = evaluate_sl_criterion(_c_s, dealer_rot, int(seat_bt), deal_row, fail_on_missing=False)
                    if _sl_res is not True:
                        _sl_failed = True
                        break
                if _sl_failed:
                    _bids_criteria_filtered.append(
                        {
                            "bid": _bid_o,
                            "agg_expr": _crits,
                            "filter_reason": "sl_sanity_fail",
                            "matching_deal_count": (_o or {}).get("matching_deal_count"),
                            "criteria_trace": _trace_criteria_failures(
                                _crits, dealer_rot, int(seat_bt), deal_row, state, deal_row_dict
                            ),
                        }
                    )
                    continue
                _sl_kept.append(_o)
            passed_opts = _sl_kept
        except Exception:
            pass

        # Strong 2C opening force: block pass below 2NT.
        # If non-pass options exist, remove pass. If pass-only, inject legal 2D fallback.
        # Stayman is also forcing for one round (opener must answer 2C/3C).
        try:
            _stayman_fallback = _stayman_forcing_fallback_bid(tokens)
            if _stayman_fallback is not None:
                _non_pass = [
                    _o for _o in list(passed_opts or [])
                    if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                ]
                if _non_pass:
                    if len(_non_pass) != len(list(passed_opts or [])):
                        passed_opts = _non_pass
                        step_rec["hard_gate_reason"] = (
                            "STAYMAN_FORCING_GATE: pass blocked while non-pass continuations exist"
                        )
                else:
                    _fallback = None
                    for _o in list(next_bids or []):
                        _b = str((_o or {}).get("bid", "") or "").strip().upper()
                        if _b != _stayman_fallback:
                            continue
                        _bt_i = (_o or {}).get("bt_index")
                        _can_complete = bool((_o or {}).get("can_complete", True))
                        _is_dead_end = bool((_o or {}).get("is_dead_end", False))
                        if _bt_i is not None and _can_complete and (not _is_dead_end):
                            _fallback = _o
                            break
                    if _fallback is not None:
                        passed_opts = [_fallback]
                        step_rec["hard_gate_reason"] = (
                            f"STAYMAN_FORCING_GATE: pass-only outcome replaced with forced {_stayman_fallback}"
                        )
            if _is_strong_2c_forcing_active(tokens, acting_side_step, dealer_actual):
                _non_pass = [
                    _o for _o in list(passed_opts or [])
                    if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                ]
                if _non_pass:
                    if len(_non_pass) != len(list(passed_opts or [])):
                        passed_opts = _non_pass
                        step_rec["hard_gate_reason"] = (
                            "STRONG_2C_FORCING_TO_2N_GATE: pass blocked below 2NT while non-pass continuations exist"
                        )
                else:
                    _fallback_2d = None
                    for _o in list(next_bids or []):
                        _b = str((_o or {}).get("bid", "") or "").strip().upper()
                        if _b != "2D":
                            continue
                        _bt_i = (_o or {}).get("bt_index")
                        _can_complete = bool((_o or {}).get("can_complete", True))
                        _is_dead_end = bool((_o or {}).get("is_dead_end", False))
                        if _bt_i is not None and _can_complete and (not _is_dead_end):
                            _fallback_2d = _o
                            break
                    if _fallback_2d is not None:
                        passed_opts = [_fallback_2d]
                        step_rec["hard_gate_reason"] = (
                            "STRONG_2C_FORCING_TO_2N_GATE: pass-only outcome replaced with forced 2D"
                        )
        except Exception:
            pass

        # Partner cue-bid force: never allow the second pass after partner's cue-bid.
        # Default action is to return to the fit.
        try:
            _cue_ctx, _fit_suit = _partner_cue_bid_fit_context(tokens, dealer_actual, acting_dir_step)
            if _cue_ctx:
                _has_pass = any(
                    str((_o or {}).get("bid", "") or "").strip().upper() in ("P", "PASS")
                    for _o in list(passed_opts or [])
                )
                if _has_pass:
                    _fit_from_passed = _pick_return_to_fit_bid(_fit_suit, list(passed_opts or []))
                    _fit_from_next = _pick_return_to_fit_bid(_fit_suit, list(next_bids or []))
                    _fit_pick = _fit_from_passed if _fit_from_passed is not None else _fit_from_next

                    if _fit_pick is not None:
                        _fit_bid = str((_fit_pick or {}).get("bid", "") or "").strip().upper()
                        passed_opts = [dict(_fit_pick)]
                        step_rec["hard_gate_reason"] = (
                            f"PARTNER_CUE_BID_NO_SECOND_PASS_GATE: pass blocked; forced return to fit via {_fit_bid}"
                        )
                    else:
                        _non_pass = [
                            _o for _o in list(passed_opts or [])
                            if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                        ]
                        if _non_pass:
                            passed_opts = _non_pass
                            step_rec["hard_gate_reason"] = (
                                "PARTNER_CUE_BID_NO_SECOND_PASS_GATE: pass blocked after partner cue-bid"
                            )
        except Exception:
            pass

        # --- Simple-raise pre-rescue for criteria-filtered bids ----------
        # Bids that failed criteria but qualify as a simple raise of
        # partner's suit at the TP-justified level are promoted into
        # passed_opts so they can compete in the scoring path.
        if use_guardrails_v2 and _bids_criteria_filtered:
            try:
                _sr_dirs = ["N", "E", "S", "W"]
                _sr_dealer_idx = _sr_dirs.index(dealer_actual) if dealer_actual in _sr_dirs else 0
                _sr_acting_side = "NS" if acting_dir_step in ("N", "S") else "EW"
                _sr_partner_dir = _partner_dir(acting_dir_step)

                _sr_partner_strains: set[str] = set()
                _sr_partner_bid_counts: dict[str, int] = {}
                for _sr_i, _sr_tk in enumerate(list(tokens or [])):
                    _sr_tk_u = str(_sr_tk or "").strip().upper()
                    if len(_sr_tk_u) < 2 or not _sr_tk_u[0].isdigit():
                        continue
                    _sr_st = _sr_tk_u[1:]
                    if _sr_st == "NT":
                        continue
                    if _sr_st not in ("C", "D", "H", "S"):
                        continue
                    _sr_d = _sr_dirs[(_sr_dealer_idx + _sr_i) % 4]
                    _sr_d_side = "NS" if _sr_d in ("N", "S") else "EW"
                    if _sr_d_side != _sr_acting_side:
                        continue
                    if _sr_d != _sr_partner_dir:
                        continue
                    _sr_partner_strains.add(_sr_st)
                    _sr_partner_bid_counts[_sr_st] = int(_sr_partner_bid_counts.get(_sr_st, 0) or 0) + 1

                if _sr_partner_strains:
                    _sr_hand = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                    _sr_sl_map: dict[str, int] = {}
                    if _sr_hand and "." in _sr_hand:
                        _sr_parts = _sr_hand.split(".")
                        if len(_sr_parts) == 4:
                            _sr_sl_map = {"S": len(_sr_parts[0]), "H": len(_sr_parts[1]),
                                          "D": len(_sr_parts[2]), "C": len(_sr_parts[3])}
                    _sr_tp: float | None = None
                    try:
                        _sr_tp_raw = deal_row.get(f"Total_Points_{acting_dir_step}")
                        if _sr_tp_raw is not None:
                            _sr_tp = float(_sr_tp_raw)
                    except Exception:
                        pass

                    _sr_tp_ranges: dict[int, tuple[float, float]] = {
                        2: (6.0, 9.0), 3: (10.0, 12.0),
                        4: (13.0, 40.0), 5: (13.0, 40.0),
                        6: (13.0, 40.0), 7: (13.0, 40.0),
                    }
                    _sr_next_bids_by_bid: dict[str, dict] = {
                        str(nb.get("bid", "")).strip().upper(): nb for nb in (next_bids or [])
                    }

                    _sr_promoted: list[str] = []
                    if _sr_tp is not None and _sr_sl_map:
                        for _sr_entry in list(_bids_criteria_filtered):
                            _sr_reason = str(_sr_entry.get("filter_reason", "")).strip().lower()
                            # Respect BT structural blocks: do not rescue cannot-complete bids.
                            if _sr_reason != "criteria_fail":
                                continue
                            _sr_bid = str(_sr_entry.get("bid", "")).strip().upper()
                            if len(_sr_bid) < 2 or not _sr_bid[0].isdigit():
                                continue
                            _sr_bid_level = int(_sr_bid[0])
                            _sr_bid_strain = _sr_bid[1:]
                            if _sr_bid_strain == "NT":
                                continue
                            if _sr_bid_strain not in ("C", "D", "H", "S"):
                                continue
                            if _sr_bid_strain not in _sr_partner_strains:
                                continue
                            _sr_support = int(_sr_sl_map.get(_sr_bid_strain, 0) or 0)
                            # Default to known 8+ fit (4+ support), but allow 3-card
                            # raises when partner has rebid the strain (typically 5+).
                            _sr_min_support = 3 if int(_sr_partner_bid_counts.get(_sr_bid_strain, 0) or 0) >= 2 else 4
                            if _sr_support < _sr_min_support:
                                continue
                            _sr_rng = _sr_tp_ranges.get(_sr_bid_level)
                            if _sr_rng is None or not (_sr_rng[0] <= _sr_tp <= _sr_rng[1]):
                                continue
                            _sr_nb = _sr_next_bids_by_bid.get(_sr_bid)
                            if _sr_nb is None:
                                continue
                            if bool((_sr_nb or {}).get("is_dead_end", False)):
                                continue
                            if not bool((_sr_nb or {}).get("can_complete", True)):
                                continue
                            _sr_nb_copy = dict(_sr_nb)
                            _sr_nb_copy["_simple_raise_pre_rescue"] = True
                            _sr_nb_copy["_simple_raise_pre_rescue_reason"] = (
                                f"SIMPLE_RAISE_PRE_RESCUE: partner bid {_sr_bid_strain}, "
                                f"self has {_sr_support} {_sr_bid_strain} "
                                f"and TP={_sr_tp:.0f} "
                                f"(range {_sr_rng[0]:.0f}-{_sr_rng[1]:.0f} for level {_sr_bid_level})"
                            )
                            passed_opts.append(_sr_nb_copy)
                            _sr_promoted.append(_sr_bid)

                    if _sr_promoted:
                        _bids_criteria_filtered = [
                            e for e in _bids_criteria_filtered
                            if str(e.get("bid", "")).strip().upper() not in _sr_promoted
                        ]
            except Exception:
                pass

        # --- Natural response rescue for opener-response dead ends ----------
        # When partner opens at the one level, RHO passes, and BT criteria leave
        # responder with only Pass, re-admit obvious natural new-suit responses
        # so scoring can choose them and existing non-pass pass-block logic can
        # suppress the second pass with constructive values.
        if use_guardrails_v2 and deal_row and _bids_criteria_filtered:
            try:
                _orr_has_non_pass = any(
                    str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                    for _o in list(passed_opts or [])
                )
                if not _orr_has_non_pass:
                    _orr_ctx = extract_second_pass_opening_context(
                        auction_tokens=list(tokens or []),
                        acting_direction=acting_dir_step,
                        dealer_actual=dealer_actual,
                    )
                    _orr_opening = str((_orr_ctx or {}).get("opening_bid") or "")
                    _orr_opening_match = re.match(r"^1([CDHS])$", _orr_opening)
                    if _orr_ctx is not None and _orr_opening_match is not None:
                        _orr_opening_strain = str(_orr_opening_match.group(1))
                        _orr_strain_rank = {"C": 0, "D": 1, "H": 2, "S": 3}
                        _orr_tp: float | None = None
                        try:
                            _orr_tp_raw = deal_row.get(f"Total_Points_{acting_dir_step}")
                            if _orr_tp_raw is not None:
                                _orr_tp = float(_orr_tp_raw)
                        except Exception:
                            pass
                        _orr_hand = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                        _orr_sl_map: dict[str, int] = {}
                        if _orr_hand and "." in _orr_hand:
                            _orr_parts = _orr_hand.split(".")
                            if len(_orr_parts) == 4:
                                _orr_sl_map = {
                                    "S": len(_orr_parts[0]),
                                    "H": len(_orr_parts[1]),
                                    "D": len(_orr_parts[2]),
                                    "C": len(_orr_parts[3]),
                                }
                        if _orr_tp is not None and _orr_sl_map:
                            _orr_next_bids_by_bid: dict[str, dict] = {
                                str(nb.get("bid", "")).strip().upper(): nb for nb in (next_bids or [])
                            }
                            _orr_promoted: list[str] = []
                            for _orr_entry in list(_bids_criteria_filtered):
                                if str(_orr_entry.get("filter_reason", "")).strip().lower() != "criteria_fail":
                                    continue
                                _orr_bid = str(_orr_entry.get("bid", "")).strip().upper()
                                _orr_bid_match = re.match(r"^([12])([CDHS])$", _orr_bid)
                                if _orr_bid_match is None:
                                    continue
                                _orr_bid_level = int(_orr_bid_match.group(1))
                                _orr_bid_strain = str(_orr_bid_match.group(2))
                                if _orr_bid_strain == _orr_opening_strain:
                                    continue
                                _orr_support = int(_orr_sl_map.get(_orr_bid_strain, 0) or 0)
                                if _orr_support < 4:
                                    continue
                                _orr_expected_level = (
                                    1
                                    if _orr_strain_rank.get(_orr_bid_strain, -1) > _orr_strain_rank.get(_orr_opening_strain, -1)
                                    else 2
                                )
                                if _orr_bid_level != _orr_expected_level:
                                    continue
                                _orr_min_tp = 6.0 if _orr_bid_level == 1 else 12.0
                                if _orr_tp < _orr_min_tp:
                                    continue
                                _orr_nb = _orr_next_bids_by_bid.get(_orr_bid)
                                if _orr_nb is None:
                                    continue
                                if bool((_orr_nb or {}).get("is_dead_end", False)):
                                    continue
                                if not bool((_orr_nb or {}).get("can_complete", True)):
                                    continue
                                _orr_nb_copy = dict(_orr_nb)
                                _orr_nb_copy["_opening_response_new_suit_rescue"] = True
                                _orr_nb_copy["_opening_response_new_suit_rescue_reason"] = (
                                    f"OPENING_RESPONSE_NEW_SUIT_RESCUE: partner opened {_orr_opening}, "
                                    f"RHO passed, self has {_orr_support} {_orr_bid_strain} and "
                                    f"TP={_orr_tp:.0f}; injecting natural response {_orr_bid}"
                                )
                                passed_opts.append(_orr_nb_copy)
                                _orr_promoted.append(_orr_bid)
                            if _orr_promoted:
                                _bids_criteria_filtered = [
                                    e for e in _bids_criteria_filtered
                                    if str(e.get("bid", "")).strip().upper() not in _orr_promoted
                                ]
                                step_rec["opening_response_new_suit_rescue"] = list(_orr_promoted)
            except Exception:
                pass

        # --- NT raise rescue: partner opened 1NT/2NT, responder has enough
        # HCP for game but the BT's 3N criteria are too strict for this hand
        # (e.g. singleton blocks SL_H >= 2).  Inject 3N (or 4N) into the
        # candidate pool so it competes in normal scoring.  Fires even if
        # other (likely penalised) candidates survived, because the real
        # problem is 3N being wrongly excluded.
        if use_guardrails_v2 and deal_row:
            try:
                _ntr_already_has_nt = any(
                    str((_o or {}).get("bid", "") or "").strip().upper() in ("2N", "3N", "3NT", "4N", "4NT")
                    for _o in list(passed_opts or [])
                )
                if not _ntr_already_has_nt:
                    _ntr_partner_nt_level: int | None = None
                    _ntr_partner_dir = _partner_dir(acting_dir_step)
                    for _ti_ntr in range(len(tokens)):
                        _tk_ntr = str(tokens[_ti_ntr] or "").strip().upper()
                        _tk_dir_ntr = _token_bidder_dir_for_dealer(_ti_ntr, dealer_actual)
                        if _tk_dir_ntr != _ntr_partner_dir:
                            continue
                        _m_ntr = re.match(r"^([12])N(?:T)?$", _tk_ntr)
                        if _m_ntr:
                            _ntr_partner_nt_level = int(_m_ntr.group(1))
                    if _ntr_partner_nt_level is not None:
                        _ntr_self_hcp: float | None = None
                        try:
                            _v = deal_row.get(f"HCP_{acting_dir_step}")
                            if _v is not None:
                                _ntr_self_hcp = float(_v)
                        except Exception:
                            pass
                        if _ntr_self_hcp is not None and _ntr_self_hcp >= 10:
                            _ntr_target = "3N"
                            if _ntr_self_hcp >= 15:
                                _ntr_target = "4N"
                            _ntr_next_bids_by_bid: dict[str, dict] = {
                                str(nb.get("bid", "")).strip().upper(): nb for nb in (next_bids or [])
                            }
                            _ntr_nb = _ntr_next_bids_by_bid.get(_ntr_target)
                            if _ntr_nb is None and _ntr_target == "4N":
                                _ntr_nb = _ntr_next_bids_by_bid.get("3N")
                                _ntr_target = "3N"
                            if _ntr_nb is not None:
                                # Respect regex CSV overlay rules (e.g. 1N-P-3N no 4-card major)
                                # before injecting NT rescue candidates.
                                _ntr_allowed = True
                                try:
                                    from bbo_bidding_queries_lib import pattern_matches as _pm
                                    _overlay_rules = state.get("custom_criteria_overlay") or []
                                    _regex_overlay = [
                                        r for r in (_overlay_rules or [])
                                        if str(r.get("partial") or "").startswith("^")
                                    ]
                                    _child_auc_ntr = f"{bt_prefix}-{_ntr_target}" if bt_prefix else _ntr_target
                                    _child_auc_ntr = _child_auc_ntr.upper()
                                    for _rr in _regex_overlay:
                                        _rr_seat = 0
                                        try:
                                            _rr_seat = int(_rr.get("seat") or 0)
                                        except Exception:
                                            _rr_seat = 0
                                        if _rr_seat and _rr_seat != int(seat_bt):
                                            continue
                                        _partial = str(_rr.get("partial") or "").strip().upper()
                                        if not _partial or not _pm(_partial, _child_auc_ntr):
                                            continue
                                        _rr_crits = [str(x) for x in (_rr.get("criteria") or []) if x is not None]
                                        if not _rr_crits:
                                            continue
                                        _overlay_check = handle_deal_criteria_pass_batch(
                                            state,
                                            int(deal_row_idx),
                                            dealer_rot,
                                            [{"seat": int(seat_bt), "criteria": list(_rr_crits)}],
                                            deal_row_dict=deal_row_dict,
                                        )
                                        _overlay_results = _overlay_check.get("results", []) or []
                                        if not _overlay_results or not bool((_overlay_results[0] or {}).get("passes", False)):
                                            _ntr_allowed = False
                                            step_rec["nt_raise_rescue_blocked_reason"] = (
                                                f"NT_RAISE_RESCUE_BLOCKED_BY_OVERLAY:{_partial}"
                                            )
                                            break
                                except Exception:
                                    pass
                                if _ntr_allowed:
                                    _ntr_nb_copy = dict(_ntr_nb)
                                    _ntr_nb_copy["_nt_raise_rescue"] = True
                                    _ntr_nb_copy["_nt_raise_rescue_reason"] = (
                                        f"NT_RAISE_RESCUE: partner opened {_ntr_partner_nt_level}N, "
                                        f"self HCP={_ntr_self_hcp:.0f}, BT 3N criteria too strict; "
                                        f"injecting {_ntr_target}"
                                    )
                                    passed_opts.append(_ntr_nb_copy)
                                    _bids_criteria_filtered = [
                                        e for e in _bids_criteria_filtered
                                        if str(e.get("bid", "")).strip().upper() != _ntr_target
                                    ]
            except Exception:
                pass

        # --- Direct-overcall gap rescue -----------------------------------
        # BT can under-specify strong direct overcalls and leave only Pass
        # after criteria filtering. In that case, re-admit sane natural bids
        # so scoring can choose the least-worst action instead of auto-passing.
        if use_guardrails_v2 and deal_row and _bids_criteria_filtered:
            try:
                passed_opts, _bids_criteria_filtered, _dor_gate_reason = _apply_direct_overcall_gap_rescue(
                    passed_opts_now=passed_opts,
                    filtered_bids_now=_bids_criteria_filtered,
                    next_bids_now=next_bids,
                    tokens_now=tokens,
                    acting_side_now=acting_side_step,
                    acting_dir_now=acting_dir_step,
                    deal_row_now=deal_row,
                    bt_seat_now=int(seat_bt),
                    dealer_rot_now=dealer_rot,
                    dealer_actual_now=dealer_actual,
                    deal_row_idx_now=int(deal_row_idx),
                )
                if _dor_gate_reason:
                    step_rec["hard_gate_reason"] = _dor_gate_reason
            except Exception:
                pass

        step_rec["pass"] = int(len(passed_opts))
        step_rec["all_bids_filtered"] = _bids_struct_filtered + _bids_criteria_filtered

        # Narrow fallback for strong-2C response context:
        # If the only passing option is Pass after 1N-P-P-2C-P (runtime seat-1 view
        # can appear as bt_prefix=2C-P with one leading pass), force 2D when legal.
        try:
            _prefix_now = str(bt_prefix or "").strip().upper()
            _tokens_now = "-".join([str(t or "").strip().upper() for t in list(tokens or []) if str(t or "").strip()])
            _is_capp_response_context = bool(
                _prefix_now == "1N-P-P-2C-P"
                or (_prefix_now == "2C-P" and int(leading_passes) == 1 and _tokens_now == "P-2C-P")
            )
            if _is_capp_response_context:
                _passed_non_pass = [
                    _o for _o in list(passed_opts or [])
                    if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                ]
                if len(_passed_non_pass) == 0:
                    _fallback_2d = None
                    for _o in list(next_bids or []):
                        _b = str((_o or {}).get("bid", "") or "").strip().upper()
                        if _b != "2D":
                            continue
                        _bt_i = (_o or {}).get("bt_index")
                        _can_complete = bool((_o or {}).get("can_complete", True))
                        _is_dead_end = bool((_o or {}).get("is_dead_end", False))
                        if _bt_i is not None and _can_complete and (not _is_dead_end):
                            _fallback_2d = _o
                            break
                    if _fallback_2d is not None:
                        passed_opts = [_fallback_2d]
                        step_rec["hard_gate_reason"] = (
                            "STRONG_2C_RESPONSE_NO_PASS_GATE: pass-only outcome replaced with forced 2D"
                        )
        except Exception:
            pass

        # Forced-preference gate: when partner has shown two suits and
        # no fit has been agreed, responder is obligated to give
        # preference back to partner's first suit even if the bid
        # violates level-cap or guard checks.
        # Fires when the ONLY non-pass bids in passed_opts are in
        # partner's first suit (i.e. the preference bid itself), or
        # when no non-pass bids passed criteria at all.
        # Does NOT apply when:
        #   - Responder has non-pass bids in OTHER suits (forward-going).
        #   - A fit is already agreed (both sides bid the same suit).
        #   - Responder holds strictly more cards in the second suit.
        # If a preference bid exists in passed_opts, mark it exempt.
        # If none passed criteria, inject from next_bids.
        try:
            _partner_pref = _partner_dir(acting_dir_step)
            _partner_suits_ordered: list[str] = []
            _self_suits: set[str] = set()
            for _i_pref, _tk_pref in enumerate(list(tokens or [])):
                _tk_str = str(_tk_pref or "").strip().upper()
                if len(_tk_str) < 2 or not _tk_str[0].isdigit():
                    continue
                _bidder = _token_bidder_dir_for_dealer(_i_pref, dealer_actual)
                _st_pref = _tk_str[1:]
                if _st_pref == "NT":
                    _st_pref = "N"
                if _st_pref not in ("C", "D", "H", "S"):
                    continue
                if _bidder == _partner_pref:
                    if _st_pref not in _partner_suits_ordered:
                        _partner_suits_ordered.append(_st_pref)
                elif _bidder == acting_dir_step:
                    _self_suits.add(_st_pref)
            _fit_overlap = _self_suits & set(_partner_suits_ordered)
            _fit_agreed = bool(_fit_overlap)
            if len(_partner_suits_ordered) >= 2 and not _fit_agreed:
                _first_suit = _partner_suits_ordered[0]
                _fp_sl_map_check: dict[str, int] = {}
                _fp_has_real_fit = False
                if deal_row:
                    _fp_hand_pbn = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                    if _fp_hand_pbn and "." in _fp_hand_pbn:
                        try:
                            _fp_hp = _fp_hand_pbn.split(".")
                            if len(_fp_hp) == 4:
                                _fp_sl_map_check = {"S": len(_fp_hp[0]), "H": len(_fp_hp[1]), "D": len(_fp_hp[2]), "C": len(_fp_hp[3])}
                                if _fp_sl_map_check.get(_first_suit, 0) >= 4:
                                    _fp_has_real_fit = True
                        except Exception:
                            pass
                _passed_non_pass_fp = [
                    _o for _o in list(passed_opts or [])
                    if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                ]
                def _bid_strain(b: str) -> str:
                    b = b.strip().upper()
                    return b[1:] if len(b) >= 2 and b[0].isdigit() else ""
                _non_pref_forward = [
                    _o for _o in _passed_non_pass_fp
                    if _bid_strain(str((_o or {}).get("bid", "") or "")) != _first_suit
                ]
                if len(_non_pref_forward) == 0 and not _fp_has_real_fit:
                    _second_suit = _partner_suits_ordered[1]
                    _sl_first: int | None = None
                    _sl_second: int | None = None
                    if deal_row:
                        _hand_pbn = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                        if _hand_pbn and "." in _hand_pbn:
                            try:
                                _hp = _hand_pbn.split(".")
                                if len(_hp) == 4:
                                    _sl_map = {"S": len(_hp[0]), "H": len(_hp[1]), "D": len(_hp[2]), "C": len(_hp[3])}
                                    _sl_first = _sl_map.get(_first_suit)
                                    _sl_second = _sl_map.get(_second_suit)
                            except Exception:
                                pass
                    _must_prefer_first = bool(
                        _sl_first is not None
                        and _sl_second is not None
                        and _sl_first >= _sl_second
                    )
                    if _must_prefer_first:
                        _pref_in_passed: Dict[str, Any] | None = None
                        for _o in list(passed_opts or []):
                            _b = str((_o or {}).get("bid", "") or "").strip().upper()
                            if len(_b) < 2 or not _b[0].isdigit():
                                continue
                            _b_st = _b[1:]
                            if _b_st == "NT":
                                _b_st = "N"
                            if _b_st == _first_suit:
                                _pref_in_passed = _o
                                break
                        if _pref_in_passed is not None:
                            _pref_in_passed["_forced_preference"] = True
                            _pref_bid_name = str((_pref_in_passed or {}).get("bid", "") or "").strip().upper()
                            passed_opts = [_pref_in_passed]
                            step_rec["hard_gate_reason"] = (
                                f"FORCED_PREFERENCE: partner showed {'/'.join(_partner_suits_ordered)}, "
                                f"self has {_sl_first}{_first_suit} vs {_sl_second}{_second_suit}, "
                                f"{_pref_bid_name} overrides level-cap/guard checks"
                            )
                        else:
                            _pref_bid_nb = None
                            _pref_bid_nb_fallback = None
                            for _o in list(next_bids or []):
                                _b = str((_o or {}).get("bid", "") or "").strip().upper()
                                if len(_b) < 2 or not _b[0].isdigit():
                                    continue
                                _b_st = _b[1:]
                                if _b_st == "NT":
                                    _b_st = "N"
                                if _b_st != _first_suit:
                                    continue
                                _bt_i = (_o or {}).get("bt_index")
                                _can_complete = bool((_o or {}).get("can_complete", True))
                                _is_dead_end = bool((_o or {}).get("is_dead_end", False))
                                if _bt_i is not None and _can_complete and (not _is_dead_end):
                                    _pref_bid_nb = _o
                                    break
                                if _pref_bid_nb_fallback is None and _bt_i is not None:
                                    _pref_bid_nb_fallback = _o
                            if _pref_bid_nb is None:
                                _pref_bid_nb = _pref_bid_nb_fallback
                            if _pref_bid_nb is not None:
                                _pref_bid_nb["_forced_preference"] = True
                                _pref_bid_name = str((_pref_bid_nb or {}).get("bid", "") or "").strip().upper()
                                passed_opts = [_pref_bid_nb]
                                step_rec["hard_gate_reason"] = (
                                    f"FORCED_PREFERENCE: partner showed {'/'.join(_partner_suits_ordered)}, "
                                    f"self has {_sl_first}{_first_suit} vs {_sl_second}{_second_suit}, "
                                    f"injected {_pref_bid_name} overrides level-cap/guard checks"
                                )
        except Exception:
            pass

        # Delayed support raise gate: when the responder previously
        # temporized (e.g. bid a new suit) but actually has 4+ cards in
        # opener's first suit, and the only surviving bid in that suit is
        # at an inappropriately low level given the hand strength, inject
        # the correct raise level.  This covers the "delayed game raise"
        # and "delayed limit raise" patterns.
        #
        # Guard conditions (all must hold):
        #   1. The bidder must NOT be the one who first bid the suit
        #      (opener rebidding own suit is not "delayed support").
        #   2. The bidder must have previously made at least one non-pass
        #      bid (the temporizing bid — e.g. 1S over 1H).
        #   3. The target raise must be above the current auction level
        #      (no illegal bids).
        #   4. The gate fires at most once per side (no looping).
        try:
            _dsr_fired = False
            if deal_row and len(tokens) >= 5 and not step_rec.get("hard_gate_reason"):
                _dsr_partner_dir = _partner_dir(acting_dir_step)
                _dsr_partner_first_suit: str | None = None
                _dsr_partner_suit_token_idx: int = -1
                _dsr_self_already_bid = False
                _dsr_self_bid_is_suit_opener = False
                _dsr_current_auction_level = 0
                _dsr_suit_rank = {"C": 0, "D": 1, "H": 2, "S": 3}
                _dsr_current_auction_strain_rank = -1
                _dsr_self_suits: list[str] = []
                for _ti_dsr in range(len(tokens)):
                    _tk_dir_dsr = _token_bidder_dir_for_dealer(_ti_dsr, dealer_actual)
                    _tk_u_dsr = str(tokens[_ti_dsr] or "").strip().upper()
                    if _tk_u_dsr in ("P", "PASS", "", "D", "X", "DOUBLE", "R", "XX", "REDOUBLE"):
                        continue
                    if len(_tk_u_dsr) >= 2 and _tk_u_dsr[0].isdigit():
                        _tl = int(_tk_u_dsr[0])
                        _ts = _tk_u_dsr[1:]
                        _ts_r = _dsr_suit_rank.get(_ts, 4)
                        if _tl > _dsr_current_auction_level or (_tl == _dsr_current_auction_level and _ts_r > _dsr_current_auction_strain_rank):
                            _dsr_current_auction_level = _tl
                            _dsr_current_auction_strain_rank = _ts_r
                    if _tk_dir_dsr == _dsr_partner_dir and _dsr_partner_first_suit is None:
                        if len(_tk_u_dsr) >= 2 and _tk_u_dsr[0].isdigit() and _tk_u_dsr[1:] in ("C", "D", "H", "S"):
                            _dsr_partner_first_suit = _tk_u_dsr[1:]
                            _dsr_partner_suit_token_idx = _ti_dsr
                    if _tk_dir_dsr == acting_dir_step:
                        if len(_tk_u_dsr) >= 2 and _tk_u_dsr[0].isdigit() and _tk_u_dsr[1:] in ("C", "D", "H", "S"):
                            _dsr_self_suits.append(_tk_u_dsr[1:])
                        if not _dsr_self_already_bid:
                            _dsr_self_already_bid = True
                            if len(_tk_u_dsr) >= 2 and _tk_u_dsr[0].isdigit() and _tk_u_dsr[1:] in ("C", "D", "H", "S"):
                                if _dsr_partner_first_suit and _tk_u_dsr[1:] == _dsr_partner_first_suit:
                                    _dsr_self_bid_is_suit_opener = True
                if _dsr_partner_first_suit and not _dsr_self_bid_is_suit_opener:
                    if _dsr_partner_first_suit in _dsr_self_suits:
                        _dsr_self_bid_is_suit_opener = True
                _dsr_partner_bid_is_artificial = False
                if _dsr_partner_first_suit and _dsr_partner_suit_token_idx >= 0:
                    _dsr_prefix_at_partner = "-".join(
                        str(tokens[i] or "").strip().upper()
                        for i in range(_dsr_partner_suit_token_idx + 1)
                    )
                    _dsr_prefix_s1 = re.sub(r"(?i)^(P-)+", "", _dsr_prefix_at_partner)
                    _dsr_overlay = state.get("custom_criteria_overlay") or []
                    from bbo_bidding_queries_lib import pattern_matches as _dsr_pm
                    for _dsr_rule in _dsr_overlay:
                        if "is_artificial" not in (_dsr_rule.get("flags") or set()):
                            continue
                        if _dsr_pm(str(_dsr_rule.get("partial", "")), _dsr_prefix_s1):
                            _dsr_partner_bid_is_artificial = True
                            break
                _dsr_higher_suit_agreed = False
                if _dsr_partner_first_suit:
                    _dsr_pfs_rank = _dsr_suit_rank.get(_dsr_partner_first_suit, 0)
                    _dsr_acting_side_ns = acting_dir_step.upper() in ("N", "S")
                    _dsr_my_side_higher_strains: set[str] = set()
                    for _ti_ag in range(len(tokens)):
                        _tk_ag = str(tokens[_ti_ag] or "").strip().upper()
                        if len(_tk_ag) < 2 or not _tk_ag[0].isdigit():
                            continue
                        _ag_strain = _tk_ag[1:]
                        if _ag_strain not in ("C", "D", "H", "S"):
                            continue
                        if _dsr_suit_rank.get(_ag_strain, 0) <= _dsr_pfs_rank:
                            continue
                        _ag_dir = _token_bidder_dir_for_dealer(_ti_ag, dealer_actual)
                        _ag_is_ns = _ag_dir.upper() in ("N", "S")
                        if _ag_is_ns == _dsr_acting_side_ns:
                            _dsr_my_side_higher_strains.add(_ag_strain)
                    for _ag_s in _dsr_my_side_higher_strains:
                        _ag_bidders: set[str] = set()
                        for _ti_ag2 in range(len(tokens)):
                            _tk_ag2 = str(tokens[_ti_ag2] or "").strip().upper()
                            if len(_tk_ag2) >= 2 and _tk_ag2[0].isdigit() and _tk_ag2[1:] == _ag_s:
                                _ag_d2 = _token_bidder_dir_for_dealer(_ti_ag2, dealer_actual)
                                if (_ag_d2.upper() in ("N", "S")) == _dsr_acting_side_ns:
                                    _ag_bidders.add(_ag_d2)
                        if len(_ag_bidders) >= 2:
                            _dsr_higher_suit_agreed = True
                            break

                if (
                    _dsr_partner_first_suit
                    and _dsr_self_already_bid
                    and not _dsr_self_bid_is_suit_opener
                    and not _dsr_partner_bid_is_artificial
                    and not _dsr_higher_suit_agreed
                ):
                    _dsr_hand_pbn = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                    _dsr_sl_map: dict[str, int] = {}
                    if _dsr_hand_pbn and "." in _dsr_hand_pbn:
                        _dsr_hp = _dsr_hand_pbn.split(".")
                        if len(_dsr_hp) == 4:
                            _dsr_sl_map = {"S": len(_dsr_hp[0]), "H": len(_dsr_hp[1]), "D": len(_dsr_hp[2]), "C": len(_dsr_hp[3])}
                    _dsr_support = _dsr_sl_map.get(_dsr_partner_first_suit, 0)
                    if _dsr_support >= 4:
                        _dsr_self_tp = float(deal_row.get("Total_Points_" + acting_dir_step, 0) or 0)
                        _dsr_is_major = _dsr_partner_first_suit in ("H", "S")
                        _dsr_target_level = 0
                        if _dsr_is_major:
                            if _dsr_self_tp >= 13:
                                _dsr_target_level = 4
                            elif _dsr_self_tp >= 10:
                                _dsr_target_level = 3
                        else:
                            if _dsr_self_tp >= 15:
                                _dsr_target_level = 5
                            elif _dsr_self_tp >= 12:
                                _dsr_target_level = 4
                            elif _dsr_self_tp >= 10:
                                _dsr_target_level = 3
                        _dsr_target_strain_rank = _dsr_suit_rank.get(_dsr_partner_first_suit, 0)
                        _dsr_target_legal = (
                            _dsr_target_level > _dsr_current_auction_level
                            or (
                                _dsr_target_level == _dsr_current_auction_level
                                and _dsr_target_strain_rank > _dsr_current_auction_strain_rank
                            )
                        )
                        _dsr_best_passed: Dict[str, Any] | None = None
                        _dsr_best_level = 0
                        for _o in list(passed_opts or []):
                            _b = str((_o or {}).get("bid", "") or "").strip().upper()
                            if len(_b) < 2 or not _b[0].isdigit():
                                continue
                            _b_st = _b[1:]
                            if _b_st == _dsr_partner_first_suit:
                                _bl = int(_b[0])
                                if _bl > _dsr_best_level:
                                    _dsr_best_level = _bl
                                    _dsr_best_passed = _o
                        if _dsr_target_level > 0 and _dsr_target_level > _dsr_best_level and _dsr_target_legal:
                            _dsr_target_bid = f"{_dsr_target_level}{_dsr_partner_first_suit}"
                            _dsr_opt: Dict[str, Any] | None = None
                            for _o in list(next_bids or []):
                                _ub = str((_o or {}).get("bid", "") or "").strip().upper()
                                if _ub == _dsr_target_bid:
                                    _dsr_opt = _o
                                    break
                            if _dsr_opt is None:
                                _dsr_opt = {
                                    "bid": _dsr_target_bid,
                                    "bt_index": None,
                                    "expr": [],
                                    "agg_expr": [],
                                    "can_complete": True,
                                    "is_completed_auction": False,
                                    "is_dead_end": False,
                                    "matching_deal_count": 0,
                                    "avg_ev": None,
                                    "avg_par": None,
                                }
                            _dsr_opt["_delayed_support_raise"] = True
                            passed_opts = [_dsr_opt]
                            step_rec["hard_gate_reason"] = (
                                f"DELAYED_SUPPORT_RAISE: {_dsr_support}-card support in "
                                f"partner's {_dsr_partner_first_suit}, self_tp={_dsr_self_tp:.0f}, "
                                f"raise to {_dsr_target_bid} "
                                f"(was {_dsr_best_level}{_dsr_partner_first_suit if _dsr_best_level > 0 else 'none'})"
                            )
                            _dsr_fired = True
        except Exception:
            pass

        # New-suit-at-1-level forcing gate: a new suit by responder at the
        # 1-level is universally forcing for one round.  The opener MUST NOT
        # pass.  When only Pass survived *or* when the only non-Pass options
        # are at level >= 4 (which will be hard-blocked by the level cap),
        # inject the best available rebid.
        # Exception: a raise to 4 of partner's major is game — treat as viable.
        try:
            _nsf_partner_last_strain: str | None = None
            if len(tokens) >= 3 and deal_row:
                _nsf_pdir_pre = _partner_dir(acting_dir_step)
                for _ti_pre in range(len(tokens) - 1, -1, -1):
                    _tk_pre = str(tokens[_ti_pre] or "").strip().upper()
                    if _token_bidder_dir_for_dealer(_ti_pre, dealer_actual) == _nsf_pdir_pre and _tk_pre not in ("P", "PASS", ""):
                        if len(_tk_pre) >= 2 and _tk_pre[0].isdigit():
                            _nsf_partner_last_strain = _tk_pre[1:]
                        break
            _nsf_no_viable_nonpass = True
            for _o in list(passed_opts or []):
                _ob = str((_o or {}).get("bid", "") or "").strip().upper()
                if _ob in ("", "P", "PASS"):
                    continue
                if len(_ob) >= 2 and _ob[0].isdigit():
                    _ob_lvl = int(_ob[0])
                    _ob_st = _ob[1:]
                    if _ob_lvl <= 3:
                        _nsf_no_viable_nonpass = False
                        break
                    if _ob_lvl == 4 and _ob_st in ("H", "S") and _ob_st == _nsf_partner_last_strain:
                        _nsf_no_viable_nonpass = False
                        break
            if _nsf_no_viable_nonpass and len(tokens) >= 3 and deal_row:
                _nsf_partner_dir = _partner_dir(acting_dir_step)
                _nsf_partner_tok_idx: int | None = None
                for _ti in range(len(tokens) - 1, -1, -1):
                    _tk_dir = _token_bidder_dir_for_dealer(_ti, dealer_actual)
                    _tk_u = str(tokens[_ti] or "").strip().upper()
                    if _tk_dir == _nsf_partner_dir and _tk_u not in ("P", "PASS", ""):
                        _nsf_partner_tok_idx = _ti
                        break
                _nsf_fire = False
                _nsf_partner_strain: str | None = None
                _nsf_self_opening_strain: str | None = None
                if _nsf_partner_tok_idx is not None:
                    _nsf_ptk = str(tokens[_nsf_partner_tok_idx] or "").strip().upper()
                    if len(_nsf_ptk) == 2 and _nsf_ptk[0] == "1" and _nsf_ptk[1] in ("C", "D", "H", "S"):
                        _nsf_partner_strain = _nsf_ptk[1]
                        for _ti2 in range(_nsf_partner_tok_idx):
                            _tk_dir2 = _token_bidder_dir_for_dealer(_ti2, dealer_actual)
                            _tk_u2 = str(tokens[_ti2] or "").strip().upper()
                            if _tk_dir2 == acting_dir_step and _tk_u2 not in ("P", "PASS", ""):
                                if len(_tk_u2) == 2 and _tk_u2[0] == "1" and _tk_u2[1] in ("C", "D", "H", "S"):
                                    _nsf_self_opening_strain = _tk_u2[1]
                                break
                        if _nsf_self_opening_strain and _nsf_partner_strain != _nsf_self_opening_strain:
                            _nsf_fire = True
                            # Forcing obligation is fulfilled once self has
                            # made any non-pass bid after partner's call.
                            for _ti3 in range(_nsf_partner_tok_idx + 1, len(tokens)):
                                if _token_bidder_dir_for_dealer(_ti3, dealer_actual) == acting_dir_step:
                                    _tk3 = str(tokens[_ti3] or "").strip().upper()
                                    if _tk3 not in ("P", "PASS", ""):
                                        _nsf_fire = False
                                        break
                if _nsf_fire:
                    _nsf_hand_pbn = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                    _nsf_sl_map: dict[str, int] = {}
                    if _nsf_hand_pbn and "." in _nsf_hand_pbn:
                        _hp = _nsf_hand_pbn.split(".")
                        if len(_hp) == 4:
                            _nsf_sl_map = {"S": len(_hp[0]), "H": len(_hp[1]), "D": len(_hp[2]), "C": len(_hp[3])}
                    _nsf_open_st = str(_nsf_self_opening_strain or "")
                    _nsf_second_suit: str | None = None
                    _nsf_second_len = 0
                    for _st, _sl in sorted(_nsf_sl_map.items(), key=lambda x: -x[1]):
                        if _st != _nsf_open_st:
                            _nsf_second_suit = _st
                            _nsf_second_len = _sl
                            break

                    # Get actual TP to avoid injecting bids whose criteria
                    # range is wildly incompatible with the hand.
                    _nsf_self_tp: float | None = None
                    try:
                        _nsf_tp_raw = deal_row.get(f"Total_Points_{acting_dir_step}")
                        if _nsf_tp_raw is not None:
                            _nsf_self_tp = float(_nsf_tp_raw)
                    except Exception:
                        pass

                    def _nsf_tp_ceiling(opt_entry: Dict[str, Any]) -> float | None:
                        """Extract the TP ceiling from an option's agg_expr list."""
                        for _ae in list(opt_entry.get("agg_expr") or []):
                            _ae_s = str(_ae).strip()
                            if "Total_Points" in _ae_s and "<=" in _ae_s:
                                try:
                                    return float(_ae_s.split("<=")[-1].strip())
                                except (ValueError, IndexError):
                                    pass
                        return None

                    _nsf_chosen: Dict[str, Any] | None = None
                    _nsf_chosen_priority = 99
                    for _o in list(next_bids or []):
                        _b = str((_o or {}).get("bid", "") or "").strip().upper()
                        if _b in ("", "P", "PASS"):
                            continue
                        if len(_b) < 2 or not _b[0].isdigit():
                            continue
                        # Skip options whose TP ceiling is far below the
                        # hand's actual strength — injecting a weak-hand
                        # bid for a strong hand is catastrophic.
                        if _nsf_self_tp is not None:
                            _nsf_ceil = _nsf_tp_ceiling(_o)
                            if _nsf_ceil is not None and _nsf_self_tp > _nsf_ceil + 4:
                                continue
                        _b_lvl = int(_b[0])
                        _b_st = _b[1:]
                        if _b_st == "NT":
                            _b_st = "N"
                        _pri = 50
                        if _nsf_second_suit and _b_st == _nsf_second_suit and _nsf_second_len >= 4:
                            _pri = 10 + _b_lvl
                        elif _b_st == _nsf_open_st:
                            _pri = 20 + _b_lvl
                        elif _b_st == "N":
                            _pri = 30 + _b_lvl
                        else:
                            _pri = 40 + _b_lvl
                        if _pri < _nsf_chosen_priority:
                            _nsf_chosen = _o
                            _nsf_chosen_priority = _pri
                    if _nsf_chosen is not None:
                        _nsf_chosen["_forced_new_suit_response"] = True
                        _nsf_bid_name = str((_nsf_chosen or {}).get("bid", "") or "").strip().upper()
                        passed_opts = [_nsf_chosen]
                        _nsf_second_desc = f"{_nsf_second_len}{_nsf_second_suit}" if _nsf_second_suit else "none"
                        step_rec["hard_gate_reason"] = (
                            f"NEW_SUIT_FORCING_GATE: partner's 1{_nsf_partner_strain} is forcing; "
                            f"self opened 1{_nsf_open_st}, second suit {_nsf_second_desc}; "
                            f"injected {_nsf_bid_name}"
                        )
        except Exception:
            pass

        # NT injection gate: when all 4 suits have been bid (or 3 + stopper
        # in the 4th), no major fit, and combined HCP is sufficient, inject
        # 2NT or 3NT as the sole non-pass candidate.
        try:
            _nt_inj = _check_nt_injection_conditions(
                tokens=tokens,
                dealer_actual=dealer_actual,
                acting_dir=acting_dir_step,
                deal_row=deal_row,
                bt_seat=seat_bt,
                dealer_rot=dealer_rot,
                deal_row_idx=int(deal_row_idx),
                passed_opts=passed_opts,
            )
            if _nt_inj is not None:
                _nt_bid_name = str(_nt_inj["bid"])
                _nt_opt: Dict[str, Any] = {
                    "bid": _nt_bid_name,
                    "bt_index": None,
                    "expr": [],
                    "agg_expr": [],
                    "can_complete": True,
                    "is_completed_auction": False,
                    "is_dead_end": False,
                    "matching_deal_count": 0,
                    "avg_ev": None,
                    "avg_par": None,
                    "_nt_injection": True,
                }
                passed_opts = [_nt_opt]
                step_rec["hard_gate_reason"] = str(_nt_inj.get("reason", "NT_INJECTION_GATE"))
                step_rec["nt_injection_evidence"] = _nt_inj.get("evidence", {})
        except Exception:
            pass

        # Major-first response gate: when responding to partner's 1-level
        # opening, bid a 1-level major before a 2-level minor, and prefer
        # the correct major when both are available.
        try:
            _mfr_partner = _partner_dir(acting_dir_step)
            _mfr_partner_opened_1 = False
            _mfr_is_first_response = False
            _mfr_partner_opening_strain: str | None = None
            if len(tokens) >= 2:
                _mfr_self_has_bid = False
                for _ti, _tk in enumerate(tokens):
                    _tk_u = str(_tk or "").strip().upper()
                    _bidder = _token_bidder_dir_for_dealer(_ti, dealer_actual)
                    if _bidder == _mfr_partner and _ti == (len(tokens) - 2):
                        if len(_tk_u) == 2 and _tk_u[0] == "1" and _tk_u[1] in ("C", "D", "H", "S"):
                            _mfr_partner_opened_1 = True
                            _mfr_partner_opening_strain = _tk_u[1]
                    if _bidder == acting_dir_step and _tk_u not in ("P", "PASS", ""):
                        _mfr_self_has_bid = True
                _mfr_is_first_response = _mfr_partner_opened_1 and not _mfr_self_has_bid
                if tokens[-1].strip().upper() not in ("P", "PASS"):
                    _mfr_is_first_response = False

            if _mfr_is_first_response and passed_opts:
                _mfr_1h: list[dict] = []
                _mfr_1s: list[dict] = []
                _mfr_2minor: list[dict] = []
                _mfr_other: list[dict] = []
                for _o in list(passed_opts):
                    _b = str((_o or {}).get("bid", "") or "").strip().upper()
                    if _b == "1H":
                        _mfr_1h.append(_o)
                    elif _b == "1S":
                        _mfr_1s.append(_o)
                    elif _b in ("2C", "2D"):
                        _mfr_2minor.append(_o)
                    else:
                        _mfr_other.append(_o)

                _mfr_has_major = bool(_mfr_1h or _mfr_1s)
                _mfr_removed: list[str] = []
                _mfr_reason_parts: list[str] = []

                if _mfr_has_major and _mfr_2minor:
                    _mfr_removed = [str((_o or {}).get("bid", "")) for _o in _mfr_2minor]
                    _mfr_2minor = []
                    _mfr_reason_parts.append(f"removed {'/'.join(_mfr_removed)}")

                if _mfr_1h and _mfr_1s:
                    _mfr_sl_h: int | None = None
                    _mfr_sl_s: int | None = None
                    if deal_row:
                        _mfr_pbn = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                        if _mfr_pbn and "." in _mfr_pbn:
                            try:
                                _mfr_hp = _mfr_pbn.split(".")
                                if len(_mfr_hp) == 4:
                                    _mfr_sl_s = len(_mfr_hp[0])
                                    _mfr_sl_h = len(_mfr_hp[1])
                            except Exception:
                                pass
                    if _mfr_sl_h is not None and _mfr_sl_s is not None:
                        if _mfr_sl_h > _mfr_sl_s:
                            _mfr_1s = []
                            _mfr_reason_parts.append(f"1H preferred (H={_mfr_sl_h} > S={_mfr_sl_s})")
                        elif _mfr_sl_s > _mfr_sl_h:
                            _mfr_1h = []
                            _mfr_reason_parts.append(f"1S preferred (S={_mfr_sl_s} > H={_mfr_sl_h})")
                        elif _mfr_sl_h == _mfr_sl_s and _mfr_sl_h == 4:
                            _mfr_1s = []
                            _mfr_reason_parts.append(f"1H preferred (4-4 up-the-line)")
                        elif _mfr_sl_h == _mfr_sl_s and _mfr_sl_h >= 5:
                            _mfr_1h = []
                            _mfr_reason_parts.append(f"1S preferred (5-5+ higher suit)")

                _mfr_new = _mfr_1h + _mfr_1s + _mfr_2minor + _mfr_other
                if len(_mfr_new) < len(passed_opts) and _mfr_new:
                    passed_opts = _mfr_new
                    if _mfr_reason_parts:
                        step_rec["hard_gate_reason"] = (
                            "MAJOR_FIRST_RESPONSE: " + "; ".join(_mfr_reason_parts)
                        )
        except Exception:
            pass

        # Hard gate:
        # after partner has shown takeout-double intent, responder with values
        # may not choose Pass if a criteria-passing non-pass continuation exists.
        try:
            _partner_step = _partner_dir(acting_dir_step)
            _takeout_live = bool(takeout_double_shown_by_dir.get(_partner_step, False))
            _tp_step = None
            try:
                _tp_raw = deal_row.get(f"Total_Points_{acting_dir_step}")
                if _tp_raw is not None:
                    _tp_step = float(_tp_raw)
            except Exception:
                _tp_step = None

            _passed_non_pass_pool = [
                _o for _o in list(passed_opts or [])
                if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
            ]
            _has_passing_non_pass = len(_passed_non_pass_pool) > 0
            _in_takeout_value_window = bool(
                _takeout_live and isinstance(_tp_step, (int, float)) and float(_tp_step) >= 11.0
            )

            if _in_takeout_value_window and _has_passing_non_pass:
                _before_n = len(passed_opts)
                passed_opts = [
                    _o for _o in list(passed_opts or [])
                    if str((_o or {}).get("bid", "") or "").strip().upper() not in ("P", "PASS")
                ]
                if len(passed_opts) != _before_n:
                    step_rec["hard_gate_reason"] = (
                        "TAKEOUT_DOUBLE_PASS_HARD_GATE: partner doubled and responder has values; "
                        "disallow Pass while criteria-passing non-pass continuations exist"
                    )
        except Exception:
            pass

        if not passed_opts:
            # Narrow fallback for strong-2C response context:
            # after 1N-P-P-2C-P (runtime seat-1 view can appear as bt_prefix=2C-P
            # with one leading pass), do not allow pass-out when criteria are too strict;
            # prefer 2D if it is a legal continuation in BT.
            try:
                _prefix_now = str(bt_prefix or "").strip().upper()
                _tokens_now = "-".join([str(t or "").strip().upper() for t in list(tokens or []) if str(t or "").strip()])
                _is_capp_response_context = bool(
                    _prefix_now == "1N-P-P-2C-P"
                    or (_prefix_now == "2C-P" and int(leading_passes) == 1 and _tokens_now == "P-2C-P")
                )
                if _is_capp_response_context:
                    _fallback_2d = None
                    for _o in list(next_bids or []):
                        _b = str((_o or {}).get("bid", "") or "").strip().upper()
                        if _b != "2D":
                            continue
                        _bt_i = (_o or {}).get("bt_index")
                        _can_complete = bool((_o or {}).get("can_complete", True))
                        _is_dead_end = bool((_o or {}).get("is_dead_end", False))
                        if _bt_i is not None and _can_complete and (not _is_dead_end):
                            _fallback_2d = _o
                            break
                    if _fallback_2d is not None:
                        tokens.append("2D")
                        step_rec["choice"] = "2D"
                        step_rec["hard_gate_reason"] = (
                            "STRONG_2C_RESPONSE_NO_PASS_GATE: forcing fallback to 2D in 1N-P-P-2C-P context"
                        )
                        step_rec["bidder_view"] = {
                            "acting_direction": acting_dir_step,
                            "chosen_bid": "2D",
                            "likely_final_contract": None,
                            "likely_par_score_ns": None,
                            "likely_par_score_acting": None,
                            "source": "strong_2c_default_2d_fallback",
                        }
                        step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
                        steps_detail.append(step_rec)
                        continue
            except Exception:
                pass

            # Semantics: choose Pass when nothing matches.
            if pass_opt is None:
                break
            tokens.append("P")
            step_rec["choice"] = "P"
            _avg_par_ns = None
            try:
                _ap1 = pass_opt.get("avg_par") if pass_opt is not None else None
                if _ap1 is not None:
                    _avg_par_ns = float(_ap1)
            except Exception:
                _avg_par_ns = None
            step_rec["bidder_view"] = {
                "acting_direction": acting_dir_step,
                "chosen_bid": "P",
                "likely_final_contract": None,
                "likely_par_score_ns": (
                    round(float(acting_sign_step) * float(_avg_par_ns), 2) if _avg_par_ns is not None else None
                ),
                "likely_par_score_acting": round(float(_avg_par_ns), 2) if _avg_par_ns is not None else None,
                "source": "criteria_empty_pass_avg_par",
            }
            step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
            steps_detail.append(step_rec)
            continue

        # If only one passing bid, take it (no scoring).
        # In v2 mode, force scoring path so TP-floor level-cap checks run uniformly.
        # EXCEPTION: If the one passing bid is Pass, but we also have Pass in candidates...
        # Actually we just want to ensure we score if there's a choice.
        # But if only 1 valid move, we must take it.
        # Wait - if 'Pass' is the only valid move, we take it.
        # If '2D' is the only valid move, we take it.
        # If 'Pass' and '2D' are valid, we score both.
        if len(passed_opts) == 1 and not use_guardrails_v2:
            bid1 = str(passed_opts[0].get("bid", "") or "").strip().upper()
            _single_v2_diag: Dict[str, Any] = {
                "v2_jump_detected": False,
                "v2_blocked": False,
                "v2_block_reason": None,
            }
            if use_guardrails_v2 and bid1 not in ("", "P", "PASS"):
                try:
                    _m_single = re.match(r"^([1-7])\\s*(NT|N|[CDHS])$", bid1)
                    _single_blocked = False
                    _single_block_reason = None
                    if _m_single:
                        _single_bid_level = int(_m_single.group(1))
                        _single_bid_strain = "N" if _m_single.group(2).upper() in ("N", "NT") else _m_single.group(2).upper()
                        _single_game_level = 3 if _single_bid_strain == "N" else (4 if _single_bid_strain in ("H", "S") else 5)
                        _single_prev_same = None
                        _acting_side_single = "NS" if str(acting_dir_step).upper() in ("N", "S") else "EW"
                        _dirs_single = ["N", "E", "S", "W"]
                        _dealer_idx_single = _dirs_single.index(dealer_actual) if dealer_actual in _dirs_single else 0
                        for _i2, _tk2 in enumerate(list(tokens or [])):
                            _m2 = re.match(r"^([1-7])\\s*(NT|N|[CDHS])$", str(_tk2 or "").strip().upper())
                            if not _m2:
                                continue
                            _lvl2 = int(_m2.group(1))
                            _st2 = "N" if _m2.group(2).upper() in ("N", "NT") else _m2.group(2).upper()
                            _dir2 = _dirs_single[(_dealer_idx_single + int(_i2)) % 4]
                            _side2 = "NS" if _dir2 in ("N", "S") else "EW"
                            if _side2 == _acting_side_single and _st2 == _single_bid_strain:
                                _single_prev_same = _lvl2
                        if (
                            _single_prev_same is not None
                            and _single_bid_level > _single_game_level
                            and _single_bid_level > _single_prev_same
                            and (_single_prev_same <= _single_game_level)
                        ):
                            _single_blocked = True
                            _single_v2_diag["v2_jump_detected"] = True
                            _single_v2_diag["v2_blocked"] = True
                            _single_block_reason = (
                                "GUARDRAILS_V2_JUMP_PAST_GAME_BLOCK: single-option jump past game "
                                f"({_single_prev_same}{_single_bid_strain}->{_single_bid_level}{_single_bid_strain})"
                            )
                            _single_v2_diag["v2_block_reason"] = _single_block_reason
                    if _single_blocked:
                        if pass_opt is not None:
                            bid1 = "P"
                            step_rec["hard_gate_reason"] = _single_block_reason
                        else:
                            step_rec["hard_gate_reason"] = _single_block_reason
                            step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
                            steps_detail.append(step_rec)
                            continue
                except Exception:
                    pass
            if use_guardrails_v2:
                step_rec["v2_single_candidate_diag"] = _single_v2_diag
            _single_hard_gate_reason: str | None = None
            _single_hard_gate_applied = False
            if use_common_sense:
                try:
                    _legal_non_pass_candidates = [
                        _o for _o in list(candidates_sorted or [])
                        if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                        and bool((_o or {}).get("can_complete", True))
                    ]
                    _blocked_cannot_complete = [
                        _r for _r in list(_bids_criteria_filtered or [])
                        if str((_r or {}).get("filter_reason", "")).strip().lower() == "cannot_complete"
                    ]
                    _blocked_criteria_fail = [
                        _r for _r in list(_bids_criteria_filtered or [])
                        if str((_r or {}).get("filter_reason", "")).strip().lower() == "criteria_fail"
                        and str((_r or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                    ]
                    _self_tp_step = None
                    _self_sl_step_single: Dict[str, int] | None = None
                    try:
                        _tp_raw = deal_row.get(f"Total_Points_{acting_dir_step}")
                        if _tp_raw is not None:
                            _self_tp_step = float(_tp_raw)
                    except Exception:
                        _self_tp_step = None
                    try:
                        _h = str(deal_row.get(f"Hand_{acting_dir_step}", "") or "").strip()
                        if _h:
                            _parts = _h.split(".")
                            if len(_parts) == 4:
                                _self_sl_step_single = {
                                    "S": len(str(_parts[0] or "")),
                                    "H": len(str(_parts[1] or "")),
                                    "D": len(str(_parts[2] or "")),
                                    "C": len(str(_parts[3] or "")),
                                }
                    except Exception:
                        _self_sl_step_single = None

                    _cs_hard = compute_common_sense_hard_override(
                        auction_tokens=list(tokens or []),
                        acting_direction=acting_dir_step,
                        dealer_actual=dealer_actual,
                        self_total_points=_self_tp_step if isinstance(_self_tp_step, (int, float)) else None,
                        partner_total_points_expected=None,
                        self_suit_lengths=_self_sl_step_single if isinstance(_self_sl_step_single, dict) else None,
                        current_best_bid=bid1,
                        legal_non_pass_candidates=list(_legal_non_pass_candidates or []),
                        blocked_candidates=list(_blocked_cannot_complete or []),
                        criteria_failed_candidates=list(_blocked_criteria_fail or []),
                    )
                    if bool((_cs_hard or {}).get("apply")):
                        _ov_bid = str((_cs_hard or {}).get("selected_bid") or "").strip().upper()
                        if _ov_bid:
                            bid1 = _ov_bid
                            _single_hard_gate_applied = True
                            _single_hard_gate_reason = str((_cs_hard or {}).get("reason") or "COMMON_SENSE_HARD_OVERRIDE")
                            step_rec["common_sense_final_gate_applied"] = True
                            step_rec["common_sense_gate_reason"] = _single_hard_gate_reason
                            step_rec["bid_scores"] = [
                                {
                                    "bid": bid1,
                                    "score": None,
                                    "agg_expr": [],
                                    "common_sense_hard_override": True,
                                    "common_sense_reason_codes": list((_cs_hard or {}).get("reason_codes") or []),
                                    "common_sense_evidence": dict((_cs_hard or {}).get("evidence") or {}),
                                }
                            ]
                except Exception:
                    _single_hard_gate_applied = False
                    _single_hard_gate_reason = None
            if not bid1:
                break
            tokens.append(bid1)
            step_rec["choice"] = bid1
            _avg_par_ns = None
            try:
                _ap2 = passed_opts[0].get("avg_par")
                if _ap2 is not None:
                    _avg_par_ns = float(_ap2)
            except Exception:
                _avg_par_ns = None
            step_rec["bidder_view"] = {
                "acting_direction": acting_dir_step,
                "chosen_bid": bid1,
                "likely_final_contract": None,
                "likely_par_score_ns": (
                    round(float(acting_sign_step) * float(_avg_par_ns), 2) if _avg_par_ns is not None else None
                ),
                "likely_par_score_acting": round(float(_avg_par_ns), 2) if _avg_par_ns is not None else None,
                "source": "single_valid_bid_avg_par",
            }
            if not _single_hard_gate_applied:
                step_rec["common_sense_final_gate_applied"] = False
                step_rec["common_sense_gate_reason"] = None
            step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
            steps_detail.append(step_rec)
            continue
        
        # PERSPECTIVE: determine acting side sign (+1 for NS, -1 for EW acting)
        acting_sign = 1.0
        acting_dir = "N"
        try:
            # Use deal_row dealer
            dealer_actual = str(deal_row.get("Dealer", "N")).upper()
            directions = ["N", "E", "S", "W"]
            dealer_idx = directions.index(dealer_actual) if dealer_actual in directions else 0
            # Seat 1 is dealer, Seat 2 is LHO, etc.
            acting_dir = directions[(dealer_idx + int(seat_display) - 1) % 4]
            acting_sign = 1.0 if acting_dir in ("N", "S") else -1.0
        except Exception:
            acting_sign = 1.0
            acting_dir = "N"

        # Score top-N passing bids via bid-details (already seed-ranked above).
        passed_sorted = passed_opts[: max(1, int(top_n))]
        auction_full = "-".join(tokens)
        _current_contract_ev_ns_cache: dict[str, Any] = {"ready": False, "value": None}

        def _current_contract_ev_ns() -> float | None:
            """Deal-specific EV (NS frame) of holding the current contract."""
            if bool(_current_contract_ev_ns_cache.get("ready", False)):
                return _current_contract_ev_ns_cache.get("value")
            _current_contract_ev_ns_cache["ready"] = True
            try:
                toks_now = [str(t or "").strip().upper() for t in list(tokens or []) if str(t or "").strip()]
                has_contract = any(bool(re.match(r"^[1-7]\s*(NT|N|[CDHS])", t)) for t in toks_now)
                if not has_contract:
                    _current_contract_ev_ns_cache["value"] = None
                    return None
                auc_now = "-".join(toks_now)
                ev_ns_now = get_ev_for_auction(auc_now, dealer_actual, deal_row)
                if ev_ns_now is None:
                    _current_contract_ev_ns_cache["value"] = None
                    return None
                _current_contract_ev_ns_cache["value"] = float(ev_ns_now)
                return _current_contract_ev_ns_cache["value"]
            except Exception:
                _current_contract_ev_ns_cache["value"] = None
                return None

        # Build agg_expr lookup for bid_scores annotation (item 1)
        _agg_expr_by_bid: Dict[str, List[str]] = {
            str(opt.get("bid", "")).strip().upper(): list(opt.get("agg_expr") or [])
            for opt in passed_sorted
            if str(opt.get("bid", "")).strip()
        }

        def _parse_contract_bid_text(bid_text: str) -> tuple[int, str] | None:
            s = str(bid_text or "").strip().upper()
            if len(s) < 2 or not s[0].isdigit():
                return None
            try:
                lvl = int(s[0])
            except Exception:
                return None
            st = s[1:]
            if st == "NT":
                st = "N"
            if lvl < 1 or lvl > 7 or st not in ("C", "D", "H", "S", "N"):
                return None
            return (lvl, st)

        def _game_level_for_strain(st: str) -> int:
            if st == "N":
                return 3
            if st in ("H", "S"):
                return 4
            return 5

        def _strain_rank(st: str) -> int:
            return {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}.get(str(st or "").upper(), -1)

        def _token_bidder_dir(token_idx: int) -> str:
            try:
                directions = ["N", "E", "S", "W"]
                d = str(dealer_actual or "N").upper()
                i = directions.index(d) if d in directions else 0
                return directions[(i + int(token_idx)) % 4]
            except Exception:
                return "N"

        def _partner_shown_hcp_floor_from_nt_calls(
            auction_tokens: List[str],
            acting_direction: str,
        ) -> float | None:
            """Infer minimum partner HCP from explicit prior NT calls."""
            try:
                partner_dir = _partner_dir(acting_direction)
                shown_floor: float | None = None
                # Conservative floors by explicit NT level:
                # 1NT: at least 15 HCP, 2NT: at least 18 HCP.
                nt_floor_by_level = {1: 15.0, 2: 18.0}
                for idx, tok in enumerate(list(auction_tokens or [])):
                    if _token_bidder_dir(idx) != partner_dir:
                        continue
                    parsed = _parse_contract_bid_text(str(tok or "").strip().upper())
                    if parsed is None:
                        continue
                    lvl, st = int(parsed[0]), str(parsed[1])
                    if st != "N":
                        continue
                    floor_now = nt_floor_by_level.get(lvl)
                    if floor_now is None:
                        continue
                    shown_floor = floor_now if shown_floor is None else max(shown_floor, float(floor_now))
                return shown_floor
            except Exception:
                return None

        def _dir_side(direction: str) -> str:
            return "NS" if str(direction).upper() in ("N", "S") else "EW"

        def _pass_would_end_auction(auction_tokens: List[str]) -> bool:
            """True when a Pass by the current actor would end the auction."""
            toks = [str(t or "").strip().upper() for t in list(auction_tokens or [])]
            last_contract = -1
            for i, t in enumerate(toks):
                if t and t[0].isdigit():
                    last_contract = i
            if last_contract < 0:
                return False
            after = toks[last_contract + 1 :]
            # If any non-pass action occurred since last contract, this is not pass-out.
            if any(a not in ("P", "") for a in after):
                return False
            # Two trailing passes means current actor is reopening/balancing.
            return len([a for a in after if a == "P"]) == 2

        def _classify_guardrail_phase(auction_tokens: List[str], candidate_bid: str) -> str:
            """Classify bidding phase for guardrail strictness."""
            contract_bid_positions: List[int] = []
            for idx, tok in enumerate(list(auction_tokens or [])):
                t = str(tok or "").strip().upper()
                if t and t[0].isdigit():
                    contract_bid_positions.append(idx)

            # No prior contract call: opening decision.
            if not contract_bid_positions:
                return "opening"

            opener_dir = _token_bidder_dir(contract_bid_positions[0])
            opener_side = _dir_side(opener_dir)

            has_opponent_contract = False
            for idx in contract_bid_positions[1:]:
                if _dir_side(_token_bidder_dir(idx)) != opener_side:
                    has_opponent_contract = True
                    break

            cand = str(candidate_bid or "").strip().upper()
            is_contract_cand = bool(cand and cand[0].isdigit())
            contract_count = len(contract_bid_positions) + (1 if is_contract_cand else 0)

            # Reopening / balancing seat: pass would end the auction, and we are
            # choosing a non-pass action.
            if cand not in ("P", "PASS") and _pass_would_end_auction(auction_tokens):
                return "reopening_or_balancing"

            # Opening/response/rebid in uncontested auctions should be mostly policy-led.
            if not has_opponent_contract and contract_count <= 3:
                return "early_uncontested"
            return "competitive_or_late"

        def _compute_nt_preference_adjustment(
            *,
            bid_text: str,
            bid_level_local: int | None,
            options: List[Dict[str, Any]],
            auction_tokens: List[str],
            acting_direction: str,
            fit_us_hist: Dict[str, Any] | None,
            par_topk_rows: List[Dict[str, Any]],
            acting_sign_local: float,
        ) -> Tuple[float, float, str | None]:
            """Prefer 3NT over 3m when partner shape suggests no major fit."""
            nt_bonus = 0.0
            nt_detour_pen = 0.0
            nt_reason: str | None = None
            try:
                has_3n_choice = any(
                    str((_o or {}).get("bid", "") or "").strip().upper() in ("3N", "3NT")
                    for _o in list(options or [])
                )
                if not has_3n_choice or bid_level_local != 3:
                    return nt_bonus, nt_detour_pen, nt_reason

                partner = _partner_dir(acting_direction)
                partner_suits_shown: set[str] = set()
                for _i, _tk in enumerate(list(auction_tokens or [])):
                    _c = _parse_contract_bid_text(str(_tk or "").strip().upper())
                    if _c is None:
                        continue
                    if _token_bidder_dir(_i) != partner:
                        continue
                    _st = str(_c[1])
                    if _st in ("C", "D", "H", "S"):
                        partner_suits_shown.add(_st)
                partner_two_suits_shown = len(partner_suits_shown) >= 2
                if not partner_two_suits_shown:
                    return nt_bonus, nt_detour_pen, nt_reason

                def _p_fit_8plus(suit: str) -> float | None:
                    if not isinstance(fit_us_hist, dict):
                        return None
                    h = fit_us_hist.get(str(suit))
                    if not isinstance(h, dict):
                        return None
                    tot = 0.0
                    good = 0.0
                    for _k, _v in h.items():
                        try:
                            _n = int(_k)
                            _cnt = float(_v)
                        except Exception:
                            continue
                        if _cnt <= 0:
                            continue
                        tot += _cnt
                        if _n >= 8:
                            good += _cnt
                    return (good / tot) if tot > 0 else None

                p_fit_h = _p_fit_8plus("H")
                p_fit_s = _p_fit_8plus("S")
                vals = [v for v in [p_fit_h, p_fit_s] if isinstance(v, (int, float))]
                p_major_fit_max = max(vals) if vals else None
                no_major_fit_likely = bool(p_major_fit_max is None or float(p_major_fit_max) < 0.35)
                if not no_major_fit_likely:
                    return nt_bonus, nt_detour_pen, nt_reason

                topk_supports_3nt = False
                for _r in list(par_topk_rows or []):
                    if not isinstance(_r, dict):
                        continue
                    if str(_r.get("contract") or "").strip().upper() not in ("3N", "3NT"):
                        continue
                    try:
                        _prob = float(_r.get("prob") or 0.0)
                        _aps = float(_r.get("avg_par_score") or 0.0)
                    except Exception:
                        continue
                    if _prob >= 0.05 and (float(acting_sign_local) * _aps) > 0:
                        topk_supports_3nt = True
                        break
                if not topk_supports_3nt:
                    return nt_bonus, nt_detour_pen, nt_reason

                b = str(bid_text or "").strip().upper()
                if b in ("3N", "3NT"):
                    nt_bonus = 120.0
                    nt_reason = (
                        "NT_PREFERENCE_BONUS: partner showed two suits, major fit unlikely, "
                        "and 3NT is supported by par top-k (+120)"
                    )
                elif b in ("3C", "3D"):
                    nt_detour_pen = 140.0
                    nt_reason = (
                        "NT_PREFERENCE_DETOUR: 3NT available/supported; penalize 3m detour "
                        "that consumes room (-140)"
                    )
            except Exception:
                return 0.0, 0.0, None
            return nt_bonus, nt_detour_pen, nt_reason

        def _compute_nt_over_minor_surplus_adjustment(
            *,
            bid_text: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            self_hcp: float | None,
            partner_hcp_est: float | None,
            self_tp: float | None,
            partner_tp_est: float | None,
        ) -> Tuple[float, float, str | None]:
            """Prefer NT over minor-suit bids when HCP surplus for 3NT exceeds
            TP surplus for 5-of-a-minor.

            When (combined_HCP - 25) > (combined_TP - 29), the partnership has
            more equity in NT game than in a minor-suit game.  Penalize minor
            bids and boost NT bids accordingly.
            """
            bonus = 0.0
            penalty = 0.0
            reason: str | None = None
            try:
                if bid_level_local is None or bid_strain_local is None:
                    return bonus, penalty, reason
                if self_hcp is None or partner_hcp_est is None:
                    return bonus, penalty, reason
                if self_tp is None or partner_tp_est is None:
                    return bonus, penalty, reason
                combined_hcp = float(self_hcp) + float(partner_hcp_est)
                combined_tp = float(self_tp) + float(partner_tp_est)
                hcp_surplus = combined_hcp - 25.0
                tp_surplus = combined_tp - 29.0
                if hcp_surplus <= 0 or hcp_surplus <= tp_surplus:
                    return bonus, penalty, reason
                surplus_gap = hcp_surplus - tp_surplus
                b = str(bid_text or "").strip().upper()
                if bid_strain_local in ("C", "D") and int(bid_level_local) >= 3:
                    penalty = max(200.0, min(600.0, 200.0 * surplus_gap))
                    reason = (
                        f"NT_OVER_MINOR_SURPLUS: combined HCP {combined_hcp:.0f} "
                        f"(surplus +{hcp_surplus:.0f} over 3NT) vs combined TP "
                        f"{combined_tp:.0f} (surplus {tp_surplus:+.0f} over 5m); "
                        f"NT preferred, penalize {b} (-{penalty:.0f})"
                    )
                elif bid_strain_local in ("N",) and int(bid_level_local) >= 3:
                    bonus = max(150.0, min(500.0, 150.0 * surplus_gap))
                    reason = (
                        f"NT_OVER_MINOR_SURPLUS: combined HCP {combined_hcp:.0f} "
                        f"(surplus +{hcp_surplus:.0f} over 3NT) vs combined TP "
                        f"{combined_tp:.0f} (surplus {tp_surplus:+.0f} over 5m); "
                        f"NT preferred, boost {b} (+{bonus:.0f})"
                    )
            except Exception:
                return 0.0, 0.0, None
            return bonus, penalty, reason

        def _compute_pull_nt_game_penalty(
            *,
            bid_text: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            auction_tokens: List[str],
            acting_direction: str,
        ) -> Tuple[float, str | None]:
            """Penalize suit bids that pull an agreed NT game contract.

            When partner bid 3N (game) and the current bidder bids a suit at
            the 4-level or higher, this is almost always wrong unless the bidder
            has a self-sufficient suit (7+ cards) or slam-level points.
            """
            penalty = 0.0
            reason: str | None = None
            try:
                if bid_level_local is None or bid_strain_local is None:
                    return penalty, reason
                if bid_strain_local in ("N",):
                    return penalty, reason
                if int(bid_level_local) < 4:
                    return penalty, reason
                partner_dir = _partner_dir(acting_direction)
                partner_last_bid: str | None = None
                for ti in range(len(auction_tokens) - 1, -1, -1):
                    tk_dir = _token_bidder_dir_for_dealer(ti, dealer_actual)
                    tk_u = str(auction_tokens[ti] or "").strip().upper()
                    if tk_dir == partner_dir and tk_u not in ("P", "PASS", ""):
                        partner_last_bid = tk_u
                        break
                if partner_last_bid not in ("3N", "3NT"):
                    return penalty, reason
                b = str(bid_text or "").strip().upper()
                penalty = 500.0
                reason = (
                    f"PULL_NT_GAME_PENALTY: partner bid 3NT (game); "
                    f"pulling to {b} is almost never correct without "
                    f"a self-sufficient suit (-{penalty:.0f})"
                )
            except Exception:
                return 0.0, None
            return penalty, reason

        def _compute_forcing_heart_state_adjustment(
            *,
            bid_text: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            acting_direction: str,
        ) -> Tuple[float, float, str | None]:
            """If partner has shown forcing-heart intent, guide continuation bids.

            Includes a Smolen-style carveout: after partner's artificial heart jump
            (e.g. 3H over 1N-2C-2D), prefer spade-accept continuations over pass.
            """
            bonus = 0.0
            penalty = 0.0
            reason: str | None = None
            try:
                partner_dir = _partner_dir(acting_direction)
                if not bool(forcing_heart_shown_by_dir.get(partner_dir, False)):
                    return bonus, penalty, reason
                if bid_level_local is None:
                    return bonus, penalty, reason
                b = str(bid_text or "").strip().upper()
                partner_last_bid: str | None = None
                for _i in range(len(tokens) - 1, -1, -1):
                    _tk_dir = _token_bidder_dir_for_dealer(_i, dealer_actual)
                    _tk_u = str(tokens[_i] or "").strip().upper()
                    if _tk_dir == partner_dir and _tk_u not in ("", "P", "PASS", "X", "XX"):
                        partner_last_bid = _tk_u
                        break
                # Forced-major acceptance: partner's 3H often asks opener to show
                # spade support; reward spade continuation here.
                if partner_last_bid == "3H" and bid_strain_local == "S" and int(bid_level_local) >= 3:
                    bonus = 165.0 if int(bid_level_local) >= 4 else 140.0
                    reason = (
                        "FORCED_MAJOR_ACCEPT_PRIORITY: partner's forcing 3H asks for spade-fit clarification; "
                        f"reward {b} (+{bonus:.0f})"
                    )
                    return bonus, penalty, reason
                if bid_strain_local == "H" and int(bid_level_local) >= 5:
                    # Treat direct 5H+ jumps as slam-commit actions, not generic
                    # forcing-heart continuations. This prevents convention bonus
                    # from overpowering guardrails in non-slam auctions.
                    penalty = 180.0
                    reason = (
                        "CONVENTION_HEART_OVERJUMP: partner showed forcing-heart intent; "
                        f"penalize direct heart overjump {b} without slam evidence (-{penalty:.0f})"
                    )
                elif bid_strain_local == "H" and int(bid_level_local) >= 3:
                    bonus = 160.0 if int(bid_level_local) >= 4 else 110.0
                    reason = (
                        "CONVENTION_HEART_FIT_PRIORITY: partner showed forcing-heart intent; "
                        f"reward heart-fit continuation {b} (+{bonus:.0f})"
                    )
                elif bid_strain_local in ("N", "C", "D", "S") and int(bid_level_local) >= 2:
                    penalty = 150.0 if bid_strain_local == "N" else 120.0
                    reason = (
                        "CONVENTION_HEART_FIT_DETOUR: partner showed forcing-heart intent; "
                        f"penalize non-heart continuation {b} (-{penalty:.0f})"
                    )
            except Exception:
                return 0.0, 0.0, None
            return bonus, penalty, reason

        def _compute_forced_major_game_commit_adjustment(
            *,
            bid_text: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            auction_tokens: List[str],
            acting_direction: str,
        ) -> Tuple[float, float, str | None]:
            """After ...3H-P-3S-P, prefer game commitment over pass.

            This captures a generic forced-major game continuation where opener's
            3S acceptance leaves responder in a game-going auction.
            """
            bonus = 0.0
            penalty = 0.0
            reason: str | None = None
            try:
                toks_u = [str(t or "").strip().upper() for t in list(auction_tokens or [])]
                if len(toks_u) < 4:
                    return bonus, penalty, reason
                # We only care about the immediate decision right after ...3H-P-3S-P.
                if toks_u[-4:] != ["3H", "P", "3S", "P"]:
                    return bonus, penalty, reason

                idx_3h = len(toks_u) - 4
                responder_dir = _token_bidder_dir_for_dealer(idx_3h, dealer_actual)
                if str(acting_direction or "").upper() != str(responder_dir):
                    return bonus, penalty, reason

                b = str(bid_text or "").strip().upper()
                if b in ("P", "PASS"):
                    penalty = 220.0
                    reason = (
                        "FORCED_MAJOR_GAME_COMMIT: opener accepted with 3S after forcing 3H; "
                        "penalize pass in game-going context (-220)"
                    )
                    return bonus, penalty, reason

                if bid_strain_local == "S" and bid_level_local is not None and int(bid_level_local) >= 4:
                    bonus = 220.0
                    reason = (
                        "FORCED_MAJOR_GAME_COMMIT: opener accepted with 3S after forcing 3H; "
                        f"reward spade game commitment {b} (+220)"
                    )
                    return bonus, penalty, reason
            except Exception:
                return 0.0, 0.0, None
            return bonus, penalty, reason

        def _compute_takeout_double_game_explore_adjustment(
            *,
            bid_text: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            options: List[Dict[str, Any]],
            bid_agg_expr: List[str],
            auction_tokens: List[str],
            acting_direction: str,
            self_total_points_est: float | None,
            self_total_points_actual: float | None,
        ) -> Tuple[float, float, str | None]:
            """After partner's takeout-double signal, require game exploration with values."""
            bonus = 0.0
            penalty = 0.0
            reason: str | None = None
            try:
                partner_dir = _partner_dir(acting_direction)
                if not bool(takeout_double_shown_by_dir.get(partner_dir, False)):
                    return bonus, penalty, reason

                tp = self_total_points_est if self_total_points_est is not None else self_total_points_actual
                if tp is None or float(tp) < 11.0:
                    return bonus, penalty, reason

                # Opponent's last shown strain helps identify cue-bid game-try actions.
                opp_last_strain: str | None = None
                acting_side = _dir_side(acting_direction)
                for _i in range(len(list(auction_tokens or [])) - 1, -1, -1):
                    _tk = str((auction_tokens or [])[int(_i)] or "").strip().upper()
                    _c = _parse_contract_bid_text(_tk)
                    if _c is None:
                        continue
                    if _dir_side(_token_bidder_dir(int(_i))) == acting_side:
                        continue
                    opp_last_strain = str(_c[1])
                    break

                b = str(bid_text or "").strip().upper()
                has_non_pass_choice = any(
                    str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                    for _o in list(options or [])
                )
                if b in ("P", "PASS") and has_non_pass_choice:
                    penalty = 190.0
                    reason = (
                        "TAKEOUT_DOUBLE_GAME_EXPLORE: partner doubled and responder has values; "
                        "penalize pass when constructive continuations are available (-190)"
                    )
                    return bonus, penalty, reason
                if bid_level_local is None:
                    return bonus, penalty, reason

                agg_u = [str(x or "").strip().upper() for x in list(bid_agg_expr or [])]
                has_forcing_flag = any("FORCING_" in x for x in agg_u)

                # Penalize minimum non-forcing level-1 suit responses.
                if int(bid_level_local) == 1 and bid_strain_local in ("C", "D", "H", "S") and not has_forcing_flag:
                    penalty = 130.0
                    reason = (
                        "TAKEOUT_DOUBLE_GAME_EXPLORE: partner doubled and responder has values; "
                        f"penalize minimum non-forcing response {b} (-{penalty:.0f})"
                    )
                    return bonus, penalty, reason

                # Reward game-exploration continuations.
                if b in ("2N", "2NT"):
                    bonus = 110.0
                    reason = (
                        "TAKEOUT_DOUBLE_GAME_EXPLORE: partner doubled and responder has values; "
                        "reward NT game-try exploration (+110)"
                    )
                    return bonus, penalty, reason
                if opp_last_strain is not None and bid_strain_local == opp_last_strain and int(bid_level_local) >= 2:
                    bonus = 95.0
                    reason = (
                        "TAKEOUT_DOUBLE_GAME_EXPLORE: cue-bid style continuation after partner double (+95)"
                    )
                    return bonus, penalty, reason
                if bid_strain_local in ("H", "S") and int(bid_level_local) >= 2:
                    bonus = 70.0
                    reason = (
                        "TAKEOUT_DOUBLE_GAME_EXPLORE: constructive major continuation after partner double (+70)"
                    )
                    return bonus, penalty, reason
            except Exception:
                return 0.0, 0.0, None
            return bonus, penalty, reason

        def _compute_takeout_double_trigger_adjustment(
            *,
            bid_text: str,
            options: List[Dict[str, Any]],
            auction_tokens: List[str],
            acting_direction: str,
            self_total_points_est: float | None,
            self_total_points_actual: float | None,
            self_suit_lengths: Dict[str, int] | None,
        ) -> Tuple[float, float, str | None]:
            """Encourage direct takeout-double actions in classic one-level spots."""
            bonus = 0.0
            penalty = 0.0
            reason: str | None = None
            try:
                b = str(bid_text or "").strip().upper()
                has_double_choice = any(
                    str((_o or {}).get("bid", "") or "").strip().upper() in ("D", "X", "DOUBLE")
                    for _o in list(options or [])
                )
                if not has_double_choice:
                    return bonus, penalty, reason

                acting_side = _dir_side(acting_direction)
                opp_contracts: list[tuple[int, str]] = []
                side_contract_count = 0
                for _i, _tk in enumerate(list(auction_tokens or [])):
                    _c = _parse_contract_bid_text(str(_tk or "").strip().upper())
                    if _c is None:
                        continue
                    _side = _dir_side(_token_bidder_dir(int(_i)))
                    if _side == acting_side:
                        side_contract_count += 1
                    else:
                        opp_contracts.append((int(_c[0]), str(_c[1])))

                tp = self_total_points_est if self_total_points_est is not None else self_total_points_actual

                # Penalize ridiculous doubles into constructive opponent auctions.
                # When opponents have bid 2+ contracts (a constructive sequence)
                # and our side has no contracts, a double by a weak hand is
                # pointless — it just helps the opponents.
                if side_contract_count == 0 and len(opp_contracts) >= 2 and b in ("D", "X", "DOUBLE"):
                    _tp_val = float(tp) if tp is not None else 0.0
                    if _tp_val < 14.0:
                        penalty = 400.0
                        reason = (
                            f"TAKEOUT_DOUBLE_TRIGGER: double into constructive "
                            f"opponent auction ({len(opp_contracts)} bids) with only "
                            f"{_tp_val:.0f} TP — heavily penalized (-400)"
                        )
                        return bonus, penalty, reason

                # Classic takeout seed: opponents opened one-level and our side
                # has not yet bid a contract.
                if side_contract_count > 0 or len(opp_contracts) != 1:
                    return bonus, penalty, reason
                open_lvl, open_st = opp_contracts[0]
                if open_lvl != 1 or open_st not in ("C", "D", "H", "S"):
                    return bonus, penalty, reason

                if tp is None or float(tp) < 12.0:
                    return bonus, penalty, reason

                sl = dict(self_suit_lengths or {})
                opener_len = sl.get(str(open_st), 0)
                if opener_len > 2:
                    return bonus, penalty, reason

                if b in ("D", "X", "DOUBLE"):
                    bonus = 240.0
                    reason = (
                        "TAKEOUT_DOUBLE_TRIGGER: classic one-level takeout shape/values; "
                        "reward direct double (+240)"
                    )
                    return bonus, penalty, reason
                if b in ("P", "PASS"):
                    penalty = 220.0
                    reason = (
                        "TAKEOUT_DOUBLE_TRIGGER: with takeout shape/values, penalize pass "
                        "instead of double (-220)"
                    )
                    return bonus, penalty, reason
            except Exception:
                return 0.0, 0.0, None
            return bonus, penalty, reason

        def _guardrails_v2_jump_past_game_block(
            *,
            auction_tokens: List[str],
            acting_direction: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            self_tp_est: float | None,
            self_tp_act: float | None,
            partner_tp_floor_local: float | None,
            partner_tp_ceiling_local: float | None = None,
            self_hcp_est_local: float | None = None,
            self_hcp_act_local: float | None = None,
            partner_hcp_floor_local: float | None = None,
            partner_hcp_ceiling_local: float | None = None,
        ) -> tuple[bool, str | None]:
            """Return hard-block decision for v2 jump-past-game rule."""
            diag = _guardrails_v2_jump_past_game_diagnostics(
                auction_tokens=auction_tokens,
                acting_direction=acting_direction,
                bid_level_local=bid_level_local,
                bid_strain_local=bid_strain_local,
                self_tp_est=self_tp_est,
                self_tp_act=self_tp_act,
                partner_tp_floor_local=partner_tp_floor_local,
                partner_tp_ceiling_local=partner_tp_ceiling_local,
                self_hcp_est_local=self_hcp_est_local,
                self_hcp_act_local=self_hcp_act_local,
                partner_hcp_floor_local=partner_hcp_floor_local,
                partner_hcp_ceiling_local=partner_hcp_ceiling_local,
            )
            return bool(diag.get("v2_blocked", False)), diag.get("v2_block_reason")

        def _guardrails_v2_jump_past_game_diagnostics(
            *,
            auction_tokens: List[str],
            acting_direction: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            self_tp_est: float | None,
            self_tp_act: float | None,
            partner_tp_floor_local: float | None,
            partner_tp_ceiling_local: float | None = None,
            self_hcp_est_local: float | None = None,
            self_hcp_act_local: float | None = None,
            partner_hcp_floor_local: float | None = None,
            partner_hcp_ceiling_local: float | None = None,
        ) -> Dict[str, Any]:
            """Return full diagnostics for v2 jump-past-game rule evaluation."""
            out: Dict[str, Any] = {
                "v2_bid_level": bid_level_local,
                "v2_bid_strain": bid_strain_local,
                "v2_prev_same_side_same_strain_level": None,
                "v2_game_level_for_strain": None,
                "v2_jump_detected": False,
                "v2_required_tp": None,
                "v2_self_tp_used": None,
                "v2_partner_tp_floor": partner_tp_floor_local,
                "v2_partner_tp_ceiling": partner_tp_ceiling_local,
                "v2_combined_tp_floor": None,
                "v2_combined_tp_ceiling": None,
                "v2_max_level_by_tp_floor": None,
                "v2_max_level_by_tp_ceiling": None,
                "v2_level_cap_exceeded": False,
                "v2_slam_qualified": False,
                "v2_slam_rejected_reason": None,
                "v2_blocked": False,
                "v2_block_reason": None,
            }
            try:
                if bid_level_local is None or bid_strain_local not in ("C", "D", "H", "S", "N"):
                    return out
                game_lvl_local = _game_level_for_strain(str(bid_strain_local))
                out["v2_game_level_for_strain"] = game_lvl_local
                prev_same_side_same_strain = None
                acting_side_local = _dir_side(acting_direction)
                for i2, tk2 in enumerate(list(auction_tokens or [])):
                    c2 = _parse_contract_bid_text(str(tk2 or "").strip().upper())
                    if c2 is None:
                        continue
                    if _dir_side(_token_bidder_dir(i2)) != acting_side_local:
                        continue
                    lvl2, st2 = int(c2[0]), str(c2[1])
                    if st2 == str(bid_strain_local):
                        prev_same_side_same_strain = lvl2
                out["v2_prev_same_side_same_strain_level"] = prev_same_side_same_strain
                is_jump_past_game = bool(
                    prev_same_side_same_strain is not None
                    and int(bid_level_local) > int(game_lvl_local)
                    and int(bid_level_local) > int(prev_same_side_same_strain)
                    and (
                        int(prev_same_side_same_strain) < int(game_lvl_local)
                        or int(prev_same_side_same_strain) == int(game_lvl_local)
                    )
                )
                out["v2_jump_detected"] = is_jump_past_game
                if not is_jump_past_game:
                    return out

                # For NT bids, use HCP only (distributional points don't help).
                # For suit bids, use Total Points (HCP + distribution).
                is_nt = str(bid_strain_local) == "N"
                out["v2_point_type"] = "HCP" if is_nt else "TP"

                if is_nt:
                    self_pts = None
                    if isinstance(self_hcp_est_local, (int, float)):
                        self_pts = float(self_hcp_est_local)
                    elif isinstance(self_hcp_act_local, (int, float)):
                        self_pts = float(self_hcp_act_local)
                    partner_floor = partner_hcp_floor_local
                    partner_ceil = partner_hcp_ceiling_local
                else:
                    self_pts = None
                    if isinstance(self_tp_est, (int, float)):
                        self_pts = float(self_tp_est)
                    elif isinstance(self_tp_act, (int, float)):
                        self_pts = float(self_tp_act)
                    partner_floor = partner_tp_floor_local
                    partner_ceil = partner_tp_ceiling_local

                out["v2_self_tp_used"] = self_pts
                combined_tp_max_local = None
                if isinstance(self_pts, (int, float)) and isinstance(partner_floor, (int, float)):
                    combined_tp_max_local = float(self_pts) + float(partner_floor)
                out["v2_combined_tp_floor"] = combined_tp_max_local
                # Required points for level L is 14 + 3*L
                # (e.g. 26->4-level, 29->5-level, 32->6-level, 35->7-level).
                # However, bidding past game in the same strain is only
                # justified as a slam invitation.  The ceiling must reach
                # slam TP (33) — otherwise there is no reason to bypass game.
                base_required = float(14 + (3 * int(bid_level_local)))
                is_past_game_same_strain = bool(int(bid_level_local) > int(game_lvl_local))
                slam_tp_threshold = 33.0
                required_tp = max(base_required, slam_tp_threshold) if is_past_game_same_strain else base_required
                out["v2_required_tp"] = required_tp
                out["v2_is_past_game_same_strain"] = is_past_game_same_strain
                max_level_by_tp = None
                if combined_tp_max_local is not None:
                    max_level_by_tp = int((float(combined_tp_max_local) - 14.0) // 3.0)
                    max_level_by_tp = max(0, min(7, max_level_by_tp))
                out["v2_max_level_by_tp_floor"] = max_level_by_tp
                level_cap_exceeded = bool(
                    max_level_by_tp is not None and int(bid_level_local) > int(max_level_by_tp)
                )
                out["v2_level_cap_exceeded"] = level_cap_exceeded
                floor_qualified = bool(combined_tp_max_local is not None and combined_tp_max_local >= required_tp)

                # Ceiling check: even if the floor qualifies, block the bid if
                # the ceiling (self + partner max) can't reach the required
                # points for this level.  Applies to every level — game tries,
                # slam tries, grand-slam tries.
                combined_tp_ceiling_local = None
                if isinstance(self_pts, (int, float)) and isinstance(partner_ceil, (int, float)):
                    combined_tp_ceiling_local = float(self_pts) + float(partner_ceil)
                out["v2_combined_tp_ceiling"] = round(float(combined_tp_ceiling_local), 2) if combined_tp_ceiling_local is not None else None
                out["v2_partner_tp_ceiling"] = round(float(partner_ceil), 2) if partner_ceil is not None else None

                max_level_by_ceiling = None
                if combined_tp_ceiling_local is not None:
                    max_level_by_ceiling = int((float(combined_tp_ceiling_local) - 14.0) // 3.0)
                    max_level_by_ceiling = max(0, min(7, max_level_by_ceiling))
                out["v2_max_level_by_tp_ceiling"] = max_level_by_ceiling

                ceiling_qualified = bool(
                    combined_tp_ceiling_local is not None and combined_tp_ceiling_local >= required_tp
                )
                level_qualified = floor_qualified and ceiling_qualified
                out["v2_slam_qualified"] = level_qualified

                if not ceiling_qualified and combined_tp_ceiling_local is not None:
                    pt_label = "HCP" if is_nt else "TP"
                    out["v2_slam_rejected_reason"] = (
                        f"ceiling {combined_tp_ceiling_local:.0f} (self {self_pts:.0f} + "
                        f"partner max {partner_ceil:.0f}) < {required_tp:.0f} required {pt_label} "
                        f"for level {bid_level_local}"
                    )

                if not level_cap_exceeded and ceiling_qualified:
                    return out

                # Block: either floor says level too high, or ceiling can't
                # reach the required points even with partner at maximum.
                prev_level_int = int(prev_same_side_same_strain) if prev_same_side_same_strain is not None else -1
                bid_level_int = int(bid_level_local) if bid_level_local is not None else -1
                pt_label_blk = "HCP" if is_nt else "TP"
                if not ceiling_qualified:
                    reason_local = (
                        f"GUARDRAILS_V2_CEILING_BLOCK: bid {bid_level_int}{bid_strain_local} — "
                        f"even at partner max, combined {pt_label_blk} ceiling "
                        f"{combined_tp_ceiling_local:.0f} < {required_tp:.0f} required "
                        f"(self={self_pts}, partner_max={partner_ceil}, "
                        f"prev_same_strain_level={prev_level_int})"
                    )
                else:
                    reason_local = (
                        f"GUARDRAILS_V2_LEVEL_CAP_BLOCK: bid {bid_level_int}{bid_strain_local} exceeds "
                        f"{pt_label_blk}-floor level cap (max_level={max_level_by_tp}, required={required_tp}, "
                        f"self={self_pts}, partner_floor={partner_floor}, "
                        f"combined_floor={combined_tp_max_local}, prev_same_strain_level={prev_level_int})"
                    )
                out["v2_blocked"] = True
                out["v2_block_reason"] = reason_local
                return out
            except Exception:
                return out

        def _v2_simple_raise_rescue(
            *,
            bid_text: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            acting_direction: str,
            acting_sign_local: float,
            auction_tokens: List[str],
            deal_row_local: Dict[str, Any],
            opt_avg_par: Any,
            base_phase_mult_local: float,
        ) -> Tuple[float, Dict[str, Any]] | None:
            """Rescue a bid whose criteria match but has no deals in the index.

            Applies when the bid is a *simple raise* of partner's suit at the
            level justified by Total Points.  Returns (score, breakdown) or
            None if the rescue does not apply.

            Simple-raise detection:
            1. Partner (same side) previously bid this strain.
            2. Self has 3+ cards in that strain.
            3. Self TP falls inside the appropriate range for the bid level
               (2-level: 6-9, 3-level game-try: 10-12, 4-level game: 13+).
            """
            if bid_level_local is None or bid_strain_local is None:
                return None
            if bid_strain_local not in ("C", "D", "H", "S"):
                return None

            acting_side = _dir_side(acting_direction)
            partner_bid_this_strain = False
            for i, tk in enumerate(list(auction_tokens or [])):
                c = _parse_contract_bid_text(str(tk or "").strip().upper())
                if c is None:
                    continue
                tk_dir = _token_bidder_dir(i)
                if _dir_side(tk_dir) != acting_side:
                    continue
                if tk_dir == acting_direction:
                    continue
                if str(c[1]) == str(bid_strain_local):
                    partner_bid_this_strain = True
                    break

            if not partner_bid_this_strain:
                return None

            _srr_suit_rank = {"C": 0, "D": 1, "H": 2, "S": 3}
            _srr_strain_rank = _srr_suit_rank.get(str(bid_strain_local), 0)
            for _srr_s in ("C", "D", "H", "S"):
                if _srr_suit_rank.get(_srr_s, 0) <= _srr_strain_rank:
                    continue
                _srr_bidders: set[str] = set()
                for _srr_i, _srr_tk in enumerate(list(auction_tokens or [])):
                    _srr_c = _parse_contract_bid_text(str(_srr_tk or "").strip().upper())
                    if _srr_c is None:
                        continue
                    if str(_srr_c[1]) != _srr_s:
                        continue
                    _srr_d = _token_bidder_dir(_srr_i)
                    if _dir_side(_srr_d) == acting_side:
                        _srr_bidders.add(_srr_d)
                if len(_srr_bidders) >= 2:
                    return None

            sl_key = f"SL_{bid_strain_local}_{acting_direction}"
            sl_val: int | None = None
            try:
                raw = deal_row_local.get(sl_key)
                if raw is not None:
                    sl_val = int(raw)
            except Exception:
                pass
            if sl_val is None:
                try:
                    hand = str(deal_row_local.get(f"Hand_{acting_direction}", "") or "").strip()
                    if hand:
                        parts = hand.split(".")
                        if len(parts) == 4:
                            idx = {"S": 0, "H": 1, "D": 2, "C": 3}.get(str(bid_strain_local), -1)
                            if idx >= 0:
                                sl_val = len(parts[idx])
                except Exception:
                    pass
            if sl_val is None or sl_val < 3:
                return None

            tp_key = f"Total_Points_{acting_direction}"
            tp_val: float | None = None
            try:
                raw = deal_row_local.get(tp_key)
                if raw is not None:
                    tp_val = float(raw)
            except Exception:
                pass
            if tp_val is None:
                return None

            tp_ranges: Dict[int, Tuple[float, float]] = {
                2: (6.0, 9.0),
                3: (10.0, 12.0),
                4: (13.0, 40.0),
                5: (13.0, 40.0),
            }
            rng = tp_ranges.get(int(bid_level_local))
            if rng is None:
                return None
            if not (rng[0] <= tp_val <= rng[1]):
                return None

            # Compute synthetic score from precomputed avg_par or a
            # neutral positive value so the raise beats Pass(~0).
            synth_base: float = 0.0
            source = "simple_raise_neutral"
            if opt_avg_par is not None:
                try:
                    synth_base = float(opt_avg_par)
                    source = "simple_raise_avg_par"
                except Exception:
                    synth_base = 0.0

            if synth_base == 0.0:
                game_lvl = _game_level_for_strain(str(bid_strain_local))
                is_major = bid_strain_local in ("H", "S")
                level = int(bid_level_local)
                if level >= game_lvl:
                    synth_base = 500.0 if is_major else 400.0
                elif level == game_lvl - 1:
                    synth_base = 400.0 if is_major else 300.0
                else:
                    synth_base = 150.0
                source = "simple_raise_heuristic"

            base_acting = float(acting_sign_local) * float(synth_base)
            final = base_acting * float(base_phase_mult_local)
            breakdown: Dict[str, Any] = {
                "cache_mode": "v2_simple_raise_rescue",
                "base": round(base_acting, 2),
                "base_shrunk": round(final, 2),
                "base_phase_mult": round(base_phase_mult_local, 4),
                "matched_n": None,
                "mean_par": round(synth_base, 2) if source == "simple_raise_avg_par" else None,
                "final_score": round(final, 2),
                "acting_sign": float(acting_sign_local),
                "simple_raise_reason": (
                    f"SIMPLE_RAISE_RESCUE: partner bid {bid_strain_local}, self has "
                    f"{sl_val} {bid_strain_local} and TP={tp_val:.0f} "
                    f"(range {rng[0]:.0f}-{rng[1]:.0f} for level {bid_level_local}); "
                    f"source={source}"
                ),
            }
            breakdown["special_case_notes"] = [breakdown["simple_raise_reason"]]
            return (final, breakdown)

        def _v2_level_acceptable_missing_criteria_rescue(
            *,
            bid_text: str,
            bid_level_local: int | None,
            bid_strain_local: str | None,
            acting_direction: str,
            acting_sign_local: float,
            deal_row_local: Dict[str, Any],
            opt_avg_par: Any,
            base_phase_mult_local: float,
            details_error_text: str,
        ) -> Tuple[float, Dict[str, Any]] | None:
            """General rescue for contract bids dropped by empty criteria masks.

            Applies only when bid-details fails with a no-match/empty-mask message,
            the bid is at or below game, and the partnership points indicate the
            target level is supportable.
            """
            if bid_level_local is None or bid_strain_local is None:
                return None
            if bid_strain_local not in ("C", "D", "H", "S", "N", "NT"):
                return None
            if int(bid_level_local) >= 6:
                return None

            msg_u = str(details_error_text or "").strip().upper()
            if ("NO MATCHED DEALS" not in msg_u) and ("CRITERIA MASK EMPTY" not in msg_u):
                return None

            game_level = _game_level_for_strain("N" if bid_strain_local in ("N", "NT") else str(bid_strain_local))
            if int(bid_level_local) > int(game_level):
                return None

            partner_direction = _partner_dir(acting_direction)
            self_hcp: float | None = None
            partner_hcp: float | None = None
            self_tp: float | None = None
            partner_tp: float | None = None
            try:
                _v = deal_row_local.get(f"HCP_{acting_direction}")
                if _v is not None:
                    self_hcp = float(_v)
            except Exception:
                self_hcp = None
            try:
                _v = deal_row_local.get(f"HCP_{partner_direction}")
                if _v is not None:
                    partner_hcp = float(_v)
            except Exception:
                partner_hcp = None
            try:
                _v = deal_row_local.get(f"Total_Points_{acting_direction}")
                if _v is not None:
                    self_tp = float(_v)
            except Exception:
                self_tp = None
            try:
                _v = deal_row_local.get(f"Total_Points_{partner_direction}")
                if _v is not None:
                    partner_tp = float(_v)
            except Exception:
                partner_tp = None

            if self_hcp is None or partner_hcp is None:
                return None
            combined_hcp = float(self_hcp) + float(partner_hcp)
            combined_tp: float | None = None
            if self_tp is not None and partner_tp is not None:
                combined_tp = float(self_tp) + float(partner_tp)

            is_nt = str(bid_strain_local) in ("N", "NT")
            combined_pts_for_level = combined_hcp if is_nt else (
                combined_tp if combined_tp is not None else combined_hcp
            )
            required_pts = 14.0 + 3.0 * float(int(bid_level_local))
            max_level = max(0, min(7, int((combined_pts_for_level - 14.0) // 3.0)))
            if int(bid_level_local) > int(max_level):
                return None

            if int(bid_level_local) >= int(game_level) and combined_hcp < 25.0:
                return None

            synth_base = 0.0
            source = "level_accept_heuristic"
            if opt_avg_par is not None:
                try:
                    synth_base = float(opt_avg_par)
                    source = "level_accept_avg_par"
                except Exception:
                    synth_base = 0.0
            if synth_base == 0.0:
                if int(bid_level_local) >= int(game_level):
                    if is_nt:
                        synth_base = 430.0
                    elif str(bid_strain_local) in ("H", "S"):
                        synth_base = 420.0
                    else:
                        synth_base = 400.0
                else:
                    synth_base = 120.0 + 40.0 * float(int(bid_level_local))

            base_acting = float(acting_sign_local) * float(synth_base)
            final = base_acting * float(base_phase_mult_local)
            breakdown: Dict[str, Any] = {
                "cache_mode": "v2_level_acceptable_missing_criteria_rescue",
                "base": round(base_acting, 2),
                "base_shrunk": round(final, 2),
                "base_phase_mult": round(float(base_phase_mult_local), 4),
                "matched_n": None,
                "mean_par": round(synth_base, 2) if source == "level_accept_avg_par" else None,
                "final_score": round(final, 2),
                "acting_sign": float(acting_sign_local),
                "missing_criteria_rescue_reason": (
                    f"LEVEL_ACCEPTABLE_MISSING_CRITERIA_RESCUE: {bid_text} had "
                    f"no matched deals ({str(details_error_text or '').strip()}); "
                    f"combined HCP={combined_hcp:.0f}"
                    + (f", combined TP={combined_tp:.0f}" if combined_tp is not None else "")
                    + f", max_level={int(max_level)} supports level {int(bid_level_local)}; "
                    f"source={source}"
                ),
            }
            breakdown["special_case_notes"] = [breakdown["missing_criteria_rescue_reason"]]
            return (final, breakdown)

        def _score_one(opt: Dict[str, Any]) -> Tuple[float, str, float, float, bool, Dict[str, Any]]:
            bid1 = str(opt.get("bid", "") or "").strip().upper()
            if not bid1:
                return float("-inf"), "", 0.0, 0.0, False, {}

            score_phase = _classify_guardrail_phase(tokens, bid1)

            def _base_phase_multiplier(phase_name: str) -> float:
                if phase_name == "opening":
                    return float(opening_base_mult)
                if phase_name == "early_uncontested":
                    return 1.0
                return 1.0

            def _special_case_breakdown_fields(opt_local: Dict[str, Any]) -> Dict[str, Any]:
                special_notes: list[str] = []
                out: Dict[str, Any] = {}
                for src_key, dst_key in (
                    ("_simple_raise_pre_rescue_reason", "simple_raise_pre_rescue_reason"),
                    ("_nt_raise_rescue_reason", "nt_raise_rescue_reason"),
                    ("_direct_overcall_gap_rescue_reason", "direct_overcall_gap_rescue_reason"),
                ):
                    reason = opt_local.get(src_key)
                    if not reason:
                        continue
                    reason_str = str(reason).strip()
                    if not reason_str:
                        continue
                    out[dst_key] = reason_str
                    if reason_str not in special_notes:
                        special_notes.append(reason_str)
                if special_notes:
                    out["special_case_notes"] = special_notes
                return out

            base_phase_mult = _base_phase_multiplier(score_phase)
            bid_contract = _parse_contract_bid_text(bid1)
            bid_level = bid_contract[0] if bid_contract is not None else None
            bid_strain = bid_contract[1] if bid_contract is not None else None
            is_level1_contract = bool(bid_level == 1)
            def _is_level1_contract_bid_text(s: str) -> bool:
                _c = _parse_contract_bid_text(s)
                return bool(_c is not None and _c[0] == 1)

            has_level1_contract_choice = any(
                _is_level1_contract_bid_text(str(_o.get("bid", "") or "").strip().upper())
                for _o in list(passed_opts or [])
            )
            
            # Handle Pass (not BT-backed)
            # Prefer the value of holding the current contract on this deal, then
            # fall back to option aggregates when EV cannot be resolved.
            if bid1 in ("P", "PASS"):
                score_val: float | None = None
                _pass_source = "opt"
                try:
                    cur_ev_ns = _current_contract_ev_ns()
                    if cur_ev_ns is not None:
                        score_val = float(cur_ev_ns)
                        _pass_source = "current_contract_ev_ns"
                except Exception:
                    pass
                try:
                    avg_ev = opt.get("avg_ev")
                    if score_val is None and avg_ev is not None:
                        score_val = float(avg_ev)
                        _pass_source = "opt_avg_ev"
                except Exception:
                    pass
                if score_val is None:
                    try:
                        avg_par = opt.get("avg_par")
                        if avg_par is not None:
                            score_val = float(avg_par)
                            _pass_source = "opt_avg_par"
                    except Exception:
                        pass

                if score_val is None:
                    score_val = 0.0

                pass_bonus = 0.0
                pass_bonus_reason: str | None = None
                try:
                    pass_bonus, pass_bonus_reason = compute_pass_signoff_bonus(
                        auction_tokens=tokens,
                        acting_direction=acting_dir,
                        dealer_actual=dealer_actual,
                    )
                except Exception:
                    pass_bonus = 0.0
                    pass_bonus_reason = None

                pass_penalty = 0.0
                pass_penalty_reason: str | None = None
                if score_phase in ("opening", "early_uncontested") and has_level1_contract_choice:
                    pass_penalty = float(opening_pass_penalty)
                    pass_penalty_reason = (
                        "EARLY_PASS_PENALTY: valid level-1 opening available in early phase"
                    )
                try:
                    partner_dir = _partner_dir(acting_dir)
                    _forcing_ctx = bool(forcing_heart_shown_by_dir.get(partner_dir, False))
                    _has_non_pass = any(
                        str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                        for _o in list(passed_opts or [])
                    )
                    if _forcing_ctx and _has_non_pass:
                        _extra = 180.0
                        pass_penalty += _extra
                        _tag = (
                            "FORCING_HEART_PASS_PENALTY: partner showed forcing-heart/Smolen intent; "
                            "penalize pass while constructive continuations exist"
                        )
                        pass_penalty_reason = (
                            _tag if not pass_penalty_reason else f"{pass_penalty_reason}; {_tag}"
                        )
                except Exception:
                    pass
                try:
                    _self_tp_actual = deal_row.get(f"Total_Points_{acting_dir}")
                    if _self_tp_actual is not None:
                        _self_tp_actual = float(_self_tp_actual)
                except Exception:
                    _self_tp_actual = None
                try:
                    _self_hcp_actual = deal_row.get(f"HCP_{acting_dir}")
                    if _self_hcp_actual is not None:
                        _self_hcp_actual = float(_self_hcp_actual)
                except Exception:
                    _self_hcp_actual = None
                try:
                    _self_hand_actual = str(deal_row.get(f"Hand_{acting_dir}", "") or "").strip() or None
                except Exception:
                    _self_hand_actual = None
                try:
                    _has_non_pass = any(
                        str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                        for _o in list(passed_opts or [])
                    )
                    if _has_non_pass:
                        _constructive_reasons: list[str] = []
                        _pass_expr = [str(_x or "").strip().upper() for _x in list(opt.get("agg_expr") or [])]

                        def _upper_bound_excess(metric_name: str, actual_value: float | None) -> tuple[float, float] | None:
                            if actual_value is None:
                                return None
                            best_cap: float | None = None
                            for _expr in _pass_expr:
                                if not _expr:
                                    continue
                                _m = re.fullmatch(rf"{metric_name}\s*(<=|<)\s*(-?\d+(?:\.\d+)?)", _expr)
                                if not _m:
                                    continue
                                _op = str(_m.group(1))
                                _cap = float(_m.group(2))
                                _eff_cap = _cap if _op == "<=" else (_cap - 1.0)
                                best_cap = _eff_cap if best_cap is None else min(best_cap, _eff_cap)
                            if best_cap is None or actual_value <= best_cap:
                                return None
                            return best_cap, float(actual_value - best_cap)

                        _tp_excess = _upper_bound_excess("TOTAL_POINTS", _self_tp_actual)
                        if _tp_excess is not None and _self_tp_actual is not None:
                            _tp_cap, _tp_over = _tp_excess
                            _tp_pen = min(220.0, 30.0 * float(_tp_over))
                            pass_penalty += _tp_pen
                            _constructive_reasons.append(
                                "PASS_RANGE_TOO_STRONG: pass criteria cap Total_Points at "
                                f"{_tp_cap:.0f}; self has {_self_tp_actual:.0f} (-{_tp_pen:.0f})"
                            )

                        _hcp_excess = _upper_bound_excess("HCP", _self_hcp_actual)
                        if _hcp_excess is not None and _self_hcp_actual is not None:
                            _hcp_cap, _hcp_over = _hcp_excess
                            _hcp_pen = min(160.0, 20.0 * float(_hcp_over))
                            pass_penalty += _hcp_pen
                            _constructive_reasons.append(
                                "PASS_RANGE_TOO_STRONG: pass criteria cap HCP at "
                                f"{_hcp_cap:.0f}; self has {_self_hcp_actual:.0f} (-{_hcp_pen:.0f})"
                            )

                        try:
                            _contract_positions: list[int] = []
                            _contract_sides: set[str] = set()
                            _last_contract_idx: int | None = None
                            _last_contract_level: int | None = None
                            _last_contract_strain: str | None = None
                            for _idx, _tok in enumerate(list(tokens or [])):
                                _parsed = _parse_contract_bid_text(str(_tok or "").strip().upper())
                                if _parsed is None:
                                    continue
                                _lvl, _strain = _parsed
                                _contract_positions.append(_idx)
                                _contract_sides.add(_dir_side(_token_bidder_dir(_idx)))
                                _last_contract_idx = _idx
                                _last_contract_level = int(_lvl)
                                _last_contract_strain = str(_strain)

                            _partner_dir_local = _partner_dir(acting_dir)
                            _last_bidder = _token_bidder_dir(_last_contract_idx) if _last_contract_idx is not None else None
                            _game_min = 3 if _last_contract_strain == "N" else (4 if _last_contract_strain in ("H", "S") else 5)
                            if (
                                _last_contract_idx is not None
                                and _last_contract_level is not None
                                and len(_contract_sides) == 1
                                and len(_contract_positions) >= 2
                                and _partner_dir_local
                                and _last_bidder == _partner_dir_local
                                and int(_last_contract_level) < int(_game_min)
                                and _self_tp_actual is not None
                                and float(_self_tp_actual) >= 13.0
                            ):
                                _extras_pen = 140.0 + min(80.0, 20.0 * max(0.0, float(_self_tp_actual) - 13.0))
                                pass_penalty += _extras_pen
                                _constructive_reasons.append(
                                    "CONSTRUCTIVE_BELOW_GAME_PASS_PENALTY: partner made the last below-game contract bid, "
                                    f"constructive continuations exist, and self has {_self_tp_actual:.0f} Total_Points (-{_extras_pen:.0f})"
                                )
                        except Exception:
                            pass

                        if _constructive_reasons:
                            _pass_reason = "; ".join(_constructive_reasons)
                            pass_penalty_reason = _pass_reason if not pass_penalty_reason else f"{pass_penalty_reason}; {_pass_reason}"
                except Exception:
                    pass
                try:
                    _sm_b, _sm_p, _sm_r = _compute_forced_major_game_commit_adjustment(
                        bid_text=bid1,
                        bid_level_local=bid_level,
                        bid_strain_local=bid_strain,
                        auction_tokens=list(tokens or []),
                        acting_direction=acting_dir,
                    )
                    pass_penalty += float(_sm_p)
                    if _sm_r:
                        pass_penalty_reason = _sm_r if not pass_penalty_reason else f"{pass_penalty_reason}; {_sm_r}"
                except Exception:
                    pass

                pass_hard_blocked = bool(
                    score_phase in ("opening", "early_uncontested") and has_level1_contract_choice
                )
                pass_hard_block_reason: str | None = None
                if pass_hard_blocked:
                    pass_hard_block_reason = (
                        "BT_OPENING_PASS_BLOCK: legal BT level-1 opening available; "
                        "pass cannot override an opening bid"
                    )
                else:
                    try:
                        _forced_non_pass = compute_forced_non_pass_policy(
                            auction_tokens=list(tokens or []),
                            acting_direction=acting_dir,
                            dealer_actual=dealer_actual,
                            has_non_pass_choice=bool(_has_non_pass),
                            self_total_points=_self_tp_actual,
                            self_hcp=_self_hcp_actual,
                            self_hand=_self_hand_actual,
                        )
                    except Exception:
                        _forced_non_pass = {"hard_block": False, "reason": None}
                    if bool((_forced_non_pass or {}).get("hard_block")):
                        pass_hard_blocked = True
                        pass_hard_block_reason = str((_forced_non_pass or {}).get("reason") or "FORCED_NON_PASS_GAME_VALUES")

                # Apply perspective flip (same as non-Pass base), unless BT opening
                # policy hard-blocks pass while a legal level-1 opening exists.
                final_score = float(acting_sign) * float(score_val) + float(pass_bonus) - float(pass_penalty)
                if pass_hard_blocked:
                    final_score = float("-inf")
                return final_score, bid1, 0.0, 0.0, False, {
                    "pass_score": float(score_val),
                    "pass_source": _pass_source,
                    "pass_bonus": float(pass_bonus),
                    "pass_bonus_reason": pass_bonus_reason,
                    "pass_penalty": float(pass_penalty),
                    "pass_penalty_reason": pass_penalty_reason,
                    "pass_hard_blocked": pass_hard_blocked,
                    "pass_hard_block_reason": pass_hard_block_reason,
                    "score_phase": score_phase,
                    "has_level1_contract_choice": bool(has_level1_contract_choice),
                    "acting_sign": float(acting_sign),
                }

            # Try bid-feature cache for fast scoring (only useful when cache row
            # has actual stats — i.e. terminal/completed auctions that joined with
            # the stats table).  Non-terminal rows have NULL stats from the LEFT JOIN
            # and would score 0.0, so we must fall through to the full bid-details path.
            cache_row = None  # Disabled for strict parity with Streamlit Advanced scorer.
            if cache_row is not None:
                def _is_vul_for_acting(vul_val: Any, dir_to_act: str) -> bool:
                    s = str(vul_val or "").strip().upper()
                    if s in ("BOTH", "ALL", "V", "VUL"):
                        return True
                    if s in ("NONE", "NV", "NONVUL", "NON-VUL", ""):
                        return False
                    if s in ("NS",):
                        return dir_to_act in ("N", "S")
                    if s in ("EW",):
                        return dir_to_act in ("E", "W")
                    try:
                        i = int(s)
                        if i == 3:
                            return True
                        if i == 1:
                            return dir_to_act in ("N", "S")
                        if i == 2:
                            return dir_to_act in ("E", "W")
                        return False
                    except Exception:
                        return False

                def _pick_num(row: Dict[str, Any], keys: List[str]) -> float | None:
                    for k in keys:
                        v = row.get(k)
                        if v is None:
                            continue
                        try:
                            return float(v)
                        except Exception:
                            continue
                    return None

                use_v = _is_vul_for_acting(board_vul, acting_dir)
                suf = "V" if use_v else "NV"
                seat_i = int(seat_display)
                mean_par = _pick_num(
                    cache_row,
                    [f"mean_par_{suf.lower()}", f"Avg_Par_S{seat_i}_{suf}", "mean_par"],
                )
                matched_n = _pick_num(
                    cache_row,
                    [f"count_{suf.lower()}", f"Count_S{seat_i}_{suf}", "matching_deal_count", "matched_n"],
                )
                # Only use cache fast-path when we have real stats (non-NULL mean_par
                # AND count > 0).  Most BT rows are non-terminal and have NULL stats
                # from the LEFT JOIN — scoring them as 0.0 causes Pass to always win.
                if mean_par is not None and matched_n is not None and float(matched_n) > 0:
                    mean_par_val = float(mean_par)
                    n_val = float(matched_n)
                    shrinkage_k = 5.0
                    # Avg_Par is seat-relative since GPU pipeline v3.1.
                    base = mean_par_val
                    base_shrunk = (base * n_val) / (n_val + shrinkage_k)
                    breakdown = {
                        "cache_mode": "bid_feature_cache",
                        "base": round(base, 2),
                        "base_shrunk": round(base_shrunk, 2),
                        "matched_n": int(n_val),
                        "mean_par": round(mean_par_val, 2),
                        "acting_sign": float(acting_sign),
                        "final_score": round(float(base_shrunk), 2),
                    }
                    breakdown.update(_special_case_breakdown_fields(opt))
                    return float(base_shrunk), bid1, 0.0, 0.0, False, breakdown
                # else: cache row lacks stats → fall through to full bid-details scoring

            details = handle_bid_details(
                state=state,
                auction=str(auction_full or ""),
                bid=bid1,
                max_deals=int(max_deals),
                seed=int(seed or 0),
                vul_filter=str(board_vul or "") if board_vul is not None else None,
                deal_index=deal_index_i,
                topk=10,
                include_phase2a=True,
                include_timing=True,
            )
            d_ms = 0.0
            p2_ms = 0.0
            hit = bool((details.get("cache") or {}).get("hit", False))
            try:
                d_ms = float(details.get("elapsed_ms") or 0.0)
            except Exception:
                d_ms = 0.0
            try:
                p2_ms = float(((details.get("timing") or {}).get("phase2a_ms")) or 0.0)
            except Exception:
                p2_ms = 0.0
            if details.get("error") or details.get("message"):
                # --- v2 simple-raise rescue -----------------------------------
                # When the BT criteria match the deal (the bid survived criteria
                # filtering above) but the deal bitmap index has no matching
                # deals, the bid-details call returns an error/message.
                # In v2 mode, rescue the bid with a synthetic score when it is a
                # *simple raise* of partner's suit at the level justified by TP.
                if use_guardrails_v2 and bid_contract is not None:
                    _sr = _v2_simple_raise_rescue(
                        bid_text=bid1,
                        bid_level_local=bid_level,
                        bid_strain_local=bid_strain,
                        acting_direction=acting_dir,
                        acting_sign_local=acting_sign,
                        auction_tokens=tokens,
                        deal_row_local=deal_row,
                        opt_avg_par=opt.get("avg_par"),
                        base_phase_mult_local=base_phase_mult,
                    )
                    if _sr is not None:
                        return _sr[0], bid1, d_ms, p2_ms, hit, _sr[1]
                    _lc = _v2_level_acceptable_missing_criteria_rescue(
                        bid_text=bid1,
                        bid_level_local=bid_level,
                        bid_strain_local=bid_strain,
                        acting_direction=acting_dir,
                        acting_sign_local=acting_sign,
                        deal_row_local=deal_row,
                        opt_avg_par=opt.get("avg_par"),
                        base_phase_mult_local=base_phase_mult,
                        details_error_text=str(details.get("error") or details.get("message") or ""),
                    )
                    if _lc is not None:
                        return _lc[0], bid1, d_ms, p2_ms, hit, _lc[1]
                return float("-inf"), bid1, d_ms, p2_ms, hit, {"error": details.get("error") or details.get("message")}

            par_score = details.get("par_score") or {}
            mean_par = par_score.get("mean")
            bt_acting_criteria: list[str] = []
            try:
                _crit = details.get("bt_acting_criteria") or []
                if isinstance(_crit, list):
                    bt_acting_criteria = [str(x) for x in _crit if str(x or "").strip()]
            except Exception:
                bt_acting_criteria = []

            eeo = {}
            try:
                eeo = compute_eeo_from_bid_details(details)
            except Exception:
                eeo = {}
            utility = eeo.get("utility")

            opp_threat = None
            try:
                threat = (details.get("phase2a") or {}).get("threat") or {}
                vals = []
                for su in ["S", "H", "D", "C"]:
                    t_su = threat.get(su) or {}
                    v = t_su.get("p_them_6plus")
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                if vals:
                    opp_threat = max(vals)
            except Exception:
                opp_threat = None

            desc_score = None
            try:
                rp = details.get("range_percentiles")
                if isinstance(rp, dict) and rp.get("role") == "self":
                    pcts = []
                    h = (rp.get("hcp") or {}).get("percentile")
                    if isinstance(h, (int, float)):
                        pcts.append(float(h))
                    sl = rp.get("suit_lengths") or {}
                    for su in ["S", "H", "D", "C"]:
                        p = ((sl.get(su) or {}).get("percentile"))
                        if isinstance(p, (int, float)):
                            pcts.append(float(p))
                    if pcts:
                        dev = sum(abs(p - 50.0) for p in pcts) / len(pcts)
                        desc_score = max(0.0, min(1.0, 1.0 - (dev / 50.0)))
            except Exception:
                desc_score = None

            # Acting player's hand string (used by trick estimation + control check)
            _self_hand_str: str | None = None
            try:
                _self_hand_str = str(deal_row.get(f"Hand_{acting_dir}", "") or "").strip() or None
            except Exception:
                _self_hand_str = None

            def _suit_lengths_from_hand_str(hand_str: str | None) -> dict[str, int]:
                if not hand_str:
                    return {}
                try:
                    parts = str(hand_str).strip().split(".")
                    if len(parts) != 4:
                        return {}
                    return {
                        "S": len(parts[0]),
                        "H": len(parts[1]),
                        "D": len(parts[2]),
                        "C": len(parts[3]),
                    }
                except Exception:
                    return {}

            _self_suit_lengths = _suit_lengths_from_hand_str(_self_hand_str)
            _fit_us_hist: dict[str, Any] | None = None
            try:
                _fit_us_any = ((details.get("phase2a") or {}).get("fit") or {}).get("us")
                if isinstance(_fit_us_any, dict):
                    _fit_us_hist = _fit_us_any
            except Exception:
                _fit_us_hist = None

            # Hand controls for INSUFFICIENT_FIRST_ROUND_CONTROLS guardrail
            _self_aces: int | None = None
            _self_helpful_voids: int | None = None
            if _self_hand_str:
                _self_aces, _self_helpful_voids = hand_controls(_self_hand_str)

            # Trick estimation for guardrail (TRICKS_SHORTFALL component).
            # Uses the acting player's hand string + Phase2a partner posteriors.
            _est_tricks_api: float | None = None
            try:
                import re as _re_strain
                _strain_m = _re_strain.match(r"^[1-7]\s*(NT|N|[CDHS])", bid1)
                if _strain_m:
                    _bs = _strain_m.group(1).upper()
                    _bid_strain = "NT" if _bs in ("N", "NT") else _bs
                    if _self_hand_str:
                        _p2a_tricks = details.get("phase2a") or {}
                        _partner_hcp_h = ((_p2a_tricks.get("roles") or {}).get("partner") or {}).get("hcp_hist")
                        _partner_sl_h = ((_p2a_tricks.get("roles") or {}).get("partner") or {}).get("sl_hist")
                        _fit_us_h = (_p2a_tricks.get("fit") or {}).get("us")
                        _fit_us_hist = _fit_us_h if isinstance(_fit_us_h, dict) else _fit_us_hist
                        _trick_res = estimate_partnership_tricks(
                            self_hand=_self_hand_str,
                            partner_hcp_hist=_partner_hcp_h,
                            partner_sl_hists=_partner_sl_h,
                            fit_us_hists=_fit_us_h,
                            strain=_bid_strain,
                        )
                        _est_tricks_api = _trick_res.get("est_tricks")
            except Exception:
                _est_tricks_api = None

            # Guardrail penalty
            guard_penalty = 0.0
            guard_penalty_raw = 0.0
            guard_reasons: list[str] = []
            guard_phase: str | None = None
            guard_enable_underbid_checks = True
            guard_enable_tp_shortfall_check = True
            guard_enable_tricks_shortfall_check = True
            self_tp_val = None
            self_tp_source: str | None = None
            self_tp_actual = None
            partner_tp_hist_val = None
            partner_tp_expected = None
            partner_tp_min = None
            partner_tp_max = None
            self_hcp_actual: float | None = None
            self_hcp_est: float | None = None
            partner_hcp_expected: float | None = None
            partner_hcp_min: float | None = None
            partner_hcp_max: float | None = None
            partner_hcp_shown_floor: float | None = None
            par_topk: list[dict[str, Any]] = []
            non_rebiddable_rebid_penalty = 0.0
            non_rebiddable_rebid_reason: str | None = None
            rebiddable_major_game_bonus = 0.0
            rebiddable_major_game_reason: str | None = None
            partner_major_game_bonus = 0.0
            partner_major_detour_penalty = 0.0
            partner_major_reason: str | None = None
            try:
                self_tp_actual = deal_row.get(f"Total_Points_{acting_dir}")
                if self_tp_actual is not None:
                    self_tp_actual = float(self_tp_actual)
            except Exception:
                self_tp_actual = None
            try:
                _hcp_raw = deal_row.get(f"HCP_{acting_dir}")
                if _hcp_raw is not None:
                    self_hcp_actual = float(_hcp_raw)
            except Exception:
                self_hcp_actual = None
            try:
                if float(w_guard) > 0:
                    guard_phase = _classify_guardrail_phase(tokens, bid1)
                    guard_uncontested_early = guard_phase in ("opening", "early_uncontested")
                    # Keep policy freedom in early uncontested auctions for low-level
                    # developmental calls, but still enforce hard risk checks on
                    # high-level commitments (5+) where each extra trick is costly.
                    is_high_level_contract = bool(bid_level is not None and int(bid_level) >= 5)
                    guard_enable_underbid_checks = not guard_uncontested_early
                    guard_enable_tp_shortfall_check = (not guard_uncontested_early) or is_high_level_contract
                    guard_enable_tricks_shortfall_check = (not guard_uncontested_early) or is_high_level_contract
                    try:
                        rp2 = details.get("range_percentiles")
                        if isinstance(rp2, dict):
                            self_tp_val = (rp2.get("total_points") or {}).get("value")
                            if self_tp_val is not None:
                                self_tp_val = float(self_tp_val)
                                self_tp_source = "range_percentiles.total_points.value"
                            _hcp_est_raw = (rp2.get("hcp") or {}).get("value")
                            if _hcp_est_raw is not None:
                                self_hcp_est = float(_hcp_est_raw)
                    except Exception:
                        self_tp_val = None
                        self_tp_source = None

                    try:
                        p2a = details.get("phase2a") or {}
                        _partner_roles = (p2a.get("roles") or {}).get("partner") or {}
                        partner_tp_hist_val = _partner_roles.get("total_points_hist")
                        partner_tp_expected = expected_from_hist(partner_tp_hist_val)
                        partner_tp_min = _min_from_hist(partner_tp_hist_val)
                        partner_tp_max = _max_from_hist(partner_tp_hist_val)
                        _partner_hcp_hist = _partner_roles.get("hcp_hist")
                        partner_hcp_expected = expected_from_hist(_partner_hcp_hist)
                        partner_hcp_min = _min_from_hist(_partner_hcp_hist)
                        partner_hcp_max = _max_from_hist(_partner_hcp_hist)
                    except Exception:
                        partner_tp_hist_val = None
                        partner_tp_expected = None
                        partner_tp_min = None
                        partner_tp_max = None
                        partner_hcp_expected = None
                        partner_hcp_min = None
                        partner_hcp_max = None

                    try:
                        partner_hcp_shown_floor = _partner_shown_hcp_floor_from_nt_calls(
                            auction_tokens=list(tokens or []),
                            acting_direction=acting_dir,
                        )
                    except Exception:
                        partner_hcp_shown_floor = None

                    par_topk = (details.get("par_contracts") or {}).get("topk") or []

                    gp, guard_reasons = compute_guardrail_penalty(
                        bid=bid1,
                        par_contracts_topk=par_topk,
                        self_total_points=self_tp_val,
                        partner_tp_hist=partner_tp_hist_val,
                        par_score_mean=mean_par,
                        acting_sign=float(acting_sign),
                        w_overbid_level=float(w_guard_overbid),
                        w_tp_shortfall=float(w_guard_tp),
                        w_neg_par_high=float(w_guard_neg),
                        w_underbid_level=float(w_guard_underbid),
                        w_tp_surplus=float(w_guard_tp_surplus),
                        w_strain_ineff=float(w_guard_strain),
                        sacrifice_discount=float(w_guard_sacrifice),
                        est_tricks=_est_tricks_api,
                        w_tricks_shortfall=float(w_guard_tricks),
                        self_aces=_self_aces,
                        self_helpful_voids=_self_helpful_voids,
                        self_suit_lengths=_self_suit_lengths,
                        fit_us_hist=_fit_us_hist,
                        bt_acting_criteria=bt_acting_criteria,
                        is_reopening=bool(guard_phase == "reopening_or_balancing"),
                        enable_tp_shortfall_check=guard_enable_tp_shortfall_check,
                        enable_tricks_shortfall_check=guard_enable_tricks_shortfall_check,
                        enable_underbid_checks=guard_enable_underbid_checks,
                        is_raise_of_partner_suit=_is_raise_of_partner_suit(
                            tokens_now=tokens,
                            bid_text=bid1,
                            acting_direction=acting_dir,
                            dealer=dealer_actual,
                        ),
                        opp_shown_strains=_opp_shown_strains,
                        debug_equivalence_bypass=True,
                    )
                    guard_penalty_raw = float(gp)
                    guard_penalty = float(gp) * float(w_guard)
            except Exception:
                guard_penalty = 0.0
                guard_penalty_raw = 0.0
                guard_reasons = []

            # Additional bridge-texture guardrail:
            # Penalize same-player suit rebids that are not rebiddable by hand.
            try:
                if float(w_guard) > 0:
                    _nr_p, _nr_reason = compute_non_rebiddable_suit_rebid_penalty(
                        bid_text=bid1,
                        auction_tokens=tokens,
                        acting_direction=acting_dir,
                        dealer_actual=dealer_actual,
                        bt_acting_criteria=bt_acting_criteria,
                    )
                    non_rebiddable_rebid_penalty = float(_nr_p) * float(w_guard)
                    non_rebiddable_rebid_reason = _nr_reason
            except Exception:
                non_rebiddable_rebid_penalty = 0.0
                non_rebiddable_rebid_reason = None

            # Bonus for directly committing to game in a strongly rebiddable
            # major already shown by acting player.
            try:
                _rb_bonus, _rb_reason = compute_rebiddable_major_game_bonus(
                    bid_text=bid1,
                    auction_tokens=tokens,
                    acting_direction=acting_dir,
                    dealer_actual=dealer_actual,
                    bt_acting_criteria=bt_acting_criteria,
                )
                rebiddable_major_game_bonus = float(_rb_bonus)
                rebiddable_major_game_reason = _rb_reason
            except Exception:
                rebiddable_major_game_bonus = 0.0
                rebiddable_major_game_reason = None

            # Partner-major context: if partner has repeatedly shown a major,
            # prefer direct game in that major over side-suit detours.
            try:
                _pm_bonus, _pm_pen, _pm_reason = compute_partner_major_game_commit_adjustment(
                    bid_text=bid1,
                    auction_tokens=tokens,
                    acting_direction=acting_dir,
                    dealer_actual=dealer_actual,
                    bt_acting_criteria=bt_acting_criteria,
                )
                partner_major_game_bonus = float(_pm_bonus)
                partner_major_detour_penalty = float(_pm_pen)
                partner_major_reason = _pm_reason
            except Exception:
                partner_major_game_bonus = 0.0
                partner_major_detour_penalty = 0.0
                partner_major_reason = None

            nt_preference_bonus, minor_nt_detour_penalty, nt_preference_reason = _compute_nt_preference_adjustment(
                bid_text=bid1,
                bid_level_local=bid_level,
                options=list(passed_opts or []),
                auction_tokens=list(tokens or []),
                acting_direction=acting_dir,
                fit_us_hist=_fit_us_hist,
                par_topk_rows=list(par_topk or []),
                acting_sign_local=float(acting_sign),
            )
            nt_over_minor_bonus, nt_over_minor_penalty, nt_over_minor_reason = _compute_nt_over_minor_surplus_adjustment(
                bid_text=bid1,
                bid_level_local=bid_level,
                bid_strain_local=bid_strain,
                self_hcp=self_hcp_actual if isinstance(self_hcp_actual, (int, float)) else None,
                partner_hcp_est=partner_hcp_expected if isinstance(partner_hcp_expected, (int, float)) else None,
                self_tp=self_tp_actual if isinstance(self_tp_actual, (int, float)) else None,
                partner_tp_est=partner_tp_expected if isinstance(partner_tp_expected, (int, float)) else None,
            )
            pull_nt_game_penalty, pull_nt_game_reason = 0.0, None
            try:
                pull_nt_game_penalty, pull_nt_game_reason = _compute_pull_nt_game_penalty(
                    bid_text=bid1,
                    bid_level_local=bid_level,
                    bid_strain_local=bid_strain,
                    auction_tokens=list(tokens or []),
                    acting_direction=acting_dir,
                )
            except Exception:
                pull_nt_game_penalty = 0.0
                pull_nt_game_reason = None
            forcing_heart_fit_bonus, forcing_heart_detour_penalty, forcing_heart_reason = _compute_forcing_heart_state_adjustment(
                bid_text=bid1,
                bid_level_local=bid_level,
                bid_strain_local=bid_strain,
                acting_direction=acting_dir,
            )
            forced_major_game_commit_bonus, forced_major_game_commit_penalty, forced_major_game_commit_reason = _compute_forced_major_game_commit_adjustment(
                bid_text=bid1,
                bid_level_local=bid_level,
                bid_strain_local=bid_strain,
                auction_tokens=list(tokens or []),
                acting_direction=acting_dir,
            )
            takeout_double_explore_bonus, takeout_double_explore_penalty, takeout_double_explore_reason = _compute_takeout_double_game_explore_adjustment(
                bid_text=bid1,
                bid_level_local=bid_level,
                bid_strain_local=bid_strain,
                options=list(passed_opts or []),
                bid_agg_expr=list(_agg_expr_by_bid.get(str(bid1 or "").strip().upper(), []) or []),
                auction_tokens=list(tokens or []),
                acting_direction=acting_dir,
                self_total_points_est=self_tp_val if isinstance(self_tp_val, (int, float)) else None,
                self_total_points_actual=self_tp_actual if isinstance(self_tp_actual, (int, float)) else None,
            )
            takeout_double_trigger_bonus, takeout_double_trigger_penalty, takeout_double_trigger_reason = _compute_takeout_double_trigger_adjustment(
                bid_text=bid1,
                options=list(passed_opts or []),
                auction_tokens=list(tokens or []),
                acting_direction=acting_dir,
                self_total_points_est=self_tp_val if isinstance(self_tp_val, (int, float)) else None,
                self_total_points_actual=self_tp_actual if isinstance(self_tp_actual, (int, float)) else None,
                self_suit_lengths=_self_suit_lengths if isinstance(_self_suit_lengths, dict) else None,
            )
            common_sense_bonus = 0.0
            common_sense_penalty = 0.0
            common_sense_reason_codes: list[str] = []
            common_sense_evidence: Dict[str, Any] = {}
            if use_common_sense:
                try:
                    _cs = compute_common_sense_adjustments(
                        bid_text=bid1,
                        auction_tokens=list(tokens or []),
                        acting_direction=acting_dir,
                        dealer_actual=dealer_actual,
                        self_total_points=(
                            self_tp_val if isinstance(self_tp_val, (int, float))
                            else (self_tp_actual if isinstance(self_tp_actual, (int, float)) else None)
                        ),
                        partner_total_points_expected=(
                            partner_tp_expected if isinstance(partner_tp_expected, (int, float)) else None
                        ),
                        self_suit_lengths=_self_suit_lengths if isinstance(_self_suit_lengths, dict) else None,
                        fit_us_hist=_fit_us_hist if isinstance(_fit_us_hist, dict) else None,
                    )
                    common_sense_bonus = float((_cs or {}).get("bonus") or 0.0)
                    common_sense_penalty = float((_cs or {}).get("penalty") or 0.0)
                    common_sense_reason_codes = list((_cs or {}).get("reason_codes") or [])
                    common_sense_evidence = dict((_cs or {}).get("evidence") or {})
                except Exception:
                    common_sense_bonus = 0.0
                    common_sense_penalty = 0.0
                    common_sense_reason_codes = []
                    common_sense_evidence = {}

            # Room-consumption + strain-change context:
            # jumping to level-5 in a new strain consumes bidding room that could
            # otherwise be used to explore fit/strain at lower levels.
            room_consumption_penalty = 0.0
            room_consumption_reason: str | None = None
            is_level5_strain_change_jump = False
            try:
                if bid_level == 5 and bid_strain in ("C", "D", "H", "S", "N"):
                    acting_side = _dir_side(acting_dir)
                    last_side_contract: tuple[int, str] | None = None
                    for i, tk in enumerate(list(tokens or [])):
                        _c = _parse_contract_bid_text(str(tk or "").strip().upper())
                        if _c is None:
                            continue
                        if _dir_side(_token_bidder_dir(i)) == acting_side:
                            last_side_contract = (int(_c[0]), str(_c[1]))
                    if last_side_contract is not None:
                        prev_lvl, prev_st = last_side_contract
                        is_level5_strain_change_jump = bool(prev_lvl <= 4 and str(prev_st) != str(bid_strain))
                    else:
                        # No prior side contract but direct level-5 contract is still
                        # a high room-consuming commitment.
                        is_level5_strain_change_jump = True

                    if is_level5_strain_change_jump:
                        lower_exploration_available = False
                        for _o in list(passed_opts or []):
                            _bo = str((_o or {}).get("bid", "") or "").strip().upper()
                            if not _bo or _bo == bid1 or _bo in ("P", "PASS", "X", "XX"):
                                continue
                            _co = _parse_contract_bid_text(_bo)
                            if _co is None:
                                continue
                            if int(_co[0]) <= 3:
                                lower_exploration_available = True
                                break
                        if lower_exploration_available:
                            room_consumption_penalty = 120.0
                            room_consumption_reason = (
                                f"LEVEL5_ROOM_CONSUMPTION: {bid1} is a level-5 strain-change jump "
                                f"while lower-level exploration bids are available (-{room_consumption_penalty:.0f})"
                            )
            except Exception:
                room_consumption_penalty = 0.0
                room_consumption_reason = None
                is_level5_strain_change_jump = False

            post_game_slam_gate_penalty = 0.0
            p_slam_tp_ge_33 = None
            combined_tp_mean = None
            game_contract_on_table = False
            post_game_slam_gate_reason: str | None = None
            v2_jump_past_game_hard_block = False
            v2_jump_past_game_reason: str | None = None
            slam_hard_block = False
            slam_hard_block_reason: str | None = None
            slam_likely_make = False
            slam_likely_make_reason: str | None = None
            _ev_acting: float | None = None
            _ev_floor: float | None = None
            _ev_source: str | None = None
            try:
                _slam_adj = compute_post_game_slam_gate_adjustment(
                    bid_text=bid1,
                    auction_tokens=tokens,
                    acting_direction=acting_dir,
                    dealer_actual=dealer_actual,
                    self_total_points=self_tp_val,
                    partner_tp_hist=partner_tp_hist_val,
                )
                post_game_slam_gate_penalty = float(_slam_adj.get("penalty") or 0.0)
                p_slam_tp_ge_33 = _slam_adj.get("p_slam_tp_ge_33")
                combined_tp_mean = _slam_adj.get("combined_tp_mean")
                game_contract_on_table = bool(_slam_adj.get("game_contract_on_table"))
                post_game_slam_gate_reason = _slam_adj.get("reason")
                try:
                    _strain_m2 = re.match(r"^[1-7]\s*(NT|N|[CDHS])", bid1)
                    _bid_strain2 = None
                    if _strain_m2:
                        _bs2 = _strain_m2.group(1).upper()
                        _bid_strain2 = "NT" if _bs2 in ("N", "NT") else _bs2
                    _required_tricks = (int(bid_level) + 6) if bid_level is not None else None
                    _tricks_ok = bool(
                        _required_tricks is not None
                        and _est_tricks_api is not None
                        and float(_est_tricks_api) >= float(_required_tricks) - 0.5
                    )
                    _is_grand = bool(bid_level is not None and int(bid_level) >= 7)
                    _tp_ok = bool(
                        combined_tp_mean is not None
                        and float(combined_tp_mean) >= (35.0 if _is_grand else 32.0)
                    )
                    _slam_prob_ok = bool(
                        p_slam_tp_ge_33 is not None
                        and float(p_slam_tp_ge_33) >= (0.70 if _is_grand else 0.45)
                    )
                    _avg_ev_opt = None
                    try:
                        _raw_avg_ev_opt = opt.get("avg_ev")
                        if _raw_avg_ev_opt is not None:
                            _avg_ev_opt = float(_raw_avg_ev_opt)
                    except Exception:
                        _avg_ev_opt = None
                    _ev_acting = None
                    _ev_source = None
                    # Prefer deal-specific EV for this exact candidate auction;
                    # fallback to aggregated avg_ev only when needed.
                    try:
                        _cand_auction = "-".join([*(str(t).strip().upper() for t in tokens), bid1])
                        _ev_deal_ns = get_ev_for_auction(_cand_auction, dealer_actual, deal_row)
                        if _ev_deal_ns is not None:
                            _ev_acting = float(acting_sign) * float(_ev_deal_ns)
                            _ev_source = "deal_ev"
                    except Exception:
                        _ev_acting = None
                    if _ev_acting is None and _avg_ev_opt is not None:
                        _ev_acting = float(acting_sign) * float(_avg_ev_opt)
                        _ev_source = "avg_ev"
                    _ev_floor = 900.0 if (bid_level is not None and int(bid_level) >= 7) else 420.0
                    _ev_ok = bool(_ev_acting is not None and float(_ev_acting) >= float(_ev_floor))
                    _par_topk_support = False
                    if _bid_strain2 and isinstance(par_topk, list) and len(par_topk) > 0:
                        for _r in par_topk:
                            if not isinstance(_r, dict):
                                continue
                            _c = str(_r.get("contract") or "").strip().upper()
                            _m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", _c)
                            if not _m:
                                continue
                            _lvl = int(_m.group(1))
                            _st = "NT" if _m.group(2).upper() in ("N", "NT") else _m.group(2).upper()
                            if bid_level is not None and _lvl == int(bid_level) and _st == _bid_strain2:
                                try:
                                    _prob = float(_r.get("prob") or 0.0)
                                    _aps = float(_r.get("avg_par_score") or 0.0)
                                except Exception:
                                    _prob = 0.0
                                    _aps = 0.0
                                _par_topk_support = bool(_prob >= 0.10 and _aps > 0)
                                break
                    if _is_grand:
                        # Grand slams need explicit structural evidence, not just top-k support.
                        _structure_ok = bool(_tricks_ok and _tp_ok and _slam_prob_ok)
                    else:
                        _structure_ok = bool((_tricks_ok and _tp_ok) or (_slam_prob_ok and _tp_ok) or _par_topk_support)
                    # EV must agree that this level is likely make / value-positive.
                    slam_likely_make = bool(_ev_ok and _structure_ok)
                    if slam_likely_make:
                        _reasons = []
                        if _ev_ok:
                            _reasons.append("ev")
                        if _tricks_ok:
                            _reasons.append("tricks")
                        if _tp_ok:
                            _reasons.append("tp")
                        if _slam_prob_ok:
                            _reasons.append("p33")
                        if _par_topk_support:
                            _reasons.append("topk")
                        slam_likely_make_reason = ",".join(_reasons) if _reasons else "evidence"
                    if (
                        post_game_slam_gate_penalty > 0
                        and isinstance(post_game_slam_gate_reason, str)
                        and (
                            post_game_slam_gate_reason.startswith("PRE_GAME_DIRECT_SLAM_COMMIT_GATE")
                            or post_game_slam_gate_reason.startswith("PRE_GAME_5M_SLAM_TRY_GATE")
                            or post_game_slam_gate_reason.startswith("PRE_GAME_5NT_SLAM_TRY_GATE")
                            or (
                                post_game_slam_gate_reason.startswith("POST_GAME_SLAM_GATE")
                                and str(bid1).strip().upper() in ("4N", "4NT", "5N", "5NT")
                            )
                        )
                    ):
                        slam_hard_block = not slam_likely_make
                        if slam_hard_block:
                            slam_hard_block_reason = "SLAM_GATE_WITHOUT_EV_MAKE_EVIDENCE"
                    if is_level5_strain_change_jump and not slam_likely_make:
                        slam_hard_block = True
                        slam_hard_block_reason = "LEVEL5_STRAIN_CHANGE_WITHOUT_EV_MAKE_EVIDENCE"
                except Exception:
                    slam_hard_block = False
                    slam_hard_block_reason = None
                    slam_likely_make = False
                    slam_likely_make_reason = None
            except Exception:
                post_game_slam_gate_penalty = 0.0
                p_slam_tp_ge_33 = None
                combined_tp_mean = None
                game_contract_on_table = False
                post_game_slam_gate_reason = None
                v2_jump_past_game_hard_block = False
                v2_jump_past_game_reason = None
                slam_hard_block = False
                slam_hard_block_reason = None
                slam_likely_make = False
                slam_likely_make_reason = None

            # Guardrails v2:
            # Reject same-side same-strain jumps past game (below slam) unless
            # bidder hand + partner max inferred TP can plausibly support slam.
            v2_diag: Dict[str, Any] = {
                "v2_bid_level": bid_level,
                "v2_bid_strain": bid_strain,
                "v2_prev_same_side_same_strain_level": None,
                "v2_game_level_for_strain": None,
                "v2_jump_detected": False,
                "v2_required_tp": None,
                "v2_self_tp_used": None,
                "v2_point_type": None,
                "v2_partner_tp_floor": partner_tp_min if isinstance(partner_tp_min, (int, float)) else None,
                "v2_partner_tp_ceiling": partner_tp_max if isinstance(partner_tp_max, (int, float)) else None,
                "v2_combined_tp_floor": None,
                "v2_combined_tp_ceiling": None,
                "v2_slam_qualified": False,
                "v2_slam_rejected_reason": None,
                "v2_blocked": False,
                "v2_block_reason": None,
            }
            if use_guardrails_v2:
                v2_diag = _guardrails_v2_jump_past_game_diagnostics(
                    auction_tokens=list(tokens or []),
                    acting_direction=acting_dir,
                    bid_level_local=bid_level,
                    bid_strain_local=bid_strain,
                    self_tp_est=(float(self_tp_val) if isinstance(self_tp_val, (int, float)) else None),
                    self_tp_act=(float(self_tp_actual) if isinstance(self_tp_actual, (int, float)) else None),
                    partner_tp_floor_local=(float(partner_tp_min) if isinstance(partner_tp_min, (int, float)) else None),
                    partner_tp_ceiling_local=(float(partner_tp_max) if isinstance(partner_tp_max, (int, float)) else None),
                    self_hcp_est_local=(float(self_hcp_est) if isinstance(self_hcp_est, (int, float)) else None),
                    self_hcp_act_local=(float(self_hcp_actual) if isinstance(self_hcp_actual, (int, float)) else None),
                    partner_hcp_floor_local=(float(partner_hcp_min) if isinstance(partner_hcp_min, (int, float)) else None),
                    partner_hcp_ceiling_local=(float(partner_hcp_max) if isinstance(partner_hcp_max, (int, float)) else None),
                )
                v2_jump_past_game_hard_block = bool(v2_diag.get("v2_blocked", False))
                v2_jump_past_game_reason = v2_diag.get("v2_block_reason")

            _pref_exempt = bool(opt.get("_forced_preference", False))

            if v2_jump_past_game_hard_block and not _pref_exempt:
                slam_hard_block = True
                slam_hard_block_reason = v2_jump_past_game_reason or "GUARDRAILS_V2_JUMP_PAST_GAME_BLOCK"

            # Universal level-cap hard block: prevent any bid whose level
            # exceeds what combined partnership points can support.
            # HCP for NT, Total Points for suit contracts.
            # Runs in ALL logic modes (including ai_bt_only).
            # Skipped for forced-preference bids (obligated by convention).
            _level_cap_blocked = False
            if (not slam_hard_block) and (not _pref_exempt) and bid_level is not None and int(bid_level) >= 1:
                try:
                    _is_nt_cap = bool(bid_strain in ("N", "NT"))
                    if _is_nt_cap:
                        _self_pts_cap = float(self_hcp_actual) if isinstance(self_hcp_actual, (int, float)) else None
                        _partner_pts_cap = float(partner_hcp_expected) if isinstance(partner_hcp_expected, (int, float)) else None
                        if isinstance(partner_hcp_shown_floor, (int, float)):
                            _shown_floor = float(partner_hcp_shown_floor)
                            _partner_pts_cap = max(_partner_pts_cap, _shown_floor) if _partner_pts_cap is not None else _shown_floor
                        _pt_label = "HCP"
                    else:
                        _self_pts_cap = float(self_tp_actual) if isinstance(self_tp_actual, (int, float)) else None
                        _partner_pts_cap = float(partner_tp_expected) if isinstance(partner_tp_expected, (int, float)) else None
                        _pt_label = "TP"
                    if _self_pts_cap is not None and _partner_pts_cap is not None:
                        _combined_pts_cap = _self_pts_cap + _partner_pts_cap
                        _required_pts_cap = 14.0 + 3.0 * float(int(bid_level))
                        _max_level_cap = max(0, min(7, int((_combined_pts_cap - 14.0) // 3.0)))
                        if int(bid_level) > _max_level_cap:
                            slam_hard_block = True
                            _level_cap_blocked = True
                            slam_hard_block_reason = (
                                f"LEVEL_CAP_BLOCK: combined {_pt_label} "
                                f"{_combined_pts_cap:.0f} (self {_self_pts_cap:.0f} + "
                                f"partner ~{_partner_pts_cap:.0f}) cannot support "
                                f"level {bid_level} (needs {_required_pts_cap:.0f}, "
                                f"max level {_max_level_cap})"
                            )
                except Exception:
                    pass

            if (not slam_hard_block) and (not _pref_exempt) and bid_level is None:
                _is_double = bool(bid1 in ("D", "X", "DOUBLE"))
                if _is_double:
                    _self_tp_dbl = float(self_tp_actual) if isinstance(self_tp_actual, (int, float)) else None
                    _self_hcp_dbl = float(self_hcp_actual) if isinstance(self_hcp_actual, (int, float)) else None
                    _partner_tp_dbl = float(partner_tp_expected) if isinstance(partner_tp_expected, (int, float)) else None
                    if _self_tp_dbl is not None and _partner_tp_dbl is not None:
                        _combined_tp_dbl = _self_tp_dbl + _partner_tp_dbl
                        _dbl_blocked = False
                        _dbl_block_reason = ""
                        if _combined_tp_dbl < 14.0:
                            _dbl_blocked = True
                            _dbl_block_reason = (
                                f"DOUBLE_MIN_POINTS_BLOCK: combined TP "
                                f"{_combined_tp_dbl:.0f} (self {_self_tp_dbl:.0f} + "
                                f"partner ~{_partner_tp_dbl:.0f}) below minimum 14 "
                                f"for Double"
                            )
                        if not _dbl_blocked and _self_hcp_dbl is not None:
                            _dbl_last_level = 0
                            _dbl_last_strain = ""
                            for _dt in reversed(tokens):
                                _dtu = str(_dt or "").strip().upper()
                                if _dtu in ("P", "PASS", "", "D", "X", "DOUBLE", "R", "XX", "REDOUBLE"):
                                    continue
                                if len(_dtu) >= 2 and _dtu[0].isdigit():
                                    _dbl_last_level = int(_dtu[0])
                                    _dbl_last_strain = _dtu[1:]
                                break
                            _dbl_is_game = (
                                (_dbl_last_strain in ("NT", "N") and _dbl_last_level >= 3)
                                or (_dbl_last_strain in ("H", "S") and _dbl_last_level >= 4)
                                or (_dbl_last_strain in ("C", "D") and _dbl_last_level >= 5)
                            )
                            if _dbl_is_game and _self_hcp_dbl < 10:
                                _dbl_blocked = True
                                _dbl_block_reason = (
                                    f"DOUBLE_MIN_POINTS_BLOCK: self HCP "
                                    f"{_self_hcp_dbl:.0f} below minimum 10 "
                                    f"to double game contract "
                                    f"{_dbl_last_level}{_dbl_last_strain}"
                                )
                        if _dbl_blocked:
                            slam_hard_block = True
                            _level_cap_blocked = True
                            slam_hard_block_reason = _dbl_block_reason

            # Hard block: pulling a fitted-NT game (3NT) to an unbid suit.
            # "Fitted NT" means both partnership members have bid NT in this
            # auction.  With no suit fit established, pulling to a suit at the
            # 4-level or higher is almost never correct.
            if (
                (not slam_hard_block)
                and (not _pref_exempt)
                and bid_level is not None
                and int(bid_level) >= 4
                and bid_strain in ("C", "D", "H", "S")
            ):
                try:
                    _pnt_partner = _partner_dir(acting_dir)
                    _pnt_self_bid_nt = False
                    _pnt_partner_bid_nt = False
                    _pnt_suit_bid_by_us: set[str] = set()
                    for _pnt_i, _pnt_tk in enumerate(tokens):
                        _pnt_dir = _token_bidder_dir_for_dealer(_pnt_i, dealer_actual)
                        _pnt_u = str(_pnt_tk or "").strip().upper()
                        if not _pnt_u or _pnt_u in ("P", "PASS", "D", "X", "DOUBLE", "R", "XX", "REDOUBLE"):
                            continue
                        _pnt_m = re.match(r"^([1-7])\s*(NT|N|[CDHS])", _pnt_u)
                        if not _pnt_m:
                            continue
                        _pnt_st = _pnt_m.group(2).upper()
                        _pnt_is_nt = _pnt_st in ("N", "NT")
                        if _pnt_dir == acting_dir:
                            if _pnt_is_nt:
                                _pnt_self_bid_nt = True
                            else:
                                _pnt_suit_bid_by_us.add(_pnt_st)
                        elif _pnt_dir == _pnt_partner:
                            if _pnt_is_nt:
                                _pnt_partner_bid_nt = True
                            else:
                                _pnt_suit_bid_by_us.add(_pnt_st)
                    if _pnt_self_bid_nt and _pnt_partner_bid_nt and bid_strain not in _pnt_suit_bid_by_us:
                        slam_hard_block = True
                        _level_cap_blocked = True
                        slam_hard_block_reason = (
                            f"PULL_FITTED_NT_BLOCK: both partners bid NT; "
                            f"pulling to {bid_level}{bid_strain} with no "
                            f"established {bid_strain} fit is not allowed"
                        )
                except Exception:
                    pass

            # Hard block: bidding opponent's naturally-shown suit without
            # sufficient length.  The soft OPP_SUIT_TRESPASS guard penalty
            # in compute_guardrail_penalty often can't overcome inflated EV
            # deltas, so we hard-block here when all of:
            #   (a) the bid strain was naturally shown by an opponent,
            #   (b) the bidder holds < 5 cards in that strain,
            #   (c) the BT criteria don't include Rebiddable/Twice_Rebiddable
            #       for that strain (which would indicate a legitimate
            #       competitive overcall or cue).
            #
            # Exempt: artificial probes (Stayman-like 2C/3C), NT bids,
            # doubles, passes, and forced-preference bids.
            #
            # Potential exceptions to monitor:
            #   - Michaels / Unusual 2NT cue-bids (opponent's suit = asking,
            #     not natural).  Currently BT criteria for those paths
            #     typically include suit-length constraints so they pass (c).
            #   - Responsive/competitive doubles later converted into cue-bid
            #     suit calls — these should show suit length >= 5 or have
            #     rebiddable criteria, so they won't be blocked.
            #   - Fit-showing cue-bids (e.g. 3S after partner opens 1H and
            #     RHO bids 1S, meaning "I have a spade control and heart
            #     support").  These are rare in GIB data and typically lack
            #     suit-length criteria, so they *will* be blocked.  If this
            #     becomes an issue, add an exemption for single-level
            #     cue-bids that also constrain partner's suit length.
            if (
                (not slam_hard_block)
                and (not _pref_exempt)
                and bid_strain in ("C", "D", "H", "S")
                and isinstance(_opp_shown_strains, set)
                and bid_strain in _opp_shown_strains
            ):
                _ost_self_len = (
                    int(_self_suit_lengths.get(bid_strain, -1))
                    if isinstance(_self_suit_lengths, dict) else -1
                )
                _ost_has_rebiddable = False
                if bt_acting_criteria:
                    _ost_crit_upper = " ".join(str(c) for c in bt_acting_criteria).upper()
                    _ost_has_rebiddable = (
                        f"TWICE_REBIDDABLE_{bid_strain}" in _ost_crit_upper
                        or f"REBIDDABLE_{bid_strain}" in _ost_crit_upper
                    )
                _ost_is_artificial = False
                try:
                    if bid1 in ("2C", "3C") and bt_acting_criteria:
                        _ost_crits_joined = " ".join(str(c or "").upper() for c in bt_acting_criteria)
                        if "SL_H" in _ost_crits_joined and "SL_S" in _ost_crits_joined and not re.search(r"\bSL_C\b", _ost_crits_joined):
                            _ost_is_artificial = True
                except Exception:
                    pass
                if (
                    (not _ost_has_rebiddable)
                    and (not _ost_is_artificial)
                    and 0 <= _ost_self_len < 5
                ):
                    slam_hard_block = True
                    _level_cap_blocked = True
                    slam_hard_block_reason = (
                        f"OPP_SUIT_TRESPASS_BLOCK: bid {bid1} in {bid_strain} "
                        f"which opponent has naturally shown; self has "
                        f"{_ost_self_len} card{'s' if _ost_self_len != 1 else ''} "
                        f"(need 5+ or rebiddable criteria to compete in "
                        f"opponent's suit)"
                    )

            if ((not use_bt_only_scoring) or use_guardrails_v2 or _level_cap_blocked) and slam_hard_block:
                breakdown = {
                    "base": None,
                    "base_shrunk": None,
                    "base_phase_mult": round(float(base_phase_mult), 4),
                    "base_shrunk_weighted": None,
                    "matched_n": None,
                    "mean_par": round(float(mean_par), 2) if mean_par is not None else None,
                    "desc_score": round(float(desc_score), 4) if desc_score is not None else None,
                    "desc_term": None,
                    "w_desc_contrib": None,
                    "opp_threat": round(float(opp_threat), 4) if opp_threat is not None else None,
                    "w_threat_contrib": None,
                    "guard_penalty": None,
                    "guard_penalty_raw": None,
                    "guard_weight": round(float(w_guard), 4),
                    "guard_reasons": list(guard_reasons or []),
                    "guard_phase": guard_phase,
                    "score_phase": score_phase,
                    "policy_bonus": 0.0,
                    "policy_bonus_reason": None,
                    "guard_inputs": {
                        "self_total_points_used": round(float(self_tp_val), 2) if self_tp_val is not None else None,
                        "self_total_points_source": self_tp_source,
                        "self_total_points_actual": round(float(self_tp_actual), 2) if self_tp_actual is not None else None,
                        "partner_total_points_expected": round(float(partner_tp_expected), 2) if partner_tp_expected is not None else None,
                        "partner_total_points_floor": round(float(partner_tp_min), 2) if partner_tp_min is not None else None,
                        "partner_total_points_hist": partner_tp_hist_val,
                        "par_contracts_topk": par_topk,
                        "acting_direction": acting_dir,
                        "enable_underbid_checks": guard_enable_underbid_checks,
                        "enable_tp_shortfall_check": guard_enable_tp_shortfall_check,
                        "enable_tricks_shortfall_check": guard_enable_tricks_shortfall_check,
                    },
                    "non_rebiddable_rebid_penalty": round(float(non_rebiddable_rebid_penalty), 2),
                    "non_rebiddable_rebid_reason": non_rebiddable_rebid_reason,
                    "post_game_slam_gate_penalty": round(float(post_game_slam_gate_penalty), 2),
                    "post_game_slam_gate_reason": post_game_slam_gate_reason,
                    "rebiddable_major_game_bonus": round(float(rebiddable_major_game_bonus), 2),
                    "rebiddable_major_game_reason": rebiddable_major_game_reason,
                    "partner_major_game_bonus": round(float(partner_major_game_bonus), 2),
                    "partner_major_detour_penalty": round(float(partner_major_detour_penalty), 2),
                    "partner_major_reason": partner_major_reason,
                    "common_sense_bonus": round(float(common_sense_bonus), 2),
                    "common_sense_penalty": round(float(common_sense_penalty), 2),
                    "common_sense_reason_codes": list(common_sense_reason_codes or []),
                    "common_sense_evidence": common_sense_evidence,
                    "common_sense_hard_override": False,
                    "game_contract_on_table": bool(game_contract_on_table),
                    "p_slam_tp_ge_33": round(float(p_slam_tp_ge_33), 4) if p_slam_tp_ge_33 is not None else None,
                    "combined_tp_mean": round(float(combined_tp_mean), 2) if combined_tp_mean is not None else None,
                    "est_tricks": round(float(_est_tricks_api), 2) if _est_tricks_api is not None else None,
                    "utility": round(float(utility), 2) if utility is not None else None,
                    "acting_sign": float(acting_sign),
                    "hard_blocked": True,
                    "hard_block_reason": slam_hard_block_reason or "SLAM_OR_LEVEL5_GATE_WITHOUT_EVIDENCE",
                    "v2_jump_detected": bool(v2_diag.get("v2_jump_detected", False)),
                    "v2_point_type": v2_diag.get("v2_point_type"),
                    "v2_required_tp": v2_diag.get("v2_required_tp"),
                    "v2_self_tp_used": v2_diag.get("v2_self_tp_used"),
                    "v2_partner_tp_floor": v2_diag.get("v2_partner_tp_floor"),
                    "v2_partner_tp_ceiling": v2_diag.get("v2_partner_tp_ceiling"),
                    "v2_combined_tp_floor": v2_diag.get("v2_combined_tp_floor"),
                    "v2_combined_tp_ceiling": v2_diag.get("v2_combined_tp_ceiling"),
                    "v2_slam_qualified": bool(v2_diag.get("v2_slam_qualified", False)),
                    "v2_slam_rejected_reason": v2_diag.get("v2_slam_rejected_reason"),
                    "v2_blocked": bool(v2_diag.get("v2_blocked", False)),
                    "v2_block_reason": v2_diag.get("v2_block_reason"),
                    "slam_likely_make": bool(slam_likely_make),
                    "slam_likely_make_reason": slam_likely_make_reason,
                    "slam_likely_make_ev_acting": round(float(_ev_acting), 2) if "_ev_acting" in locals() and _ev_acting is not None else None,
                    "slam_likely_make_ev_floor": round(float(_ev_floor), 2) if "_ev_floor" in locals() and _ev_floor is not None else None,
                    "slam_likely_make_ev_source": _ev_source if "_ev_source" in locals() else None,
                    "final_score": None,
                }
                breakdown.update(_special_case_breakdown_fields(opt))
                return float("-inf"), bid1, d_ms, p2_ms, hit, breakdown

            try:
                # Bid-details `mean_par` is already normalized into acting-side utility.
                mean_par_val = float(mean_par) if mean_par is not None else 0.0
                base = mean_par_val
                
                # Bayesian Shrinkage for low-N bids
                matched_n = float(details.get("matched_deals_total_excluding_pinned") or details.get("matched_deals_total") or 0.0)
                shrinkage_k = 5.0
                base_shrunk = (base * matched_n) / (matched_n + shrinkage_k)
                base_shrunk_weighted = float(base_shrunk) * float(base_phase_mult)
                
                desc_term = float(desc_score - 0.5) if desc_score is not None else 0.0
                threat_term = float(opp_threat) if opp_threat is not None else 0.0
                policy_bonus = 0.0
                policy_bonus_reason: str | None = None
                if score_phase in ("opening", "early_uncontested") and is_level1_contract:
                    policy_bonus = float(opening_policy_bonus)
                    policy_bonus_reason = "EARLY_LEVEL1_POLICY_BONUS: valid level-1 contract in early phase"
                v2_current_ev_ns = _current_contract_ev_ns() if use_guardrails_v2 else None
                v2_current_ev_acting = (
                    float(acting_sign) * float(v2_current_ev_ns) if v2_current_ev_ns is not None else None
                )

                if use_bt_only_scoring:
                    score_val = float(base_shrunk_weighted)
                    early_phase = score_phase in ("opening", "early_uncontested")
                    if use_guardrails_v2 and _ev_acting is not None and v2_current_ev_acting is not None and not early_phase:
                        score_val = float(_ev_acting) - float(v2_current_ev_acting)
                    score_val += float(nt_over_minor_bonus) - float(nt_over_minor_penalty)
                    score_val -= float(pull_nt_game_penalty)
                else:
                    # Use full scoring stack (optionally with common-sense stage disabled later).
                    score_val = (
                        base_shrunk_weighted
                        + float(w_desc) * desc_term
                        - float(w_threat) * threat_term
                        - guard_penalty
                        - float(non_rebiddable_rebid_penalty)
                        - float(post_game_slam_gate_penalty)
                        - float(room_consumption_penalty)
                        + float(rebiddable_major_game_bonus)
                        + float(partner_major_game_bonus)
                        - float(partner_major_detour_penalty)
                        + float(forcing_heart_fit_bonus)
                        - float(forcing_heart_detour_penalty)
                        + float(forced_major_game_commit_bonus)
                        - float(forced_major_game_commit_penalty)
                        + float(takeout_double_trigger_bonus)
                        - float(takeout_double_trigger_penalty)
                        + float(takeout_double_explore_bonus)
                        - float(takeout_double_explore_penalty)
                        + float(nt_preference_bonus)
                        - float(minor_nt_detour_penalty)
                        + float(nt_over_minor_bonus)
                        - float(nt_over_minor_penalty)
                        - float(pull_nt_game_penalty)
                        + float(policy_bonus)
                    )
                breakdown = {
                    "base": round(base, 2),
                    "base_shrunk": round(base_shrunk, 2),
                    "base_phase_mult": round(float(base_phase_mult), 4),
                    "base_shrunk_weighted": round(float(base_shrunk_weighted), 2),
                    "matched_n": int(matched_n),
                    "mean_par": round(float(mean_par), 2) if mean_par is not None else None,
                    "desc_score": round(float(desc_score), 4) if desc_score is not None else None,
                    "desc_term": round(desc_term, 4),
                    "w_desc_contrib": round(float(w_desc) * desc_term, 2),
                    "opp_threat": round(float(opp_threat), 4) if opp_threat is not None else None,
                    "w_threat_contrib": round(float(w_threat) * threat_term, 2),
                    "guard_penalty": round(guard_penalty, 2),
                    "guard_penalty_raw": round(float(guard_penalty_raw), 2),
                    "guard_weight": round(float(w_guard), 4),
                    "guard_reasons": list(guard_reasons or []),
                    "guard_phase": guard_phase,
                    "score_phase": score_phase,
                    "policy_bonus": round(float(policy_bonus), 2),
                    "policy_bonus_reason": policy_bonus_reason,
                    "guard_inputs": {
                        "self_total_points_used": round(float(self_tp_val), 2) if self_tp_val is not None else None,
                        "self_total_points_source": self_tp_source,
                        "self_total_points_actual": round(float(self_tp_actual), 2) if self_tp_actual is not None else None,
                        "partner_total_points_expected": round(float(partner_tp_expected), 2) if partner_tp_expected is not None else None,
                        "partner_total_points_floor": round(float(partner_tp_min), 2) if partner_tp_min is not None else None,
                        "partner_total_points_hist": partner_tp_hist_val,
                        "par_contracts_topk": par_topk,
                        "acting_direction": acting_dir,
                        "enable_underbid_checks": guard_enable_underbid_checks,
                        "enable_tp_shortfall_check": guard_enable_tp_shortfall_check,
                        "enable_tricks_shortfall_check": guard_enable_tricks_shortfall_check,
                    },
                    "non_rebiddable_rebid_penalty": round(float(non_rebiddable_rebid_penalty), 2),
                    "non_rebiddable_rebid_reason": non_rebiddable_rebid_reason,
                    "post_game_slam_gate_penalty": round(float(post_game_slam_gate_penalty), 2),
                    "post_game_slam_gate_reason": post_game_slam_gate_reason,
                    "rebiddable_major_game_bonus": round(float(rebiddable_major_game_bonus), 2),
                    "rebiddable_major_game_reason": rebiddable_major_game_reason,
                    "partner_major_game_bonus": round(float(partner_major_game_bonus), 2),
                    "partner_major_detour_penalty": round(float(partner_major_detour_penalty), 2),
                    "partner_major_reason": partner_major_reason,
                    "forcing_heart_fit_bonus": round(float(forcing_heart_fit_bonus), 2),
                    "forcing_heart_detour_penalty": round(float(forcing_heart_detour_penalty), 2),
                    "forcing_heart_reason": forcing_heart_reason,
                    "forced_major_game_commit_bonus": round(float(forced_major_game_commit_bonus), 2),
                    "forced_major_game_commit_penalty": round(float(forced_major_game_commit_penalty), 2),
                    "forced_major_game_commit_reason": forced_major_game_commit_reason,
                    "takeout_double_explore_bonus": round(float(takeout_double_explore_bonus), 2),
                    "takeout_double_explore_penalty": round(float(takeout_double_explore_penalty), 2),
                    "takeout_double_explore_reason": takeout_double_explore_reason,
                    "takeout_double_trigger_bonus": round(float(takeout_double_trigger_bonus), 2),
                    "takeout_double_trigger_penalty": round(float(takeout_double_trigger_penalty), 2),
                    "takeout_double_trigger_reason": takeout_double_trigger_reason,
                    "common_sense_bonus": round(float(common_sense_bonus), 2),
                    "common_sense_penalty": round(float(common_sense_penalty), 2),
                    "common_sense_reason_codes": list(common_sense_reason_codes or []),
                    "common_sense_evidence": common_sense_evidence,
                    "common_sense_hard_override": False,
                    "nt_preference_bonus": round(float(nt_preference_bonus), 2),
                    "minor_nt_detour_penalty": round(float(minor_nt_detour_penalty), 2),
                    "nt_preference_reason": nt_preference_reason,
                    "nt_over_minor_bonus": round(float(nt_over_minor_bonus), 2),
                    "nt_over_minor_penalty": round(float(nt_over_minor_penalty), 2),
                    "nt_over_minor_reason": nt_over_minor_reason,
                    "pull_nt_game_penalty": round(float(pull_nt_game_penalty), 2),
                    "pull_nt_game_reason": pull_nt_game_reason,
                    "room_consumption_penalty": round(float(room_consumption_penalty), 2),
                    "room_consumption_reason": room_consumption_reason,
                    "is_level5_strain_change_jump": bool(is_level5_strain_change_jump),
                    "v2_jump_detected": bool(v2_diag.get("v2_jump_detected", False)),
                    "v2_point_type": v2_diag.get("v2_point_type"),
                    "v2_required_tp": v2_diag.get("v2_required_tp"),
                    "v2_self_tp_used": v2_diag.get("v2_self_tp_used"),
                    "v2_partner_tp_floor": v2_diag.get("v2_partner_tp_floor"),
                    "v2_partner_tp_ceiling": v2_diag.get("v2_partner_tp_ceiling"),
                    "v2_combined_tp_floor": v2_diag.get("v2_combined_tp_floor"),
                    "v2_combined_tp_ceiling": v2_diag.get("v2_combined_tp_ceiling"),
                    "v2_slam_qualified": bool(v2_diag.get("v2_slam_qualified", False)),
                    "v2_slam_rejected_reason": v2_diag.get("v2_slam_rejected_reason"),
                    "v2_blocked": bool(v2_diag.get("v2_blocked", False)),
                    "v2_block_reason": v2_diag.get("v2_block_reason"),
                    "v2_current_contract_ev_ns": round(float(v2_current_ev_ns), 2) if v2_current_ev_ns is not None else None,
                    "v2_current_contract_ev_acting": round(float(v2_current_ev_acting), 2) if v2_current_ev_acting is not None else None,
                    "v2_candidate_ev_acting": round(float(_ev_acting), 2) if _ev_acting is not None else None,
                    "v2_ev_delta_acting": (
                        round(float(_ev_acting) - float(v2_current_ev_acting), 2)
                        if (_ev_acting is not None and v2_current_ev_acting is not None)
                        else None
                    ),
                    "game_contract_on_table": bool(game_contract_on_table),
                    "p_slam_tp_ge_33": round(float(p_slam_tp_ge_33), 4) if p_slam_tp_ge_33 is not None else None,
                    "combined_tp_mean": round(float(combined_tp_mean), 2) if combined_tp_mean is not None else None,
                    "est_tricks": round(float(_est_tricks_api), 2) if _est_tricks_api is not None else None,
                    "utility": round(float(utility), 2) if utility is not None else None,
                    "acting_sign": float(acting_sign),
                    "final_score": round(float(score_val), 2),
                }
                breakdown.update(_special_case_breakdown_fields(opt))
                try:
                    _top = par_topk[0] if isinstance(par_topk, list) and len(par_topk) > 0 and isinstance(par_topk[0], dict) else None
                    if _top is not None:
                        _c = str(_top.get("contract") or "").strip()
                        _p = str(_top.get("pair") or "").strip().upper()
                        breakdown["likely_final_contract"] = f"{_c} by {_p}" if _c and _p else (_c or None)
                        _aps = _top.get("avg_par_score")
                        if _aps is not None:
                            _aps_f = float(_aps)
                            breakdown["likely_par_score_from_top_contract_ns"] = round(float(acting_sign) * _aps_f, 2)
                            breakdown["likely_par_score_from_top_contract_acting"] = round(_aps_f, 2)
                except Exception:
                    pass
                return float(score_val), bid1, d_ms, p2_ms, hit, breakdown
            except Exception:
                return float("-inf"), bid1, d_ms, p2_ms, hit, {}

        best_bid = ""
        best_score = float("-inf")
        max_workers = min(10, max(1, len(passed_sorted)))
        bd_calls = 0
        bd_ms_sum = 0.0
        bd_p2_ms_sum = 0.0
        bd_hits = 0
        bid_scores: List[Dict[str, Any]] = []
        scored_rows: List[Tuple[float, str, Dict[str, Any]]] = []
        _self_suit_lengths: dict[str, int] = {}
        try:
            _ssh = str(deal_row.get(f"Hand_{acting_dir}", "") or "").strip()
            if _ssh and "." in _ssh:
                _ssh_parts = _ssh.split(".")
                if len(_ssh_parts) == 4:
                    _self_suit_lengths = {"S": len(_ssh_parts[0]), "H": len(_ssh_parts[1]), "D": len(_ssh_parts[2]), "C": len(_ssh_parts[3])}
        except Exception:
            pass
        _opp_shown_strains = opponent_shown_natural_strains(tokens, acting_dir, dealer_actual)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_score_one, opt) for opt in passed_sorted]
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        sc, b, d_ms, p2_ms, hit, bkdn = fut.result()
                    except Exception:
                        continue
                    bd_calls += 1
                    bd_ms_sum += float(d_ms or 0.0)
                    bd_p2_ms_sum += float(p2_ms or 0.0)
                    bd_hits += 1 if hit else 0
                    bid_scores.append({"bid": b, "score": round(sc, 2) if sc != float("-inf") else None, "agg_expr": _agg_expr_by_bid.get(str(b or "").strip().upper(), []), **bkdn})
                    if b:
                        scored_rows.append((float(sc), str(b), bkdn))
        except Exception:
            for opt in passed_sorted:
                sc, b, d_ms, p2_ms, hit, bkdn = _score_one(opt)
                bd_calls += 1
                bd_ms_sum += float(d_ms or 0.0)
                bd_p2_ms_sum += float(p2_ms or 0.0)
                bd_hits += 1 if hit else 0
                bid_scores.append({"bid": b, "score": round(sc, 2) if sc != float("-inf") else None, "agg_expr": _agg_expr_by_bid.get(str(b or "").strip().upper(), []), **bkdn})
                if b:
                    scored_rows.append((float(sc), str(b), bkdn))
        # Post-scoring sweep: if a suit bid was LEVEL_CAP_BLOCKed and the
        # bidder's suit length there is >= the suit length for any surviving
        # (non-blocked) suit bid at the same or higher level, block that bid
        # too.  A weaker suit cannot succeed if the strongest one can't.
        if isinstance(_self_suit_lengths, dict) and _self_suit_lengths:
            _lcb_entries: list[tuple[int, str, int]] = []  # (level, strain, suit_len)
            for _bs in bid_scores:
                _b_name = str((_bs or {}).get("bid", "")).strip().upper()
                if not _b_name or len(_b_name) < 2 or not _b_name[0].isdigit():
                    continue
                _b_reason = str((_bs or {}).get("hard_block_reason", "") or "")
                if "LEVEL_CAP_BLOCK" not in _b_reason:
                    continue
                _b_lvl = int(_b_name[0])
                _b_str = _b_name[1:]
                if _b_str not in ("C", "D", "H", "S"):
                    continue
                _b_sl = _self_suit_lengths.get(_b_str)
                if _b_sl is not None:
                    _lcb_entries.append((_b_lvl, _b_str, int(_b_sl)))
            if _lcb_entries:
                _new_scored_rows: list[tuple[float, str, dict[str, Any]]] = []
                for _sc, _b_s, _bk in scored_rows:
                    _bn = str(_b_s).strip().upper()
                    if len(_bn) >= 2 and _bn[0].isdigit():
                        _blvl = int(_bn[0])
                        _bstr = _bn[1:]
                        if _bstr in ("C", "D", "H", "S") and not _bk.get("hard_blocked"):
                            _bsl = _self_suit_lengths.get(_bstr)
                            if _bsl is not None:
                                _bsl = int(_bsl)
                                for _clvl, _cstr, _csl in _lcb_entries:
                                    if _blvl >= _clvl and _bsl <= _csl:
                                        _bk["hard_blocked"] = True
                                        _bk["hard_block_reason"] = (
                                            f"LEVEL_CAP_BLOCK_PROPAGATED: "
                                            f"{_bn} blocked because {_clvl}{_cstr} "
                                            f"(best suit, {_csl} cards) was level-cap "
                                            f"blocked and {_bstr} is weaker ({_bsl} cards)"
                                        )
                                        _bk["final_score"] = None
                                        _sc = float("-inf")
                                        for _bsr in bid_scores:
                                            if str((_bsr or {}).get("bid", "")).strip().upper() == _bn:
                                                _bsr["hard_blocked"] = True
                                                _bsr["hard_block_reason"] = _bk["hard_block_reason"]
                                                _bsr["final_score"] = None
                                                _bsr["score"] = None
                                        break
                    _new_scored_rows.append((_sc, _b_s, _bk))
                scored_rows = _new_scored_rows

        common_sense_gate_reason: str | None = None
        common_sense_final_gate_applied = False
        # Common-sense adjudication stage (post-score).
        if scored_rows:
            _scored_rows_adj: List[Tuple[float, str, Dict[str, Any]]] = []
            for _sc, _b, _bk in list(scored_rows or []):
                if use_common_sense:
                    _cs_bonus = float((_bk or {}).get("common_sense_bonus") or 0.0)
                    _cs_pen = float((_bk or {}).get("common_sense_penalty") or 0.0)
                    _adj = float(_sc) + _cs_bonus - _cs_pen
                else:
                    _adj = float(_sc)
                _scored_rows_adj.append((float(_adj), str(_b), dict(_bk or {})))
            _scored_rows_adj.sort(key=lambda t: (-float(t[0]), str(t[1])))
            best_score = float(_scored_rows_adj[0][0])
            best_bid = str(_scored_rows_adj[0][1])

            # Optional hard override for whitelisted high-confidence actions.
            try:
                if not use_common_sense:
                    raise RuntimeError("common-sense hard override disabled by logic_mode")
                _legal_non_pass_candidates = [
                    _o for _o in list(candidates_sorted or [])
                    if str((_o or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                    and bool((_o or {}).get("can_complete", True))
                ]
                _blocked_cannot_complete = [
                    _r for _r in list(_bids_criteria_filtered or [])
                    if str((_r or {}).get("filter_reason", "")).strip().lower() == "cannot_complete"
                ]
                _blocked_criteria_fail = [
                    _r for _r in list(_bids_criteria_filtered or [])
                    if str((_r or {}).get("filter_reason", "")).strip().lower() == "criteria_fail"
                    and str((_r or {}).get("bid", "") or "").strip().upper() not in ("", "P", "PASS")
                ]
                _self_tp_step = None
                _self_sl_step: Dict[str, int] | None = None
                try:
                    _tp_raw = deal_row.get(f"Total_Points_{acting_dir}")
                    if _tp_raw is not None:
                        _self_tp_step = float(_tp_raw)
                except Exception:
                    _self_tp_step = None
                try:
                    _h = str(deal_row.get(f"Hand_{acting_dir}", "") or "").strip()
                    if _h:
                        _parts = _h.split(".")
                        if len(_parts) == 4:
                            _self_sl_step = {
                                "S": len(str(_parts[0] or "")),
                                "H": len(str(_parts[1] or "")),
                                "D": len(str(_parts[2] or "")),
                                "C": len(str(_parts[3] or "")),
                            }
                except Exception:
                    _self_sl_step = None
                _cs_hard = compute_common_sense_hard_override(
                    auction_tokens=list(tokens or []),
                    acting_direction=acting_dir,
                    dealer_actual=dealer_actual,
                    self_total_points=_self_tp_step if isinstance(_self_tp_step, (int, float)) else None,
                    partner_total_points_expected=None,
                    self_suit_lengths=_self_sl_step if isinstance(_self_sl_step, dict) else None,
                    current_best_bid=best_bid,
                    legal_non_pass_candidates=list(_legal_non_pass_candidates or []),
                    blocked_candidates=list(_blocked_cannot_complete or []),
                    criteria_failed_candidates=list(_blocked_criteria_fail or []),
                )
                if bool((_cs_hard or {}).get("apply")):
                    _ov_bid = str((_cs_hard or {}).get("selected_bid") or "").strip().upper()
                    if _ov_bid:
                        _ov_score = None
                        _ov_row = None
                        for _adj_sc, _adj_b, _adj_bk in list(_scored_rows_adj or []):
                            if str(_adj_b).strip().upper() == _ov_bid:
                                _ov_score = float(_adj_sc)
                                _ov_row = _adj_bk
                                break
                        if _ov_score is None:
                            _ov_score = float(best_score) + 1.0
                        best_bid = _ov_bid
                        best_score = float(_ov_score)
                        common_sense_final_gate_applied = True
                        common_sense_gate_reason = str((_cs_hard or {}).get("reason") or "COMMON_SENSE_HARD_OVERRIDE")
                        _ov_reason_codes = list((_cs_hard or {}).get("reason_codes") or [])
                        _ov_evidence = dict((_cs_hard or {}).get("evidence") or {})
                        _found = False
                        for _row in bid_scores:
                            if str((_row or {}).get("bid", "")).strip().upper() == _ov_bid:
                                _row["common_sense_hard_override"] = True
                                _row["common_sense_reason_codes"] = _ov_reason_codes
                                _row["common_sense_evidence"] = _ov_evidence
                                _row["score"] = round(float(best_score), 2)
                                _found = True
                                break
                        if not _found:
                            bid_scores.append(
                                {
                                    "bid": _ov_bid,
                                    "score": round(float(best_score), 2),
                                    "agg_expr": [],
                                    "common_sense_bonus": 0.0,
                                    "common_sense_penalty": 0.0,
                                    "common_sense_reason_codes": _ov_reason_codes,
                                    "common_sense_evidence": _ov_evidence,
                                    "common_sense_hard_override": True,
                                    "hard_override_without_bid_details": True,
                                }
                            )
            except Exception:
                pass
        # Sort bid_scores by score descending for readability
        bid_scores.sort(key=lambda x: float(x.get("score") or float("-inf")), reverse=True)

        if not best_bid:
            best_bid = str(passed_sorted[0].get("bid", "") or "").strip().upper()
        if not best_bid:
            break

        tokens.append(best_bid)
        step_rec["choice"] = best_bid
        step_rec["score"] = best_score
        step_rec["acting_sign"] = acting_sign
        step_rec["scored_n"] = int(len(passed_sorted))
        step_rec["bid_details_calls"] = int(bd_calls)
        step_rec["bid_details_cache_hits"] = int(bd_hits)
        step_rec["bid_details_elapsed_ms_sum"] = round(float(bd_ms_sum), 1)
        step_rec["bid_details_phase2a_ms_sum"] = round(float(bd_p2_ms_sum), 1)
        step_rec["bid_scores"] = bid_scores
        step_rec["common_sense_final_gate_applied"] = bool(common_sense_final_gate_applied)
        step_rec["common_sense_gate_reason"] = common_sense_gate_reason
        try:
            _chosen_row = None
            for _r in bid_scores:
                if str(_r.get("bid", "")).strip().upper() == str(best_bid).strip().upper():
                    _chosen_row = _r
                    break
            _likely_contract = None
            _likely_top_par_ns = None
            if isinstance(_chosen_row, dict):
                _likely_contract = _chosen_row.get("likely_final_contract")
                if _likely_contract is None:
                    _topk = (((_chosen_row.get("guard_inputs") or {}).get("par_contracts_topk")) or [])
                    if isinstance(_topk, list) and _topk and isinstance(_topk[0], dict):
                        _c = str(_topk[0].get("contract") or "").strip()
                        _p = str(_topk[0].get("pair") or "").strip().upper()
                        _likely_contract = f"{_c} by {_p}" if _c and _p else (_c or None)
                        _aps = _topk[0].get("avg_par_score")
                        if _aps is not None:
                            _likely_top_par_ns = float(acting_sign) * float(_aps)
                _mean_par = _chosen_row.get("mean_par")
                _mean_par_ns = float(acting_sign) * float(_mean_par) if _mean_par is not None else None
                _mean_par_acting = float(_mean_par) if _mean_par is not None else None
            else:
                _mean_par_ns = None
                _mean_par_acting = None
            step_rec["bidder_view"] = {
                "acting_direction": acting_dir,
                "chosen_bid": best_bid,
                "likely_final_contract": _likely_contract,
                "likely_par_score_ns": round(float(_mean_par_ns), 2) if _mean_par_ns is not None else None,
                "likely_par_score_acting": round(float(_mean_par_acting), 2) if _mean_par_acting is not None else None,
                "likely_top_contract_par_score_ns": round(float(_likely_top_par_ns), 2) if _likely_top_par_ns is not None else None,
                "source": "chosen_bid_details",
            }
            try:
                _chosen_agg = _chosen_row.get("agg_expr") if isinstance(_chosen_row, dict) else []
                _agg_u = [str(x or "").strip().upper() for x in list(_chosen_agg or [])]
                _shows_forcing_hearts = False
                if any("FORCING_ONE_ROUND" in x for x in _agg_u) and any("SL_H >=" in x for x in _agg_u):
                    _shows_forcing_hearts = True
                _bc = _parse_contract_bid_text(str(best_bid))
                if _bc is not None and str(_bc[1]) == "H" and int(_bc[0]) >= 2:
                    _shows_forcing_hearts = True
                if _shows_forcing_hearts:
                    forcing_heart_shown_by_dir[str(acting_dir)] = True
                if str(best_bid).strip().upper() in ("D", "X", "DOUBLE"):
                    takeout_double_shown_by_dir[str(acting_dir)] = True
                step_rec["convention_state"] = {
                    "forcing_heart_shown_by_dir": dict(forcing_heart_shown_by_dir),
                    "takeout_double_shown_by_dir": dict(takeout_double_shown_by_dir),
                }
            except Exception:
                pass
        except Exception:
            step_rec["bidder_view"] = {
                "acting_direction": acting_dir_step,
                "chosen_bid": best_bid,
                "likely_final_contract": None,
                "likely_par_score_ns": None,
                "likely_par_score_acting": None,
                "source": "chosen_bid_details_error",
            }
        step_rec["elapsed_ms"] = round((time.perf_counter() - t_step0) * 1000, 1)
        steps_detail.append(step_rec)

    out_auc = "-".join(tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "auction": out_auc,
        "steps": int(len(tokens)),
        "steps_detail": steps_detail,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_custom_criteria_impact(
    state: Dict[str, Any],
    *,
    auction: str,
    criteria: List[str],
    seat: int = 1,
    dealer: str = "N",
    sample_size: int = 10_000,
) -> Dict[str, Any]:
    """Estimate the impact of proposed new criteria on deals traversing an auction node.

    For each criterion string, reports what fraction of sampled deals would fail
    (i.e., be excluded) if the criterion were added to the bidding table node
    at `auction`.  Uses vectorized Polars evaluation where possible, with a
    per-row Python fallback for complex expressions.

    Returns a dict with:
      - auction, seat, dealer, sample_size, total_deals_in_db
      - impact_per_criterion: list of {criterion, pass_count, fail_count, fail_pct, unknown_count}
      - combined: counts for deals failing ANY proposed criterion
    """
    t0 = time.perf_counter()
    deal_df = state.get("deal_df")
    if deal_df is None:
        raise ValueError("deal_df not loaded in server state")

    total_deals = deal_df.height
    _sample_n = min(int(sample_size), total_deals)

    # Sample deterministically for reproducibility
    if total_deals > _sample_n:
        rng = random.Random(42)
        idxs = sorted(rng.sample(range(total_deals), _sample_n))
        sample_df = deal_df[idxs]
    else:
        sample_df = deal_df

    direction = seat_to_direction(dealer, seat)

    # ---------------------------------------------------------------------------
    # Vectorized fast-path: map common criterion patterns to Polars expressions
    # ---------------------------------------------------------------------------
    def _criterion_to_polars(crit_s: str) -> pl.Expr | None:
        """Return a Polars boolean expression for simple criterion patterns, or None."""
        c = crit_s.strip()
        # Simple: VAR OP NUMBER  (HCP >= 12, Total_Points >= 20, SL_S >= 5)
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<|==|!=)\s*(\d+(?:\.\d+)?)$", c)
        if not m:
            return None
        var_raw, op_s, num_s = m.group(1), m.group(2), m.group(3)
        try:
            num = float(num_s) if "." in num_s else int(num_s)
        except ValueError:
            return None
        _op_map = {">=": "__ge__", "<=": "__le__", ">": "__gt__", "<": "__lt__", "==": "__eq__", "!=": "__ne__"}
        op_fn = _op_map.get(op_s)
        if op_fn is None:
            return None
        var_up = var_raw.upper()
        SUIT_MAP = {"S": 0, "H": 1, "D": 2, "C": 3}
        if var_up == "HCP":
            col = f"HCP_{direction}"
            if col not in sample_df.columns:
                return None
            return getattr(pl.col(col), op_fn)(num)
        if var_up in ("TOTAL_POINTS", "TOTALPOINTS"):
            col = f"Total_Points_{direction}"
            if col not in sample_df.columns:
                return None
            return getattr(pl.col(col), op_fn)(num)
        if var_up.startswith("SL_") and len(var_up) == 4:
            suit = var_up[-1]
            suit_idx = SUIT_MAP.get(suit)
            if suit_idx is None:
                return None
            hand_col = f"Hand_{direction}"
            if hand_col not in sample_df.columns:
                return None
            sl_expr = pl.col(hand_col).str.split(".").list.get(suit_idx).str.len_chars()
            return getattr(sl_expr, op_fn)(num)
        return None

    impact_per_criterion: List[Dict[str, Any]] = []
    combined_fail_mask: pl.Series | None = None

    for crit in (criteria or []):
        crit_s = str(crit).strip()
        pass_count: int = 0
        fail_count: int = 0
        unknown_count: int = 0
        fail_mask: pl.Series = pl.Series([False] * _sample_n)

        polars_expr = _criterion_to_polars(crit_s)
        _used_polars = False
        if polars_expr is not None:
            try:
                result_series = sample_df.select(polars_expr.alias("_pass")).get_column("_pass")
                pass_count = int(result_series.sum())
                fail_count = _sample_n - pass_count
                unknown_count = 0
                fail_mask = ~result_series
                _used_polars = True
            except Exception:
                _used_polars = False

        if not _used_polars:
            # Per-row Python fallback (handles complex / relative-SL criteria)
            pass_count = fail_count = unknown_count = 0
            fail_mask_list: List[bool] = []
            for row in sample_df.iter_rows(named=True):
                result = evaluate_sl_criterion(crit_s, dealer, seat, row, fail_on_missing=False)
                if result is True:
                    pass_count += 1
                    fail_mask_list.append(False)
                elif result is False:
                    fail_count += 1
                    fail_mask_list.append(True)
                else:
                    unknown_count += 1
                    fail_mask_list.append(False)
            fail_mask = pl.Series(fail_mask_list)

        # Accumulate combined mask (ANY criterion fails → excluded)
        if combined_fail_mask is None:
            combined_fail_mask = fail_mask
        else:
            combined_fail_mask = combined_fail_mask | fail_mask

        impact_per_criterion.append({
            "criterion": crit_s,
            "pass_count": int(pass_count),
            "fail_count": int(fail_count),
            "unknown_count": int(unknown_count),
            "fail_pct": round(100.0 * fail_count / max(1, pass_count + fail_count), 1),
        })

    combined_fail = int(combined_fail_mask.sum()) if combined_fail_mask is not None else 0
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "auction": auction,
        "seat": seat,
        "dealer": dealer,
        "sample_size": _sample_n,
        "total_deals_in_db": total_deals,
        "impact_per_criterion": impact_per_criterion,
        "combined": {
            "fail_count": combined_fail,
            "pass_count": _sample_n - combined_fail,
            "fail_pct": round(100.0 * combined_fail / max(1, _sample_n), 1),
        },
        "elapsed_ms": elapsed_ms,
    }


def _merge_numeric_bound(bounds: dict[str, float | None], op: str, value: float) -> None:
    """Merge a single comparator into min/max bounds."""
    cur_min = bounds.get("min")
    cur_max = bounds.get("max")
    if op == ">=":
        bounds["min"] = value if cur_min is None else max(float(cur_min), value)
    elif op == ">":
        # Keep integer-like strictness in explainable form (+1 for strict integer criteria).
        strict_min = value + 1.0
        bounds["min"] = strict_min if cur_min is None else max(float(cur_min), strict_min)
    elif op == "<=":
        bounds["max"] = value if cur_max is None else min(float(cur_max), value)
    elif op == "<":
        strict_max = value - 1.0
        bounds["max"] = strict_max if cur_max is None else min(float(cur_max), strict_max)
    elif op == "==":
        bounds["min"] = value
        bounds["max"] = value


def _infer_belief_ranges_from_criteria(criteria: list[str]) -> dict[str, dict[str, float | None]]:
    """Infer coarse numeric ranges from Agg_Expr criteria list."""
    out: dict[str, dict[str, float | None]] = {
        "HCP": {"min": None, "max": None},
        "Total_Points": {"min": None, "max": None},
        "SL_S": {"min": None, "max": None},
        "SL_H": {"min": None, "max": None},
        "SL_D": {"min": None, "max": None},
        "SL_C": {"min": None, "max": None},
    }
    pat = re.compile(r"^\s*(HCP|Total_Points|SL_[SHDC])\s*(>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
    for crit in criteria:
        c = str(crit or "").strip()
        m = pat.match(c)
        if not m:
            continue
        var_raw, op, num_s = m.group(1), m.group(2), m.group(3)
        var = "Total_Points" if var_raw.upper().startswith("TOTAL") else var_raw.upper()
        if var not in out:
            continue
        try:
            num_v = float(num_s)
        except Exception:
            continue
        _merge_numeric_bound(out[var], op, num_v)
    return out


def _build_known_seat_features(known_hands: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Compute per-direction known hand feature dicts for explainability."""
    out: dict[str, dict[str, Any]] = {}
    for d in ("N", "E", "S", "W"):
        hs = str((known_hands or {}).get(d) or "").strip()
        if not hs:
            continue
        feats = compute_hand_features(hs) or {}
        out[d] = {
            "hand": hs,
            "HCP": feats.get("HCP"),
            "Total_Points": feats.get("Total_Points"),
            "SL_S": feats.get("SL_S"),
            "SL_H": feats.get("SL_H"),
            "SL_D": feats.get("SL_D"),
            "SL_C": feats.get("SL_C"),
        }
    return out


def handle_belief_snapshot(
    state: Dict[str, Any],
    *,
    auction: str,
    dealer: str = "N",
    vul: str | None = None,
    step: int | None = None,
    known_hands: dict[str, str] | None = None,
    deal_row_idx: int | None = None,
    deal_row_dict: dict[str, Any] | None = None,
    compact: bool = True,
) -> Dict[str, Any]:
    """Return seat-wise belief snapshot from BT criteria + known hand facts."""
    t0 = time.perf_counter()
    auction_in = normalize_auction_input(str(auction or ""))
    tok_all = [t for t in str(auction_in or "").split("-") if t]
    step_i = int(step) if step is not None else (len(tok_all) + 1)
    if step_i < 1:
        step_i = 1
    prefix_tokens = tok_all[: max(0, step_i - 1)]
    auction_prefix = "-".join(prefix_tokens)

    dealer_u = str(dealer or "N").strip().upper() or "N"
    if dealer_u not in DIRECTIONS_LIST:
        dealer_u = "N"
    known_map: dict[str, str] = {str(k).upper(): str(v) for k, v in (known_hands or {}).items() if str(v or "").strip()}

    # Pull known hands from deal row when caller provides db index or inline row.
    row_dict: dict[str, Any] = {}
    if isinstance(deal_row_dict, dict):
        row_dict = dict(deal_row_dict)
    elif deal_row_idx is not None and int(deal_row_idx) >= 0:
        try:
            deal_df = state.get("deal_df")
            if isinstance(deal_df, pl.DataFrame) and deal_df.height > int(deal_row_idx):
                one = _take_rows_by_index(deal_df, [int(deal_row_idx)])
                if one.height > 0:
                    row_dict = one.to_dicts()[0]
        except Exception:
            row_dict = {}
    if row_dict:
        dealer_u = str(row_dict.get("Dealer", dealer_u) or dealer_u).upper()
        if dealer_u not in DIRECTIONS_LIST:
            dealer_u = "N"
        for d in ("N", "E", "S", "W"):
            hv = str(row_dict.get(f"Hand_{d}", "") or "").strip()
            if hv and d not in known_map:
                known_map[d] = hv

    bt_index: int | None = None
    bt_row: dict[str, Any] = {}
    if auction_prefix:
        bt_index = _resolve_bt_index_by_traversal(state, auction_prefix)
        if bt_index is not None:
            try:
                bt_file = state.get("bt_seat1_file")
                if isinstance(bt_file, (str, pathlib.Path)):
                    bt_map = _load_agg_expr_for_bt_indices([int(bt_index)], bt_file)
                    bt_row = {"bt_index": int(bt_index)}
                    bt_row.update(bt_map.get(int(bt_index), {}))
                    bt_row = _apply_all_rules_to_bt_row(bt_row, state)
            except Exception:
                bt_row = {}

    known_features = _build_known_seat_features(known_map)
    seats_out: dict[str, Any] = {}
    for seat in (1, 2, 3, 4):
        dir_map = _seat_direction_map(seat)
        direction = str(dir_map.get(dealer_u, "N"))
        crits = list(bt_row.get(agg_expr_col(seat)) or [])
        ranges = _infer_belief_ranges_from_criteria([str(c) for c in crits])
        known = known_features.get(direction)
        seat_payload: dict[str, Any] = {
            "seat": int(seat),
            "direction": direction,
            "criteria_count": int(len(crits)),
            "ranges": ranges,
            "is_known": bool(known),
        }
        if known:
            seat_payload["known"] = known
        if not compact:
            seat_payload["criteria"] = crits
        seats_out[direction] = seat_payload

    explanation_facts: list[str] = []
    for d in ("N", "E", "S", "W"):
        s = seats_out.get(d) or {}
        rg = s.get("ranges") or {}
        hcp_r = rg.get("HCP") or {}
        tp_r = rg.get("Total_Points") or {}
        hcp_min = hcp_r.get("min")
        hcp_max = hcp_r.get("max")
        tp_min = tp_r.get("min")
        tp_max = tp_r.get("max")
        if hcp_min is not None or hcp_max is not None or tp_min is not None or tp_max is not None:
            explanation_facts.append(
                f"{d}: HCP[{hcp_min},{hcp_max}] TP[{tp_min},{tp_max}] from criteria"
            )

    return {
        "auction_input": auction_in,
        "auction_prefix_for_step": auction_prefix,
        "step": int(step_i),
        "dealer": dealer_u,
        "vul": vul,
        "bt_index": bt_index,
        "belief_version": "criteria-ranges-v1",
        "seats": seats_out,
        "known_hands": known_map,
        "explanation_facts": explanation_facts,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def handle_belief_trace(
    state: Dict[str, Any],
    *,
    auction: str,
    dealer: str = "N",
    vul: str | None = None,
    known_hands: dict[str, str] | None = None,
    deal_row_idx: int | None = None,
    deal_row_dict: dict[str, Any] | None = None,
    compact: bool = True,
    max_steps: int = 64,
) -> Dict[str, Any]:
    """Return step-wise belief snapshots for full-auction explainability."""
    t0 = time.perf_counter()
    auction_in = normalize_auction_input(str(auction or ""))
    toks = [t for t in auction_in.split("-") if t]
    n_steps = min(int(len(toks) + 1), max(1, int(max_steps)))

    snapshots: list[dict[str, Any]] = []
    for s in range(1, n_steps + 1):
        snap = handle_belief_snapshot(
            state=state,
            auction=auction_in,
            dealer=dealer,
            vul=vul,
            step=s,
            known_hands=known_hands,
            deal_row_idx=deal_row_idx,
            deal_row_dict=deal_row_dict,
            compact=compact,
        )
        snapshots.append(snap)

    deltas: list[dict[str, Any]] = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        delta_rows: list[dict[str, Any]] = []
        for d in ("N", "E", "S", "W"):
            p = ((prev.get("seats") or {}).get(d) or {}).get("ranges") or {}
            c = ((curr.get("seats") or {}).get(d) or {}).get("ranges") or {}
            p_hcp = p.get("HCP") or {}
            c_hcp = c.get("HCP") or {}
            if p_hcp != c_hcp:
                delta_rows.append({"direction": d, "feature": "HCP", "before": p_hcp, "after": c_hcp})
        if delta_rows:
            deltas.append({
                "step_from": int(i),
                "step_to": int(i + 1),
                "changes": delta_rows,
            })

    return {
        "auction_input": auction_in,
        "dealer": str(dealer or "N").upper(),
        "vul": vul,
        "steps": snapshots,
        "deltas": deltas,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def handle_critical_mistake_analysis(
    state: Dict[str, Any],
    *,
    ai_model_steps: list[dict[str, Any]] | None,
    dealer: str = "N",
    us_pair: str = "NS",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Identify high-impact opponent mistakes from per-step bid scores.

    This consumes the `ai_model_steps` payload produced by advanced path scoring and
    finds steps where the acting side selected a bid that scored materially below
    the best available alternative.
    """
    _ = state  # reserved for future richer analysis using stateful lookups
    t0 = time.perf_counter()
    dealer_u = str(dealer or "N").strip().upper() or "N"
    if dealer_u not in DIRECTIONS_LIST:
        dealer_u = "N"
    us_pair_u = str(us_pair or "NS").strip().upper()
    if us_pair_u not in ("NS", "EW"):
        us_pair_u = "NS"
    k = max(1, int(top_k or 1))

    rows: list[dict[str, Any]] = []
    for st in list(ai_model_steps or []):
        if not isinstance(st, dict):
            continue
        scores = st.get("bid_scores") or []
        if not isinstance(scores, list) or not scores:
            continue

        # Build best candidate and chosen candidate lookup.
        best_row: dict[str, Any] | None = None
        best_score = float("-inf")
        by_bid: dict[str, dict[str, Any]] = {}
        for r in scores:
            if not isinstance(r, dict):
                continue
            bid = str(r.get("bid", "")).strip().upper()
            if bid:
                by_bid[bid] = r
            sc = _safe_float(r.get("score"))
            if sc is None:
                sc = _safe_float(r.get("final_score"))
            if sc is None:
                continue
            if sc > best_score:
                best_score = float(sc)
                best_row = r
        if best_row is None:
            continue

        chosen_bid = str(st.get("choice", "")).strip().upper()
        chosen_row = by_bid.get(chosen_bid)
        chosen_score = _safe_float(chosen_row.get("score")) if isinstance(chosen_row, dict) else None
        if chosen_score is None and isinstance(chosen_row, dict):
            chosen_score = _safe_float(chosen_row.get("final_score"))
        if chosen_score is None:
            continue

        seat_i = int(st.get("seat", 0) or 0)
        seat_map = _seat_direction_map(seat_i if 1 <= seat_i <= 4 else 1)
        direction = str(seat_map.get(dealer_u, "N"))
        actor_pair = "NS" if direction in ("N", "S") else "EW"
        is_opponent_step = actor_pair != us_pair_u

        delta_actor = float(best_score) - float(chosen_score)
        if not math.isfinite(delta_actor) or delta_actor <= 0.0:
            continue

        scored_n = int(st.get("scored_n", len(scores)) or len(scores))
        # Confidence proxy: bigger deltas and larger candidate sets increase trust.
        conf = max(
            0.05,
            min(
                0.99,
                0.70 * min(delta_actor / 120.0, 1.0) + 0.30 * min(scored_n / 5.0, 1.0),
            ),
        )

        # Per-step bidder belief snapshot for chosen and best-alternative bids.
        _step_belief_raw = st.get("bidder_view")
        step_belief: dict[str, Any] = _step_belief_raw if isinstance(_step_belief_raw, dict) else {}
        chosen_belief = {
            "acting_direction": step_belief.get("acting_direction", direction),
            "chosen_bid": chosen_bid,
            "likely_final_contract": step_belief.get("likely_final_contract"),
            "likely_par_score_ns": _safe_float(step_belief.get("likely_par_score_ns")),
            "likely_par_score_acting": _safe_float(step_belief.get("likely_par_score_acting")),
            "source": step_belief.get("source"),
        }
        best_belief = {
            "acting_direction": step_belief.get("acting_direction", direction),
            "chosen_bid": str(best_row.get("bid", "")).strip().upper(),
            "likely_final_contract": best_row.get("likely_final_contract"),
            "likely_par_score_ns": None,
            "likely_par_score_acting": None,
            "source": "best_alternative_bid_details",
        }
        _best_par_acting = _safe_float(best_row.get("mean_par"))
        if _best_par_acting is not None:
            _actor_sign = 1.0 if actor_pair == "NS" else -1.0
            best_belief["likely_par_score_acting"] = float(_best_par_acting)
            best_belief["likely_par_score_ns"] = float(_actor_sign * _best_par_acting)

        rows.append(
            {
                "step": int(st.get("step", 0) or 0),
                "auction_prefix": st.get("bt_prefix"),
                "seat": seat_i,
                "direction": direction,
                "actor_pair": actor_pair,
                "is_opponent_step": bool(is_opponent_step),
                "chosen_bid": chosen_bid,
                "chosen_score": round(float(chosen_score), 2),
                "best_alternative_bid": str(best_row.get("bid", "")).strip().upper(),
                "best_alternative_score": round(float(best_score), 2),
                "delta_actor": round(delta_actor, 2),
                "delta_us": round(delta_actor if is_opponent_step else -delta_actor, 2),
                "scored_n": scored_n,
                "confidence": round(conf, 3),
                "chosen_guard_penalty": _safe_float((chosen_row or {}).get("guard_penalty")),
                "best_guard_penalty": _safe_float((best_row or {}).get("guard_penalty")),
                "chosen_bidder_view": chosen_belief,
                "best_alternative_bidder_view": best_belief,
            }
        )

    opponent_rows = [r for r in rows if bool(r.get("is_opponent_step"))]
    opponent_rows.sort(key=lambda r: float(r.get("delta_actor") or 0.0), reverse=True)
    our_rows = [r for r in rows if not bool(r.get("is_opponent_step"))]
    our_rows.sort(key=lambda r: float(r.get("delta_actor") or 0.0), reverse=True)

    critical_opp = opponent_rows[0] if opponent_rows else None
    critical_our = our_rows[0] if our_rows else None

    return {
        "dealer": dealer_u,
        "us_pair": us_pair_u,
        "total_steps_analyzed": len(rows),
        "critical_opponent_mistake": critical_opp,
        "critical_our_mistake": critical_our,
        "top_opponent_mistakes": opponent_rows[:k],
        "top_our_mistakes": our_rows[:k],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
