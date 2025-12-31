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
import random
import re
import statistics
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
    get_dd_score_for_auction,
    get_ev_for_auction,
    compute_hand_features,
    compute_par_score,
    parse_pbn_deal,
    build_distribution_sql_for_bt,
    build_distribution_sql_for_deals,
    add_suit_length_columns,
)

from mlBridgeLib.mlBridgeBiddingLib import DIRECTIONS
import mlBridgeLib.mlBridgeAugmentLib as mlBridgeAugmentLib

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
    # Vectorized join helpers (performance optimization)
    prepare_deals_with_bid_str,
    prepare_bt_for_join,
    join_deals_with_bt,
    batch_check_wrong_bids,
    # SL (Suit Length) evaluation helpers
    seat_to_direction,
    format_seat_notation as _format_seat_notation,
    hand_suit_length,
    parse_sl_comparison_relative,
    parse_sl_comparison_numeric,
    eval_comparison,
    annotate_criterion_with_value,
)


from plugins.custom_criteria_overlay import apply_custom_criteria_overlay_to_bt_row as _apply_custom_criteria_overlay_to_bt_row


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


def handle_random_auction_sequences(
    state: Dict[str, Any],
    n_samples: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Handle /random-auction-sequences endpoint.
    
    Requires bt_seat1_df (pipeline invariant).
    """
    t0 = time.perf_counter()
    
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded (pipeline error): missing bbo_bt_seat1.parquet")

    # Hard fail if is_completed_auction is missing
    if "is_completed_auction" not in bt_seat1_df.columns:
        raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")

    base_df = bt_seat1_df
    completed_df = base_df.filter(pl.col("is_completed_auction"))
    if "bt_index" not in completed_df.columns:
        raise ValueError("REQUIRED column 'bt_index' missing from bt_seat1_df. Pipeline error.")
    index_col = "bt_index"

    if completed_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[random-auction-sequences] {format_elapsed(elapsed_ms)} (empty)")
        return {"samples": [], "elapsed_ms": round(elapsed_ms, 1)}

    sample_n = min(n_samples, completed_df.height)
    effective_seed = _effective_seed(seed)
    sampled_df = completed_df.sample(n=sample_n, seed=effective_seed)

    sql_query = f"""SELECT * 
FROM auctions 
WHERE is_completed_auction = true 
ORDER BY RANDOM() 
LIMIT {sample_n}"""

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
        prev_rows_df = base_df.filter(pl.col(index_col).is_in(list(all_prev_indices))).select(
            [c for c in available_display_cols if c in base_df.columns]
        )
        prev_rows_lookup = {row[index_col]: row for row in prev_rows_df.iter_rows(named=True)}
    else:
        prev_rows_lookup = {}

    out_samples: List[Dict[str, Any]] = []
    overlay = state.get("custom_criteria_overlay") or []
    merged_rules_lookup = state.get("merged_rules_lookup")

    def _with_rules_seat1_cols(raw_row: dict[str, Any]) -> dict[str, Any]:
        """Return the overlay-applied BT row plus extra columns showing Rules (merged+overlay) seat-1 criteria."""
        base_row = _apply_custom_criteria_overlay_to_bt_row(dict(raw_row), overlay)
        try:
            bt_idx_val = base_row.get(index_col)
            bt_idx_i = int(bt_idx_val) if bt_idx_val is not None else None
        except Exception:
            bt_idx_i = None
        merged = None
        if merged_rules_lookup is not None and bt_idx_i is not None:
            try:
                merged = merged_rules_lookup.get(bt_idx_i)
            except Exception:
                merged = None
        if merged:
            rules_row = dict(raw_row)
            rules_row[agg_expr_col(1)] = list(merged)
            rules_row = _apply_custom_criteria_overlay_to_bt_row(rules_row, overlay)
            base_row["Merged_Rules_Seat_1"] = list(merged)
            base_row["Agg_Expr_Seat_1_Rules"] = rules_row.get(agg_expr_col(1)) or []
        else:
            base_row["Merged_Rules_Seat_1"] = []
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

        # Get deal count and wrong bid rate for this auction
        deal_df = state.get("deal_df")
        deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
        deal_count = 0
        wrong_bid_count = 0
        wrong_bid_rate = 0.0
        
        if deal_df is not None and matched_auction:
            # Build bt_info for wrong bid checking
            bt_info = {
                "Agg_Expr_Seat_1": row.get("Agg_Expr_Seat_1"),
                "Agg_Expr_Seat_2": row.get("Agg_Expr_Seat_2"),
                "Agg_Expr_Seat_3": row.get("Agg_Expr_Seat_3"),
                "Agg_Expr_Seat_4": row.get("Agg_Expr_Seat_4"),
            }
            
            auction_lower = matched_auction.lower()
            auction_variations = [auction_lower]
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
            
            deal_df_with_str = deal_df_with_str.with_row_index("_row_idx")
            matching_deals = deal_df_with_str.filter(
                pl.col("_bid_str").str.to_lowercase().is_in(auction_variations)
            )
            deal_count = matching_deals.height
            
            # Sample for wrong bid calculation
            if deal_count > 0:
                sample_n = min(100, deal_count)
                if sample_n < deal_count:
                    sample_deals = matching_deals.sample(n=sample_n, seed=42)
                else:
                    sample_deals = matching_deals
                
                for deal_row in sample_deals.iter_rows(named=True):
                    deal_idx = deal_row.get("_row_idx", 0)
                    dealer = deal_row.get("Dealer", "N")
                    bid_str = deal_row.get("_bid_str", "")
                    
                    conformance = _check_deal_criteria_conformance_bitmap(
                        int(deal_idx), bt_info, dealer, deal_criteria_by_seat_dfs, auction=bid_str
                    )
                    if conformance["first_wrong_seat"] is not None:
                        wrong_bid_count += 1
                
                wrong_bid_rate = wrong_bid_count / sample_n if sample_n > 0 else 0.0

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
    merged_rules_lookup = state.get("merged_rules_lookup")

    def _with_rules_seat1_cols(raw_row: dict[str, Any]) -> dict[str, Any]:
        """Return the overlay-applied BT row plus extra columns showing Rules (merged+overlay) seat-1 criteria."""
        base_row = _apply_custom_criteria_overlay_to_bt_row(dict(raw_row), overlay)
        try:
            bt_idx_val = base_row.get(index_col)
            bt_idx_i = int(bt_idx_val) if bt_idx_val is not None else None
        except Exception:
            bt_idx_i = None
        merged = None
        if merged_rules_lookup is not None and bt_idx_i is not None:
            try:
                merged = merged_rules_lookup.get(bt_idx_i)
            except Exception:
                merged = None
        if merged:
            # Apply merged rules to seat 1, then re-apply overlay so CSV can override merged rules.
            rules_row = dict(raw_row)
            rules_row[agg_expr_col(1)] = list(merged)
            rules_row = _apply_custom_criteria_overlay_to_bt_row(rules_row, overlay)
            base_row["Merged_Rules_Seat_1"] = list(merged)
            base_row["Agg_Expr_Seat_1_Rules"] = rules_row.get(agg_expr_col(1)) or []
        else:
            base_row["Merged_Rules_Seat_1"] = []
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
        
        seq_df = seq_df.select(out_cols)
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
                    sequence_data.append(
                        _apply_custom_criteria_overlay_to_bt_row(dict(prev_rows_lookup[idx]), overlay)
                    )
        sequence_data.append(
            _apply_custom_criteria_overlay_to_bt_row(
                {c: row[c] for c in available_display_cols if c in row},
                overlay,
            )
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
            print(f"[pbn-lookup] Warning: Could not find matched BT auctions: {e}")
    
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
    
    # Initialize loop variables for type checker (they're always bound when used due to early return)
    auction: str = ""
    auction_info: Dict[str, Any] = {}
    combined_df: pl.DataFrame = pl.DataFrame()

    for auction_row in sampled_auctions.iter_rows(named=True):
        auction_row = _apply_custom_criteria_overlay_to_bt_row(dict(auction_row), overlay)
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
                    auction=auction
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
        raise ValueError(f"Invalid regex pattern: {e}")
    
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
    overlay = state.get("custom_criteria_overlay") or []
    if overlay:
        result_rows = [_apply_custom_criteria_overlay_to_bt_row(r, overlay) for r in result_rows]
    
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
    
    result_cols = ["Auction"]
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
    overlay = state.get("custom_criteria_overlay") or []
    if overlay:
        result_rows = [_apply_custom_criteria_overlay_to_bt_row(r, overlay) for r in result_rows]
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
    
    bt_info = {
        "Agg_Expr_Seat_1": bt_row.get("Agg_Expr_Seat_1"),
        "Agg_Expr_Seat_2": bt_row.get("Agg_Expr_Seat_2"),
        "Agg_Expr_Seat_3": bt_row.get("Agg_Expr_Seat_3"),
        "Agg_Expr_Seat_4": bt_row.get("Agg_Expr_Seat_4"),
    }
    
    for row in sample_df.iter_rows(named=True):
        deal_idx = row.get("_row_idx", 0)
        dealer = row.get("Dealer", "N")
        bid_str = row.get("_bid_str", "")
        
        conformance = _check_deal_criteria_conformance_bitmap(
            int(deal_idx), bt_info, dealer, deal_criteria_by_seat_dfs, auction=bid_str
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
    # Apply hot-reloadable custom criteria overlay so stats reflect current CSV rules.
    overlay = state.get("custom_criteria_overlay") or []
    bt_row = _apply_custom_criteria_overlay_to_bt_row(dict(bt_row), overlay)

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
    overlay = state.get("custom_criteria_overlay") or []
    
    bt_lookup_df = bt_seat1_df
    # Hard fail if is_completed_auction is missing
    if "is_completed_auction" not in bt_lookup_df.columns:
        raise ValueError("REQUIRED column 'is_completed_auction' missing from bt_seat1_df. Pipeline error.")
        bt_lookup_df = bt_lookup_df.filter(pl.col("is_completed_auction"))
    available_bt_cols = [c for c in bt_cols if c in bt_lookup_df.columns]
    
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
        if available_bt_cols:
            # Strip leading "P-" prefixes (use regex, not lstrip which removes individual chars)
            auction_for_search = re.sub(r"^(P-)+", "", (bid_auction.upper() if bid_auction else ""))

            bt_match = bt_lookup_df.filter(pl.col("Auction").cast(pl.Utf8).str.to_uppercase() == auction_for_search)
            if bt_match.height == 0 and not auction_for_search.endswith("-P-P-P"):
                auction_with_passes = auction_for_search + "-P-P-P"
                bt_match = bt_lookup_df.filter(pl.col("Auction").cast(pl.Utf8).str.to_uppercase() == auction_with_passes)
            if bt_match.height > 0:
                bt_row = _apply_custom_criteria_overlay_to_bt_row(dict(bt_match.row(0, named=True)), overlay)
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
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index for bitmap lookups
    deal_df = deal_df.with_row_index("_row_idx")
    
    # Prepare for join: add _bid_str and _auction_key
    deals_prepared = prepare_deals_with_bid_str(deal_df)
    
    # Filter deals by auction pattern if provided
    if auction_pattern:
        try:
            regex_pattern = f"(?i){normalize_auction_user_text(auction_pattern)}"
            deals_prepared = deals_prepared.filter(pl.col("_bid_str").str.contains(regex_pattern))
        except Exception as e:
            raise ValueError(f"Invalid auction pattern: {e}")
    
    total_deals = deals_prepared.height
    
    # Sample deals to analyze (for performance, limit to 10000)
    sample_size = min(10000, total_deals)
    if sample_size < total_deals:
        sample_df = deals_prepared.sample(n=sample_size, seed=42)
    else:
        sample_df = deals_prepared
    
    analyzed_deals = sample_df.height
    
    # Prepare BT for join and join once (instead of per-row lookups)
    bt_prepared = prepare_bt_for_join(bt_seat1_df)
    joined_df = join_deals_with_bt(sample_df, bt_prepared)
    overlay = state.get("custom_criteria_overlay") or []
    
    # Batch check wrong bids (still loops but no per-row filter operations)
    result_df = batch_check_wrong_bids(joined_df, deal_criteria_by_seat_dfs, seat, criteria_overlay=overlay)
    
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

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[wrong-bid-stats] {format_elapsed(elapsed_ms)} ({analyzed_deals} analyzed, {deals_with_wrong_bid} wrong)")
    
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
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index for bitmap lookups
    deal_df = deal_df.with_row_index("_row_idx")
    
    # Prepare for join: add _bid_str and _auction_key
    deals_prepared = prepare_deals_with_bid_str(deal_df)
    
    # Filter deals by auction pattern if provided
    if auction_pattern:
        try:
            regex_pattern = f"(?i){normalize_auction_user_text(auction_pattern)}"
            deals_prepared = deals_prepared.filter(pl.col("_bid_str").str.contains(regex_pattern))
        except Exception as e:
            raise ValueError(f"Invalid auction pattern: {e}")
    
    total_deals = deals_prepared.height
    
    # Sample for performance
    sample_size = min(10000, total_deals)
    if sample_size < total_deals:
        sample_df = deals_prepared.sample(n=sample_size, seed=42)
    else:
        sample_df = deals_prepared
    
    analyzed_deals = sample_df.height
    
    # Prepare BT for join and join once (eliminates per-row filter operations)
    bt_prepared = prepare_bt_for_join(bt_seat1_df)
    joined_df = join_deals_with_bt(sample_df, bt_prepared)
    overlay = state.get("custom_criteria_overlay") or []
    
    # Track criteria failures
    criteria_fail_counts: Dict[str, int] = {}
    criteria_check_counts: Dict[str, int] = {}
    criteria_by_seat: Dict[int, Dict[str, int]] = {1: {}, 2: {}, 3: {}, 4: {}}
    
    # Process each deal - now without per-row BT lookups (already joined)
    seats_to_check = [seat] if seat else list(range(1, 5))
    for row in joined_df.iter_rows(named=True):
        if overlay:
            row = _apply_custom_criteria_overlay_to_bt_row(dict(row), overlay)
        deal_idx = row.get("_row_idx", 0)
        dealer = row.get("Dealer", "N")
        
        # Check each seat's criteria
        for s in seats_to_check:
            if s is None:
                continue
            criteria_list = row.get(f"Agg_Expr_Seat_{s}")
            if not criteria_list:
                continue
            
            seat_dfs = deal_criteria_by_seat_dfs.get(s, {})
            criteria_df = seat_dfs.get(dealer)
            if criteria_df is None or criteria_df.is_empty():
                continue
            
            for criterion in criteria_list:
                if criterion not in criteria_df.columns:
                    continue
                
                # Track check count
                criteria_check_counts[criterion] = criteria_check_counts.get(criterion, 0) + 1
                
                try:
                    bitmap_value = criteria_df[criterion][deal_idx]
                    if not bitmap_value:
                        criteria_fail_counts[criterion] = criteria_fail_counts.get(criterion, 0) + 1
                        criteria_by_seat[s][criterion] = criteria_by_seat[s].get(criterion, 0) + 1
                except (IndexError, KeyError):
                    continue
    
    # Build results sorted by fail count
    criteria_results = []
    for criterion, fail_count in sorted(criteria_fail_counts.items(), key=lambda x: -x[1]):
        check_count = criteria_check_counts.get(criterion, 0)
        fail_rate = fail_count / check_count if check_count > 0 else 0.0
        criteria_results.append({
            "criterion": criterion,
            # Aliases for Streamlit visualization
            "fail_count": fail_count,
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
        seat_breakdown[f"seat_{s}"] = [{"criterion": c, "fail_count": cnt} for c, cnt in seat_top]
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[failed-criteria-summary] {format_elapsed(elapsed_ms)} ({analyzed_deals} analyzed)")
    
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
    
    deal_df = state["deal_df"]
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index
    deal_df = deal_df.with_row_index("_row_idx")
    
    # Prepare for join: add _bid_str and _auction_key
    deals_prepared = prepare_deals_with_bid_str(deal_df)
    
    # Sample for performance
    total_deals = deals_prepared.height
    sample_size = min(10000, total_deals)
    if sample_size < total_deals:
        sample_df = deals_prepared.sample(n=sample_size, seed=42)
    else:
        sample_df = deals_prepared
    
    analyzed_deals = sample_df.height
    
    # Prepare BT for join and join once (eliminates per-row filter operations)
    bt_prepared = prepare_bt_for_join(bt_seat1_df)
    joined_df = join_deals_with_bt(sample_df, bt_prepared)
    overlay = state.get("custom_criteria_overlay") or []
    
    # Track wrong bids by (bid, seat)
    bid_seat_wrong: Dict[Tuple[str, int], int] = {}
    bid_seat_total: Dict[Tuple[str, int], int] = {}
    bid_failed_criteria: Dict[Tuple[str, int], Dict[str, int]] = {}
    
    # Process each deal - now without per-row BT lookups (already joined)
    seats_to_check = [seat] if seat else list(range(1, 5))
    for row in joined_df.iter_rows(named=True):
        if overlay:
            row = _apply_custom_criteria_overlay_to_bt_row(dict(row), overlay)
        deal_idx = row.get("_row_idx", 0)
        dealer = row.get("Dealer", "N")
        bid_str = row.get("_bid_str", "")
        
        # For each seat, track the bid and whether it's wrong
        for s in seats_to_check:
            if s is None:
                continue
            bid_at_seat = _extract_bid_at_seat(bid_str, s)
            if not bid_at_seat:
                continue
            
            key = (bid_at_seat.upper(), s)
            bid_seat_total[key] = bid_seat_total.get(key, 0) + 1
            
            # Check this seat's criteria (from joined data)
            criteria_list = row.get(f"Agg_Expr_Seat_{s}")
            if not criteria_list:
                continue
            
            seat_dfs = deal_criteria_by_seat_dfs.get(s, {})
            criteria_df = seat_dfs.get(dealer)
            if criteria_df is None or criteria_df.is_empty():
                continue
            
            seat_failed = []
            for criterion in criteria_list:
                if criterion not in criteria_df.columns:
                    continue
                try:
                    bitmap_value = criteria_df[criterion][deal_idx]
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
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[wrong-bid-leaderboard] {format_elapsed(elapsed_ms)} ({analyzed_deals} analyzed)")
    
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
) -> Dict[str, Any]:
    """Handle /bidding-arena endpoint (Bidding Arena).
    
    Provides comprehensive head-to-head comparison between two bidding models.
    Currently supports "Rules", "Raw_Rules", and "Actual" models.
    Future: "NN", "RF", "Fuzzy", etc.
    
    IMPORTANT: "Rules" model uses learned criteria + CSV overlay.
    "Raw_Rules" model finds the BT completed auction whose criteria ALL match
    the deal's hand features (using bitmap lookups). This is NOT the same as looking
    up the BT row by the actual auction string.
    
    Args:
        deals_uri: Optional file path or URL to load custom deals from.
                   Supports .parquet and .csv files. If None, uses deal_df.
    """
    t0 = time.perf_counter()
    
    # Load deals from URI or use default
    deals_source = "default"
    if deals_uri:
        try:
            deal_df = _load_deals_from_uri(deals_uri)
            deals_source = deals_uri
            print(f"[bidding-arena] Loaded {deal_df.height} deals from {deals_uri}")
        except Exception as e:
            raise ValueError(f"Failed to load deals from URI: {e}")
    else:
        deal_df = state["deal_df"]
    
    deal_criteria_by_seat_dfs = state.get("deal_criteria_by_seat_dfs", {})
    bt_seat1_df = state.get("bt_seat1_df")
    
    # Validate models
    # "Rules" uses learned criteria from bbo_bt_merged_rules.parquet + CSV overlay (default)
    # "Raw_Rules" uses original BT criteria + CSV overlay (no learned criteria)
    merged_rules_lookup = state.get("merged_rules_lookup")
    valid_models = {"Raw_Rules", "Actual"}
    if merged_rules_lookup is not None:
        valid_models.add("Rules")
    
    if model_a not in valid_models:
        raise ValueError(f"Invalid model_a: {model_a}. Valid models: {valid_models}")
    if model_b not in valid_models:
        raise ValueError(f"Invalid model_b: {model_b}. Valid models: {valid_models}")
    if model_a == model_b:
        raise ValueError("model_a and model_b must be different")
    
    # Check if Rules model is requested but not available
    if (model_a == "Rules" or model_b == "Rules") and merged_rules_lookup is None:
        raise ValueError("Rules model requested but merged_rules_lookup not loaded. "
                         "Ensure bbo_bt_merged_rules.parquet exists in data directory.")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index
    deal_df = deal_df.with_row_index("_row_idx")
    
    # Prepare deals: add _bid_str (for Actual auction display)
    deals_prepared = prepare_deals_with_bid_str(deal_df)
    
    # Filter deals by auction pattern if provided (filters on ACTUAL auction)
    if auction_pattern:
        try:
            regex_pattern = f"(?i){normalize_auction_user_text(auction_pattern)}"
            deals_prepared = deals_prepared.filter(pl.col("_bid_str").str.contains(regex_pattern))
        except Exception as e:
            raise ValueError(f"Invalid auction pattern: {e}")
    
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
        if pinned_list and "index" in deals_prepared.columns:
            pinned_set = set(pinned_list)
            order_df = pl.DataFrame({"index": pinned_list, "_pin_rank": list(range(len(pinned_list)))})
            pinned_df = (
                deals_prepared
                .join(order_df, on="index", how="inner")
                .sort("_pin_rank")
                .drop("_pin_rank")
            )

    if pinned_df is not None and pinned_df.height > 0:
        remaining_df = deals_prepared.filter(~pl.col("index").is_in(list(pinned_set)))
        remaining_n = max(0, int(sample_size) - int(pinned_df.height))
        if remaining_n > 0 and remaining_df.height > 0:
            remaining_n = min(remaining_n, remaining_df.height)
            sampled_rest = remaining_df.sample(n=remaining_n, seed=effective_seed) if remaining_n < remaining_df.height else remaining_df
            sample_df = pl.concat([pinned_df, sampled_rest], how="vertical")
        else:
            sample_df = pinned_df
    else:
        if sample_size < total_deals:
            sample_df = deals_prepared.sample(n=sample_size, seed=effective_seed)
        else:
            sample_df = deals_prepared
    
    analyzed_deals = sample_df.height
    
    # ---------------------------------------------------------------------------
    # Rules Auction Matching Strategy
    # ---------------------------------------------------------------------------
    # Deal-level precomputed candidates (Matched_BT_Indices) appear unreliable in this dataset,
    # so we do NOT use them for Rules matching.
    has_precomputed_matches = False

    # Rules candidate source:
    # - Default: restrict to BT rows that have a merged-rules entry (bt_index in merged_rules_lookup)
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
        
    def _find_rules_matches_precomputed(
        deal_idx: int,
        dealer: str,
        matched_indices: List[int] | None,
        deal_row: Dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return all matching BT candidates among the precomputed list (up to max returned)."""
        if not matched_indices:
            return [], False
        out: list[dict[str, Any]] = []
        truncated = False
        for bt_idx in matched_indices:
            bt_idx_i = int(bt_idx)
            bt_row = bt_idx_to_row.get(bt_idx_i)
            if bt_row is None:
                continue
            if _deal_meets_all_seat_criteria(deal_idx, dealer, bt_row, deal_row):
                out.append({"bt_index": bt_idx_i, "auction": bt_row.get("Auction")})
                if len(out) >= RULES_MATCHES_MAX_RETURNED:
                    truncated = True
                    break
        return out, truncated

    def _find_rules_matches_onthefly(deal_idx: int, dealer: str, deal_row: Dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], bool]:
        """Return matching BT candidates from the on-the-fly candidate pool (up to max returned)."""
        out: list[dict[str, Any]] = []
        truncated = False
        for bt_row in bt_completed_rows:
            if _deal_meets_all_seat_criteria(deal_idx, dealer, bt_row, deal_row):
                out.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction")})
                if len(out) >= RULES_MATCHES_MAX_RETURNED:
                    truncated = True
                    break
        return out, truncated

    def _find_first_rules_match_precomputed(
        deal_idx: int,
        dealer: str,
        matched_indices: List[int] | None,
        deal_row: Dict[str, Any] | None = None,
    ) -> str | None:
        """Fast path: return first matching auction among the precomputed candidate list."""
        if not matched_indices:
            return None
        for bt_idx in matched_indices:
            bt_idx_i = int(bt_idx)
            bt_row = bt_idx_to_row.get(bt_idx_i)
            if bt_row is None:
                continue
            if _deal_meets_all_seat_criteria(deal_idx, dealer, bt_row, deal_row):
                auc = bt_row.get("Auction")
                return str(auc) if auc is not None else None
            return None

    def _find_first_rules_match_onthefly(deal_idx: int, dealer: str, deal_row: Dict[str, Any] | None = None) -> str | None:
        """Fast path: return first matching auction from the on-the-fly candidate pool."""
        for bt_row in bt_completed_rows:
            if _deal_meets_all_seat_criteria(deal_idx, dealer, bt_row, deal_row):
                auc = bt_row.get("Auction")
                return str(auc) if auc is not None else None
        return None

    if has_precomputed_matches:
        # Use precomputed candidate indices but re-check against (base + overlay) criteria so the CSV can override.
        overlay = state.get("custom_criteria_overlay") or []

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
                cols = ["bt_index", "Auction", agg_expr_col(1), agg_expr_col(2), agg_expr_col(3), agg_expr_col(4)]
                cols = [c for c in cols if c in bt_seat1_df.columns]
                lookup_df = bt_seat1_df.select(cols).filter(pl.col("bt_index").is_in(needed_idxs))
                lookup_rows_base = lookup_df.to_dicts()
                # Store base rows (without overlay) for Rules model
                for r in lookup_rows_base:
                    bt_idx = r.get("bt_index")
                    if bt_idx is not None:
                        bt_idx_to_row_base[int(bt_idx)] = r
                    auc = r.get("Auction")
                    if auc:
                        # Normalize key to seat-1 view + lowercase so deal auctions with leading passes
                        # (e.g. "p-p-p-1N-p-p-p") can be looked up reliably.
                        k = _auction_key_seat1(auc)
                        bt_auction_to_row_base[k] = r
                # Apply overlay for Rules model
                if overlay:
                    lookup_rows = [_apply_custom_criteria_overlay_to_bt_row(r, overlay) for r in lookup_rows_base]
                else:
                    lookup_rows = lookup_rows_base
                for r in lookup_rows:
                    bt_idx = r.get("bt_index")
                    auction = r.get("Auction")
                    if bt_idx is None:
                        continue
                    bt_idx_i = int(bt_idx)
                    bt_idx_to_row[bt_idx_i] = r
                    if auction:
                        bt_idx_to_auction[bt_idx_i] = str(auction)
            except Exception:
                bt_idx_to_row = {}
                bt_idx_to_row_base = {}
                bt_idx_to_auction = {}

        def _find_rules_auction_precomputed(deal_idx: int, dealer: str, matched_indices: List[int] | None) -> Optional[str]:
            """Pick the first precomputed candidate that satisfies (base + overlay) criteria for this deal."""
            return _find_first_rules_match_precomputed(deal_idx, dealer, matched_indices)
    else:
        # Candidate pool(s) for criteria matching (no deal-level Matched_BT_Indices)
        if "is_completed_auction" in bt_seat1_df.columns:
            bt_completed = bt_seat1_df.filter(pl.col("is_completed_auction"))
        else:
            bt_completed = bt_seat1_df

        # Default Rules candidate pool: BT rows that have a merged rules entry.
        merged_rules_lookup = state.get("merged_rules_lookup") or {}
        if merged_rules_lookup:
            try:
                merged_bt_indices = list(merged_rules_lookup.keys())
            except Exception:
                merged_bt_indices = []
        else:
            merged_bt_indices = []

        bt_rules_default = bt_completed
        if merged_bt_indices and "bt_index" in bt_rules_default.columns:
            bt_rules_default = bt_rules_default.filter(pl.col("bt_index").is_in(merged_bt_indices))
        else:
            # If merged rules aren't available, default pool degenerates to the generic completed pool.
            rules_search_mode = "onthefly"
        
        if not search_all_bt_rows:
            # Limit pools for performance (default + fallback)
            if "matching_deal_count" in bt_rules_default.columns:
                bt_rules_default = bt_rules_default.sort("matching_deal_count", descending=True).head(5000)
            elif "bt_index" in bt_rules_default.columns:
                bt_rules_default = bt_rules_default.sort("bt_index").head(5000)
            else:
                bt_rules_default = bt_rules_default.head(5000)

            if "matching_deal_count" in bt_completed.columns:
                bt_completed = bt_completed.sort("matching_deal_count", descending=True).head(5000)
            elif "bt_index" in bt_completed.columns:
                bt_completed = bt_completed.sort("bt_index").head(5000)
            else:
                bt_completed = bt_completed.head(5000)
        
        # Convert pools to list of dicts for iteration
        bt_cols = ["Auction", "bt_index", agg_expr_col(1), agg_expr_col(2), agg_expr_col(3), agg_expr_col(4)]
        bt_cols = [c for c in bt_cols if c in bt_completed.columns]

        bt_rules_default_rows_base = bt_rules_default.select(bt_cols).to_dicts() if bt_rules_default is not None else []
        bt_completed_rows_base = bt_completed.select(bt_cols).to_dicts()

        # Build Auction->row caches for fast lookup.
        # - default: merged-rules pool only (preferred when searching candidates)
        # - full: generic completed pool (used ONLY for "is actual auction in BT?" checks)
        #
        # Key by seat-1 normalized, lowercased auction string for robust lookup.
        def _build_bt_auction_cache(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            """Build a robust seat-1 auction lookup cache.
            
            Multiple BT rows can map to the same seat-1 key if the source BT includes
            leading-pass variants (e.g., '1N-p-p-p' and 'p-1N-p-p-p'). When that happens,
            we must prefer the *canonical* row with the fewest leading passes; otherwise the
            cache can silently point at an already-rotated criteria row and seats will look
            'confused' in downstream debug output.
            """
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
                # Prefer the row with fewer leading passes (more canonical seat-1 representation).
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
        overlay = state.get("custom_criteria_overlay") or []
        if overlay:
            bt_completed_rows = [_apply_custom_criteria_overlay_to_bt_row(r, overlay) for r in bt_completed_rows_base]
        else:
            bt_completed_rows = bt_completed_rows_base
        
        def _find_rules_auction_onthefly(deal_idx: int, dealer: str) -> Optional[str]:
            """Find the first completed auction whose criteria ALL match this deal."""
            return _find_first_rules_match_onthefly(deal_idx, dealer)
    
    # ---------------------------------------------------------------------------
    # Rules Model Matching Setup (Learned Criteria)
    # ---------------------------------------------------------------------------
    # Rules model uses learned criteria from bbo_bt_merged_rules.parquet.
    # We key the lookup by bt_index to avoid fragile auction-string matching.
    
    use_rules = model_a == "Rules" or model_b == "Rules"  # Rules = learned + CSV overlay
    
    def _apply_merged_rules_to_bt_row(bt_row: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Merged_Rules to seat-1, then apply CSV overlay on top.
        
        Flow:
        1. Get merged rules for seat 1 from merged_rules_lookup (if available)
        2. Apply CSV overlay on top (so CSV can disable/modify merged rules)
        
        IMPORTANT: CSV overlay is ALWAYS applied, even if no merged rules exist.
        """
        overlay = state.get("custom_criteria_overlay") or []
        
        # Start with a copy to avoid mutating the original
        result = dict(bt_row)
        
        # Apply merged rules to seat 1 if available
        if merged_rules_lookup is not None:
            bt_idx = bt_row.get("bt_index")
            if bt_idx is not None:
                try:
                    bt_idx_i = int(bt_idx)
                    merged = merged_rules_lookup.get(bt_idx_i)
                    if merged:
                        # Rules model: seat-1 criteria is driven by merged rules (pipeline output).
                        result[agg_expr_col(1)] = list(merged)
                except Exception:
                    pass
        
        # ALWAYS apply CSV overlay on top (so CSV can modify/disable rules)
        if overlay:
            result = _apply_custom_criteria_overlay_to_bt_row(result, overlay)
        
        return result
    
    def _deal_meets_merged_criteria(deal_idx: int, dealer: str, bt_row: Dict[str, Any], deal_row: Dict[str, Any] | None = None) -> bool:
        """Check if a deal meets criteria using Merged_Rules for seat 1."""
        merged_row = _apply_merged_rules_to_bt_row(bt_row)
        return _deal_meets_all_seat_criteria(deal_idx, dealer, merged_row, deal_row)
    
    def _find_first_merged_rules_match_precomputed(
        deal_idx: int,
        dealer: str,
        matched_indices: List[int] | None,
        deal_row: Dict[str, Any] | None = None,
    ) -> str | None:
        """Find first matching auction using Merged_Rules criteria (precomputed path).
        
        Uses bt_idx_to_row_base (without overlay), then _apply_merged_rules_to_bt_row
        applies merged rules + CSV overlay.
        """
        if not matched_indices:
            return None
        for bt_idx in matched_indices:
            # Use base rows - overlay is applied inside _apply_merged_rules_to_bt_row
            bt_row = bt_idx_to_row_base.get(int(bt_idx))
            if bt_row and _deal_meets_merged_criteria(deal_idx, dealer, bt_row, deal_row):
                auc = bt_row.get("Auction")
                return str(auc) if auc is not None else None
        return None
    
    def _find_first_merged_rules_match_onthefly(deal_idx: int, dealer: str, deal_row: Dict[str, Any] | None = None) -> str | None:
        """Find first matching auction using Merged_Rules criteria (on-the-fly path).
        
        Uses bt_completed_rows_base (without overlay), then _apply_merged_rules_to_bt_row
        applies merged rules + CSV overlay.
        """
        for bt_row in bt_completed_rows_base:
            if _deal_meets_merged_criteria(deal_idx, dealer, bt_row, deal_row):
                auc = bt_row.get("Auction")
                return str(auc) if auc is not None else None
        return None

    def _find_first_merged_rules_match_default(deal_idx: int, dealer: str, deal_row: Dict[str, Any] | None = None) -> str | None:
        """Default path: search the merged-rules candidate pool first, then fall back to None."""
        for bt_row in bt_rules_default_rows_base:
            if _deal_meets_merged_criteria(deal_idx, dealer, bt_row, deal_row):
                auc = bt_row.get("Auction")
                return str(auc) if auc is not None else None
        return None
    
    def _find_merged_rules_matches_precomputed(
        deal_idx: int,
        dealer: str,
        matched_indices: List[int] | None,
        deal_row: Dict[str, Any] | None = None,
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
            if _deal_meets_merged_criteria(deal_idx, dealer, bt_row, deal_row):
                out.append({"bt_index": bt_idx_i, "auction": bt_row.get("Auction")})
                if len(out) >= RULES_MATCHES_MAX_RETURNED:
                    truncated = True
                    break
        return out, truncated

    def _find_merged_rules_matches_onthefly(deal_idx: int, dealer: str, deal_row: Dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], bool]:
        """Return matching BT candidates using merged rules, default pool first (up to max returned)."""
        out: list[dict[str, Any]] = []
        truncated = False
        # Default source: merged-rules rows
        for bt_row in bt_rules_default_rows_base:
            if _deal_meets_merged_criteria(deal_idx, dealer, bt_row, deal_row):
                out.append({"bt_index": int(bt_row.get("bt_index", 0)), "auction": bt_row.get("Auction")})
                if len(out) >= RULES_MATCHES_MAX_RETURNED:
                    truncated = True
                    break
        # Backstop: generic completed rows (avoid duplicates)
        if not truncated and len(out) < RULES_MATCHES_MAX_RETURNED:
            seen = set((m.get("bt_index"), m.get("auction")) for m in out)
            for bt_row in bt_completed_rows_base:
                if _deal_meets_merged_criteria(deal_idx, dealer, bt_row, deal_row):
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
        if model == "Actual":
            if dd_score_val is None:
                return "Actual: missing DD_Score_Declarer"
            return None
        # Rules or Raw_Rules
        model_name = model  # "Rules" or "Raw_Rules"
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

            Uses merged rules + CSV overlay (the Rules path), not Raw_Rules.
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
                    # Use up to N candidates from the precomputed list, apply merged rules + overlay
                    for bt_idx in matched_indices[:25]:
                        bt_row = bt_idx_to_row_base.get(int(bt_idx))
                        if bt_row is not None:
                            # Apply merged rules + CSV overlay (same as _apply_merged_rules_to_bt_row)
                            candidates.append(_apply_merged_rules_to_bt_row(bt_row))
                else:
                    # On-the-fly path: use base rows, apply merged rules + overlay
                    # Default pool first, then fallback pool
                    for bt_row in bt_rules_default_rows_base[:25]:
                        candidates.append(_apply_merged_rules_to_bt_row(bt_row))
                    if len(candidates) < 25:
                        for bt_row in bt_completed_rows_base[:25]:
                            candidates.append(_apply_merged_rules_to_bt_row(bt_row))

                if not candidates:
                    return "no candidates available"

                # Show variety of candidate auctions
                unique_auctions = list(set(str(c.get("Auction", "?")) for c in candidates[:10]))[:5]
                
                # Diagnostic: how many matched_indices are actually found in bt_idx_to_row_base?
                if matched_indices:
                    found_count = sum(1 for idx in matched_indices[:25] if bt_idx_to_row_base.get(int(idx)) is not None)
                    total_count = min(25, len(matched_indices))
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

        need_rules = (model_a == "Rules") or (model_b == "Rules")  # Learned rules
        need_raw_rules = (model_a == "Raw_Rules") or (model_b == "Raw_Rules")  # Original BT rules
        need_rules_matches_list = need_rules and (len(sample_deals) < 50)

        # Pick a single Rules auction (learned criteria) for scoring
        # Pass `row` as deal_row for dynamic SL evaluation
        matched_indices = row.get("Matched_BT_Indices") if has_precomputed_matches else None
        # Always define these for the sample-deals output schema (even if Rules model not requested).
        rules_actual_lead_passes: int | None = None
        rules_actual_opener_seat: int | None = None
        if need_rules:
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
            if bid_str:
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
                                txt = "; ".join(str(x) for x in crits[:25])
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

                def _apply_merged_rules_to_bt_row_no_overlay(bt_row: Dict[str, Any]) -> Dict[str, Any]:
                    """Apply merged rules to seat 1 ONLY (no CSV overlay).
                    
                    This is a debug helper so we can distinguish issues coming from:
                    - bbo_bt_merged_rules.parquet (merged rules), vs
                    - CSV overlay (runtime)
                    """
                    result = dict(bt_row)
                    if merged_rules_lookup is not None:
                        bt_idx = bt_row.get("bt_index")
                        if bt_idx is not None:
                            try:
                                bt_idx_i = int(bt_idx)
                                merged = merged_rules_lookup.get(bt_idx_i)
                                if merged:
                                    result[agg_expr_col(1)] = list(merged)
                            except Exception:
                                pass
                    return result

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
                        # As a last resort, fall back to the shared helper (still slower; should be rare)
                        try:
                            bt_match, _auction_for_search = _lookup_bt_row(bt_seat1_df, str(bid_str))
                            if bt_match.height > 0:
                                cols = ["Auction", "bt_index", agg_expr_col(1), agg_expr_col(2), agg_expr_col(3), agg_expr_col(4)]
                                cols = [c for c in cols if c in bt_match.columns]
                                bt_row_actual = bt_match.select(cols).to_dicts()[0]
                                rules_actual_bt_lookup = "found_bt_full"
                                if len(diag_first_failure_reasons) < 3:
                                    diag_first_failure_reasons.append(f"FOUND in bt_seat1_df: {_auction_for_search}")
                        except Exception as e:
                            rules_actual_bt_lookup = "lookup_error"
                            if len(diag_first_failure_reasons) < 3:
                                diag_first_failure_reasons.append(f"LOOKUP ERROR: {e}")
                
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
                        merged_bt_row_actual = _apply_merged_rules_to_bt_row(bt_row_actual)
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
                        merged_only = _apply_merged_rules_to_bt_row_no_overlay(bt_row_actual)
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
                        rules_actual_seat1_criteria = "; ".join(str(x) for x in s1[:50])
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
            
            # If actual auction didn't match (or isn't in BT), search other candidates:
            # 1) default: merged-rules candidate pool
            # 2) fallback: generic on-the-fly candidate pool
            if rules_auction is None:
                if has_precomputed_matches:
                    if not matched_indices:
                        diag_no_matched_indices += 1
                    rules_auction = _find_first_merged_rules_match_precomputed(int(deal_idx), dealer, matched_indices, row)
                else:
                    rules_auction = _find_first_merged_rules_match_default(int(deal_idx), dealer, row)
                    if rules_auction is None and rules_search_mode == "merged_default":
                        rules_auction = _find_first_merged_rules_match_onthefly(int(deal_idx), dealer, row)
            
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
        
        # Pick Raw_Rules auction (original BT criteria)
        if need_raw_rules:
            if has_precomputed_matches:
                matched_indices = row.get("Matched_BT_Indices")
                raw_rules_auction = _find_first_rules_match_precomputed(int(deal_idx), dealer, matched_indices, row)
            else:
                raw_rules_auction = _find_first_rules_match_onthefly(int(deal_idx), dealer, row)
        else:
            raw_rules_auction = None

        # Collect all matches only for deals that are surfaced in Sample Deal Comparisons.
        # Uses merged rules (the new Rules model) for consistency with rules_auction
        rules_matches: list[dict[str, Any]] = []
        rules_matches_truncated = False
        if need_rules_matches_list:
            if has_precomputed_matches:
                matched_indices = row.get("Matched_BT_Indices")
                rules_matches, rules_matches_truncated = _find_merged_rules_matches_precomputed(int(deal_idx), dealer, matched_indices, row)
            else:
                rules_matches, rules_matches_truncated = _find_merged_rules_matches_onthefly(int(deal_idx), dealer, row)
        
        # Helper to get auction result for a model
        def _get_model_result(model: str) -> Tuple[Any, Any, Any]:
            """Get (contract, auction, dd_score) for a model.
            
            All auction strings are normalized to canonical uppercase for consistent comparison.
            """
            if model == "Actual":
                # Normalize to canonical case for consistent display/comparison
                auction_out = normalize_auction_case(bid_str)
                return row.get("Contract", ""), auction_out, row.get("DD_Score_Declarer")
            elif model == "Rules":
                if rules_auction:
                    auction_out = normalize_auction_case(rules_auction)
                    return (
                        get_ai_contract(rules_auction, dealer),
                        auction_out,
                        get_dd_score_for_auction(rules_auction, dealer, row),
                    )
                return None, None, None
            elif model == "Raw_Rules":
                if raw_rules_auction:
                    auction_out = normalize_auction_case(raw_rules_auction)
                    return (
                        get_ai_contract(raw_rules_auction, dealer),
                        auction_out,
                        get_dd_score_for_auction(raw_rules_auction, dealer, row),
                    )
                return None, None, None
            return None, None, None
        
        # Model A
        contract_a, auction_a, dd_score_a = _get_model_result(model_a)
        
        # Model B
        contract_b, auction_b, dd_score_b = _get_model_result(model_b)
        
        # Collect sample deals (include rejected deals, with reason)
        if len(sample_deals_output) < max_sample_output:
            reason_a = _rejection_reason_for(model_a, auction_a, dd_score_a, dealer)
            reason_b = _rejection_reason_for(model_b, auction_b, dd_score_b, dealer)
            rejection_reason = reason_a or reason_b

            # Add a small diagnostic for the common confusion case: Rules has no match.
            rules_debug = ""
            if (model_a != "Actual" and auction_a is None) or (model_b != "Actual" and auction_b is None):
                mi = row.get("Matched_BT_Indices") if has_precomputed_matches else None
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
                    "Result": row.get("Result", ""),
                    "Score": row.get("Score", ""),
                    "ParScore": row.get("ParScore", par_score),
                }
            )

        # Skip if we don't have scores for both models
        if dd_score_a is None or dd_score_b is None:
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
    print(f"[bidding-arena] {format_elapsed(elapsed_ms)} ({contracts_compared}/{analyzed_deals} compared)")
    print(f"[bidding-arena] DIAG: no_matched_indices={diag_no_matched_indices}, rules_none={diag_rules_none}")
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
                "mode=merged_default means Rules first searches BT rows with a merged-rules entry (bt_index in merged_rules_lookup), "
                "then falls back to the generic on-the-fly pool if no match is found."
            ),
        },
        "total_deals": total_deals,
        "analyzed_deals": analyzed_deals,
        "deals_compared": contracts_compared,
        "auction_pattern": auction_pattern,
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
_DD_ANALYSIS_STATS_THRESHOLD = 5000  # Use full data if <= this many matches
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


def _find_bt_row_for_auction(bt_df: pl.DataFrame, auction_normalized: str) -> pl.DataFrame:
    """Find a BT row matching the normalized auction string.
    
    Works with both partial and completed auctions.
    Tries: exact match → case-insensitive exact → prefix match.
    Returns the matching rows DataFrame or empty DataFrame if no match.
    """
    auction_col = bt_df["Auction"].cast(pl.Utf8).str.replace(r"(?i)^(p-)+", "")
    
    # Try exact match first (case-sensitive)
    matches = bt_df.filter(auction_col == auction_normalized)
    if matches.height > 0:
        return matches
    
    # Try case-insensitive exact match
    try:
        regex_pattern = f"(?i)^{re.escape(auction_normalized)}$"
        matches = bt_df.filter(auction_col.str.contains(regex_pattern))
        if matches.height > 0:
            return matches
    except Exception as e:
        print(f"[auction-dd-analysis] Regex match failed: {e}")
    
    # Try prefix match (for finding auctions that start with the input)
    try:
        prefix_pattern = f"(?i)^{re.escape(auction_normalized)}(-|$)"
        matches = bt_df.filter(auction_col.str.contains(prefix_pattern))
        if matches.height > 0:
            return matches
    except Exception as e:
        print(f"[auction-dd-analysis] Prefix match failed: {e}")
    
    return bt_df.head(0)  # Empty DataFrame


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
    
    # DO NOT filter to completed auctions - we want to support partial auctions like "1N"
    # The BT contains rows for all auction prefixes with their accumulated criteria
    bt_df = bt_seat1_df
    
    # Normalize auction input (supports '-', whitespace, or ',' separators), then strip leading passes to match seat-1 view
    auction_normalized = re.sub(r"(?i)^(p-)+", "", normalize_auction_input(auction))
    if not auction_normalized:
        raise ValueError(f"Invalid auction: '{auction}'")
    
    # Find matching BT row (exact, regex, or prefix match)
    exact_matches = _find_bt_row_for_auction(bt_df, auction_normalized)
    
    if exact_matches.height == 0:
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
    
    # Use first match
    overlay = state.get("custom_criteria_overlay") or []
    bt_row = _apply_custom_criteria_overlay_to_bt_row(dict(exact_matches.row(0, named=True)), overlay)
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
    """Fast lookup of available next bids using BT's next_bid_indices.
    
    Uses the sorted BT structure for efficient lookups:
    - For opening bids: filter by is_opening_bid=True
    - For continuations: find parent row, then lookup next_bid_indices
    
    Returns:
        - auction_input: The input auction
        - auction_normalized: Normalized form
        - next_bids: List of dicts with bid, bt_index, agg_expr, is_completed_auction
        - elapsed_ms: Processing time
    """
    t0 = time.perf_counter()
    
    bt_seat1_df = state.get("bt_seat1_df")
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    if "next_bid_indices" not in bt_seat1_df.columns:
        raise ValueError("next_bid_indices column not loaded in bt_seat1_df")
    
    auction_input = normalize_auction_input(auction)
    auction_normalized = re.sub(r"(?i)^(p-)+", "", auction_input) if auction_input else ""
    
    next_bids: List[Dict[str, Any]] = []
    
    if not auction_input:
        # Opening bids: use pre-computed bt_openings_df (tiny, fast)
        bt_openings_df = state.get("bt_openings_df")
        
        if bt_openings_df is not None and bt_openings_df.height > 0:
            # Use pre-computed opening bids table (very fast)
            # bt_openings_df has: bt_index, Auction (the bid), seat, Agg_Expr_Seat_1..4
            # Get unique Auction values (opening bids) for seat=1
            seat1_df = bt_openings_df.filter(pl.col("seat") == 1) if "seat" in bt_openings_df.columns else bt_openings_df
            unique_auctions = seat1_df.select("Auction").unique()
            
            seen_bids: set[str] = set()
            for row in seat1_df.iter_rows(named=True):
                bid = row.get("Auction") or ""
                if bid and bid.upper() not in seen_bids:
                    seen_bids.add(bid.upper())
                    next_bids.append({
                        "bid": bid.upper(),
                        "bt_index": row.get("bt_index"),
                        "agg_expr": row.get("Agg_Expr_Seat_1") or [],
                        "is_completed_auction": False,  # Opening bids are not complete
                    })
        else:
            # Fallback: filter bt_seat1_df (slower)
            if "is_opening_bid" not in bt_seat1_df.columns:
                raise ValueError("is_opening_bid column not found")
            if "candidate_bid" not in bt_seat1_df.columns:
                raise ValueError("candidate_bid column not found")
            
            opening_df = bt_seat1_df.filter(pl.col("is_opening_bid"))
            
            # Get unique bids efficiently
            cols_needed = ["candidate_bid", "bt_index", "Agg_Expr_Seat_1", "is_completed_auction"]
            cols_available = [c for c in cols_needed if c in opening_df.columns]
            
            unique_bids_df = opening_df.select(cols_available).unique(subset=["candidate_bid"], keep="first")
            
            for row in unique_bids_df.iter_rows(named=True):
                bid = row.get("candidate_bid") or ""
                if bid:
                    next_bids.append({
                        "bid": bid.upper(),
                        "bt_index": row.get("bt_index"),
                        "agg_expr": row.get("Agg_Expr_Seat_1") or [],
                        "is_completed_auction": row.get("is_completed_auction", False),
                    })
    else:
        # Find parent BT row for this auction prefix using FAST exact match
        auction_for_lookup = auction_input.rstrip("-").upper() if auction_input else ""
        
        # Check for pass-only prefix
        is_passes_only = not auction_normalized or auction_normalized.lower() in ("p", "")
        expected_passes = _count_leading_passes(auction_input)
        
        if is_passes_only and expected_passes > 0:
            auction_for_lookup = "-".join(["P"] * expected_passes)
        
        # Fast exact match on Auction column (no regex, no column transformation)
        parent_matches = bt_seat1_df.filter(
            pl.col("Auction").cast(pl.Utf8).str.to_uppercase() == auction_for_lookup
        )
        
        if parent_matches.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "auction_input": auction_input,
                "auction_normalized": auction_normalized,
                "next_bids": [],
                "error": f"No BT row found for auction '{auction_for_lookup}'",
                "elapsed_ms": round(elapsed_ms, 1),
            }
        
        parent_row = parent_matches.row(0, named=True)
        next_indices = parent_row.get("next_bid_indices") or []
        
        if not next_indices:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "auction_input": auction_input,
                "auction_normalized": auction_normalized,
                "parent_bt_index": parent_row.get("bt_index"),
                "next_bids": [],
                "elapsed_ms": round(elapsed_ms, 1),
            }
        
        # Lookup next bid rows by bt_index
        next_df = bt_seat1_df.filter(pl.col("bt_index").is_in(list(next_indices)))
        
        # Determine seat for Agg_Expr lookup
        call_tokens = [t for t in auction_for_lookup.split("-") if t] if auction_for_lookup else []
        next_seat = (len(call_tokens) % 4) + 1
        agg_expr_col = f"Agg_Expr_Seat_{next_seat}"
        
        # Get unique bids efficiently (next_df is typically small, but be safe)
        cols_needed = ["candidate_bid", "bt_index", agg_expr_col, "is_completed_auction"]
        cols_available = [c for c in cols_needed if c in next_df.columns]
        
        unique_next_df = next_df.select(cols_available).unique(subset=["candidate_bid"], keep="first")
        
        for row in unique_next_df.iter_rows(named=True):
            bid = row.get("candidate_bid") or ""
            if bid:
                next_bids.append({
                    "bid": bid.upper(),
                    "bt_index": row.get("bt_index"),
                    "agg_expr": row.get(agg_expr_col) or [],
                    "is_completed_auction": row.get("is_completed_auction", False),
                })
    
    # Sort: regular bids by level/suit, then D/R, then P
    def sort_key(item: Dict[str, Any]) -> tuple:
        bid = item.get("bid", "")
        if bid == "P":
            return (2, 0, "")
        if bid in ("D", "R"):
            return (1, 0, bid)
        try:
            level = int(bid[0])
            suit = bid[1:] if len(bid) > 1 else ""
            suit_order = {"C": 1, "D": 2, "H": 3, "S": 4, "N": 5}.get(suit, 0)
            return (0, level, suit_order)
        except:
            return (0, 0, bid)
    
    next_bids.sort(key=sort_key)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[list-next-bids] {format_elapsed(elapsed_ms)} ({len(next_bids)} bids for '{auction_input or '(opening)'}')")
    
    return {
        "auction_input": auction_input,
        "auction_normalized": auction_normalized,
        "next_bids": next_bids,
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
        # Truly empty auction: get all opening bids (seat 1 bids)
        next_bid_rows = bt_seat1_df.filter(pl.col("is_opening_bid"))
    elif is_passes_only and expected_passes > 0:
        # Pass-only prefix (e.g., "p-", "p-p-", "p-p-p-"): look up the passes in BT
        # Build the pass sequence: "p", "p-p", or "p-p-p"
        pass_auction = "-".join(["p"] * expected_passes)
        bt_matches = _find_bt_row_for_auction(bt_seat1_df, pass_auction)
        
        if bt_matches.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "auction_input": auction_input,
                "auction_normalized": pass_auction,
                "parent_bt_row": None,
                "bid_rankings": [],
                "total_next_bids": 0,
                "error": f"Pass sequence '{pass_auction}' not found in BT",
                "elapsed_ms": round(elapsed_ms, 1),
            }
        
        parent_row = bt_matches.row(0, named=True)
        parent_bt_row = {
            "bt_index": parent_row.get("bt_index"),
            "Auction": parent_row.get("Auction"),
            "candidate_bid": parent_row.get("candidate_bid"),
        }
        
        # Get next_bid_indices for bids after the passes
        next_indices = parent_row.get("next_bid_indices") or []
        if not next_indices:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "auction_input": auction_input,
                "auction_normalized": pass_auction,
                "parent_bt_row": parent_bt_row,
                "bid_rankings": [],
                "total_next_bids": 0,
                "message": f"No next bids found after {expected_passes} pass(es)",
                "elapsed_ms": round(elapsed_ms, 1),
            }
        
        # Get the BT rows for all next bids
        next_bid_rows = bt_seat1_df.filter(pl.col("bt_index").is_in(list(next_indices)))
        
        # Update auction_normalized for display
        auction_normalized = pass_auction
    else:
        # Find the BT row for the auction (includes any leading passes)
        bt_matches = _find_bt_row_for_auction(bt_seat1_df, auction_for_lookup)
        
        if bt_matches.height == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "auction_input": auction_input,
                "auction_normalized": auction_for_lookup,
                "parent_bt_row": None,
                "bid_rankings": [],
                "total_next_bids": 0,
                "error": f"Auction '{auction_for_lookup}' not found in BT",
                "elapsed_ms": round(elapsed_ms, 1),
            }
        
        parent_row = bt_matches.row(0, named=True)
        parent_bt_row = {
            "bt_index": parent_row.get("bt_index"),
            "Auction": parent_row.get("Auction"),
            "candidate_bid": parent_row.get("candidate_bid"),
        }
        
        # Get next_bid_indices
        next_indices = parent_row.get("next_bid_indices") or []
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
        
        # Get the BT rows for all next bids
        next_bid_rows = bt_seat1_df.filter(pl.col("bt_index").is_in(list(next_indices)))
    
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
        first_row = _apply_custom_criteria_overlay_to_bt_row(dict(next_bid_rows.row(0, named=True)), overlay)
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
        
        # Build criteria mask using optimized parent mask where possible
        global_mask = None
        
        # Optimization: Pre-fetch non-constant criteria lists
        # Apply overlay per candidate row before extracting criteria lists
        row_overlayed = _apply_custom_criteria_overlay_to_bt_row(dict(row), overlay) if overlay else row
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
            # ... (no changes to continue block) ...
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
            
        # Combine with opening seat mask ONLY if parent_mask optimization was NOT used
        if not constant_seats and opening_seat_mask is not None:
            global_mask = global_mask & opening_seat_mask
            
        if not global_mask.any():
            # ... (no changes to continue block) ...
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
            # ... (already handled by mask.any() above, but kept for safety)
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
        
        if "Vul" in matched_df.columns and "ParScore" in matched_df.columns:
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
                
                # Compute EV for NV deals relative to bidder direction (seat-based)
                if bid_level and bid_strain and "Dealer" in nv_df.columns:
                    for bidder in ["N", "E", "S", "W"]:
                        pair = "NS" if bidder in ["N", "S"] else "EW"
                        ev_col = f"EV_{pair}_{bidder}_{bid_strain}_{bid_level}_NV"
                        if ev_col in bidder_nv_df.columns:
                            sub = bidder_nv_df.filter(pl.col("_Bidder") == bidder)[ev_col].drop_nulls()
                            n = sub.len()
                            if n:
                                s = float(sub.sum())
                                ss = float((sub * sub).sum())
                                ev_nv_sum += s
                                ev_nv_sum_sq += ss
                                ev_nv_n += int(n)
            
            if v_count > 0:
                bidder_v_df = v_df.with_columns(bidder_expr.alias("_Bidder"))
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
                
                # Compute EV for V deals relative to bidder direction (seat-based)
                if bid_level and bid_strain and "Dealer" in v_df.columns:
                    for bidder in ["N", "E", "S", "W"]:
                        pair = "NS" if bidder in ["N", "S"] else "EW"
                        ev_col = f"EV_{pair}_{bidder}_{bid_strain}_{bid_level}_V"
                        if ev_col in bidder_v_df.columns:
                            sub = bidder_v_df.filter(pl.col("_Bidder") == bidder)[ev_col].drop_nulls()
                            n = sub.len()
                            if n:
                                s = float(sub.sum())
                                ss = float((sub * sub).sum())
                                ev_v_sum += s
                                ev_v_sum_sq += ss
                                ev_v_n += int(n)
        
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
        # Opening bids: bt rows are directly selectable by is_opening_bid + candidate_bid
        open_df = bt_seat1_df.filter(pl.col("is_opening_bid"))
        if "candidate_bid" not in open_df.columns:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        bid_df = open_df.filter(pl.col("candidate_bid") == next_bid)
        if bid_df.height == 0:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        bt_row = bid_df.row(0, named=True)
    else:
        # Find the parent BT row for this auction prefix (including pass-only prefixes)
        if is_passes_only and expected_passes > 0:
            pass_auction = "-".join(["p"] * expected_passes)
            parent_matches = _find_bt_row_for_auction(bt_seat1_df, pass_auction)
        else:
            parent_matches = _find_bt_row_for_auction(bt_seat1_df, auction_for_lookup)

        if parent_matches.height == 0:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        parent_row = parent_matches.row(0, named=True)

        # Find candidate row for the selected next bid using next_bid_indices (same as handle_rank_bids_by_ev)
        next_indices = parent_row.get("next_bid_indices") or []
        if not next_indices:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        idx_df = bt_seat1_df.filter(pl.col("bt_index").is_in(list(next_indices)))
        if idx_df.height == 0:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        if "candidate_bid" not in idx_df.columns:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        next_bid_row_df = idx_df.filter(pl.col("candidate_bid") == next_bid)
        if next_bid_row_df.height == 0:
            return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

        bt_row = next_bid_row_df.row(0, named=True)

    if bt_row is None:
        return {"deals": [], "total_matches": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    # Apply hot-reloadable custom criteria overlay before building criteria masks.
    overlay = state.get("custom_criteria_overlay") or []
    bt_row = _apply_custom_criteria_overlay_to_bt_row(dict(bt_row), overlay)
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
