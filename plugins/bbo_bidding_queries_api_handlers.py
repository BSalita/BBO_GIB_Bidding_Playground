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

import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import duckdb  # pyright: ignore[reportMissingImports]
import polars as pl

# These imports are needed for the handler logic
from endplay.types import Deal, Vul, Player
from endplay.dds import calc_dd_table, par

from bbo_bidding_queries_lib import (
    calculate_imp,
    normalize_auction_pattern,
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

# Import from common module (eliminates code duplication)
from plugins.bbo_handlers_common import (
    # Constants
    SEAT_RANGE,
    MAX_SAMPLE_SIZE,
    DEFAULT_SEED,
    agg_expr_col,
    wrong_bid_col,
    invalid_criteria_col,
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
    # Vectorized join helpers (performance optimization)
    prepare_deals_with_bid_str,
    prepare_bt_for_join,
    join_deals_with_bt,
    batch_check_wrong_bids,
)


# ===========================================================================
# Note: Common helpers (_display_auction_with_seat_prefix, _expand_row_to_all_seats,
# _normalize_to_seat1, _take_rows_by_index, _safe_float, _bid_value_to_str,
# _par_contract_signature, _dedup_par_contracts, _format_par_contracts,
# _ev_list_for_par_contracts, _effective_seed, _seat_direction_map,
# _extract_bid_at_seat, _check_deal_criteria_conformance_bitmap, _lookup_bt_row)
# are imported from handlers_common.py to eliminate code duplication.
# ===========================================================================


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
        seat_criteria_df = seat_criteria_for_seat.get(dealer)
        if seat_criteria_df is None or seat_criteria_df.is_empty():
            continue

        dealer_mask = dealer_series == dealer
        crit_mask: pl.Series | None = None
        for crit in valid_criteria:
            col = seat_criteria_df[crit]
            crit_mask = col if crit_mask is None else (crit_mask & col)

        if crit_mask is None:
            continue

        combined = dealer_mask & crit_mask
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
    print(f"[openings-by-deal-index] {elapsed_ms:.1f}ms, {len(out_deals)} deals")
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
        print(f"[random-auction-sequences] {elapsed_ms:.1f}ms (empty)")
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

    for row in sampled_df.iter_rows(named=True):
        prev_indices = row.get(prev_idx_col, [])

        sequence_data: List[Dict[str, Any]] = []
        if prev_indices:
            sequence_data.extend(prev_rows_lookup[idx] for idx in prev_indices if idx in prev_rows_lookup)
        sequence_data.append({c: row[c] for c in available_display_cols if c in row})

        seq_df = pl.DataFrame(sequence_data)
        if index_col in seq_df.columns:
            seq_df = seq_df.sort(index_col)
        
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
    print(f"[random-auction-sequences] {elapsed_ms:.1f}ms ({len(out_samples)} samples)")
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
        pattern = normalize_auction_pattern(pattern)
        auction_expr = pl.col("Auction").cast(pl.Utf8)
    
    is_regex = pattern.startswith("^") or pattern.endswith("$")
    regex_pattern = f"(?i){pattern}"
    filtered_df = base_df.filter(auction_expr.str.contains(regex_pattern))

    filtered_df, rejected_df = apply_auction_criteria_fn(filtered_df, track_rejected=True)

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "pattern": pattern,
            "samples": [],
            "total_matching": 0,
            "rejected_count": rejected_df.height if rejected_df is not None else 0,
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

    for row in sampled_df.iter_rows(named=True):
        prev_indices = row.get(prev_idx_col, [])

        sequence_data: List[Dict[str, Any]] = []
        if prev_indices:
            sequence_data.extend(prev_rows_lookup[idx] for idx in prev_indices if idx in prev_rows_lookup)
        sequence_data.append({c: row[c] for c in available_display_cols if c in row})

        seq_df = pl.DataFrame(sequence_data)
        if index_col in seq_df.columns:
            seq_df = seq_df.sort(index_col)

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
    print(f"[auction-sequences-matching] {elapsed_ms:.1f}ms ({len(out_samples)} samples)")
    return {
        "pattern": pattern,
        "samples": out_samples,
        "total_matching": total_matching,
        "rejected_count": rejected_df.height if rejected_df is not None else 0,
        "elapsed_ms": round(elapsed_ms, 1),
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
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[pbn-lookup] Found {matching.height} matches in {elapsed_ms:.1f}ms")
    
    # Rename 'bid' to 'Actual_Auction' in output
    matches_list = matching.to_dicts()
    for m in matches_list:
        if "bid" in m:
            m["Actual_Auction"] = _bid_value_to_str(m.pop("bid"))
    
    return {
        "matches": matches_list,
        "count": matching.height,
        "total_in_df": deal_df.height,
        "pbn_searched": pbn_input,
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
        pattern = normalize_auction_pattern(pattern)
        auction_expr = pl.col("Auction").cast(pl.Utf8)

    regex_pattern = f"(?i){pattern}"
    filtered_df = base_df.filter(auction_expr.str.contains(regex_pattern))

    filtered_df, rejected_df = apply_auction_criteria_fn(filtered_df, track_rejected=True)

    if filtered_df.height == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[deals-matching-auction] {elapsed_ms:.1f}ms (no matches)")
        result: Dict[str, Any] = {"pattern": pattern, "auctions": [], "elapsed_ms": round(elapsed_ms, 1)}
        if rejected_df is not None and rejected_df.height > 0:
            result["criteria_rejected"] = rejected_df.to_dicts()
        return result

    sample_n = min(n_auction_samples, filtered_df.height)
    effective_seed = _effective_seed(seed)
    sampled_auctions = filtered_df.sample(n=sample_n, seed=effective_seed)

    deal_display_cols = [
        "index", "Dealer", "Vul", "Actual_Auction", "Contract", "Hand_N", "Hand_E", "Hand_S", "Hand_W",
        "Declarer", "Result", "Tricks", "Score", "DD_Score_Declarer", "EV_Score_Declarer",
        "EV_Declarer", "ParScore", "ParContracts", "EV_ParContracts",
    ]

    out_auctions: List[Dict[str, Any]] = []
    
    # Initialize loop variables for type checker (they're always bound when used due to early return)
    auction: str = ""
    auction_info: Dict[str, Any] = {}
    combined_df: pl.DataFrame = pl.DataFrame()

    for auction_row in sampled_auctions.iter_rows(named=True):
        auction = auction_row["Auction"]

        auction_info = {
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
        
        actual_final_seat = 0
        for s in range(1, 5):
            agg_col = f"Agg_Expr_Seat_{s}"
            if agg_col in auction_row and auction_row[agg_col]:
                actual_final_seat = s
        
        for s in range(1, actual_final_seat + 1):
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
                                if criterion in seat_criteria_df.columns:
                                    if criterion not in criteria_found[seat_key]:
                                        criteria_found[seat_key].append(criterion)
                                else:
                                    if criterion not in criteria_missing[seat_key]:
                                        criteria_missing[seat_key].append(criterion)
                            break

        for dealer in DIRECTIONS:
            dealer_mask = deal_df["Dealer"] == dealer
            if not dealer_mask.any():
                continue

            combined_mask = dealer_mask.clone()

            for s in range(1, actual_final_seat + 1):
                agg_col = f"Agg_Expr_Seat_{s}"
                if agg_col not in auction_row:
                    continue
                criteria_list = auction_row[agg_col]
                if not criteria_list:
                    continue

                if s in deal_criteria_by_seat_dfs and dealer in deal_criteria_by_seat_dfs[s]:
                    seat_criteria_df = deal_criteria_by_seat_dfs[s][dealer]
                    for criterion in criteria_list:
                        if criterion in seat_criteria_df.columns:
                            criterion_values = seat_criteria_df[criterion]
                            combined_mask = combined_mask & criterion_values

            matching_idx = combined_mask.arg_true()
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
            # Remove internal row index from output
            deal_row.pop("_row_idx", None)
            
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
                agg_exprs.append(pl.col("EV_Score_Declarer").cast(pl.Float64, strict=False).mean().alias("Avg_EV_Contract"))
            if has_ev_rules:
                agg_exprs.append(pl.col("EV_Rules").cast(pl.Float64, strict=False).mean().alias("Avg_EV_Rules"))
            if has_ev_contract and has_ev_rules:
                agg_exprs.append(
                    (pl.col("EV_Rules").cast(pl.Float64, strict=False) - pl.col("EV_Score_Declarer").cast(pl.Float64, strict=False))
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
            
            auction_info["contract_makes_count"] = int((dd_actual >= 0).sum())
            auction_info["rules_makes_count"] = int((dd_rules >= 0).sum())
            auction_info["contract_par_count"] = int((dd_actual == par_score_col).sum())
            auction_info["rules_par_count"] = int((dd_rules == par_score_col).sum())
            
            if has_ev_contract:
                ev_contract_mean = deals_with_computed["EV_Score_Declarer"].cast(pl.Float64, strict=False).mean()
                ev_contract_f = _safe_float(ev_contract_mean)
                auction_info["avg_ev_contract"] = round(ev_contract_f, 2) if ev_contract_f is not None else None
            if has_ev_rules:
                ev_rules_mean = deals_with_computed["EV_Rules"].cast(pl.Float64, strict=False).mean()
                ev_rules_f = _safe_float(ev_rules_mean)
                auction_info["avg_ev_rules"] = round(ev_rules_f, 2) if ev_rules_f is not None else None
            if has_ev_contract and has_ev_rules:
                ev_diff_mean = (deals_with_computed["EV_Rules"].cast(pl.Float64, strict=False) - 
                                deals_with_computed["EV_Score_Declarer"].cast(pl.Float64, strict=False)).mean()
                ev_diff_f = _safe_float(ev_diff_mean)
                auction_info["avg_ev_diff"] = round(ev_diff_f, 2) if ev_diff_f is not None else None
        
        display_cols_set = set(deal_display_cols) | {"DD_Score_Rules", "EV_Rules", "Rules_Contract", "IMP_Rules_vs_Actual"}
        auction_info["deals"] = [{k: v for k, v in d.items() if k in display_cols_set} for d in deals_list]

        out_auctions.append(auction_info)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_deals_count = sum(len(a.get("deals", [])) for a in out_auctions)
    print(f"[deals-matching-auction] {elapsed_ms:.1f}ms ({len(out_auctions)} auctions, {total_deals_count} deals)")
    
    response: Dict[str, Any] = {"pattern": pattern, "auctions": out_auctions, "elapsed_ms": round(elapsed_ms, 1)}
    if dist_pattern or sorted_shape:
        response["dist_filter"] = {"dist_pattern": dist_pattern, "sorted_shape": sorted_shape, "direction": dist_direction}
    if wrong_bid_filter != "all":
        response["wrong_bid_filter"] = wrong_bid_filter
    if rejected_df is not None and rejected_df.height > 0:
        response["criteria_rejected"] = rejected_df.to_dicts()
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
            auction_pattern = normalize_auction_pattern(auction_pattern)
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
    
    # Expand each matched row to 4 seat variants when allow_initial_passes is True
    if allow_initial_passes:
        expanded_rows = []
        for row in result_rows:
            expanded_rows.extend(_expand_row_to_all_seats(row, allow_initial_passes=True))
        result_rows = expanded_rows
        # Update total_matches to reflect expanded count
        total_matches = total_matches * 4
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[bidding-table-statistics] {elapsed_ms:.1f}ms ({len(result_rows)} rows from {total_matches} matches)")
    
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
    # Display: prepend leading passes for the requested seat.
    for r in result_rows:
        if "Auction" in r:
            r["Auction"] = _display_auction_with_seat_prefix(r.get("Auction"), seat)
        if "Rules_Auction" in r:
            r["Rules_Auction"] = _display_auction_with_seat_prefix(r.get("Rules_Auction"), seat)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    criteria_filtered = pre_criteria_count - post_criteria_count
    print(f"[find-matching-auctions] {elapsed_ms:.1f}ms ({len(result_rows)} matches, {criteria_filtered} filtered by CSV criteria)")
    
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
    print(f"[process-pbn] {elapsed_ms:.1f}ms ({len(results)} deals, type={input_type})")
    
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
    
    pattern = normalize_auction_pattern(auction_pattern)
    
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
            # Strip leading "p-" prefixes (use regex, not lstrip which removes individual chars)
            auction_for_search = re.sub(r"^(p-)+", "", (bid_auction.lower() if bid_auction else ""))

            bt_match = bt_lookup_df.filter(pl.col("Auction").cast(pl.Utf8).str.to_lowercase() == auction_for_search)
            if bt_match.height == 0 and not auction_for_search.endswith("-p-p-p"):
                auction_with_passes = auction_for_search + "-p-p-p"
                bt_match = bt_lookup_df.filter(pl.col("Auction").cast(pl.Utf8).str.to_lowercase() == auction_with_passes)
            if bt_match.height > 0:
                bt_row = bt_match.row(0, named=True)
                bt_info = {c: bt_row.get(c) for c in available_bt_cols if c in bt_row}
                bt_auction = bt_row.get("Auction")
        
        if bt_auction:
            group_deals = group_deals.with_columns(pl.lit(bt_auction).alias("Auction"))
        
        if "Score" in group_deals.columns and "ParScore" in group_deals.columns:
            group_deals = group_deals.with_columns(
                (pl.col("Score").cast(pl.Int64, strict=False) - pl.col("ParScore").cast(pl.Int64, strict=False)).alias("Score_Delta")
            )
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
            # Remove internal row index from output
            deal_row.pop("_row_idx", None)

        auction_groups.append({
            "auction": bid_auction, "bt_auction": bt_auction, "deal_count": deal_count,
            "sample_count": group_deals.height, "bt_info": bt_info, "stats": stats,
            "wrong_bid_count": wrong_bid_count,
            "deals": deals_list,
        })
    
    # Sort groups by number of leading passes (seat order: 1, 2, 3, 4)
    def count_leading_passes(auction: str) -> int:
        count = 0
        parts = auction.lower().split("-")
        for part in parts:
            if part == "p":
                count += 1
            else:
                break
        return count
    
    auction_groups.sort(key=lambda g: count_leading_passes(g.get("auction", "")))
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_deals_out = sum(g["sample_count"] for g in auction_groups)
    print(f"[group-by-bid] {elapsed_ms:.1f}ms ({len(auction_groups)} groups, {total_deals_out} deals)")
    
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
            regex_pattern = f"(?i){normalize_auction_pattern(auction_pattern)}"
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
    
    # Batch check wrong bids (still loops but no per-row filter operations)
    result_df = batch_check_wrong_bids(joined_df, deal_criteria_by_seat_dfs, seat)
    
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
    print(f"[wrong-bid-stats] {elapsed_ms:.1f}ms ({analyzed_deals} analyzed, {deals_with_wrong_bid} wrong)")
    
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
            regex_pattern = f"(?i){normalize_auction_pattern(auction_pattern)}"
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
    
    # Track criteria failures
    criteria_fail_counts: Dict[str, int] = {}
    criteria_check_counts: Dict[str, int] = {}
    criteria_by_seat: Dict[int, Dict[str, int]] = {1: {}, 2: {}, 3: {}, 4: {}}
    
    # Process each deal - now without per-row BT lookups (already joined)
    seats_to_check = [seat] if seat else list(range(1, 5))
    for row in joined_df.iter_rows(named=True):
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
    print(f"[failed-criteria-summary] {elapsed_ms:.1f}ms ({analyzed_deals} analyzed)")
    
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
    
    # Track wrong bids by (bid, seat)
    bid_seat_wrong: Dict[Tuple[str, int], int] = {}
    bid_seat_total: Dict[Tuple[str, int], int] = {}
    bid_failed_criteria: Dict[Tuple[str, int], Dict[str, int]] = {}
    
    # Process each deal - now without per-row BT lookups (already joined)
    seats_to_check = [seat] if seat else list(range(1, 5))
    for row in joined_df.iter_rows(named=True):
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
    print(f"[wrong-bid-leaderboard] {elapsed_ms:.1f}ms ({analyzed_deals} analyzed)")
    
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
) -> Dict[str, Any]:
    """Handle /bidding-arena endpoint (Bidding Arena).
    
    Provides comprehensive head-to-head comparison between two bidding models.
    Currently supports "Rules" and "Actual" models.
    Future: "NN", "RF", "Fuzzy", etc.
    Uses vectorized join instead of per-row BT lookups for performance.
    
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
    valid_models = {"Rules", "Actual"}  # Future: add "NN", "RF", "Fuzzy"
    if model_a not in valid_models:
        raise ValueError(f"Invalid model_a: {model_a}. Valid models: {valid_models}")
    if model_b not in valid_models:
        raise ValueError(f"Invalid model_b: {model_b}. Valid models: {valid_models}")
    if model_a == model_b:
        raise ValueError("model_a and model_b must be different")
    
    if bt_seat1_df is None:
        raise ValueError("bt_seat1_df not loaded")
    
    # Add row index
    deal_df = deal_df.with_row_index("_row_idx")
    
    # Prepare for join: add _bid_str and _auction_key
    deals_prepared = prepare_deals_with_bid_str(deal_df)
    
    # Filter deals by auction pattern if provided
    if auction_pattern:
        try:
            regex_pattern = f"(?i){normalize_auction_pattern(auction_pattern)}"
            deals_prepared = deals_prepared.filter(pl.col("_bid_str").str.contains(regex_pattern))
        except Exception as e:
            raise ValueError(f"Invalid auction pattern: {e}")
    
    total_deals = deals_prepared.height
    
    # Sample deals
    effective_seed = _effective_seed(seed)
    if sample_size < total_deals:
        sample_df = deals_prepared.sample(n=sample_size, seed=effective_seed)
    else:
        sample_df = deals_prepared
    
    analyzed_deals = sample_df.height
    
    # Prepare BT for join - include Auction column for Rules contract lookup
    bt_prepared = prepare_bt_for_join(bt_seat1_df)
    bt_cols = ["Auction"] + [agg_expr_col(s) for s in SEAT_RANGE]
    joined_df = join_deals_with_bt(sample_df, bt_prepared, bt_cols=bt_cols)
    
    # Rename joined Auction to avoid conflict with deal's bid
    if "Auction" in joined_df.columns and "Auction_right" not in joined_df.columns:
        # The join may have created Auction_right if there was a conflict
        pass  # Auction column from BT is what we want for Rules model
    
    # Metrics accumulators
    model_a_wins = 0
    model_b_wins = 0
    ties = 0
    total_imp_a = 0
    total_imp_b = 0
    imp_diffs: List[int] = []
    sum_dd_a = 0
    sum_dd_b = 0
    sample_deals: List[Dict[str, Any]] = []
    
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
    
    # Process each deal - now using joined data (no per-row BT lookups)
    for row in joined_df.iter_rows(named=True):
        deal_idx = row.get("_row_idx", 0)
        dealer = row.get("Dealer", "N")
        vul = row.get("Vul", "None")
        bid_str = row.get("_bid_str", "")
        par_score = row.get("ParScore")
        
        # BT auction from the join (for Rules model)
        bt_auction = row.get("Auction")  # From joined BT data
        
        # Get opener's HCP (seat 1)
        dir_map = _seat_direction_map(1)
        opener_dir = dir_map.get(dealer, "N")
        opener_hcp = row.get(f"HCP_{opener_dir}")
        hcp_label = _hcp_range_label(opener_hcp)
        if hcp_label not in by_hcp_range:
            by_hcp_range[hcp_label] = {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0}
        
        # Determine contracts and scores for each model
        # Model A
        if model_a == "Actual":
            contract_a = row.get("Contract", "")
            dd_score_a = row.get("DD_Score_Declarer")
            auction_a = bid_str
        else:  # Rules
            # Use pre-joined BT auction
            if bt_auction:
                contract_a = get_ai_contract(bt_auction, dealer)
                auction_a = bt_auction
                # Compute DD score for Rules contract on-the-fly
                dd_score_a = get_dd_score_for_auction(bt_auction, dealer, row)
            else:
                contract_a = None
                auction_a = None
                dd_score_a = None
        
        # Model B
        if model_b == "Actual":
            contract_b = row.get("Contract", "")
            dd_score_b = row.get("DD_Score_Declarer")
            auction_b = bid_str
        else:  # Rules
            # Use pre-joined BT auction
            if bt_auction:
                contract_b = get_ai_contract(bt_auction, dealer)
                auction_b = bt_auction
                # Compute DD score for Rules contract on-the-fly
                dd_score_b = get_dd_score_for_auction(bt_auction, dealer, row)
            else:
                contract_b = None
                auction_b = None
                dd_score_b = None
        
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

        # Collect a small sample of deal-level comparisons for UI
        if len(sample_deals) < 50:
            sample_deals.append(
                {
                    "PBN": row.get("PBN"),
                    "Dealer": dealer,
                    "Vul": vul,
                    f"Auction_{model_a}": auction_a,
                    f"Auction_{model_b}": auction_b,
                    f"DD_Score_{model_a}": score_a,
                    f"DD_Score_{model_b}": score_b,
                    "IMP_Diff": imp_signed,
                }
            )
        
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
    print(f"[bidding-arena] {elapsed_ms:.1f}ms ({contracts_compared} compared)")
    
    return {
        "model_a": model_a,
        "model_b": model_b,
        "deals_source": deals_source,
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
        "sample_deals": sample_deals,
        "elapsed_ms": round(elapsed_ms, 1),
    }

