"""
Common utilities, constants, and types for API handlers.

This module contains shared helper functions, constants, and the HandlerState
dataclass to eliminate primitive obsession and magic numbers/strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from bbo_bidding_queries_lib import normalize_auction_pattern, normalize_auction_input, normalize_auction_user_text
from mlBridgeLib.mlBridgeBiddingLib import DIRECTIONS

# ===========================================================================
# Constants (eliminates magic numbers/strings)
# ===========================================================================

SEAT_RANGE = range(1, 5)  # Seats 1-4
MAX_SAMPLE_SIZE = 10_000  # Default sample limit for performance
DEFAULT_SEED = 42

# Suit index mapping (Spades, Hearts, Diamonds, Clubs in PBN order)
SUIT_IDX: Dict[str, int] = {"S": 0, "H": 1, "D": 2, "C": 3}

# Directions in clockwise order starting from North
DIRECTIONS_LIST: List[str] = ["N", "E", "S", "W"]

def agg_expr_col(seat: int) -> str:
    """Get the Agg_Expr column name for a seat."""
    return f"Agg_Expr_Seat_{seat}"


def wrong_bid_col(seat: int) -> str:
    """Get the Wrong_Bid column name for a seat."""
    return f"Wrong_Bid_S{seat}"


def invalid_criteria_col(seat: int) -> str:
    """Get the Invalid_Criteria column name for a seat."""
    return f"Invalid_Criteria_S{seat}"


# ===========================================================================
# Typed State (eliminates primitive obsession)
# ===========================================================================

@dataclass
class HandlerState:
    """Typed container for handler state, replacing Dict[str, Any].
    
    This provides type safety and IDE autocompletion for the state object
    passed to all handlers.
    """
    deal_df: pl.DataFrame
    bt_seat1_df: Optional[pl.DataFrame] = None
    bt_stats_df: Optional[pl.DataFrame] = None
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, pl.DataFrame]] = field(default_factory=dict)
    duckdb_conn: Any = None  # DuckDB connection
    
    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "HandlerState":
        """Create HandlerState from a legacy dict."""
        return cls(
            deal_df=state["deal_df"],
            bt_seat1_df=state.get("bt_seat1_df"),
            bt_stats_df=state.get("bt_stats_df"),
            deal_criteria_by_seat_dfs=state.get("deal_criteria_by_seat_dfs", {}),
            duckdb_conn=state.get("duckdb_conn"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict for backward compatibility."""
        return {
            "deal_df": self.deal_df,
            "bt_seat1_df": self.bt_seat1_df,
            "bt_stats_df": self.bt_stats_df,
            "deal_criteria_by_seat_dfs": self.deal_criteria_by_seat_dfs,
            "duckdb_conn": self.duckdb_conn,
        }


# ===========================================================================
# Auction Helpers
# ===========================================================================

def display_auction_with_seat_prefix(auction: Any, seat: Any) -> Any:
    """Display helper: prepend leading passes for the requested seat.
    
    We keep auctions canonicalized (no leading p-) for matching, but for UI display
    we want seat-relative prefixes:
      seat=1 -> ""
      seat=2 -> "p-"
      seat=3 -> "p-p-"
      seat=4 -> "p-p-p-"
    """
    if auction is None:
        return None
    try:
        s = str(auction)
    except Exception:
        return auction
    try:
        seat_i = int(seat)
    except Exception:
        seat_i = 1
    seat_i = max(1, min(4, seat_i))
    base = re.sub(r"^(p-)+", "", s)
    prefix = "p-" * (seat_i - 1)
    return prefix + base


def normalize_to_seat1(pattern: str) -> str:
    """Normalize auction regex then strip any leading 'p-' prefixes (seat-1 view)."""
    if not pattern or not pattern.strip():
        return pattern
    p = normalize_auction_user_text(pattern)
    # Strip leading p- (case-insensitive) after optional start anchor
    p = re.sub(r"(?i)^\^(p-)+", "^", p)
    p = re.sub(r"(?i)^(p-)+", "", p)
    return p


def expand_row_to_all_seats(row: Dict[str, Any], allow_initial_passes: bool) -> List[Dict[str, Any]]:
    """Expand a single bt_seat1 row to 4 seat variants (or 1 if allow_initial_passes=False).
    
    When allow_initial_passes is True, each matched auction is expanded to 4 rows
    representing the 4 possible dealer positions. The Agg_Expr_Seat_X columns are
    rotated to maintain correct seat semantics.
    """
    if not allow_initial_passes:
        return [row]
    
    auction = row.get("Auction", "")
    if not auction:
        return [row]
    
    # Strip any existing leading passes to get the base auction
    base_auction = re.sub(r"(?i)^(p-)+", "", str(auction))
    
    # Identify all seat-indexed column groups to rotate
    agg_expr_pattern = re.compile(r"^Agg_Expr_Seat_([1-4])$")
    suffix_s_pattern = re.compile(r"^(.+)_S([1-4])$")
    
    expanded_rows = []
    for num_passes in range(4):
        new_row = {}
        
        # Copy non-seat-specific columns
        for col_name, value in row.items():
            if agg_expr_pattern.match(col_name) or suffix_s_pattern.match(col_name):
                continue
            new_row[col_name] = value
        
        # Build the auction with appropriate prefix
        prefix = "p-" * num_passes
        new_row["Auction"] = prefix + base_auction
        new_row["_opener_seat"] = num_passes + 1
        
        # Rotate seat-indexed columns
        for display_seat in SEAT_RANGE:
            original_seat = ((display_seat - 1 - num_passes) % 4) + 1
            
            orig_agg = agg_expr_col(original_seat)
            display_agg = agg_expr_col(display_seat)
            if orig_agg in row:
                new_row[display_agg] = row[orig_agg]
            
            for col_name in row.keys():
                match = suffix_s_pattern.match(col_name)
                if match:
                    base_name = match.group(1)
                    col_seat = int(match.group(2))
                    if col_seat == original_seat:
                        display_col = f"{base_name}_S{display_seat}"
                        new_row[display_col] = row[col_name]
        
        expanded_rows.append(new_row)
    
    return expanded_rows


# ===========================================================================
# DataFrame Helpers
# ===========================================================================

def take_rows_by_index(df: pl.DataFrame, row_indices: List[int]) -> pl.DataFrame:
    """Version-tolerant row selection by integer indices (preserves order).
    
    Polars versions differ on whether `DataFrame.take()` exists. This helper works
    without it by joining on a generated row index.
    """
    if not row_indices:
        return df.head(0)
    idx_df = pl.DataFrame(
        {"_row": [int(i) for i in row_indices], "_pos": list(range(len(row_indices)))}
    )
    return (
        df.with_row_index("_row")
        .join(idx_df, on="_row", how="inner")
        .sort("_pos")
        .drop(["_row", "_pos"])
    )


def effective_seed(seed: int | None) -> int | None:
    """Convert seed=0 to None (non-reproducible) for Polars .sample()."""
    if seed is None:
        return None
    return None if seed == 0 else seed


def safe_float(x: Any) -> float | None:
    """Convert a value to float if possible; otherwise return None."""
    try:
        return float(x)
    except Exception:
        return None


# ===========================================================================
# Bid/Contract Helpers
# ===========================================================================

def bid_value_to_str(bid_val: Any) -> str:
    """Stringize deal_df['bid'] consistently across endpoints.

    - If it's a list of bids, join with '-' (e.g., ['1N','p','3N'] -> '1N-p-3N')
    - If it's already a string, return as-is
    - Else, best-effort str(...)
    """
    if bid_val is None:
        return ""
    if isinstance(bid_val, list):
        try:
            return "-".join(map(str, bid_val))
        except Exception:
            return "-".join([str(x) for x in bid_val])
    if isinstance(bid_val, str):
        return bid_val
    return str(bid_val)


def count_leading_passes(auction: Any) -> int:
    """Count leading passes in an auction.
    
    Args:
        auction: Auction as string ("p-p-1N-...") or list (["p", "p", "1N", ...])
    
    Returns:
        Number of leading passes (0-3)
    """
    if auction is None:
        return 0
    
    # Convert to list of bids
    if isinstance(auction, str):
        if not auction.strip():
            return 0
        # Normalize separators (space/comma/dash) to canonical dash format
        norm = normalize_auction_input(auction)
        bids = [b.strip().lower() for b in norm.split("-")] if norm else []
    elif isinstance(auction, list):
        bids = [str(b).strip().lower() for b in auction]
    else:
        return 0
    
    count = 0
    for bid in bids:
        if bid in ("p", "pass"):
            count += 1
        else:
            break
    return min(count, 3)  # Max 3 leading passes before all-pass


def auction_matches_opening_seat(actual_auction: Any, expected_passes: int) -> bool:
    """Check if a deal's actual auction starts with the expected number of passes.
    
    Args:
        actual_auction: The deal's actual auction (string or list)
        expected_passes: Expected number of leading passes (0-3)
    
    Returns:
        True if the actual auction starts with exactly the expected number of passes
    """
    actual_passes = count_leading_passes(actual_auction)
    return actual_passes == expected_passes


def extract_bid_at_seat(auction: str, seat: int) -> str | None:
    """Extract the bid at a specific seat position from an auction string.
    
    Args:
        auction: Auction string like "1N-p-3N-p-p-p"
        seat: 1-based seat position (1-4)
    
    Returns:
        The bid at that seat, or None if not found
    """
    if not auction:
        return None
    bids = auction.split("-")
    seat_idx = seat - 1
    if seat_idx < 0 or seat_idx >= len(bids):
        return None
    return bids[seat_idx]


def seat_direction_map(seat: int) -> Dict[str, str]:
    """Map dealer -> actual hand direction for a given seat (1–4).
    
    Seat 1 is the dealer; seat 2 is LHO; seat 3 is partner; seat 4 is RHO.
    """
    seat_i = max(1, min(4, int(seat)))
    mapping: Dict[str, str] = {}
    for dealer in DIRECTIONS:
        dealer_idx = DIRECTIONS.index(dealer)
        direction = DIRECTIONS[(dealer_idx + seat_i - 1) % 4]
        mapping[dealer] = direction
    return mapping


# ===========================================================================
# Par Contract Helpers
# ===========================================================================

def par_contract_signature(c: dict) -> str:
    """Stable signature for a par-contract dict (used for de-duping)."""
    level = c.get("Level", "")
    strain = c.get("Strain", "")
    dbl = c.get("Doubled", "")
    if dbl == "":
        dbl = c.get("Double", "")
    pair_dir = c.get("Pair_Direction", "")
    result = c.get("Result", "")
    return f"{level}|{strain}|{dbl}|{pair_dir}|{result}"


def dedup_par_contracts(par_contracts: Any) -> List[dict]:
    """Return de-duplicated par contracts (preserving first-seen order)."""
    if not isinstance(par_contracts, list):
        return []
    seen: set[str] = set()
    out: List[dict] = []
    for c in par_contracts:
        if not isinstance(c, dict):
            continue
        sig = par_contract_signature(c)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def format_par_contracts(par_contracts: Any) -> str | None:
    """Format ParContracts into a readable string, de-duped and with correct 'Doubled' key."""
    if par_contracts is None:
        return None
    if not isinstance(par_contracts, list):
        return str(par_contracts)
    formatted: List[str] = []
    for c in dedup_par_contracts(par_contracts):
        level = c.get("Level", "")
        strain = c.get("Strain", "")
        dbl = c.get("Doubled", "")
        if dbl == "":
            dbl = c.get("Double", "")
        pair_dir = c.get("Pair_Direction", "")
        result = c.get("Result", "")
        contract_str = f"{level}{strain}{dbl}"
        if pair_dir:
            contract_str += f" {pair_dir}"
        if result is not None and result != "":
            if isinstance(result, int):
                if result == 0:
                    contract_str += " ="
                elif result > 0:
                    contract_str += f" +{result}"
                else:
                    contract_str += f" {result}"
            else:
                contract_str += f" {result}"
        formatted.append(contract_str.strip())
    return ", ".join(formatted)


def ev_list_for_par_contracts(deal_row: dict) -> List[float | None]:
    """EV values aligned to the de-duplicated ParContracts order."""
    par_contracts = dedup_par_contracts(deal_row.get("ParContracts"))
    if not par_contracts:
        return []

    def _vul_flag(vul_raw: Any, pair: str) -> str:
        vul_raw_s = "None" if vul_raw is None else str(vul_raw).strip()
        if vul_raw_s in ["Both", "All", "b", "B"]:
            return "V"
        if vul_raw_s == pair or (vul_raw_s == "NS" and pair == "NS") or (vul_raw_s == "EW" and pair == "EW"):
            return "V"
        if vul_raw_s in ["None", "O", "o", "-", ""]:
            return "NV"
        return "NV"

    out: List[float | None] = []
    for c in par_contracts:
        level_raw = c.get("Level")
        strain_raw = c.get("Strain")
        pair = c.get("Pair_Direction")
        if not level_raw or not strain_raw or pair not in ("NS", "EW"):
            out.append(None)
            continue
        try:
            level = int(level_raw)
        except Exception:
            out.append(None)
            continue
        strain = str(strain_raw).strip().upper()
        if strain not in ["C", "D", "H", "S", "N"]:
            out.append(None)
            continue
        vul = _vul_flag(deal_row.get("Vul", "None"), str(pair))
        decls = ["N", "S"] if pair == "NS" else ["E", "W"]
        best_for_contract: float | None = None
        for declarer in decls:
            col = f"EV_{pair}_{declarer}_{strain}_{level}_{vul}"
            ev_f = safe_float(deal_row.get(col))
            if ev_f is None:
                continue
            if best_for_contract is None or ev_f > best_for_contract:
                best_for_contract = ev_f
        out.append(best_for_contract)
    return out


# ===========================================================================
# BT Lookup Helper (extracted from duplicated code pattern)
# ===========================================================================

def lookup_bt_row(
    bt_lookup_df: pl.DataFrame,
    bid_str: str,
) -> Tuple[pl.DataFrame, str]:
    """Look up bidding table row with pass-suffix fallback.
    
    This extracts the duplicated pattern that appeared 6 times in the codebase.
    
    Args:
        bt_lookup_df: Bidding table DataFrame (filtered to completed auctions)
        bid_str: Auction string to look up
    
    Returns:
        Tuple of (matched_df, normalized_auction_string)
    """
    # Canonicalize auction tokens (e.g. "1nt" -> "1N", "pass" -> "p"), then strip leading passes (seat-1 view).
    # IMPORTANT: This is used for *literal* auction lookups, not regex matching.
    auction_norm = normalize_auction_input(bid_str) if bid_str else ""
    auction_for_search = re.sub(r"(?i)^(p-)+", "", auction_norm).lower() if auction_norm else ""

    # Try common variants with/without trailing "-p-p-p" because BT rows aren't always consistent
    # across endpoints (some use completed-auction rows, some use prefix rows).
    candidates: list[str] = []
    if auction_for_search:
        candidates.append(auction_for_search)
        if auction_for_search.endswith("-p-p-p"):
            candidates.append(auction_for_search[: -len("-p-p-p")])
        else:
            candidates.append(auction_for_search + "-p-p-p")

    if not candidates:
        return bt_lookup_df.head(0), auction_for_search

    auc_col = pl.col("Auction").cast(pl.Utf8).str.to_lowercase()
    bt_match = bt_lookup_df.filter(auc_col.is_in(candidates))
    return bt_match, auction_for_search


def get_bt_info_from_match(bt_match: pl.DataFrame) -> Dict[str, Any] | None:
    """Extract bt_info dict from a bt_match DataFrame.
    
    Returns None if no match found.
    """
    if bt_match.height == 0:
        return None
    
    bt_row = bt_match.row(0, named=True)
    return {
        agg_expr_col(seat): bt_row.get(agg_expr_col(seat))
        for seat in SEAT_RANGE
    }


# ===========================================================================
# Vectorized Deal-to-BT Join Helpers (performance optimization)
# ===========================================================================

def prepare_deals_with_bid_str(deal_df: pl.DataFrame) -> pl.DataFrame:
    """Add _bid_str and _auction_key columns to deal DataFrame for joining.
    
    _auction_key is normalized: lowercase, leading passes stripped, trailing -p-p-p removed.
    """
    bid_dtype = deal_df.schema.get("bid")
    
    # Add bid_str column
    if bid_dtype == pl.List(pl.Utf8):
        df = deal_df.with_columns(pl.col("bid").list.join("-").alias("_bid_str"))
    elif bid_dtype == pl.Utf8:
        df = deal_df.with_columns(pl.col("bid").fill_null("").alias("_bid_str"))
    else:
        df = deal_df.with_columns(
            pl.col("bid").map_elements(
                lambda x: "-".join(map(str, x)) if isinstance(x, list) else (str(x) if x is not None else ""),
                return_dtype=pl.Utf8,
            ).alias("_bid_str")
        )
    
    # Add normalized auction key for joining
    df = df.with_columns(
        pl.col("_bid_str")
        .str.to_lowercase()
        .str.replace(r"^(p-)+", "")  # Strip leading passes
        .str.replace(r"-p-p-p$", "")  # Strip trailing passes for matching
        .alias("_auction_key")
    )
    
    return df


def prepare_bt_for_join(bt_df: pl.DataFrame) -> pl.DataFrame:
    """Add _auction_key column to BT DataFrame for joining.
    
    Filters to completed auctions only and normalizes the auction key.
    """
    if "is_completed_auction" in bt_df.columns:
        bt_df = bt_df.filter(pl.col("is_completed_auction"))
    
    # Add normalized key (BT auctions are already seat-1 normalized, just need lowercase + strip trailing passes)
    return bt_df.with_columns(
        pl.col("Auction")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace(r"-p-p-p$", "")  # Strip trailing passes for matching
        .alias("_auction_key")
    )


def join_deals_with_bt(
    deals_df: pl.DataFrame,
    bt_df: pl.DataFrame,
    bt_cols: List[str] | None = None,
) -> pl.DataFrame:
    """Join deals with BT data using pre-computed auction keys.
    
    Both DataFrames must have _auction_key column (use prepare_* functions).
    
    Args:
        deals_df: Deal DataFrame with _auction_key column
        bt_df: BT DataFrame with _auction_key column  
        bt_cols: Columns to select from BT (defaults to Agg_Expr_Seat_1-4)
    
    Returns:
        Joined DataFrame with BT columns added
    """
    if bt_cols is None:
        bt_cols = [agg_expr_col(s) for s in SEAT_RANGE]
    
    # Select only needed columns from BT to avoid column conflicts
    bt_select = ["_auction_key"] + [c for c in bt_cols if c in bt_df.columns]
    bt_slim = bt_df.select(bt_select).unique(subset=["_auction_key"])
    
    return deals_df.join(bt_slim, on="_auction_key", how="left")


def batch_check_wrong_bids(
    joined_df: pl.DataFrame,
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, Any]],
    seat_filter: int | None = None,
    criteria_overlay: list[dict[str, Any]] | None = None,
) -> pl.DataFrame:
    """Batch-check wrong bids for a joined deals+BT DataFrame.
    
    This is more efficient than row-by-row checking when processing many deals.
    Still uses row iteration for bitmap checks (criteria vary per auction),
    but eliminates the expensive per-row BT lookups.
    
    Args:
        joined_df: DataFrame with _row_idx, Dealer, _bid_str, and Agg_Expr_Seat_1-4 columns
        deal_criteria_by_seat_dfs: Pre-computed criteria bitmap DataFrames
        seat_filter: Optional seat to check (None = all seats)
    
    Returns:
        DataFrame with Wrong_Bid_S1-4, first_wrong_seat columns added
    """
    # Prepare result columns
    wrong_cols = {wrong_bid_col(s): [] for s in SEAT_RANGE}
    first_wrong_list: List[int | None] = []
    
    seats_to_check = [seat_filter] if seat_filter else list(SEAT_RANGE)

    from plugins.custom_criteria_overlay import apply_custom_criteria_overlay_to_bt_row as _apply_overlay
    
    # Process rows - still a loop but without per-row DataFrame filters
    for row in joined_df.iter_rows(named=True):
        if criteria_overlay:
            row = _apply_overlay(dict(row), criteria_overlay)
        deal_idx = row.get("_row_idx", 0)
        dealer = row.get("Dealer", "N")
        
        first_wrong_seat: int | None = None
        row_wrong = {s: False for s in SEAT_RANGE}
        
        for seat in seats_to_check:
            if seat is None:
                continue
            criteria_list = row.get(agg_expr_col(seat)) or []
            if not criteria_list:
                continue
            
            seat_dfs = deal_criteria_by_seat_dfs.get(seat, {})
            criteria_df = seat_dfs.get(dealer)
            if criteria_df is None or criteria_df.is_empty():
                continue
            
            for criterion in criteria_list:
                if criterion not in criteria_df.columns:
                    continue
                try:
                    bitmap_value = criteria_df[criterion][deal_idx]
                    if not bitmap_value:
                        row_wrong[seat] = True
                        if first_wrong_seat is None:
                            first_wrong_seat = seat
                        break  # One failed criterion is enough
                except (IndexError, KeyError):
                    continue
        
        for s in SEAT_RANGE:
            wrong_cols[wrong_bid_col(s)].append(row_wrong[s])
        first_wrong_list.append(first_wrong_seat)
    
    # Add columns to DataFrame
    result = joined_df.with_columns([
        pl.Series(name=wrong_bid_col(s), values=wrong_cols[wrong_bid_col(s)])
        for s in SEAT_RANGE
    ] + [
        pl.Series(name="first_wrong_seat", values=first_wrong_list)
    ])
    
    return result


# ===========================================================================
# Wrong Bid Conformance Checking
# ===========================================================================

def check_deal_criteria_conformance_bitmap(
    deal_idx: int,
    bt_info: Dict[str, Any] | None,
    dealer: str,
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, Any]],
    auction: str | None = None,
) -> Dict[str, Any]:
    """Check if a deal conforms to the criteria for each seat using bitmap lookups.
    
    Args:
        deal_idx: Row index in deal_df
        bt_info: Dict with Agg_Expr_Seat_1..4 keys
        dealer: Dealer direction (N/E/S/W)
        deal_criteria_by_seat_dfs: Pre-computed criteria bitmap DataFrames
        auction: Optional auction string for including bid in failed criteria
    
    Returns:
        Dict with Wrong_Bid_S1-S4, Invalid_Criteria_S1-S4, and first_wrong_seat
    """
    import json
    
    result: Dict[str, Any] = {
        wrong_bid_col(s): False for s in SEAT_RANGE
    }
    result.update({
        invalid_criteria_col(s): None for s in SEAT_RANGE
    })
    result["first_wrong_seat"] = None
    
    if bt_info is None:
        return result
    
    for seat in SEAT_RANGE:
        criteria_list = bt_info.get(agg_expr_col(seat))
        if not criteria_list:
            continue
        
        seat_dfs = deal_criteria_by_seat_dfs.get(seat, {})
        criteria_df = seat_dfs.get(dealer)
        if criteria_df is None or criteria_df.is_empty():
            continue
        
        failed_criteria: List[str] = []
        for criterion in criteria_list:
            if criterion not in criteria_df.columns:
                continue
            try:
                bitmap_value = criteria_df[criterion][deal_idx]
                if not bitmap_value:
                    failed_criteria.append(criterion)
            except (IndexError, KeyError):
                continue
        
        if failed_criteria:
            result[wrong_bid_col(seat)] = True
            bid_at_seat = extract_bid_at_seat(auction, seat) if auction else None
            if bid_at_seat:
                result[invalid_criteria_col(seat)] = json.dumps([bid_at_seat, failed_criteria])
            else:
                result[invalid_criteria_col(seat)] = json.dumps(failed_criteria)
            
            if result["first_wrong_seat"] is None:
                result["first_wrong_seat"] = seat
    
    return result


# ===========================================================================
# Suit Length (SL) Evaluation Helpers
# ===========================================================================
# These are used for dynamic evaluation of SL criteria (e.g., SL_S >= 5)
# to ensure correct seat-direction mapping.

def seat_to_direction(dealer: str, seat: int) -> str:
    """Convert seat number (1-4) to direction (N/E/S/W) given the dealer.
    
    Seat 1 is always the dealer. Seats rotate clockwise.
    
    Args:
        dealer: The dealer direction ('N', 'E', 'S', 'W')
        seat: The seat number (1-4)
    
    Returns:
        The direction for that seat
    """
    try:
        dealer_i = DIRECTIONS_LIST.index(dealer.upper())
    except ValueError:
        dealer_i = 0
    seat_i = max(1, min(4, int(seat)))
    return DIRECTIONS_LIST[(dealer_i + seat_i - 1) % 4]


def hand_suit_length(deal_row: Dict[str, Any], direction: str, suit: str) -> Optional[int]:
    """Get the length of a suit from a deal row's hand.
    
    Args:
        deal_row: Dict containing Hand_N, Hand_E, Hand_S, Hand_W keys
        direction: The direction ('N', 'E', 'S', 'W')
        suit: The suit ('S', 'H', 'D', 'C')
    
    Returns:
        The suit length, or None if the hand can't be parsed
    """
    hand = deal_row.get(f"Hand_{direction}")
    if hand is None:
        return None
    suits = str(hand).split(".")
    if len(suits) != 4:
        return None
    idx = SUIT_IDX.get(suit.upper())
    if idx is None:
        return None
    try:
        return len(suits[idx])
    except (IndexError, TypeError):
        return None


def parse_sl_comparison_relative(criterion: str) -> Optional[Tuple[str, str, str]]:
    """Parse suit-to-suit comparison like 'SL_S >= SL_H'.
    
    Args:
        criterion: The criterion string
    
    Returns:
        Tuple of (left_suit, operator, right_suit) or None if not matched
    """
    c = str(criterion).strip().replace("≥", ">=").replace("≤", "<=")
    m = re.match(r"^SL_([SHDC])\s*(>=|<=|>|<|==|!=)\s*SL_([SHDC])$", c)
    if m is None:
        return None
    return m.group(1), m.group(2), m.group(3)


def parse_sl_comparison_numeric(criterion: str) -> Optional[Tuple[str, str, int]]:
    """Parse suit-to-number comparison like 'SL_S >= 5'.
    
    Args:
        criterion: The criterion string
    
    Returns:
        Tuple of (suit, operator, number) or None if not matched
    """
    c = str(criterion).strip().replace("≥", ">=").replace("≤", "<=")
    m = re.match(r"^SL_([SHDC])\s*(>=|<=|>|<|==|!=)\s*(\d+)$", c)
    if m is None:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def eval_comparison(left: int, op: str, right: int) -> bool:
    """Evaluate a comparison between two integers.
    
    Args:
        left: Left operand
        op: Operator ('>=', '<=', '>', '<', '==', '!=')
        right: Right operand
    
    Returns:
        The result of the comparison
    """
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    return False


def evaluate_sl_criterion(
    criterion: str,
    dealer: str,
    seat: int,
    deal_row: Dict[str, Any],
    fail_on_missing: bool = True,
) -> Optional[bool]:
    """Evaluate a suit-length criterion dynamically.
    
    Args:
        criterion: The criterion string (e.g., 'SL_S >= 5' or 'SL_S >= SL_H')
        dealer: The dealer direction
        seat: The seat number (1-4)
        deal_row: Dict containing hand data
        fail_on_missing: If True, return False when hand data is missing.
                        If False, return None (treat as untracked).
    
    Returns:
        True if criterion passes, False if fails (or can't evaluate when fail_on_missing=True), 
        None if not an SL criterion OR can't evaluate when fail_on_missing=False
    """
    direction = seat_to_direction(dealer, seat)
    
    # Try suit-to-suit comparison (e.g., SL_S >= SL_H)
    parsed_rel = parse_sl_comparison_relative(criterion)
    if parsed_rel is not None:
        left_s, op, right_s = parsed_rel
        lv = hand_suit_length(deal_row, direction, left_s)
        rv = hand_suit_length(deal_row, direction, right_s)
        if lv is None or rv is None:
            return False if fail_on_missing else None
        return eval_comparison(lv, op, rv)
    
    # Try suit-to-number comparison (e.g., SL_S >= 5)
    parsed_num = parse_sl_comparison_numeric(criterion)
    if parsed_num is not None:
        suit, op, num_val = parsed_num
        lv = hand_suit_length(deal_row, direction, suit)
        if lv is None:
            return False if fail_on_missing else None
        return eval_comparison(lv, op, num_val)
    
    # Not an SL criterion - fall through to bitmap lookup
    return None

