"""
Common utilities, constants, and types for API handlers.

This module contains shared helper functions, constants, and the HandlerState
dataclass to eliminate primitive obsession and magic numbers/strings.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

from bbo_bidding_queries_lib import normalize_auction_pattern, normalize_auction_input, normalize_auction_user_text, pattern_matches
from mlBridgeLib.mlBridgeBiddingLib import DIRECTIONS


# ===========================================================================
# On-Demand Agg_Expr Loading (Memory Optimization)
# ===========================================================================

def load_agg_expr_for_bt_indices(
    bt_indices: List[int],
    bt_parquet_file: Union[pathlib.Path, str],
) -> Dict[int, Dict[str, List[str]]]:
    """Load Agg_Expr columns for specific bt_indices from the Parquet file.
    
    This is a memory-efficient way to get criteria data for specific rows
    without loading the heavy Agg_Expr columns (100+ GB) for all 461M rows.
    
    Uses DuckDB for efficient lookup (row group pruning, bloom filters) instead of
    scanning all 461M rows with Polars.
    
    Args:
        bt_indices: List of bt_index values to load
        bt_parquet_file: Path to the compiled BT parquet file
    
    Returns:
        Dict mapping bt_index -> {col_name: list of criteria strings}
        e.g., {12345: {"Agg_Expr_Seat_1": ["HCP >= 10", "SL_S >= 4"], ...}}
    """
    if not bt_indices:
        return {}

    # Safety rail: prevent accidental "load Agg_Expr for the whole BT" behavior.
    # Typical requests should load O(1..10_000) indices. Larger loads are almost certainly a bug
    # and can reintroduce pagefile thrashing.
    _MAX_ON_DEMAND_BT_INDICES = 250_000
    uniq = sorted({int(x) for x in bt_indices if x is not None})
    if len(uniq) > _MAX_ON_DEMAND_BT_INDICES:
        raise ValueError(
            f"Refusing to load Agg_Expr for {len(uniq):,} bt_indices (limit={_MAX_ON_DEMAND_BT_INDICES:,}). "
            "This would be extremely slow and memory-heavy. Load smaller subsets or implement caching/batching."
        )
    
    # Use DuckDB for efficient lookup - it can use Parquet row group pruning and bloom filters
    # which is MUCH faster than Polars' full scan for small IN lists.
    #
    # NOTE: We intentionally do NOT provide a Polars fallback here; if DuckDB is unavailable,
    # we'd rather fail loudly than silently reintroduce multi-second timeouts.
    import duckdb

    # Create a fresh connection for thread safety
    conn = duckdb.connect(":memory:")

    # Build the IN list as comma-separated values
    in_list = ", ".join(str(x) for x in uniq)

    # Escape backslashes in the file path for SQL
    file_path = str(bt_parquet_file).replace("\\", "/")

    query = f"""
        SELECT bt_index, Agg_Expr_Seat_1, Agg_Expr_Seat_2, Agg_Expr_Seat_3, Agg_Expr_Seat_4
        FROM read_parquet('{file_path}')
        WHERE bt_index IN ({in_list})
    """

    try:
        result_rel = conn.execute(query)
        rows = result_rel.fetchall()
        col_names = [desc[0] for desc in result_rel.description]
    finally:
        conn.close()

    result: Dict[int, Dict[str, List[str]]] = {}
    for row in rows:
        row_dict = dict(zip(col_names, row))
        bt_idx = row_dict["bt_index"]
        result[bt_idx] = {}
        for seat in range(1, 5):
            col = f"Agg_Expr_Seat_{seat}"
            if col in row_dict:
                val = row_dict[col]
                # DuckDB may return list or None
                result[bt_idx][col] = list(val) if val else []

    return result


def enrich_bt_row_with_agg_expr(
    bt_row: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Enrich a BT row with Agg_Expr columns loaded on-demand from Parquet.
    
    If the row already has Agg_Expr columns, returns as-is.
    Otherwise, loads them from the Parquet file using bt_index.
    
    Args:
        bt_row: Row dict from bt_seat1_df (may lack Agg_Expr columns)
        state: API state dict containing bt_seat1_file
    
    Returns:
        Row dict with Agg_Expr columns populated
    """
    # Check if already has Agg_Expr data
    if "Agg_Expr_Seat_1" in bt_row and bt_row.get("Agg_Expr_Seat_1") is not None:
        return bt_row
    
    # Get bt_index for lookup
    bt_index = bt_row.get("bt_index")
    if bt_index is None:
        return bt_row
    
    # Get parquet file path from state
    bt_parquet_file = state.get("bt_seat1_file")
    if bt_parquet_file is None:
        raise RuntimeError(
            "Agg_Expr columns are not loaded in-memory, but state['bt_seat1_file'] is missing. "
            "This is a wiring bug: set STATE['bt_seat1_file'] during initialization."
        )
    
    # Load Agg_Expr for this bt_index
    agg_data = load_agg_expr_for_bt_indices([int(bt_index)], bt_parquet_file)
    if bt_index in agg_data:
        bt_row = dict(bt_row)  # Copy to avoid mutation
        bt_row.update(agg_data[bt_index])
    
    return bt_row

# ===========================================================================
# Criteria Deduplication (keep least restrictive bounds)
# ===========================================================================

# Regex to parse numeric inequality criteria: e.g., "HCP >= 10", "SL_S <= 5"
_INEQ_PATTERN = re.compile(r'^(\w+)\s*(>=|<=|>|<|==)\s*(-?\d+)$')


def dedupe_criteria_least_restrictive(criteria: List[str]) -> List[str]:
    """Deduplicate criteria list, keeping least restrictive bounds for each variable.
    
    When multiple criteria refer to the same variable with the same operator:
    - For >= or >: keep the SMALLEST value (allows more hands through)
    - For <= or <: keep the LARGEST value (allows more hands through)
    - For ==: keep first occurrence (exact match, can't merge)
    
    Non-numeric (boolean) criteria are kept as-is.
    
    Examples:
        ["HCP >= 3", "HCP >= 5", "HCP <= 10"]  -> ["HCP >= 3", "HCP <= 10"]
        ["SL_S >= 4", "SL_S >= 2", "Balanced"] -> ["SL_S >= 2", "Balanced"]
    """
    if not criteria:
        return []
    
    # Track inequalities: (var_name, operator) -> best_value
    inequalities: Dict[Tuple[str, str], int] = {}
    # Track non-numeric criteria (preserve order)
    other: List[str] = []
    # Track original inequality strings to preserve spacing/formatting
    ineq_format: Dict[Tuple[str, str], str] = {}
    
    for crit in criteria:
        crit_str = str(crit).strip()
        match = _INEQ_PATTERN.match(crit_str)
        if match:
            var_name, op, value_str = match.groups()
            value = int(value_str)
            key = (var_name, op)
            
            if key not in inequalities:
                inequalities[key] = value
                ineq_format[key] = f"{var_name} {op} {value}"
            else:
                existing = inequalities[key]
                # For lower bounds (>=, >): keep smallest (least restrictive)
                if op in ('>=', '>'):
                    if value < existing:
                        inequalities[key] = value
                        ineq_format[key] = f"{var_name} {op} {value}"
                # For upper bounds (<=, <): keep largest (least restrictive)
                elif op in ('<=', '<'):
                    if value > existing:
                        inequalities[key] = value
                        ineq_format[key] = f"{var_name} {op} {value}"
                # For ==: first wins (can't merge different equality values)
        else:
            # Non-numeric criterion - add if not already present
            if crit_str not in other:
                other.append(crit_str)
    
    # Combine: inequalities first (sorted for consistency), then other
    result = sorted(ineq_format.values()) + other
    return result


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

# Model name constants (for rules matching)
MODEL_RULES = "Rules"  # Full pipeline: compiled BT + CSV overlay
MODEL_RULES_BASE = "Rules_Base"  # Compiled BT only (base + learned pre-merged)
MODEL_RULES_LEARNED = "Rules_Learned"  # Same as Rules_Base (backwards compat)
MODEL_ACTUAL = "Actual"  # Use actual auction from deal

# Models that use pre-compiled BT without overlay
MODELS_NO_OVERLAY = frozenset({MODEL_RULES_BASE, MODEL_RULES_LEARNED})

# NOTE (backwards compatibility):
# - MODEL_RULES_LEARNED is retained for UI/API compatibility, but is now equivalent to MODEL_RULES_BASE
#   because learned rules are pre-compiled into `bbo_bt_compiled.parquet`.

# ===========================================================================
# Canonical Auction Casing (single source of truth)
# ===========================================================================

def normalize_auction_case(auction: str) -> str:
    """Normalize auction string to canonical UPPERCASE form.
    
    This is the single source of truth for auction casing throughout the codebase.
    All auction strings for display/comparison should use this function.
    
    Examples:
        normalize_auction_case("1n-p-2c") -> "1N-P-2C"
        normalize_auction_case("p-p-1h-d-r") -> "P-P-1H-D-R"
    """
    return auction.upper() if auction else ""


def format_elapsed(ms: float) -> str:
    """Format elapsed time in seconds (e.g., 5380.4ms -> '5.38s').
    
    This is the single source of truth for elapsed time display.
    """
    return f"{ms / 1000:.2f}s"


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
# Auction Rule Application (Learned + Custom)
# ===========================================================================

def normalize_auction_for_overlay(auction: str) -> str:
    """Normalize auction for overlay prefix matching: UPPERCASE and strip leading passes.
    
    This uses the canonical uppercase form for consistency across the codebase.
    """
    a = (auction or "").strip().upper()
    # Strip leading 'P-' prefixes (seat-1 view matching)
    while a.startswith("P-"):
        a = a[2:]
    return a


def _ensure_list_criteria(val: Any) -> list[str]:
    """Ensure criteria is a list of strings. 
    Handles both pl.List(pl.Utf8) and pipe-separated strings (memory optimization).
    """
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x]
    if isinstance(val, str):
        if not val:
            return []
        return [x for x in val.split("|") if x]
    return []


def dedupe_criteria_all_seats(bt_row: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate criteria for all seats in a BT row.
    
    Modifies the row in place and returns it.
    Uses least-restrictive bounds when deduplicating.
    """
    for seat in SEAT_RANGE:
        col = f"Agg_Expr_Seat_{seat}"
        if col in bt_row:
            # Handle both list and pipe-separated categorical string
            val = bt_row.get(col)
            lst = _ensure_list_criteria(val)
            if lst:
                bt_row[col] = dedupe_criteria_least_restrictive(lst)
            else:
                bt_row[col] = []
    return bt_row


def apply_custom_criteria_overlay_to_bt_row(
    bt_row: dict[str, Any],
    overlay: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Apply overlay criteria rules to a BT row dict in-place (returns the dict).

    Overlay rule format: {"partial": str, "seat": int, "criteria": list[str]}

    This helper is intentionally lightweight and safe:
    - If overlay is empty, returns the row unchanged.
    - If Auction is missing, returns unchanged.
    - De-dupes criteria while preserving order.
    """
    if not overlay:
        # Still need to ensure lists if they were pipe-separated
        for seat in SEAT_RANGE:
            col = f"Agg_Expr_Seat_{seat}"
            if col in bt_row:
                bt_row[col] = _ensure_list_criteria(bt_row[col])
        return bt_row

    auction = bt_row.get("Auction")
    if not auction:
        return bt_row

    auction_norm = normalize_auction_for_overlay(str(auction))
    if not auction_norm:
        return bt_row

    # Ensure existing criteria are lists before appending
    for seat in SEAT_RANGE:
        col = f"Agg_Expr_Seat_{seat}"
        if col in bt_row:
            bt_row[col] = _ensure_list_criteria(bt_row[col])

    for rule in overlay:
        partial = str(rule.get("partial") or "")
        if not partial:
            continue
        # Hybrid matching: literal prefix for simple patterns, regex for complex ones
        if not pattern_matches(partial, auction_norm):
            continue

        try:
            seat = int(rule.get("seat") or 0)
        except Exception:
            continue
        if seat < 1 or seat > 4:
            continue

        crit_to_add = rule.get("criteria") or []
        if not crit_to_add:
            continue

        col = f"Agg_Expr_Seat_{seat}"
        existing = bt_row.get(col) or []
        combined = list(existing)
        for c in crit_to_add:
            if c not in combined:
                combined.append(c)
        bt_row[col] = combined

    return bt_row


def apply_overlay_and_dedupe(
    bt_row: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Apply CSV overlay to a BT row and deduplicate criteria.
    
    Note: Merged rules are pre-compiled into bbo_bt_compiled.parquet,
    so this function only applies the CSV overlay on top.
    
    If the row lacks Agg_Expr columns (memory optimization), they are
    loaded on-demand from the Parquet file using bt_index.
    
    Args:
        bt_row: BT row dict (will be copied, not mutated)
        state: API state dict containing custom_criteria_overlay and bt_seat1_file
    
    Returns:
        New dict with overlay applied and criteria deduplicated
    """
    result = dict(bt_row)
    # Enrich with Agg_Expr columns if missing (on-demand load from Parquet)
    result = enrich_bt_row_with_agg_expr(result, state)
    overlay = state.get("custom_criteria_overlay")
    result = apply_custom_criteria_overlay_to_bt_row(result, overlay)
    return dedupe_criteria_all_seats(result)


# Backwards compatibility alias
apply_all_rules_to_bt_row = apply_overlay_and_dedupe


def apply_rules_by_model(
    bt_row: dict[str, Any],
    state: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Apply rules based on model variant.
    
    Note: Merged rules are pre-compiled into bbo_bt_compiled.parquet.
    
    Model mapping:
    - MODEL_RULES_BASE / MODEL_RULES_LEARNED: Return row as-is (already compiled)
    - MODEL_RULES: Apply CSV overlay + deduplicate
    
    Args:
        bt_row: BT row dict (will be copied, not mutated)
        state: API state dict
        model_name: One of MODEL_RULES, MODEL_RULES_BASE, MODEL_RULES_LEARNED
    
    Returns:
        New dict with appropriate rules applied
    """
    if model_name in MODELS_NO_OVERLAY:
        return dict(bt_row)
    
    if model_name == MODEL_RULES:
        return apply_overlay_and_dedupe(bt_row, state)
    
    # Unknown model - return copy unchanged
    return dict(bt_row)


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
        auction: Auction as string ("P-P-1N-...") or list (["P", "P", "1N", ...])
    
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
        bids = [b.strip().upper() for b in norm.split("-")] if norm else []
    elif isinstance(auction, list):
        bids = [str(b).strip().upper() for b in auction]
    else:
        return 0
    
    count = 0
    for bid in bids:
        if bid in ("P", "PASS"):
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
    # Canonicalize auction tokens (e.g. "1nt" -> "1N", "pass" -> "P"), then strip leading passes (seat-1 view).
    # IMPORTANT: This is used for *literal* auction lookups, not regex matching.
    auction_norm = normalize_auction_input(bid_str) if bid_str else ""
    auction_for_search = re.sub(r"(?i)^(P-)+", "", auction_norm).upper() if auction_norm else ""
    
    # Try common variants with/without trailing "-P-P-P" because BT rows aren't always consistent
    # across endpoints (some use completed-auction rows, some use prefix rows).
    candidates: list[str] = []
    if auction_for_search:
        candidates.append(auction_for_search)
        if auction_for_search.endswith("-P-P-P"):
            candidates.append(auction_for_search[: -len("-P-P-P")])
        else:
            candidates.append(auction_for_search + "-P-P-P")

    if not candidates:
        return bt_lookup_df.head(0), auction_for_search

    auc_col = pl.col("Auction").cast(pl.Utf8).str.to_uppercase()
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
    
    _auction_key is normalized: UPPERCASE (canonical), leading passes stripped, trailing -P-P-P removed.
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
    
    # Add normalized auction key for joining (canonical UPPERCASE)
    df = df.with_columns(
        pl.col("_bid_str")
        .str.to_uppercase()
        .str.replace(r"^(P-)+", "")  # Strip leading passes
        .str.replace(r"-P-P-P$", "")  # Strip trailing passes for matching
        .alias("_auction_key")
    )
    
    return df


def prepare_bt_for_join(bt_df: pl.DataFrame) -> pl.DataFrame:
    """Add _auction_key column to BT DataFrame for joining.
    
    Filters to completed auctions only and normalizes the auction key (canonical UPPERCASE).
    """
    if "is_completed_auction" in bt_df.columns:
        bt_df = bt_df.filter(pl.col("is_completed_auction"))
    
    # Add normalized key (BT auctions are already seat-1 normalized, just need uppercase + strip trailing passes)
    return bt_df.with_columns(
        pl.col("Auction")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace(r"-P-P-P$", "")  # Strip trailing passes for matching
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
    # Always carry bt_index when present so callers can do on-demand enrichment later.
    bt_select = ["_auction_key"]
    if "bt_index" in bt_df.columns:
        bt_select.append("bt_index")
    bt_select += [c for c in bt_cols if c in bt_df.columns]
    bt_slim = bt_df.select(bt_select).unique(subset=["_auction_key"])
    
    return deals_df.join(bt_slim, on="_auction_key", how="left")


def join_deals_with_bt_on_demand(
    deals_df: pl.DataFrame,
    bt_df: pl.DataFrame,
    state: Dict[str, Any],
    bt_cols: List[str] | None = None,
) -> pl.DataFrame:
    """Join deals with BT data, ensuring Agg_Expr_Seat_* are available (loaded on-demand).

    This is required when the API runs with a lightweight in-memory `bt_seat1_df` that
    intentionally excludes Agg_Expr_Seat_1..4 to avoid 1TB pagefile thrashing.

    Strategy:
    - Join by `_auction_key` to get at least `bt_index`.
    - If Agg_Expr columns are missing, load them from the BT Parquet file for just the
      small set of bt_index values present in the joined result, and join back on bt_index.
    """
    if bt_cols is None:
        bt_cols = [agg_expr_col(s) for s in SEAT_RANGE]

    joined = join_deals_with_bt(deals_df, bt_df, bt_cols=bt_cols)

    needs_agg = any(c.startswith("Agg_Expr_Seat_") for c in bt_cols)
    has_any_agg = any((c in joined.columns) for c in bt_cols)
    if not needs_agg or has_any_agg:
        return joined

    if "bt_index" not in joined.columns:
        # Without bt_index we cannot load Agg_Expr on-demand safely.
        return joined

    bt_parquet_file = state.get("bt_seat1_file")
    if bt_parquet_file is None:
        raise RuntimeError(
            "join_deals_with_bt_on_demand requires state['bt_seat1_file'] to load Agg_Expr columns on-demand."
        )

    bt_indices = (
        joined
        .select(pl.col("bt_index").drop_nulls().cast(pl.Int64).unique())
        .to_series()
        .to_list()
    )
    bt_indices = [int(x) for x in bt_indices if x is not None]
    if not bt_indices:
        return joined

    # Load Agg_Expr columns for only the needed bt_index values and join back.
    cols = ["bt_index"] + [c for c in bt_cols if c.startswith("Agg_Expr_Seat_")]
    scan = pl.scan_parquet(str(bt_parquet_file))
    available = scan.collect_schema().names()
    cols_to_load = [c for c in cols if c in available]
    if len(cols_to_load) <= 1:
        return joined

    agg_df = (
        scan.filter(pl.col("bt_index").is_in(bt_indices))
        .select(cols_to_load)
        .collect()
    )

    return joined.join(agg_df, on="bt_index", how="left")


def batch_check_wrong_bids(
    joined_df: pl.DataFrame,
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, Any]],
    seat_filter: int | None = None,
    criteria_overlay: list[dict[str, Any]] | None = None,
    state: Dict[str, Any] | None = None,
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

    from plugins.bbo_bt_custom_criteria_overlay import apply_custom_criteria_overlay_to_bt_row as _apply_overlay
    
    # Process rows - still a loop but without per-row DataFrame filters
    for row in joined_df.iter_rows(named=True):
        if state is not None:
            # Canonical path: enrich (if needed) + overlay + dedupe
            row = apply_overlay_and_dedupe(dict(row), state)
        elif criteria_overlay:
            # Backwards-compat path
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


def format_seat_notation(
    dealer: str,
    seat: int,
    *,
    lead_passes: int = 0,
    include_bt_seat: bool = False,
) -> str:
    """Format a seat label in direction-first notation.

    Examples:
    - dealer='N', seat=1 -> 'N(S1)'
    - dealer='E', seat=2 -> 'S(S2)'
    - include_bt_seat=True, lead_passes=1 -> 'S(S2, BT_S1)'

    Notes:
    - `seat` is always dealer-relative (Seat 1 = dealer).
    - When a BT row has been rotated for leading passes, `lead_passes` lets us annotate
      which *BT seat* (seat-1/opener-relative) the dealer-relative seat corresponds to.
    """
    try:
        d = str(dealer or "N").upper()
    except Exception:
        d = "N"
    seat_i = max(1, min(4, int(seat)))
    try:
        dir_i = seat_to_direction(d, seat_i)
    except Exception:
        dir_i = "?"

    if include_bt_seat:
        try:
            lp = int(lead_passes or 0)
        except Exception:
            lp = 0
        bt_seat = ((seat_i - 1 - (lp % 4)) % 4) + 1 if lp else seat_i
        return f"{dir_i}(S{seat_i}, BT_S{bt_seat})"

    return f"{dir_i}(S{seat_i})"


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


def annotate_criterion_with_value(
    criterion: str,
    dealer: str,
    seat: int,
    deal_row: Dict[str, Any],
) -> str:
    """Annotate a criterion with the actual value of its left-hand side variable.
    
    Examples:
        'HCP <= 11' -> 'HCP(10) <= 11'
        'SL_S >= 5' -> 'SL_S(3) >= 5'
        'Total_Points >= 20' -> 'Total_Points(18) >= 20'
        'SL_S >= SL_H' -> 'SL_S(3) >= SL_H(4)'
    
    Args:
        criterion: The criterion string
        dealer: The dealer direction (N/E/S/W)
        seat: The seat number (1-4)
        deal_row: Dict containing hand data
    
    Returns:
        The criterion string with actual values annotated, or original if can't parse
    """
    direction = seat_to_direction(dealer, seat)
    crit_s = str(criterion).strip().replace("≥", ">=").replace("≤", "<=")
    
    # Try suit-to-suit comparison: SL_S >= SL_H
    parsed_rel = parse_sl_comparison_relative(crit_s)
    if parsed_rel is not None:
        left_suit, op, right_suit = parsed_rel
        lv = hand_suit_length(deal_row, direction, left_suit)
        rv = hand_suit_length(deal_row, direction, right_suit)
        lv_str = f"({lv})" if lv is not None else "(?)"
        rv_str = f"({rv})" if rv is not None else "(?)"
        return f"SL_{left_suit}{lv_str} {op} SL_{right_suit}{rv_str}"
    
    # Try suit-to-number comparison: SL_S >= 5
    parsed_num = parse_sl_comparison_numeric(crit_s)
    if parsed_num is not None:
        suit, op, num_val = parsed_num
        lv = hand_suit_length(deal_row, direction, suit)
        lv_str = f"({lv})" if lv is not None else "(?)"
        return f"SL_{suit}{lv_str} {op} {num_val}"
    
    # Try general variable comparison: VAR op NUMBER (e.g., HCP <= 11, Total_Points >= 20)
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<|==|!=)\s*(\d+)$", crit_s)
    if m:
        var_name = m.group(1)
        op = m.group(2)
        num_val = m.group(3)
        
        # Look up the variable value from deal_row
        actual_val = None
        var_upper = var_name.upper()
        
        if var_upper == "HCP":
            actual_val = deal_row.get(f"HCP_{direction}")
        elif var_upper == "TOTAL_POINTS":
            actual_val = deal_row.get(f"Total_Points_{direction}")
        else:
            # Try with direction suffix first, then without
            actual_val = deal_row.get(f"{var_name}_{direction}")
            if actual_val is None:
                actual_val = deal_row.get(var_name)
        
        if actual_val is not None:
            try:
                actual_val = int(actual_val)
                return f"{var_name}({actual_val}) {op} {num_val}"
            except (ValueError, TypeError):
                pass
    
    # Can't parse or annotate - return original
    return criterion


def evaluate_sl_criterion(
    criterion: str,
    dealer: str,
    seat: int,
    deal_row: Dict[str, Any],
    fail_on_missing: bool = True,
) -> Optional[bool]:
    """Evaluate a suit-length or complex criterion dynamically.
    
    Args:
        criterion: The criterion string. Supports:
            - Simple: 'SL_S >= 5', 'SL_S >= SL_H'
            - Complex: '(SL_D > SL_H | SL_D > SL_S)', 'SL_D >= SL_C & SL_D > SL_H'
        dealer: The dealer direction
        seat: The seat number (1-4)
        deal_row: Dict containing hand data
        fail_on_missing: If True, return False when hand data is missing.
                        If False, return None (treat as untracked).
    
    Returns:
        True if criterion passes, False if fails (or can't evaluate when fail_on_missing=True), 
        None if not an SL/complex criterion OR can't evaluate when fail_on_missing=False
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
    
    # Try complex expression with logical operators (e.g., SL_D > SL_H | SL_D > SL_S)
    if is_complex_expression(criterion):
        return evaluate_complex_expression(criterion, dealer, seat, deal_row, fail_on_missing)
    
    # Not an SL/complex criterion - fall through to bitmap lookup
    return None


# ===========================================================================
# Complex Criteria Expression Evaluator
# ===========================================================================
# Uses CriteriaEvaluator from mlBridgeBiddingLib for parsing complex expressions
# with logical operators (&, |, and, or, not) and parentheses.

# Lazy-loaded singleton instance of CriteriaEvaluator
_criteria_evaluator = None


def _get_criteria_evaluator():
    """Get or create the CriteriaEvaluator singleton."""
    global _criteria_evaluator
    if _criteria_evaluator is None:
        from mlBridgeLib.mlBridgeBiddingLib import CriteriaEvaluator
        _criteria_evaluator = CriteriaEvaluator()
    return _criteria_evaluator


def strip_criterion_comments(expr: str) -> str:
    """Strip inline comments from a criterion expression.
    
    A '#' character marks the beginning of a comment - everything from '#'
    to the end of the line is ignored.
    
    Args:
        expr: The expression string, possibly containing comments
        
    Returns:
        The expression with comments removed and whitespace stripped
    """
    evaluator = _get_criteria_evaluator()
    return evaluator.strip_comments(expr)


def is_complex_expression(expr: str) -> bool:
    """Check if an expression contains logical operators or parentheses.
    
    Complex expressions require the full CriteriaEvaluator; simple expressions
    can use the faster regex-based parsers.
    """
    # Check for logical operators or grouping parentheses
    return bool(re.search(r'[&|()]|\band\b|\bor\b|\bnot\b', expr, re.IGNORECASE))


def parse_complex_expression(expr: str) -> Tuple[Tuple[str, ...], List[str]]:
    """Parse a complex expression into postfix tokens.
    
    Args:
        expr: The expression string (e.g., "(SL_D > SL_H | SL_D > SL_S)")
        
    Returns:
        Tuple of (postfix_tokens, variables)
        - postfix_tokens: Tuple of tokens in postfix notation
        - variables: List of variable names found in the expression
    """
    evaluator = _get_criteria_evaluator()
    
    # Strip comments first
    expr = evaluator.strip_comments(expr)
    if not expr:
        return tuple(), []
    
    # Tokenize and convert to postfix
    tokens = re.findall(evaluator.token_pattern, expr)
    postfix = evaluator.infix_to_postfix(tokens)
    
    # Extract variable names (anything that matches identifier pattern and isn't an operator)
    ops = set(evaluator.ops.keys())
    variables = [t for t in postfix if re.match(r'^[a-zA-Z_]\w*$', t) and t not in ops]
    
    return postfix, variables


def evaluate_complex_expression(
    expr: str,
    dealer: str,
    seat: int,
    deal_row: Dict[str, Any],
    fail_on_missing: bool = True,
) -> Optional[bool]:
    """Evaluate a complex expression against a single deal's hand data.
    
    Supports logical operators (&, |, and, or, not) and parentheses.
    
    Args:
        expr: The expression string (e.g., "(SL_D > SL_H | SL_D > SL_S)")
        dealer: The dealer direction (N/E/S/W)
        seat: The seat number (1-4)
        deal_row: Dict containing hand data (Hand_N, Hand_E, Hand_S, Hand_W, HCP_N, etc.)
        fail_on_missing: If True, return False when hand data is missing.
                        If False, return None (treat as untracked).
    
    Returns:
        True if expression passes, False if fails, None if can't evaluate
    """
    evaluator = _get_criteria_evaluator()
    
    # Strip comments
    expr = evaluator.strip_comments(expr)
    if not expr:
        return True  # Empty expression = passes
    
    # Get direction for this seat
    direction = seat_to_direction(dealer, seat)
    
    # Tokenize and convert to postfix
    tokens = re.findall(evaluator.token_pattern, expr)
    if not tokens:
        return True  # No tokens = passes
    
    postfix = evaluator.infix_to_postfix(tokens)
    
    # Build variable values dict for this deal/direction
    var_values: Dict[str, Any] = {}
    missing_vars: List[str] = []
    
    for token in postfix:
        if token in evaluator.ops or token.isnumeric() or token in ('(', ')'):
            continue
        if not re.match(r'^[a-zA-Z_]\w*$', token):
            continue
        if token in var_values:
            continue
            
        # Resolve variable value
        val = _resolve_variable_value(token, direction, deal_row)
        if val is None:
            missing_vars.append(token)
        else:
            var_values[token] = val
    
    if missing_vars:
        return False if fail_on_missing else None
    
    # Evaluate postfix expression
    try:
        result = _evaluate_postfix_single(postfix, var_values, evaluator.ops)
        return bool(result)
    except Exception:
        return False if fail_on_missing else None


def _resolve_variable_value(
    var_name: str,
    direction: str,
    deal_row: Dict[str, Any],
) -> Optional[int]:
    """Resolve a variable name to its value for a given direction.
    
    Handles:
    - SL_S, SL_H, SL_D, SL_C -> suit length from hand
    - HCP -> high card points (from HCP_{direction} column)
    - Total_Points -> total points (from Total_Points_{direction} column)
    """
    var_upper = var_name.upper()
    
    # Suit length variables
    if var_upper.startswith("SL_") and len(var_upper) == 4:
        suit = var_upper[-1]
        if suit in SUIT_IDX:
            return hand_suit_length(deal_row, direction, suit)
    
    # HCP variable
    if var_upper == "HCP":
        hcp_col = f"HCP_{direction}"
        val = deal_row.get(hcp_col)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        return None
    
    # Total_Points variable
    if var_upper == "TOTAL_POINTS":
        tp_col = f"Total_Points_{direction}"
        val = deal_row.get(tp_col)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        return None
    
    # Try direct column lookup (for any other variable)
    # First try with direction suffix, then without
    for col in [f"{var_name}_{direction}", var_name]:
        val = deal_row.get(col)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    
    return None


def _evaluate_postfix_single(
    postfix: Tuple[str, ...],
    var_values: Dict[str, Any],
    ops: Dict[str, Any],
) -> Any:
    """Evaluate a postfix expression using single values (not vectorized).
    
    Args:
        postfix: Tuple of tokens in postfix notation
        var_values: Dict mapping variable names to their values
        ops: Dict mapping operator strings to operator functions
        
    Returns:
        The result of evaluating the expression
    """
    import operator as op_module
    
    # Map operators to scalar versions (not numpy vectorized)
    scalar_ops = {
        '==': op_module.eq,
        '!=': op_module.ne,
        '<=': op_module.le,
        '>=': op_module.ge,
        '<': op_module.lt,
        '>': op_module.gt,
        '&': lambda a, b: bool(a) and bool(b),
        '|': lambda a, b: bool(a) or bool(b),
        '+': op_module.add,
        '-': op_module.sub,
        '*': op_module.mul,
        '/': op_module.truediv,
        '//': op_module.floordiv,
        '%': op_module.mod,
        '**': op_module.pow,
        'and': lambda a, b: bool(a) and bool(b),
        'or': lambda a, b: bool(a) or bool(b),
        'not': lambda a: not bool(a),
    }
    
    stack: List[Any] = []
    
    for token in postfix:
        if token.isnumeric():
            stack.append(int(token))
        elif token in scalar_ops:
            if token == 'not':
                if not stack:
                    raise ValueError(f"Empty stack for 'not' operator")
                a = stack.pop()
                stack.append(scalar_ops[token](a))
            else:
                if len(stack) < 2:
                    raise ValueError(f"Insufficient operands for '{token}'")
                b = stack.pop()
                a = stack.pop()
                stack.append(scalar_ops[token](a, b))
        elif re.match(r'^[a-zA-Z_]\w*$', token):
            if token not in var_values:
                raise ValueError(f"Unknown variable: {token}")
            stack.append(var_values[token])
        else:
            raise ValueError(f"Invalid token: {token}")
    
    if len(stack) != 1:
        raise ValueError(f"Invalid expression result: stack has {len(stack)} items")
    
    return stack[0]

