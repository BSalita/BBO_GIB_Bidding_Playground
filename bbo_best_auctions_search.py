#!/usr/bin/env python3
"""
Standalone script to find best auctions by DD and EV.
This version runs entirely locally without any API calls.

Usage:
    python bbo_best_auctions_search.py --deal-index 1
    python bbo_best_auctions_search.py --deal-index 1 --metric EV
    python bbo_best_auctions_search.py --deal-index 1 --verbose
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import psutil

from bbo_bidding_queries_lib import (
    get_ai_contract,
    get_dd_score_for_auction,
    get_ev_for_auction,
    parse_contract_from_auction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------

class Direction(Enum):
    """Cardinal directions for bridge seats."""
    N = 0
    E = 1
    S = 2
    W = 3

    @classmethod
    def from_str(cls, s: str) -> "Direction":
        """Parse direction from string, defaulting to North."""
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.N

    def rotate(self, offset: int) -> "Direction":
        """Rotate direction by offset positions clockwise."""
        return Direction((self.value + offset) % 4)

    def __str__(self) -> str:
        return self.name


DIRECTIONS_LIST = [d.name for d in Direction]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

SCORE_MIN = float("-inf")
DD_SCORE_MIN = -99999
PASSES_TO_END_BIDDED = 3
PASSES_TO_END_EMPTY = 4
PROGRESS_REPORT_INTERVAL = 50
DD_SCORE_STRAINS = ("C", "D", "H", "S", "N")
DD_SCORE_LEVELS = tuple(range(1, 8))

# Pre-compiled Regex Patterns
CRITERION_NUM_RE = re.compile(r"^(\w+)\s*(>=|<=|==|!=|>|<)\s*(\d+)$")
CRITERION_REL_RE = re.compile(r"^(\w+)\s*(>=|<=|==|!=|>|<)\s*(\w+)$")


# ---------------------------------------------------------------------------
# Bid Encoding
# ---------------------------------------------------------------------------

class BidVocab:
    """Encapsulates bid-to-code and code-to-bid mappings."""

    PASS_BID = "P"

    def __init__(self) -> None:
        self._code_to_bid: list[str] = [""]
        self._bid_to_code: dict[str, int] = {"": 0}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build the bid vocabulary."""
        for b in ["P", "X", "XX", "D", "R"]:
            self._add(b)
        for level in range(1, 8):
            for strain in ["C", "D", "H", "S", "N"]:
                self._add(f"{level}{strain}")
        # Add lowercase aliases
        for b in ["p", "x", "xx", "d", "r"]:
            if b.upper() in self._bid_to_code:
                self._bid_to_code[b] = self._bid_to_code[b.upper()]

    def _add(self, bid: str) -> None:
        bid = bid.upper()
        if bid not in self._bid_to_code:
            self._bid_to_code[bid] = len(self._code_to_bid)
            self._code_to_bid.append(bid)

    def encode(self, bid: str) -> int:
        """Get code for a bid, returns 0 for unknown bids."""
        return self._bid_to_code.get(bid, 0)

    def decode(self, code: int) -> str:
        """Get bid string for a code."""
        if 0 <= code < len(self._code_to_bid):
            return self._code_to_bid[code]
        return ""

    @property
    def bid_to_code(self) -> dict[str, int]:
        return self._bid_to_code

    @property
    def code_to_bid(self) -> list[str]:
        return self._code_to_bid


# Singleton instance
BID_VOCAB = BidVocab()


# ---------------------------------------------------------------------------
# Auction Utilities
# ---------------------------------------------------------------------------

def normalize_auction_tokens(auc: str) -> list[str]:
    """Normalize an auction string to uppercase bid tokens (no empties)."""
    if not auc:
        return []
    return [t.strip().upper() for t in auc.split("-") if t.strip()]


def is_pass_bid(bid: str) -> bool:
    """Check if a bid is a pass."""
    return bid.upper() == BidVocab.PASS_BID


def count_leading_passes(tokens: list[str]) -> int:
    """Count the number of leading pass bids in a token list."""
    count = 0
    for t in tokens:
        if is_pass_bid(t):
            count += 1
        else:
            break
    return count


def strip_leading_passes(auc: str) -> tuple[str, int]:
    """Strip leading passes from an auction string, returning (stripped, count)."""
    tokens = normalize_auction_tokens(auc)
    if not tokens:
        return "", 0
    count = count_leading_passes(tokens)
    return "-".join(tokens[count:]), count


def is_auction_complete(auc: str) -> bool:
    """Check if an auction is complete (ended with sufficient passes)."""
    bids = normalize_auction_tokens(auc)
    if not bids:
        return False

    # All passes case
    if len(bids) >= PASSES_TO_END_EMPTY and all(is_pass_bid(b) for b in bids[:PASSES_TO_END_EMPTY]):
        return True

    # Find last contract bid
    last_contract_idx = _find_last_contract_index(bids)
    if last_contract_idx < 0:
        return False

    required_length = last_contract_idx + 1 + PASSES_TO_END_BIDDED
    return len(bids) >= required_length and all(is_pass_bid(b) for b in bids[-PASSES_TO_END_BIDDED:])


def _find_last_contract_index(bids: list[str]) -> int:
    """Find the index of the last contract bid (level+strain)."""
    last_idx = -1
    for i, b in enumerate(bids):
        if b not in ("P", "X", "XX") and b and b[0].isdigit():
            last_idx = i
    return last_idx


def get_contract_str(auc: str) -> str:
    """Get a human-readable contract string from an auction."""
    contract = parse_contract_from_auction(auc)
    if contract:
        level, strain, _ = contract
        strain_s = "NT" if str(strain).upper() == "N" else str(strain).upper()
        return f"{level}{strain_s}"
    if all(is_pass_bid(t) for t in auc.split("-") if t.strip()):
        return "Passed out"
    return "?"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class G3Index:
    """Gemini-3.2 CSR index for O(1) bidding table traversal."""
    offsets: np.ndarray    # uint32: [bt_index] -> edges start
    children: np.ndarray   # uint32: flat child indices
    bidcodes: np.ndarray   # uint8: flat bid codes
    openings: dict[str, int]
    is_complete: np.ndarray    # bool: [bt_index] -> is completed auction
    crit_offsets: np.ndarray   # uint32: [bt_index] -> crit_ids start
    crit_ids: np.ndarray       # uint16: flat criteria IDs
    bid_str_map: list[str]     # code -> bid string

    def walk(self, auction: str) -> int | None:
        """O(1) direct-address traversal of the bidding table."""
        tokens = normalize_auction_tokens(auction)
        # Skip leading passes
        tokens = tokens[count_leading_passes(tokens):]
        if not tokens:
            return None

        curr = self.openings.get(tokens[0])
        if curr is None:
            return None

        for tok in tokens[1:]:
            code = BID_VOCAB.encode(tok)
            start, end = self.offsets[curr], self.offsets[curr + 1]
            found = False
            for i in range(start, end):
                if self.bidcodes[i] == code:
                    curr = int(self.children[i])
                    found = True
                    break
            if not found:
                return None
        return curr


@dataclass
class TimingStats:
    """Tracks compute timing statistics."""
    compute_time_total: float = 0.0
    compute_latencies_ms: list[float] = field(default_factory=list)

    def record(self, latency_sec: float) -> None:
        """Record a compute call latency."""
        self.compute_time_total += latency_sec
        self.compute_latencies_ms.append(latency_sec * 1000)

    def get_summary(self) -> dict[str, Any]:
        """Get timing statistics summary."""
        if not self.compute_latencies_ms:
            return {"compute_calls": 0}
        latencies = sorted(self.compute_latencies_ms)
        n = len(latencies)
        return {
            "compute_calls": n,
            "compute_time_total_sec": round(self.compute_time_total, 3),
            "compute_avg_ms": round(sum(latencies) / n, 2),
            "compute_min_ms": round(latencies[0], 2),
            "compute_max_ms": round(latencies[-1], 2),
            "compute_p50_ms": round(latencies[n // 2], 2),
        }


@dataclass
class SearchConfig:
    """Configuration for auction search."""
    max_depth: int = 20
    max_results: int = 10
    metric: Literal["DD", "EV"] = "DD"
    verbose: bool = False
    target_score: float | None = None
    min_matches: int = 0


@dataclass
class SearchProgress:
    """Tracks search progress and results."""
    nodes_explored: int = 0
    completed_found: int = 0
    par_found: int = 0
    last_progress_report: int = 0
    alpha: float = SCORE_MIN
    early_termination_triggered: bool = False
    completed_auctions: list[tuple[float, int, str, str, int, list[str]]] = field(default_factory=list)
    best_score_cache: dict[str, float] = field(default_factory=dict)

    def record_completion(
        self,
        score: float,
        dd_score: int,
        auction: str,
        contract: str,
        depth: int,
        path: list[str],
        par_score: int | None
    ) -> None:
        """Record a completed auction."""
        self.completed_auctions.append((score, dd_score, auction, contract, depth, path))
        self.completed_found += 1
        if par_score is not None and dd_score == par_score:
            self.par_found += 1


@dataclass
class DealContext:
    """Context for a deal being searched."""
    deal: dict[str, Any]
    dealer: Direction
    row_idx: int | None
    par_score: int | None
    bitmap_row: dict[str, Any] | None = None

    @classmethod
    def from_deal(cls, deal: dict[str, Any]) -> "DealContext":
        """Create context from a deal dictionary."""
        dealer = Direction.from_str(str(deal.get("Dealer", "N")))
        row_idx = deal.get("_row_idx")
        par_score = cls._extract_par_score(deal)
        return cls(deal=deal, dealer=dealer, row_idx=row_idx, par_score=par_score)

    @classmethod
    def from_deal_with_loader(cls, deal: dict[str, Any], loader: "DataLoader") -> "DealContext":
        """Create context from a deal and attach cached bitmap row when available."""
        ctx = cls.from_deal(deal)
        if ctx.row_idx is not None and loader.criteria_bitmaps.height > ctx.row_idx:
            ctx.bitmap_row = loader.criteria_bitmaps.row(ctx.row_idx, named=True)
        return ctx

    @staticmethod
    def _extract_par_score(deal: dict[str, Any]) -> int | None:
        """Extract par score from deal, handling different key names."""
        for key in ["ParScore", "Par_Score"]:
            val = deal.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
        return None


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

class DataLoader:
    """Handles loading and preparing data for search."""

    def __init__(self, max_deal_rows: int | None = None):
        self.max_deal_rows = max_deal_rows
        self.deal_df: pl.DataFrame = pl.DataFrame()
        self.bt_seat1_df: pl.DataFrame = pl.DataFrame()
        self.criteria_bitmaps: pl.DataFrame = pl.DataFrame()
        self.criteria_bitmap_cols: set[str] = set()
        self.expr_map_by_direction: dict[str, dict[str, str]] = {}
        self.deal_index_arr: np.ndarray = np.array([])
        self.deal_index_monotonic: bool = True
        self.g3_index: G3Index | None = None
        self.unique_criteria: list[str] = []
        self.criterion_to_id: dict[str, int] = {}

    def load_all(self) -> None:
        """Load all necessary DataFrames and build indices."""
        print(f"[{datetime.now()}] [init] Loading standalone data...")
        t0 = time.time()

        plan = self._load_execution_plan()
        valid_cols = list(plan["valid_deal_columns"])

        self._load_deals(valid_cols)
        self._load_bidding_table()
        self._load_bitmaps()
        self._print_memory_usage("parquet load")
        self._build_g3_index()
        self._collect_unique_criteria()

        elapsed = time.time() - t0
        print(f"[{datetime.now()}] [init] Data loading completed in {elapsed:.2f}s")

    def _collect_unique_criteria(self) -> None:
        """Collect all unique criteria from the bidding table for pre-evaluation."""
        print("  Collecting unique criteria...")
        if "Expr" not in self.bt_seat1_df.columns:
            return
        
        # Explode Expr column and get unique values
        expr_series = self.bt_seat1_df.select(pl.col("Expr").explode().drop_nulls().unique()).get_column("Expr")
        self.unique_criteria = sorted([str(x) for x in expr_series.to_list()])
        self.criterion_to_id = {c: i for i, c in enumerate(self.unique_criteria)}
        print(f"  Found {len(self.unique_criteria)} unique criteria.")

    def _load_execution_plan(self) -> dict[str, Any]:
        """Load the execution plan containing expression mappings."""
        plan_file = DATA_DIR / "bbo_bt_execution_plan_data.pkl"
        with open(plan_file, "rb") as f:
            plan = pickle.load(f)
        self.expr_map_by_direction = plan["expr_map_by_direction"]
        self._print_stats("Execution Plan", plan_file, plan)
        return plan

    def _load_deals(self, valid_cols: list[str]) -> None:
        """Load the deals DataFrame."""
        deal_file = DATA_DIR / "bbo_mldf_augmented.parquet"
        available_cols = pl.scan_parquet(deal_file).collect_schema().names()
        load_cols = [c for c in valid_cols if c in available_cols]

        for col in self._build_additional_deal_columns():
            if col in available_cols and col not in load_cols:
                load_cols.append(col)

        for mandatory in ["index", "ParScore", "Par_Score", "Dealer", "Vul", "Vulnerability"]:
            if mandatory in available_cols and mandatory not in load_cols:
                load_cols.append(mandatory)

        self.deal_df = pl.read_parquet(deal_file, columns=load_cols, n_rows=self.max_deal_rows)
        self.deal_index_arr = self.deal_df.get_column("index").to_numpy()
        self.deal_index_monotonic = self._check_monotonic(self.deal_index_arr)
        self._print_stats("Deals", deal_file, self.deal_df)

    @staticmethod
    def _build_additional_deal_columns() -> list[str]:
        """Build list of additional columns needed for DD/EV computation."""
        additional_cols = [
            "PBN", "Vul", "Vulnerability", "Declarer", "bid", "Contract", "Result", "Tricks", "Score",
            "ParScore", "Par_Score", "DD_Score_Declarer", "EV_Score_Declarer", "ParContracts",
        ]

        for direction in DIRECTIONS_LIST:
            for strain in DD_SCORE_STRAINS:
                additional_cols.append(f"DD_{direction}_{strain}")

        for level in DD_SCORE_LEVELS:
            for strain in DD_SCORE_STRAINS:
                for direction in DIRECTIONS_LIST:
                    additional_cols.append(f"DD_Score_{level}{strain}_{direction}")

        for pair in ["NS", "EW"]:
            declarers = ["N", "S"] if pair == "NS" else ["E", "W"]
            for declarer in declarers:
                for strain in DD_SCORE_STRAINS:
                    for level in DD_SCORE_LEVELS:
                        additional_cols.append(f"EV_{pair}_{declarer}_{strain}_{level}")

        return additional_cols

    def _load_bidding_table(self) -> None:
        """Load the bidding table DataFrame."""
        bt_file = DATA_DIR / "bbo_bt_compiled.parquet"
        bt_cols = [
            "bt_index",
            "candidate_bid",
            "next_bid_indices",
            "is_opening_bid",
            "Auction",
            "is_completed_auction",
            "Expr",
        ]
        bt_available = pl.scan_parquet(bt_file).collect_schema().names()
        bt_load_cols = [c for c in bt_cols if c in bt_available]
        self.bt_seat1_df = pl.read_parquet(bt_file, columns=bt_load_cols)
        self._print_stats("Bidding Table", bt_file, self.bt_seat1_df)

    def _load_bitmaps(self) -> None:
        """Load the criteria bitmaps DataFrame."""
        bitmap_file = DATA_DIR / "bbo_mldf_augmented_criteria_bitmaps.parquet"
        bitmap_available = set(pl.scan_parquet(bitmap_file).collect_schema().names())
        needed_cols = self._compute_needed_bitmap_columns(bitmap_available)
        self.criteria_bitmaps = pl.read_parquet(bitmap_file, columns=needed_cols, n_rows=self.max_deal_rows)
        self.criteria_bitmap_cols = set(self.criteria_bitmaps.columns)
        self._print_stats("Bitmaps", bitmap_file, self.criteria_bitmaps)

    def _compute_needed_bitmap_columns(self, bitmap_available: set[str]) -> list[str]:
        """Determine which bitmap columns are actually needed for search."""
        if "Expr" not in self.bt_seat1_df.columns:
            return sorted(bitmap_available)

        crit_series = (
            self.bt_seat1_df
            .select(pl.col("Expr").explode().cast(pl.Utf8).drop_nulls().unique().alias("_crit"))
            .get_column("_crit")
        )
        crits = [str(x) for x in crit_series.to_list() if x is not None]

        needed: set[str] = set()
        for crit in crits:
            if crit in bitmap_available:
                needed.add(crit)
                continue

            for direction, mapping in self.expr_map_by_direction.items():
                directionless = mapping.get(crit)
                if directionless:
                    col_name = f"DIR_{direction}_{directionless}"
                    if col_name in bitmap_available:
                        needed.add(col_name)

        return sorted(needed) if needed else sorted(bitmap_available)

    def _build_g3_index(self) -> None:
        """Build the Gemini-3.2 CSR index from the bidding table."""
        print("  Building Gemini-3.2 CSR index...")
        t0 = time.time()
        
        # 1. Get max bt_index
        max_val = self.bt_seat1_df["bt_index"].max()
        max_idx = int(cast(Any, max_val)) if max_val is not None else 0

        # 2. Build is_complete array
        print("    Building is_complete array...")
        is_complete = np.zeros(max_idx + 1, dtype=bool)
        comp_indices = self.bt_seat1_df.filter(pl.col("is_completed_auction") == True).get_column("bt_index").to_numpy().astype(np.uint32)
        is_complete[comp_indices] = True

        # 3. Build criteria CSR
        print("    Building criteria CSR...")
        # Map Expr strings to IDs
        crit_ids_list = (
            self.bt_seat1_df
            .select([
                pl.col("bt_index"),
                pl.col("Expr")
                .list.eval(pl.element().replace_strict(self.criterion_to_id, default=None).drop_nulls())
                .cast(pl.List(pl.UInt16))
                .alias("ids")
            ])
        )
        
        # We need these to be in bt_index order for the offsets to work
        crit_ids_list = crit_ids_list.sort("bt_index")
        
        # Extract flat IDs and counts for offsets
        flat_crit_ids = crit_ids_list.get_column("ids").explode().drop_nulls().to_numpy().astype(np.uint16)
        crit_counts = crit_ids_list.get_column("ids").list.len().to_numpy().astype(np.uint32)
        crit_indices = crit_ids_list.get_column("bt_index").to_numpy().astype(np.uint32)
        
        # Build dense offsets
        crit_offsets = np.zeros(max_idx + 2, dtype=np.uint32)
        # crit_counts is the count for each bt_index in crit_indices
        # We need to accumulate them into crit_offsets[crit_indices + 1]
        dense_counts = np.zeros(max_idx + 1, dtype=np.uint32)
        dense_counts[crit_indices] = crit_counts
        crit_offsets[1:] = np.cumsum(dense_counts, dtype=np.uint32)

        # 4. Build children/offsets CSR
        print("    Building children CSR...")
        parent_info = self._build_parent_info()
        p_indices = parent_info["p"].to_numpy().astype(np.uint32)
        flat_children = parent_info["c"].to_numpy().astype(np.uint32)
        
        # Pre-map bid codes for children
        bid_codes_map = self._build_bid_codes_map(max_idx)
        flat_bidcodes = bid_codes_map[flat_children]
        
        offsets = self._build_offsets(p_indices, max_idx)
        openings = self._build_openings_map()

        self.g3_index = G3Index(
            offsets=offsets,
            children=flat_children,
            bidcodes=flat_bidcodes,
            openings=openings,
            is_complete=is_complete,
            crit_offsets=crit_offsets,
            crit_ids=flat_crit_ids,
            bid_str_map=BID_VOCAB.code_to_bid
        )
        print(f"    G3 index built in {time.time() - t0:.2f}s")

    def _build_bid_codes_map(self, max_idx: int) -> np.ndarray:
        """Build mapping from bt_index to bid codes."""
        bid_codes_map = np.zeros(max_idx + 1, dtype=np.uint8)
        bt_index_np = self.bt_seat1_df.get_column("bt_index").to_numpy().astype(int, copy=False)
        codes_np = (
            self.bt_seat1_df
            .select(
                pl.col("candidate_bid")
                .cast(pl.Utf8)
                .fill_null("")
                .str.to_uppercase()
                .replace_strict(BID_VOCAB.bid_to_code, default=0)
                .cast(pl.UInt8)
                .alias("_code")
            )
            .get_column("_code")
            .to_numpy()
        )
        bid_codes_map[bt_index_np] = codes_np
        return bid_codes_map

    def _build_parent_info(self) -> pl.DataFrame:
        """Build parent-child relationship DataFrame."""
        return (
            self.bt_seat1_df.filter(pl.col("next_bid_indices").list.len() > 0)
            .select(["bt_index", "next_bid_indices"])
            .explode("next_bid_indices")
            .rename({"bt_index": "p", "next_bid_indices": "c"})
            .sort("p")
        )

    def _build_offsets(self, p_indices: np.ndarray, max_idx: int) -> np.ndarray:
        """Build CSR offset array."""
        unique_p, counts = np.unique(p_indices, return_counts=True)
        degrees = np.zeros(max_idx + 1, dtype=np.uint32)
        degrees[unique_p] = counts.astype(np.uint32)
        offsets = np.zeros(max_idx + 2, dtype=np.uint32)
        offsets[1:] = np.cumsum(degrees, dtype=np.uint32)
        return offsets

    def _build_openings_map(self) -> dict[str, int]:
        """Build opening bid to bt_index mapping."""
        openings: dict[str, int] = {}
        opening_df = self.bt_seat1_df.filter(pl.col("is_opening_bid") == True).select(["Auction", "bt_index"])
        for row in opening_df.iter_rows():
            openings[str(row[0]).upper()] = int(row[1])
        return openings

    @staticmethod
    def _check_monotonic(arr: np.ndarray) -> bool:
        """Check if array is monotonically increasing."""
        return bool(np.all(arr[1:] >= arr[:-1])) if len(arr) > 1 else True

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes into a human-readable string."""
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        value = float(num_bytes)
        for unit in units:
            if value < 1024:
                return f"{value:.2f} {unit}"
            value /= 1024
        return f"{value:.2f} PB"

    def _print_memory_usage(self, stage: str) -> None:
        """Print process memory usage at a given stage."""
        proc = psutil.Process()
        mem = proc.memory_info()
        rss = self._format_bytes(mem.rss)
        vms = self._format_bytes(mem.vms)
        print(f"  Memory usage after {stage}: RSS={rss}, VMS={vms}")

    @staticmethod
    def _print_stats(name: str, path: Path, data: Any) -> None:
        """Print loading statistics."""
        size_mb = path.stat().st_size / (1024 * 1024)
        if hasattr(data, "shape"):
            shape_str = str(data.shape)
        elif isinstance(data, (dict, list)):
            shape_str = f"len={len(data)}"
        else:
            shape_str = "N/A"
        print(f"  Loaded {name:<18} from {path.name:<45} | Shape: {shape_str:<15} | Size: {size_mb:>7.2f} MB")


# ---------------------------------------------------------------------------
# Criteria Evaluation
# ---------------------------------------------------------------------------

class CriteriaEvaluator:
    """Evaluates bid criteria against deals."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.cache: dict[tuple[int, str, int, tuple[str, ...]], bool] = {}
        self.pre_eval_results: dict[str, np.ndarray] = {}  # direction -> boolean array of results

    def clear_cache(self) -> None:
        """Clear the criteria evaluation cache."""
        self.cache.clear()
        self.pre_eval_results.clear()

    def pre_evaluate_all(
        self,
        deal_row_idx: int,
        deal_row: dict[str, Any] | None = None,
        bitmap_row: dict[str, Any] | None = None
    ) -> None:
        """Pre-evaluate all unique criteria for the current deal and all directions."""
        if not self.loader.unique_criteria:
            return

        if deal_row is None:
            deal_row = self.loader.deal_df.row(deal_row_idx, named=True)
        if bitmap_row is None:
            bitmap_row = self.loader.criteria_bitmaps.row(deal_row_idx, named=True)

        num_crits = len(self.loader.unique_criteria)
        for direction in Direction:
            dir_str = str(direction)
            results = np.zeros(num_crits, dtype=bool)
            for i, crit in enumerate(self.loader.unique_criteria):
                res = self._evaluate_single_criterion(crit, direction, 0, direction, deal_row, bitmap_row)
                results[i] = bool(res) if res is not None else True # Treat untracked as pass
            self.pre_eval_results[dir_str] = results

    def is_valid(self, direction: str, criteria_ids: list[int]) -> bool:
        """Check if all criteria are met for a given direction using pre-evaluated results."""
        if not criteria_ids:
            return True
        results = self.pre_eval_results.get(direction)
        if results is None:
            return True
        # Check all criteria IDs in the pre-evaluated boolean array
        for cid in criteria_ids:
            if not results[cid]:
                return False
        return True

    def evaluate_batch(
        self,
        deal_row_idx: int,
        dealer: Direction,
        checks: list[dict[str, Any]],
        deal_row: dict[str, Any] | None = None,
        bitmap_row: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Evaluate criteria batch against a deal row."""
        if deal_row is None:
            deal_row = self.loader.deal_df.row(deal_row_idx, named=True)
        if bitmap_row is None:
            bitmap_row = self.loader.criteria_bitmaps.row(deal_row_idx, named=True)

        results = []
        for chk in checks:
            seat, criteria = int(chk["seat"]), chk["criteria"]
            direction = dealer.rotate(seat - 1)
            passed, failed, untracked = self._evaluate_criteria_list(
                criteria, dealer, seat, direction, deal_row, bitmap_row
            )
            results.append({"seat": seat, "passed": passed, "failed": failed, "untracked": untracked})
        return results

    def _evaluate_criteria_list(
        self,
        criteria: list[Any],
        dealer: Direction,
        seat: int,
        direction: Direction,
        deal_row: dict[str, Any],
        bitmap_row: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str]]:
        """Evaluate a list of criteria, categorizing results."""
        passed: list[str] = []
        failed: list[str] = []
        untracked: list[str] = []

        for crit in criteria:
            crit_s = str(crit)
            result = self._evaluate_single_criterion(crit_s, dealer, seat, direction, deal_row, bitmap_row)
            if result is True:
                passed.append(crit_s)
            elif result is False:
                failed.append(crit_s)
            else:
                untracked.append(crit_s)

        return passed, failed, untracked

    def _evaluate_single_criterion(
        self,
        criterion: str,
        dealer: Direction,
        seat: int,
        direction: Direction,
        deal_row: dict[str, Any],
        bitmap_row: dict[str, Any]
    ) -> bool | None:
        """Evaluate a single criterion against a deal."""
        # Try SL criterion first
        sl_result = self._evaluate_sl_criterion(criterion, direction, deal_row)
        if sl_result is not None:
            return sl_result

        # Try directionless bitmap
        dir_str = str(direction)
        directionless = self.loader.expr_map_by_direction.get(dir_str, {}).get(criterion)
        if directionless:
            col_name = f"DIR_{dir_str}_{directionless}"
            if col_name in self.loader.criteria_bitmap_cols:
                return bool(bitmap_row[col_name])

        # Try direct bitmap
        if criterion in self.loader.criteria_bitmap_cols:
            return bool(bitmap_row[criterion])

        return None

    def _evaluate_sl_criterion(
        self,
        criterion: str,
        direction: Direction,
        row: dict[str, Any]
    ) -> bool | None:
        """Evaluate a single SL criterion against a deal row."""
        dir_str = str(direction)

        m_num = CRITERION_NUM_RE.match(criterion)
        if m_num:
            col_base, op_str, val_str = m_num.groups()
            col = f"{col_base}_{dir_str}"
            if col not in row:
                return None
            return self._compare(row[col], op_str, int(val_str))

        m_rel = CRITERION_REL_RE.match(criterion)
        if m_rel:
            col1_base, op_str, col2_base = m_rel.groups()
            col1 = f"{col1_base}_{dir_str}"
            col2 = f"{col2_base}_{dir_str}"
            if col1 not in row or col2 not in row:
                return None
            return self._compare(row[col1], op_str, row[col2])

        return None

    @staticmethod
    def _compare(actual: Any, op: str, target: Any) -> bool:
        """Compare two values with the given operator."""
        ops = {
            ">=": lambda a, t: a >= t,
            "<=": lambda a, t: a <= t,
            ">": lambda a, t: a > t,
            "<": lambda a, t: a < t,
            "==": lambda a, t: a == t,
            "!=": lambda a, t: a != t,
        }
        return ops.get(op, lambda a, t: False)(actual, target)


# ---------------------------------------------------------------------------
# Search Engine
# ---------------------------------------------------------------------------

class SearchEngine:
    """Coordinates bid search with data access and criteria evaluation."""

    def __init__(self, max_deal_rows: int | None = None):
        self.loader = DataLoader(max_deal_rows)
        self.loader.load_all()
        self.evaluator = CriteriaEvaluator(self.loader)
        self.ui_valid_cache: dict[tuple[str, int | None, str | None], list[dict[str, Any]]] = {}
        self.bt_option_cache: dict[int, dict[str, Any]] = {}

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.ui_valid_cache.clear()
        self.evaluator.clear_cache()

    def fetch_deal(self, deal_index: int) -> dict[str, Any] | None:
        """Fetch a deal by its index."""
        row_idx = self._find_deal_row_idx(deal_index)
        if row_idx is None:
            return None
        row = self.loader.deal_df.row(row_idx, named=True)
        row["_row_idx"] = row_idx
        return row

    def _find_deal_row_idx(self, deal_index: int) -> int | None:
        """Find the row index for a deal index."""
        if self.loader.deal_index_monotonic:
            pos = np.searchsorted(self.loader.deal_index_arr, deal_index)
            if pos < len(self.loader.deal_index_arr) and int(self.loader.deal_index_arr[pos]) == deal_index:
                return int(pos)
            return None
        else:
            res = self.loader.deal_df.with_row_count().filter(pl.col("index") == deal_index)
            if res.height == 0:
                return None
            return int(res.get_column("row_nr")[0])

    def list_next_bids(self, auction: str) -> list[dict[str, Any]]:
        """List valid next bids for a given auction prefix."""
        g3 = self.loader.g3_index
        if g3 is None:
            return []

        if not auction:
            next_indices = list(g3.openings.values())
        else:
            bt_idx = g3.walk(auction)
            if bt_idx is None:
                return []
            start, end = g3.offsets[bt_idx], g3.offsets[bt_idx + 1]
            next_indices = [int(x) for x in g3.children[start:end]]

        if not next_indices:
            return []

        missing = [idx for idx in next_indices if idx not in self.bt_option_cache]
        if missing:
            rows = self.loader.bt_seat1_df.filter(pl.col("bt_index").is_in(missing))
            for row in rows.iter_rows(named=True):
                is_complete = bool(row.get("is_completed_auction"))
                next_idx_list = row.get("next_bid_indices") or []
                expr = row.get("Expr") or []
                criteria_ids = [self.loader.criterion_to_id[str(c)] for c in expr if str(c) in self.loader.criterion_to_id]
                self.bt_option_cache[int(row["bt_index"])] = {
                    "bid": str(row["candidate_bid"]).upper(),
                    "bt_index": row["bt_index"],
                    "agg_expr": expr,
                    "criteria_ids": criteria_ids,
                    "is_completed_auction": is_complete,
                    "is_dead_end": not is_complete and not next_idx_list,
                }

        return [self.bt_option_cache[idx] for idx in next_indices if idx in self.bt_option_cache]

    def get_valid_options(
        self,
        prefix_auc: str,
        deal_ctx: DealContext | None,
        timing: TimingStats,
        bt_index: int | None = None
    ) -> list[dict[str, Any]]:
        """Get valid next bids based on the auction prefix and deal context."""
        prefix_norm = "-".join(normalize_auction_tokens(prefix_auc))
        cache_key = (
            prefix_norm,
            deal_ctx.row_idx if deal_ctx else None,
            str(deal_ctx.dealer) if deal_ctx else None,
        )
        if cache_key in self.ui_valid_cache:
            return self.ui_valid_cache[cache_key]

        t0 = time.perf_counter()
        
        # If bt_index is provided, use it directly instead of walking
        if bt_index is not None:
            g3 = self.loader.g3_index
            if g3 is not None:
                start, end = g3.offsets[bt_index], g3.offsets[bt_index + 1]
                next_indices = [int(x) for x in g3.children[start:end]]
                opts = self._get_options_for_indices(next_indices)
            else:
                opts = []
        else:
            bt_prefix, leading_passes = strip_leading_passes(prefix_norm)
            opts = self.list_next_bids(bt_prefix)

        timing.record(time.perf_counter() - t0)

        if not opts:
            self.ui_valid_cache[cache_key] = []
            return []

        candidates = [o for o in opts if not o.get("is_dead_end")]
        if deal_ctx is None or deal_ctx.row_idx is None:
            self.ui_valid_cache[cache_key] = candidates
            return candidates

        toks = normalize_auction_tokens(prefix_norm)
        # Re-strip leading passes to find seat index correctly
        _, leading_passes = strip_leading_passes(prefix_norm)
        seat_index = self._bt_seat_from_display_seat((len(toks) % 4) + 1, leading_passes)
        dealer_rot = deal_ctx.dealer.rotate(len(toks)) # Use len(toks) for total rotation from dealer

        valid = self._filter_by_criteria_fast(candidates, dealer_rot)
        self.ui_valid_cache[cache_key] = valid
        return valid

    def _get_options_for_indices(self, next_indices: list[int]) -> list[dict[str, Any]]:
        """Fetch option dictionaries for the given bt_index values, using cache."""
        missing = [idx for idx in next_indices if idx not in self.bt_option_cache]
        if missing:
            rows = self.loader.bt_seat1_df.filter(pl.col("bt_index").is_in(missing))
            for row in rows.iter_rows(named=True):
                is_complete = bool(row.get("is_completed_auction"))
                next_idx_list = row.get("next_bid_indices") or []
                expr = row.get("Expr") or []
                criteria_ids = [self.loader.criterion_to_id[str(c)] for c in expr if str(c) in self.loader.criterion_to_id]
                self.bt_option_cache[int(row["bt_index"])] = {
                    "bid": str(row["candidate_bid"]).upper(),
                    "bt_index": row["bt_index"],
                    "agg_expr": expr,
                    "criteria_ids": criteria_ids,
                    "is_completed_auction": is_complete,
                    "is_dead_end": not is_complete and not next_idx_list,
                }
        return [self.bt_option_cache[idx] for idx in next_indices if idx in self.bt_option_cache]

    def _filter_by_criteria_fast(
        self,
        candidates: list[dict[str, Any]],
        direction: Direction
    ) -> list[dict[str, Any]]:
        """Filter candidates by criteria evaluation using pre-evaluated results."""
        dir_str = str(direction)
        valid: list[dict[str, Any]] = []
        for opt in candidates:
            if self.evaluator.is_valid(dir_str, opt.get("criteria_ids", [])):
                valid.append(opt)
        return valid

    def _filter_by_criteria(
        self,
        candidates: list[dict[str, Any]],
        deal_ctx: DealContext,
        seat_index: int,
        dealer: Direction,
        timing: TimingStats
    ) -> list[dict[str, Any]]:
        """Filter candidates by criteria evaluation."""
        valid: list[dict[str, Any]] = []
        unknown: list[dict[str, Any]] = []
        deal_row_idx = int(cast(int, deal_ctx.row_idx))

        for opt in candidates:
            criteria = tuple(str(x) for x in (opt.get("agg_expr") or []) if x is not None)
            if not criteria:
                valid.append(opt)
                continue

            cache_key = (deal_row_idx, str(dealer), seat_index, criteria)
            cached_res = self.evaluator.cache.get(cache_key)
            if cached_res is True:
                valid.append(opt)
            elif cached_res is False:
                continue
            else:
                unknown.append(opt)

        if unknown and deal_ctx.row_idx is not None:
            t0 = time.perf_counter()
            batch_checks = [{"seat": seat_index, "criteria": list(opt.get("agg_expr", []))} for opt in unknown]
            results = self.evaluator.evaluate_batch(
                deal_ctx.row_idx,
                dealer,
                batch_checks,
                deal_row=deal_ctx.deal,
                bitmap_row=deal_ctx.bitmap_row,
            )
            timing.record(time.perf_counter() - t0)

            for j, res in enumerate(results):
                opt = unknown[j]
                criteria = tuple(str(x) for x in (opt.get("agg_expr") or []) if x is not None)
                cache_key = (deal_row_idx, str(dealer), seat_index, criteria)
                if not res.get("failed") and not res.get("untracked"):
                    valid.append(opt)
                    self.evaluator.cache[cache_key] = True
                else:
                    self.evaluator.cache[cache_key] = False

        return valid

    @staticmethod
    def _bt_seat_from_display_seat(display_seat: int, leading_passes: int) -> int:
        """Convert display seat to BT seat, accounting for leading passes."""
        return ((display_seat - 1 - leading_passes) % 4) + 1


# ---------------------------------------------------------------------------
# Search Runner
# ---------------------------------------------------------------------------

class SearchRunner:
    """Executes auction search with configurable parameters."""

    def __init__(self, engine: SearchEngine, config: SearchConfig, deal_ctx: DealContext):
        self.engine = engine
        self.config = config
        self.deal_ctx = deal_ctx
        self.progress = SearchProgress()
        self.timing = TimingStats()
        self.search_start_time: float = 0.0
        self.score_cache: dict[tuple[str, str], float] = {}
        self.dd_score_cache: dict[str, int] = {}

    def run(self, start_auction: str = "") -> list[dict[str, Any]]:
        """Run the search and return formatted results."""
        self.search_start_time = time.perf_counter()
        
        # Pre-evaluate all criteria for this deal
        if self.deal_ctx.row_idx is not None:
            self.engine.evaluator.pre_evaluate_all(
                self.deal_ctx.row_idx,
                deal_row=self.deal_ctx.deal,
                bitmap_row=self.deal_ctx.bitmap_row
            )
        
        # Initial bt_index
        start_bt_index = None
        if start_auction:
            start_bt_index = self.engine.loader.g3_index.walk(start_auction) if self.engine.loader.g3_index else None

        self._explore(start_bt_index, start_auction, [], 0)
        self._print_summary()
        return self._format_results()

    def _explore(self, bt_index: int | None, prefix_auc: str, path_bids: list[str], depth: int) -> float:
        """Recursive search function."""
        if depth >= self.config.max_depth or self.progress.early_termination_triggered:
            return SCORE_MIN

        if prefix_auc in self.progress.best_score_cache:
            return self.progress.best_score_cache[prefix_auc]

        options = self.engine.get_valid_options(prefix_auc, self.deal_ctx, self.timing, bt_index=bt_index)
        self.progress.nodes_explored += 1

        self._maybe_report_progress(depth, bool(options))

        if not options:
            self.progress.best_score_cache[prefix_auc] = SCORE_MIN
            return SCORE_MIN

        best_here = SCORE_MIN
        for opt in options:
            bid = str(opt.get("bid", "")).upper()
            child_bt_index = opt.get("bt_index")
            if not bid:
                continue

            child_auc = f"{prefix_auc}-{bid}" if prefix_auc else bid
            child_path = path_bids + [bid]

            if opt.get("is_completed_auction"):
                score = self._evaluate_leaf(child_auc, child_path, depth + 1)
            else:
                score = self._explore(child_bt_index, child_auc, child_path, depth + 1)

            best_here = max(best_here, score)

            if self._check_termination():
                break

        self.progress.best_score_cache[prefix_auc] = best_here
        return best_here

    def _evaluate_leaf(self, auction: str, path: list[str], depth: int) -> float:
        """Evaluate and record a completed auction."""
        score = self._compute_score(auction)
        if score <= SCORE_MIN:
            return SCORE_MIN

        self.progress.alpha = max(self.progress.alpha, score)
        dd_score = self._get_dd_score(auction)
        contract = get_contract_str(auction)

        self.progress.record_completion(
            score, dd_score, auction, contract, depth, path, self.deal_ctx.par_score
        )

        if self.config.verbose:
            metric_str = "EV" if self.config.metric == "EV" else "DD"
            print(f"    -> Complete: {auction} => {contract} {metric_str}={score:.1f}")

        return score

    def _compute_score(self, auction: str) -> float:
        """Compute the score for an auction based on the configured metric."""
        deal = self.deal_ctx.deal
        dealer_str = str(self.deal_ctx.dealer)
        contract_key = get_ai_contract(auction, dealer_str)
        cache_key = (self.config.metric, contract_key) if contract_key else None
        if cache_key is not None and cache_key in self.score_cache:
            return self.score_cache[cache_key]

        try:
            if self.config.metric == "EV":
                ev = get_ev_for_auction(auction, dealer_str, deal)
                score = float(ev) if ev is not None else SCORE_MIN
            else:
                ddv = get_dd_score_for_auction(auction, dealer_str, deal)
                score = float(ddv) if ddv is not None else SCORE_MIN
        except (ValueError, TypeError, KeyError) as e:
            logger.debug("Score computation failed for %s: %s", auction, e)
            score = SCORE_MIN

        if cache_key is not None:
            self.score_cache[cache_key] = score
        return score

    def _get_dd_score(self, auction: str) -> int:
        """Get DD score for an auction (needed for Par checking)."""
        dealer_str = str(self.deal_ctx.dealer)
        contract_key = get_ai_contract(auction, dealer_str)
        if contract_key and contract_key in self.dd_score_cache:
            return self.dd_score_cache[contract_key]
        try:
            ddv = get_dd_score_for_auction(auction, dealer_str, self.deal_ctx.deal)
            dd_score = int(ddv) if ddv is not None else DD_SCORE_MIN
        except (ValueError, TypeError, KeyError):
            dd_score = DD_SCORE_MIN

        if contract_key:
            self.dd_score_cache[contract_key] = dd_score
        return dd_score

    def _check_termination(self) -> bool:
        """Check if early termination criteria are met."""
        if self.config.metric == "DD":
            if (
                self.deal_ctx.par_score is not None
                and self.config.min_matches > 0
                and self.progress.alpha >= self.deal_ctx.par_score
                and self.progress.par_found >= self.config.min_matches
            ):
                self.progress.early_termination_triggered = True
                return True
        elif self.config.metric == "EV":
            if self.config.target_score is not None and self.progress.alpha >= self.config.target_score:
                self.progress.early_termination_triggered = True
                return True
        return False

    def _maybe_report_progress(self, depth: int, has_options: bool) -> None:
        """Report progress if verbose and interval reached."""
        if not self.config.verbose or not has_options:
            return
        if self.progress.nodes_explored - self.progress.last_progress_report >= PROGRESS_REPORT_INTERVAL:
            self.progress.last_progress_report = self.progress.nodes_explored
            print(
                f"  [Progress] Nodes: {self.progress.nodes_explored}, "
                f"Completed: {self.progress.completed_found}, "
                f"Par: {self.progress.par_found}, Depth: {depth}",
                flush=True,
            )

    def _print_summary(self) -> None:
        """Print search summary."""
        elapsed = time.perf_counter() - self.search_start_time
        print(
            f"\n[{datetime.now()}] [Search] Completed in {elapsed:.2f}s. "
            f"Nodes: {self.progress.nodes_explored}, "
            f"Completed: {self.progress.completed_found}, "
            f"Par: {self.progress.par_found}"
        )

        timing_stats = self.timing.get_summary()
        if timing_stats.get("compute_calls", 0) > 0:
            print(
                f"  Compute Time: total={timing_stats['compute_time_total_sec']:.2f}s, "
                f"avg={timing_stats['compute_avg_ms']:.1f}ms"
            )

    def _format_results(self) -> list[dict[str, Any]]:
        """Format and filter search results."""
        # Sort: Primary score desc, then depth asc (shorter auctions first for same score)
        self.progress.completed_auctions.sort(key=lambda x: (x[0], -x[4]), reverse=True)

        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        for score, dd_score, auction, contract, depth, _ in self.progress.completed_auctions:
            if auction in seen:
                continue
            seen.add(auction)

            res_item: dict[str, Any] = {
                "Auction": auction,
                "Contract": contract,
                "DD_Score": dd_score,
                "Par": "Par" if self.deal_ctx.par_score and dd_score == self.deal_ctx.par_score else "",
                "Depth": depth,
            }
            if self.config.metric == "EV":
                res_item["EV"] = round(score, 1)
            results.append(res_item)

        return self._apply_result_filters(results)

    def _apply_result_filters(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply filtering logic based on metric and configuration."""
        max_results = self.config.max_results

        if self.config.metric == "DD" and self.deal_ctx.par_score is not None:
            par_results = [r for r in results if r.get("Par") == "✅"]
            if len(par_results) < max_results:
                other_results = [r for r in results if r.get("Par") != "✅"]
                par_results.extend(other_results[: max(0, max_results - len(par_results))])
            return par_results

        return results[:max_results]


# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

def find_best_auctions(
    engine: SearchEngine,
    deal: dict[str, Any],
    config: SearchConfig | None = None,
    start_auction: str = "",
) -> list[dict[str, Any]]:
    """Find best auctions for a deal.

    Args:
        engine: The search engine instance.
        deal: Deal dictionary with hand and score information.
        config: Search configuration. Uses defaults if None.
        start_auction: Starting auction prefix (empty for fresh search).

    Returns:
        List of result dictionaries with auction details.
    """
    if config is None:
        config = SearchConfig()

    deal_ctx = DealContext.from_deal_with_loader(deal, engine.loader)
    runner = SearchRunner(engine, config, deal_ctx)
    return runner.run(start_auction)


# ---------------------------------------------------------------------------
# Result Formatting
# ---------------------------------------------------------------------------

class ResultFormatter:
    """Formats search results for display."""

    DD_HEADER = f"{'Auction':<45} {'Contract':<12} {'DD Score':<10} {'Par':<5} {'Depth':<6}"
    EV_HEADER = f"{'Auction':<45} {'Contract':<12} {'EV':<10} {'DD Score':<10} {'Par':<5} {'Depth':<6}"

    @classmethod
    def print_results(cls, results: list[dict[str, Any]], metric: str) -> None:
        """Print formatted results to stdout."""
        print(f"\nResults for {metric}:")

        if metric == "EV":
            cls._print_ev_results(results)
        else:
            cls._print_dd_results(results)

    @classmethod
    def _print_dd_results(cls, results: list[dict[str, Any]]) -> None:
        """Print DD metric results."""
        print(cls.DD_HEADER)
        print("-" * len(cls.DD_HEADER))
        for r in results:
            print(
                f"{r['Auction']:<45} {r['Contract']:<12} "
                f"{r['DD_Score']:<10} {r['Par']:<5} {r['Depth']:<6}"
            )

    @classmethod
    def _print_ev_results(cls, results: list[dict[str, Any]]) -> None:
        """Print EV metric results."""
        print(cls.EV_HEADER)
        print("-" * len(cls.EV_HEADER))
        for r in results:
            print(
                f"{r['Auction']:<45} {r['Contract']:<12} "
                f"{r['EV']:<10} {r['DD_Score']:<10} {r['Par']:<5} {r['Depth']:<6}"
            )


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Find best auctions standalone")
    parser.add_argument("--deal-index", type=int, default=1)
    parser.add_argument("--metric", choices=["DD", "EV", "BOTH"], default="BOTH")
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--target-score", type=float, default=None)
    parser.add_argument("--min-matches", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    start_time = datetime.now()
    print(f"[{start_time}] Starting best auction search script...")

    engine = SearchEngine()
    deal = engine.fetch_deal(args.deal_index)

    if not deal:
        print(f"ERROR: Deal {args.deal_index} not found")
        return 1

    metrics_to_run: list[Literal["DD", "EV"]] = (
        ["DD", "EV"] if args.metric == "BOTH" else [cast(Literal["DD", "EV"], args.metric)]
    )

    for metric in metrics_to_run:
        engine.clear_caches()

        config = SearchConfig(
            max_depth=args.max_depth,
            max_results=args.max_results,
            metric=metric,
            verbose=args.verbose,
            target_score=args.target_score,
            min_matches=args.min_matches,
        )

        results = find_best_auctions(engine, deal, config)
        ResultFormatter.print_results(results, metric)

    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    print(f"\n[{end_time}] Script completed in {total_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
