"""
Bridge Bidding Queries Library

Reusable utility functions for auction parsing, hand analysis, scoring,
and distribution filtering. Used by bbo_bidding_queries_api.py and
other bridge analysis scripts.
"""

from __future__ import annotations

import re
from itertools import permutations
from typing import Dict, Any

import polars as pl

# Optional imports for par score calculation
try:
    from endplay.types import Deal, Vul, Player  # type: ignore[import-untyped]
    from endplay.dds import calc_dd_table, par  # type: ignore[import-untyped]
    HAS_ENDPLAY = True
except ImportError:
    HAS_ENDPLAY = False
    Deal = Vul = Player = calc_dd_table = par = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# IMP Calculation
# ---------------------------------------------------------------------------

def calculate_imp(score_diff: int) -> int:
    """
    Calculate IMPs for a given score difference using the standard IMP table.
    
    Args:
        score_diff: The score difference (absolute value used)
        
    Returns:
        IMP value (0-24)
    """
    diff = abs(int(score_diff))
    if diff < 20: return 0
    if diff < 50: return 1
    if diff < 90: return 2
    if diff < 130: return 3
    if diff < 170: return 4
    if diff < 220: return 5
    if diff < 270: return 6
    if diff < 320: return 7
    if diff < 370: return 8
    if diff < 430: return 9
    if diff < 500: return 10
    if diff < 600: return 11
    if diff < 750: return 12
    if diff < 900: return 13
    if diff < 1100: return 14
    if diff < 1300: return 15
    if diff < 1500: return 16
    if diff < 1750: return 17
    if diff < 2000: return 18
    if diff < 2250: return 19
    if diff < 2500: return 20
    if diff < 3000: return 21
    if diff < 3500: return 22
    if diff < 4000: return 23
    return 24


# ---------------------------------------------------------------------------
# Elapsed Time Formatting
# ---------------------------------------------------------------------------

def format_elapsed(ms: float) -> str:
    """Format elapsed time in seconds (e.g., 5380.4ms -> '5.38s').
    
    This is the single source of truth for elapsed time display.
    """
    return f"{ms / 1000:.2f}s"


# ---------------------------------------------------------------------------
# Hybrid Pattern Matching (literal prefix vs regex)
# ---------------------------------------------------------------------------

# Regex metacharacters that indicate a pattern needs regex matching
_REGEX_METACHAR_RE = re.compile(r'[\[\].*+?^${}()|\\]')

# Cache for compiled regex patterns (thread-safe via GIL for reads)
_pattern_cache: Dict[str, re.Pattern] = {}


def is_regex_pattern(pattern: str) -> bool:
    """Check if a pattern contains regex metacharacters.
    
    Returns True if the pattern should be matched using regex,
    False if it can use fast string prefix matching.
    
    Examples:
        is_regex_pattern("1C") -> False (literal)
        is_regex_pattern("1C-P") -> False (literal)
        is_regex_pattern("^1[CD]$") -> True (regex)
        is_regex_pattern("1.*") -> True (regex)
    """
    return bool(_REGEX_METACHAR_RE.search(pattern))


def get_cached_regex(pattern: str) -> re.Pattern:
    """Get a compiled regex pattern, using cache for performance.
    
    Patterns are compiled with re.IGNORECASE for case-insensitive matching.
    """
    if pattern not in _pattern_cache:
        # Auto-anchor if not already anchored (for prefix matching behavior)
        if not pattern.startswith('^'):
            pattern = '^' + pattern
        _pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE)
    return _pattern_cache[pattern]


def pattern_matches(pattern: str, text: str) -> bool:
    """Match a pattern against text using hybrid literal/regex matching.
    
    For literal patterns (no regex metacharacters): uses fast startswith()
    For regex patterns: uses compiled regex with caching
    
    Args:
        pattern: The pattern to match (literal prefix or regex)
        text: The text to match against
        
    Returns:
        True if the pattern matches, False otherwise
    """
    if not pattern:
        return True  # Empty pattern matches everything
    
    if is_regex_pattern(pattern):
        # Regex path (slower but flexible)
        return bool(get_cached_regex(pattern).match(text))
    else:
        # Literal path (fast string prefix matching)
        return text.upper().startswith(pattern.upper())


# ---------------------------------------------------------------------------
# Auction/Contract Parsing
# ---------------------------------------------------------------------------

def normalize_auction_input(auction: str) -> str:
    """Normalize a user-entered auction *string* into canonical dash-separated form.

    - Accepts bid separators: '-', whitespace, or ','
    - Canonical output separator is '-'
    - Normalizes passes: 'pass' -> 'P'
    - Normalizes NT: '1nt', '1n' -> '1N'
    - Uppercases all tokens (canonical UPPERCASE form)

    This is intended for literal auction strings (not regex patterns).
    """
    if auction is None:
        return ""
    s = str(auction).strip()
    if not s:
        return ""

    # Replace any supported separator (dash, comma, whitespace) with '-'
    parts = [p for p in re.split(r"[\s,\-]+", s) if p]
    out: list[str] = []
    suit_map = {"♣": "C", "♦": "D", "♥": "H", "♠": "S"}

    for raw in parts:
        t = raw.strip()
        if not t:
            continue
        tl = t.lower()
        if tl in ("p", "pass"):
            out.append("P")  # Canonical uppercase
            continue
        # Normalize suit symbols if user pasted them
        if any(sym in t for sym in suit_map):
            for sym, rep in suit_map.items():
                t = t.replace(sym, rep)
            tl = t.lower()

        # Contract bids like 1c/1d/1h/1s/1n/1nt
        if tl[0].isdigit() and len(tl) >= 2:
            level = tl[0]
            strain_raw = tl[1:]
            if strain_raw in ("n", "nt"):
                out.append(f"{level}N")
                continue
            if strain_raw and strain_raw[0] in ("c", "d", "h", "s"):
                out.append(f"{level}{strain_raw[0].upper()}")
                continue

        # Fallback: preserve token but uppercase it (e.g., X/XX)
        out.append(t.upper())

    return "-".join(out)


_REGEX_META_RE = re.compile(r"[\\^$.*+?()[\]{}|]")


def normalize_auction_user_text(text: str) -> str:
    """Normalize user-entered auction text, choosing literal vs regex normalization.

    Policy:
    - If **regex meta characters** are detected, treat input as a regex pattern and run
      `normalize_auction_pattern()` on the text as-is.
    - Otherwise, treat input as a literal auction string and run `normalize_auction_input()`.

    This provides a single point of truth for both Streamlit and API server inputs.
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    if _REGEX_META_RE.search(s):
        # Regex mode: do not rewrite user input; assume '-' is used in the pattern.
        return normalize_auction_pattern(s)

    # Literal mode
    return normalize_auction_input(s)

def normalize_auction_pattern(pattern: str) -> str:
    """
    Normalize an auction regex pattern by:
    1. Uppercasing bid tokens (1n->1N, p->P, etc.) while preserving regex syntax
    2. Appending only the passes needed to complete (not blindly adding 3)

    Bridge auctions end with 3 consecutive passes after any bid/double/redouble.

    Examples:
        '1n-p-3n' → '1N-P-3N-P-P-P' (uppercased, 3 passes needed)
        '^1N-p-3N$' → '^1N-P-3N-P-P-P$'
        '1c-p-p-d' → '1C-P-P-D-P-P-P' (3 passes after D)
        '1N-p' → '1N-P-P-P' (2 more passes needed, already has 1)
        '1N-p-p' → '1N-P-P-P' (1 more pass needed, already has 2)
        'p-p-p-p' → 'P-P-P-P' (pass-out, already complete)
        '1n-p-3n-p-p-p' → '1N-P-3N-P-P-P' (already complete, uppercased)
        '.*-3n' → '.*-3N-P-P-P' (wildcards preserved, bids uppercased)
    """
    if not pattern or not pattern.strip():
        return pattern

    pattern = pattern.strip()
    
    # Uppercase bid tokens while preserving regex metacharacters
    # Bid tokens: level+suit (1C-7N), P/pass, D/X/double, R/XX/redouble
    def uppercase_bids(m: re.Match) -> str:
        token = m.group(0)
        tl = token.lower()
        # Pass
        if tl in ("p", "pass"):
            return "P"
        # Double
        if tl in ("d", "x", "dbl", "double"):
            return "X"
        # Redouble
        if tl in ("r", "xx", "rdbl", "redouble"):
            return "XX"
        # Level + strain (1C, 2H, 3N, etc.)
        if len(tl) >= 2 and tl[0].isdigit():
            level = tl[0]
            strain = tl[1:]
            if strain in ("n", "nt"):
                return f"{level}N"
            if strain and strain[0] in "cdhs":
                return f"{level}{strain[0].upper()}"
        # Fallback: uppercase the whole token
        return token.upper()
    
    # Match bid-like tokens (word characters that look like bids) but not regex syntax
    # This pattern matches: digits+letters sequences that could be bids
    pattern = re.sub(r"\b([1-7]?[A-Za-z]+)\b", uppercase_bids, pattern)

    # Check for end anchor and temporarily remove it
    has_end_anchor = pattern.endswith("$")
    if has_end_anchor:
        pattern = pattern[:-1]

    # Don't append if pattern ends with open-ended wildcards that could match passes
    # e.g., '.*', '.+', '[^-]*' at the end
    if re.search(r"(\.\*|\.\+|\[[^\]]*\]\*|\[[^\]]*\]\+)$", pattern):
        return pattern + ("$" if has_end_anchor else "")

    # Check if the pattern contains any non-pass bid (opening bid exists)
    # Strip start anchor for this check
    pattern_no_anchor = pattern.lstrip("^")
    tokens = pattern_no_anchor.split("-")
    has_opening_bid = any(t.upper() not in ("P", "") for t in tokens if t and not re.search(r"[.*+?\[\](){}|\\]", t))
    
    if has_opening_bid:
        # Auction with bids: need 3 trailing passes after the last non-pass action
        if pattern.endswith("-P-P-P"):
            return pattern + ("$" if has_end_anchor else "")
        
        # Count trailing passes
        trailing_pass_match = re.search(r"(-P)+$", pattern)
        trailing_passes = trailing_pass_match.group(0).count("-P") if trailing_pass_match else 0
        
        passes_needed = max(0, 3 - trailing_passes)
        if passes_needed > 0:
            pattern = pattern + "-P" * passes_needed
    else:
        # Pass-only auction: need exactly 4 passes for pass-out
        # Count total passes (including leading P if present)
        pass_match = re.match(r"^\^?(P-)*P?$", pattern)
        if pass_match:
            total_passes = pattern.count("P")
            if total_passes < 4:
                passes_needed = 4 - total_passes
                if pattern.endswith("P"):
                    pattern = pattern + "-P" * passes_needed
                else:
                    pattern = pattern + "P" + "-P" * (passes_needed - 1) if passes_needed > 0 else pattern
    
    return pattern + ("$" if has_end_anchor else "")


def normalize_auction_pattern_to_seat1(pattern: str) -> str:
    """Normalize auction regex then strip any leading 'p-' prefixes (seat-1 view).
    
    This makes matching robust when callers provide seat-relative prefixes:
      '^p-1N-p-3N$' -> '^1N-p-3N-p-p-p$'
      'p-p-1C'      -> '1C-p-p-p' (after normalize_auction_pattern)
    
    Note: This is ONLY for matching. For display, prefixes can be re-applied
    based on seat (see API handler display helpers).
    """
    if not pattern or not pattern.strip():
        return pattern
    p = normalize_auction_pattern(pattern)
    # Strip leading p- after optional start anchor.
    p = re.sub(r"^\^(p-)+", "^", p)
    p = re.sub(r"^(p-)+", "", p)
    return p


def parse_contract_from_auction(auction: str) -> tuple[int, str, int] | None:
    """
    Parse the final contract from an auction string.
    
    Args:
        auction: Auction string like "1N-p-3N-p-p-p" or "1S-p-2S-p-p-p"
        
    Returns:
        Tuple of (level, strain, bid_position) where bid_position is the 0-based
        position of the final contract bid in the auction, or None if no valid contract.
        strain is one of 'C', 'D', 'H', 'S', 'N'
    """
    if not auction:
        return None
    
    # Split auction into bids
    bids = auction.upper().replace('P-P-P', '').rstrip('-').split('-')
    
    # Find the last non-pass bid (the contract)
    for i in range(len(bids) - 1, -1, -1):
        bid = bids[i].strip()
        if bid and bid != 'P' and len(bid) >= 2:
            level_char = bid[0]
            if level_char.isdigit():
                level = int(level_char)
                strain = bid[1].upper()
                # Normalize strain: NT -> N
                if strain == 'T' and len(bid) > 2:
                    strain = 'N'
                elif strain == 'N' and len(bid) > 2 and bid[2].upper() == 'T':
                    strain = 'N'
                if strain in 'CDHSN' and 1 <= level <= 7:
                    return (level, strain, i)
    return None


def get_declarer_for_auction(auction: str, dealer: str) -> str | None:
    """
    Determine the declarer direction for an auction.
    
    The declarer is the first player in the declaring partnership to bid the 
    contract's denomination.
    
    Args:
        auction: Auction string like "1N-p-3N-p-p-p"
        dealer: Dealer direction ('N', 'E', 'S', 'W')
        
    Returns:
        Declarer direction ('N', 'E', 'S', 'W') or None if cannot determine.
    """
    contract = parse_contract_from_auction(auction)
    if not contract:
        return None
    
    level, strain, contract_pos = contract
    
    # Direction order starting from dealer
    dirs = ['N', 'E', 'S', 'W']
    dealer_idx = dirs.index(dealer.upper()) if dealer.upper() in dirs else 0
    
    # Split auction into bids
    bids = auction.upper().split('-')
    
    # Determine the declaring side (who made the final contract bid)
    contract_bidder_idx = (dealer_idx + contract_pos) % 4
    contract_bidder = dirs[contract_bidder_idx]
    
    # The declaring partnership
    if contract_bidder in ['N', 'S']:
        partnership = ['N', 'S']
    else:
        partnership = ['E', 'W']
    
    # Find the first bid of this strain by the declaring partnership
    for i, bid in enumerate(bids):
        bid = bid.strip().upper()
        if bid and bid != 'P' and len(bid) >= 2:
            bid_strain = bid[1]
            if bid_strain == 'T' and len(bid) > 2:
                bid_strain = 'N'
            elif bid_strain == 'N' and len(bid) > 2 and bid[2].upper() == 'T':
                bid_strain = 'N'
            
            if bid_strain == strain:
                bidder_idx = (dealer_idx + i) % 4
                bidder = dirs[bidder_idx]
                if bidder in partnership:
                    return bidder
    
    # Fallback: the contract bidder is the declarer
    return contract_bidder


def get_ai_contract(auction: str, dealer: str) -> str | None:
    """
    Get the final contract string from an auction, including declarer and any doubles/redoubles.
    
    Args:
        auction: Auction string like "1N-p-3N-p-p-p" or "1S-X-XX-2S-p-p-p"
        dealer: Dealer direction ('N', 'E', 'S', 'W')
        
    Returns:
        Contract string like "3NN" (3NT by North), "3NXS" (3NT doubled by South), 
        or "3NXXE" (3NT redoubled by East), or None if no valid contract.
    """
    if not auction:
        return None
    
    # Split auction into bids (keep the trailing passes to track doubles after contract)
    bids = auction.upper().split('-')
    
    # Find the last contract bid and track any doubles/redoubles after it
    last_contract_idx = -1
    contract_str = None
    
    for i in range(len(bids) - 1, -1, -1):
        bid = bids[i].strip()
        if not bid or bid == 'P':
            continue
        if bid in ('X', 'D', 'DBL', 'DOUBLE'):
            continue  # Skip doubles when looking for contract
        if bid in ('XX', 'R', 'RDBL', 'REDOUBLE'):
            continue  # Skip redoubles when looking for contract
        
        # This should be a contract bid
        if len(bid) >= 2:
            level_char = bid[0]
            if level_char.isdigit():
                level = int(level_char)
                strain = bid[1].upper()
                # Normalize strain: NT -> N
                if strain == 'N' and len(bid) > 2 and bid[2].upper() == 'T':
                    strain = 'N'
                elif strain == 'T':
                    # Handle cases like "1T" which might mean 1NT
                    strain = 'N'
                
                if strain in 'CDHSN' and 1 <= level <= 7:
                    last_contract_idx = i
                    contract_str = f"{level}{strain}"
                    break
    
    if contract_str is None:
        return None
    
    # Check for doubles/redoubles after the contract bid
    is_doubled = False
    is_redoubled = False
    
    for i in range(last_contract_idx + 1, len(bids)):
        bid = bids[i].strip().upper()
        if bid in ('X', 'D', 'DBL', 'DOUBLE'):
            is_doubled = True
            is_redoubled = False  # A new double resets redouble
        elif bid in ('XX', 'R', 'RDBL', 'REDOUBLE'):
            is_redoubled = True
    
    if is_redoubled:
        contract_str += "XX"
    elif is_doubled:
        contract_str += "X"
    
    # Add declarer direction
    declarer = get_declarer_for_auction(auction, dealer)
    if declarer:
        contract_str += declarer
    
    return contract_str


def get_dd_score_for_auction(auction: str, dealer: str, deal_row: dict) -> int | None:
    """
    Get the double-dummy score for a contract derived from an auction.
    
    Args:
        auction: Auction string like "1N-p-3N-p-p-p"
        dealer: Dealer direction
        deal_row: Dictionary containing DD_Score_{level}{strain}_{direction} columns
        
    Returns:
        The DD score for the auction's contract, or None if cannot compute.
    """
    contract = parse_contract_from_auction(auction)
    if not contract:
        return None
    
    level, strain, _ = contract
    declarer = get_declarer_for_auction(auction, dealer)
    if not declarer:
        return None
    
    # Look up the DD score column
    col_name = f"DD_Score_{level}{strain}_{declarer}"
    return deal_row.get(col_name)


def get_dd_tricks_for_auction(auction: str, dealer: str, deal_row: dict) -> int | None:
    """
    Get the double-dummy trick count for a contract derived from an auction.
    
    Args:
        auction: Auction string like "1N-p-3N-p-p-p"
        dealer: Dealer direction
        deal_row: Dictionary containing DD_{direction}_{strain} columns (raw trick counts)
        
    Returns:
        The DD trick count (0-13) for the auction's contract, or None if cannot compute.
    """
    contract = parse_contract_from_auction(auction)
    if not contract:
        return None
    
    _, strain, _ = contract  # level not needed for trick lookup
    declarer = get_declarer_for_auction(auction, dealer)
    if not declarer:
        return None
    
    # Look up the DD tricks column: DD_{direction}_{strain}
    col_name = f"DD_{declarer}_{strain}"
    val = deal_row.get(col_name)
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    return None


def get_ev_for_auction(auction: str, dealer: str, deal_row: dict) -> float | None:
    """
    Get the expected value (EV) for a contract derived from an auction.
    
    Args:
        auction: Auction string like "1N-p-3N-p-p-p"
        dealer: Dealer direction
        deal_row: Dictionary containing EV_{pair}_{declarer}_{strain}_{level}_{vul} columns
                  and 'Vul' column for vulnerability
        
    Returns:
        The EV for the auction's contract, or None if cannot compute.
    """
    contract = parse_contract_from_auction(auction)
    if not contract:
        return None
    
    level, strain, _ = contract
    declarer = get_declarer_for_auction(auction, dealer)
    if not declarer:
        return None
    
    # Determine pair from declarer
    pair = 'NS' if declarer in ['N', 'S'] else 'EW'
    
    # Get vulnerability and convert to column format
    # Vul column values: 'None', 'Both', 'NS', 'EW' (or similar)
    vul_raw = deal_row.get('Vul', 'None')
    if vul_raw is None:
        vul_raw = 'None'
    vul_raw = str(vul_raw).strip()
    
    # Determine if declarer's side is vulnerable
    if vul_raw in ['Both', 'All', 'b', 'B']:
        vul = 'V'
    elif vul_raw == pair or (vul_raw == 'NS' and pair == 'NS') or (vul_raw == 'EW' and pair == 'EW'):
        vul = 'V'
    elif vul_raw in ['None', 'O', 'o', '-', '']:
        vul = 'NV'
    else:
        vul = 'NV'  # Default to not vulnerable
    
    # Look up the EV column: EV_{pair}_{declarer}_{strain}_{level}_{vul}
    col_name = f"EV_{pair}_{declarer}_{strain}_{level}_{vul}"
    return deal_row.get(col_name)


# ---------------------------------------------------------------------------
# Hand Analysis
# ---------------------------------------------------------------------------

def compute_hand_features(hand_str: str) -> dict:
    """
    Compute HCP, suit lengths, and total points for a hand.
    
    Args:
        hand_str: Hand in dot notation like "AKQ2.J98.T75.643"
        
    Returns:
        Dictionary with keys: HCP, SL_S, SL_H, SL_D, SL_C, Total_Points
    """
    hcp_values = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    
    suits = hand_str.split('.')
    if len(suits) != 4:
        return {}
    
    suit_names = ['S', 'H', 'D', 'C']
    result = {}
    total_hcp = 0
    dp = 0  # Distribution points
    
    for suit_name, suit_cards in zip(suit_names, suits):
        length = len(suit_cards)
        result[f'SL_{suit_name}'] = length
        
        suit_hcp = sum(hcp_values.get(c.upper(), 0) for c in suit_cards)
        total_hcp += suit_hcp
        
        # Distribution points: void=3, singleton=2, doubleton=1
        if length == 0:
            dp += 3
        elif length == 1:
            dp += 2
        elif length == 2:
            dp += 1
    
    result['HCP'] = total_hcp
    result['Total_Points'] = total_hcp + dp
    
    return result


# ---------------------------------------------------------------------------
# Par Score Calculation
# ---------------------------------------------------------------------------

def compute_par_score(pbn_str: str, dealer: str, vul: str = "None") -> dict:
    """
    Compute par score and contracts using endplay.
    
    Args:
        pbn_str: PBN deal string
        dealer: Dealer direction ('N', 'E', 'S', 'W')
        vul: Vulnerability ('None', 'Both', 'All', 'NS', 'N-S', 'EW', 'E-W')
        
    Returns:
        Dictionary with 'Par_Score' and 'Par_Contract' keys
    """
    if not HAS_ENDPLAY:
        return {'Par_Score': None, 'Par_Contract': 'Error: endplay not installed'}
    
    try:
        deal = Deal(pbn_str)  # type: ignore[misc]
        
        dealer_map = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}  # type: ignore[union-attr]
        dealer_player = dealer_map.get(dealer, Player.north)  # type: ignore[union-attr]
        
        vul_map = {
            'None': Vul.none, 'Both': Vul.both, 'All': Vul.both,  # type: ignore[union-attr]
            'NS': Vul.ns, 'N-S': Vul.ns,  # type: ignore[union-attr]
            'EW': Vul.ew, 'E-W': Vul.ew  # type: ignore[union-attr]
        }
        vul_enum = vul_map.get(vul, Vul.none)  # type: ignore[union-attr]
        
        dd_table = calc_dd_table(deal)  # type: ignore[misc]
        parlist = par(dd_table, vul_enum, dealer_player)  # type: ignore[misc]
        
        par_score = parlist.score
        
        # endplay Denom enum: spades=0, hearts=1, diamonds=2, clubs=3, nt=4
        strain_map = {0: 'S', 1: 'H', 2: 'D', 3: 'C', 4: 'N'}
        contracts_list = []
        for contract in parlist:
            level = contract.level
            strain = strain_map.get(int(contract.denom), '?')
            declarer = contract.declarer.abbr
            penalty = contract.penalty.abbr if contract.penalty.abbr != 'U' else ''
            result = contract.result
            
            result_str = f"+{result}" if result > 0 else str(result) if result < 0 else "="
            contract_str = f"{level}{strain}{penalty} {declarer} {result_str}"
            contracts_list.append(contract_str)
        
        par_contracts = ", ".join(contracts_list) if contracts_list else "Pass"
        
        return {'Par_Score': par_score, 'Par_Contract': par_contracts}
    except Exception as e:
        return {'Par_Score': None, 'Par_Contract': f"Error: {e}"}


# ---------------------------------------------------------------------------
# PBN Parsing
# ---------------------------------------------------------------------------

def parse_pbn_deal(pbn_str: str) -> dict | None:
    """
    Parse a PBN deal string into individual hands.
    
    Args:
        pbn_str: PBN format like "N:AKQ2.J98.T75.643 ..."
        
    Returns:
        Dictionary with Hand_N, Hand_E, Hand_S, Hand_W keys, plus:
        - Dealer: inferred from the leading "<Dir>:" prefix if present, else "N"
        - pbn: original input string
        or None if parsing fails
    """
    try:
        if not pbn_str:
            return None
        
        parts = pbn_str.strip().split()
        if len(parts) < 4:
            return None
        
        first_part = parts[0]
        if ':' in first_part:
            start_dir, first_hand = first_part.split(':', 1)
            hands = [first_hand] + parts[1:4]
        else:
            start_dir = 'N'
            hands = parts[:4]
        
        dirs = ['N', 'E', 'S', 'W']
        start_idx = dirs.index(start_dir.upper()) if start_dir.upper() in dirs else 0
        
        dealer = start_dir.upper() if start_dir.upper() in dirs else "N"
        result: dict[str, str] = {"Dealer": dealer, "pbn": pbn_str}
        for i, hand in enumerate(hands):
            dir_idx = (start_idx + i) % 4
            result[f'Hand_{dirs[dir_idx]}'] = hand
        
        return result

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Distribution Pattern Parsing
# ---------------------------------------------------------------------------

def parse_distribution_pattern(pattern: str) -> dict | None:
    """
    Parse a suit distribution pattern into filter criteria (S-H-D-C order).
    
    Supports formats:
    - Compact: '4333', '5332'
    - Dash-separated: '4-3-3-3'
    - Ranges: '[3:5]-x-x-x', '4+-x-x-x', '3--x-x-x'
    
    Args:
        pattern: Distribution pattern string
        
    Returns:
        Dictionary mapping suit ('S', 'H', 'D', 'C') to (min, max) tuple or None
    """
    if not pattern or not pattern.strip():
        return None
    
    pattern = pattern.strip()
    suits = ['S', 'H', 'D', 'C']
    result: dict[str, tuple[int, int] | None] = {s: None for s in suits}
    
    # Try compact numeric format (e.g., '4333', '5332')
    if re.match(r'^[0-9]{4}$', pattern):
        for i, suit in enumerate(suits):
            val = int(pattern[i])
            result[suit] = (val, val)
        return result
    
    # Split by dash for other formats
    parts = pattern.split('-')
    if len(parts) != 4:
        return None
    
    for i, (part, suit) in enumerate(zip(parts, suits)):
        part = part.strip()
        
        if part.lower() == 'x' or part in ('.*', '.+', '*', ''):
            result[suit] = None
            continue
        
        bracket_match = re.match(r'^\[(\d+)[-:](\d+)\]$', part)
        if bracket_match:
            result[suit] = (int(bracket_match.group(1)), int(bracket_match.group(2)))
            continue
        
        colon_match = re.match(r'^(\d+):(\d+)$', part)
        if colon_match:
            result[suit] = (int(colon_match.group(1)), int(colon_match.group(2)))
            continue
        
        plus_match = re.match(r'^(\d+)\+$', part)
        if plus_match:
            result[suit] = (int(plus_match.group(1)), 13)
            continue
        
        minus_match = re.match(r'^(\d+)-$', part)
        if minus_match:
            result[suit] = (0, int(minus_match.group(1)))
            continue
        
        if re.match(r'^\d+$', part):
            val = int(part)
            result[suit] = (val, val)
            continue
        
        result[suit] = None
    
    return result


def parse_sorted_shape(pattern: str) -> list[int] | None:
    """
    Parse a sorted shape pattern (e.g., '5431', '4432').
    
    Args:
        pattern: Shape pattern string
        
    Returns:
        List of 4 integers sorted descending, or None if invalid
    """
    if not pattern or not pattern.strip():
        return None
    
    pattern = pattern.strip()
    
    if re.match(r'^[0-9]{4}$', pattern):
        lengths = [int(c) for c in pattern]
        if sum(lengths) == 13:
            return sorted(lengths, reverse=True)
        return None
    
    parts = pattern.split('-')
    if len(parts) == 4:
        try:
            lengths = [int(p.strip()) for p in parts]
            if sum(lengths) == 13:
                return sorted(lengths, reverse=True)
        except ValueError:
            pass
    
    return None


def build_distribution_sql_for_bt(
    dist_pattern: str | None,
    sorted_shape: str | None,
    seat: int,
    available_columns: list[str]
) -> str:
    """
    Build SQL WHERE clause for bt_df distribution filtering.
    
    Args:
        dist_pattern: Distribution pattern or None
        sorted_shape: Sorted shape pattern or None
        seat: Seat number (1-4)
        available_columns: List of column names available in the DataFrame
        
    Returns:
        SQL WHERE clause string (without 'WHERE')
    """
    conditions = []
    suits = ['S', 'H', 'D', 'C']
    
    if dist_pattern:
        parsed = parse_distribution_pattern(dist_pattern)
        if parsed:
            for suit, constraint in parsed.items():
                if constraint is None:
                    continue
                min_val, max_val = constraint
                min_col = f"SL_{suit}_min_S{seat}"
                max_col = f"SL_{suit}_max_S{seat}"
                
                if min_col in available_columns and max_col in available_columns:
                    conditions.append(f'"{min_col}" <= {max_val}')
                    conditions.append(f'"{max_col}" >= {min_val}')
    
    if sorted_shape:
        shape = parse_sorted_shape(sorted_shape)
        if shape:
            perm_conditions = []
            unique_perms = set(permutations(shape))
            
            for perm in unique_perms:
                perm_parts = []
                for suit, expected_len in zip(suits, perm):
                    min_col = f"SL_{suit}_min_S{seat}"
                    max_col = f"SL_{suit}_max_S{seat}"
                    if min_col in available_columns and max_col in available_columns:
                        perm_parts.append(
                            f'("{min_col}" <= {expected_len} AND "{max_col}" >= {expected_len})'
                        )
                if len(perm_parts) == 4:
                    perm_conditions.append(f"({' AND '.join(perm_parts)})")
            
            if perm_conditions:
                conditions.append(f"({' OR '.join(perm_conditions)})")
    
    return " AND ".join(conditions) if conditions else ""


def build_distribution_sql_for_deals(
    dist_pattern: str | None,
    sorted_shape: str | None,
    direction: str
) -> str:
    """
    Build SQL WHERE clause for deal_df distribution filtering (Hand_* columns).
    
    Args:
        dist_pattern: Distribution pattern or None
        sorted_shape: Sorted shape pattern or None
        direction: Direction ('N', 'E', 'S', 'W')
        
    Returns:
        SQL WHERE clause string (without 'WHERE')
    """
    conditions = []
    suits = ['S', 'H', 'D', 'C']
    
    if dist_pattern:
        parsed = parse_distribution_pattern(dist_pattern)
        if parsed:
            for suit, constraint in parsed.items():
                if constraint is None:
                    continue
                min_val, max_val = constraint
                col = f"SL_{suit}_{direction}"
                
                if min_val == max_val:
                    conditions.append(f'"{col}" = {min_val}')
                else:
                    conditions.append(f'"{col}" >= {min_val} AND "{col}" <= {max_val}')
    
    if sorted_shape:
        shape = parse_sorted_shape(sorted_shape)
        if shape:
            perm_conditions = []
            unique_perms = set(permutations(shape))
            
            for perm in unique_perms:
                perm_parts = []
                for suit, expected_len in zip(suits, perm):
                    col = f"SL_{suit}_{direction}"
                    perm_parts.append(f'"{col}" = {expected_len}')
                perm_conditions.append(f"({' AND '.join(perm_parts)})")
            
            if perm_conditions:
                conditions.append(f"({' OR '.join(perm_conditions)})")
    
    return " AND ".join(conditions) if conditions else ""


def add_suit_length_columns(df: pl.DataFrame, direction: str) -> pl.DataFrame:
    """
    Add suit length columns for a specific direction's hand.
    
    Args:
        df: DataFrame with Hand_{direction} column
        direction: Direction ('N', 'E', 'S', 'W')
        
    Returns:
        DataFrame with SL_S_{direction}, SL_H_{direction}, etc. columns added
    """
    hand_col = f"Hand_{direction}"
    if hand_col not in df.columns:
        return df
    
    suits = ['S', 'H', 'D', 'C']
    for suit_idx, suit in enumerate(suits):
        col_name = f"SL_{suit}_{direction}"
        df = df.with_columns(
            pl.col(hand_col).str.split('.').list.get(suit_idx).str.len_chars().alias(col_name)
        )
    
    return df


# ---------------------------------------------------------------------------
# Criteria Evaluation (supports complex and suit-relational via shared helpers)
# ---------------------------------------------------------------------------

def evaluate_criterion_for_hand(criterion: str, hand_values: Dict[str, int]) -> bool:
    """
    Evaluate a criterion string (including complex/suit relational) against specific hand values.
    
    Delegates to evaluate_sl_criterion from plugins.bbo_handlers_common to keep
    behavior consistent across frontend/backend and support expressions like
    'SL_D >= SL_C & ((SL_D > SL_H & SL_D > SL_S) | (SL_H <= 4 & SL_S <= 4))'.
    
    Returns True if the criterion passes, False if it fails, and True for
    unparseable criteria (maintains prior permissive behavior).
    """
    from plugins.bbo_handlers_common import evaluate_sl_criterion

    crit_s = str(criterion or "").strip()
    if not crit_s:
        return True

    # Build a minimal deal_row for direction 'N' using provided hand values.
    deal_row = {
        "HCP_N": hand_values.get("HCP"),
        "Total_Points_N": hand_values.get("Total_Points"),
        "SL_N_S": hand_values.get("SL_S"),
        "SL_N_H": hand_values.get("SL_H"),
        "SL_N_D": hand_values.get("SL_D"),
        "SL_N_C": hand_values.get("SL_C"),
    }

    try:
        res = evaluate_sl_criterion(crit_s, dealer="N", seat=1, deal_row=deal_row, fail_on_missing=True)
        if res is None:
            # Not an SL/complex criterion; treat as pass to avoid over-filtering
            return True
        return bool(res)
    except Exception:
        # Preserve prior behavior: unparseable criteria do not filter out rows
        return True


# ---------------------------------------------------------------------------
# On-the-fly BT Auction Matching (for PBN/non-deal_df)
# ---------------------------------------------------------------------------

DIRECTIONS_LIST = ["N", "E", "S", "W"]
RANGE_METRICS = ["HCP", "SL_C", "SL_D", "SL_H", "SL_S", "Total_Points"]

DEFAULT_METRIC_RANGES = {
    "HCP": (0, 40),
    "SL_C": (0, 13),
    "SL_D": (0, 13),
    "SL_H": (0, 13),
    "SL_S": (0, 13),
    "Total_Points": (0, 50),
}

# Regex patterns for parsing criteria
_PATTERN_LE = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*<=\s*(\d+)$")
_PATTERN_GE = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*>=\s*(\d+)$")
_PATTERN_EQ = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*==\s*(\d+)$")
_PATTERN_LT = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*<\s*(\d+)$")
_PATTERN_GT = re.compile(r"^(HCP|SL_[CDHS]|Total_Points)\s*>\s*(\d+)$")


def _parse_criteria_to_ranges(criteria_list: list | None) -> Dict[str, tuple]:
    """Parse criteria expressions into min/max ranges per metric."""
    if not criteria_list:
        return {m: DEFAULT_METRIC_RANGES[m] for m in RANGE_METRICS}
    
    mins = {m: [] for m in RANGE_METRICS}
    maxs = {m: [] for m in RANGE_METRICS}
    
    for expr in criteria_list:
        expr = str(expr).strip()
        
        for pattern, is_min, offset in [
            (_PATTERN_GE, True, 0),
            (_PATTERN_LE, False, 0),
            (_PATTERN_GT, True, 1),
            (_PATTERN_LT, False, -1),
        ]:
            m = pattern.match(expr)
            if m:
                metric, val = m.groups()
                if is_min:
                    mins[metric].append(int(val) + offset)
                else:
                    maxs[metric].append(int(val) + offset)
                break
        else:
            m = _PATTERN_EQ.match(expr)
            if m:
                metric, val = m.groups()
                mins[metric].append(int(val))
                maxs[metric].append(int(val))
    
    result = {}
    for metric in RANGE_METRICS:
        d_min, d_max = DEFAULT_METRIC_RANGES[metric]
        final_min = max(mins[metric]) if mins[metric] else d_min
        final_max = min(maxs[metric]) if maxs[metric] else d_max
        result[metric] = (final_min, final_max)
    
    return result


def _get_seat_direction(dealer: str, seat: int) -> str:
    """Map seat number (1-4) to direction (N/E/S/W) based on dealer."""
    dealer_idx = DIRECTIONS_LIST.index(dealer) if dealer in DIRECTIONS_LIST else 0
    return DIRECTIONS_LIST[(dealer_idx + seat - 1) % 4]


def _hand_matches_ranges(
    hand_features: Dict[str, int],
    ranges: Dict[str, tuple],
) -> bool:
    """Check if a hand's features fall within the specified ranges."""
    for metric, (lo, hi) in ranges.items():
        val = hand_features.get(metric)
        if val is None:
            continue
        if val < lo or val > hi:
            return False
    return True


def find_matching_bt_auctions(
    hands: Dict[str, str],
    dealer: str,
    bt_completed_df,
    max_matches: int = 10,
) -> list:
    """Find completed BT auctions matching a deal's hand criteria.
    
    This is for on-the-fly matching of PBN deals not in deal_df.
    Uses range pre-filtering for efficiency.
    
    Args:
        hands: Dict mapping direction to hand string, e.g.:
               {"N": "AKQ2.JT9.876.543", "E": "...", "S": "...", "W": "..."}
        dealer: Dealer direction ('N', 'E', 'S', 'W')
        bt_completed_df: DataFrame of completed auctions with Agg_Expr_Seat_1..4
        max_matches: Maximum number of matches to return
    
    Returns:
        List of matching auction strings (e.g., ["1N-p-3N-p-p-p", ...])
    """
    import polars as pl
    
    # Compute hand features for all directions
    hand_features = {}
    for direction in DIRECTIONS_LIST:
        hand_str = hands.get(direction, "")
        if hand_str:
            features = compute_hand_features(hand_str)
            hand_features[direction] = features
    
    if not hand_features:
        return []
    
    matches = []
    
    for row in bt_completed_df.iter_rows(named=True):
        auction = row.get("Auction")
        if not auction:
            continue
        
        # Check each seat's criteria
        all_seats_pass = True
        
        for seat in range(1, 5):
            criteria_list = row.get(f"Agg_Expr_Seat_{seat}")
            if not criteria_list:
                continue
            
            direction = _get_seat_direction(dealer, seat)
            features = hand_features.get(direction)
            
            if not features:
                all_seats_pass = False
                break
            
            # Parse criteria to ranges
            ranges = _parse_criteria_to_ranges(criteria_list)
            
            # Check range criteria
            if not _hand_matches_ranges(features, ranges):
                all_seats_pass = False
                break
            
            # For non-range criteria (Biddable, Stopper, etc.), we'd need
            # additional logic. For now, range-based is a good approximation.
        
        if all_seats_pass:
            matches.append(auction)
    
    # Sort alphabetically descending, then limit to max_matches
    matches.sort(reverse=True)
    return matches[:max_matches]


def find_matching_bt_auctions_from_pbn(
    pbn_str: str,
    bt_completed_df,
    max_matches: int = 10,
) -> list:
    """Find matching BT auctions for a PBN deal string.
    
    Convenience wrapper that parses the PBN and calls find_matching_bt_auctions.
    
    Args:
        pbn_str: PBN format deal string (e.g., "N:AKQ2.JT9.876.543 ...")
        bt_completed_df: DataFrame of completed auctions
        max_matches: Maximum matches to return
    
    Returns:
        List of matching auction strings
    """
    parsed = parse_pbn_deal(pbn_str)
    if not parsed:
        return []
    
    hands = {
        "N": parsed.get("Hand_N", ""),
        "E": parsed.get("Hand_E", ""),
        "S": parsed.get("Hand_S", ""),
        "W": parsed.get("Hand_W", ""),
    }
    dealer = parsed.get("Dealer", "N")
    
    return find_matching_bt_auctions(hands, dealer, bt_completed_df, max_matches)