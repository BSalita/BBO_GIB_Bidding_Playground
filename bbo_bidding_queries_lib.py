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
    from endplay.types import Deal, Vul, Player
    from endplay.dds import calc_dd_table, par
    HAS_ENDPLAY = True
except ImportError:
    HAS_ENDPLAY = False


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
# Auction/Contract Parsing
# ---------------------------------------------------------------------------

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
        deal = Deal(pbn_str)
        
        dealer_map = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}
        dealer_player = dealer_map.get(dealer, Player.north)
        
        vul_map = {
            'None': Vul.none, 'Both': Vul.both, 'All': Vul.both,
            'NS': Vul.ns, 'N-S': Vul.ns,
            'EW': Vul.ew, 'E-W': Vul.ew
        }
        vul_enum = vul_map.get(vul, Vul.none)
        
        dd_table = calc_dd_table(deal)
        parlist = par(dd_table, vul_enum, dealer_player)
        
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
        Dictionary with Hand_N, Hand_E, Hand_S, Hand_W keys, or None if parsing fails
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
        
        result = {}
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
# Criteria Evaluation
# ---------------------------------------------------------------------------

def evaluate_criterion_for_hand(criterion: str, hand_values: Dict[str, int]) -> bool:
    """
    Evaluate a criterion string like 'SL_S >= SL_H' against specific hand values.
    
    Args:
        criterion: Criterion expression like 'HCP >= 15', 'SL_S >= SL_H'
        hand_values: Dictionary with keys: HCP, SL_S, SL_H, SL_D, SL_C, Total_Points
        
    Returns:
        True if the criterion is satisfied, False otherwise.
        Returns True if criterion cannot be parsed (don't filter on unparseable criteria).
    """
    pattern = r'(\w+)\s*(>=|<=|>|<|==|!=)\s*(\w+|\d+)'
    match = re.match(pattern, criterion.strip())
    if not match:
        return True  # Can't parse, don't filter
    
    left, op, right = match.groups()
    
    # Get left value
    left_val = hand_values.get(left)
    if left_val is None:
        return True  # Unknown column, don't filter
    
    # Get right value (either from hand_values or as a number)
    try:
        right_val = float(right)
    except ValueError:
        right_val = hand_values.get(right)
        if right_val is None:
            return True  # Unknown column, don't filter
    
    # Evaluate
    if op == '>=':
        return left_val >= right_val
    elif op == '<=':
        return left_val <= right_val
    elif op == '>':
        return left_val > right_val
    elif op == '<':
        return left_val < right_val
    elif op == '==':
        return left_val == right_val
    elif op == '!=':
        return left_val != right_val
    
    return True

