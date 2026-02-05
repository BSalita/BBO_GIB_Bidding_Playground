"""
Bid Category Classification

Classifies bridge bids into 100 semantic categories based on auction context.

Input: data/bbo_bt_merged_rules.parquet
Output: data/bbo_bt_categories.parquet

Run: python bbo_bt_classify_bids.py
"""

import polars as pl
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("E:/bridge/data/bbo/bidding")
INPUT_FILE = DATA_DIR / "bbo_bt_merged_rules.parquet"
OUTPUT_FILE = DATA_DIR / "bbo_bt_categories.parquet"

# Strain ranking for comparison
STRAINS = {'C': 1, 'D': 2, 'H': 3, 'S': 4, 'N': 5}

# ============================================================================
# BID PARSING UTILITIES
# ============================================================================

def parse_bid(bid_str: str) -> Optional[Tuple]:
    """Parse a bid string like '1H', '2N', 'P', 'D', 'R' into structured form."""
    if not bid_str:
        return None
    bid_str = bid_str.strip().upper()
    if bid_str in ('P', 'PASS'):
        return ('P',)
    if bid_str in ('D', 'X', 'DBL', 'DOUBLE'):
        return ('D',)
    if bid_str in ('R', 'XX', 'RDBL', 'REDOUBLE'):
        return ('R',)
    match = re.match(r'^(\d)([CDHSN])$', bid_str)
    if match:
        level = int(match.group(1))
        strain = match.group(2)
        return (level, strain)
    return None

def parse_auction(auction_str: str) -> List[Optional[Tuple]]:
    """Parse an auction string like '1H-P-2H-P' into list of parsed bids."""
    if not auction_str:
        return []
    bids = auction_str.replace(' ', '-').split('-')
    return [parse_bid(b) for b in bids if b]

# ============================================================================
# BID CLASSIFICATION FUNCTION
# ============================================================================

def classify_bid(step_auction: str, next_bid: str, pos_count: int, neg_count: int) -> Dict[str, bool]:
    """
    Classify a bid into semantic categories based on auction context.
    
    Args:
        step_auction: The auction so far (e.g., '1H-P-2H')
        next_bid: The candidate bid being classified
        pos_count: Number of positive matches for this rule
        neg_count: Number of negative matches for this rule
    
    Returns:
        Dictionary of boolean category flags (100 categories)
    """
    # Parse the auction
    # NOTE: step_auction INCLUDES the next_bid as the last element
    # So we need to parse the full auction and exclude the last bid
    full_auction = parse_auction(step_auction)
    
    # Exclude the last bid (which is the candidate) to get auction_before
    if full_auction:
        auction_before = full_auction[:-1]
    else:
        auction_before = []
    
    candidate = parse_bid(next_bid)
    
    # Determine candidate's seat position (0-indexed position in full auction)
    candidate_seat = len(auction_before)
    
    # Find partner's bids (every 4th bid starting from seat-2)
    partner_bids = []
    for i in range(candidate_seat - 2, -1, -4):
        if 0 <= i < len(auction_before):
            partner_bids.append(auction_before[i])
    
    # Find our previous bids
    our_bids = []
    for i in range(candidate_seat - 4, -1, -4):
        if 0 <= i < len(auction_before):
            our_bids.append(auction_before[i])
    
    # Find partner's first natural bid (not P/D/R)
    partner_natural = None
    for b in partner_bids:
        if b and isinstance(b[0], int):
            partner_natural = b
            break
    
    # Check if opponents have bid
    opponents_bid = False
    highest_opp_level = 0
    highest_opp_strain = None
    for i, b in enumerate(auction_before):
        if (candidate_seat - i) % 2 == 1:
            if b and isinstance(b[0], int):
                opponents_bid = True
                if b[0] > highest_opp_level or (b[0] == highest_opp_level and STRAINS.get(b[1], 0) > STRAINS.get(highest_opp_strain or '', 0)):
                    highest_opp_level = b[0]
                    highest_opp_strain = b[1]
    
    # Initialize all 102 categories to False
    result = {
        # Original categories (9)
        # NOTE: is_Raise = expresses fit in a strain (not just bidding same strain)
        #   Examples: 1S-2S (direct), 1N-2N (NT), 1S-2N (Jacoby shows spade fit)
        #   Includes: direct raises, NT raises, Jacoby 2NT, Bergen, Splinter, Drury
        'is_Preempt': False, 'is_Raise': False, 'is_Artificial': False,
        'is_Sacrifice': False, 'is_Penalty': False, 'is_Takeout': False,
        'is_FitSeek': False, 'is_Feature': False, 'is_LeadDirect': False,
        # Bid Strength/Intent (5)
        'is_Forcing': False, 'is_Invitational': False, 'is_SignOff': False,
        'is_GameTry': False, 'is_SlamTry': False,
        # Auction Structure (6)
        'is_Opening': False, 'is_Overcall': False, 'is_Response': False,
        'is_Rebid': False, 'is_Balancing': False, 'is_Competitive': False,
        # Strain Categories (3)
        'is_Major': False, 'is_Minor': False, 'is_Notrump': False,
        # Level Categories (3)
        'is_GameLevel': False, 'is_SlamLevel': False, 'is_PartScore': False,
        # Special Conventions (5)
        'is_Cuebid': False, 'is_Splinter': False, 'is_TwoSuited': False,
        'is_Weak': False, 'is_Strong': False,
        # Relationship Categories (4)
        'is_SupportShowing': False, 'is_Preference': False,
        'is_NewSuit': False, 'is_Fourth': False,
        # Bid Nature Categories (4)
        'is_Natural': False, 'is_Conventional': False,
        'is_Jump': False, 'is_Minimum': False,
        # Timing/Position Categories (3)
        'is_Direct': False, 'is_PassedHand': False, 'is_Reopening': False,
        # Intent Categories (4)
        'is_Constructive': False, 'is_Destructive': False,
        'is_Asking': False, 'is_Telling': False,
        # Escape/Correction Categories (3)
        'is_Escape': False, 'is_Rescue': False, 'is_Correction': False,
        # Agreement Categories (4)
        'is_FitEstablished': False, 'is_Denial': False,
        'is_TransferCompletion': False, 'is_SuperAccept': False,
        # Strength Clarity (2)
        'is_LimitBid': False, 'is_Unlimited': False,
        # Special Bid Types (4)
        'is_Pass': False, 'is_Double': False,
        'is_Redouble': False, 'is_Alertable': False,
        # Competitive Dynamics (2)
        # - is_Push: Reduces opponent's EV at the risk of reducing bidder's own EV
        #   (both sides accept increased risk of suboptimal par score)
        'is_FreeBid': False, 'is_Push': False,
        # Slam Mechanics (4)
        'is_ControlBid': False, 'is_KeyCardAsk': False,
        'is_KeyCardResponse': False, 'is_QueenAsk': False,
        # Force Mechanics (3)
        'is_GameForcing': False, 'is_NonForcing': False, 'is_SemiForcing': False,
        # Vulnerability Awareness (2)
        'is_VulAction': False, 'is_NonVulPreempt': False,
        # Shape Indicators (3)
        'is_BalancedShowing': False, 'is_LongSuitShowing': False, 'is_ShortShowing': False,
        # Risk Profile (3)
        # All bids aim to maximize expected value (EV) / par score for their side.
        # - is_Conservative: Safe, sound bid with positive EV
        # - is_Aggressive: Bold bid accepting some risk for reward
        # - is_Gambling: Accepts negative EV for self, hoping to get lucky
        'is_Conservative': False, 'is_Aggressive': False, 'is_Gambling': False,
        # Special Patterns (4)
        'is_Trap': False, 'is_Slow': False,
        'is_WaitingBid': False, 'is_LastTrain': False,
        # Double Types (5)
        'is_NegativeDouble': False, 'is_ResponsiveDouble': False,
        'is_SupportDouble': False, 'is_LeadDirectingDouble': False, 'is_OptionalDouble': False,
        # Partnership Role (2)
        'is_Captain': False, 'is_Describer': False,
        # Bid Characteristics (4)
        'is_Reverse': False, 'is_JumpShift': False,
        'is_CheapestBid': False, 'is_HighCard': False,
        # Specific Conventions (6)
        'is_Transfer': False, 'is_Relay': False, 'is_Checkback': False,
        'is_Drury': False, 'is_Bergen': False, 'is_Jacoby2NT': False,
        # Hand Evaluation Signals (3)
        'is_MinimumShowing': False, 'is_MaximumShowing': False, 'is_ExtraValues': False,
        # Strong Bid Responses (3)
        'is_NegativeResponse': False, 'is_PositiveResponse': False, 'is_SecondNegative': False,
    }
    
    if not candidate:
        return result
    
    # =========================================================================
    # SPECIAL BID TYPES
    # =========================================================================
    if candidate[0] == 'P':
        result['is_Pass'] = True
        result['is_SignOff'] = True
        result['is_Conservative'] = True
        return result
    elif candidate[0] == 'D':
        result['is_Double'] = True
    elif candidate[0] == 'R':
        result['is_Redouble'] = True
    
    # For suit/NT bids
    if isinstance(candidate[0], int):
        level, strain = candidate
        
        # --- STRAIN CATEGORIES ---
        if strain in ('H', 'S'):
            result['is_Major'] = True
        elif strain in ('C', 'D'):
            result['is_Minor'] = True
        elif strain == 'N':
            result['is_Notrump'] = True
            result['is_BalancedShowing'] = True
        
        # --- LEVEL CATEGORIES ---
        if level >= 6:
            result['is_SlamLevel'] = True
        elif (level == 3 and strain == 'N') or \
             (level == 4 and strain in ('H', 'S')) or \
             (level == 5 and strain in ('C', 'D')):
            result['is_GameLevel'] = True
        else:
            result['is_PartScore'] = True
        
        # --- AUCTION STRUCTURE ---
        def is_pass_or_none(bid: Optional[Tuple]) -> bool:
            return bid is None or bid[0] == 'P'
        all_passes_before_us = all(
            is_pass_or_none(auction_before[i])
            for i in range(candidate_seat - 4, -1, -4)
            if 0 <= i < len(auction_before)
        )
        # Opening = first non-pass bid in the ENTIRE auction
        # Must check that NO ONE has bid before us (not just our side)
        all_passes_in_auction = all(
            is_pass_or_none(b) for b in auction_before
        )
        if all_passes_in_auction:
            result['is_Opening'] = True
            result['is_Describer'] = True
        
        if not result['is_Opening'] and opponents_bid:
            our_side_has_bid = any(b and isinstance(b[0], int) for b in our_bids + partner_bids)
            if not our_side_has_bid:
                result['is_Overcall'] = True
        
        if partner_natural and not our_bids:
            if not result['is_Overcall']:
                result['is_Response'] = True
        
        if our_bids:
            result['is_Rebid'] = True
            result['is_Describer'] = True
        
        if opponents_bid:
            result['is_Competitive'] = True
        
        if len(auction_before) >= 2:
            if auction_before[-1] and auction_before[-1][0] == 'P':
                if len(auction_before) >= 3 and auction_before[-2] and auction_before[-2][0] == 'P':
                    result['is_Balancing'] = True
                    result['is_Reopening'] = True
                    result['is_Aggressive'] = True
        
        # --- RAISE DETECTION ---
        # A raise expresses fit/agreement in a strain (not necessarily bidding same strain)
        # 
        # Direct raises: 1S-2S, 1H-3H (same strain at higher level)
        # NT raises: 1N-2N, 2N-3N (balanced values agreeing on NT)
        # Artificial raises: 1S-2N (Jacoby), 1H-3C (Bergen), 1S-4D (Splinter), 1H-2C (Drury)
        #   - These bid a DIFFERENT strain but show fit in partner's suit
        #   - Criteria expectations differ: suit raises need SL_X >= 3, NT raises need HCP/TP
        if partner_natural and isinstance(partner_natural[0], int):
            partner_strain = partner_natural[1]
            # Direct raise: same strain at higher level
            if strain == partner_strain:
                result['is_Raise'] = True
                result['is_SupportShowing'] = True
                result['is_FitEstablished'] = True
            # NT raise: partner opened NT, we raise NT
            elif partner_strain == 'N' and strain == 'N':
                result['is_Raise'] = True
                result['is_SupportShowing'] = True
                result['is_FitEstablished'] = True
        
        # --- NEW SUIT ---
        all_prior_strains = set()
        for b in auction_before:
            if b and isinstance(b[0], int):
                all_prior_strains.add(b[1])
        if strain not in all_prior_strains and strain != 'N':
            result['is_NewSuit'] = True
            result['is_FitSeek'] = True
        
        # --- CUEBID ---
        # A cuebid is a SPECIFIC conventional pattern, not just any bid in opp's suit
        # Be conservative - only mark clear cuebid patterns
        our_side_strains = set()
        for b in partner_bids + our_bids:
            if b and isinstance(b[0], int):
                our_side_strains.add(b[1])
        
        if highest_opp_strain and strain == highest_opp_strain:
            # Pattern 1: Michaels cuebid - direct 2-level overcall of opponent's major
            is_michaels = (
                result['is_Overcall'] and 
                not our_side_strains and  # First bid by our side
                level == 2 and 
                highest_opp_level == 1 and
                highest_opp_strain in ('H', 'S')  # Opponent opened a major
            )
            
            # Pattern 2: Cuebid raise - after fit established, bid opp's suit (slam try)
            # Only if we have a clear fit in a DIFFERENT suit
            is_cuebid_raise = (
                result['is_FitEstablished'] and
                our_side_strains and 
                strain not in our_side_strains and
                level >= 3  # At 3+ level for slam exploration
            )
            
            if is_michaels:
                result['is_Cuebid'] = True
                result['is_Artificial'] = True
                result['is_TwoSuited'] = True
            elif is_cuebid_raise:
                result['is_Cuebid'] = True
                result['is_Artificial'] = True
                result['is_ControlBid'] = True
        
        # --- JUMP DETECTION ---
        min_legal_level = 1
        for b in auction_before:
            if b and isinstance(b[0], int):
                bid_level, bid_strain = b
                if STRAINS.get(strain, 0) > STRAINS.get(bid_strain, 0):
                    min_legal_level = max(min_legal_level, bid_level)
                else:
                    min_legal_level = max(min_legal_level, bid_level + 1)
        
        if level > min_legal_level:
            result['is_Jump'] = True
        elif level == min_legal_level:
            result['is_Minimum'] = True
            result['is_CheapestBid'] = True
        
        # --- PREEMPT ---
        # Preempt = weak jump bid in a SUIT with intent to disrupt opponents
        # NOT notrump (2NT/3NT openings are strong, not preemptive)
        # NOT 2C (strong artificial opening)
        if result['is_Jump'] and result['is_Opening'] and level >= 2 and strain != 'N':
            # Exclude 2C which is the strong artificial opening
            if not (level == 2 and strain == 'C'):
                result['is_Preempt'] = True
                result['is_Weak'] = True
                result['is_Destructive'] = True
                result['is_Aggressive'] = True
                result['is_LongSuitShowing'] = True
        
        # Weak two openings (2D, 2H, 2S) - NOT 2C (strong) or 2N (strong balanced)
        if result['is_Opening'] and level == 2 and strain in ('D', 'H', 'S'):
            result['is_Weak'] = True
            result['is_Preempt'] = True
        
        # 2NT opening is strong balanced, not preemptive
        if result['is_Opening'] and candidate == (2, 'N'):
            result['is_Strong'] = True
            result['is_BalancedShowing'] = True
        
        # Gambling 3NT opening - long running minor, blocks lead info exchange
        # Intent: "Hail Mary" for game + prevent opponents from finding best lead
        if result['is_Opening'] and candidate == (3, 'N'):
            result['is_Gambling'] = True  # High-risk "Hail Mary" bid
            result['is_Aggressive'] = True
            result['is_Destructive'] = True  # Blocks opponent communication
            result['is_LongSuitShowing'] = True  # Implies long minor
        
        # --- STRONG BIDS ---
        if result['is_Opening'] and candidate == (2, 'C'):
            result['is_Strong'] = True
            result['is_Artificial'] = True
            result['is_GameForcing'] = True
        
        # --- SPLINTER ---
        if result['is_Jump'] and result['is_Response'] and result['is_Minor']:
            if partner_natural and partner_natural[1] in ('H', 'S'):
                if level >= 3:
                    result['is_Splinter'] = True
                    result['is_Artificial'] = True
                    result['is_ShortShowing'] = True
                    # Splinter shows fit in partner's major
                    result['is_Raise'] = True
                    result['is_SupportShowing'] = True
                    result['is_FitEstablished'] = True
        
        # --- TWO-SUITED ---
        if result['is_Cuebid'] and result['is_Overcall']:
            result['is_TwoSuited'] = True
        if candidate == (2, 'N') and result['is_Overcall']:
            # Unusual 2NT = two-suited overcall with preemptive intent
            result['is_TwoSuited'] = True
            result['is_Artificial'] = True
            result['is_Destructive'] = True  # Intent to disrupt opponents
            result['is_Weak'] = True  # Typically shows weak hand
            result['is_Aggressive'] = True
        
        # --- PREFERENCE ---
        if len(partner_bids) >= 2:
            partner_strains = [b[1] for b in partner_bids if b and isinstance(b[0], int)]
            if len(set(partner_strains)) >= 2 and strain in partner_strains:
                if not result['is_Raise']:
                    result['is_Preference'] = True
                    result['is_Correction'] = True
        
        # --- FOURTH SUIT FORCING ---
        if result['is_NewSuit'] and len(all_prior_strains) == 3:
            if result['is_Rebid'] or result['is_Response']:
                result['is_Fourth'] = True
                result['is_Forcing'] = True
                result['is_Artificial'] = True
                result['is_Conventional'] = True
        
        # --- GAME/SLAM TRIES ---
        if result['is_Raise'] and result['is_Jump'] and not result['is_Preempt']:
            if result['is_GameLevel']:
                result['is_GameTry'] = True
            elif result['is_SlamLevel']:
                result['is_SlamTry'] = True
            else:
                result['is_Invitational'] = True
        
        if candidate == (4, 'N') and result['is_FitEstablished']:
            result['is_KeyCardAsk'] = True
            result['is_Asking'] = True
            result['is_SlamTry'] = True
        
        # --- NATURAL vs ARTIFICIAL ---
        if not result['is_Artificial']:
            result['is_Natural'] = True
        
        # --- CONVENTIONAL ---
        if result['is_Artificial'] or result['is_TwoSuited'] or result['is_Splinter'] or result['is_Fourth']:
            result['is_Conventional'] = True
            result['is_Alertable'] = True
        
        # --- CONSTRUCTIVE ---
        if not result['is_Preempt'] and not result['is_Sacrifice']:
            if result['is_Raise'] or result['is_Response'] or result['is_Rebid']:
                result['is_Constructive'] = True
            if result['is_SlamTry'] or result['is_GameTry']:
                result['is_Constructive'] = True
        
        # --- DIRECT ---
        if len(auction_before) >= 1:
            last_bid = auction_before[-1]
            if last_bid and last_bid[0] not in ('P',):
                result['is_Direct'] = True
            elif last_bid and last_bid[0] == 'P' and not result['is_Balancing']:
                result['is_Direct'] = True
    
    # =========================================================================
    # DOUBLE CLASSIFICATIONS
    # =========================================================================
    if candidate and candidate[0] == 'D':
        if len(auction_before) >= 1:
            doubled_bid = auction_before[-1]
            if doubled_bid and isinstance(doubled_bid[0], int):
                level = doubled_bid[0]
                if level <= 3 and not partner_natural:
                    result['is_Takeout'] = True
                elif level >= 4:
                    result['is_Penalty'] = True
                    result['is_Destructive'] = True
        
        if partner_bids:
            partner_last = partner_bids[0] if partner_bids else None
            if partner_last and isinstance(partner_last[0], int):
                if len(auction_before) >= 2:
                    rho_bid = auction_before[-1]
                    if rho_bid and isinstance(rho_bid[0], int):
                        if not our_bids:
                            result['is_NegativeDouble'] = True
        
        if partner_bids:
            partner_last = partner_bids[0] if partner_bids else None
            if partner_last and partner_last[0] == 'D':
                result['is_ResponsiveDouble'] = True
        
        if result['is_Rebid']:
            if partner_bids:
                partner_resp = partner_bids[0] if partner_bids else None
                if partner_resp and isinstance(partner_resp[0], int):
                    if partner_resp[1] in ('H', 'S'):
                        result['is_SupportDouble'] = True
        
        if len(auction_before) >= 1:
            doubled_bid = auction_before[-1]
            if doubled_bid and isinstance(doubled_bid[0], int):
                level, strain = doubled_bid
                if level >= 4 and strain in ('C', 'D', 'H'):
                    result['is_LeadDirectingDouble'] = True
                if doubled_bid == (2, 'C'):
                    result['is_LeadDirectingDouble'] = True
        
        if len(auction_before) >= 1:
            doubled_bid = auction_before[-1]
            if doubled_bid and isinstance(doubled_bid[0], int):
                level = doubled_bid[0]
                if level >= 3:
                    if not result['is_Penalty'] and not result['is_Takeout']:
                        result['is_OptionalDouble'] = True
    
    # =========================================================================
    # PASSED HAND DETECTION
    # =========================================================================
    our_prior_bids_all = []
    for i in range(candidate_seat - 4, -1, -4):
        if 0 <= i < len(auction_before):
            our_prior_bids_all.append(auction_before[i])
    if any(b and b[0] == 'P' for b in our_prior_bids_all):
        result['is_PassedHand'] = True
    
    # =========================================================================
    # FREE BID
    # =========================================================================
    if candidate and candidate[0] != 'P':
        if len(auction_before) >= 1:
            rho_bid = auction_before[-1]
            if rho_bid and rho_bid[0] == 'P':
                result['is_FreeBid'] = True
    
    # =========================================================================
    # SPECIFIC CONVENTIONS
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        level, strain = candidate
        
        # Transfer
        if partner_natural == (1, 'N'):
            if candidate in [(2, 'D'), (2, 'H'), (4, 'D'), (4, 'H')]:
                result['is_Transfer'] = True
                result['is_Artificial'] = True
                result['is_Conventional'] = True
                result['is_Alertable'] = True
        if partner_natural == (2, 'N'):
            if candidate in [(3, 'D'), (3, 'H')]:
                result['is_Transfer'] = True
                result['is_Artificial'] = True
        
        # Relay (Stayman)
        if partner_natural and partner_natural[1] == 'N':
            if partner_natural[0] == 1 and candidate == (2, 'C'):
                result['is_Relay'] = True
                result['is_Artificial'] = True
                result['is_Asking'] = True
            if partner_natural[0] == 2 and candidate == (3, 'C'):
                result['is_Relay'] = True
                result['is_Artificial'] = True
        
        # Checkback
        if level == 2 and strain in ('C', 'D'):
            if partner_bids:
                last_partner = partner_bids[0] if partner_bids else None
                if last_partner == (1, 'N'):
                    our_prior = None
                    for i in range(candidate_seat - 4, -1, -4):
                        if 0 <= i < len(auction_before):
                            our_prior = auction_before[i]
                            break
                    if our_prior and isinstance(our_prior[0], int) and our_prior[0] == 1:
                        result['is_Checkback'] = True
                        result['is_Conventional'] = True
        
        # Drury
        if candidate == (2, 'C') and result['is_PassedHand']:
            if partner_natural and isinstance(partner_natural[0], int):
                if partner_natural[0] == 1 and partner_natural[1] in ('H', 'S'):
                    result['is_Drury'] = True
                    result['is_Conventional'] = True
                    result['is_Alertable'] = True
                    # Drury shows fit in partner's major
                    result['is_Raise'] = True
                    result['is_SupportShowing'] = True
                    result['is_FitEstablished'] = True
        
        # Bergen
        if level == 3 and strain in ('C', 'D'):
            if partner_natural and isinstance(partner_natural[0], int):
                if partner_natural[0] == 1 and partner_natural[1] in ('H', 'S'):
                    result['is_Bergen'] = True
                    result['is_Conventional'] = True
                    result['is_Alertable'] = True
                    # Bergen shows fit in partner's major
                    result['is_Raise'] = True
                    result['is_SupportShowing'] = True
                    result['is_FitEstablished'] = True
        
        # Jacoby 2NT
        if candidate == (2, 'N'):
            if partner_natural and isinstance(partner_natural[0], int):
                if partner_natural[0] == 1 and partner_natural[1] in ('H', 'S'):
                    result['is_Jacoby2NT'] = True
                    result['is_Conventional'] = True
                    result['is_GameForcing'] = True
                    result['is_Alertable'] = True
                    # Jacoby 2NT shows fit in partner's major
                    result['is_Raise'] = True
                    result['is_SupportShowing'] = True
                    result['is_FitEstablished'] = True
        
        # Transfer completion
        if partner_bids:
            last_partner = partner_bids[0] if partner_bids else None
            if last_partner and isinstance(last_partner[0], int):
                p_level, p_strain = last_partner
                if p_level == 2 and p_strain == 'D' and level == 2 and strain == 'H':
                    result['is_TransferCompletion'] = True
                if p_level == 2 and p_strain == 'H' and level == 2 and strain == 'S':
                    result['is_TransferCompletion'] = True
        
        # Super accept
        if partner_bids:
            last_partner = partner_bids[0] if partner_bids else None
            if last_partner and isinstance(last_partner[0], int):
                p_level, p_strain = last_partner
                if p_level == 2 and p_strain == 'D' and level == 3 and strain == 'H':
                    result['is_SuperAccept'] = True
                    result['is_MaximumShowing'] = True
                if p_level == 2 and p_strain == 'H' and level == 3 and strain == 'S':
                    result['is_SuperAccept'] = True
                    result['is_MaximumShowing'] = True
    
    # =========================================================================
    # STRONG BID RESPONSES
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        if partner_natural == (2, 'C'):
            if candidate == (2, 'D'):
                result['is_NegativeResponse'] = True
                result['is_WaitingBid'] = True
                result['is_Relay'] = True
            else:
                result['is_PositiveResponse'] = True
    
    # =========================================================================
    # REVERSE
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        level, strain = candidate
        if level == 2 and result['is_Rebid']:
            our_opening = None
            for i in range(candidate_seat - 4, -1, -4):
                if 0 <= i < len(auction_before):
                    b = auction_before[i]
                    if b and isinstance(b[0], int) and b[0] == 1:
                        our_opening = b
                        break
            if our_opening and isinstance(our_opening[0], int):
                opening_strain = our_opening[1]
                if strain != opening_strain and strain != 'N':
                    if STRAINS.get(strain, 0) > STRAINS.get(opening_strain, 0):
                        result['is_Reverse'] = True
                        result['is_ExtraValues'] = True
    
    # =========================================================================
    # JUMP SHIFT
    # =========================================================================
    if result['is_Jump'] and result['is_NewSuit']:
        result['is_JumpShift'] = True
        if result['is_Rebid']:
            result['is_ExtraValues'] = True
    
    # =========================================================================
    # GAME FORCING (2/1)
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        level, strain = candidate
        if result['is_Response'] and level == 2 and result['is_NewSuit']:
            if partner_natural and isinstance(partner_natural[0], int) and partner_natural[0] == 1:
                result['is_GameForcing'] = True
    
    # =========================================================================
    # NON-FORCING / SEMI-FORCING
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        level, strain = candidate
        
        if result['is_Raise'] and not result['is_Jump']:
            if not result['is_GameForcing']:
                result['is_NonForcing'] = True
                result['is_MinimumShowing'] = True
        
        if result['is_Preference']:
            result['is_NonForcing'] = True
        
        if result['is_SignOff']:
            result['is_NonForcing'] = True
        
        if result['is_Response'] and candidate == (1, 'N'):
            if partner_natural and isinstance(partner_natural[0], int):
                if partner_natural[0] == 1 and partner_natural[1] in ('H', 'S'):
                    result['is_SemiForcing'] = True
                    result['is_LimitBid'] = True
    
    # =========================================================================
    # PARTNERSHIP ROLE
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        if partner_natural and isinstance(partner_natural[0], int):
            if partner_natural[1] == 'N' and partner_natural[0] in (1, 2):
                result['is_Captain'] = True
        if result['is_Asking'] or result['is_KeyCardAsk']:
            result['is_Captain'] = True
        
        if result['is_LimitBid']:
            result['is_Describer'] = True
            result['is_Telling'] = True
    
    # =========================================================================
    # LONG SUIT SHOWING
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        level, strain = candidate
        if result['is_Rebid'] and not result['is_NewSuit']:
            our_first_suit = None
            for i in range(candidate_seat - 4, -1, -4):
                if 0 <= i < len(auction_before):
                    b = auction_before[i]
                    if b and isinstance(b[0], int):
                        our_first_suit = b[1]
                        break
            if our_first_suit and strain == our_first_suit:
                result['is_LongSuitShowing'] = True
    
    # =========================================================================
    # RISK PROFILE
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        level, strain = candidate
        
        if result['is_Raise'] and not result['is_Jump'] and not result['is_Preempt']:
            result['is_Conservative'] = True
        if result['is_Minimum'] and not result['is_Preempt']:
            result['is_Conservative'] = True
        if result['is_LimitBid']:
            result['is_Conservative'] = True
        
        if result['is_Overcall'] and level >= 2:
            result['is_Aggressive'] = True
        if result['is_Overcall'] and result['is_Jump']:
            result['is_Aggressive'] = True
        if level >= 4 and result['is_Competitive']:
            result['is_Aggressive'] = True
        
        # --- PUSH ---
        # A push bid reduces opponent's EV at the risk of reducing bidder's own EV
        # Both sides accept increased risk of suboptimal par score
        # 
        # REQUIRES EV-BASED DETECTION (see SUPERHUMAN_INFRASTRUCTURE.md):
        # - Component A: Historical auction EV (continuous values from historical data)
        # - Component B: Par score probabilities (key cards, shape, lead probability)
        # - Heuristics alone cannot determine if a bid risks negative EV
        #
        # is_Push = False by default (awaiting EV infrastructure)
    
    # =========================================================================
    # VULNERABILITY AWARENESS
    # =========================================================================
    if candidate and isinstance(candidate[0], int):
        level, strain = candidate
        if result['is_Sacrifice']:
            result['is_VulAction'] = True
        if level >= 4 and result['is_Competitive']:
            result['is_VulAction'] = True
    
    if result['is_Preempt']:
        if candidate and isinstance(candidate[0], int):
            level = candidate[0]
            if level >= 4:
                result['is_NonVulPreempt'] = True
    
    # =========================================================================
    # ESCAPE
    # =========================================================================
    if candidate and candidate[0] == 'R':
        if len(auction_before) >= 1 and auction_before[-1] and auction_before[-1][0] == 'D':
            result['is_Escape'] = True
    
    if candidate and isinstance(candidate[0], int):
        if len(auction_before) >= 1:
            last = auction_before[-1]
            if last and last[0] == 'D':
                if result['is_NewSuit']:
                    result['is_Escape'] = True
    
    # =========================================================================
    # HIGH CARD SHOWING
    # =========================================================================
    if result['is_ControlBid'] or result['is_Feature']:
        result['is_HighCard'] = True
    
    return result


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)
    print(f"  Loaded {df.height:,} rows, {df.width} columns")
    
    # Get all category names
    sample_cats = classify_bid("", "1H", 0, 0)
    print(f"  Classifying into {len(sample_cats)} categories")
    
    # Initialize category columns
    cat_columns = {k: [] for k in sample_cats.keys()}
    
    # Process each row
    print("\nClassifying bids...")
    total = df.height
    for i, row in enumerate(df.to_dicts()):
        if i % 10000 == 0:
            print(f"  Processing {i:,}/{total:,} ({100*i/total:.1f}%)")
        
        step_auction = row.get("step_auction", "")
        next_bid = row.get("next_bid", "")
        pos_count = row.get("pos_count", 0)
        neg_count = row.get("neg_count", 0)
        
        cats = classify_bid(step_auction, next_bid, pos_count, neg_count)
        for col in cat_columns:
            cat_columns[col].append(cats.get(col, False))
    
    print(f"  Completed {total:,} rows")
    
    # Add columns to dataframe
    result_df = df
    for col, values in cat_columns.items():
        result_df = result_df.with_columns(pl.Series(col, values))
    
    print(f"\nAdded {len(cat_columns)} bid category columns")
    print(f"Result shape: {result_df.height:,} rows x {result_df.width} columns")
    
    # Show category distribution
    print("\nCategory Distribution (top 20):")
    print("=" * 50)
    cat_counts = []
    for col in sorted(cat_columns.keys()):
        count = result_df.filter(pl.col(col)).height
        if count > 0:
            pct = 100 * count / result_df.height
            cat_counts.append((col, count, pct))
    
    cat_counts.sort(key=lambda x: -x[1])
    for col, count, pct in cat_counts[:20]:
        print(f"  {col:25s}: {count:>8,} ({pct:5.1f}%)")
    
    # Save output
    print(f"\nSaving to {OUTPUT_FILE}...")
    result_df.write_parquet(OUTPUT_FILE)
    print(f"  Saved {result_df.height:,} rows")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
