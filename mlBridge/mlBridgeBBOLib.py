"""
Takes 37m for 22m files. 8m to update an existing data file.
Memory-optimized batch processor that avoids RAM exhaustion by using streaming processing.
This version processes data in chunks and writes incrementally to avoid memory buildup.
"""

import polars as pl
import pathlib
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
from typing import List, Dict, Any, Tuple
import gc
import tempfile
import re

try:
    import endplay.parsers.lin as lin  # type: ignore[import-not-found]
except ImportError:
    print("Warning: endplay not installed. Install with: pip install endplay")
    lin = None  # type: ignore[assignment]
    lin = None


# Removed MemoryMonitor - no longer needed with zero-copy processing


def process_lin_file_batch_streaming(file_batch: List[pathlib.Path]) -> List[Dict[str, Any]]:
    """Process a batch of lin files with filtering applied at batch level for better performance."""
    if lin is None:
        raise ImportError("endplay library is required for processing lin files. Install with: pip install endplay")
    
    records = []
    invalid_count = 0
    total_boards_processed = 0
    files_processed = 0
    
    # Detailed filtering statistics
    filter_stats = {
        'invalid_pbn': 0,
        'null_contract': 0,
        'null_vulnerability': 0,
        'null_dealer': 0,
        'other_invalid': 0
    }
    
    # Track extraction issues
    extraction_issue_counts = {}
    
    for lin_file in file_batch:
        files_processed += 1
        try:
            with open(lin_file, 'r', encoding='utf-8', buffering=65536) as f:
                boards = lin.load(f)
            
            for board in boards:
                total_boards_processed += 1
                record = extract_board_record_optimized(board, lin_file)
                
                # Track extraction issues for debugging
                if 'extraction_issues' in record:
                    for issue in record['extraction_issues']:
                        extraction_issue_counts[issue] = extraction_issue_counts.get(issue, 0) + 1
                
                # Filter invalid records immediately at batch level
                if is_valid_record(record):
                    # Double-check: ensure no null vulnerability records slip through
                    if record.get('vulnerability') is None:
                        print(f"   ERROR: Null vulnerability record passed validation! File: {lin_file}")
                        print(f"   Record: {record}")
                        invalid_count += 1
                        filter_stats['null_vulnerability'] += 1
                    else:
                        records.append(record)
                else:
                    invalid_count += 1
                    # Track detailed filtering reasons
                    if record.get('PBN') == 'N:... ... ... ...':
                        filter_stats['invalid_pbn'] += 1
                    elif record.get('vulnerability') is None:
                        filter_stats['null_vulnerability'] += 1
                    elif record.get('dealer') is None:
                        filter_stats['null_dealer'] += 1
                    else:
                        filter_stats['other_invalid'] += 1
                        # Note: null contracts are now allowed (passout hands)
                
        except Exception as e:
            # Add error record instead of yielding
            records.append({
                'error': True,
                'source_file': str(lin_file),
                'error_message': str(e),
                'board_num': None
            })
    
    # Report detailed filtering stats for this batch
    if invalid_count > 0 or total_boards_processed > 0:
        valid_count = len(records)
        drop_rate = (invalid_count / total_boards_processed * 100) if total_boards_processed > 0 else 0
        
        print(f"   Batch: {files_processed} files, {total_boards_processed} boards -> {valid_count} valid ({drop_rate:.1f}% dropped)")
        
        if invalid_count > 0:
            print(f"   Filtered: {filter_stats['invalid_pbn']} invalid PBN, "
                  f"{filter_stats['null_vulnerability']} null vulnerability, {filter_stats['null_dealer']} null dealer, "
                  f"{filter_stats['other_invalid']} other (passout hands now allowed)")
        
        # Report top extraction issues
        if extraction_issue_counts:
            top_issues = sorted(extraction_issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            issue_str = ', '.join([f"{issue}: {count}" for issue, count in top_issues])
            print(f"   Extraction issues: {issue_str}")
    
    return records


# =============================================================================
# GIB bidding criteria transformation helpers
# =============================================================================

def HCP_Exact(regex, match):
    return f"HCP == {match.group(1)}"


def HCP_At_Most(regex, match):
    return f"HCP <= {match.group(1)}"


def HCP_At_Least(regex, match):
    return f"HCP >= {match.group(1)}"


def HCP_Between(regex, match):
    return [f"HCP >= {match.group(1)}", f"HCP <= {match.group(2)}"]


def Losers_Exact(regex, match):
    return f"Losers == {match.group(1)}"


def Losers_At_Most(regex, match):
    return f"Losers <= {match.group(1)}"


def Losers_At_Least(regex, match):
    return f"Losers >= {match.group(1)}"


def Losers_Between(regex, match):
    return [f"Losers >= {match.group(1)}", f"Losers <= {match.group(2)}"]


def Suit_Length_Exact(regex, match):
    return f"SL_{match.group(2)} == {match.group(1)}"


def Suit_Length_At_Most(regex, match):
    return f"SL_{match.group(2)} <= {match.group(1)}"


def Suit_Length_At_Least(regex, match):
    return f"SL_{match.group(2)} >= {match.group(1)}"


def Suit_Length_Between(regex, match):
    return [f"SL_{match.group(3)} >= {match.group(1)}", f"SL_{match.group(3)} <= {match.group(2)}"]


def Total_Points_Exact(regex, match):
    return f"Total_Points == {match.group(1)}"


def Total_Points_At_Most(regex, match):
    return f"Total_Points <= {match.group(1)}"


def Total_Points_At_Least(regex, match):
    return f"Total_Points >= {match.group(1)}"


def Total_Points_Between(regex, match):
    return [f"Total_Points >= {match.group(1)}", f"Total_Points <= {match.group(2)}"]


def Cards_In_Suit(regex, match):
    return [f"C_{match.group(2)}{c}" for c in match.group(1)]


def Suit_Has_Cards(regex, match):
    return [f"C_{match.group(1)}{c}" for c in match.group(2)]


def Suit_Has_No_Cards(regex, match):
    return [f"C_{match.group(1)}{c} == False" for c in match.group(2)]


def At_Best_Cards_In_Suit(regex, match):
    return [f"SL_{match.group(2)} <= {match.group(1)}"]


def At_Best_Stopper_In_Suit(regex, match):
    return f"At_Best_Stop_In_{match.group(1)}"


def At_Best_Likely_Stopper_In_Suit(regex, match):
    return f"At_Best_Likely_Stop_In_{match.group(1)}"


def At_Best_Partial_Stopper_In_Suit(regex, match):
    return f"At_Best_Partial_Stop_In_{match.group(1)}"


def Biddable_Suit(regex, match):
    return f"Biddable_{match.group(1)}"


def Forcing_One_Round(regex, match):
    return f"Forcing_One_Round"


def Forcing_To(regex, match):
    return f"Forcing_To_{match.group(1)}"


def Likely_Stopper_In_Suit(regex, match):
    return f"Likely_Stop_In_{match.group(1)}"


def No_Cards_In_Suit(regex, match):
    return f"SL_{match.group(1)} == 0"


def Cards_Not_In_Suit(regex, match):
    return [f"C_{match.group(1)}{c} == False" for c in match.group(2)]


def Opponents_Cannot_Play_Undoubled_Below_Bid(regex, match):
    return f"Opponents_Cannot_Play_Undoubled_Below_{match.group(1)}"


def Partial_Stopper_In_Suit(regex, match):
    return f"Partial_Stop_In_{match.group(1)}"


def Partial_Stop_Likely_Stop_In_Suit(regex, match):
    return f"Partial_Stop_Likely_Stop_In_{match.group(1)}"


def Rebiddable_Suit(regex, match):
    return f"Rebiddable_{match.group(1)}"


def Solid_SL_Suit(regex, match):
    return [f"Solid_{match.group(2)}", f"SL_{match.group(2)} >= {match.group(1)}"]


def Stopper_In_Suit(regex, match):
    return f"Stop_In_{match.group(1)}"


def Strong_Rebiddable_Suit(regex, match):
    return f"Strong_Rebiddable_{match.group(1)}"


def Twice_Rebiddable_Suit(regex, match):
    return f"Twice_Rebiddable_{match.group(1)}"


def Two_Stoppers_In_Suit(regex, match):
    return f"Two_Stops_In_{match.group(1)}"


GIB_EXPR_REGEX = [
    (r"^(\d+)$", HCP_Exact),
    (r"^(\d+)-$", HCP_At_Most),
    (r"^(\d+)\+$", HCP_At_Least),
    (r"^(\d+)-(\d+)$", HCP_Between),
    (r"^(\d+) losers$", Losers_Exact),
    (r"^(\d+)- losers$", Losers_At_Most),
    (r"^(\d+)\+ losers$", Losers_At_Least),
    (r"^(\d+)-(\d+) losers$", Losers_Between),
    (r"^(\d+) !([CDHS])$", Suit_Length_Exact),
    (r"^(\d+)- !([CDHS])$", Suit_Length_At_Most),
    (r"^(\d+)-card !([CDHS])$", Suit_Length_Exact),
    (r"^(\d+)\+ !([CDHS])$", Suit_Length_At_Least),
    (r"^(\d+)-(\d+) !([CDHS])$", Suit_Length_Between),
    (r"^(\d+) HCP$", HCP_Exact),
    (r"^(\d+)- HCP$", HCP_At_Most),
    (r"^(\d+)\+ HCP$", HCP_At_Least),
    (r"^(\d+)-(\d+) HCP$", HCP_Between),
    (r"^(\d+) total_points$", Total_Points_Exact),
    (r"^(\d+)- total_points$", Total_Points_At_Most),
    (r"^(\d+)\+ total_points$", Total_Points_At_Least),
    (r"^(\d+)-(\d+) total_points$", Total_Points_Between),
    (r"^([AKQ]+)\+_in !([CDHS])$", Cards_In_Suit),
    (r"^!?([CDHS])([AKQ]+)$", Suit_Has_Cards),
    #(r"^!([CDHS])([AKQ]+)$", Suit_Has_No_Cards), # !HA -> C_HA, not C_HA == False
    (r"^at_best (\d+)-card !([CDHS])$", At_Best_Cards_In_Suit),
    (r"^at_best_stop_in !([CDHS])$", At_Best_Stopper_In_Suit),
    (r"^at_best_likely_stop_in !([CDHS])$", At_Best_Likely_Stopper_In_Suit),
    (r"^at_best_partial_stop_in !([CDHS])$", At_Best_Partial_Stopper_In_Suit),
    (r"^biddable !([CDHS])$", Biddable_Suit),
    (r"^forcing$", Forcing_One_Round),
    (r"^forcing_to (\d[CDHSN])$", Forcing_To),
    (r"^likely_stop(:?-stop)_in !([CDHS])$", Likely_Stopper_In_Suit),
    (r"^no !([CDHS])$", No_Cards_In_Suit),
    (r"^no !([CDHS])([AKQ]+)$", Cards_Not_In_Suit),
    (r"^opponents_cannot_play_undoubled_below (\d[CDHSN])$", Opponents_Cannot_Play_Undoubled_Below_Bid),
    (r"^partial_stop(:?-stop)_in !([CDHS])$", Partial_Stopper_In_Suit),
    (r"^partial_stop[-_]likely_stop_in !([CDHS])$", Partial_Stop_Likely_Stop_In_Suit),
    (r"^rebiddable !([CDHS])$", Rebiddable_Suit),
    (r"^solid (\d)-card !([CDHS])$", Solid_SL_Suit),
    (r"^stop_in !([CDHS])$", Stopper_In_Suit),
    (r"^strong_rebiddable !([CDHS])$", Strong_Rebiddable_Suit),
    (r"^twice_rebiddable !([CDHS])$", Twice_Rebiddable_Suit),
    (r"^two_stops_in !([CDHS])$", Two_Stoppers_In_Suit),
]


def transform_gib_expression(sub_expr: str) -> List[str]:
    """Transform a GIB expression into eval expression(s).
    
    Args:
        sub_expr: The GIB expression string to transform
        
    Returns:
        List of eval expression strings, or empty list if no match
    """
    for regex, func in GIB_EXPR_REGEX:
        match = re.match(regex, sub_expr)
        if match:
            result = func(regex, match)
            if result is None:
                return []
            if isinstance(result, str):
                return [result]
            return list(result)
    return []


def transform_gib_expression_with_details(sub_expr: str) -> Tuple[str | None, tuple | None, List[str]]:
    """Transform a GIB expression and return detailed match information.
    
    Args:
        sub_expr: The GIB expression string to transform
        
    Returns:
        Tuple of (regex_pattern, match_groups, eval_expressions)
        Returns (None, None, []) if no match found
    """
    for regex, func in GIB_EXPR_REGEX:
        match = re.match(regex, sub_expr)
        if match:
            result = func(regex, match)
            if result is None:
                return (regex, match.groups(), [])
            if isinstance(result, str):
                return (regex, match.groups(), [result])
            return (regex, match.groups(), list(result))
    return (None, None, [])


def transform_gib_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    sub_expr = ' '.join(tokens)
    return transform_gib_expression(sub_expr)


def create_gib_expressions(keyword: str, raw_values: List[Any]) -> List[str]:
    tokens = [keyword]
    if raw_values:
        first = raw_values[0]
        if isinstance(first, list):
            tokens.extend(first)
        else:
            tokens.extend(raw_values)
    return transform_gib_tokens(tokens)


def is_valid_record(record: Dict[str, Any]) -> bool:
    """Check if a record is valid and should be kept."""
    vulnerability = record.get('vulnerability')
    # Allow passout hands (contract can be None) - they are valid bridge outcomes
    return (
        record.get('PBN') != 'N:... ... ... ...' and
        vulnerability is not None and isinstance(vulnerability, int) and 0 <= vulnerability <= 3 and
        record.get('dealer') is not None
    )


def extract_board_record_optimized(board, source_file: pathlib.Path) -> Dict[str, Any]:
    """Memory-optimized board record extraction."""
    # Pre-compute common values to avoid repeated calculations
    has_contract = board._contract and not board._contract.is_passout()
    
    # Track extraction issues for debugging
    extraction_issues = []
    
    # Process bidding more efficiently with list comprehensions
    bid_data = []
    for bid in board.auction:
        if hasattr(bid, 'denom'):
            bid_data.append({
                'type': 'Contract',
                'denom': bid.denom.name,
                'penalty': None,
                'level': bid.level,
                'alertable': bid.alertable,
                'announcement': bid.announcement
            })
        else:
            bid_data.append({
                'type': 'Penalty',
                'denom': None,
                'penalty': bid.penalty.name,
                'level': None,
                'alertable': bid.alertable,
                'announcement': bid.announcement
            })
    
    # Extract player names efficiently
    player_names = {}
    if hasattr(board, 'info') and board.info:
        for position in ['North', 'East', 'South', 'West']:
            if hasattr(board.info, position):
                player_names[f'Player_{position[0]}'] = str(getattr(board.info, position))
    
    # Build record with defensive extraction and error tracking
    try:
        # Defensive dealer extraction
        dealer = None
        if hasattr(board, 'dealer') and board.dealer:
            if hasattr(board.dealer, 'abbr'):
                dealer = board.dealer.abbr
            else:
                extraction_issues.append('dealer_no_abbr')
        else:
            extraction_issues.append('no_dealer')
        
        # Defensive vulnerability extraction
        vulnerability = None
        if hasattr(board, '_vul'):
            if isinstance(board._vul, int) and not isinstance(board._vul, bool) and 0 <= board._vul <= 3:
                vulnerability = board._vul
            else:
                extraction_issues.append(f'invalid_vul_{type(board._vul).__name__}_{board._vul}')
        else:
            extraction_issues.append('no_vul_attr')
        
        # Defensive PBN extraction
        pbn = 'N:... ... ... ...'  # Default invalid PBN
        try:
            if hasattr(board, 'deal') and board.deal:
                pbn = str(board.deal.to_pbn())
            else:
                extraction_issues.append('no_deal')
        except Exception as e:
            extraction_issues.append(f'pbn_error_{str(e)[:20]}')
        
        record = {
            'board_num': getattr(board, 'board_num', None),
            'dealer': dealer,
            'vulnerability': vulnerability,
            'passout': board._contract.is_passout() if board._contract else None,
            'contract': str(board._contract) if board._contract else None,
            'level': board._contract.level if has_contract else None,
            'denom': board._contract.denom.name if has_contract else None,
            'trump': board.deal.trump.name if hasattr(board, 'deal') and hasattr(board.deal, 'trump') else None,
            'penalty': board._contract.penalty.name if has_contract else None,
            'declarer': board._contract.declarer.name if has_contract else None,
            'result': board._contract.result if has_contract else None,
            'score': board._contract.score(board._vul) if has_contract and vulnerability is not None else None,
            'claimed': getattr(board, 'claimed', None),
            'PBN': pbn,
            'Hand_N': str(board.deal.north) if hasattr(board, 'deal') else '',
            'Hand_E': str(board.deal.east) if hasattr(board, 'deal') else '',
            'Hand_S': str(board.deal.south) if hasattr(board, 'deal') else '',
            'Hand_W': str(board.deal.west) if hasattr(board, 'deal') else '',
            'info': getattr(board, 'info', None),
            'source_file': str(source_file),
            # Unpack bid data
            'bid_type': [b['type'] for b in bid_data],
            'bid_denom': [b['denom'] for b in bid_data],
            'bid_penalty': [b['penalty'] for b in bid_data],
            'bid_level': [b['level'] for b in bid_data],
            'bid_alertable': [b['alertable'] for b in bid_data],
            'bid_announcement': [b['announcement'] for b in bid_data],
            'play_rank': [play.rank.name for play in board.play] if hasattr(board, 'play') else [],
            'play_suit': [play.suit.name for play in board.play] if hasattr(board, 'play') else [],
        }
        
        # Add extraction issues for debugging
        if extraction_issues:
            record['extraction_issues'] = extraction_issues
            
    except Exception as e:
        # Fallback record for completely broken boards - ensure it gets filtered out
        record = {
            'board_num': None,
            'dealer': None,  # This will cause filtering
            'vulnerability': None,  # This will cause filtering  
            'passout': None,
            'contract': None,
            'PBN': 'N:... ... ... ...',  # This will cause filtering
            'source_file': str(source_file),
            'extraction_error': str(e),
            'extraction_issues': extraction_issues + [f'total_failure_{str(e)[:30]}'],
            'Hand_N': '',
            'Hand_E': '',
            'Hand_S': '',
            'Hand_W': '',
            'info': None,
            'bid_type': [],
            'bid_denom': [],
            'bid_penalty': [],
            'bid_level': [],
            'bid_alertable': [],
            'bid_announcement': [],
            'play_rank': [],
            'play_suit': [],
        }
    
    # Add player names
    record.update(player_names)
    
    return record


def write_batches_to_parquet(batch_list: List[List[Dict]], temp_dir: pathlib.Path, chunk_id: int) -> pathlib.Path | None:
    """Write multiple batches to a parquet file without expensive memory operations."""
    if not batch_list or not any(batch_list):
        return None
        
    chunk_file = temp_dir / f"chunk_{chunk_id:06d}.parquet"
    
    # Flatten batches efficiently using itertools
    from itertools import chain
    all_records = list(chain.from_iterable(batch_list))
    
    df_chunk = pl.DataFrame(all_records)
    df_chunk.write_parquet(chunk_file)
    
    # Clear memory
    del df_chunk
    del all_records
    del batch_list
    gc.collect()
    
    return chunk_file


def align_dataframe_schemas(df1: pl.DataFrame, df2: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Align two DataFrames to have the same schema for concatenation."""
    # Get all columns from both DataFrames
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    all_cols = cols1.union(cols2)
    
    print(f"     Schema alignment: df1 has {len(cols1)} cols, df2 has {len(cols2)} cols")
    missing_in_df1 = cols2 - cols1
    missing_in_df2 = cols1 - cols2
    
    if missing_in_df1:
        print(f"     Adding to df1: {missing_in_df1}")
    if missing_in_df2:
        print(f"     Adding to df2: {missing_in_df2}")
    
    # Add missing columns to df1
    for col in missing_in_df1:
        # CRITICAL: Never add null vulnerability columns - this would bypass all our filtering!
        if col == 'vulnerability':
            print(f"   ERROR: Attempting to add null vulnerability column to df1 - this should never happen!")
            raise ValueError("Schema alignment tried to add null vulnerability column")
        # Infer the data type from df2
        dtype = df2[col].dtype
        df1 = df1.with_columns(pl.lit(None).cast(dtype).alias(col))
    
    # Add missing columns to df2
    for col in missing_in_df2:
        # CRITICAL: Never add null vulnerability columns - this would bypass all our filtering!
        if col == 'vulnerability':
            print(f"   ERROR: Attempting to add null vulnerability column to df2 - this should never happen!")
            raise ValueError("Schema alignment tried to add null vulnerability column")
        # Infer the data type from df1
        dtype = df1[col].dtype
        df2 = df2.with_columns(pl.lit(None).cast(dtype).alias(col))
    
    # Ensure both DataFrames have columns in the same order
    sorted_cols = sorted(all_cols)
    df1 = df1.select(sorted_cols)
    df2 = df2.select(sorted_cols)
    
    print(f"     After alignment: df1 has {len(df1.columns)} cols, df2 has {len(df2.columns)} cols")
    
    return df1, df2


def process_chunk_batch(batch_info: Tuple[List[pathlib.Path], int, pathlib.Path]) -> pathlib.Path | None:
    """Process a batch of chunk files in parallel."""
    batch_files, batch_id, temp_path = batch_info
    
    # Read batch of files
    dfs = []
    for chunk_file in batch_files:
        if chunk_file and chunk_file.exists():
            df = pl.read_parquet(chunk_file)
            
            # Filter out error records and drop error columns if they exist
            if 'error' in df.columns:
                error_count = df.filter(pl.col('error') == True).height
                if error_count > 0:
                    print(f"     Batch {batch_id}: Filtering out {error_count} error records from {chunk_file.name}")
                df = df.filter(pl.col('error') != True)
                df = df.drop(['error', 'error_message'])
            
            dfs.append(df)
    
    if not dfs:
        return None
    
    # Align schemas before concatenation
    if len(dfs) > 1:
        # Always ensure consistent column ordering
        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        
        # Always align to ensure consistent column order
        sorted_cols = sorted(all_columns)
        aligned_dfs = []
        for j, df in enumerate(dfs):
            missing_cols = all_columns - set(df.columns)
            if missing_cols:
                print(f"     Batch {batch_id}: Adding {len(missing_cols)} columns to DataFrame {j}: {missing_cols}")
                for col in missing_cols:
                    # CRITICAL: Never add null vulnerability columns - this would bypass all our filtering!
                    if col == 'vulnerability':
                        print(f"   ERROR: Attempting to add null vulnerability column in batch {batch_id} - this should never happen!")
                        raise ValueError("Schema alignment tried to add null vulnerability column")
                    df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
            
            # Always reorder columns to ensure consistency
            df = df.select(sorted_cols)
            aligned_dfs.append(df)
        
        batch_df = pl.concat(aligned_dfs)
    else:
        batch_df = dfs[0]
    
    # Write merged batch to temp file
    temp_merged_file = temp_path / f"merged_batch_{batch_id:03d}.parquet"
    batch_df.write_parquet(temp_merged_file)
    
    print(f"     Batch {batch_id}: Processed {len(batch_files)} chunks -> {batch_df.shape}")
    
    # Clear memory
    del dfs, batch_df
    import gc
    gc.collect()
    
    return temp_merged_file


def merge_parquet_files_streaming(chunk_files: List[pathlib.Path], 
                                output_file: pathlib.Path,
                                existing_file: pathlib.Path | None = None,
                                max_workers: int | None = None) -> pl.DataFrame:
    """Merge parquet files using parallel processing to maximize CPU usage."""
    print(f"   Merging {len(chunk_files)} chunk files with parallel processing...")
    
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 1))  # Use up to 8 workers for I/O bound tasks
    
    # Read and concatenate in smaller batches to avoid memory issues
    batch_size = 10  # Process 10 files at a time
    temp_merged_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        
        # Create batch jobs for parallel processing
        batch_jobs = []
        for i in range(0, len(chunk_files), batch_size):
            batch_files = chunk_files[i:i + batch_size]
            batch_id = i // batch_size + 1
            batch_jobs.append((batch_files, batch_id, temp_path))
        
        print(f"   Processing {len(batch_jobs)} batches in parallel with {max_workers} workers...")
        
        # Process batches in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            futures = [executor.submit(process_chunk_batch, job) for job in batch_jobs]
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    temp_file = future.result()
                    if temp_file:
                        temp_merged_files.append(temp_file)
                    completed += 1
                    if completed % 5 == 0 or completed == len(futures):
                        print(f"   Completed {completed}/{len(futures)} batches")
                except Exception as e:
                    print(f"   Error in batch processing: {e}")
                    completed += 1
        
        # Final merge of batch files
        if temp_merged_files:
            print("   Final merge of batched files...")
            final_dfs = []
            for f in temp_merged_files:
                df = pl.read_parquet(f)
                
                # Filter out error records and drop error columns if they exist
                if 'error' in df.columns:
                    error_count = df.filter(pl.col('error') == True).height
                    if error_count > 0:
                        print(f"   Filtering out {error_count} error records from final merge")
                    df = df.filter(pl.col('error') != True)
                    df = df.drop(['error', 'error_message'])
                
                final_dfs.append(df)
            
            # Check schema consistency for final merge
            if len(final_dfs) > 1:
                print(f"   Checking schema consistency for {len(final_dfs)} final DataFrames...")
                
                # Always ensure consistent column ordering for final merge
                all_columns = set()
                for df in final_dfs:
                    all_columns.update(df.columns)
                
                # Check for column name and order consistency
                column_sets = [set(df.columns) for df in final_dfs]
                column_orders = [list(df.columns) for df in final_dfs]
                
                if len(set(tuple(sorted(cols)) for cols in column_sets)) > 1:
                    print(f"   WARNING: Different column sets in final merge, applying alignment...")
                elif len(set(tuple(cols) for cols in column_orders)) > 1:
                    print(f"   WARNING: Different column orders in final merge, standardizing order...")
                else:
                    print(f"   All final DataFrames have consistent schema and order ({len(final_dfs[0].columns)} columns)")
                
                # Always align to ensure consistent column order
                sorted_cols = sorted(all_columns)
                aligned_final_dfs = []
                for i, df in enumerate(final_dfs):
                    missing_cols = all_columns - set(df.columns)
                    if missing_cols:
                        print(f"   Adding {len(missing_cols)} columns to final DataFrame {i}: {missing_cols}")
                        for col in missing_cols:
                            # CRITICAL: Never add null vulnerability columns - this would bypass all our filtering!
                            if col == 'vulnerability':
                                print(f"   ERROR: Attempting to add null vulnerability column in final merge - this should never happen!")
                                raise ValueError("Schema alignment tried to add null vulnerability column")
                            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
                    
                    # Always reorder columns to ensure consistency
                    df = df.select(sorted_cols)
                    aligned_final_dfs.append(df)
                
                print(f"   Concatenating {len(aligned_final_dfs)} aligned DataFrames...")
                
                # Final safety check before concatenation
                column_counts = [df.shape[1] for df in aligned_final_dfs]
                if len(set(column_counts)) > 1:
                    print(f"   ERROR: Final column count mismatch: {column_counts}")
                    for i, df in enumerate(aligned_final_dfs):
                        print(f"     DataFrame {i}: {df.shape[1]} cols - {list(df.columns)[:5]}...")
                    raise ValueError("Final schema alignment failed")
                
                final_df = pl.concat(aligned_final_dfs)
                print(f"   Final concatenated DataFrame: {final_df.shape}")
            else:
                final_df = final_dfs[0]
                print(f"   Single DataFrame: {final_df.shape}")
            
            # Merge with existing data if needed
            if existing_file and existing_file.exists():
                print("   Merging with existing data...")
                existing_df = pl.read_parquet(existing_file)
                
                # Filter existing data to remove invalid records (including null vulnerabilities)
                print(f"   Existing data before filtering: {existing_df.shape}")
                
                # Debug: Check vulnerability column in existing data
                null_vul_count = existing_df['vulnerability'].null_count()
                vul_dtype = existing_df['vulnerability'].dtype
                print(f"   DEBUG: Existing data null vulnerability count: {null_vul_count}")
                print(f"   DEBUG: Existing data vulnerability dtype: {vul_dtype}")
                
                # Apply the same validation logic to existing data (allow passout hands)
                existing_df = existing_df.filter(
                    (pl.col('PBN') != 'N:... ... ... ...') &
                    (pl.col('vulnerability').is_not_null()) &
                    (pl.col('vulnerability').is_between(0, 3, closed="both")) &
                    (pl.col('dealer').is_not_null())
                )
                
                print(f"   Existing data after filtering: {existing_df.shape}")
                
                # Debug: Check vulnerability column after filtering
                null_vul_count_after = existing_df['vulnerability'].null_count()
                print(f"   DEBUG: After filtering null vulnerability count: {null_vul_count_after}")
                
                # Align schemas before merging with existing data
                existing_aligned, final_aligned = align_dataframe_schemas(existing_df, final_df)
                
                print(f"   Schema alignment: existing {existing_df.shape[1]} cols -> {existing_aligned.shape[1]} cols")
                print(f"   Schema alignment: new {final_df.shape[1]} cols -> {final_aligned.shape[1]} cols")
                
                final_df = pl.concat([existing_aligned, final_aligned])
                del existing_df, existing_aligned, final_aligned
                gc.collect()
            
            # Debug: Check final result for null vulnerabilities
            final_null_vul_count = final_df['vulnerability'].null_count()
            print(f"   DEBUG: Final DataFrame null vulnerability count: {final_null_vul_count}")
            if final_null_vul_count > 0:
                print(f"   WARNING: {final_null_vul_count} null vulnerability values found in final result!")
                # Sample some null vulnerability records for debugging
                null_sample = final_df.filter(pl.col('vulnerability').is_null()).head(3)
                print(f"   DEBUG: Sample null vulnerability records:")
                print(null_sample.select(['source_file', 'board_num', 'vulnerability', 'dealer', 'contract', 'PBN']))
                
                # CRITICAL FIX: Remove all null vulnerability records before writing
                print(f"   FIXING: Removing {final_null_vul_count} null vulnerability records from final result")
                final_df = final_df.filter(
                    (pl.col('vulnerability').is_not_null()) &
                    (pl.col('dealer').is_not_null()) &
                    (pl.col('PBN') != 'N:... ... ... ...')
                )
                print(f"   FIXED: Final DataFrame after cleanup: {final_df.shape}")
                final_null_vul_count_after = final_df['vulnerability'].null_count()
                print(f"   VERIFIED: Null vulnerability count after cleanup: {final_null_vul_count_after}")
            
            # Write final result
            print(f"   Writing final result to {output_file}")
            final_df.write_parquet(output_file)
            
            return final_df
    
    return pl.DataFrame()


def convert_lin_files_memory_optimized(lin_files_l: List[pathlib.Path],
                                     output_parquet: pathlib.Path,
                                     max_cpu_workers: int | None = None,
                                     base_batch_size: int | None = None,
                                     chunk_size: int = 50000,
                                     use_incremental: bool = True) -> pl.DataFrame:
    """
    Memory-optimized conversion that processes data in chunks to avoid RAM exhaustion.
    
    Args:
        lin_files_l: List of .lin file paths
        output_parquet: Output Parquet file path
        max_cpu_workers: Maximum CPU workers (None for auto-detect)
        base_batch_size: Base files per batch (None for auto-detect)
        chunk_size: Records per chunk for streaming processing
        use_incremental: Whether to use incremental processing
    
    Returns:
        Processed DataFrame
    """
    
    # Simplified settings - no memory monitoring needed
    if max_cpu_workers is None:
        import os
        if os.name == 'nt':  # Windows
            system_limit = 61
        else:  # Linux/Mac
            system_limit = min(mp.cpu_count() * 4, 128)
        
        optimal_workers = mp.cpu_count() * 2  # 2x cores for maximum throughput
        max_cpu_workers = min(optimal_workers, system_limit)
    
    if base_batch_size is None:
        # Simple fixed batch size - no adaptive sizing needed
        base_batch_size = 1000  # Fixed optimal size
    
    print(f"🧠 Memory-Optimized Processing Configuration:")
    print(f"   CPU Workers: {max_cpu_workers}")
    print(f"   Base Batch Size: {base_batch_size:,}")
    print(f"   Chunk Size: {chunk_size:,}")
    print(f"   Files to process: {len(lin_files_l):,}")
    
    # Handle incremental processing
    files_to_process = lin_files_l
    existing_file = None
    
    if use_incremental and output_parquet.exists():
        try:
            existing_df = pl.read_parquet(output_parquet, columns=['source_file'])
            processed_files = set(existing_df['source_file'].unique())
            files_to_process = [f for f in lin_files_l if str(f) not in processed_files]
            existing_file = output_parquet
            print(f"   Incremental: {len(files_to_process):,} new files (of {len(lin_files_l):,} total)")
            del existing_df  # Free memory immediately
            gc.collect()
        except Exception as e:
            print(f"   Could not load existing data: {e}")
            files_to_process = lin_files_l
    
    if not files_to_process:
        print("   No files need processing")
        return pl.read_parquet(output_parquet) if output_parquet.exists() else pl.DataFrame()
    
    # Create temporary directory for chunk files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        chunk_files = []
        
        # Process files in adaptive batches
        current_batch_size = base_batch_size
        chunk_id = 0
        current_chunk_records = []
        
        start_time = time.time()
        total_records = 0
        total_files_processed = 0
        total_boards_processed = 0
        
        # Dynamic work distribution for optimal load balancing
        # Create smaller batches to enable better work redistribution
        optimal_batch_size = 50  # Smaller batches for better load balancing
        
        # Process batches with streaming and concurrent result collection
        # Use separate executor for I/O operations to prevent blocking
        with ProcessPoolExecutor(max_workers=max_cpu_workers) as cpu_executor, \
             ProcessPoolExecutor(max_workers=4) as io_executor:  # Separate I/O workers
            
            # Dynamic work distribution using a work queue approach
            from concurrent.futures import as_completed
            import queue
            import threading
            
            # Create work queue with adaptive batch sizing for better load balancing
            work_queue = queue.Queue()
            
            # Use adaptive batch sizing - smaller batches at the end for better load balancing
            total_files = len(files_to_process)
            for i in range(0, total_files, optimal_batch_size):
                remaining_files = total_files - i
                
                # Use smaller batch sizes when approaching the end to improve load balancing
                if remaining_files < max_cpu_workers * optimal_batch_size:
                    # For the final batches, use even smaller sizes to maximize parallelism
                    adaptive_batch_size = max(10, remaining_files // (max_cpu_workers * 2))
                    adaptive_batch_size = min(adaptive_batch_size, optimal_batch_size)
                else:
                    adaptive_batch_size = optimal_batch_size
                
                batch = files_to_process[i:i + adaptive_batch_size]
                work_queue.put(batch)
            
            total_batches = work_queue.qsize()
            print(f"   Created {total_batches} batches (adaptive sizing: {optimal_batch_size} -> 10 files) with dynamic distribution")
            
            # Submit initial batch of work (2x workers to keep queue full)
            initial_submissions = min(max_cpu_workers * 2, total_batches)
            futures = {}
            
            for _ in range(initial_submissions):
                if not work_queue.empty():
                    batch = work_queue.get()
                    future = cpu_executor.submit(process_lin_file_batch_streaming, batch)
                    futures[future] = len(batch)
            
            completed_count = 0
            io_futures = []  # Track I/O operations
            batch_buffers = []  # Buffer for batch results
            
            # Stream results and dynamically submit new work as workers become available
            while futures:
                # Wait for at least one future to complete (no timeout to avoid TimeoutError)
                done_futures = []
                for future in as_completed(futures.keys()):
                    done_futures.append(future)
                    break  # Process one at a time for better responsiveness
                
                for future in done_futures:
                    try:
                        # Get results as they complete (concurrent, not sequential)
                        batch_records = future.result()
                        batch_buffers.append(batch_records)  # Just append the batch, no extend
                        total_records += len(batch_records)
                        completed_count += 1
                        
                        # Remove completed future
                        del futures[future]
                        
                        # Submit new work if available (dynamic work distribution)
                        if not work_queue.empty():
                            new_batch = work_queue.get()
                            new_future = cpu_executor.submit(process_lin_file_batch_streaming, new_batch)
                            futures[new_future] = len(new_batch)
                        
                        # Calculate total records in buffers
                        buffer_record_count = sum(len(batch) for batch in batch_buffers)
                        
                        # Write chunk when buffer reaches chunk_size (asynchronously)
                        if buffer_record_count >= chunk_size:
                            # Submit chunk writing to I/O executor (non-blocking)
                            # Pass the batch buffers directly, no copying needed
                            io_future = io_executor.submit(
                                write_batches_to_parquet,  # New function that handles batch list
                                batch_buffers,  # Pass list of batches directly
                                temp_path, 
                                chunk_id
                            )
                            io_futures.append(io_future)
                            
                            chunk_id += 1
                            batch_buffers = []  # Reset buffer
                            
                            # Clean up completed I/O operations (non-blocking check)
                            completed_io = [f for f in io_futures if f.done()]
                            for io_f in completed_io:
                                try:
                                    chunk_file = io_f.result()
                                    if chunk_file:
                                        chunk_files.append(chunk_file)
                                except Exception as e:
                                    print(f"   I/O error: {e}")
                            io_futures = [f for f in io_futures if not f.done()]
                        
                        # Progress reporting with dynamic queue status
                        if completed_count % 10 == 0:  # Less frequent reporting
                            elapsed = time.time() - start_time
                            rate = total_records / elapsed if elapsed > 0 else 0
                            remaining_batches = work_queue.qsize()
                            active_workers = len(futures)
                            print(f"   Batch {completed_count:3d}/{total_batches}: {total_records:8,} records "
                                  f"({rate:6.0f} rec/sec) [{len(io_futures)} I/O pending, {active_workers} active, {remaining_batches} queued]")
                        
                    except Exception as e:
                        print(f"   Error in batch {completed_count}: {e}")
                        # Remove failed future
                        if future in futures:
                            del futures[future]
            
            # Wait for all I/O operations to complete
            print("   Waiting for remaining I/O operations...")
            for io_future in io_futures:
                try:
                    chunk_file = io_future.result()
                    if chunk_file:
                        chunk_files.append(chunk_file)
                except Exception as e:
                    print(f"   Final I/O error: {e}")
        
        # Write final chunk if any batch buffers remain
        if batch_buffers:
            chunk_file = write_batches_to_parquet(
                batch_buffers, temp_path, chunk_id
            )
            if chunk_file:
                chunk_files.append(chunk_file)
        
        # Merge all chunks using parallel streaming
        if chunk_files:
            final_df = merge_parquet_files_streaming(
                chunk_files, output_parquet, existing_file, max_workers=max_cpu_workers
            )
            
            elapsed = time.time() - start_time
            rate = total_records / elapsed
            file_size = output_parquet.stat().st_size
            
            print(f"\n🎉 Processing Complete:")
            print(f"   Files processed: {len(files_to_process):,}")
            print(f"   Records: {total_records:,} new, {final_df.shape[0]:,} total")
            print(f"   Time: {elapsed:.1f}s ({rate:.0f} records/sec)")
            print(f"   Output: {file_size/1e9:.1f}GB")
            
            # Calculate file-to-record ratio
            if len(files_to_process) > 0:
                records_per_file = total_records / len(files_to_process)
                print(f"   Ratio: {records_per_file:.2f} records per file")
                if records_per_file < 0.3:
                    print(f"   ⚠️  WARNING: Very low records per file ratio - check filtering logic!")
            
            return final_df
        
        else:
            print("   No records processed")
            return pl.read_parquet(output_parquet) if output_parquet.exists() else pl.DataFrame()


if __name__ == "__main__":
    print("🧠 Memory-Optimized BBO Lin Processor")
    
    # System info
    memory_info = psutil.virtual_memory()
    cpu_cores = mp.cpu_count()
    
    print(f"\nSystem Info:")
    print(f"   CPU Cores: {cpu_cores}")
    print(f"   Memory: {memory_info.total/1e9:.1f}GB total, {memory_info.available/1e9:.1f}GB available")
    print(f"   Memory Usage: {memory_info.percent:.1f}%")
    
    # Conservative recommendations for memory-constrained systems
    recommended_workers = min(cpu_cores, 8)
    recommended_batch = max(100, min(int(memory_info.available/1e9 * 50), 5000))
    
    print(f"\nMemory-Optimized Settings:")
    print(f"   CPU Workers: {recommended_workers} (conservative)")
    print(f"   Batch Size: {recommended_batch:,} (adaptive)")
    print(f"   Chunk Size: 50,000 (streaming)")
    print(f"   Expected Memory: <70% RAM usage")
