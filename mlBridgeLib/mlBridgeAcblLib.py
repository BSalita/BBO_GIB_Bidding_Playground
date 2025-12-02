# this file confunctions which are specific to acbl; downloading acbl webpages, api calls.

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # or DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def print_to_log_info(*args):
    print_to_log(logging.INFO, *args)
def print_to_log_debug(*args):
    print_to_log(logging.DEBUG, *args)
def print_to_log(level, *args):
    logging.log(level, ' '.join(str(arg) for arg in args))

import pandas as pd
import polars as pl
import re
import traceback
import requests
from bs4 import BeautifulSoup
from io import StringIO
import urllib
from collections import defaultdict
import time
import json
import pathlib
import sys
import asyncio
import threading
import sqlalchemy
from sqlalchemy import create_engine, inspect
import sqlalchemy_utils
from sqlalchemy_utils.functions import database_exists, create_database
from typing import List, Optional
sys.path.append(str(pathlib.Path.cwd().parent.joinpath('mlBridgeLib'))) # removed .parent
sys.path

from mlBridgeLib.mlBridgeLib import (
    json_to_sql_walk,
    CreateSqlFile,
    ContractToScores,
    Direction_to_NESW_d,
    brs_to_pbn,
    Vulnerability_to_Vul_d,
    vul_sym_to_index_d,
    BoardNumberToVul,
    hrs_to_brss,
)


def get_club_results(cns, base_url, acbl_url, acblPath, read_local):
    htmls = {}
    total_clubs = len(cns)
    failed_urls = []
    #headers = {"user-agent":None} # Not sure why this has become necessary
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    for ncn,cn in enumerate(sorted(cns)):
        ncn += 1
        url = base_url+str(cn)+'/'
        file = url.replace(acbl_url,'')+str(cn)+'.html'
        print_to_log_info(f'Processing file ({ncn}/{total_clubs}): {file}')
        path = acblPath.joinpath(file)
        if read_local and path.exists() and path.stat().st_size > 200:
            html = path.read_text(encoding="utf-8")
            print_to_log_info(f'Reading local {file}: len={len(html)}')
        else:
            print_to_log_info(f'Requesting {url}')
            try:
                r = requests.get(url,headers=headers)
            except:
                print_to_log_info(f'Except: status:{r.status_code} {url}')
            else:
                html = r.text
                print_to_log_info(f'Creating {file}: len={len(html)}')
            if r.status_code != 200:
                print_to_log_info(f'Error: status:{r.status_code} {url}')
                time.sleep(250) # not sure what the minimum sleep time is after a 403. Need to sleep 60 seconds and return 10 times?
                failed_urls.append(url)
                continue
            # pathlib.Path.mkdir(path.parent, parents=True, exist_ok=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            time.sleep(5) # need to self-throttle otherwise acbl returns 403 "forbidden". Minimum is 5 to 7 seconds?
        htmls[str(cn)] = html
    print_to_log_info(f'Failed Urls: len:{len(failed_urls)} Urls:{failed_urls}')
    print_to_log_info(f"Done: Total clubs processed:{total_clubs}: Total url failures:{len(failed_urls)}")
    return htmls, total_clubs, failed_urls



def extract_club_games(htmls, acbl_url):
    dfs = {}
    ClubInfos = {}
    total_htmls = len(htmls)
    for n,(cn,html) in enumerate(htmls.items()):
        n += 1
        print_to_log_info(f'Processing club ({n}/{total_htmls}) {cn}')
        bs = BeautifulSoup(html, "html.parser") # todo: do this only once.
        html_table = bs.find('table')
        if html_table is None:
            print_to_log_info(f'Invalid club-result for {cn}')
            continue
        # /html/body/div[2]/div/div[2]/div[1]/div[2]
        ClubInfo = bs.find('div', 'col-md-8')
        #print_to_log(ClubInfo)
        ci = {}
        ci['Name'] = ClubInfo.find('h1').contents[0].strip() # get first text and strip
        ci['Location'] = ClubInfo.find('h5').contents[0].strip() # get first text and strip
        if ClubInfo.find('a'):
            ci['WebSite'] = ClubInfo.find('a')['href'] # get href of first a
        ClubInfos[cn] = ci
        print_to_log_info(f'{ci}')
        # assumes first table is our target
        d = pd.read_html(StringIO(str(html_table)))
        assert len(d) == 1
        df = pd.DataFrame(d[0])
        df.insert(0,'Club',cn)
        df.insert(1,'EventID','?')
        hrefs = [acbl_url+link.get('href')[1:] for link in html_table.find_all('a', href=re.compile(r"^/club-results/details/\d*$"))]
        df.drop('Unnamed: 6', axis=1, inplace=True)
        df['ResultID'] = [result.rsplit('/', 1)[-1] for result in hrefs]
        df['ResultUrl'] = hrefs
        dfs[cn] = df
    print_to_log_info(f"Done: Total clubs processed:{len(dfs)}")
    return dfs, ClubInfos    


def extract_club_result_json(dfs, filtered_clubs, starting_nclub, ending_nclub, total_local_files, acblPath,acbl_url, read_local=True):
    total_clubs = len(filtered_clubs)
    failed_urls = []
    total_urls_processed = 0
    total_local_files_read = 0
    #headers={"user-agent":None} # Not sure why this has become necessary. Failed 2021-Sep-02 so using Chrome curl user-agent.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    for ndf,(kdf,df) in enumerate(filtered_clubs.items()):
        if ndf < starting_nclub or ndf >= ending_nclub:
            #print_to_log_info(f"Skipping club #{ndf} {kdf}") # obsolete when filtered_clubs works
            continue
        ndf += 1
        except_count = 0
        total_results = len(df['ResultUrl'])
        for cn, (nurl, url) in zip(df['Club'],enumerate(df['ResultUrl'])):
            #nurl += 1
            total_urls_processed += 1
            html_file = url.replace(acbl_url,'').replace('club-results','club-results/'+str(cn))+'.html'
            json_file = html_file.replace('.html','.data.json')
            if nurl % 100 == 0: # commented out because overloaded notebook output causing system instability.
                print_to_log_info(f'Processing club ({ndf}/{total_clubs}): result file ({nurl}/{total_results}): {html_file}')
            #if ndf < 1652:
            #    continue
            html_path = acblPath.joinpath(html_file)
            json_path = acblPath.joinpath(json_file)
            html = None
            data_json = None
            if read_local and json_path.exists():
                #if html_path.exists():
                #    print_to_log(f'Found local html file: {html_file}')
                #else:
                #    print_to_log(f'Missing local html file: {html_file}')
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data_json = json.load(f)
                except:
                    print_to_log_info(f'Exception when reading json file: {json_file}. Deleting html and json files.')
                else:
                    total_local_files_read += 1
                    #print_to_log_info(f'Reading local ({total_local_files_read}/{total_local_files}) file:{json_path}: len:{json_path.stat().st_size}') # commented out because overloaded notebook output causing system instability.
            else:
                print_to_log_info(f'Requesting {url}')
                try:
                    r = requests.get(url,headers=headers)
                except Exception as ex:
                    print_to_log_info(f'Exception: count:{except_count} type:{type(ex).__name__} args:{ex.args}')
                    if except_count > 5:
                        print_to_log_info('Except count exceeded')
                        break # skip url
                    except_count += 1
                    time.sleep(1) # just in case the exception is transient
                    continue # retry url
                except KeyboardInterrupt as e:
                    print_to_log_info(f"Error: {type(e).__name__} while processing file:{url}")
                    print_to_log_info(traceback.format_exc())
                    canceled = True
                    break
                else:
                    except_count = 0            
                html = r.text
                print_to_log_info(f'Creating {html_file}: len={len(html)}')
                # some clubs return 200 (ok) but with instructions to login (len < 200).
                # skip clubs returning errors or tiny files. assumes one failed club result will be true for all club's results.
                if r.status_code != 200 or len(html) < 200:
                    print_to_log_info(f'Error: {r.status_code} len:{len(html)} {url}. Waiting 60s.')
                    failed_urls.append(url)
                    time.sleep(60) # wait 1 minute before retrying.
                    break
                # pathlib.Path.mkdir(html_path.parent, parents=True, exist_ok=True)
                html_path.parent.mkdir(parents=True, exist_ok=True)
                html_path.write_text(html, encoding="utf-8")
                bs = BeautifulSoup(html, "html.parser")
                scripts = bs.find_all('script')
                #print_to_log(scripts)
                for script in scripts:
                    if script.string: # not defined for all scripts
                        #print_to_log(script.string)
                        vardata = re.search('var data = (.*);\n', script.string)
                        if vardata:
                            data_json = json.loads(vardata.group(1))
                            #print_to_log(json.dumps(data_json, indent=4))
                            print_to_log_info(f"Writing {json_path}")
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(data_json, f, indent=2)
                            bbo_tournament_id = data_json["bbo_tournament_id"]
                            print_to_log_info(f'bbo_tournament_id: {bbo_tournament_id}')
                time.sleep(2) # need to self-throttle otherwise acbl returns 403 "forbidden". looks like sleep of 1 second has become insufficient in 2025-Jun.
            # if no data_json file read, must be an error so delete both html and json files.
            if not data_json:
                html_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
            #print_to_log(f'Files processed ({total_urls_processed}/{total_local_files_read}/{total_urls_to_process})')
    print_to_log_info(f'Failed Urls: len:{len(failed_urls)} Urls:{failed_urls}')
    print_to_log_info(f"Done: Totals: clubs:{total_clubs} urls:{total_urls_processed} local files read:{total_local_files_read}: failed urls:{len(failed_urls)}")
    return total_urls_processed, total_local_files_read, failed_urls


def club_results_json_to_sql(urls, starting_nfile=0, ending_nfile=0, initially_delete_all_output_files=False, skip_existing_files=True, event_types=[]):
    total_files_written = 0
    if ending_nfile == 0: ending_nfile = len(urls)
    filtered_urls = urls[starting_nfile:ending_nfile]
    total_urls = len(filtered_urls)
    start_time = time.time()

    # delete files first, using filtered list of urls
    if initially_delete_all_output_files:
        for nfile,url in enumerate(filtered_urls):
            sql_file = url.with_suffix('.sql')
            sql_file.unlink(missing_ok=True)

    for nfile,url in enumerate(filtered_urls):
        nfile += 1
        #url = 'https://my.acbl.org/club-results/details/290003' # todo: insert code to extract json from script
        #r = requests.get(url)
        json_file = url
        sql_file = url.with_suffix('.sql')
        print_to_log_info(f"Processing ({nfile}/{total_urls}): file:{json_file.as_posix()}")
        if skip_existing_files:
            if sql_file.exists():
               #print_to_log_info(f"Skipping: File exists:{sql_file.as_posix()}") # removed to avoid too much output
               continue
        try:
            data_json = None
            with open(json_file, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            #print_to_log(f"Reading {json_file.as_posix()} dict len:{len(data_json)}")
            if len(event_types) > 0 and data_json['type'] not in event_types:
                print_to_log(f"Skipping type:{data_json['type']}: file{json_file.as_posix()}") # removed to avoid too much output
                continue
            tables = defaultdict(lambda :defaultdict(dict))
            primary_keys = ['id']
            json_to_sql_walk(tables,"events","","",data_json,primary_keys) # "events" is the main table.
            with open(sql_file,'w', encoding='utf-8') as f:
                CreateSqlFile(tables,f,primary_keys)
            total_files_written += 1
        except Exception as e:
            print_to_log_info(f"Error: {e}: type:{data_json['type']} file:{url.as_posix()}")
        else:
            print_to_log_info(f"Writing: type:{data_json['type']} file:{sql_file.as_posix()}")

    print_to_log_info(f"All files processed:{total_urls} files written:{total_files_written} total time:{round(time.time()-start_time,2)}")
    return total_urls, total_files_written


# todo: can acblPath be removed?
def club_results_create_sql_db(db_file_connection_string, create_tables_sql_file, db_file_path,  acblPath, input_subdir, db_memory_connection_string='sqlite://', starting_nfile=0, ending_nfile=0, write_direct_to_disk=False, create_tables=True, delete_db=False, perform_integrity_checks=False, create_engine_echo=False):
    if write_direct_to_disk:
        db_connection_string = db_file_connection_string # disk file based db
    else:
        db_connection_string = db_memory_connection_string # memory based db

    if delete_db and sqlalchemy_utils.functions.database_exists(db_file_connection_string):
        print_to_log_info(f"Deleting db:{db_file_connection_string}")
        sqlalchemy_utils.functions.drop_database(db_file_connection_string) # warning: can't delete file if in use by another app (restart kernel).

    if not sqlalchemy_utils.functions.database_exists(db_connection_string):
        print_to_log_info(f"Creating db:{db_connection_string}")
        sqlalchemy_utils.functions.create_database(db_connection_string)
        create_tables = True
        
    engine = sqlalchemy.create_engine(db_connection_string, echo=create_engine_echo)
    raw_connection = engine.raw_connection()

    if create_tables:
        print_to_log_info(f"Creating tables from:{create_tables_sql_file}")
        with open(create_tables_sql_file, 'r', encoding='utf-8') as f:
            create_sql = f.read()
        raw_connection.executescript(create_sql) # create tables

    # Discover input SQL scripts from one or more subdirectories under acblPath
    urls = []
    base = acblPath.joinpath(input_subdir)
    if base.is_dir():
        for path in base.rglob('*.data.sql'):
            urls.append(path)

    #urls = [acblPath.joinpath(f) for f in ['club-results/108571/details/280270.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in ['club-results/275966/details/99197.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in ['club-results/275966/details/98557.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in ['club-results/104034/details/100661.data.sql','club-results/104034/details/100663.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in 100*['club-results/108571/details/191864.data.sql']]

    total_script_execution_time = 0
    total_scripts_executed = 0
    canceled = False
    if ending_nfile == 0: ending_nfile = len(urls)
    filtered_urls = urls[starting_nfile:ending_nfile]
    total_filtered_urls = len(filtered_urls)
    start_time = time.time()
    for nfile,url in enumerate(filtered_urls):
        sql_file = url
        #if (nfile % 1000) == 0:
        #    print_to_log_info(f"Executing SQL script ({nfile}/{total_filtered_urls}): file:{sql_file.as_posix()}")
        
        try:
            sql_script = None
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            start_script_time = time.time()
            raw_connection.executescript(sql_script)
        except Exception as e:
            print_to_log_info(f"Error: {type(e).__name__} while processing file:{url.as_posix()}")
            print_to_log_info(traceback.format_exc())
            print_to_log_info(f"Every json field must be an entry in the schema file. Update schema if needed.")
            print_to_log_info(f"Removing {url.as_posix()}")
            sql_file.unlink(missing_ok=True) # delete any bad files, fix issues, rerun.
            continue # todo: log error.
            #break
        except KeyboardInterrupt as e:
            print_to_log_info(f"Error: {type(e).__name__} while processing file:{url.as_posix()}")
            print_to_log_info(traceback.format_exc())
            canceled = True
            break
        else:
            script_execution_time = time.time()-start_script_time
            if (nfile % 1000) == 0:
                print_to_log_info(f"{nfile}/{total_filtered_urls} SQL script executed: file:{url.as_posix()}: time:{round(script_execution_time,2)}")
            total_script_execution_time += script_execution_time
            total_scripts_executed += 1

    print_to_log_info(f"SQL scripts executed ({total_scripts_executed}/{total_filtered_urls}/{len(urls)}): total changes:{raw_connection.total_changes} total script execution time:{round(time.time()-start_time,2)}: avg script execution time:{round(total_script_execution_time/max(1,total_scripts_executed),2)}")
    # if using memory db, write memory db to disk file.
    if not canceled:
        if perform_integrity_checks:
            # todo: research how to detect and display failures? Which checks are needed?
            print_to_log_info(f"Performing quick_check on file")
            raw_connection.execute("PRAGMA quick_check;") # takes 7m on disk
            print_to_log_info(f"Performing foreign_key_check on file")
            raw_connection.execute("PRAGMA foreign_key_check;") # takes 3m on disk
            print_to_log_info(f"Performing integrity_check on file")
            raw_connection.execute("PRAGMA integrity_check;") # takes 25m on disk
        if not write_direct_to_disk:
            print_to_log_info(f"Writing memory db to file:{db_file_connection_string}")
            engine_file = sqlalchemy.create_engine(db_file_connection_string)
            raw_connection_file = engine_file.raw_connection()
            raw_connection.backup(raw_connection_file.connection) # takes 45m
            raw_connection_file.close()
            engine_file.dispose()
            print_to_log_info(f"Saved {db_file_path}: size:{db_file_path.stat().st_size}")

    raw_connection.close()
    engine.dispose()
    print_to_log_info("Done.")
    return total_scripts_executed # not sure if any return value is needed.


def get_club_results_details_data(url):
    """
    Retrieve detailed data for a specific club event.
    
    NOTE: This function now uses Playwright due to ACBL's 403 blocking of requests.
    Will check cache first (club-results/*/details/<event_id>.data.json) before downloading.
    Return format maintained for backward compatibility: dict or None
    
    Args:
        url: URL to club results details page (e.g., https://my.acbl.org/club-results/details/993420)
    
    Returns:
        dict or None: Event details data or None if not found/team event
    """
    print_to_log_info('details url:', url)
    
    # Extract event_id from URL
    event_id = url.rstrip('/').split('/')[-1]
    
    # Check for cached file in club-results/*/details/<event_id>.data.json
    cache_pattern = pathlib.Path('club-results') / '*' / 'details' / f'{event_id}.data.json'
    cached_files = list(pathlib.Path('.').glob(str(cache_pattern)))
    
    if cached_files:
        cached_file = cached_files[0]  # Use first match
        print_to_log_info(f'Using cached file: {cached_file}')
        try:
            with open(cached_file, 'r', encoding='utf-8') as f:
                details_data = json.load(f)
            print_to_log_info(f'Loaded from cache with {len(details_data.get("boards", []))} boards')
            return details_data
        except Exception as e:
            print_to_log_info(f'Error loading cache file: {e}, will download instead')
    
    # Not in cache, download using Playwright
    print_to_log_info('Not in cache, downloading with Playwright (requires: pip install playwright && playwright install chromium)')
    
    # Use Playwright version (defined later in this file)
    details_data = get_club_results_details_data_playwright(
        url,
        headless=True,
        verbose=False  # Use logging instead
    )
    
    if details_data:
        print_to_log_info(f'Downloaded details data with {len(details_data.get("boards", []))} boards')
        
        # Save to cache
        try:
            club_id = details_data.get('club_id_number') or details_data.get('club_id')
            if club_id:
                cache_dir = pathlib.Path('club-results') / str(club_id) / 'details'
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f'{event_id}.data.json'
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(details_data, f, indent=2, ensure_ascii=False)
                print_to_log_info(f'Saved to cache: {cache_file}')
            else:
                print_to_log_info('No club_id found in data, skipping cache save')
        except Exception as e:
            print_to_log_info(f'Error saving to cache: {e}')
    else:
        print_to_log_info('No details data found (possibly team event)')
    
    return details_data


# def get_tournament_results_details_data(session_id, acbl_api_key):
#     print_to_log_info('details url:',session_id)
#     response = get_tournament_session_results(session_id, acbl_api_key)
#     assert response.status_code == 200, [session_id, response.status_code]

#     # soup = BeautifulSoup(response.content, "html.parser")

#     # if soup.find('result-details-combined-section'):
#     #     data = soup.find('result-details-combined-section')['v-bind:data']
#     # elif soup.find('result-details'):
#     #     data = soup.find('result-details')['v-bind:data']
#     # elif soup.find('team-result-details'):
#     #     return None # todo: handle team events
#     #     data = soup.find('team-result-details')['v-bind:data']
#     # else:
#     #     return None # "Can't find data tag."
#     # assert data is not None and isinstance(data,str) and len(data), [session_id, data]

#     # details_data = json.loads(data) # returns dict from json
#     details_data = response.json()
#     return details_data


def get_club_results_from_acbl_number(acbl_number):
    """
    Retrieve club results for a specific ACBL member number.
    
    NOTE: This function now uses Playwright due to ACBL's 403 blocking of requests.
    Return format maintained for backward compatibility: (url, detail_url, msg)
    For more features (pagination, club_id), use get_club_results_from_acbl_number_playwright() directly.
    
    Args:
        acbl_number: ACBL member number (e.g., 2663279)
    
    Returns:
        dict: {event_id: (url, detail_url, msg)}
    """
    url = f"https://my.acbl.org/club-results/my-results/{acbl_number}"
    print_to_log_info('my-results url:', url)
    print_to_log_info('Note: Using Playwright (requires: pip install playwright && playwright install chromium)')
    
    # Use Playwright version (defined later in this file, get all events, no limit)
    playwright_results = get_club_results_from_acbl_number_playwright(
        acbl_number, 
        headless=True,
        limit=0,  # Get all events
        verbose=False  # Use logging instead
    )
    
    # Convert to old format (drop club_id for backward compatibility)
    my_results_details_data = {}
    for event_id, (my_results_url, detail_url, msg, club_id) in playwright_results.items():
        my_results_details_data[event_id] = (url, detail_url, msg)
    
    print_to_log_info(f'Found {len(my_results_details_data)} event(s)')
    return my_results_details_data


# todo: need to get the player's club game history without re-calling get_club_results_from_acbl_number(). Similar to tournament history/session results.
def get_club_player_history(acbl_number):
    return get_club_results_from_acbl_number(acbl_number)


# get a single tournament session result
def get_tournament_session_results(session_id, acbl_api_key):
    """
    Retrieve tournament session results.
    Will check cache first (tournaments/<session_id>.session.json) before downloading.
    
    Args:
        session_id: Tournament session ID (e.g., '2310369-2801-2')
        acbl_api_key: ACBL API key
    
    Returns:
        requests.Response object (for backward compatibility)
    """
    # Check for cached file
    cache_file = pathlib.Path('tournaments') / f'{session_id}.session.json'
    
    if cache_file.exists():
        print_to_log_info(f'Using cached file: {cache_file}')
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Create a mock response object to maintain backward compatibility
            class CachedResponse:
                def __init__(self, data):
                    self.status_code = 200
                    self._json_data = data
                
                def json(self):
                    return self._json_data
            
            print_to_log_info(f'Loaded from cache')
            return CachedResponse(cached_data)
        except Exception as e:
            print_to_log_info(f'Error loading cache file: {e}, will download instead')
    
    # Not in cache, download from API
    headers = {
        'accept': 'application/json', 
        'Authorization': f'Bearer {acbl_api_key}',
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    path = 'https://api.acbl.org/v1/tournament/session'
    query = {'id':session_id,'full_monty':1}
    params = urllib.parse.urlencode(query)
    url = path+'?'+params
    print_to_log_info('tournament session url:',url)
    response = requests.get(url, headers=headers)
    
    # Save to cache if successful
    if response.status_code == 200:
        try:
            data = response.json()
            cache_dir = pathlib.Path('tournaments')
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print_to_log_info(f'Saved to cache: {cache_file}')
        except Exception as e:
            print_to_log_info(f'Error saving to cache: {e}')
    else:
        print_to_log_info(f'Download failed with status {response.status_code}, not caching')
    
    return response


# get a list of tournament session results
def get_tournament_sessions_from_acbl_number(acbl_number, acbl_api_key):
    url, json_responses = download_tournament_player_history(acbl_number, acbl_api_key)
    tournament_sessions_urls = {d['session_id']:(url, f"https://live.acbl.org/event/{d['session_id'].replace('-','/')}/summary", f"{d['date']}, {d['score_tournament_name']}, {d['score_event_name']}, {d['score_session_time_description']}, {d['percentage']}", d) for r in json_responses for d in r['data']} # https://live.acbl.org/event/NABC232/23FP/1/summary
    return tournament_sessions_urls


# get a single player's tournament history
def download_tournament_player_history(player_id, acbl_api_key):
    headers = {
        'accept': 'application/json', 
        'Authorization': f'Bearer {acbl_api_key}',
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    path = 'https://api.acbl.org/v1/tournament/player/history_query'
    query = {'acbl_number':player_id,'page':1,'page_size':50,'start_date':'1900-01-01'}
    params = urllib.parse.urlencode(query)
    url = path+'?'+params
    sessions_count = 0
    except_count = 0
    json_responses = []
    while url:
        try:
            print_to_log_info(f'url:{url}')
            print_to_log_info(f'headers:{headers}')
            response = requests.get(url, headers=headers)
            print_to_log_info(f'Status Code:{response.status_code}')
        except Exception as ex:
            print_to_log_info(f'Exception: count:{except_count} type:{type(ex).__name__} args:{ex.args}')
            if except_count > 5:
                print_to_log_info('Except count exceeded')
                break # skip url
            except_count += 1
            time.sleep(1) # just in case the exception is transient
            continue # retry url
        except KeyboardInterrupt as e:
            print_to_log_info(f"Error: {type(e).__name__} while processing file:{url}")
            print_to_log_info(traceback.format_exc())
            return None
        else:
            except_count = 0
        if response.status_code in [400,500,504]: # 500 is unknown response code. try skipping player
            print_to_log_info(f'Status Code:{response.status_code}: count:{len(json_responses)} skipping') # 4476921 - Thx Merle.
            # next_page_url = None
            # sessions_total = 0
            break
        assert response.status_code == 200, (url, response.status_code) # 401 is authorization error often because Personal Access Token has expired.
        json_response = response.json()
        #json_pretty = json.dumps(json_response, indent=4)
        #print_to_log(json_pretty)
        json_responses.append(json_response)
        url = json_response['next_page_url']
    return path, json_responses


# get a list of player's tournament history
def download_tournament_players_history(player_ids, acbl_api_key, dirPath):
    start_time = time.time()
    get_count = 0 # total number of gets
    #canceled = False
    for n,player_id in enumerate(sorted(player_ids)):
        if player_id.startswith('tmp:') or player_id.startswith('#'): # somehow #* crept into player_id
            print_to_log_info(f'Skipping player_id:{player_id}')
            continue
        else:
            print_to_log_info(f'Processing player_id:{player_id}')
        if dirPath.exists():
            session_file_count = len(list(dirPath.glob('*.session.json')))
            print_to_log_info(f'dir exists: file count:{session_file_count} dir:{dirPath}')
            #if session_file_count == 0: # todo: ignore players who never played a tournament?
            #    print_to_log(f'dir empty -- skipping')
            #    continue
            #if session_file_count > 0: # todo: temp?
            #    print_to_log(f'dir not empty -- skipping')
            #    continue
        else:
            print_to_log_info(f'Creating dir:{dirPath}')
            dirPath.mkdir(parents=True,exist_ok=True)
            session_file_count = 0
        url, json_responses = download_tournament_player_history(player_id, acbl_api_key)
        if json_responses is None: # canceled
            break
        get_count = len(json_responses)
        if get_count == 0: # skip player_id's generating errors. e.g. player_id 5103045, 5103045, 5103053
            continue
        print_to_log_info(f"{n}/{len(player_ids)} gets:{get_count} rate:{round((time.time()-start_time)/get_count,2)} {player_id=}")
        #time.sleep(1) # throttle api calling. Maybe not needed as api is taking longer than 1s.
        sessions_count = 0
        for json_response in json_responses:
            sessions_total = json_response['total'] # is same for every page
            if sessions_total == session_file_count: # sometimes won't agree because identical sessions. revised results?
                print_to_log_info(f'File count correct: {dirPath}: terminating {player_id} early.')
                sessions_count = sessions_total
                break
            for data in json_response['data']:
                sessions_count += 1 # todo: oops, starts first one at 2. need to move
                session_id = data['session_id']
                filePath_sql = dirPath.joinpath(session_id+'.session.sql')
                filePath_json = dirPath.joinpath(session_id+'.session.json')
                if filePath_sql.exists() and filePath_json.exists() and filePath_sql.stat().st_ctime > filePath_json.stat().st_ctime:
                    print_to_log_info(f'{sessions_count}/{sessions_total}: File exists: {filePath_sql}: skipping')
                    #if filePath_json.exists(): # json file is no longer needed?
                    #    print_to_log(f'Deleting JSON file: {filePath_json}')
                    #    filePath_json.unlink(missing_ok=True)
                    break # continue will skip file. break will move on to next player
                if filePath_json.exists():
                    print_to_log_info(f'{sessions_count}/{sessions_total}: File exists: {filePath_json}: skipping')
                    break # continue will skip file. break will move on to next player
                response = get_tournament_session_results(session_id, acbl_api_key)
                assert response.status_code == 200, response.status_code
                session_json = response.json()
                #json_pretty = json.dumps(json_response, indent=4)
                print_to_log_info(f'{sessions_count}/{sessions_total}: Writing:{filePath_json} len:{len(session_json)}')
                with open(filePath_json,'w',encoding='utf-8') as f:
                    f.write(json.dumps(session_json, indent=4))
        if sessions_count != sessions_total:
            print_to_log_info(f'Session count mismatch: {dirPath}: variance:{sessions_count-sessions_total}')

# def post_with_auth_token(url, data, auth_token, headers=None):
#     """
#     Performs a POST request with authorization bearer token.
    
#     Args:
#         url (str): The URL to send the POST request to
#         data (dict): The data to send in the POST request body
#         auth_token (str): The authorization bearer token
#         headers (dict, optional): Additional headers to include. Defaults to None.
    
#     Returns:
#         requests.Response: The response from the server
#     """
#     # Set up default headers
#     default_headers = {
#         'Authorization': f'Bearer {auth_token}',
#         'Content-Type': 'application/json',
#         'Accept': 'application/json',
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/124.0.0.0 Safari/537.36"
#         )
#     }
    
#     # Merge with any additional headers
#     if headers:
#         default_headers.update(headers)
    
#     # Make the POST request
#     response = requests.post(
#         url,
#         json=data,
#         headers=default_headers
#     )
    
#     print_to_log_info(f'POST request to {url}')
#     print_to_log_info(f'Status Code: {response.status_code}')
    
#     return response

# # Example usage:
# """
# # Example of how to use the function:
# url = 'https://api.example.com/endpoint'
# data = {
#     'key1': 'value1',
#     'key2': 'value2'
# }
# auth_token = 'your_auth_token_here'

# response = post_with_auth_token(url, data, auth_token)

# if response.status_code == 200:
#     result = response.json()
#     print_to_log_info('Success:', result)
# else:
#     print_to_log_info('Error:', response.text)
# """

# def get_curl_command(url, data, auth_token, headers=None):
#     """
#     Generates a curl command string for making a POST request with authorization bearer token.
    
#     Args:
#         url (str): The URL to send the POST request to
#         data (dict): The data to send in the POST request body
#         auth_token (str): The authorization bearer token
#         headers (dict, optional): Additional headers to include. Defaults to None.
    
#     Returns:
#         str: The curl command string
#     """
#     # Set up default headers
#     default_headers = {
#         'Authorization': f'Bearer {auth_token}',
#         'Content-Type': 'application/json',
#         'Accept': 'application/json'
#     }
    
#     # Merge with any additional headers
#     if headers:
#         default_headers.update(headers)
    
#     # Build header arguments
#     header_args = ' '.join([f'-H "{k}: {v}"' for k, v in default_headers.items()])
    
#     # Convert data to JSON string, escaping quotes
#     json_data = json.dumps(data).replace('"', '\\"')
    
#     # Build the complete curl command
#     curl_cmd = f'curl -X POST {header_args} -d "{json_data}" {url}'
    
#     return curl_cmd

# # Example usage:
# """
# url = 'https://api.example.com/endpoint'
# data = {
#     'key1': 'value1',
#     'key2': 'value2'
# }
# auth_token = 'your_auth_token_here'

# curl_command = get_curl_command(url, data, auth_token)
# print_to_log_info('Curl command:')
# print_to_log_info(curl_command)

# # The output would look like:
# # curl -X POST -H "Authorization: Bearer your_auth_token_here" -H "Content-Type: application/json" -H "Accept: application/json" -d "{"key1":"value1","key2":"value2"}" https://api.example.com/endpoint
# """



# todo: finish converting from pandas to polars. hitch is that pd.json_normalize() is pandas only.
def create_club_dfs(data):
    dfs = {}
    event_df = pd.json_normalize(data,max_level=0) # todo: convert to polars
    dfs['event'] = pl.from_pandas(event_df)
    for k,v in event_df.items():
        if isinstance(v[0],dict) or isinstance(v[0],list):
            assert k not in dfs, k
            df = pd.json_normalize(data,max_level=0)[k] # todo: convert to polars
            # must test whether df is all scalers. Very difficult to debug.
            if isinstance(v[0],dict) and not any([isinstance(vv,dict) or isinstance(vv,list) for kk,vv in df[0].items()]):
                dfs[k] = pl.from_pandas(pd.DataFrame.from_records(df[0],index=[0])) # must use from_records to avoid 'all values are scaler must specify index' error
            else:
                dfs[k] = pl.from_pandas(pd.DataFrame.from_records(df[0]).astype('string')) # warning: needed astype('string') to avoid int error
            event_df.drop(columns=[k],inplace=True)
            #if all(isinstance(kk,int) or (isinstance(kk,str) and kk.isnumeric()) for kk,vv in v.items()):
    dfs['hand_records'] = pl.from_pandas(pd.json_normalize(data,['sessions','hand_records']))
    dfs['strat_place'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','pair_summaries','strat_place']))
    dfs['sections'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections']))
    dfs['boards'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','boards']))
    dfs['pair_summaries'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','pair_summaries']))
    dfs['players'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','pair_summaries','players']))
    dfs['board_results'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','boards','board_results']))
    return dfs


# def create_tournament_dfs(data):
#     return create_club_dfs(data)



def merge_clean_augment_club_dfs(dfs, sd_cache_d, acbl_number):  # todo: acbl_number obsolete?

    print_to_log_info('merge_clean_augment_club_dfs: dfs keys:', dfs.keys())

    df_brs = dfs['board_results']
    print_to_log_info(df_brs.head(1))
    #assert not df_brs.columns.contains(r'^.*_[xy]$').any()

    df_b = dfs['boards'].rename({'id': 'board_id'}).select(['board_id', 'section_id', 'board_number'])
    print_to_log_info(df_b.head(1))
    #assert not df_b.columns.str.contains(r'^.*_[xy]$').any()

    df_br_b = df_brs.join(df_b, on='board_id', how='left')
    print_to_log_info(df_br_b.head(1))
    assert df_br_b.height == df_brs.height
    #assert not df_br_b.columns.str.contains(r'^.*_[xy]$').any()

    df_sections = dfs['sections'].rename({'id': 'section_id', 'name': 'section_name'}).drop(['created_at', 'updated_at', 'transaction_date', 'pair_summaries', 'boards'])
    print_to_log_info(df_sections.head(1))

    df_br_b_sections = df_br_b.join(df_sections, on='section_id', how='left')
    print_to_log_info(df_br_b_sections.head(1))
    assert df_br_b_sections.height == df_br_b.height
    #assert not df_br_b_sections.columns.str.contains(r'^.*_[xy]$').any()

    df_sessions = dfs['sessions'].rename({'id': 'session_id', 'number': 'session_number'}).drop(['created_at', 'updated_at', 'transaction_date', 'hand_records', 'sections'])
    print_to_log_info(df_sessions.head(1))

    df_sessions = df_sessions.with_columns(pl.col("session_id").cast(pl.Int64)) # todo: hack to fix hack in pandas to polars conversion error
    df_br_b_sections_sessions = df_br_b_sections.join(df_sessions, on='session_id', how='left')
    print_to_log_info(df_br_b_sections_sessions.head(1))
    assert df_br_b_sections_sessions.height == df_br_b_sections.height
    #assert not df_br_b_sections_sessions.columns.str.contains(r'^.*_[xy]$').any()

    df_clubs = dfs['club'].rename({'id': 'event_id', 'name': 'club_name', 'type': 'club_type'}).drop(['created_at', 'updated_at', 'transaction_date'])
    print_to_log_info(df_clubs.head(1))

    df_br_b_sections_sessions = df_br_b_sections_sessions.with_columns(pl.col("event_id").cast(pl.Int64)) # todo: hack to fix hack in pandas to polars conversion error
    df_br_b_sections_sessions_clubs = df_br_b_sections_sessions.join(df_clubs, on='event_id', how='left')
    print_to_log_info(df_br_b_sections_sessions_clubs.head(1))
    assert df_br_b_sections_sessions_clubs.height == df_br_b_sections_sessions.height
    #assert not df_sections.columns.str.contains(r'^.*_[xy]$').any()

    df_events = dfs['event'].rename({'id': 'event_id', 'club_name': 'event_club_name', 'type': 'event_type'}).drop(['created_at', 'updated_at', 'transaction_date', 'deleted_at'])
    print_to_log_info(df_events.head(1))

    df_br_b_sections_sessions_events = df_br_b_sections_sessions_clubs.join(df_events, on='event_id', how='left')
    print_to_log_info(df_br_b_sections_sessions_events.head(1))
    assert df_br_b_sections_sessions_events.height == df_br_b_sections_sessions_clubs.height
    #assert not df_br_b_sections_sessions_events.columns.str.contains(r'^.*_[xy]$').any()

    df_pair_summaries = dfs['pair_summaries'].rename({'id': 'pair_summary_id'}).drop(['created_at', 'updated_at', 'transaction_date'])
    print_to_log_info(df_pair_summaries.head(1))

    df_br_b_pair_summary_ns = df_pair_summaries.filter(pl.col('direction') == 'NS').with_columns(pl.col('pair_number').alias('ns_pair'), pl.col('section_id').alias('section_id'))
    #assert not df_br_b_pair_summary_ns.columns.str.contains(r'^.*_[xy]$').any()
    df_br_b_pair_summary_ew = df_pair_summaries.filter(pl.col('direction') == 'EW').with_columns(pl.col('pair_number').alias('ew_pair'), pl.col('section_id').alias('section_id'))
    #assert not df_br_b_pair_summary_ew.columns.str.contains(r'^.*_[xy]$').any()

    df_players = dfs['players'].drop(['id', 'created_at', 'updated_at', 'transaction_date']).rename({'id_number': 'player_number', 'name': 'player_name'})
    
    # Get all column names except the grouping column
    player_columns = [col for col in df_players.columns if col != 'pair_summary_id']
    
    # Create aggregations for each direction with all columns
    player_n_aggs = [pl.first(col).alias(f'{col}_n') for col in player_columns]
    player_s_aggs = [pl.last(col).alias(f'{col}_s') for col in player_columns]
    player_e_aggs = [pl.first(col).alias(f'{col}_e') for col in player_columns]
    player_w_aggs = [pl.last(col).alias(f'{col}_w') for col in player_columns]
    
    player_n = df_players.group_by('pair_summary_id').agg(player_n_aggs)
    player_s = df_players.group_by('pair_summary_id').agg(player_s_aggs)
    player_e = df_players.group_by('pair_summary_id').agg(player_e_aggs)
    player_w = df_players.group_by('pair_summary_id').agg(player_w_aggs)

    player_ns = player_n.join(player_s, on='pair_summary_id', how='left')
    print_to_log_info(player_ns.head(1))
    assert player_ns.height == player_n.height
    #assert not player_ns.columns.str.contains(r'^.*_[xy]$').any()
    player_ew = player_e.join(player_w, on='pair_summary_id', how='left')
    print_to_log_info(player_ew.head(1))
    assert player_ew.height == player_e.height
    #assert not player_ew.columns.str.contains(r'^.*_[xy]$').any()

    df_pair_summary_players_ns = df_br_b_pair_summary_ns.join(player_ns, on='pair_summary_id', how='left')
    assert df_pair_summary_players_ns.height == df_br_b_pair_summary_ns.height
    df_pair_summary_players_ew = df_br_b_pair_summary_ew.join(player_ew, on='pair_summary_id', how='left')
    assert df_pair_summary_players_ew.height == df_br_b_pair_summary_ew.height

    # Rename pair summary columns with _NS and _EW suffixes to avoid _right suffix confusion
    pair_summary_columns_to_rename = [
        'pair_summary_id', 'event_pair_type', 'pair_type_id', 'pair_number', 
        'direction', 'strat', 'boards_played', 'score', 'percentage', 
        'adjustment', 'handicap', 'raw_score', 'is_eligible', 'awardFinal', 
        'strat_place', 'session_scores', 'players'
    ]
    
    # Rename NS columns
    ns_rename_dict = {col: f'{col}_NS' for col in pair_summary_columns_to_rename if col in df_pair_summary_players_ns.columns}
    df_pair_summary_players_ns = df_pair_summary_players_ns.rename(ns_rename_dict)
    
    # Rename EW columns
    ew_rename_dict = {col: f'{col}_EW' for col in pair_summary_columns_to_rename if col in df_pair_summary_players_ew.columns}
    df_pair_summary_players_ew = df_pair_summary_players_ew.rename(ew_rename_dict)

    df_br_b_sections_sessions_events_pair_summary_players = df_br_b_sections_sessions_events.join(df_pair_summary_players_ns, on=['section_id', 'ns_pair'], how='left')
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert df_br_b_sections_sessions_events_pair_summary_players.height == df_br_b_sections_sessions_events.height
    #assert not df_br_b_sections_sessions_events_pair_summary_players.columns.str.contains(r'^.*_[xy]$').any()
    df_br_b_sections_sessions_events_pair_summary_players = df_br_b_sections_sessions_events_pair_summary_players.join(df_pair_summary_players_ew, on=['section_id', 'ew_pair'], how='left')
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert df_br_b_sections_sessions_events_pair_summary_players.height == df_br_b_sections_sessions_events.height
    #assert not df_br_b_sections_sessions_events_pair_summary_players.columns.str.contains(r'^.*_[xy]$').any()

    df_hrs = dfs['hand_records'].rename({'hand_record_set_id': 'hand_record_id'}).drop(['points.N', 'points.E', 'points.S', 'points.W'])
    print_to_log_info(df_hrs.head(1))

    df_br_b_sections_sessions_events_pair_summary_players = df_br_b_sections_sessions_events_pair_summary_players.with_columns(pl.col("hand_record_id").cast(pl.Int64)) # todo: hack to fix hack in pandas to polars conversion error
    df_br_b_sections_sessions_events_pair_summary_players_hrs = df_br_b_sections_sessions_events_pair_summary_players.join(df_hrs.drop(['id', 'created_at', 'updated_at']), left_on=['hand_record_id', 'board_number'], right_on=['hand_record_id', 'board'], how='left')
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players_hrs.head(1))
    assert df_br_b_sections_sessions_events_pair_summary_players_hrs.height == df_br_b_sections_sessions_events_pair_summary_players.height
    #assert not df_br_b_sections_sessions_events_pair_summary_players_hrs.columns.str.contains(r'^.*_[xy]$').any()

    df = df_br_b_sections_sessions_events_pair_summary_players_hrs
    for col in df.columns:
        print_to_log_info(f'cols: {col} {df[col].dtype}')

    df = df.drop(['id', 'created_at', 'updated_at', 'board_id', 'double_dummy_ns', 'double_dummy_ew'])

    df = df.rename({
        'board_number': 'Board',
        'club_id_number': 'Club',
        'contract': 'Contract',
        'game_date': 'Date',
        'ns_match_points': 'MP_NS',
        'ew_match_points': 'MP_EW',
        'ns_pair': 'Pair_Number_NS',
        'ew_pair': 'Pair_Number_EW',
        #'percentage_ns': 'Final_Standing_NS', # percentage_ns is not a column in the data. use percentage instead?
        #'percentage_ew': 'Final_Standing_EW', # percentage_ew is not a column in the data. use percentage instead?
        'result': 'Result',
        'round_number': 'Round',
        'ns_score': 'Score_NS',
        'ew_score': 'Score_EW',
        'session_number': 'Session',
        'table_number': 'Table',
        'tricks_taken': 'Tricks',
    })

    df = df.with_columns([
        pl.col('Club').cast(pl.Int32), # todo: convert to Categorical but must align with acbl_club_model_data or convert in predict?
        pl.col('board_record_string').cast(pl.Utf8),
        pl.col('Date').str.strptime(pl.Date, format="%Y-%m-%d %H:%M:%S"), # game_date has both date and time, but tournament start_date has only date.
        #pl.col('Final_Standing_NS').cast(pl.Float32), # Final_Standing_NS is not a column in the data. use percentage instead?
        #pl.col('Final_Standing_EW').cast(pl.Float32), # Final_Standing_EW is not a column in the data. use percentage instead?
        pl.col('hand_record_id').cast(pl.Int64),
        pl.col('Pair_Number_NS').cast(pl.UInt8),
        pl.col('Pair_Number_EW').cast(pl.UInt8),
        pl.col('pair_summary_id_NS').alias('pair_summary_id_ns'), # todo: remove this alias after ai model is updated to use pair_summary_id
        pl.col('pair_summary_id_EW').alias('pair_summary_id_ew'), # todo: remove this alias after ai model is updated to use pair_summary_id
    ])

    df = acbldf_to_mldf(df) # todo: temporarily convert to pandas to use augment_df until clean_validate_df is converted to polars

    return df

    #df, sd_cache_d, matchpoint_ns_d = augment_df(df, sd_cache_d)

    #return df, sd_cache_d, matchpoint_ns_d

def merge_clean_augment_tournament_dfs(dfs, json_results_d, sd_cache_d, player_id):
    # dfs_results contains: tournament[], event[], overalls[], handrecord[], sections[]
    print_to_log_info('merge_clean_augment_tournament_dfs: dfs keys:', dfs.keys())

    # tournament: ['_schedule_id', 'sanction', 'alt_sanction', 'name', 'start_date', 'end_date', 'district', 'unit', 'category', 'type', 'mp_restrictions', 'allowed_conventions', 'schedule_pdf', 'schedule_link', 'locations', 'last_updated', 'schedule_available', 'cancelled', 'contacts']
    # 'event': ['sanction', '_id', '_schedule_id', 'id', 'name', 'event_code', 'start_date', 'start_time', 'game_type', 'event_type', 'mp_limit', 'mp_color', 'mp_rating', 'is_charity', 'is_juniors', 'is_mixed', 'is_playthrough', 'is_seniors', 'is_side_game', 'is_womens', 'session_count', 'is_online', 'results_available', 'strat_count', 'strat_letters']
    # 'handrecord': ['box_number', 'board_number',
    #     'north_spades', 'north_hearts', 'north_diamonds', 'north_clubs', 'east_spades', 'east_hearts', 'east_diamonds', 'east_clubs', 'south_spades', 'south_hearts', 'south_diamonds', 'south_clubs', 'west_spades', 'west_hearts', 'west_diamonds', 'west_clubs',
    #     'double_dummy_north_south', 'double_dummy_east_west', 'double_dummy_par_score', 'dealer', 'vulnerability']
    # 'sections': ['id', 'number', 'transaction_date', 'hand_records', 'sections']
    # 'overall': ['session_id', 'mp_won', 'mp_color', 'score', 'team_number', 'pair_number', 'percentage', 'rank_strat_1', 'rank_strat_2', 'rank_strat_3', 'section', 'players']
    results_dfs_d = {}
    for k,v in json_results_d.items():
        print_to_log_info(k,type(v))
        if isinstance(v,list):
            results_dfs_d[k] = pl.DataFrame(v,strict=False) # concat the list of dicts into a single dict. 'sections' needs strict=False.
        elif isinstance(v,dict):
            results_dfs_d[k] = pl.DataFrame(v)
        else:
            results_dfs_d[k] = v

    section_label = dfs['section']
    df = results_dfs_d['sections'].filter(pl.col('section_label').eq(section_label)) # filter for the section of interest
    print_to_log_info(df.head(1))

    df = df.explode('board_results').unnest('board_results')
    print_to_log_info(df.head(1))

    # todo: no need to filter by orientation and concat. instead do everything in 1 df by filtering on orientation.
    ns_df = df.filter(pl.col('orientation') == 'N-S').drop(['orientation'])
    print_to_log_info(ns_df.head(1))
    ew_df = df.filter(pl.col('orientation') == 'E-W').drop(['orientation'])
    print_to_log_info(ew_df.head(1))
    assert ns_df.height == ew_df.height, f'ns_df.height: {ns_df.height}, ew_df.height: {ew_df.height}'
    assert ns_df.columns == ew_df.columns, f'ns_df.columns: {ns_df.columns}, ew_df.columns: {ew_df.columns}'

    identical_columns = ['session_id', 'section_label', 'movement_type', 'scoring_type', 'board_number', 'contract', 'declarer']
    print_to_log_info(identical_columns)
    ns_df = ns_df.rename({
        'match_points': 'match_points_ns',
        'percentage': 'percentage_ns',
    })
    print_to_log_info(ns_df.head(1))
    ew_df = ew_df.rename({
        'match_points': 'match_points_ew',
        'percentage': 'percentage_ew',
    })
    print_to_log_info(ew_df.head(1))

    ns_df = ns_df.with_columns(
        pl.col('pair_acbl').list.get(0).cast(pl.UInt32).alias('player_number_n'),
        pl.col('pair_acbl').list.get(1).cast(pl.UInt32).alias('player_number_s'),
        pl.col('pair_names').list.get(0).cast(pl.String).alias('player_name_n'),
        pl.col('pair_names').list.get(1).cast(pl.String).alias('player_name_s'),
        pl.col('opponent_pair_names').list.get(0).cast(pl.String).alias('opponent_pair_name_n'),
        pl.col('opponent_pair_names').list.get(1).cast(pl.String).alias('opponent_pair_name_s'),
    ).drop(['pair_acbl','pair_names','opponent_pair_names'])
    print_to_log_info(ns_df.head(1))

    ew_df = ew_df.with_columns(
        pl.col('pair_acbl').list.get(0).cast(pl.UInt32).alias('player_number_e'),
        pl.col('pair_acbl').list.get(1).cast(pl.UInt32).alias('player_number_w'),
        pl.col('pair_names').list.get(0).cast(pl.String).alias('player_name_e'),
        pl.col('pair_names').list.get(1).cast(pl.String).alias('player_name_w'),
        pl.col('opponent_pair_names').list.get(0).cast(pl.String).alias('opponent_pair_name_e'),
        pl.col('opponent_pair_names').list.get(1).cast(pl.String).alias('opponent_pair_name_w'),
    ).drop(['pair_acbl','pair_names','opponent_pair_names'])
    print_to_log_info(ew_df.head(1))

    ew_cols = [
        'player_number_e','player_number_w',
        'player_name_e','player_name_w',
        'opponent_pair_name_e','opponent_pair_name_w',
        'match_points_ew',
        'percentage_ew',
        'board_number',
        'pair_number',
        'opponent_pair_number',
        ]
    df = ns_df.join(ew_df[ew_cols],left_on=['pair_number','opponent_pair_number','board_number'],right_on=['opponent_pair_number','pair_number','board_number'],how='left')
    print_to_log_info(df.head(1))
    assert df.height == ns_df.height == ew_df.height # todo: pro tip. always assert heights after joins. it can save hours of debugging.

    # using df['section_results'].to_frame() because explode() creates a duplicate field 'session_id' unless selected.
    # section_results columns:
    # ['session_id', 'section_label', 'pair_number', 'team_number', 'orientation', 'strat', 'next_session_assignment',
    #     'next_session_qualification', 'score_cumulative', 'score_carryover', 'score_adjustment', 'percentage', 'mp_won', 'mp_color',
    #     'section_rank_strat_1', 'section_rank_strat_2', 'section_rank_strat_3',
    #     'overall_rank_strat_1', 'overall_rank_strat_2', 'overall_rank_strat_3', 'players', 'score_section']
    df_section_results = df['section_results'].to_frame().explode('section_results').unnest('section_results')

    # from mlBridgeLib. todo: convert to polars
    # def hrs_to_brss(hrs,void='',ten='10'):
    #     cols = [d+'_'+s for d in ['north','west','east','south'] for s in ['spades','hearts','diamonds','clubs']] # remake of hands below, comments says the order needs to be NWES?????
    #     return hrs[cols].apply(lambda r: ''.join(['SHDC'[i%4]+c for i,c in enumerate(r.values)]).replace(' ','').replace('-',void).replace('10',ten), axis='columns')

    df_handrecord = results_dfs_d['handrecord']
    df_handrecord = df_handrecord.with_columns(
        pl.Series('board_record_string',hrs_to_brss(df_handrecord.to_pandas()),pl.Utf8), # todo: eliminate polars to pandas conversion and back
    )

    df = df.join(df_handrecord['board_number','board_record_string','dealer','vulnerability'],on='board_number',how='left')

    df = df.rename({
        'board_number': 'Board',
        #'club_id_number': 'Club',
        'contract': 'Contract',
        #'game_date': 'Date',
        'match_points_ns': 'MP_NS',
        'match_points_ew': 'MP_EW',
        'pair_number': 'Pair_Number_NS',
        'opponent_pair_number': 'Pair_Number_EW',
        #'percentage_ns': 'Final_Standing_NS', # percentage_ns is not a column in the data. use percentage instead?
        #'percentage_ew': 'Final_Standing_EW', # percentage_ew is not a column in the data. use percentage instead?
        #'result': 'Result',
        #'round_number': 'Round',
        #'score_ns': 'Score_NS',
        #'score_ew': 'Score_EW',
        'score': 'Score_NS',
        'section_label': 'section_name', # change to section_name for compatibility with clubs
        #'table_number': 'Table',
        #'tricks_taken': 'Tricks',
    })

    df = df.with_columns([
        (pl.col('percentage_ns')/100).cast(pl.Float32).alias('Pct_NS'),
        (pl.col('percentage_ew')/100).cast(pl.Float32).alias('Pct_EW'),
        #pl.col('board_record_string').cast(pl.Utf8),
        pl.lit(results_dfs_d['event']['start_date'].to_list()[0]).str.strptime(pl.Date, format="%Y-%m-%d").alias('Date'), # tournament start_date has only date so can't use %Y-%m-%d
        pl.lit(results_dfs_d['event']['id'].to_list()[0]).alias('event_id'),
        #pl.col('hand_record_id').cast(pl.Int64), # only box_number is present in the data
        pl.col('Pair_Number_NS').cast(pl.UInt8),
        pl.col('Pair_Number_EW').cast(pl.UInt8),
        #pl.col('pair_summary_id').alias('pair_summary_id_ns'), # todo: remove this alias after ai model is updated to use pair_summary_id
        #pl.col('pair_summary_id').alias('pair_summary_id_ew'), # todo: remove this alias after ai model is updated to use pair_summary_id
    ])

    df = acbldf_to_mldf(df) # todo: temporarily convert to pandas to use augment_df until clean_validate_df is converted to polars

    return df


def acbldf_to_mldf(df: pl.DataFrame) -> pl.DataFrame:
    # Rename columns
    df = df.rename({'declarer': 'Declarer_Direction'})
    df = df.with_columns(pl.col('Declarer_Direction').replace_strict(Direction_to_NESW_d,return_dtype=pl.String))

    # Drop rows where 'Board' is NaN
    df = df.filter(pl.col('Board').is_not_null() & pl.col('Board').gt(0))

    # Convert 'Board' to UInt8.
    # todo: use UInt32 instead of UInt8?
    df = df.with_columns(pl.col('Board').cast(pl.UInt8))

    if 'board_record_string' in df.columns:
        df = df.with_columns(pl.col('board_record_string').map_elements(brs_to_pbn,return_dtype=pl.Utf8).alias('PBN'))

    df = df.rename({'dealer': 'Dealer'})
    df = df.with_columns(pl.col('Dealer').replace_strict(Direction_to_NESW_d,return_dtype=pl.String))

    # todo: Shouldn't MP_Top/Pct_NS/Pct_EW calculations be left to mlBridgeAugmentLib?
    # This is flawed. We need 2 of 3 of (Pct_NS,Pct_EW,MP_Top) or (MP_NS,MP_EW,MP_Top) to be present to calculate the others.
    for col in ['Pct_NS','Pct_EW','MP_NS','MP_EW','MP_Top']:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float32))

    if 'MP_Top' not in df.columns:
        # Calculate 'MP_Top'
        df = df.with_columns([
            # which is better?
            #pl.col('MP_NS').count().over('Board').sub().cast(pl.UInt32).alias('MP_Top')
            pl.col("MP_NS").add(pl.col('MP_EW')).cast(pl.Float64).round(0).cast(pl.UInt32).alias("MP_Top"),
        ])

    # Calculate percentages. strange values and with multiple section computations. Can't all be director's adjustments?
    # todo: Shouldn't this be done in mlBridgeAugmentLib?
    if 'Pct_NS' not in df.columns and 'Pct_EW' not in df.columns:
        df = df.with_columns([
            (pl.col('MP_NS') / pl.col('MP_Top')).cast(pl.Float32).alias('Pct_NS'),
            #(pl.col('MP_EW').cast(pl.Float32) / pl.col('MP_Top')).alias('Pct_EW')
        ])
        df = df.with_columns([
            pl.when(pl.col('Pct_NS') > 1).then(1).otherwise(pl.col('Pct_NS')).alias('Pct_NS'),
            #pl.when(pl.col('Pct_EW') > 1).then(1).otherwise(pl.col('Pct_EW')).alias('Pct_EW')
        ])
        df = df.with_columns([
            (1 - pl.col('Pct_NS')).cast(pl.Float32).alias('Pct_EW'),
        ])
    else:   
        # Cap percentages at 1
        # todo: is > 1 really a thing?
        df = df.with_columns([
            pl.when(pl.col('Pct_NS') > 1).then(1).otherwise(pl.col('Pct_NS')).alias('Pct_NS'),
            pl.when(pl.col('Pct_EW') > 1).then(1).otherwise(pl.col('Pct_EW')).alias('Pct_EW')
        ])
        # I've seen some seemingly correct Pct_NS but null for Pct_EW. Mystery. Can't all be director's adjustments?
        df = df.with_columns([
            (1 - pl.col('Pct_NS')).cast(pl.Float32).alias('Pct_EW'),
        ])


    # Function to transform names into "first last" format
    def last_first_to_first_last(name):
        # Replace commas with spaces and split
        parts = name.replace(',', ' ').split()
        # Return "first last" format
        return ' '.join(parts[1:] + parts[:1]) if len(parts) > 1 else name

    # Transpose player names
    for d in 'NESW':
        df = df.rename({f'player_number_{d.lower()}': f'Player_ID_{d}'})
        df = df.rename({f'player_name_{d.lower()}': f'Player_Name_{d}'})
        df = df.with_columns([
            pl.col(f'Player_ID_{d}').cast(pl.Utf8).alias(f'Player_ID_{d}'),
            #pl.col(f'Player_ID_{d}').cast(pl.UInt32).alias(f'iPlayer_Number_{d}'),
            pl.col(f'Player_Name_{d}').map_elements(last_first_to_first_last, return_dtype=pl.Utf8).alias(f'Player_Name_{d}')
        ])

    # chatgpt suggested this. nothing else looked as good. seems like something else would be better. it's one thing where pandas codes better than polars.
    df = df.with_columns(
        pl.when(pl.col("Declarer_Direction") == "N").then(pl.col("Player_ID_N"))
        .when(pl.col("Declarer_Direction") == "E").then(pl.col("Player_ID_E"))
        .when(pl.col("Declarer_Direction") == "S").then(pl.col("Player_ID_S"))
        .when(pl.col("Declarer_Direction") == "W").then(pl.col("Player_ID_W"))
        .otherwise(None)
        .alias("Number_Declarer")
    )

    # Clean up contracts
    # todo: Shouldn't this be done in mlBridgeAugmentLib?
    df = df.with_columns(
           pl.col('Contract')
            .str.replace(' ', '',n=2)
            .str.to_uppercase()
            .str.replace('NT', 'N')
            .alias('Contract')
        )

    # drop rows where 'Score' is null. They are different from 'PASS' in 'Score'.
    df = df.filter(pl.col('Score_NS').is_not_null())
    # 'Score' is a string if no 'PASS' otherwise it might be an int (i64).
    if df.schema['Score_NS'] == pl.Utf8:
        # acbl tournament data has 'PASS' in 'score'.
        df = df.with_columns(
            pl.when(pl.col('Score_NS') == 'PASS')
            .then(pl.lit('PASS'))
            .otherwise(pl.col('Contract'))
            .alias('Contract')
        )
        # Drop invalid contracts
        drop_rows = (
            (pl.col('Contract') != 'PASS') & 
            (pl.col('Contract').is_null() | pl.col('Score_NS').is_null())
        )
        print_to_log_info('Dropping rows:',df.filter(drop_rows))
        df = df.filter(~drop_rows)
        df = df.with_columns(pl.col('Score_NS').str.replace('PASS', '0'))
    df = df.with_columns(pl.col('Score_NS').cast(pl.Int16,strict=False).alias('Score_NS'))
    df = df.with_columns(pl.col('Score_NS').neg().cast(pl.Int16,strict=False).alias('Score_EW'))

    # todo: Shouldn't this be done in mlBridgeAugmentLib?
    df = df.with_columns([
        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.lit(None))
        .otherwise(pl.col('Contract').str.slice(0, 1))
        .cast(pl.UInt8,strict=False)
        .alias('BidLvl'),

        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.lit(None))
        .otherwise(pl.col('Contract').str.slice(1, 1))
        .cast(pl.String)
        .alias('BidSuit'),

        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.lit(None))
        .otherwise(pl.col('Contract').str.slice(2))
        .cast(pl.String)
        .alias('Dbl'),
    ])

    # reformat contract to standard format. Using endplay's contract format.
    # todo: Shouldn't this be done in mlBridgeAugmentLib?
    df = df.with_columns([
        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.col('Contract'))
        .otherwise(pl.col('BidLvl').cast(pl.Utf8)+pl.col('BidSuit')+pl.col('Dbl')+pl.col('Declarer_Direction'))
        .cast(pl.String)
        .alias('Contract'),
    ])

    if 'vulnerability' in df.columns:
        df = df.rename({'vulnerability':'Vul'})
        df = df.with_columns([
            pl.col('Vul').replace_strict(Vulnerability_to_Vul_d,return_dtype=pl.Utf8)
        ])
   
    if 'iVul' not in df.columns and 'Vul' not in df.columns:
        df = df.with_columns([
            pl.col('Board').map_elements(BoardNumberToVul,return_dtype=pl.UInt8).alias('iVul'),
        ])

    if 'Vul' not in df.columns:
        df = df.with_columns([
            pl.col('iVul').replace_strict(Vulnerability_to_Vul_d,return_dtype=pl.Utf8).alias('Vul')
        ])

    if 'iVul' not in df.columns:
        df = df.with_columns([
            pl.col('Vul').replace_strict(vul_sym_to_index_d,return_dtype=pl.UInt8).alias('iVul'),
        ])

    # Create 'Result' and 'Tricks' columns
    if 'Result' in df.columns:
        df = df.with_columns([
            pl.when(pl.col('Result').is_not_null()) # todo: make unrecognized data into null? added '+' to list because of #96851 Duncan Open. A '+' was manually entered in results as part of an adjusted score.
            .then(pl.col('Result').map_elements(lambda x: 0 if x in ['=', '0', '', '+'] else int(x[1:]) if x[0] == '+' else int(x),return_dtype=pl.Int8))
            .cast(pl.Int8)
            .alias('Result')
        ])
    else:
        assert 'Tricks' not in df.columns
        df = df.with_columns(pl.Series('scores_l',ContractToScores(df),pl.List(pl.Int16)))

        # todo: use mlBridgeAugmentLib.ContractToScores?
        # adjusted score is the reason for any unexpected scores.
        df = df.with_columns(
            pl.Series('Result',[None if (None in r) or (r[1] not in r[2]) else r[2].index(r[1])-(r[3]+6) for r in df['Contract','Score_NS','scores_l','BidLvl'].rows()],dtype=pl.Int8),
        )
        # can be null if errata.
        df.filter(pl.col('Result').is_null())['Board','Contract','Score_NS','BidLvl','Vul','iVul','scores_l']
        df.drop_in_place('scores_l') # ugh, con.register() hangs unless this is done. Probably from when it was pl.Object.

    df = df.with_columns(
        pl.when((pl.col('Contract') == 'PASS') | (pl.col('Result').is_null())) # 'Result' can be null if errata.
        .then(pl.lit(None))
        .otherwise(pl.col('BidLvl') + 6 + pl.col('Result'))
        .alias('Tricks')

    )
    print_to_log_info('PASS:',df.filter(pl.col('Contract') == 'PASS').height)


    # Fill missing values
    if 'Round' in df.columns:
        df = df.with_columns([
            pl.col('Round').fill_null(0).cast(pl.UInt8),
        ])

    if 'tb_count' in df.columns:
        df = df.with_columns([
            pl.col('tb_count').fill_null(0).cast(pl.Float32), # could be half table
        ])

    if 'Table' in df.columns:
        df = df.with_columns([
            pl.col('Table').fill_null(0).cast(pl.UInt8)
        ])

    # Assert no columns start with 'ns_' or 'ew_'
    for col in df.columns:
        assert not (col.startswith('ns_') or col.startswith('ew_') or col.startswith('NS_') or col.startswith('EW_')), col

    assert len(df) > 0
    return df


# ============================================================================
# Playwright-based functions for retrieving ACBL club results
# Added to support modern anti-bot protection
# ============================================================================

# Constants for Playwright browser configuration
ACBL_PAGE_LOAD_TIMEOUT = 60000  # milliseconds
ACBL_VIEWPORT_WIDTH = 1920
ACBL_VIEWPORT_HEIGHT = 1080
ACBL_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def create_acbl_browser_context(p, headless=True):
    """
    Helper to create browser context with consistent settings for ACBL scraping.
    
    Args:
        p: Playwright instance
        headless: Run browser in headless mode
    
    Returns:
        tuple: (browser, context) objects
    """
    browser = p.chromium.launch(headless=headless)
    context = browser.new_context(
        viewport={'width': ACBL_VIEWPORT_WIDTH, 'height': ACBL_VIEWPORT_HEIGHT},
        user_agent=ACBL_USER_AGENT
    )
    return browser, context


def _run_in_thread_with_new_loop(func, *args, **kwargs):
    """Run function in a thread with a fresh event loop (Windows-compatible)."""
    import sys
    result_container = []
    exception_container = []
    
    def thread_worker():
        try:
            # Set Windows-compatible event loop policy
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
            # Create and set a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                result_container.append(result)
            finally:
                loop.close()
        except Exception as e:
            exception_container.append(e)
    
    thread = threading.Thread(target=thread_worker)
    thread.start()
    thread.join()
    
    if exception_container:
        raise exception_container[0]
    return result_container[0] if result_container else None


def _get_club_results_sync(acbl_number, headless=True, save_screenshot=None, limit=0, verbose=True):
    """
    Internal sync version that runs Playwright. 
    Use get_club_results_from_acbl_number_playwright() instead.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    
    url = f"https://my.acbl.org/club-results/my-results/{acbl_number}"
    if verbose:
        print(f"Fetching with Playwright: {url}")
    
    all_events = {}
    page_num = 1
    
    with sync_playwright() as p:
        browser, context = create_acbl_browser_context(p, headless)
        page = context.new_page()
        
        try:
            # Navigate to the page
            if verbose:
                print("  Loading page...")
            response = page.goto(url, wait_until='networkidle', timeout=ACBL_PAGE_LOAD_TIMEOUT)
            
            if response.status != 200:
                raise Exception(f"Failed to load page. Status code: {response.status}")
            
            # Take screenshot if requested
            if save_screenshot:
                page.screenshot(path=save_screenshot)
                if verbose:
                    print(f"  Screenshot saved to: {save_screenshot}")
            
            # Process all pages
            while True:
                if verbose:
                    print(f"  Processing page {page_num}...")
                
                # Wait for DataTables to render
                try:
                    page.wait_for_selector('.dataTables_paginate', timeout=5000)
                except:
                    pass  # Continue even if timeout
                
                html_content = page.content()
                events_on_page = parse_acbl_events_from_html(html_content, url)
                all_events.update(events_on_page)
                if verbose:
                    print(f"    Found {len(events_on_page)} event(s) on page {page_num}")
                
                # Check if we have enough events
                if limit > 0 and len(all_events) >= limit:
                    if verbose:
                        print(f"    Reached limit of {limit} events, stopping pagination")
                    break
                
                # Check if there's a "Next" button
                # Try multiple selectors for the Next button
                next_button_found = False
                selectors = [
                    'a.paginate_button.next',  # DataTables pagination (ACBL uses this)
                    '#DataTables_Table_0_next',  # DataTables Next button ID
                    'a.page-link:has-text("Next")',  # Bootstrap pagination
                    'button:has-text("Next")',
                    'a:has-text("Next")',
                    '.pagination a[aria-label="Next"]',
                    'a[rel="next"]',
                    '.next a',
                    'li.page-item.next a',
                ]
                
                for selector in selectors:
                    try:
                        next_button = page.locator(selector).first
                        if next_button.count() > 0:
                            # Check if button is visible and not disabled
                            if not next_button.is_visible():
                                continue
                            
                            class_attr = next_button.get_attribute('class')
                            if class_attr and 'disabled' in class_attr:
                                # Button is disabled, no more pages
                                continue
                            
                            if verbose:
                                print(f"    Found Next button with selector: {selector}")
                            next_button.click()
                            page.wait_for_load_state('networkidle', timeout=ACBL_PAGE_LOAD_TIMEOUT)
                            page_num += 1
                            time.sleep(0.5)  # Small delay to be respectful
                            next_button_found = True
                            break
                    except Exception as e:
                        # Try next selector
                        if verbose:
                            print(f"    Selector {selector} failed: {e}")
                        continue
                
                if not next_button_found:
                    if verbose:
                        print(f"    No Next button found, stopping pagination")
                    break
            
        finally:
            browser.close()
    
    if verbose:
        print(f"  Total events found across {page_num} page(s): {len(all_events)}")
    return all_events


def get_club_results_from_acbl_number_playwright(acbl_number, headless=True, save_screenshot=None, limit=0, verbose=True):
    """
    Retrieve club results for a specific ACBL member number using Playwright.
    Returns a dictionary of event data keyed by event ID.
    Automatically pages through all available pages to get complete results.
    
    This function is safe to call from async contexts (like Streamlit) as it runs
    Playwright in a separate thread with its own event loop.
    
    Args:
        acbl_number: ACBL member number (e.g., 2663279)
        headless: Run browser in headless mode (default: True)
        save_screenshot: Path to save screenshot (optional)
        limit: Maximum number of events to retrieve (0 = unlimited)
        verbose: Print progress messages (default: True)
    
    Returns:
        dict: Dictionary with event_id as key and tuple (my_results_url, details_url, msg, club_id) as value
    
    Example:
        >>> from mlBridgeLib.mlBridgeAcblLib import get_club_results_from_acbl_number_playwright
        >>> results = get_club_results_from_acbl_number_playwright(2663279, limit=5)
        >>> for event_id, (url, details_url, msg, club_id) in results.items():
        ...     print(f"Event {event_id}: {msg}")
    """
    # Run in a thread with a fresh event loop to avoid conflicts with Streamlit
    return _run_in_thread_with_new_loop(_get_club_results_sync, acbl_number, headless, save_screenshot, limit, verbose)


def parse_acbl_events_from_html(html_content, base_url):
    """
    Helper function to parse events from ACBL HTML content.
    Returns a dictionary of event data keyed by event ID.
    
    Args:
        html_content: HTML content string
        base_url: Base URL for the my-results page
    
    Returns:
        dict: Dictionary with event_id as key and tuple (base_url, details_url, msg, club_id) as value
    """
    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find all anchor tags with href attributes pointing to club results details
    anchor_pattern = re.compile(r'/club-results/details/\d+$')
    anchor_tags = soup.find_all('a', href=anchor_pattern)
    
    if not anchor_tags:
        # Empty page or no results on this page
        return {}
    
    anchor_d = {a['href']: a for a in anchor_tags}
    sorted_anchor_d = dict(sorted(
        {int(k.split('/')[-1]): v for k, v in anchor_d.items()}.items(), 
        reverse=True
    ))
    
    # Extract event information from table rows
    msgs = []
    club_ids = []
    for k, v in sorted_anchor_d.items():
        try:
            td_elements = v.parent.parent.find_all('td')
            if len(td_elements) >= 6:
                msg = ', '.join([
                    td_elements[i].text.replace('\n', '').strip() 
                    for i in [0, 1, 2, 3, 5]  # event_id, date, club, event, score
                ])
                msgs.append(msg)
                
                # Extract club_id from any column that has a club-results link
                club_id_found = None
                for td in td_elements:
                    club_link = td.find('a', href=re.compile(r'/club-results/\d+$'))
                    if club_link and 'href' in club_link.attrs:
                        # Extract club ID from href like "/club-results/267096"
                        club_href = club_link['href']
                        club_id_match = re.search(r'/club-results/(\d+)$', club_href)
                        if club_id_match:
                            club_id_found = club_id_match.group(1)
                            break
                club_ids.append(club_id_found)
        except (AttributeError, IndexError):
            msgs.append(f"Event {k} (parsing error)")
            club_ids.append(None)
    
    # Build results dictionary
    my_results_details_data = {}
    for (event_id, href), msg, club_id in zip(sorted_anchor_d.items(), msgs, club_ids):
        detail_url = 'https://my.acbl.org' + href['href']
        my_results_details_data[event_id] = (base_url, detail_url, msg, club_id)
    
    return my_results_details_data


def extract_json_from_var_data(html_content):
    """
    Extract JSON data from 'var data =' assignment in JavaScript.
    Returns the parsed JSON data or None if not found.
    
    Args:
        html_content: HTML content string containing JavaScript
    
    Returns:
        dict or None: Parsed JSON data or None if not found
    """
    # Look for patterns like: var data = {...}; or var data={...};
    
    # Pattern to match var data = {...};
    patterns = [
        r'var\s+data\s*=\s*(\{[^;]*\});',  # var data = {...};
        r'var\s+data\s*=\s*(\{.*?\});',     # var data = {...}; (non-greedy)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    return None


def _get_club_results_details_sync(url, headless=True, verbose=True):
    """
    Internal sync version that runs Playwright.
    Use get_club_results_details_data_playwright() instead.
    """
    from playwright.sync_api import sync_playwright
    
    if verbose:
        print(f"  Fetching details with Playwright: {url}")
    
    with sync_playwright() as p:
        browser, context = create_acbl_browser_context(p, headless)
        page = context.new_page()
        
        try:
            response = page.goto(url, wait_until='networkidle', timeout=ACBL_PAGE_LOAD_TIMEOUT)
            
            if response.status != 200:
                raise Exception(f"Failed to load page. Status code: {response.status}")
            
            # Get page content immediately - no need to wait for specific components
            # since we have reliable extraction methods that work with the raw HTML
            html_content = page.content()
            
        finally:
            browser.close()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Method 1: Try to find the data attribute in various result detail tags
    data = None
    if soup.find('result-details-combined-section'):
        data = soup.find('result-details-combined-section').get('v-bind:data')
    elif soup.find('result-details'):
        data = soup.find('result-details').get('v-bind:data')
    elif soup.find('team-result-details'):
        data = soup.find('team-result-details').get('v-bind:data')
    
    if data and isinstance(data, str) and len(data) > 0:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            pass
    
    # Method 2: Try to extract from 'var data =' in script
    var_data = extract_json_from_var_data(html_content)
    if var_data:
        return var_data
    
    # No data found
    return None


def get_club_results_details_data_playwright(url, headless=True, verbose=True):
    """
    Retrieve detailed data for a specific club event using Playwright.
    Returns the parsed JSON data embedded in the HTML page.
    
    This function is safe to call from async contexts (like Streamlit) as it runs
    Playwright in a separate thread with its own event loop.
    
    Args:
        url: URL to the club results details page (e.g., https://my.acbl.org/club-results/details/993420)
        headless: Run browser in headless mode (default: True)
        verbose: Print progress messages (default: True)
    
    Returns:
        dict or None: Parsed JSON data from the page or None if not found
    
    Example:
        >>> from mlBridgeLib.mlBridgeAcblLib import get_club_results_details_data_playwright
        >>> details = get_club_results_details_data_playwright("https://my.acbl.org/club-results/details/993420")
        >>> print(details['event_name'])
    """
    # Run in a thread with a fresh event loop to avoid conflicts with Streamlit
    return _run_in_thread_with_new_loop(_get_club_results_details_sync, url, headless, verbose)

