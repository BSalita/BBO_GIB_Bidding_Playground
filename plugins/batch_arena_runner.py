from __future__ import annotations

import json
import math
import pathlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import quote, urljoin, urlparse

import pandas as pd
import requests

from bbo_bidding_queries_lib import (
    calculate_imp,
    get_ai_contract,
    get_dd_score_for_auction,
    get_dd_tricks_for_auction,
    get_declarer_for_auction,
    get_ev_for_auction_pre,
    normalize_auction_input,
    parse_contract_from_auction,
)
from plugins.auction_error_taxonomy import (
    add_auction_error_taxonomy_to_row,
    build_auction_error_summary_df,
)
from plugins.bbo_handlers_common import format_par_contracts


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BATCH_ARENA_EXPORT_ROOT = PROJECT_ROOT / "data"
DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_API_TIMEOUT = 60
_PBN_VUL_MAP = {
    "None": "None",
    "Love": "None",
    "-": "None",
    "NS": "NS",
    "N-S": "NS",
    "EW": "EW",
    "E-W": "EW",
    "All": "Both",
    "Both": "Both",
    "B": "Both",
}

ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class BatchRunnerConfig:
    api_base: str = DEFAULT_API_BASE
    seed: int = 0
    auction_filter_mode: str = "All Auctions"
    ai_logic_mode: str = "all_logic"
    ai_logic_mode_label: str = "All Logic"
    use_guardrails_v2: bool = False
    max_steps: int = 40
    top_n: int = 12
    max_deals: int = 500
    permissive_pass: bool = True
    start_timeout_s: int = 15
    lookup_timeout_s: int = 30
    status_timeout_s: int = 10
    bt_mean_timeout_s: int = 5
    max_polls: int = 300
    poll_interval_s: float = 0.5


@dataclass(slots=True)
class BatchRunResult:
    results: list[dict[str, Any]]
    deal_row_map: dict[str, dict[str, Any]]
    ai_steps_map: dict[str, list[dict[str, Any]]]
    batch_debug_entries: list[dict[str, Any]]
    batch_debug_payload: dict[str, Any]
    export_paths: dict[str, pathlib.Path]
    input_cfg: dict[str, Any]


class BatchArenaApiClient:
    def __init__(self, *, api_base: str = DEFAULT_API_BASE, default_timeout: int = DEFAULT_API_TIMEOUT) -> None:
        self.api_base = str(api_base or DEFAULT_API_BASE).rstrip("/")
        self.default_timeout = int(default_timeout)

    def get(self, path: str, *, timeout: int | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        resp = requests.get(f"{self.api_base}{path}", timeout=timeout or self.default_timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data["_client_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return data

    def post(self, path: str, payload: dict[str, Any], *, timeout: int | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        resp = requests.post(f"{self.api_base}{path}", json=payload, timeout=timeout or self.default_timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise requests.HTTPError(f"{exc}\nServer detail: {detail}", response=resp) from exc
        data = resp.json()
        if isinstance(data, dict):
            data["_client_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return data


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    tmp.replace(path)


def _json_safe_export_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe_export_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_export_value(v) for v in value]
    try:
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return str(value)


def _sanitize_filename_component(value: Any, *, allow_plus: bool = False) -> str:
    s = str(value or "").strip()
    if not s:
        return "unknown"
    allowed = r"A-Za-z0-9._"
    if allow_plus:
        allowed += r"\+"
    allowed += r"-"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(rf"[^{allowed}]", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "unknown"


def _sanitize_range_expr_for_filename(ranges_text: Any) -> str:
    s = str(ranges_text or "").strip()
    if not s:
        return "all"
    s = re.sub(r"\s*,\s*", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_+\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "all"


def _batch_result_export_path(*, batch_mode: str, input_cfg: dict[str, Any]) -> pathlib.Path:
    source_type = str(input_cfg.get("source_type") or "").strip().lower()
    if batch_mode == "Deals DF":
        range_token = _sanitize_range_expr_for_filename(input_cfg.get("ranges_text"))
        return BATCH_ARENA_EXPORT_ROOT / "deals" / f"result_{range_token}.csv"

    source_name = str(input_cfg.get("source_name") or "").strip()
    filename_regex = str(input_cfg.get("filename_regex") or "").strip()
    if batch_mode == "PBN Local/URL":
        subdir = "pbn"
        if source_type == "url":
            stem = _sanitize_filename_component(source_name, allow_plus=True)
        else:
            stem = _sanitize_filename_component(source_name or "pbn_input")
        if filename_regex:
            stem = f"{stem}__rx_{_sanitize_filename_component(filename_regex)}"
        return BATCH_ARENA_EXPORT_ROOT / subdir / f"result_{stem}.csv"

    subdir = "csv"
    if source_type == "url":
        stem = _sanitize_filename_component(source_name, allow_plus=True)
        if not stem.lower().endswith(".csv"):
            stem = f"{stem}.csv"
    else:
        stem = _sanitize_filename_component(source_name or "input.csv")
        if not stem.lower().endswith(".csv"):
            stem = f"{stem}.csv"
    return BATCH_ARENA_EXPORT_ROOT / subdir / f"result_{stem}"


def _batch_results_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "Source" in df.columns:
        cols = ["Source"] + [c for c in df.columns if c != "Source"]
        df = df.loc[:, cols]
    return df


def _auto_export_batch_results(
    *,
    all_results: list[dict[str, Any]],
    batch_mode: str,
    input_cfg: dict[str, Any],
    seed: int,
    auction_filter_mode: str,
    ai_logic_mode: str,
    ai_logic_mode_label: str,
    use_guardrails_v2: bool,
) -> dict[str, pathlib.Path]:
    export_path = _batch_result_export_path(batch_mode=batch_mode, input_cfg=input_cfg)
    json_path = export_path.with_suffix(".json")
    results_df = _batch_results_to_dataframe(all_results)
    csv_text = results_df.to_csv(index=False)
    _atomic_write_text(export_path, csv_text)

    export_payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": "ai_model_batch_arena",
        "batch_mode": batch_mode,
        "settings": {
            "seed": int(seed),
            "auction_filter_mode": auction_filter_mode,
            "ai_logic_mode": ai_logic_mode,
            "ai_logic_mode_label": ai_logic_mode_label,
            "use_guardrails_v2": bool(use_guardrails_v2),
        },
        "input": {
            "source_type": input_cfg.get("source_type"),
            "source_name": input_cfg.get("source_name"),
            "filename_regex": input_cfg.get("filename_regex"),
            "ranges_text": input_cfg.get("ranges_text"),
            "indices": input_cfg.get("indices"),
            "boards": input_cfg.get("boards"),
        },
        "result_files": {
            "csv_path": str(export_path),
            "json_path": str(json_path),
        },
        "result_summary": {
            "deal_count": len(all_results),
            "columns": list(results_df.columns),
        },
        "results_csv_text": csv_text,
        "results_rows": all_results,
    }
    _atomic_write_text(
        json_path,
        json.dumps(_json_safe_export_value(export_payload), indent=2),
    )
    return {"csv_path": export_path, "json_path": json_path}


def _batch_arena_result_row_key(
    *,
    deal_idx: Any,
    row_source: str,
    row_pos: int,
    use_external: bool,
    row_idx: Any = None,
) -> str:
    source_part = str(row_source or "").strip() or "local"
    deal_part = str(deal_idx if deal_idx is not None else "?").strip() or "?"
    if use_external:
        return f"ext:{source_part}:{deal_part}:{int(row_pos)}"
    if row_idx is not None:
        try:
            return f"db_row:{int(row_idx)}"
        except Exception:
            pass
    return f"db:{deal_part}:{int(row_pos)}"


def _hand_hcp(hand: str) -> int:
    hcp_map = {"A": 4, "K": 3, "Q": 2, "J": 1}
    return sum(hcp_map.get(c, 0) for c in str(hand or "").upper() if c in hcp_map)


def _hand_total_points(hand: str) -> int:
    hcp = _hand_hcp(hand)
    suits = str(hand or "").split(".")
    dist = sum(max(0, 3 - len(s)) for s in suits)
    return hcp + dist


def _json_safe_deal_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (bool, int, str)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = None if (math.isnan(v) or math.isinf(v)) else v
        else:
            try:
                if pd.isna(v):
                    out[k] = None
                    continue
            except Exception:
                pass
            try:
                out[k] = int(v)
            except Exception:
                try:
                    fv = float(v)
                    out[k] = None if (math.isnan(fv) or math.isinf(fv)) else fv
                except Exception:
                    out[k] = str(v)
    return out


def _pbn_deal_tag_to_hands(val: str, dealer: str) -> dict[str, str] | None:
    parts = str(val or "").strip().split(":")
    if len(parts) != 2:
        return None
    deal_dealer = str(parts[0] or dealer or "N").strip().upper()[:1]
    hands = [h.strip() for h in str(parts[1] or "").split()]
    if len(hands) != 4:
        return None
    seat_order = ["N", "E", "S", "W"]
    try:
        start_idx = seat_order.index(deal_dealer)
    except ValueError:
        start_idx = seat_order.index(str(dealer or "N").upper()[:1]) if str(dealer or "N").upper()[:1] in seat_order else 0
    mapped: dict[str, str] = {}
    for offset, hand in enumerate(hands):
        seat = seat_order[(start_idx + offset) % 4]
        mapped[seat] = hand
    return mapped


def _pbn_bids_to_dash(tokens: list[str]) -> str:
    out: list[str] = []
    for tok in tokens:
        t = str(tok or "").strip()
        if not t:
            continue
        u = t.upper()
        if u in {"PASS", "P"}:
            out.append("P")
        elif u in {"X", "XX"}:
            out.append(u)
        elif re.fullmatch(r"[1-7](NT|N|[CDHS])", u):
            out.append(u.replace("NT", "N"))
        else:
            out.append(u)
    return "-".join(out)


def parse_pbn_file_to_boards(content: str) -> list[dict[str, Any]]:
    tag_re = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
    boards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    auction_tokens: list[str] = []
    in_auction = False

    def _flush() -> None:
        if not current.get("_deal"):
            return
        dealer = current.get("dealer", "N")
        hands = current["_deal"]
        vul_raw = current.get("vul_raw", "None")
        seat_order = ["N", "E", "S", "W"]
        try:
            dealer_idx = seat_order.index(str(dealer).upper()[:1])
        except ValueError:
            dealer_idx = 0
        pbn_hands = [hands.get(seat_order[(dealer_idx + offset) % 4], "") for offset in range(4)]
        board: dict[str, Any] = {
            "board": current.get("board"),
            "dealer": dealer,
            "vul": _PBN_VUL_MAP.get(str(vul_raw), str(vul_raw)),
            "Hand_N": hands.get("N", ""),
            "Hand_E": hands.get("E", ""),
            "Hand_S": hands.get("S", ""),
            "Hand_W": hands.get("W", ""),
            "auction": _pbn_bids_to_dash(auction_tokens),
            "pbn": f"{dealer}:{' '.join(pbn_hands)}",
        }
        if current.get("Declarer"):
            board["Declarer"] = current["Declarer"]
        if current.get("Contract"):
            board["Contract"] = current["Contract"]
        pbn_result = current.get("Result")
        pbn_contract = current.get("Contract")
        if pbn_result is not None:
            board["Tricks"] = pbn_result
            if pbn_contract:
                try:
                    level = int(str(pbn_contract)[0])
                    board["Result"] = int(pbn_result) - (level + 6)
                except Exception:
                    pass
        boards.append(board)

    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line:
            in_auction = False
            continue
        m = tag_re.match(line)
        if m:
            in_auction = False
            tag, val = m.group(1), m.group(2)
            if tag in ("Board", "Event") and current.get("_deal") and tag == "Board":
                _flush()
                current = {}
                auction_tokens = []
            if tag == "Board":
                try:
                    current["board"] = int(val)
                except ValueError:
                    current["board"] = None
            elif tag == "Dealer":
                current["dealer"] = val.upper()[:1]
            elif tag == "Vulnerable":
                current["vul_raw"] = val
            elif tag == "Deal":
                dealer = current.get("dealer", "N")
                hands = _pbn_deal_tag_to_hands(val, dealer)
                if hands:
                    current["_deal"] = hands
            elif tag == "Result":
                try:
                    current["Result"] = int(val)
                except Exception:
                    pass
            elif tag == "Contract":
                current["Contract"] = val.strip()
            elif tag == "Declarer":
                d = val.strip().upper()[:1]
                if d in "NESW":
                    current["Declarer"] = d
            elif tag == "Auction":
                in_auction = True
                auction_tokens = []
        elif in_auction:
            for tok in line.split():
                if tok.startswith(";"):
                    break
                auction_tokens.append(tok)

    _flush()
    return boards


def _annotate_boards_with_source_name(boards: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    source_s = str(source_name or "").strip()
    out: list[dict[str, Any]] = []
    for board in boards:
        row = dict(board)
        if source_s:
            row["Source"] = source_s
        out.append(row)
    return out


def _normalize_github_raw_url(url: str) -> str:
    fetch_url = str(url or "").strip()
    if "github.com" in fetch_url and "/blob/" in fetch_url:
        fetch_url = fetch_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return fetch_url


def _download_text_from_url(url: str, label: str) -> str:
    url = str(url or "").strip()
    if not url:
        raise ValueError(f"{label} URL is empty.")
    fetch_url = _normalize_github_raw_url(url)
    resp = requests.get(fetch_url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _compile_optional_filename_regex(pattern: str) -> re.Pattern[str] | None:
    pattern_s = str(pattern or "").strip()
    if not pattern_s:
        return None
    try:
        return re.compile(pattern_s, flags=re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid filename regex: {exc}") from exc


def _enumerate_github_directory_file_urls(directory_url: str) -> list[str] | None:
    parsed = urlparse(str(directory_url or "").strip())
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 5 or parts[2].lower() != "tree":
        return None
    owner, repo, _, ref = parts[:4]
    repo_path = "/".join(parts[4:])
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(repo_path, safe='/')}?ref={quote(ref, safe='')}"
    resp = requests.get(api_url, timeout=30, headers={"Accept": "application/vnd.github+json"})
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError("GitHub directory URL did not return a directory listing.")
    file_urls: list[str] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type") or "").lower() != "file":
            continue
        download_url = str(entry.get("download_url") or "").strip()
        if download_url:
            file_urls.append(download_url)
    return file_urls


def _enumerate_indexed_directory_file_urls(directory_url: str) -> list[str]:
    html = _download_text_from_url(directory_url, "PBN directory")
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    urls: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        href_s = str(href or "").strip()
        if not href_s or href_s.startswith(("#", "javascript:", "mailto:")):
            continue
        full_url = urljoin(directory_url, href_s)
        parsed = urlparse(full_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean_url in seen:
            continue
        seen.add(clean_url)
        urls.append(clean_url)
    return urls


def _resolve_pbn_urls_from_input(url: str, filename_regex: str = "") -> tuple[list[str], bool]:
    url_s = str(url or "").strip()
    if not url_s:
        raise ValueError("PBN URL is empty.")
    filename_re = _compile_optional_filename_regex(filename_regex)
    normalized_url = _normalize_github_raw_url(url_s)
    path_lower = urlparse(normalized_url).path.lower()
    looks_like_file = path_lower.endswith((".pbn", ".lin"))
    looks_like_github_dir = "github.com" in url_s and "/tree/" in url_s
    if not looks_like_file:
        candidate_urls: list[str] | None = None
        if looks_like_github_dir:
            candidate_urls = _enumerate_github_directory_file_urls(url_s)
        if candidate_urls is None:
            candidate_urls = _enumerate_indexed_directory_file_urls(url_s)
        filtered: list[str] = []
        seen_filtered: set[str] = set()
        for candidate in candidate_urls:
            candidate_norm = _normalize_github_raw_url(candidate)
            basename = pathlib.PurePosixPath(urlparse(candidate_norm).path).name
            if not basename or not basename.lower().endswith((".pbn", ".lin")):
                continue
            if filename_re and not filename_re.search(basename):
                continue
            if candidate_norm in seen_filtered:
                continue
            seen_filtered.add(candidate_norm)
            filtered.append(candidate_norm)
        filtered.sort(key=lambda s: pathlib.PurePosixPath(urlparse(s).path).name.lower())
        if not filtered:
            regex_note = f" matching regex `{filename_regex}`" if str(filename_regex or "").strip() else ""
            raise ValueError(f"No .pbn/.lin files found in directory URL{regex_note}.")
        return filtered, True
    basename = pathlib.PurePosixPath(urlparse(normalized_url).path).name
    if filename_re and basename and not filename_re.search(basename):
        raise ValueError(f"Single-file URL basename `{basename}` does not match filename regex.")
    return [normalized_url], False


def load_pbn_boards_from_url_input(url: str, filename_regex: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    file_urls, is_directory_source = _resolve_pbn_urls_from_input(url, filename_regex=filename_regex)
    all_boards: list[dict[str, Any]] = []
    loaded_files: list[str] = []
    per_file_counts: list[tuple[str, int]] = []
    for file_url in file_urls:
        content = _download_text_from_url(file_url, "PBN")
        file_name = pathlib.PurePosixPath(urlparse(file_url).path).name
        boards = _annotate_boards_with_source_name(parse_pbn_file_to_boards(content), file_name)
        if not boards:
            continue
        all_boards.extend(boards)
        loaded_files.append(file_url)
        per_file_counts.append((file_name, len(boards)))
    if not all_boards:
        raise ValueError("No valid boards found in the provided PBN/LIN URL input.")
    return all_boards, {
        "is_directory_source": bool(is_directory_source),
        "loaded_file_urls": loaded_files,
        "loaded_file_names": [name for name, _ in per_file_counts],
        "loaded_file_count": len(loaded_files),
        "filename_regex": str(filename_regex or "").strip(),
        "per_file_board_counts": per_file_counts,
    }


def load_pbn_boards_from_file(path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p = pathlib.Path(path)
    content = p.read_text(encoding="utf-8", errors="replace")
    boards = _annotate_boards_with_source_name(parse_pbn_file_to_boards(content), p.name)
    if not boards:
        raise ValueError(f"No valid boards found in `{p}`.")
    return boards, {
        "is_directory_source": False,
        "loaded_file_urls": [],
        "loaded_file_names": [p.name],
        "loaded_file_count": 1,
        "filename_regex": "",
        "per_file_board_counts": [(p.name, len(boards))],
    }


def _get_dd_scoring_table() -> dict[tuple[int, str, int, bool], int]:
    import mlBridge.mlBridgeAugmentLib as ml_bridge_augment_lib

    score_tables = ml_bridge_augment_lib.precompute_contract_score_tables()
    if isinstance(score_tables, tuple):
        score_map = score_tables[1]
        if isinstance(score_map, dict):
            return score_map
    if isinstance(score_tables, dict):
        return score_tables
    raise TypeError("Unsupported score-table return type from precompute_contract_score_tables().")


def _canonicalize_pbn_to_north_prefix(pbn_str: str) -> str:
    """Normalize dealer-relative PBN hand order to `N:hand_N hand_E hand_S hand_W`."""
    pbn_s = str(pbn_str or "").strip()
    if ":" not in pbn_s:
        return pbn_s
    dealer_token, hands_token = pbn_s.split(":", 1)
    dealer = dealer_token.strip().upper()[:1]
    hands = hands_token.strip().split()
    if dealer not in "NESW" or len(hands) != 4:
        return pbn_s
    seat_order = ["N", "E", "S", "W"]
    dealer_idx = seat_order.index(dealer)
    by_seat = {seat_order[(dealer_idx + offset) % 4]: hands[offset] for offset in range(4)}
    return f"N:{by_seat['N']} {by_seat['E']} {by_seat['S']} {by_seat['W']}"


def _compute_board_dds_augmentation(pbn_str: str, dealer: str, vul: str) -> dict[str, Any]:
    try:
        from endplay.dds import calc_dd_table as calc_dd_table, par as endplay_par
        from endplay.types import Deal as EndplayDeal, Denom as Denom, Player as Player, Vul as EndplayVul
    except ImportError:
        return {}
    try:
        deal = EndplayDeal(pbn_str)
    except Exception:
        return {}
    result: dict[str, Any] = {}
    try:
        dd_table = calc_dd_table(deal)
    except Exception:
        return result
    dir_to_player = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}
    strain_to_denom = {"S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds, "C": Denom.clubs, "N": Denom.nt}
    for d in "NESW":
        for s in "SHDCN":
            try:
                result[f"DD_{d}_{s}"] = int(dd_table[dir_to_player[d], strain_to_denom[s]])  # type: ignore[index]
            except Exception:
                pass
    vul_map = {"None": EndplayVul.none, "NS": EndplayVul.ns, "EW": EndplayVul.ew, "Both": EndplayVul.both, "N_S": EndplayVul.ns, "E_W": EndplayVul.ew}
    dealer_map = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}
    vul_enum = vul_map.get(str(vul), EndplayVul.none)
    dealer_enum = dealer_map.get(str(dealer).upper(), Player.north)
    try:
        par_result = endplay_par(dd_table, vul_enum, dealer_enum)
        result["ParScore"] = int(par_result.score)
        result["ParContracts"] = [
            {
                "Level": str(contract.level),
                "Strain": "SHDCN"[int(contract.denom)],
                "Doubled": contract.penalty.abbr,
                "Pair_Direction": "NS" if contract.declarer.abbr in "NS" else "EW",
                "Result": contract.result,
            }
            for contract in par_result  # type: ignore[union-attr]
        ]
    except Exception:
        pass
    try:
        scores_d = _get_dd_scoring_table()
        vul_ns = str(vul) in ("NS", "Both", "N_S")
        vul_ew = str(vul) in ("EW", "Both", "E_W")
        for level in range(1, 8):
            for s in "CDHSN":
                for d in "NESW":
                    pair_vul = vul_ns if d in "NS" else vul_ew
                    tricks = result.get(f"DD_{d}_{s}")
                    if tricks is not None:
                        score = scores_d.get((level, s, tricks, pair_vul))
                        result[f"DD_Score_{level}{s}_{d}"] = score
    except Exception:
        pass
    return result


def _compute_board_sd_ev_augmentation(pbn_str: str, dealer: str, vul: str, produce: int = 100) -> dict[str, Any]:
    try:
        from mlBridge.mlBridgeAugmentLib import estimate_sd_trick_distributions as estimate_sd_trick_distributions
    except ImportError:
        return {}
    try:
        canonical_pbn = _canonicalize_pbn_to_north_prefix(pbn_str)
        _, (_, ns_ew_rows) = estimate_sd_trick_distributions(canonical_pbn, int(produce))
    except Exception:
        return {}
    result: dict[str, Any] = {}
    for (pair_dir, decl_dir, strain), probs in ns_ew_rows.items():
        for tricks, prob in enumerate(probs):
            result[f"Probs_{pair_dir}_{decl_dir}_{strain}_{tricks}"] = float(prob)
    try:
        scores_d = _get_dd_scoring_table()
        vul_ns = str(vul) in ("NS", "Both", "N_S")
        vul_ew = str(vul) in ("EW", "Both", "E_W")
        for pair_dir in ("NS", "EW"):
            pair_vul = vul_ns if pair_dir == "NS" else vul_ew
            for decl_dir in pair_dir:
                for strain in "SHDCN":
                    probs = ns_ew_rows.get((pair_dir, decl_dir, strain))
                    if probs is None:
                        continue
                    for level in range(1, 8):
                        ev = sum(
                            probs[tricks] * (scores_d.get((level, strain, tricks, pair_vul)) or 0.0)
                            for tricks in range(14)
                        )
                        result[f"EV_{pair_dir}_{decl_dir}_{strain}_{level}"] = ev
    except Exception:
        pass
    return result


def _contract_str(auction: str | None, dealer: str) -> str:
    if not auction:
        return ""
    toks = [t.strip().upper() for t in str(auction).split("-") if t.strip()]
    if toks and all(t == "P" for t in toks):
        return "Pass"
    c = get_ai_contract(auction, dealer)
    return str(c) if c else ""


def _dd_score_ns(auction: str | None, dealer: str, deal_row: dict[str, Any]) -> int | None:
    if not auction:
        return None
    toks = [t.strip().upper() for t in str(auction).split("-") if t.strip()]
    if toks and all(t == "P" for t in toks):
        return 0
    raw = get_dd_score_for_auction(auction, dealer, deal_row)
    if raw is None:
        return None
    decl = get_declarer_for_auction(auction, dealer)
    if decl and str(decl).upper() in ("E", "W"):
        return -int(raw)
    return int(raw)


def _imp_vs_par_signed(score_contract: int | None, par_score: int | None, auction: str | None, dealer: str) -> int | None:
    if score_contract is None or par_score is None:
        return None
    decl = get_declarer_for_auction(auction or "", dealer)
    par_for_contract_side = int(par_score)
    if decl and str(decl).upper() in ("E", "W"):
        par_for_contract_side = -par_for_contract_side
    diff = int(score_contract) - int(par_for_contract_side)
    sign = 1 if diff >= 0 else -1
    return sign * calculate_imp(abs(diff))


def _imp_diff(imp_actual: int | None, imp_ai: int | None) -> int | None:
    if imp_ai is None or imp_actual is None:
        return None
    return int(imp_ai) - int(imp_actual)


def _dd_imp_diff_signed(actual_score: int | None, ai_score: int | None) -> int | None:
    """Signed IMP from declarer-side AI DD score minus Actual DD score."""
    if actual_score is None or ai_score is None:
        return None
    diff = int(ai_score) - int(actual_score)
    sign = 1 if diff >= 0 else -1
    return sign * calculate_imp(abs(diff))


def _ev_imp_vs_par_signed(
    ev_declarer: float | None,
    par_score: int | None,
    auction: str | None,
    dealer: str,
) -> int | None:
    if ev_declarer is None or par_score is None:
        return None
    decl = get_declarer_for_auction(auction or "", dealer)
    par_for_declarer = float(par_score)
    if decl and str(decl).upper() in ("E", "W"):
        par_for_declarer = -par_for_declarer
    diff = float(ev_declarer) - par_for_declarer
    sign = 1 if diff >= 0 else -1
    return sign * calculate_imp(int(round(abs(diff))))


def _ev_declarer(auction: str | None, dealer: str, deal_row: dict[str, Any]) -> float | None:
    if not auction:
        return None
    toks = [t.strip().upper() for t in str(auction).split("-") if t.strip()]
    if toks and all(t == "P" for t in toks):
        return 0.0
    raw = get_ev_for_auction_pre(auction, dealer, deal_row)
    if raw is None:
        return None
    try:
        val = float(raw)
    except Exception:
        return None
    return round(val, 1)


def _is_pass_call(call: Any) -> bool:
    return str(call or "").strip().upper() in {"P", "PASS"}


def classify_actual_auction_competitive(auction: Any) -> bool | None:
    if auction is None:
        return None
    calls = [x.strip() for x in re.split(r"[-\s]+", str(auction)) if x and x.strip()]
    if not calls:
        return None
    opener_idx: int | None = None
    for i, call in enumerate(calls):
        if not _is_pass_call(call):
            opener_idx = i
            break
    if opener_idx is None:
        return None
    opener_parity = opener_idx % 2
    for i, call in enumerate(calls):
        if i % 2 != opener_parity and not _is_pass_call(call):
            return True
    return False


def _append_unique_bt_error_note(notes: list[str], note: Any, *, step: int | None = None) -> None:
    note_str = str(note or "").strip()
    if not note_str:
        return
    if step is not None:
        note_str = f"Step {step}: {note_str}"
    if note_str not in notes:
        notes.append(note_str)


def _append_reasonish_notes_from_dict(notes: list[str], data: Any, *, step: int | None = None, prefix: str = "") -> None:
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if value is None:
            continue
        label = f"{prefix}{key}: " if prefix else ""
        if str(key).endswith("_reason"):
            _append_unique_bt_error_note(notes, f"{label}{value}", step=step)
        elif str(key).endswith("_reasons") and isinstance(value, list):
            for item in value:
                _append_unique_bt_error_note(notes, f"{label}{item}", step=step)
        elif key in ("special_case_notes", "guard_reasons", "common_sense_reason_codes") and isinstance(value, list):
            for item in value:
                _append_unique_bt_error_note(notes, f"{label}{item}", step=step)


def _collect_special_case_bt_error_notes(ai_steps_detail_local: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for step_data in ai_steps_detail_local or []:
        if not isinstance(step_data, dict):
            continue
        step_no = step_data.get("step")
        try:
            step_no = int(step_no) if step_no is not None else None
        except Exception:
            step_no = None
        _append_reasonish_notes_from_dict(notes, step_data, step=step_no)
        for filtered in step_data.get("all_bids_filtered") or []:
            if not isinstance(filtered, dict):
                continue
            fbid = str(filtered.get("bid") or "").strip().upper()
            freason = str(filtered.get("filter_reason") or "").strip()
            if fbid and freason:
                _append_unique_bt_error_note(notes, f"blocked {fbid}: {freason}", step=step_no)
        chosen_bid = str(step_data.get("chosen_bid") or ((step_data.get("bidder_view") or {}).get("chosen_bid")) or "").strip().upper()
        if not chosen_bid:
            continue
        chosen_score_row = None
        for score_row in step_data.get("bid_scores") or []:
            if str((score_row or {}).get("bid") or "").strip().upper() == chosen_bid:
                chosen_score_row = score_row
                break
        if isinstance(chosen_score_row, dict):
            _append_reasonish_notes_from_dict(notes, chosen_score_row, step=step_no)
    return notes


def _build_batch_debug_payload(
    *,
    input_cfg: dict[str, Any],
    results: list[dict[str, Any]],
    batch_debug_entries: list[dict[str, Any]],
    seed: int,
    imp_running_total: int,
) -> dict[str, Any]:
    worst_n = 10
    sorted_entries = sorted(batch_debug_entries, key=lambda d: float(d.get("imp_diff") or 0.0))
    taxonomy_summary_df = build_auction_error_summary_df(results)
    worst_deals_summary: list[dict[str, Any]] = []
    for entry in sorted_entries[:worst_n]:
        imp = entry.get("imp_diff")
        if imp is None or imp >= 0:
            continue
        divergence = entry.get("divergence") or {}
        summary: dict[str, Any] = {
            "deal_index": entry.get("deal_index"),
            "imp_actual": entry.get("imp_actual"),
            "imp_ai": entry.get("imp_ai"),
            "imp_diff": imp,
            "actual_auction": entry.get("actual_auction"),
            "ai_auction": entry.get("ai_auction"),
            "actual_contract": entry.get("actual_contract"),
            "ai_contract": entry.get("ai_contract"),
            "dd_score_actual_ns": entry.get("dd_score_actual_ns"),
            "dd_score_ai_ns": entry.get("dd_score_ai_ns"),
            "par": entry.get("par"),
            "par_contracts": entry.get("par_contracts"),
        }
        if divergence:
            summary["divergence_step"] = divergence.get("step")
            summary["divergence_auction_so_far"] = divergence.get("auction_so_far")
            summary["actual_bid_at_divergence"] = divergence.get("actual_bid")
            summary["ai_bid_at_divergence"] = divergence.get("ai_bid")
            bid_scores = divergence.get("bid_scores") or []
            if bid_scores:
                summary["bid_scores_at_divergence"] = [
                    {
                        "bid": bid.get("bid"),
                        "score": bid.get("score"),
                        "guard_penalty": bid.get("guard_penalty"),
                        "pass_source": bid.get("pass_source"),
                    }
                    for bid in bid_scores[:5]
                ]
            summary["bt_node_at_divergence"] = {
                "auction_so_far": divergence.get("auction_so_far"),
                "seat": divergence.get("seat"),
                "seat_bt": divergence.get("seat_bt"),
                "ai_bid": divergence.get("ai_bid"),
                "actual_bid": divergence.get("actual_bid"),
                "bids_scored": [
                    {
                        "bid": bid.get("bid"),
                        "score": bid.get("score"),
                        "agg_expr": bid.get("agg_expr"),
                        "desc_score": bid.get("desc_score"),
                        "opp_threat": bid.get("opp_threat"),
                        "guard_penalty": bid.get("guard_penalty"),
                        "final_score": bid.get("final_score"),
                    }
                    for bid in bid_scores
                ],
                "bids_filtered": divergence.get("all_bids_filtered") or [],
            }
        deal = entry.get("deal") or {}
        summary["hands"] = {d: deal.get(f"Hand_{d}") for d in ("N", "E", "S", "W")}
        summary["hcp"] = {d: deal.get(f"HCP_{d}") for d in ("N", "E", "S", "W")}
        summary["dealer"] = entry.get("dealer")
        summary["vul"] = entry.get("vul")
        if entry.get("auction_error_taxonomy") is not None:
            summary["auction_error_taxonomy"] = entry.get("auction_error_taxonomy")
        worst_deals_summary.append(summary)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deal_ranges": input_cfg.get("ranges_text"),
        "deal_count": len(results),
        "seed": int(seed),
        "deals_processed": len(results),
        "imp_running_total": imp_running_total,
        "auction_error_summary": taxonomy_summary_df.to_dict("records"),
        "worst_deals": worst_deals_summary,
        "deals": batch_debug_entries,
    }


def write_batch_debug_json(batch_debug_payload: dict[str, Any], *, path: pathlib.Path | None = None) -> pathlib.Path:
    debug_path = path or (PROJECT_ROOT / "data" / "batch_arena_debug.json")
    _atomic_write_text(debug_path, json.dumps(_json_safe_export_value(batch_debug_payload), indent=2))
    return debug_path


def _progress(callback: ProgressCallback | None, current: int, total: int, message: str) -> None:
    if callback is not None:
        callback(int(current), int(total), str(message))


def run_batch_on_external_boards(
    *,
    boards: list[dict[str, Any]],
    input_cfg: dict[str, Any],
    config: BatchRunnerConfig,
    progress_callback: ProgressCallback | None = None,
) -> BatchRunResult:
    client = BatchArenaApiClient(api_base=config.api_base)
    external_boards = list(boards)
    if config.auction_filter_mode != "All Auctions":
        filtered: list[dict[str, Any]] = []
        for board in external_boards:
            cls = classify_actual_auction_competitive(board.get("auction"))
            if config.auction_filter_mode == "Non-competitive auctions only":
                if cls is False:
                    filtered.append(board)
            else:
                if cls is True:
                    filtered.append(board)
        external_boards = filtered

    if not external_boards:
        raise ValueError("No external records loaded after applying the auction filter.")

    results: list[dict[str, Any]] = []
    deal_row_map: dict[str, dict[str, Any]] = {}
    ai_steps_map: dict[str, list[dict[str, Any]]] = {}
    batch_debug_entries: list[dict[str, Any]] = []
    imp_running = 0
    dd_imp_running = 0
    ev_imp_running = 0
    t_batch_start = time.perf_counter()
    total = len(external_boards)

    for i, src_board in enumerate(external_boards):
        row_source = str(src_board.get("Source") or src_board.get("_source_file") or input_cfg.get("source_name") or "").strip()
        deal_idx = src_board.get("board") or (i + 1)
        result_row_key = _batch_arena_result_row_key(
            deal_idx=deal_idx,
            row_source=row_source,
            row_pos=i,
            use_external=True,
        )
        dealer = str(src_board.get("dealer", "N")).upper()
        vul = {"All": "Both", "None": "None", "NS": "NS", "EW": "EW"}.get(str(src_board.get("vul", "None")), str(src_board.get("vul", "None")))
        actual_auction = normalize_auction_input(str(src_board.get("auction", "") or "").strip())

        _progress(progress_callback, i, total, f"Board {i+1}/{total} (#{deal_idx}) lookup")
        try:
            lookup_resp = client.post("/pbn-lookup", {"pbn": src_board["pbn"], "max_results": 1}, timeout=config.lookup_timeout_s)
            matches = lookup_resp.get("matches") or []
        except Exception as exc:
            results.append({"Source": row_source, "Deal": deal_idx, "_Batch_Row_Key": result_row_key, "Error": f"lookup_error: {exc}"})
            continue

        if not matches:
            _progress(progress_callback, i, total, f"Board {i+1}/{total} (#{deal_idx}) DD+EV augmentation")
            pbn_for_aug = str(src_board.get("pbn", "") or "")
            aug = _compute_board_dds_augmentation(pbn_for_aug, dealer, vul)
            aug_sd = _compute_board_sd_ev_augmentation(pbn_for_aug, dealer, vul)
            deal_row = {
                "Dealer": dealer,
                "Vul": vul,
                **{f"Hand_{d}": src_board.get(f"Hand_{d}", "") for d in "NESW"},
                **{f"HCP_{d}": _hand_hcp(str(src_board.get(f"Hand_{d}", "") or "")) for d in "NESW"},
                **{f"Total_Points_{d}": _hand_total_points(str(src_board.get(f"Hand_{d}", "") or "")) for d in "NESW"},
                **aug,
                **aug_sd,
            }
            for pbn_key in ("Tricks", "Result", "Declarer", "Contract"):
                pbn_val = src_board.get(pbn_key)
                if pbn_val is not None:
                    deal_row[pbn_key] = pbn_val
            deal_row_map[result_row_key] = deal_row
            row_idx = None
        else:
            deal_row = dict(matches[0])
            if "Actual_Auction" in deal_row and "bid" not in deal_row:
                deal_row["bid"] = deal_row["Actual_Auction"]
            for d in ("N", "E", "S", "W"):
                hk = f"HCP_{d}"
                tk = f"Total_Points_{d}"
                hand = str(src_board.get(f"Hand_{d}", "") or "")
                if hk not in deal_row or deal_row[hk] is None:
                    deal_row[hk] = _hand_hcp(hand)
                if tk not in deal_row or deal_row[tk] is None:
                    deal_row[tk] = _hand_total_points(hand)
            if deal_row.get("DD_N_S") is None:
                aug = _compute_board_dds_augmentation(str(src_board.get("pbn", "") or ""), dealer, vul)
                for col, val in aug.items():
                    if deal_row.get(col) is None:
                        deal_row[col] = val
            if deal_row.get("EV_NS_N_S_3") is None:
                aug_sd = _compute_board_sd_ev_augmentation(str(src_board.get("pbn", "") or ""), dealer, vul)
                for col, val in aug_sd.items():
                    if deal_row.get(col) is None:
                        deal_row[col] = val
            for pbn_key in ("Tricks", "Result", "Declarer", "Contract"):
                pbn_val = src_board.get(pbn_key)
                if pbn_val is not None:
                    deal_row[pbn_key] = pbn_val
            deal_row_map[result_row_key] = deal_row
            row_idx = deal_row.get("_row_idx")

        elapsed_batch_s = max(time.perf_counter() - t_batch_start, 1e-9)
        deals_per_min = (i * 60.0) / elapsed_batch_s if i > 0 else 0.0
        _progress(progress_callback, i, total, f"Board {i+1}/{total} (#{deal_idx}) AI auction {deals_per_min:.2f} deals/min")
        if row_idx is not None:
            ai_start_params: dict[str, Any] = {
                "deal_row_idx": int(row_idx),
                "seed": int(config.seed),
                "logic_mode": str(config.ai_logic_mode),
                "use_guardrails_v2": bool(config.use_guardrails_v2),
                "max_steps": int(config.max_steps),
                "top_n": int(config.top_n),
                "max_deals": int(config.max_deals),
                "permissive_pass": bool(config.permissive_pass),
            }
        else:
            ai_start_params = {
                "deal_row_idx": -1,
                "deal_row_dict": _json_safe_deal_row(deal_row),
                "seed": int(config.seed),
                "logic_mode": str(config.ai_logic_mode),
                "use_guardrails_v2": bool(config.use_guardrails_v2),
                "max_steps": int(config.max_steps),
                "top_n": int(config.top_n),
                "max_deals": int(config.max_deals),
                "permissive_pass": bool(config.permissive_pass),
            }
        try:
            start_resp = client.post("/ai-model-advanced-path/start", ai_start_params, timeout=config.start_timeout_s)
            job_id = str(start_resp.get("job_id") or "").strip()
        except Exception as exc:
            results.append({"Source": row_source, "Deal": deal_idx, "_Batch_Row_Key": result_row_key, "Error": f"start_failed: {exc}"})
            continue
        if not job_id:
            results.append({"Source": row_source, "Deal": deal_idx, "_Batch_Row_Key": result_row_key, "Error": "no_job_id"})
            continue

        ai_auction = ""
        ai_steps_detail: list[dict[str, Any]] = []
        poll_ok = False
        for _poll in range(int(config.max_polls)):
            time.sleep(float(config.poll_interval_s))
            try:
                job = client.get(f"/ai-model-advanced-path/status/{job_id}", timeout=config.status_timeout_s)
            except Exception:
                continue
            status_val = str(job.get("status") or "")
            if status_val == "completed":
                res = job.get("result") or {}
                ai_auction = normalize_auction_input(str(res.get("auction") or "").strip())
                ai_steps_detail = list(res.get("steps_detail") or [])
                ai_steps_map[result_row_key] = ai_steps_detail
                poll_ok = True
                break
            if status_val == "failed":
                results.append({"Source": row_source, "Deal": deal_idx, "_Batch_Row_Key": result_row_key, "Error": f"job_failed: {job.get('error')}"})
                break
        else:
            results.append({"Source": row_source, "Deal": deal_idx, "_Batch_Row_Key": result_row_key, "Error": "timeout"})
        if not poll_ok:
            continue

        actual_contract = _contract_str(actual_auction, dealer)
        ai_contract = _contract_str(ai_auction, dealer)
        bt_mean_auctions = [a for a in [actual_auction, ai_auction] if a and a.strip()]
        bt_mean_map: dict[str, float | None] = {}
        if bt_mean_auctions:
            try:
                bt_resp = client.post("/bt-dd-mean-tricks", {"auctions": bt_mean_auctions, "dealer": dealer}, timeout=config.bt_mean_timeout_s)
                bt_mean_map = bt_resp.get("results") or {}
            except Exception:
                pass

        def bt_mean_result(auction: str | None) -> float | None:
            if not auction:
                return None
            mean_tricks = bt_mean_map.get(auction.strip())
            if mean_tricks is None:
                return None
            contract = parse_contract_from_auction(auction)
            if not contract:
                return None
            level_i, _, _ = contract
            return round(float(mean_tricks) - (int(level_i) + 6), 1)

        actual_tricks = bt_mean_map.get((actual_auction or "").strip())
        ai_dd_tricks = bt_mean_map.get((ai_auction or "").strip())
        actual_result = bt_mean_result(actual_auction)
        ai_result = bt_mean_result(ai_auction)

        def contract_score_or_passout(auction: str | None) -> int | None:
            if not auction:
                return None
            toks = [t.strip().upper() for t in auction.split("-") if t.strip()]
            if toks and all(t == "P" for t in toks):
                return 0
            return get_dd_score_for_auction(auction, dealer, deal_row)

        actual_score_contract = contract_score_or_passout(actual_auction)
        ai_score_contract = contract_score_or_passout(ai_auction)
        actual_score_ns = _dd_score_ns(actual_auction, dealer, deal_row)
        ai_score_ns = _dd_score_ns(ai_auction, dealer, deal_row)
        actual_ev = _ev_declarer(actual_auction, dealer, deal_row)
        ai_ev = _ev_declarer(ai_auction, dealer, deal_row)
        par_score = deal_row.get("ParScore")
        par_contracts = format_par_contracts(deal_row.get("ParContracts")) or ""

        ev_imp_actual = _ev_imp_vs_par_signed(actual_ev, par_score, actual_auction, dealer)
        ev_imp_ai = _ev_imp_vs_par_signed(ai_ev, par_score, ai_auction, dealer)
        ev_imp_diff = None
        if ev_imp_actual is not None and ev_imp_ai is not None:
            ev_imp_diff = int(ev_imp_ai) - int(ev_imp_actual)
            ev_imp_running += int(ev_imp_diff)

        imp_actual = _imp_vs_par_signed(actual_score_contract, par_score, actual_auction, dealer)
        imp_ai = _imp_vs_par_signed(ai_score_contract, par_score, ai_auction, dealer)
        imp = _imp_diff(imp_actual, imp_ai)
        if imp is not None:
            imp_running += imp

        dd_imp_diff = _dd_imp_diff_signed(actual_score_contract, ai_score_contract)
        if dd_imp_diff is not None:
            dd_imp_running += dd_imp_diff

        div_str = ""
        try:
            actual_toks = [t.strip().upper() for t in (actual_auction or "").split("-") if t.strip()]
            ai_toks = [t.strip().upper() for t in (ai_auction or "").split("-") if t.strip()]
            if actual_toks or ai_toks:
                div_str = "✓"
                for dsi in range(max(len(actual_toks), len(ai_toks))):
                    da = actual_toks[dsi] if dsi < len(actual_toks) else "—"
                    db = ai_toks[dsi] if dsi < len(ai_toks) else "—"
                    if da != db:
                        div_str = f"{dsi + 1}: {db}→{da}"
                        break
        except Exception:
            div_str = ""

        error_flags: list[str] = []
        try:
            ai_toks_err = [t.strip().upper() for t in (ai_auction or "").split("-") if t.strip()]
            directions = ["N", "E", "S", "W"]
            dealer_idx_err = directions.index(dealer) if dealer in directions else 0
            suit_bids_by_dir: dict[str, list[str]] = {"N": [], "E": [], "S": [], "W": []}
            sl_cache: dict[str, dict[str, int]] = {}
            for d in directions:
                hpbn = str((deal_row or {}).get(f"Hand_{d}", "") or "").strip()
                if hpbn and "." in hpbn:
                    hparts = hpbn.split(".")
                    if len(hparts) == 4:
                        sl_cache[d] = {"S": len(hparts[0]), "H": len(hparts[1]), "D": len(hparts[2]), "C": len(hparts[3])}
            for ti_err, tk_err in enumerate(ai_toks_err):
                if len(tk_err) < 2 or not tk_err[0].isdigit():
                    continue
                bidder_dir = directions[(dealer_idx_err + ti_err) % 4]
                strain_err = tk_err[1:]
                if strain_err not in ("C", "D", "H", "S"):
                    continue
                prior = suit_bids_by_dir[bidder_dir]
                if strain_err in prior:
                    sl_val = (sl_cache.get(bidder_dir) or {}).get(strain_err)
                    if sl_val is not None and sl_val <= 4:
                        error_flags.append(f"{bidder_dir} {tk_err}: {strain_err} not rebiddable (SL={sl_val})")
                prior.append(strain_err)
        except Exception:
            pass
        try:
            error_flags.extend(_collect_special_case_bt_error_notes(ai_steps_detail))
        except Exception:
            pass
        error_str = "; ".join(error_flags) if error_flags else ""

        result_row = {
            "Source": row_source,
            "Deal": deal_idx,
            "_Batch_Row_Key": result_row_key,
            "Dealer": dealer,
            "Vul": vul,
            "Divergence": div_str,
            "BT Error": error_str,
            "Actual_Auction": actual_auction,
            "AI_Auction": ai_auction,
            "Actual_Contract": actual_contract,
            "AI_Contract": ai_contract,
            "BT_Tricks_Actual": actual_tricks,
            "BT_Tricks_AI": ai_dd_tricks,
            "Actual_Result": actual_result,
            "AI_Result": ai_result,
            "DD_Score_Actual": actual_score_contract,
            "DD_Score_AI": ai_score_contract,
            "EV_Actual": actual_ev,
            "EV_AI": ai_ev,
            "EV_IMP_Actual": ev_imp_actual,
            "EV_IMP_AI": ev_imp_ai,
            "EV_IMP_Diff": ev_imp_diff,
            "EV_IMP_Running": ev_imp_running,
            "Par": par_score,
            "ParContracts": par_contracts,
            "DD_IMP_Diff": dd_imp_diff,
            "DD_IMP_Running": dd_imp_running,
            "IMP_Actual": imp_actual,
            "IMP_AI": imp_ai,
            "IMP_Diff": imp,
            "IMP_Running": imp_running,
        }
        result_row = add_auction_error_taxonomy_to_row(
            result_row,
            dealer=dealer,
            chosen_auction=ai_auction,
            chosen_contract=ai_contract,
            actual_auction=actual_auction,
            actual_contract=actual_contract,
            par_score=par_score,
            par_contracts=(deal_row.get("ParContracts") if isinstance(deal_row, dict) else None) or par_contracts,
            ai_model_steps=ai_steps_detail,
            deal_context=deal_row,
            chosen_dd_score=ai_score_contract,
            chosen_ev_score=ai_ev,
        )
        results.append(result_row)

        try:
            dbg_deal_info: dict[str, Any] = {}
            for dk in ["_row_idx", "index", "Dealer", "Vul", "Hand_N", "Hand_E", "Hand_S", "Hand_W", "ParContracts", "ParScore", "bid"]:
                dv = deal_row.get(dk)
                if dv is not None:
                    dbg_deal_info[dk] = dv
            for d in ("N", "E", "S", "W"):
                for stat in ("HCP", "Total_Points"):
                    sk = f"{stat}_{d}"
                    sv = deal_row.get(sk)
                    if sv is not None:
                        dbg_deal_info[sk] = sv
            actual_toks = [t.strip().upper() for t in (actual_auction or "").split("-") if t.strip()]
            ai_toks = [t.strip().upper() for t in (ai_auction or "").split("-") if t.strip()]
            divergence: dict[str, Any] | None = None
            for si in range(max(len(actual_toks), len(ai_toks))):
                a_bid = actual_toks[si] if si < len(actual_toks) else None
                ai_bid = ai_toks[si] if si < len(ai_toks) else None
                if a_bid != ai_bid:
                    step_data = None
                    for sd in ai_steps_detail:
                        if isinstance(sd, dict) and sd.get("step") == si + 1:
                            step_data = sd
                            break
                    divergence = {
                        "step": si + 1,
                        "actual_bid": a_bid,
                        "ai_bid": ai_bid,
                        "auction_so_far": "-".join(ai_toks[:si]) if si > 0 else "(opening)",
                    }
                    if step_data:
                        divergence["seat"] = step_data.get("seat")
                        divergence["seat_bt"] = step_data.get("seat_bt")
                        divergence["scored_n"] = step_data.get("scored_n")
                        divergence["bid_scores"] = step_data.get("bid_scores")
                        divergence["all_bids_filtered"] = step_data.get("all_bids_filtered")
                    break
            batch_debug_entry = {
                "deal_index": deal_idx,
                "deal": dbg_deal_info,
                "dealer": dealer,
                "vul": vul,
                "actual_auction": actual_auction,
                "ai_auction": ai_auction,
                "actual_contract": actual_contract,
                "ai_contract": ai_contract,
                "dd_score_actual_ns": actual_score_ns,
                "dd_score_ai_ns": ai_score_ns,
                "ev_actual": actual_ev,
                "ev_ai": ai_ev,
                "ev_imp_actual": ev_imp_actual,
                "ev_imp_ai": ev_imp_ai,
                "ev_imp_diff": ev_imp_diff,
                "par": par_score,
                "par_contracts": par_contracts,
                "imp_actual": imp_actual,
                "imp_ai": imp_ai,
                "imp_diff": imp,
                "divergence": divergence,
                "ai_model_steps": ai_steps_detail,
            }
            batch_debug_entry = add_auction_error_taxonomy_to_row(
                batch_debug_entry,
                dealer=dealer,
                chosen_auction=ai_auction,
                chosen_contract=ai_contract,
                actual_auction=actual_auction,
                actual_contract=actual_contract,
                par_score=par_score,
                par_contracts=dbg_deal_info.get("ParContracts") or par_contracts,
                ai_model_steps=ai_steps_detail,
                deal_context=deal_row,
                chosen_dd_score=ai_score_contract,
                chosen_ev_score=ai_ev,
            )
            batch_debug_entries.append(batch_debug_entry)
        except Exception:
            pass

        elapsed_s = time.perf_counter() - t_batch_start
        avg_s = elapsed_s / (i + 1)
        eta_s = avg_s * (total - i - 1)
        _progress(progress_callback, i + 1, total, f"Board {i+1}/{total} (#{deal_idx}) IMP {imp_running:+d} ETA {eta_s:.0f}s")

    export_paths = _auto_export_batch_results(
        all_results=results,
        batch_mode="PBN Local/URL",
        input_cfg=input_cfg,
        seed=int(config.seed),
        auction_filter_mode=str(config.auction_filter_mode),
        ai_logic_mode=str(config.ai_logic_mode),
        ai_logic_mode_label=str(config.ai_logic_mode_label),
        use_guardrails_v2=bool(config.use_guardrails_v2),
    )
    batch_debug_payload = _build_batch_debug_payload(
        input_cfg=input_cfg,
        results=results,
        batch_debug_entries=batch_debug_entries,
        seed=int(config.seed),
        imp_running_total=imp_running,
    )
    write_batch_debug_json(batch_debug_payload)
    return BatchRunResult(
        results=results,
        deal_row_map=deal_row_map,
        ai_steps_map=ai_steps_map,
        batch_debug_entries=batch_debug_entries,
        batch_debug_payload=batch_debug_payload,
        export_paths=export_paths,
        input_cfg=input_cfg,
    )


def run_pbn_batch(
    *,
    pbn_path: pathlib.Path | None = None,
    pbn_url: str | None = None,
    filename_regex: str = "",
    config: BatchRunnerConfig,
    progress_callback: ProgressCallback | None = None,
) -> BatchRunResult:
    if pbn_path is None and not str(pbn_url or "").strip():
        raise ValueError("Provide either pbn_path or pbn_url.")
    if pbn_path is not None and str(pbn_url or "").strip():
        raise ValueError("Provide only one of pbn_path or pbn_url.")

    if pbn_path is not None:
        path = pathlib.Path(pbn_path)
        boards, meta = load_pbn_boards_from_file(path)
        input_cfg = {
            "source_type": "local",
            "source_name": path.name,
            "filename_regex": "",
            "ranges_text": None,
            "indices": None,
            "boards": None,
            **meta,
        }
    else:
        boards, meta = load_pbn_boards_from_url_input(str(pbn_url), filename_regex=filename_regex)
        input_cfg = {
            "source_type": "url",
            "source_name": str(pbn_url),
            "filename_regex": str(filename_regex or ""),
            "ranges_text": None,
            "indices": None,
            "boards": None,
            **meta,
        }
    return run_batch_on_external_boards(
        boards=boards,
        input_cfg=input_cfg,
        config=config,
        progress_callback=progress_callback,
    )
