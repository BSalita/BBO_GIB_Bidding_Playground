from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class CandidateOption:
    bid: str
    raw: Dict[str, Any]


def select_top_candidate_pool(
    passed_opts: Sequence[Dict[str, Any]],
    top_n: int,
) -> List[CandidateOption]:
    limit = max(1, int(top_n))
    pool: List[CandidateOption] = []
    for opt in list(passed_opts or [])[:limit]:
        bid = str((opt or {}).get("bid", "") or "").strip().upper()
        pool.append(CandidateOption(bid=bid, raw=dict(opt or {})))
    return pool
