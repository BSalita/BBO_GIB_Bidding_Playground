from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from plugins.bidding_guardrails import InformationClass


@dataclass(frozen=True)
class ScoreTerm:
    name: str
    value: float
    source: InformationClass


def assert_live_score_terms(terms: Iterable[ScoreTerm]) -> None:
    for term in list(terms or []):
        if term.source == InformationClass.ORACLE:
            raise ValueError(f"oracle score term not allowed in live bidding: {term.name}")


def combine_live_score_terms(terms: Iterable[ScoreTerm]) -> float:
    items = list(terms or [])
    assert_live_score_terms(items)
    return float(sum(float(term.value) for term in items))


def score_live_pass(
    *,
    current_contract_score: Optional[float],
    opt_avg_ev: Optional[float],
    opt_avg_par: Optional[float],
    acting_sign: float,
    pass_bonus: float,
    pass_penalty: float,
) -> Tuple[float, str]:
    terms: list[ScoreTerm] = []
    source = "opt"
    if current_contract_score is not None:
        terms.append(
            ScoreTerm(
                name="current_contract",
                value=float(acting_sign) * float(current_contract_score),
                source=InformationClass.PUBLIC_AUCTION,
            )
        )
        source = "current_contract_projection"
    elif opt_avg_ev is not None:
        terms.append(
            ScoreTerm(
                name="opt_avg_ev",
                value=float(opt_avg_ev),
                source=InformationClass.POSTERIOR,
            )
        )
        source = "opt_avg_ev"
    elif opt_avg_par is not None:
        terms.append(
            ScoreTerm(
                name="opt_avg_par",
                value=float(opt_avg_par),
                source=InformationClass.POSTERIOR,
            )
        )
        source = "opt_avg_par"
    else:
        terms.append(ScoreTerm(name="fallback_zero", value=0.0, source=InformationClass.PUBLIC_AUCTION))

    if float(pass_bonus):
        terms.append(ScoreTerm(name="pass_bonus", value=float(pass_bonus), source=InformationClass.PUBLIC_AUCTION))
    if float(pass_penalty):
        terms.append(ScoreTerm(name="pass_penalty", value=-float(pass_penalty), source=InformationClass.SELF_HAND))
    return combine_live_score_terms(terms), source
