from __future__ import annotations

import re
from typing import Any

import polars as pl

from bbo_bidding_queries_lib import get_ai_contract
from plugins.bbo_handlers_common import dedup_par_contracts, ev_list_for_par_contracts

_CONTRACT_RE = re.compile(r"^\s*([1-7])\s*(NT|[CDHSN])\s*((?:XX|X)?)\s*([NESW])?\s*$", re.IGNORECASE)
_PAR_STR_RE = re.compile(
    r"^\s*([1-7])\s*(NT|[CDHSN])\s*((?:XX|X)?)\s+(NS|EW|SELF|OPP)\s*(=|[+-]\d+|-?\d+)?\s*$",
    re.IGNORECASE,
)
_TOLERANCE = 0.0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _normalize_strain(value: Any) -> str | None:
    s = str(value or "").strip().upper()
    if s in {"NT", "N", "T"}:
        return "N"
    if s in {"C", "D", "H", "S"}:
        return s
    return None


def _contract_rank(level: int | None, strain: str | None) -> int | None:
    if level is None or strain is None:
        return None
    order = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}
    return (int(level) - 1) * 5 + order[strain]


def _zone_for_contract(level: int | None, strain: str | None) -> str | None:
    if level is None or strain is None:
        return None
    if level == 7:
        return "grand_slam"
    if level == 6:
        return "slam"
    if (strain == "N" and level >= 3) or (strain in {"H", "S"} and level >= 4) or (strain in {"C", "D"} and level >= 5):
        return "game"
    return "partscore"


def _max_ignore_none(*values: Any) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None


def _min_ignore_none(*values: Any) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return min(vals) if vals else None


def _parse_contract_string(contract: Any) -> dict[str, Any] | None:
    raw = str(contract or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in {"P", "PASS", "PASSED OUT", "PASSOUT"}:
        return {
            "raw": raw,
            "contract": "Pass",
            "level": None,
            "strain": None,
            "declarer": None,
            "pair": None,
            "rank": None,
            "zone": "pass",
            "is_pass": True,
        }
    token = upper.split(" ", 1)[0]
    m = _CONTRACT_RE.match(token)
    if not m:
        return None
    level = int(m.group(1))
    strain = _normalize_strain(m.group(2))
    declarer = (m.group(4) or "").upper() or None
    pair = "NS" if declarer in {"N", "S"} else "EW" if declarer in {"E", "W"} else None
    contract_text = f"{level}{'NT' if strain == 'N' else strain}{(m.group(3) or '').upper()}"
    return {
        "raw": raw,
        "contract": contract_text,
        "level": level,
        "strain": strain,
        "declarer": declarer,
        "pair": pair,
        "rank": _contract_rank(level, strain),
        "zone": _zone_for_contract(level, strain),
        "is_pass": False,
    }


def _parse_auction_contract(auction: Any, dealer: str | None) -> dict[str, Any] | None:
    auction_s = str(auction or "").strip()
    if not auction_s:
        return None
    contract_text = get_ai_contract(auction_s, str(dealer or "N").upper())
    return _parse_contract_string(contract_text or "Pass")


def _parse_par_contracts(par_contracts: Any) -> list[dict[str, Any]]:
    if par_contracts is None:
        return []
    out: list[dict[str, Any]] = []
    if isinstance(par_contracts, str):
        for chunk in [c.strip() for c in par_contracts.split(",") if c.strip()]:
            m = _PAR_STR_RE.match(chunk)
            if not m:
                continue
            level = int(m.group(1))
            strain = _normalize_strain(m.group(2))
            pair = str(m.group(4) or "").upper()
            result_raw = str(m.group(5) or "").strip()
            result = 0 if result_raw == "=" else int(result_raw) if result_raw else None
            out.append(
                {
                    "contract": f"{level}{'NT' if strain == 'N' else strain}{(m.group(3) or '').upper()}",
                    "level": level,
                    "strain": strain,
                    "pair": pair,
                    "result": result,
                    "rank": _contract_rank(level, strain),
                    "zone": _zone_for_contract(level, strain),
                }
            )
        return out
    for rec in dedup_par_contracts(par_contracts):
        level = _safe_int(rec.get("Level"))
        if level is None:
            continue
        strain = _normalize_strain(rec.get("Strain"))
        pair = str(rec.get("Pair_Direction") or "").strip().upper() or None
        result = _safe_int(rec.get("Result"))
        out.append(
            {
                "contract": f"{level}{'NT' if strain == 'N' else strain}{str(rec.get('Doubled') or rec.get('Double') or '').upper()}",
                "level": level,
                "strain": strain,
                "pair": pair,
                "result": result,
                "rank": _contract_rank(level, strain),
                "zone": _zone_for_contract(level, strain),
            }
        )
    return out


def _parse_likely_contract(text: Any) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if " by " in raw:
        raw = raw.split(" by ", 1)[0].strip()
    return _parse_contract_string(raw)


def _first_for_pair(par_contracts: list[dict[str, Any]], pair: str | None) -> dict[str, Any] | None:
    if pair:
        for rec in par_contracts:
            if str(rec.get("pair") or "").upper() == str(pair).upper():
                return rec
    return par_contracts[0] if par_contracts else None


def _extract_critical_step(ai_model_steps: Any) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for step in list(ai_model_steps or []):
        if not isinstance(step, dict):
            continue
        scores = step.get("bid_scores") or []
        if not isinstance(scores, list) or not scores:
            continue
        chosen_bid = str(step.get("choice") or "").strip().upper()
        chosen_row = None
        best_row = None
        best_score = None
        for row in scores:
            if not isinstance(row, dict):
                continue
            row_bid = str(row.get("bid") or "").strip().upper()
            row_score = _safe_float(row.get("score"))
            if row_score is None:
                row_score = _safe_float(row.get("final_score"))
            if row_bid == chosen_bid:
                chosen_row = row
            if row_score is not None and (best_score is None or row_score > best_score):
                best_score = row_score
                best_row = row
        if not chosen_row or not best_row:
            continue
        chosen_score = _safe_float(chosen_row.get("score"))
        if chosen_score is None:
            chosen_score = _safe_float(chosen_row.get("final_score"))
        if chosen_score is None or best_score is None:
            continue
        delta = float(best_score) - float(chosen_score)
        if delta <= 0:
            continue
        candidate = {
            "step": int(step.get("step") or 0),
            "auction_prefix": step.get("bt_prefix"),
            "chosen_bid": chosen_bid,
            "best_bid": str(best_row.get("bid") or "").strip().upper(),
            "delta": round(delta, 2),
            "confidence": round(min(0.99, max(0.05, delta / 120.0)), 3),
            "chosen_contract": _parse_likely_contract((step.get("bidder_view") or {}).get("likely_final_contract")),
            "best_contract": _parse_likely_contract(best_row.get("likely_final_contract")),
        }
        if best is None or float(candidate["delta"]) > float(best["delta"]):
            best = candidate
    return best


def _best_par_ev(par_contracts: Any, deal_context: Any) -> float | None:
    if not isinstance(deal_context, dict):
        return None
    ctx = dict(deal_context)
    if par_contracts is not None and ctx.get("ParContracts") is None:
        ctx["ParContracts"] = par_contracts
    try:
        evs = ev_list_for_par_contracts(ctx)
    except Exception:
        return None
    vals = [float(v) for v in evs if v is not None]
    return max(vals) if vals else None


def classify_auction_error(
    *,
    dealer: str | None,
    chosen_auction: Any = None,
    chosen_contract: Any = None,
    actual_auction: Any = None,
    actual_contract: Any = None,
    best_auction: Any = None,
    best_contract: Any = None,
    par_score: Any = None,
    par_contracts: Any = None,
    ai_model_steps: Any = None,
    chosen_dd_score: Any = None,
    chosen_ev_score: Any = None,
    deal_context: Any = None,
    tolerance: float = _TOLERANCE,
) -> dict[str, Any]:
    chosen = _parse_contract_string(chosen_contract) or _parse_auction_contract(chosen_auction, dealer)
    actual = _parse_contract_string(actual_contract) or _parse_auction_contract(actual_auction, dealer)
    best = _parse_contract_string(best_contract) or _parse_auction_contract(best_auction, dealer)
    par_list = _parse_par_contracts(par_contracts)
    critical = _extract_critical_step(ai_model_steps)

    recommended = best or (critical or {}).get("best_contract") or _first_for_pair(par_list, (chosen or {}).get("pair")) or actual
    recommended_source = (
        "best_auction" if best else
        "best_alternative" if critical and critical.get("best_contract") else
        "par_topk" if par_list else
        "actual_auction" if actual else
        None
    )

    chosen_pair = (chosen or {}).get("pair")
    our_par = _first_for_pair(par_list, chosen_pair)
    opp_par = None
    if chosen_pair == "NS":
        opp_par = _first_for_pair(par_list, "EW")
    elif chosen_pair == "EW":
        opp_par = _first_for_pair(par_list, "NS")

    contract_dd_score = _safe_float(chosen_dd_score)
    contract_ev_score = _safe_float(chosen_ev_score)
    par_score_f = _safe_float(par_score)
    par_ev_best = _best_par_ev(par_contracts, deal_context)
    contract_benchmark = _max_ignore_none(contract_dd_score, contract_ev_score)
    par_benchmark = _min_ignore_none(par_score_f, par_ev_best)
    is_error = (
        (contract_benchmark + float(tolerance) < par_benchmark)
        if contract_benchmark is not None and par_benchmark is not None
        else None
    )

    luck_high_ref = _max_ignore_none(par_ev_best, contract_ev_score)
    luck_low_ref = _min_ignore_none(par_ev_best, contract_ev_score)
    good_luck = (
        contract_dd_score > (luck_high_ref + float(tolerance))
        if contract_dd_score is not None and luck_high_ref is not None
        else None
    )
    bad_luck = (
        (contract_dd_score + float(tolerance) < luck_low_ref)
        if contract_dd_score is not None and luck_low_ref is not None
        else None
    )
    luck_swing = (
        float(contract_dd_score) - float(contract_ev_score)
        if contract_dd_score is not None and contract_ev_score is not None
        else None
    )

    final_error_family = "other"
    decision_error_family = "other"
    reasons: list[str] = []

    if critical:
        if critical.get("chosen_bid") in {"P", "PASS"} and critical.get("best_bid") not in {"", "P", "PASS", None}:
            decision_error_family = "premature_pass"
            reasons.append("CHOSEN_PASS_WITH_BETTER_NON_PASS")
        elif critical.get("chosen_contract") and critical.get("best_contract"):
            c1 = critical["chosen_contract"]
            c2 = critical["best_contract"]
            if c1.get("strain") != c2.get("strain"):
                decision_error_family = "wrong_strain"
                reasons.append("STEP_WRONG_STRAIN")
            elif c1.get("rank") is not None and c2.get("rank") is not None:
                if int(c1["rank"]) > int(c2["rank"]):
                    decision_error_family = "overbid"
                    reasons.append("STEP_TOO_HIGH")
                elif int(c1["rank"]) < int(c2["rank"]):
                    decision_error_family = "underbid"
                    reasons.append("STEP_TOO_LOW")

    sacrifice_contract = None
    if opp_par and our_par:
        chosen_rank = (chosen or {}).get("rank")
        if (
            opp_par.get("rank") is not None
            and our_par.get("rank") is not None
            and int(our_par["rank"]) > int(opp_par["rank"])
            and (chosen is None or chosen.get("is_pass") or (chosen_rank is not None and int(chosen_rank) < int(our_par["rank"])))
        ):
            sacrifice_contract = our_par
            final_error_family = "missed_sacrifice"
            reasons.extend(["PAR_OWNED_BY_OPP", "SACRIFICE_AVAILABLE"])

    if final_error_family == "other":
        if decision_error_family == "premature_pass" and (chosen is None or chosen.get("is_pass")):
            final_error_family = "premature_pass"
        elif chosen and recommended and chosen.get("strain") != recommended.get("strain"):
            final_error_family = "wrong_strain"
            reasons.append("FINAL_WRONG_STRAIN")
        elif chosen and recommended and chosen.get("rank") is not None and recommended.get("rank") is not None:
            if int(chosen["rank"]) > int(recommended["rank"]):
                final_error_family = "overbid"
                reasons.append("FINAL_TOO_HIGH")
            elif int(chosen["rank"]) < int(recommended["rank"]):
                final_error_family = "underbid"
                reasons.append("FINAL_TOO_LOW")

    if is_error is False:
        final_error_family = "other"
        reasons.append("ERROR_GATE_BLOCKED")
    elif is_error is True:
        reasons.append("ERROR_GATE_PASSED")

    severity = (recommended or chosen or {}).get("zone")

    should_have_been = {
        "wrong_strain": {
            "contract": (recommended or {}).get("contract"),
            "strain": (recommended or {}).get("strain"),
            "source": recommended_source,
        },
        "overbid": {
            "max_safe_contract": (recommended or our_par or {}).get("contract"),
            "max_level": (recommended or our_par or {}).get("level"),
            "max_strain": (recommended or our_par or {}).get("strain"),
        },
        "premature_pass": {
            "contract": (recommended or {}).get("contract"),
            "best_non_pass_bid": (critical or {}).get("best_bid"),
            "step": (critical or {}).get("step"),
        },
        "missed_sacrifice": {
            "contract": (sacrifice_contract or {}).get("contract"),
            "result_estimate": (sacrifice_contract or {}).get("result"),
        },
    }

    chosen_level = _safe_int((chosen or {}).get("level"))
    recommended_level = _safe_int((recommended or {}).get("level"))
    safe_level = _safe_int((recommended or our_par or {}).get("level"))
    overbid_by_levels = (
        chosen_level - safe_level
        if final_error_family == "overbid" and chosen_level is not None and safe_level is not None
        else None
    )
    underbid_by_levels = (
        recommended_level - chosen_level
        if final_error_family == "underbid" and recommended_level is not None and chosen_level is not None
        else None
    )

    return {
        "final_error_family": final_error_family,
        "decision_error_family": decision_error_family,
        "severity": severity,
        "chosen_contract": (chosen or {}).get("contract"),
        "recommended_contract": (recommended or {}).get("contract"),
        "recommended_strain": (recommended or {}).get("strain"),
        "recommended_source": recommended_source,
        "max_safe_contract": (recommended or our_par or {}).get("contract"),
        "max_safe_level": (recommended or our_par or {}).get("level"),
        "max_safe_strain": (recommended or our_par or {}).get("strain"),
        "is_error": is_error,
        "contract_dd_score": contract_dd_score,
        "contract_ev_score": contract_ev_score,
        "contract_benchmark": contract_benchmark,
        "par_score": par_score_f,
        "par_ev_best": par_ev_best,
        "par_benchmark": par_benchmark,
        "overbid_luck": good_luck,
        "underbid_luck": bad_luck,
        "luck_swing": luck_swing,
        "overbid_by_levels": overbid_by_levels,
        "underbid_by_levels": underbid_by_levels,
        "missed_target_zone": (recommended or {}).get("zone") if final_error_family == "underbid" else None,
        "pass_step": (critical or {}).get("step") if decision_error_family == "premature_pass" else None,
        "auction_prefix": (critical or {}).get("auction_prefix"),
        "chosen_bid": (critical or {}).get("chosen_bid"),
        "best_non_pass_bid": (critical or {}).get("best_bid") if decision_error_family == "premature_pass" else None,
        "confidence": (critical or {}).get("confidence"),
        "opponent_par_contract": (opp_par or {}).get("contract"),
        "sacrifice_contract": (sacrifice_contract or {}).get("contract"),
        "sacrifice_result_estimate": (sacrifice_contract or {}).get("result"),
        "why": list(dict.fromkeys(reasons)),
        "evidence_source": [src for src in ["ai_model_steps" if critical else None, "par_contracts" if par_list else None, recommended_source] if src],
        "should_have_been": should_have_been,
    }


def add_auction_error_taxonomy_to_row(
    row: dict[str, Any],
    *,
    dealer: str | None,
    chosen_auction: Any = None,
    chosen_contract: Any = None,
    actual_auction: Any = None,
    actual_contract: Any = None,
    best_auction: Any = None,
    best_contract: Any = None,
    par_score: Any = None,
    par_contracts: Any = None,
    ai_model_steps: Any = None,
    deal_context: Any = None,
    chosen_dd_score: Any = None,
    chosen_ev_score: Any = None,
) -> dict[str, Any]:
    taxonomy = classify_auction_error(
        dealer=dealer,
        chosen_auction=chosen_auction,
        chosen_contract=chosen_contract,
        actual_auction=actual_auction,
        actual_contract=actual_contract,
        best_auction=best_auction,
        best_contract=best_contract,
        par_score=par_score,
        par_contracts=par_contracts,
        ai_model_steps=ai_model_steps,
        chosen_dd_score=chosen_dd_score,
        chosen_ev_score=chosen_ev_score,
        deal_context=deal_context,
    )
    enriched = dict(row)
    enriched["auction_error_taxonomy"] = taxonomy
    enriched["Final_Error_Family"] = taxonomy.get("final_error_family")
    enriched["Decision_Error_Family"] = taxonomy.get("decision_error_family")
    enriched["Error_Severity"] = taxonomy.get("severity")
    enriched["Recommended_Contract"] = taxonomy.get("recommended_contract")
    enriched["Recommended_Strain"] = taxonomy.get("recommended_strain")
    enriched["Max_Safe_Level"] = taxonomy.get("max_safe_level")
    enriched["Missed_Sacrifice_Contract"] = taxonomy.get("sacrifice_contract")
    enriched["Is_Error"] = taxonomy.get("is_error")
    enriched["Contract_DD_Score"] = taxonomy.get("contract_dd_score")
    enriched["Contract_EV_Score"] = taxonomy.get("contract_ev_score")
    enriched["Contract_Benchmark"] = taxonomy.get("contract_benchmark")
    enriched["Par_Score_Benchmark"] = taxonomy.get("par_score")
    enriched["Par_EV_Best"] = taxonomy.get("par_ev_best")
    enriched["Par_Benchmark"] = taxonomy.get("par_benchmark")
    enriched["Overbid_Luck"] = taxonomy.get("overbid_luck")
    enriched["Underbid_Luck"] = taxonomy.get("underbid_luck")
    enriched["Luck_Swing"] = taxonomy.get("luck_swing")
    imp_diff = _safe_float(
        enriched.get(
            "IMP_Diff",
            enriched.get(
                "Par_IMP_Diff",
                enriched.get("imp_diff"),
            ),
        )
    )
    ev_actual = _safe_float(enriched.get("EV_Actual", enriched.get("ev_actual", enriched.get("ev_actual_ns"))))
    ev_ai = _safe_float(enriched.get("EV_AI", enriched.get("ev_ai", enriched.get("ev_ai_ns"))))
    enriched["EV_Diff"] = (ev_ai - ev_actual) if ev_ai is not None and ev_actual is not None else None
    enriched["IMP_Loss"] = -min(imp_diff or 0.0, 0.0)
    ev_diff = _safe_float(enriched.get("EV_Diff"))
    enriched["EV_Loss"] = -min(ev_diff or 0.0, 0.0)
    return enriched


def build_auction_error_summary_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    if not any("Final_Error_Family" in row for row in rows if isinstance(row, dict)):
        return pl.DataFrame()

    alias_map = {
        "IMP_Diff": ["Par_IMP_Diff", "imp_diff"],
        "EV_Diff": ["ev_diff"],
        "EV_IMP_Diff": ["ev_imp_diff"],
        "IMP_Loss": ["imp_loss"],
        "EV_Loss": ["ev_loss"],
        "Error_Severity": ["severity"],
        "Decision_Error_Family": ["decision_error_family"],
        "Overbid_Luck": ["overbid_luck"],
        "Underbid_Luck": ["underbid_luck"],
        "Luck_Swing": ["luck_swing"],
        "Is_Error": ["is_error"],
        "Par_EV_Best": ["par_ev_best"],
        "Par_Benchmark": ["par_benchmark"],
    }
    numeric_cols = ["IMP_Diff", "EV_Diff", "EV_IMP_Diff", "IMP_Loss", "EV_Loss", "Luck_Swing", "Par_EV_Best", "Par_Benchmark"]
    bool_cols = ["Overbid_Luck", "Underbid_Luck", "Is_Error"]
    required_str_cols = ["Final_Error_Family", "Decision_Error_Family", "Error_Severity"]

    normalized_rows: list[dict[str, Any]] = []
    for src in rows:
        if not isinstance(src, dict):
            continue
        row = dict(src)
        if "Final_Error_Family" not in row:
            continue
        for canonical, aliases in alias_map.items():
            if canonical not in row:
                for alias in aliases:
                    if alias in row:
                        row[canonical] = row.get(alias)
                        break
        for col in ["IMP_Diff", "EV_Diff", "EV_IMP_Diff", "IMP_Loss", "EV_Loss"]:
            row.setdefault(col, 0.0)
        for col in required_str_cols:
            row.setdefault(col, None)
        severity_text = "" if row.get("Error_Severity") is None else str(row.get("Error_Severity")).strip()
        final_family = "other" if row.get("Final_Error_Family") is None else str(row.get("Final_Error_Family"))
        row["Error_Type"] = f"{final_family}_{severity_text}" if severity_text else final_family
        normalized_rows.append(row)

    if not normalized_rows:
        return pl.DataFrame()

    df = pl.from_dicts(normalized_rows)
    casts: list[pl.Expr] = []
    for col in numeric_cols:
        if col in df.columns:
            casts.append(pl.col(col).cast(pl.Float64, strict=False))
    for col in bool_cols:
        if col in df.columns:
            casts.append(pl.col(col).cast(pl.Boolean, strict=False).fill_null(False))
    for col in required_str_cols + ["Error_Type"]:
        if col in df.columns:
            casts.append(pl.col(col).cast(pl.Utf8, strict=False))
    if casts:
        df = df.with_columns(casts)

    grouped = (
        df.group_by(["Error_Type", "Final_Error_Family", "Error_Severity", "Decision_Error_Family"], maintain_order=True)
        .agg(
            pl.len().alias("Boards"),
            pl.col("Is_Error").sum().alias("Error_Boards"),
            pl.col("IMP_Diff").sum().alias("IMP_Diff_Total"),
            pl.col("IMP_Diff").mean().alias("IMP_Diff_Mean"),
            pl.col("IMP_Loss").sum().alias("IMP_Loss_Total"),
            pl.col("EV_Diff").sum().alias("EV_Diff_Total"),
            pl.col("EV_Diff").mean().alias("EV_Diff_Mean"),
            pl.col("EV_Loss").sum().alias("EV_Loss_Total"),
            pl.col("EV_IMP_Diff").sum().alias("EV_IMP_Diff_Total"),
            pl.col("IMP_Diff").min().alias("Worst_IMP_Diff"),
            pl.col("Overbid_Luck").sum().alias("Overbid_Luck_Boards"),
            pl.col("Underbid_Luck").sum().alias("Underbid_Luck_Boards"),
            pl.col("Luck_Swing").mean().alias("Avg_Luck_Swing"),
            pl.col("Par_EV_Best").mean().alias("Avg_Par_EV_Best"),
            pl.col("Par_Benchmark").mean().alias("Avg_Par_Benchmark"),
        )
        .with_columns(
            pl.when(pl.col("Boards") > 0)
            .then((pl.col("Overbid_Luck_Boards") / pl.col("Boards") * 100.0).round(1))
            .otherwise(None)
            .alias("Overbid_Luck_%"),
            pl.when(pl.col("Boards") > 0)
            .then((pl.col("Underbid_Luck_Boards") / pl.col("Boards") * 100.0).round(1))
            .otherwise(None)
            .alias("Underbid_Luck_%"),
            (pl.col("IMP_Loss_Total").fill_null(0.0) * 1000.0 + pl.col("EV_Loss_Total").fill_null(0.0)).alias("Priority_Score"),
        )
        .sort(by=["Priority_Score", "IMP_Loss_Total", "Boards"], descending=[True, True, True])
    )
    return grouped
