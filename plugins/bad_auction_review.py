from __future__ import annotations

import json
import os
import pathlib
import re
from datetime import datetime, timezone
from typing import Any, Optional

from chatlib.expert_chat import DEFAULT_MODELS, call_provider
from plugins.batch_arena_runner import (
    BatchArenaApiClient,
    BatchRunResult,
    PROJECT_ROOT,
    _json_safe_deal_row,
    _json_safe_export_value,
    _sanitize_filename_component,
)
from plugins.self_improve_proposals import (
    markdown_from_fix_proposal,
    normalize_fix_proposal_response,
    summarize_fix_proposal_records,
)


DEFAULT_REVIEW_PROVIDER = "OpenAI"
DEFAULT_REVIEW_OUTPUT_ROOT = PROJECT_ROOT / "quality_reports" / "bad_auctions"


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    tmp.replace(path)


def _result_slug(result_row: dict[str, Any]) -> str:
    source = _sanitize_filename_component(result_row.get("Source") or "local", allow_plus=True)
    deal = _sanitize_filename_component(result_row.get("Deal") or "unknown")
    row_key = _sanitize_filename_component(result_row.get("_Batch_Row_Key") or f"{source}_{deal}", allow_plus=True)
    return f"{source}__deal_{deal}__{row_key}"


def _extract_divergence_detail(result_row: dict[str, Any], ai_steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    actual_auction = str(result_row.get("Actual_Auction") or "").strip()
    ai_auction = str(result_row.get("AI_Auction") or "").strip()
    actual_toks = [t.strip().upper() for t in actual_auction.split("-") if t.strip()]
    ai_toks = [t.strip().upper() for t in ai_auction.split("-") if t.strip()]
    for idx in range(max(len(actual_toks), len(ai_toks))):
        actual_bid = actual_toks[idx] if idx < len(actual_toks) else None
        ai_bid = ai_toks[idx] if idx < len(ai_toks) else None
        if actual_bid == ai_bid:
            continue
        step_no = idx + 1
        step_data = None
        for candidate in ai_steps:
            if isinstance(candidate, dict) and candidate.get("step") == step_no:
                step_data = candidate
                break
        detail: dict[str, Any] = {
            "step": step_no,
            "auction_so_far": "-".join(ai_toks[:idx]) if idx > 0 else "",
            "actual_bid": actual_bid,
            "ai_bid": ai_bid,
        }
        if isinstance(step_data, dict):
            detail["step_data"] = step_data
            chosen_bid = str(step_data.get("chosen_bid") or ((step_data.get("bidder_view") or {}).get("chosen_bid")) or "").strip().upper()
            bid_scores = list(step_data.get("bid_scores") or [])
            detail["chosen_bid_score"] = next(
                (row for row in bid_scores if str((row or {}).get("bid") or "").strip().upper() == chosen_bid),
                None,
            )
            if actual_bid:
                detail["actual_bid_score"] = next(
                    (row for row in bid_scores if str((row or {}).get("bid") or "").strip().upper() == str(actual_bid).upper()),
                    None,
                )
            top_alternatives = []
            for row in bid_scores:
                bid = str((row or {}).get("bid") or "").strip().upper()
                if bid and bid != chosen_bid:
                    top_alternatives.append(row)
            detail["top_alternatives"] = top_alternatives[:5]
        return detail
    return None


def _deal_known_hands(deal_row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for direction in ("N", "E", "S", "W"):
        hand = str(deal_row.get(f"Hand_{direction}") or "").strip()
        if hand:
            out[direction] = hand
    return out


def _packet_base_context(
    *,
    result_row: dict[str, Any],
    deal_row: dict[str, Any],
    ai_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_row": _json_safe_export_value(result_row),
        "deal_row": _json_safe_export_value(_json_safe_deal_row(deal_row)),
        "ai_model_steps": _json_safe_export_value(ai_steps),
    }


def build_review_packet(
    *,
    result_row: dict[str, Any],
    deal_row: dict[str, Any],
    ai_steps: list[dict[str, Any]],
    api_client: BatchArenaApiClient,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    packet = _packet_base_context(result_row=result_row, deal_row=deal_row, ai_steps=ai_steps)
    dealer = str(result_row.get("Dealer") or deal_row.get("Dealer") or "N").upper()
    vul = result_row.get("Vul") or deal_row.get("Vul")
    row_idx = deal_row.get("_row_idx")
    deal_index = deal_row.get("index") or result_row.get("Deal")
    divergence = _extract_divergence_detail(result_row, ai_steps)
    packet["divergence"] = _json_safe_export_value(divergence)

    known_hands = _deal_known_hands(deal_row)
    actual_auction = str(result_row.get("Actual_Auction") or "").strip()
    ai_auction = str(result_row.get("AI_Auction") or "").strip()

    try:
        packet["critical_mistake_analysis"] = api_client.post(
            "/critical-mistake-analysis",
            {
                "ai_model_steps": ai_steps,
                "dealer": dealer,
                "us_pair": "NS",
                "top_k": 3,
            },
            timeout=20,
        )
    except Exception as exc:
        packet["critical_mistake_analysis_error"] = str(exc)

    belief_request = {
        "auction": ai_auction,
        "dealer": dealer,
        "vul": vul,
        "known_hands": known_hands or None,
        "deal_row_idx": int(row_idx) if row_idx is not None else None,
        "deal_row_dict": _json_safe_deal_row(deal_row),
        "compact": True,
    }
    try:
        packet["belief_snapshot"] = api_client.post("/belief-snapshot", belief_request, timeout=20)
    except Exception as exc:
        packet["belief_snapshot_error"] = str(exc)

    if divergence:
        prefix = str(divergence.get("auction_so_far") or "").strip()
        ai_bid = str(divergence.get("ai_bid") or "").strip()
        actual_bid = str(divergence.get("actual_bid") or "").strip()
        explain_payload: dict[str, Any] = {
            "auction": prefix,
            "bid": ai_bid,
            "why_not_bid": actual_bid or None,
            "max_deals": 5000,
            "seed": 0,
            "topk": 10,
            "include_phase2a": True,
        }
        try:
            if deal_index is not None:
                explain_payload["deal_index"] = int(deal_index)
        except Exception:
            pass
        try:
            packet["divergence_explanation"] = api_client.post("/explain-bid", explain_payload, timeout=30)
        except Exception as exc:
            packet["divergence_explanation_error"] = str(exc)

    slug = _result_slug(result_row)
    packet_path = pathlib.Path(output_dir) / "packets" / f"{slug}.json"
    _atomic_write_text(packet_path, json.dumps(_json_safe_export_value(packet), indent=2))
    return packet_path


def build_negative_imp_review_packets(
    *,
    batch_result: BatchRunResult,
    api_base: str,
    output_root: pathlib.Path,
    max_bad_deals: int | None = None,
    min_imp_loss: int = 1,
) -> list[pathlib.Path]:
    api_client = BatchArenaApiClient(api_base=api_base)
    sorted_rows = sorted(
        (
            row for row in batch_result.results
            if isinstance(row, dict)
            and row.get("IMP_Diff") is not None
            and int(row["IMP_Diff"]) <= -abs(int(min_imp_loss))
        ),
        key=lambda row: int(row.get("IMP_Diff") or 0),
    )
    if max_bad_deals is not None:
        sorted_rows = sorted_rows[: int(max_bad_deals)]

    packet_paths: list[pathlib.Path] = []
    for row in sorted_rows:
        row_key = str(row.get("_Batch_Row_Key") or "")
        deal_row = batch_result.deal_row_map.get(row_key)
        ai_steps = batch_result.ai_steps_map.get(row_key) or []
        if not deal_row:
            continue
        packet_paths.append(
            build_review_packet(
                result_row=row,
                deal_row=deal_row,
                ai_steps=ai_steps,
                api_client=api_client,
                output_dir=output_root,
            )
        )
    return packet_paths


def _provider_api_key(provider: str) -> str:
    p = str(provider or "").strip()
    env_map = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Vercel": "VERCEL_AI_GATEWAY_API_KEY",
    }
    primary = env_map.get(p, "")
    if primary and os.environ.get(primary):
        return str(os.environ.get(primary) or "")
    if p == "Vercel":
        return str(os.environ.get("AI_GATEWAY_API_KEY") or "")
    return ""


def _provider_default_model(provider: str) -> str:
    p = str(provider or "").strip()
    if p == "OpenAI":
        return str(os.environ.get("OPENAI_MODEL") or "").strip() or str(DEFAULT_MODELS.get("OpenAI") or "").strip()
    return str(DEFAULT_MODELS.get(p) or "").strip()


def _review_system_prompt() -> str:
    return (
        "You are reviewing a bridge bidding engine failure. "
        "Analyze the supplied packet and return a single JSON object only. "
        "Focus on concrete causes in BT coverage, scoring heuristics, guardrails, "
        "or range interpretation. Be specific about likely code changes. "
        "Do not propose vague product ideas. "
        "Required JSON fields: "
        "root_cause_category, root_cause_summary, divergence_step, "
        "why_chosen_bid_won, why_better_bid_lost, suggested_code_changes, "
        "suggested_test_case, risk_of_regression, confidence. "
        "suggested_code_changes must be an array of objects with fields: "
        "title, target_files, change_type, rationale, implementation_notes."
    )


def _review_question() -> str:
    return (
        "Review this negative-IMP auction result. "
        "Identify the most likely root cause and offer targeted code changes. "
        "Prefer the smallest safe fixes that address the specific failure. "
        "If the evidence is ambiguous, say so explicitly."
    )


def _fix_proposal_system_prompt() -> str:
    return (
        "You are designing bounded fix proposals for a bridge bidding engine failure. "
        "Return a single JSON object only. "
        "Do not return markdown. "
        "Required top-level field: candidate_fixes. "
        "candidate_fixes must be an array of 1 to 3 objects. "
        "Each candidate must include: "
        "root_cause_category, root_cause_summary, change_scope, change_type, target_files, proposed_edits, risk_notes, confidence. "
        "target_files must be exact repo-relative file paths when you can identify them. "
        "proposed_edits must be an array of objects with: title, target_file, change_type, rationale, implementation_notes, search_hint. "
        "Keep fixes bounded and patch-friendly. "
        "Prefer the smallest safe fix. "
        "If no safe patch is justified, emit one candidate with change_type set to no_safe_patch and no target files. "
        "Use root_cause_category values such as bt_criteria_overlay, scoring_heuristic, guardrail_policy, range_interpretation, or ambiguous. "
        "Do not invent files that are not supported by the evidence in the packet."
    )


def _fix_proposal_question() -> str:
    return (
        "Produce 1 to 3 concrete fix proposals for this negative-IMP result. "
        "Each proposal should identify likely target files, bounded edits, and the safest next patch to try. "
        "Prefer repo-grounded proposals that can later be verified by replaying the failed board and running safety tests."
    )


def _extract_json_text(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1].strip()
    return None


def _markdown_from_review(packet: dict[str, Any], parsed_review: dict[str, Any] | None, raw_text: str) -> str:
    result_row = dict(packet.get("result_row") or {})
    lines = [
        f"# Bad Auction Review: Deal {result_row.get('Deal')}",
        "",
        f"- Source: `{result_row.get('Source')}`",
        f"- IMP_Diff: `{result_row.get('IMP_Diff')}`",
        f"- Actual_Auction: `{result_row.get('Actual_Auction')}`",
        f"- AI_Auction: `{result_row.get('AI_Auction')}`",
        "",
    ]
    if parsed_review:
        lines.extend(
            [
                "## Root Cause",
                "",
                str(parsed_review.get("root_cause_summary") or ""),
                "",
                "## Suggested Code Changes",
                "",
            ]
        )
        for change in list(parsed_review.get("suggested_code_changes") or []):
            if not isinstance(change, dict):
                continue
            lines.append(f"- {change.get('title')}: {change.get('rationale')}")
        lines.extend(
            [
                "",
                "## Suggested Test Case",
                "",
                str(parsed_review.get("suggested_test_case") or ""),
                "",
                "## Risk Of Regression",
                "",
                str(parsed_review.get("risk_of_regression") or ""),
                "",
            ]
        )
    lines.extend(["## Raw Model Output", "", "```text", raw_text.strip(), "```", ""])
    return "\n".join(lines)


def run_llm_review_for_packet(
    *,
    packet_path: pathlib.Path,
    provider: str = DEFAULT_REVIEW_PROVIDER,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 90,
) -> dict[str, Any]:
    packet = json.loads(pathlib.Path(packet_path).read_text(encoding="utf-8"))
    provider_name = str(provider or DEFAULT_REVIEW_PROVIDER)
    model_name = str(model or _provider_default_model(provider_name) or "").strip()
    key = str(api_key or _provider_api_key(provider_name) or "").strip()
    if not key:
        raise ValueError(f"Missing API key for provider `{provider_name}`.")
    if not model_name:
        raise ValueError(f"Missing model for provider `{provider_name}`.")

    review_text = call_provider(
        provider=provider_name,
        api_key=key,
        model=model_name,
        system_prompt=_review_system_prompt(),
        question=_review_question(),
        context=packet,
        timeout_s=int(timeout_s),
    )

    parsed_review = None
    json_text = _extract_json_text(review_text)
    if json_text:
        try:
            parsed_review = json.loads(json_text)
        except Exception:
            parsed_review = None

    record = {
        "packet_path": str(packet_path),
        "provider": provider_name,
        "model": model_name,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "parsed_review": parsed_review,
        "raw_text": review_text,
    }

    packet_slug = pathlib.Path(packet_path).stem
    output_root = pathlib.Path(packet_path).parent.parent
    json_out = output_root / "reviews_json" / f"{packet_slug}.json"
    md_out = output_root / "reviews_md" / f"{packet_slug}.md"
    _atomic_write_text(json_out, json.dumps(_json_safe_export_value(record), indent=2))
    _atomic_write_text(md_out, _markdown_from_review(packet, parsed_review, review_text))
    record["json_output_path"] = str(json_out)
    record["markdown_output_path"] = str(md_out)
    return record


def run_llm_reviews_for_packets(
    *,
    packet_paths: list[pathlib.Path],
    provider: str = DEFAULT_REVIEW_PROVIDER,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 90,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for packet_path in packet_paths:
        records.append(
            run_llm_review_for_packet(
                packet_path=packet_path,
                provider=provider,
                model=model,
                api_key=api_key,
                timeout_s=timeout_s,
            )
        )
    return records


def run_fix_proposal_for_packet(
    *,
    packet_path: pathlib.Path,
    provider: str = DEFAULT_REVIEW_PROVIDER,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 90,
    max_proposals: int | None = None,
) -> dict[str, Any]:
    packet = json.loads(pathlib.Path(packet_path).read_text(encoding="utf-8"))
    provider_name = str(provider or DEFAULT_REVIEW_PROVIDER)
    model_name = str(model or _provider_default_model(provider_name) or "").strip()
    key = str(api_key or _provider_api_key(provider_name) or "").strip()
    if not key:
        raise ValueError(f"Missing API key for provider `{provider_name}`.")
    if not model_name:
        raise ValueError(f"Missing model for provider `{provider_name}`.")

    proposal_text = call_provider(
        provider=provider_name,
        api_key=key,
        model=model_name,
        system_prompt=_fix_proposal_system_prompt(),
        question=_fix_proposal_question(),
        context=packet,
        timeout_s=int(timeout_s),
    )

    parsed_response = None
    json_text = _extract_json_text(proposal_text)
    if json_text:
        try:
            parsed_response = json.loads(json_text)
        except Exception:
            parsed_response = None

    normalized_record = normalize_fix_proposal_response(
        packet=packet,
        parsed_response=parsed_response,
        provider=provider_name,
        model=model_name,
        raw_text=proposal_text,
        max_proposals=max_proposals,
    )
    result_row = dict(packet.get("result_row") or {})
    record = {
        "packet_path": str(packet_path),
        "provider": provider_name,
        "model": model_name,
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "parsed_response": parsed_response,
        "normalized_record": normalized_record,
        "raw_text": proposal_text,
        "result_row": result_row,
    }

    packet_slug = pathlib.Path(packet_path).stem
    output_root = pathlib.Path(packet_path).parent.parent
    raw_out = output_root / "proposals_raw_json" / f"{packet_slug}.json"
    normalized_out = output_root / "proposals_normalized_json" / f"{packet_slug}.json"
    md_out = output_root / "proposals_md" / f"{packet_slug}.md"
    _atomic_write_text(
        raw_out,
        json.dumps(
            _json_safe_export_value(
                {
                    "packet_path": str(packet_path),
                    "provider": provider_name,
                    "model": model_name,
                    "proposed_at": record["proposed_at"],
                    "parsed_response": parsed_response,
                    "raw_text": proposal_text,
                }
            ),
            indent=2,
        ),
    )
    _atomic_write_text(normalized_out, json.dumps(_json_safe_export_value(normalized_record), indent=2))
    _atomic_write_text(md_out, markdown_from_fix_proposal(packet, normalized_record))
    record["raw_output_path"] = str(raw_out)
    record["normalized_output_path"] = str(normalized_out)
    record["markdown_output_path"] = str(md_out)
    return record


def run_fix_proposals_for_packets(
    *,
    packet_paths: list[pathlib.Path],
    provider: str = DEFAULT_REVIEW_PROVIDER,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 90,
    max_proposals: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for packet_path in packet_paths:
        records.append(
            run_fix_proposal_for_packet(
                packet_path=packet_path,
                provider=provider,
                model=model,
                api_key=api_key,
                timeout_s=timeout_s,
                max_proposals=max_proposals,
            )
        )
    return records


def summarize_review_records(
    *,
    review_records: list[dict[str, Any]],
    output_root: pathlib.Path,
    batch_result: BatchRunResult,
) -> pathlib.Path:
    by_category: dict[str, int] = {}
    deals: list[dict[str, Any]] = []
    for record in review_records:
        parsed_review = dict(record.get("parsed_review") or {})
        category = str(parsed_review.get("root_cause_category") or "unparsed")
        by_category[category] = by_category.get(category, 0) + 1
        packet_path = pathlib.Path(str(record.get("packet_path") or ""))
        packet = json.loads(packet_path.read_text(encoding="utf-8")) if packet_path.exists() else {}
        result_row = dict(packet.get("result_row") or {})
        deals.append(
            {
                "deal": result_row.get("Deal"),
                "source": result_row.get("Source"),
                "imp_diff": result_row.get("IMP_Diff"),
                "root_cause_category": category,
                "root_cause_summary": parsed_review.get("root_cause_summary"),
                "json_output_path": record.get("json_output_path"),
                "markdown_output_path": record.get("markdown_output_path"),
            }
        )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_count": len(review_records),
        "categories": by_category,
        "batch_export_paths": {k: str(v) for k, v in batch_result.export_paths.items()},
        "deals": deals,
    }
    summary_path = pathlib.Path(output_root) / "summary.json"
    _atomic_write_text(summary_path, json.dumps(_json_safe_export_value(summary), indent=2))
    return summary_path


def summarize_fix_proposals(
    *,
    proposal_records: list[dict[str, Any]],
    output_root: pathlib.Path,
    batch_result: BatchRunResult,
) -> pathlib.Path:
    summary = summarize_fix_proposal_records(
        proposal_records=proposal_records,
        batch_export_paths=batch_result.export_paths,
    )
    summary_path = pathlib.Path(output_root) / "proposals_summary.json"
    _atomic_write_text(summary_path, json.dumps(_json_safe_export_value(summary), indent=2))
    return summary_path
