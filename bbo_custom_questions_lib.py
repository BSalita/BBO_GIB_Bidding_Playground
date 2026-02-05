from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Iterable, Optional


AUTO_BEGIN = "<!-- BEGIN AUTO-GENERATED QUESTIONS -->"
AUTO_END = "<!-- END AUTO-GENERATED QUESTIONS -->"


@dataclass(frozen=True)
class CustomQuestion:
    id: str
    order: int
    enabled: bool
    title: str
    prompt_player: str
    prompt_template: str
    answer_mode: str
    requires_candidate_bid: bool
    requires_pinned_deal: bool


_ALLOWED_PLACEHOLDERS = frozenset({"deal_index", "auction", "candidate_bid", "why_not_bid"})


def file_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(data).hexdigest()


def _parse_bool01(v: str, *, field: str) -> bool:
    s = str(v or "").strip()
    if s in ("0", "false", "False", "FALSE", ""):
        return False
    if s in ("1", "true", "True", "TRUE"):
        return True
    raise ValueError(f"Invalid boolean for {field}: {v!r} (expected 0/1)")


def _validate_template(s: str, *, field: str) -> None:
    try:
        fmt = Formatter()
        for literal_text, field_name, format_spec, conversion in fmt.parse(str(s)):
            if field_name is None:
                continue
            # field_name can include formatting like "x!r" etc; Formatter.parse gives raw field.
            name = str(field_name).strip()
            # Disallow attribute/index lookups for safety/stability.
            if any(ch in name for ch in ".[]"):
                raise ValueError(f"{field} contains unsupported placeholder syntax: {{{name}}}")
            if name not in _ALLOWED_PLACEHOLDERS:
                raise ValueError(f"{field} contains unknown placeholder: {{{name}}}")
            if format_spec:
                raise ValueError(f"{field} contains unsupported format_spec for {{{name}}}: {format_spec!r}")
            if conversion:
                raise ValueError(f"{field} contains unsupported conversion for {{{name}}}: {conversion!r}")
    except Exception as e:
        raise ValueError(f"Invalid template in {field}: {e}") from e


def load_custom_questions_csv(path: Path) -> list[CustomQuestion]:
    if not path.exists():
        raise FileNotFoundError(f"Custom questions CSV not found: {path}")
    raw = path.read_text(encoding="utf-8")
    rdr = csv.DictReader(io.StringIO(raw))
    required = [
        "id",
        "order",
        "enabled",
        "title",
        "prompt_player",
        "prompt_template",
        "answer_mode",
        "requires_candidate_bid",
        "requires_pinned_deal",
    ]
    if rdr.fieldnames is None:
        raise ValueError("CSV has no header row")
    missing = [c for c in required if c not in set(rdr.fieldnames)]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    out: list[CustomQuestion] = []
    seen_ids: set[str] = set()
    for i, row in enumerate(rdr, start=2):
        qid = str(row.get("id") or "").strip().upper()
        if not qid:
            raise ValueError(f"Row {i}: empty id")
        if qid in seen_ids:
            raise ValueError(f"Row {i}: duplicate id {qid!r}")
        seen_ids.add(qid)

        try:
            order = int(str(row.get("order") or "").strip())
        except Exception:
            raise ValueError(f"Row {i}: invalid order {row.get('order')!r} (expected int)")

        enabled = _parse_bool01(str(row.get("enabled") or ""), field=f"enabled (row {i})")
        title = str(row.get("title") or "").strip()
        if not title:
            raise ValueError(f"Row {i}: empty title")
        prompt_player = str(row.get("prompt_player") or "").strip()
        if not prompt_player:
            raise ValueError(f"Row {i}: empty prompt_player")
        prompt_template = str(row.get("prompt_template") or "").strip()
        if not prompt_template:
            raise ValueError(f"Row {i}: empty prompt_template")
        _validate_template(prompt_template, field=f"prompt_template (row {i})")
        answer_mode = str(row.get("answer_mode") or "").strip()
        if not answer_mode:
            raise ValueError(f"Row {i}: empty answer_mode")

        requires_candidate_bid = _parse_bool01(
            str(row.get("requires_candidate_bid") or ""), field=f"requires_candidate_bid (row {i})"
        )
        requires_pinned_deal = _parse_bool01(
            str(row.get("requires_pinned_deal") or ""), field=f"requires_pinned_deal (row {i})"
        )

        out.append(
            CustomQuestion(
                id=qid,
                order=order,
                enabled=enabled,
                title=title,
                prompt_player=prompt_player,
                prompt_template=prompt_template,
                answer_mode=answer_mode,
                requires_candidate_bid=requires_candidate_bid,
                requires_pinned_deal=requires_pinned_deal,
            )
        )

    out = sorted(out, key=lambda q: (q.order, q.id))
    return out


def render_example_prompts_list_md(questions: Iterable[CustomQuestion]) -> str:
    lines: list[str] = []
    for q in questions:
        if not q.enabled:
            continue
        qn = q.id.lower()
        lines.append(f"- [{q.id}: {q.prompt_player}](#{qn})")
    return "\n".join(lines).rstrip() + "\n"


def replace_between_markers(md_text: str, *, begin: str = AUTO_BEGIN, end: str = AUTO_END, new_block: str) -> str:
    if begin not in md_text or end not in md_text:
        raise ValueError(f"Missing markers {begin!r} / {end!r} in markdown")
    pre, rest = md_text.split(begin, 1)
    _mid, post = rest.split(end, 1)
    # Preserve surrounding spacing: ensure exactly one newline after begin and before end.
    new_mid = "\n" + new_block.rstrip() + "\n"
    return pre + begin + new_mid + end + post


_Q_PROMPT_RE = re.compile(r'^(###\s+(Q\d+)\s+\(prompt\):\s*).*$')


def update_player_prompt_headings(md_text: str, questions: Iterable[CustomQuestion]) -> str:
    """Replace lines like `### Q1 (prompt): "..."` with values from CSV."""
    by_id = {q.id.upper(): q for q in questions if q.enabled}
    out_lines: list[str] = []
    for line in md_text.splitlines():
        m = _Q_PROMPT_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        qid = str(m.group(2)).upper()
        q = by_id.get(qid)
        if q is None:
            out_lines.append(line)
            continue
        # Normalize to quoted style used in the doc.
        out_lines.append(f'{m.group(1)}"{q.prompt_player}"')
    return "\n".join(out_lines).rstrip() + "\n"


def sync_questions_into_docs(
    *,
    questions_csv: Path,
    players_md: Path,
    coders_md: Path,
) -> dict[str, str]:
    """Sync docs from CSV. Returns metadata (sha1 + counts)."""
    qs = load_custom_questions_csv(questions_csv)
    sha1 = file_sha1(questions_csv)
    enabled = [q for q in qs if q.enabled]

    block = render_example_prompts_list_md(enabled)

    p_txt = players_md.read_text(encoding="utf-8")
    p_txt2 = replace_between_markers(p_txt, new_block=block)
    p_txt3 = update_player_prompt_headings(p_txt2, enabled)
    if p_txt3 != p_txt:
        players_md.write_text(p_txt3, encoding="utf-8", newline="\n")

    c_txt = coders_md.read_text(encoding="utf-8")
    c_txt2 = replace_between_markers(c_txt, new_block=block)
    if c_txt2 != c_txt:
        coders_md.write_text(c_txt2, encoding="utf-8", newline="\n")

    return {"csv_sha1": sha1, "enabled_count": str(len(enabled)), "total_count": str(len(qs))}

