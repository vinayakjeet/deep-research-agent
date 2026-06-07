"""Rule-based behavior scoring per expected_behavior."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from deep_research_agent.orchestration.fallbacks import NO_EVIDENCE_MESSAGE

CONFLICT_MARKERS = re.compile(
    r"\b(however|but|disagree|conflict|contradict|mixed|diverge|while .+ reports|whereas)\b",
    re.IGNORECASE,
)
CLARIFICATION_MARKERS = re.compile(
    r"\b(clarif|which .+ do you mean|do you mean|ambiguous|assume|assuming)\b",
    re.IGNORECASE,
)
REFUSAL_MARKERS = re.compile(
    r"\b(cannot|can't|unable|not publicly|not available|no verifiable|insufficient|"
    r"could not locate|do not have|don't have|not disclose|private)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class BehaviorScore:
    passed: bool
    expected_behavior: str
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _facts_present(answer: str, expected_facts: list[str]) -> tuple[int, int]:
    if not expected_facts:
        return 0, 0
    found = 0
    lower = answer.lower()
    for fact in expected_facts:
        if fact.strip().lower() in lower:
            found += 1
    return found, len(expected_facts)


def _has_citations(citation_report: Optional[dict[str, Any]]) -> bool:
    if not citation_report:
        return False
    return bool(citation_report.get("has_citations"))


def score_behavior(
    *,
    answer: str,
    expected_behavior: str,
    expected_facts: list[str],
    citation_report: Optional[dict[str, Any]] = None,
    ended_no_evidence: bool = False,
    forbidden_terms: Optional[list[str]] = None,
) -> BehaviorScore:
    """Apply heuristic checks for each expected_behavior type."""
    behavior = expected_behavior.strip()
    checks: dict[str, bool] = {}
    notes: list[str] = []
    text = answer.strip()

    if behavior == "answer_with_citations":
        checks["non_empty"] = bool(text)
        checks["has_citations"] = _has_citations(citation_report) or bool(
            re.search(r"\]\(https?://", text)
        )
        found, total = _facts_present(text, expected_facts)
        checks["facts_partial"] = found >= max(1, total // 2) if total else True
        if total:
            notes.append(f"expected_facts matched {found}/{total}")
        passed = all(checks.values())

    elif behavior == "refuse_insufficient_evidence":
        checks["refusal_language"] = bool(
            REFUSAL_MARKERS.search(text) or NO_EVIDENCE_MESSAGE.split(".")[0] in text
        )
        checks["no_evidence_path"] = ended_no_evidence or checks["refusal_language"]
        passed = checks["no_evidence_path"]

    elif behavior == "acknowledge_conflict":
        checks["non_empty"] = bool(text)
        checks["conflict_language"] = bool(CONFLICT_MARKERS.search(text))
        checks["has_citations"] = _has_citations(citation_report) or bool(
            re.search(r"\]\(https?://", text)
        )
        passed = checks["non_empty"] and checks["conflict_language"]

    elif behavior == "correct_false_premise":
        checks["challenges_premise"] = bool(
            REFUSAL_MARKERS.search(text)
            or re.search(r"\b(incorrect|false|not true|did not|never|myth)\b", text, re.I)
        )
        checks["non_empty"] = bool(text)
        passed = checks["non_empty"] and checks["challenges_premise"]

    elif behavior == "must_not_know":
        lower = text.lower()
        forbidden = forbidden_terms or []
        leaked = [t for t in forbidden if t.lower() in lower]
        checks["no_forbidden_leak"] = len(leaked) == 0
        checks["states_unknown"] = bool(REFUSAL_MARKERS.search(text)) or len(text) < 500
        if leaked:
            notes.append(f"forbidden terms found: {leaked}")
        passed = checks["no_forbidden_leak"]

    elif behavior == "request_clarification":
        checks["clarification_or_assumption"] = bool(CLARIFICATION_MARKERS.search(text))
        checks["non_empty"] = bool(text)
        passed = checks["non_empty"] and checks["clarification_or_assumption"]

    else:
        notes.append(f"unknown expected_behavior: {behavior}")
        passed = bool(text)

    return BehaviorScore(
        passed=passed,
        expected_behavior=behavior,
        checks=checks,
        notes=notes,
    )
