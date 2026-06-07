"""Aggregate and write evaluation reports."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from deep_research_agent.eval.runner import CaseRunResult


def _case_to_dict(result: CaseRunResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "case_id": result.case_id,
        "session_id": result.session_id,
        "query": result.query,
        "final_answer": result.final_answer,
        "search_urls": result.search_urls,
        "fetched_urls": result.fetched_urls,
        "citation_urls": result.citation_urls,
        "search_query_count": result.search_query_count,
        "ended_no_evidence": result.ended_no_evidence,
        "phases_executed": result.phases_executed,
        "duration_sec": result.duration_sec,
        "error": result.error,
    }
    if result.retrieval:
        data["retrieval"] = asdict(result.retrieval)
    if result.behavior:
        data["behavior"] = asdict(result.behavior)
    if result.streaming:
        data["streaming"] = asdict(result.streaming)
    if result.judge:
        data["judge"] = asdict(result.judge)
    if result.citation_report:
        data["citation_report"] = result.citation_report
    return data


@dataclass(slots=True)
class AggregateReport:
    total_cases: int
    errors: int
    retrieval_mean_f1: Optional[float]
    behavior_pass_rate: float
    streaming_pass_rate: float
    judge_behavior_pass_rate: Optional[float]
    confident_hallucination_count: int
    by_behavior: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_label: dict[str, dict[str, Any]] = field(default_factory=dict)


def aggregate_results(
    results: list[CaseRunResult],
    case_labels: dict[str, list[str]],
) -> AggregateReport:
    """Compute summary metrics across case results."""
    errors = sum(1 for r in results if r.error)
    f1_scores: list[float] = []
    behavior_pass = 0
    streaming_pass = 0
    judge_pass = 0
    judge_total = 0
    confident_hallucinations = 0

    by_behavior: dict[str, list[bool]] = defaultdict(list)
    by_label: dict[str, list[bool]] = defaultdict(list)

    for result in results:
        if result.error:
            continue
        if result.retrieval and not result.retrieval.skipped and result.retrieval.f1 is not None:
            f1_scores.append(result.retrieval.f1)
        if result.behavior:
            behavior_pass += int(result.behavior.passed)
            by_behavior[result.behavior.expected_behavior].append(result.behavior.passed)
        if result.streaming:
            streaming_pass += int(result.streaming.passed)
        if result.judge and not result.judge.skipped:
            judge_total += 1
            judge_pass += int(result.judge.behavior_pass)
            if result.judge.confident_hallucination:
                confident_hallucinations += 1

        labels = case_labels.get(result.case_id, [])
        overall_pass = (
            (result.behavior.passed if result.behavior else False)
            and (result.streaming.passed if result.streaming else True)
        )
        for label in labels:
            by_label[label].append(overall_pass)

    n = len(results) or 1
    valid = max(n - errors, 1)

    def _rate(items: list[bool]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "pass_rate": None}
        passed = sum(items)
        return {"count": len(items), "pass_rate": round(passed / len(items), 4)}

    return AggregateReport(
        total_cases=len(results),
        errors=errors,
        retrieval_mean_f1=round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None,
        behavior_pass_rate=round(behavior_pass / valid, 4),
        streaming_pass_rate=round(streaming_pass / valid, 4),
        judge_behavior_pass_rate=round(judge_pass / judge_total, 4) if judge_total else None,
        confident_hallucination_count=confident_hallucinations,
        by_behavior={k: _rate(v) for k, v in by_behavior.items()},
        by_label={k: _rate(v) for k, v in by_label.items()},
    )


def write_report(
    results: list[CaseRunResult],
    aggregate: AggregateReport,
    output_path: str | Path,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Write JSON report and companion CSV summary."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "aggregate": asdict(aggregate),
        "cases": [_case_to_dict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "case_id", "behavior_pass", "streaming_pass", "retrieval_f1",
            "judge_pass", "error", "duration_sec",
        ])
        for r in results:
            writer.writerow([
                r.case_id,
                r.behavior.passed if r.behavior else "",
                r.streaming.passed if r.streaming else "",
                r.retrieval.f1 if r.retrieval and r.retrieval.f1 is not None else "",
                r.judge.behavior_pass if r.judge and not r.judge.skipped else "",
                r.error,
                r.duration_sec,
            ])
    return out
