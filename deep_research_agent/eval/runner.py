"""Run a single dataset case through the research orchestrator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from deep_research_agent.eval.behavior import BehaviorScore, score_behavior
from deep_research_agent.eval.judge import JudgeScore, run_judge
from deep_research_agent.eval.metrics import RetrievalScore, score_retrieval
from deep_research_agent.eval.streaming import StreamingScore, score_streaming
from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.orchestration.citations import MARKDOWN_CITATION_RE, PAREN_CITATION_RE
from deep_research_agent.orchestration.orchestrator import ResearchOrchestrator


@dataclass(slots=True)
class CaseRunResult:
    case_id: str
    session_id: str
    query: str
    final_answer: str
    events: list[dict[str, Any]] = field(default_factory=list)
    search_urls: list[str] = field(default_factory=list)
    fetched_urls: list[str] = field(default_factory=list)
    citation_urls: list[str] = field(default_factory=list)
    search_query_count: int = 0
    ended_no_evidence: bool = False
    error: str = ""
    retrieval: Optional[RetrievalScore] = None
    behavior: Optional[BehaviorScore] = None
    streaming: Optional[StreamingScore] = None
    judge: Optional[JudgeScore] = None
    phases_executed: list[str] = field(default_factory=list)
    citation_report: Optional[dict[str, Any]] = None
    duration_sec: float = 0.0


def _extract_urls_from_answer(answer: str) -> list[str]:
    urls: list[str] = []
    for pattern in (MARKDOWN_CITATION_RE, PAREN_CITATION_RE):
        for match in pattern.finditer(answer):
            url = match.group(2).strip()
            if url not in urls:
                urls.append(url)
    return urls


def _collect_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    search_urls: list[str] = []
    fetched_urls: list[str] = []
    final_answer = ""
    phases_executed: list[str] = []
    citation_report: Optional[dict[str, Any]] = None
    ended_no_evidence = False
    search_query_count = 0

    for event in events:
        etype = event.get("event")
        if etype == "no_evidence":
            ended_no_evidence = True
        if etype == "phase_complete":
            phase = event.get("phase", "")
            details = event.get("details") or {}
            if phase == "searching":
                for url in details.get("urls") or []:
                    if url and url not in search_urls:
                        search_urls.append(url)
            if phase == "acquiring":
                for url in details.get("urls") or []:
                    if url and url not in fetched_urls:
                        fetched_urls.append(url)
            if phase == "planning":
                queries = details.get("search_queries") or []
                search_query_count = max(search_query_count, len(queries))
        if etype == "complete":
            final_answer = event.get("final_answer") or ""
            details = event.get("details") or {}
            phases_executed = list(details.get("phases_executed") or [])
            citation_report = details.get("citation_report")

    return {
        "search_urls": search_urls,
        "fetched_urls": fetched_urls,
        "final_answer": final_answer,
        "phases_executed": phases_executed,
        "citation_report": citation_report,
        "ended_no_evidence": ended_no_evidence,
        "search_query_count": search_query_count,
    }


def run_query(
    orchestrator: ResearchOrchestrator,
    session_id: str,
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute one query and return raw events plus extracted fields."""
    events: list[dict[str, Any]] = []
    try:
        for event in orchestrator.run_stream(session_id, query):
            events.append(event)
    except Exception as exc:
        return events, {"error": str(exc), "final_answer": ""}
    return events, _collect_from_events(events)


def run_case(
    case: dict[str, Any],
    *,
    store: MemoryStore,
    orchestrator: ResearchOrchestrator,
    prior_queries: Optional[list[str]] = None,
    run_judge_flag: bool = True,
    forbidden_terms: Optional[list[str]] = None,
) -> CaseRunResult:
    """Run target case (with optional prior turn replay) and score."""
    import time

    case_id = case.get("id", "unknown")
    query = case.get("query", "")
    session = store.create_session()
    session_id = session["session_id"]

    start = time.perf_counter()
    all_events: list[dict[str, Any]] = []

    for prior in prior_queries or []:
        prior_events, _ = run_query(orchestrator, session_id, prior)
        all_events.extend(prior_events)

    target_events, extracted = run_query(orchestrator, session_id, query)
    all_events.extend(target_events)
    duration = time.perf_counter() - start

    if extracted.get("error"):
        return CaseRunResult(
            case_id=case_id,
            session_id=session_id,
            query=query,
            final_answer="",
            events=all_events,
            error=extracted["error"],
            duration_sec=round(duration, 2),
        )

    answer = extracted["final_answer"]
    citation_urls = _extract_urls_from_answer(answer)
    all_retrieved = (
        extracted["search_urls"] + extracted["fetched_urls"] + citation_urls
    )

    constraints = case.get("execution_constraints") or {}
    required_stream = list(constraints.get("require_streaming_events") or [])

    retrieval = score_retrieval(
        all_retrieved,
        case.get("ground_truth_urls") or [],
        case.get("acceptable_domains") or [],
    )
    behavior = score_behavior(
        answer=answer,
        expected_behavior=case.get("expected_behavior", ""),
        expected_facts=case.get("expected_facts") or [],
        citation_report=extracted.get("citation_report"),
        ended_no_evidence=extracted.get("ended_no_evidence", False),
        forbidden_terms=forbidden_terms,
    )
    streaming = score_streaming(all_events, required_stream)

    judge_result: Optional[JudgeScore] = None
    if run_judge_flag and answer.strip():
        judge_result = run_judge(
            query=query,
            answer=answer,
            expected_facts=case.get("expected_facts") or [],
            expected_behavior=case.get("expected_behavior", ""),
            ideal_answer_sketch=case.get("ideal_answer_sketch", ""),
        )

    max_queries = constraints.get("max_search_queries")
    if max_queries is not None and extracted["search_query_count"] > max_queries:
        behavior.notes.append(
            f"search_query_count {extracted['search_query_count']} > max {max_queries}"
        )

    return CaseRunResult(
        case_id=case_id,
        session_id=session_id,
        query=query,
        final_answer=answer,
        events=all_events,
        search_urls=extracted["search_urls"],
        fetched_urls=extracted["fetched_urls"],
        citation_urls=citation_urls,
        search_query_count=extracted["search_query_count"],
        ended_no_evidence=extracted.get("ended_no_evidence", False),
        retrieval=retrieval,
        behavior=behavior,
        streaming=streaming,
        judge=judge_result,
        phases_executed=extracted.get("phases_executed") or [],
        citation_report=extracted.get("citation_report"),
        duration_sec=round(duration, 2),
    )
