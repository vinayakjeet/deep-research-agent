"""Master research orchestration loop: PLAN → SEARCH → ACQUIRE → ANSWER."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Optional

from deep_research_agent.context import ContextBuilder
from deep_research_agent.ingestion import IngestionPipeline
from deep_research_agent.ingestion.pipeline import IngestionPipelineResult
from deep_research_agent.ingestion.search import SearchProvider, get_search_provider
from deep_research_agent.llm.gemini_client import GeminiClient
from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.memory.turn_tracker import TurnHistoryTracker
from deep_research_agent.orchestration.citations import (
    CitationReport,
    build_authorized_sources,
    validate_citations,
)
from deep_research_agent.orchestration.context_assembly import AnswerGenerator
from deep_research_agent.orchestration.events import (
    error_event,
    no_evidence_event,
    phase_complete,
    phase_start,
    terminal_event,
)
from deep_research_agent.orchestration.fallbacks import build_no_evidence_response
from deep_research_agent.orchestration.planner import Planner
from deep_research_agent.state.adapters import load_agent_state
from deep_research_agent.state.enums import AgentPhase, MessageRole
from deep_research_agent.state.schema import AgentState, ContextSnippet, SearchResult, SourceContext


def _next_phase(current: AgentPhase) -> AgentPhase:
    """Return the only legal successor phase in the research loop."""
    transitions = {
        AgentPhase.START: AgentPhase.PLANNING,
        AgentPhase.PLANNING: AgentPhase.SEARCHING,
        AgentPhase.SEARCHING: AgentPhase.ACQUIRING,
        AgentPhase.ACQUIRING: AgentPhase.ANSWERING,
        AgentPhase.ANSWERING: AgentPhase.COMPLETE,
    }
    if current not in transitions:
        return AgentPhase.ERROR
    return transitions[current]


@dataclass(slots=True)
class ResearchTurnResult:
    """Outcome of one orchestrated research turn."""

    session_id: str
    turn_id: int
    final_answer: str
    state: AgentState
    phases_executed: list[str] = field(default_factory=list)
    citation_report: Optional[CitationReport] = None


class ResearchOrchestrator:
    """
    Runs the quadripartite research loop for a single user query.

    Phases execute strictly in order; ACQUIRING cannot be skipped before ANSWERING.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        gemini_client: Optional[GeminiClient] = None,
        search_provider: Optional[SearchProvider] = None,
        ingestion_pipeline: Optional[IngestionPipeline] = None,
        planner: Optional[Planner] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        context_builder: Optional[ContextBuilder] = None,
        max_replan_attempts: int = 1,
    ) -> None:
        self.store = store
        self.gemini_client = gemini_client or GeminiClient()
        self.search_provider = search_provider or get_search_provider()
        self.pipeline = ingestion_pipeline or IngestionPipeline()
        self.planner = planner or Planner(self.gemini_client)
        self.answer_generator = answer_generator or AnswerGenerator(self.gemini_client)
        self.context_builder = context_builder or ContextBuilder()
        self.max_replan_attempts = max_replan_attempts

    def run(self, session_id: str, user_query: str) -> ResearchTurnResult:
        """Execute a full research turn and return the final answer."""
        state = load_agent_state(self.store, session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}")

        final_answer = ""
        phases_executed: list[str] = []
        turn_id = state.turn_id or 0
        citation_report: Optional[CitationReport] = None

        for event in self.run_stream(session_id, user_query):
            if event.get("event") == "complete":
                final_answer = event.get("final_answer", "")
                details = event.get("details", {})
                phases_executed = list(details.get("phases_executed", []))
                turn_id = int(details.get("turn_id", turn_id))
                cr = details.get("citation_report")
                if cr:
                    citation_report = self._citation_report_from_dict(cr)
            state = load_agent_state(self.store, session_id) or state

        return ResearchTurnResult(
            session_id=session_id,
            turn_id=turn_id,
            final_answer=final_answer,
            state=state,
            phases_executed=phases_executed,
            citation_report=citation_report,
        )

    def run_stream(
        self,
        session_id: str,
        user_query: str,
    ) -> Generator[dict[str, Any], None, None]:
        """Yield operational status events, then a terminal event with the answer."""
        query = user_query.strip()
        if not query:
            raise ValueError("user_query must be non-empty")

        state = load_agent_state(self.store, session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}")

        state.current_query = query
        state.phase = AgentPhase.START
        state.search_results = []
        state.urls_opened = []
        state.source_contexts = []
        state.selected_snippets = []
        state.final_answer = None
        state.error_message = None

        self.store.add_message(session_id, MessageRole.USER.value, query)

        tracker = TurnHistoryTracker(self.store, session_id)
        turn_id = tracker.begin(query)
        state.turn_id = turn_id

        phases_executed: list[str] = []
        replans = 0
        citation_report: Optional[CitationReport] = None

        while not state.is_terminal():
            phase = state.phase

            if phase == AgentPhase.START:
                state.advance_to(_next_phase(phase))
                phases_executed.append(state.phase.value)
                yield phase_start(state.phase)
                continue

            if phase == AgentPhase.PLANNING:
                yield phase_start(AgentPhase.PLANNING)
                self._run_planning(state, tracker)
                state.advance_to(_next_phase(phase))
                phases_executed.append(state.phase.value)
                yield phase_complete(
                    AgentPhase.PLANNING,
                    details={
                        "search_queries": list(state.search_queries),
                        "summary": state.plan.summary if state.plan else "",
                    },
                )
                continue

            if phase == AgentPhase.SEARCHING:
                yield phase_start(
                    AgentPhase.SEARCHING,
                    message="Searching the web...",
                )
                found = self._run_search(state)
                if not found:
                    if replans < self.max_replan_attempts:
                        replans += 1
                        state.advance_to(AgentPhase.PLANNING)
                        phases_executed.append(AgentPhase.PLANNING.value)
                        yield phase_complete(
                            AgentPhase.SEARCHING,
                            details={"result_count": 0, "replan": True},
                            message="No results; replanning.",
                        )
                        continue
                    yield from self._finalize_no_evidence(
                        session_id,
                        query,
                        state,
                        tracker,
                        phases_executed,
                        reason="No search results were returned for the planned queries.",
                    )
                    citation_report = None
                    yield terminal_event(
                        final_answer=state.final_answer or "",
                        session_id=session_id,
                        turn_id=turn_id,
                        phases_executed=phases_executed,
                    )
                    return
                state.advance_to(_next_phase(phase))
                phases_executed.append(state.phase.value)
                yield phase_complete(
                    AgentPhase.SEARCHING,
                    details={
                        "result_count": len(state.search_results),
                        "urls": [r.url for r in state.search_results[:10]],
                    },
                )
                continue

            if phase == AgentPhase.ACQUIRING:
                yield phase_start(
                    AgentPhase.ACQUIRING,
                    message="Fetching and selecting relevant context...",
                )
                acquired = self._run_acquire(state, tracker)
                if not acquired:
                    yield from self._finalize_no_evidence(
                        session_id,
                        query,
                        state,
                        tracker,
                        phases_executed,
                        reason="No usable text could be extracted from the fetched pages.",
                    )
                    yield terminal_event(
                        final_answer=state.final_answer or "",
                        session_id=session_id,
                        turn_id=turn_id,
                        phases_executed=phases_executed,
                    )
                    return
                state.advance_to(_next_phase(phase))
                phases_executed.append(state.phase.value)
                yield phase_complete(
                    AgentPhase.ACQUIRING,
                    details={
                        "sources_acquired": len(state.source_contexts),
                        "snippets_selected": len(state.selected_snippets),
                        "urls": state.urls_opened[:10],
                    },
                )
                continue

            if phase == AgentPhase.ANSWERING:
                yield phase_start(
                    AgentPhase.ANSWERING,
                    message="Generating answer with citations...",
                )
                summary = self._conversation_summary(state)
                answer = self.answer_generator.generate(
                    query,
                    state,
                    conversation_summary=summary,
                )
                state.final_answer = answer
                citation_report = validate_citations(
                    answer,
                    build_authorized_sources(state),
                )
                if citation_report.hallucination_flags:
                    tracker.record_citation_anomalies(citation_report.to_dict())
                tracker.finalize_answer(answer)
                self.store.add_message(session_id, MessageRole.ASSISTANT.value, answer)
                state.advance_to(_next_phase(phase))
                phases_executed.append(state.phase.value)
                yield phase_complete(
                    AgentPhase.ANSWERING,
                    details={
                        "citation_count": len(citation_report.citations),
                        "hallucination_flags": citation_report.hallucination_flags,
                    },
                )
                continue

            state.advance_to(AgentPhase.ERROR)
            state.error_message = "An unexpected error occurred during research."
            yield error_event(state.error_message)
            break

        if state.phase == AgentPhase.ERROR and state.error_message:
            self.store.add_message(
                session_id,
                MessageRole.ASSISTANT.value,
                state.error_message,
            )

        final = state.final_answer or state.error_message or ""
        yield terminal_event(
            final_answer=final,
            session_id=session_id,
            turn_id=turn_id,
            phases_executed=phases_executed,
            citation_report=citation_report.to_dict() if citation_report else None,
        )

    def _finalize_no_evidence(
        self,
        session_id: str,
        query: str,
        state: AgentState,
        tracker: TurnHistoryTracker,
        phases_executed: list[str],
        *,
        reason: str,
    ) -> Generator[dict[str, Any], None, None]:
        """Bypass ANSWERING and return a hardcoded no-evidence response."""
        answer = build_no_evidence_response(query, reason=reason)
        state.final_answer = answer
        state.error_message = None
        tracker.finalize_answer(answer)
        self.store.add_message(session_id, MessageRole.ASSISTANT.value, answer)
        state.advance_to(AgentPhase.COMPLETE)
        phases_executed.append("no_evidence")
        phases_executed.append(AgentPhase.COMPLETE.value)
        yield no_evidence_event(reason=reason, user_query=query)

    @staticmethod
    def _citation_report_from_dict(data: dict[str, Any]) -> CitationReport:
        from deep_research_agent.orchestration.citations import CitationRecord

        records = [
            CitationRecord(
                raw=c["raw"],
                url=c["url"],
                domain=c["domain"],
                title=c["title"],
                format=c["format"],
                is_valid=c["is_valid"],
            )
            for c in data.get("citations", [])
        ]
        return CitationReport(
            citations=records,
            invalid_domains=list(data.get("invalid_domains", [])),
            hallucination_flags=list(data.get("hallucination_flags", [])),
            has_citations=bool(data.get("has_citations")),
        )

    def _run_planning(self, state: AgentState, tracker: TurnHistoryTracker) -> None:
        built = self.context_builder.build_from_state(state)
        history_text = ""
        if built.conversation_summary:
            history_text = built.conversation_summary
        if built.messages:
            lines = [f"{m['role']}: {m['content']}" for m in built.messages[-6:]]
            history_text = (history_text + "\n" + "\n".join(lines)).strip()

        plan = self.planner.plan(state.current_query or "", conversation_context=history_text)
        state.plan = plan
        state.search_queries = list(plan.search_queries)
        tracker.record_plan(plan.to_dict(), search_queries=plan.search_queries)

    def _run_search(self, state: AgentState) -> bool:
        queries = state.search_queries or (
            state.plan.search_queries if state.plan else []
        )
        if not queries:
            return False

        seen_urls: set[str] = set()
        aggregated: list[SearchResult] = []

        for q in queries:
            try:
                hits = self.search_provider.search(q, max_results=5)
            except Exception:
                continue
            for hit in hits:
                if hit.url and hit.url not in seen_urls:
                    seen_urls.add(hit.url)
                    aggregated.append(hit)

        state.search_results = aggregated
        return len(aggregated) > 0

    def _run_acquire(
        self,
        state: AgentState,
        tracker: TurnHistoryTracker,
    ) -> bool:
        urls = [r.url for r in state.search_results if r.url]
        if not urls:
            return False

        titles = {r.url: r.title for r in state.search_results if r.url}
        query = state.current_query or ""

        result = self.pipeline.acquire_from_urls(
            urls,
            query,
            titles=titles,
            tracker=tracker,
        )
        self._apply_acquisition(state, result)
        return len(state.selected_snippets) > 0 or len(state.source_contexts) > 0

    def _apply_acquisition(
        self,
        state: AgentState,
        result: IngestionPipelineResult,
    ) -> None:
        state.urls_opened = [src.url for src in result.acquired]
        snippets: list[ContextSnippet] = []
        contexts: list[SourceContext] = []

        for index, src in enumerate(result.acquired, start=1):
            snippets.extend(src.snippets)
            contexts.append(
                SourceContext(
                    context_id=index,
                    turn_id=state.turn_id or 0,
                    url=src.url,
                    title=src.title,
                    domain=src.domain,
                    text_block=src.full_text,
                )
            )

        state.selected_snippets = snippets
        state.source_contexts = contexts

    def _conversation_summary(self, state: AgentState) -> Optional[str]:
        built = self.context_builder.build_from_state(state)
        return built.conversation_summary
