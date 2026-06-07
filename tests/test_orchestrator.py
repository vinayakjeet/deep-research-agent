"""Tests for the master research orchestration loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deep_research_agent.ingestion.pipeline import AcquiredSource, IngestionPipelineResult
from deep_research_agent.memory import MemoryStore
from deep_research_agent.orchestration import ResearchOrchestrator
from deep_research_agent.state.enums import AgentPhase
from deep_research_agent.state.schema import ContextSnippet, ResearchPlan, SearchResult


class ResearchOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "orch.db")
        self.session = self.store.create_session()
        self.session_id = self.session["session_id"]

        self.mock_gemini = MagicMock()
        self.mock_search = MagicMock()
        self.mock_pipeline = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_answer = MagicMock()

        self.mock_planner.plan.return_value = ResearchPlan(
            summary="Compare GDP growth",
            search_queries=["India GDP 2024", "China GDP 2024"],
            steps=["search", "compare", "answer"],
        )
        self.mock_search.search.return_value = [
            SearchResult(title="India", url="https://example.com/india", snippet="7%"),
            SearchResult(title="China", url="https://example.com/china", snippet="5%"),
        ]
        self._acquire_result = IngestionPipelineResult(
            acquired=[
                AcquiredSource(
                    url="https://example.com/india",
                    title="India GDP",
                    domain="example.com",
                    full_text="India GDP grew by 7%.",
                    snippets=[
                        ContextSnippet(
                            url="https://example.com/india",
                            snippet="India GDP grew by 7%.",
                            title="India GDP",
                            domain="example.com",
                        )
                    ],
                )
            ],
            failed_urls=[],
        )
        self.mock_pipeline.acquire_from_urls.side_effect = self._mock_acquire
        self.mock_answer.generate.return_value = (
            "India grew faster than China [India GDP — example.com]."
        )

        self.orchestrator = ResearchOrchestrator(
            self.store,
            gemini_client=self.mock_gemini,
            search_provider=self.mock_search,
            ingestion_pipeline=self.mock_pipeline,
            planner=self.mock_planner,
            answer_generator=self.mock_answer,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _mock_acquire(self, urls, query, *, titles=None, tracker=None):
        """Simulate pipeline side effects on the turn tracker."""
        if tracker is not None:
            for src in self._acquire_result.acquired:
                tracker.record_url_extraction(
                    url=src.url,
                    text_block=src.full_text,
                    title=src.title,
                    domain=src.domain,
                )
                if src.snippets:
                    tracker.record_snippets(src.snippets)
        return self._acquire_result

    def test_full_phase_sequence(self) -> None:
        result = self.orchestrator.run(self.session_id, "Compare GDP of India and China")

        self.assertIn("India", result.final_answer)
        self.assertEqual(result.state.phase, AgentPhase.COMPLETE)

        expected = [
            AgentPhase.PLANNING.value,
            AgentPhase.SEARCHING.value,
            AgentPhase.ACQUIRING.value,
            AgentPhase.ANSWERING.value,
            AgentPhase.COMPLETE.value,
        ]
        self.assertEqual(result.phases_executed, expected)

    def test_acquire_called_before_answer(self) -> None:
        call_order: list[str] = []

        def track_plan(*_a, **_k):
            call_order.append("plan")
            return self.mock_planner.plan.return_value

        def track_search(*_a, **_k):
            call_order.append("search")
            return self.mock_search.search.return_value

        def track_acquire(*_a, **kwargs):
            call_order.append("acquire")
            return self._mock_acquire(*_a, **kwargs)

        def track_answer(*_a, **_k):
            call_order.append("answer")
            return self.mock_answer.generate.return_value

        self.mock_planner.plan.side_effect = track_plan
        self.mock_search.search.side_effect = track_search
        self.mock_pipeline.acquire_from_urls.side_effect = track_acquire
        self.mock_answer.generate.side_effect = track_answer

        self.orchestrator.run(self.session_id, "Compare GDP")

        self.assertLess(call_order.index("acquire"), call_order.index("answer"))
        self.mock_pipeline.acquire_from_urls.assert_called_once()

    def test_persists_turn_to_sqlite(self) -> None:
        result = self.orchestrator.run(self.session_id, "What is RAG?")
        turn = self.store.get_turn(result.turn_id)
        assert turn is not None
        self.assertEqual(turn["search_queries"], ["India GDP 2024", "China GDP 2024"])
        self.assertGreater(len(turn["context_snippets"]), 0)
        self.assertIn("India grew faster", turn["final_answer"])

    def test_empty_search_returns_no_evidence_without_llm(self) -> None:
        self.mock_search.search.return_value = []
        self.mock_planner.plan.return_value = ResearchPlan(
            summary="x",
            search_queries=["obscure query"],
            steps=["search"],
        )
        orch = ResearchOrchestrator(
            self.store,
            gemini_client=self.mock_gemini,
            search_provider=self.mock_search,
            ingestion_pipeline=self.mock_pipeline,
            planner=self.mock_planner,
            answer_generator=self.mock_answer,
            max_replan_attempts=0,
        )
        result = orch.run(self.session_id, "Obscure topic xyz")
        self.assertEqual(result.state.phase, AgentPhase.COMPLETE)
        self.assertIn("could not locate verifiable evidence", result.final_answer.lower())
        self.mock_answer.generate.assert_not_called()
        self.mock_pipeline.acquire_from_urls.assert_not_called()


class PhaseGuardTests(unittest.TestCase):
    def test_next_phase_sequence(self) -> None:
        from deep_research_agent.orchestration.orchestrator import _next_phase

        self.assertEqual(_next_phase(AgentPhase.START), AgentPhase.PLANNING)
        self.assertEqual(_next_phase(AgentPhase.PLANNING), AgentPhase.SEARCHING)
        self.assertEqual(_next_phase(AgentPhase.SEARCHING), AgentPhase.ACQUIRING)
        self.assertEqual(_next_phase(AgentPhase.ACQUIRING), AgentPhase.ANSWERING)
        self.assertEqual(_next_phase(AgentPhase.ANSWERING), AgentPhase.COMPLETE)


if __name__ == "__main__":
    unittest.main()
