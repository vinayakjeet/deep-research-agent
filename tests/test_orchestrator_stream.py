"""Tests for generator-based orchestrator streaming."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from deep_research_agent.ingestion.pipeline import AcquiredSource, IngestionPipelineResult
from deep_research_agent.memory import MemoryStore
from deep_research_agent.orchestration import ResearchOrchestrator
from deep_research_agent.state.enums import AgentPhase
from deep_research_agent.state.schema import ContextSnippet, ResearchPlan, SearchResult


class OrchestratorStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "stream.db")
        self.session_id = self.store.create_session()["session_id"]

        self.mock_gemini = MagicMock()
        self.mock_search = MagicMock()
        self.mock_pipeline = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_answer = MagicMock()

        self.mock_planner.plan.return_value = ResearchPlan(
            summary="GDP",
            search_queries=["India GDP"],
            steps=["search"],
        )
        self.mock_search.search.return_value = [
            SearchResult(title="India", url="https://example.com/india", snippet="7%"),
        ]
        self.mock_pipeline.acquire_from_urls.return_value = IngestionPipelineResult(
            acquired=[
                AcquiredSource(
                    url="https://example.com/india",
                    title="India GDP",
                    domain="example.com",
                    full_text="India GDP grew.",
                    snippets=[
                        ContextSnippet(
                            url="https://example.com/india",
                            snippet="India GDP grew.",
                            title="India GDP",
                            domain="example.com",
                        )
                    ],
                )
            ],
            failed_urls=[],
        )
        self.mock_answer.generate.return_value = (
            "Answer [India GDP — example.com](https://example.com/india)."
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

    def test_stream_yields_phase_events_then_complete(self) -> None:
        events = list(self.orchestrator.run_stream(self.session_id, "India GDP"))
        types = [e["event"] for e in events]
        self.assertIn("phase_start", types)
        self.assertIn("phase_complete", types)
        self.assertEqual(types[-1], "complete")
        final = events[-1]
        self.assertIn("final_answer", final)
        self.assertNotIn("token", str(final).lower())

    def test_no_evidence_stream_emits_no_evidence_event(self) -> None:
        self.mock_search.search.return_value = []
        orch = ResearchOrchestrator(
            self.store,
            gemini_client=self.mock_gemini,
            search_provider=self.mock_search,
            ingestion_pipeline=self.mock_pipeline,
            planner=self.mock_planner,
            answer_generator=self.mock_answer,
            max_replan_attempts=0,
        )
        events = list(orch.run_stream(self.session_id, "Obscure"))
        statuses = [e.get("status") for e in events]
        self.assertIn("NoEvidence", statuses)
        self.mock_answer.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
