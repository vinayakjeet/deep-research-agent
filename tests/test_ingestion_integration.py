"""Integration tests for the Phase 2 ingestion pipeline (mocked network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deep_research_agent.ingestion.fetch.models import FetchResult, FetchStatus
from deep_research_agent.ingestion.pipeline import IngestionPipeline
from deep_research_agent.memory import MemoryStore, TurnHistoryTracker
from deep_research_agent.state.schema import SearchResult

SAMPLE_HTML = """
<html><body>
<nav>Menu</nav>
<p>India GDP growth reached seven percent in the latest fiscal year.</p>
<p>China GDP growth was five percent according to official statistics.</p>
<footer>Copyright</footer>
</body></html>
"""


class IngestionPipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "pipeline.db")
        session = self.store.create_session()
        self.tracker = TurnHistoryTracker(self.store, session["session_id"])
        self.tracker.begin("Compare India and China GDP")

        self.mock_search = MagicMock()
        self.mock_search.search.return_value = [
            SearchResult(
                title="India GDP",
                url="https://example.com/india",
                snippet="India growth",
            ),
            SearchResult(
                title="China GDP",
                url="https://example.com/china",
                snippet="China growth",
            ),
        ]

        self.pipeline = IngestionPipeline(search_provider=self.mock_search)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    @patch.object(IngestionPipeline, "acquire_from_urls")
    def test_search_and_acquire_wires_tracker(self, mock_acquire: MagicMock) -> None:
        from deep_research_agent.ingestion.pipeline import AcquiredSource, IngestionPipelineResult
        from deep_research_agent.ingestion.select import select_context_for_query

        text = "India GDP growth reached seven percent."
        snippets = select_context_for_query(
            text,
            "India GDP",
            "https://example.com/india",
            title="India GDP",
        )
        mock_acquire.return_value = IngestionPipelineResult(
            search_results=self.mock_search.search.return_value,
            acquired=[
                AcquiredSource(
                    url="https://example.com/india",
                    title="India GDP",
                    domain="example.com",
                    full_text=text,
                    snippets=snippets,
                )
            ],
        )

        result = self.pipeline.search_and_acquire(
            "India GDP 2024",
            max_results=2,
            tracker=self.tracker,
        )

        self.assertEqual(len(result.search_results), 2)
        mock_acquire.assert_called_once()
        audit = self.tracker.to_audit_dict()
        self.assertIn("India GDP 2024", audit["search"]["queries"])

    def test_acquire_from_urls_with_mocked_fetch(self) -> None:
        def fake_fetch_many_sync(urls: list[str]) -> list[FetchResult]:
            out: list[FetchResult] = []
            for url in urls:
                if "missing" in url:
                    out.append(
                        FetchResult(
                            url=url,
                            status=FetchStatus.HTTP_ERROR,
                            status_code=404,
                            error="HTTP 404",
                        )
                    )
                else:
                    out.append(
                        FetchResult(
                            url=url,
                            status=FetchStatus.OK,
                            status_code=200,
                            html=SAMPLE_HTML,
                        )
                    )
            return out

        self.pipeline.fetcher.fetch_many_sync = fake_fetch_many_sync  # type: ignore[method-assign]

        result = self.pipeline.acquire_from_urls(
            ["https://example.com/india", "https://example.com/missing"],
            "India GDP",
            titles={"https://example.com/india": "India GDP"},
            tracker=self.tracker,
        )

        self.assertEqual(len(result.acquired), 1)
        self.assertEqual(len(result.failed_urls), 1)
        self.assertGreater(len(result.acquired[0].snippets), 0)
        combined = " ".join(s.snippet for s in result.acquired[0].snippets).lower()
        self.assertIn("india", combined)

        audit = self.tracker.to_audit_dict()
        failed_events = [e for e in audit["events"] if e["event_type"] == "fetch_failed"]
        self.assertEqual(len(failed_events), 1)


if __name__ == "__main__":
    unittest.main()
