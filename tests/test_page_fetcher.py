"""Tests for async page fetcher and turn tracker fetch audit."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from deep_research_agent.ingestion.fetch import FetchStatus, PageFetcher, PageFetcherConfig
from deep_research_agent.memory import MemoryStore, TurnHistoryTracker


class PageFetcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_many_returns_all_results(self) -> None:
        fetcher = PageFetcher(PageFetcherConfig(timeout_sec=5.0, max_concurrent=3))

        ok_result = MagicMock()
        ok_result.status = 200
        ok_result.charset = "utf-8"
        ok_result.read = AsyncMock(return_value=b"<html>ok</html>")

        err_result = MagicMock()
        err_result.status = 404
        err_result.charset = "utf-8"
        err_result.read = AsyncMock(return_value=b"not found")

        class FakeCtx:
            def __init__(self, resp: MagicMock) -> None:
                self._resp = resp

            async def __aenter__(self) -> MagicMock:
                return self._resp

            async def __aexit__(self, *_a: object) -> None:
                return None

        urls = [
            "https://example.com/ok",
            "https://example.com/missing",
            "https://example.com/ok2",
        ]

        def fake_get(url: str, **_kwargs: object) -> FakeCtx:
            if "missing" in url:
                return FakeCtx(err_result)
            return FakeCtx(ok_result)

        with patch("aiohttp.ClientSession.get", side_effect=fake_get):
            results = await fetcher.fetch_many(urls)

        self.assertEqual(len(results), 3)
        statuses = {r.url: r.status for r in results}
        self.assertEqual(statuses["https://example.com/ok"], FetchStatus.OK)
        self.assertEqual(statuses["https://example.com/missing"], FetchStatus.HTTP_ERROR)
        self.assertEqual(statuses["https://example.com/ok2"], FetchStatus.OK)

    async def test_timeout_classified(self) -> None:
        fetcher = PageFetcher(PageFetcherConfig(timeout_sec=0.01, max_retries=0))

        class TimeoutCtx:
            async def __aenter__(self) -> None:
                raise asyncio.TimeoutError()

            async def __aexit__(self, *_a: object) -> None:
                return None

        with patch("aiohttp.ClientSession.get", return_value=TimeoutCtx()):
            result = await fetcher.fetch_many(["https://slow.example.com"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, FetchStatus.TIMEOUT)

    def test_fetch_many_sync(self) -> None:
        fetcher = PageFetcher()

        async def fake_fetch_many(urls: list[str]) -> list:
            from deep_research_agent.ingestion.fetch.models import FetchResult

            return [
                FetchResult(url=u, status=FetchStatus.OK, status_code=200, html="<p>x</p>")
                for u in urls
            ]

        with patch.object(fetcher, "fetch_many", side_effect=fake_fetch_many):
            results = fetcher.fetch_many_sync(["https://a.com", "https://b.com"])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.ok for r in results))


class TurnTrackerFetchAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "fetch_audit.db")
        session = self.store.create_session()
        self.tracker = TurnHistoryTracker(self.store, session["session_id"])
        self.tracker.begin("Fetch audit test")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_record_fetch_failure_in_audit(self) -> None:
        self.tracker.record_fetch_outcome(
            "https://example.com/missing",
            fetch_status="http_error",
            status_code=404,
            error="HTTP 404",
        )
        audit = self.tracker.to_audit_dict()
        events = [e for e in audit["events"] if e["event_type"] == "fetch_failed"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["payload"]["status_code"], 404)

    def test_record_fetch_ok_updates_urls(self) -> None:
        self.tracker.record_fetch_outcome(
            "https://example.com/page",
            fetch_status="ok",
            status_code=200,
        )
        audit = self.tracker.to_audit_dict()
        self.assertIn("https://example.com/page", audit["search"]["urls_accessed"])


if __name__ == "__main__":
    unittest.main()
