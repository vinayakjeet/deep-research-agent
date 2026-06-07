"""Tests for search provider network wrappers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import requests

from deep_research_agent.ingestion.errors import SearchProviderError
from deep_research_agent.ingestion.retry import retry_with_backoff
from deep_research_agent.ingestion.search.base import get_search_provider, normalize_result
from deep_research_agent.ingestion.search.serper import SerperSearchProvider
from deep_research_agent.ingestion.search.tavily import TavilySearchProvider


class RetryTests(unittest.TestCase):
    def test_retries_on_429_then_succeeds(self) -> None:
        calls = {"n": 0}

        def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                exc = requests.HTTPError("rate limited")
                exc.response = MagicMock(status_code=429)
                raise exc
            return "ok"

        with patch("deep_research_agent.ingestion.retry.time.sleep"):
            result = retry_with_backoff(
                fn,
                max_retries=3,
                get_status_code=lambda e: getattr(e.response, "status_code", None),
            )
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 2)

    def test_raises_after_max_retries(self) -> None:
        def fn() -> str:
            exc = requests.HTTPError("rate limited")
            exc.response = MagicMock(status_code=429)
            raise exc

        with patch("deep_research_agent.ingestion.retry.time.sleep"):
            with self.assertRaises(requests.HTTPError):
                retry_with_backoff(fn, max_retries=2, get_status_code=lambda e: 429)


class NormalizeResultTests(unittest.TestCase):
    def test_normalize_maps_snippet_and_content(self) -> None:
        r1 = normalize_result({"title": "T", "url": "https://x.com", "snippet": "s"})
        r2 = normalize_result({"title": "T", "url": "https://y.com", "content": "c"})
        self.assertEqual(r1.snippet, "s")
        self.assertEqual(r2.snippet, "c")


class TavilySearchProviderTests(unittest.TestCase):
    def _mock_response(self, payload: dict, status: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        return resp

    @patch("deep_research_agent.ingestion.search.tavily.requests.post")
    def test_search_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = self._mock_response(
            {
                "results": [
                    {
                        "title": "RAG Guide",
                        "url": "https://example.com/rag",
                        "content": "Retrieval augmented generation explained.",
                        "score": 0.9,
                    }
                ]
            }
        )
        provider = TavilySearchProvider(api_key="test-key")
        results = provider.search("what is RAG", max_results=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "RAG Guide")
        self.assertEqual(results[0].url, "https://example.com/rag")
        self.assertIn("Retrieval", results[0].snippet)
        mock_post.assert_called_once()

    @patch("deep_research_agent.ingestion.search.tavily.requests.post")
    def test_search_retries_on_429(self, mock_post: MagicMock) -> None:
        ok = self._mock_response(
            {"results": [{"title": "T", "url": "https://a.com", "content": "x"}]}
        )
        err = MagicMock()
        err.status_code = 429
        err.__enter__ = MagicMock(return_value=err)
        http_err = requests.HTTPError("429", response=err)

        def side_effect(*_a, **_k):
            side_effect.calls += 1
            if side_effect.calls == 1:
                raise http_err
            return ok

        side_effect.calls = 0
        mock_post.side_effect = side_effect

        provider = TavilySearchProvider(api_key="test-key", max_retries=2)
        with patch("deep_research_agent.ingestion.retry.time.sleep"):
            results = provider.search("query")
        self.assertEqual(len(results), 1)
        self.assertEqual(mock_post.call_count, 2)

    def test_missing_api_key_raises(self) -> None:
        provider = TavilySearchProvider(api_key="")
        with self.assertRaises(SearchProviderError):
            provider.search("test")


class SerperSearchProviderTests(unittest.TestCase):
    @patch("deep_research_agent.ingestion.search.serper.requests.post")
    def test_search_success(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "organic": [
                {
                    "title": "GDP India",
                    "link": "https://example.com/india",
                    "snippet": "Growth 7%",
                    "position": 1,
                }
            ]
        }
        mock_post.return_value = resp

        provider = SerperSearchProvider(api_key="serper-key")
        results = provider.search("India GDP", max_results=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/india")
        self.assertEqual(results[0].snippet, "Growth 7%")


class SearchProviderFactoryTests(unittest.TestCase):
    @patch.dict("os.environ", {"SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "k"})
    def test_factory_tavily(self) -> None:
        provider = get_search_provider()
        self.assertIsInstance(provider, TavilySearchProvider)

    @patch.dict("os.environ", {"SEARCH_PROVIDER": "serper", "SERPER_API_KEY": "k"})
    def test_factory_serper(self) -> None:
        provider = get_search_provider()
        self.assertIsInstance(provider, SerperSearchProvider)


if __name__ == "__main__":
    unittest.main()
