"""High-level ingestion pipeline chaining search, fetch, parse, and select."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from deep_research_agent.ingestion.fetch import FetchResult, FetchStatus, PageFetcher, PageFetcherConfig
from deep_research_agent.ingestion.parse import HtmlExtractor, HtmlExtractorConfig, domain_from_url
from deep_research_agent.ingestion.search import SearchProvider, get_search_provider
from deep_research_agent.ingestion.select import TfSelectorConfig, select_context_for_query
from deep_research_agent.memory.turn_tracker import TurnHistoryTracker
from deep_research_agent.state.schema import ContextSnippet, SearchResult


@dataclass(slots=True)
class AcquiredSource:
    """One successfully processed source URL."""

    url: str
    title: Optional[str]
    domain: str
    full_text: str
    snippets: list[ContextSnippet] = field(default_factory=list)
    fetch_result: Optional[FetchResult] = None


@dataclass(slots=True)
class IngestionPipelineResult:
    """Output of a full search-and-acquire pass."""

    search_results: list[SearchResult] = field(default_factory=list)
    acquired: list[AcquiredSource] = field(default_factory=list)
    failed_urls: list[dict[str, Any]] = field(default_factory=list)


class IngestionPipeline:
    """
    Orchestrates Phase 2 modules without the Phase 3 LLM loop.

    Flow: search -> fetch URLs -> extract text -> TF-select snippets -> audit log.
    """

    def __init__(
        self,
        *,
        search_provider: Optional[SearchProvider] = None,
        fetcher: Optional[PageFetcher] = None,
        extractor: Optional[HtmlExtractor] = None,
        selector_config: Optional[TfSelectorConfig] = None,
    ) -> None:
        self.search_provider = search_provider or get_search_provider()
        self.fetcher = fetcher or PageFetcher()
        self.extractor = extractor or HtmlExtractor()
        self.selector_config = selector_config or TfSelectorConfig()

    def run_search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return self.search_provider.search(query, max_results=max_results)

    def acquire_from_urls(
        self,
        urls: list[str],
        query: str,
        *,
        titles: Optional[dict[str, str]] = None,
        tracker: Optional[TurnHistoryTracker] = None,
    ) -> IngestionPipelineResult:
        titles = titles or {}
        result = IngestionPipelineResult()

        fetch_results = self.fetcher.fetch_many_sync(urls)

        for fr in fetch_results:
            status_value = fr.status.value
            if tracker is not None:
                tracker.record_fetch_outcome(
                    fr.url,
                    fetch_status=status_value,
                    status_code=fr.status_code,
                    error=fr.error,
                    elapsed_ms=fr.elapsed_ms,
                )

            if not fr.ok:
                result.failed_urls.append(
                    {
                        "url": fr.url,
                        "status": status_value,
                        "status_code": fr.status_code,
                        "error": fr.error,
                    }
                )
                continue

            text = self.extractor.extract(fr.html or "", url=fr.url)
            if not text.strip():
                result.failed_urls.append(
                    {"url": fr.url, "status": "empty_text", "error": "No extractable text"}
                )
                if tracker is not None:
                    tracker.record_fetch_outcome(
                        fr.url,
                        fetch_status="http_error",
                        error="No extractable text",
                    )
                continue

            title = titles.get(fr.url)
            domain = domain_from_url(fr.url)
            snippets = select_context_for_query(
                text,
                query,
                fr.url,
                title=title,
                domain=domain,
                config=self.selector_config,
            )

            if tracker is not None:
                tracker.record_url_extraction(
                    url=fr.url,
                    text_block=text,
                    title=title,
                    domain=domain,
                )
                if snippets:
                    tracker.record_snippets(snippets)

            result.acquired.append(
                AcquiredSource(
                    url=fr.url,
                    title=title,
                    domain=domain,
                    full_text=text,
                    snippets=snippets,
                    fetch_result=fr,
                )
            )

        return result

    def search_and_acquire(
        self,
        query: str,
        *,
        max_results: int = 5,
        tracker: Optional[TurnHistoryTracker] = None,
    ) -> IngestionPipelineResult:
        """Search for a query then fetch and process result URLs."""
        search_results = self.run_search(query, max_results=max_results)
        urls = [r.url for r in search_results if r.url]
        titles = {r.url: r.title for r in search_results if r.url}

        if tracker is not None:
            tracker.record_search_queries([query])

        acquired = self.acquire_from_urls(
            urls,
            query,
            titles=titles,
            tracker=tracker,
        )
        acquired.search_results = search_results
        return acquired
