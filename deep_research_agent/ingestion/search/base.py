"""Search provider protocol and factory."""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from deep_research_agent.state.schema import SearchResult


@runtime_checkable
class SearchProvider(Protocol):
    """Contract for search API clients."""

    provider_name: str

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Execute a search and return normalized results."""
        ...


def normalize_result(raw: dict[str, Any]) -> SearchResult:
    """Map provider-specific fields to SearchResult."""
    return SearchResult.from_dict(raw)


def get_search_provider(name: str | None = None) -> SearchProvider:
    """Instantiate the configured search provider from environment."""
    provider = (name or os.environ.get("SEARCH_PROVIDER", "tavily")).strip().lower()

    if provider == "tavily":
        from deep_research_agent.ingestion.search.tavily import TavilySearchProvider

        return TavilySearchProvider()
    if provider == "serper":
        from deep_research_agent.ingestion.search.serper import SerperSearchProvider

        return SerperSearchProvider()

    raise ValueError(f"Unknown search provider: {provider!r}. Use 'tavily' or 'serper'.")
