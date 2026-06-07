"""Search provider clients."""

from deep_research_agent.ingestion.search.base import (
    SearchProvider,
    get_search_provider,
    normalize_result,
)
from deep_research_agent.ingestion.search.serper import SerperSearchProvider
from deep_research_agent.ingestion.search.tavily import TavilySearchProvider

__all__ = [
    "SearchProvider",
    "TavilySearchProvider",
    "SerperSearchProvider",
    "get_search_provider",
    "normalize_result",
]
