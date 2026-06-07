"""Web ingestion: search, fetch, parse, and context selection."""

from deep_research_agent.ingestion.errors import (
    FetchError,
    IngestionError,
    ParseError,
    SearchProviderError,
)
from deep_research_agent.ingestion.fetch import FetchResult, FetchStatus, PageFetcher, PageFetcherConfig
from deep_research_agent.ingestion.parse import (
    HtmlExtractor,
    HtmlExtractorConfig,
    domain_from_url,
    to_source_context,
)
from deep_research_agent.ingestion.pipeline import (
    AcquiredSource,
    IngestionPipeline,
    IngestionPipelineResult,
)
from deep_research_agent.ingestion.search import (
    SearchProvider,
    SerperSearchProvider,
    TavilySearchProvider,
    get_search_provider,
)
from deep_research_agent.ingestion.select import (
    TfSelectorConfig,
    select_context_for_query,
    select_relevant_blocks,
)

__all__ = [
    "AcquiredSource",
    "FetchError",
    "FetchResult",
    "FetchStatus",
    "HtmlExtractor",
    "HtmlExtractorConfig",
    "IngestionError",
    "IngestionPipeline",
    "IngestionPipelineResult",
    "PageFetcher",
    "PageFetcherConfig",
    "ParseError",
    "SearchProvider",
    "SearchProviderError",
    "SerperSearchProvider",
    "TavilySearchProvider",
    "TfSelectorConfig",
    "domain_from_url",
    "get_search_provider",
    "select_context_for_query",
    "select_relevant_blocks",
    "to_source_context",
]
