"""Contextual chunking and TF-based selection."""

from deep_research_agent.ingestion.select.chunker import split_by_max_chars, split_paragraphs
from deep_research_agent.ingestion.select.tf_selector import (
    TfSelectorConfig,
    blocks_to_snippets,
    select_context_for_query,
    select_relevant_blocks,
    tf_score,
    tokenize,
)

__all__ = [
    "TfSelectorConfig",
    "blocks_to_snippets",
    "select_context_for_query",
    "select_relevant_blocks",
    "split_by_max_chars",
    "split_paragraphs",
    "tf_score",
    "tokenize",
]
