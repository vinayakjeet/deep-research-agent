"""Term-frequency based block selection for query relevance."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from deep_research_agent.ingestion.select.chunker import split_paragraphs
from deep_research_agent.state.schema import ContextSnippet

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "with",
        "from",
        "by",
        "about",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "again",
        "further",
        "then",
        "once",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "how",
        "when",
        "where",
        "why",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, excluding stopwords."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def tf_score(block: str, query_terms: list[str]) -> float:
    """Score a block by term frequency overlap with query terms."""
    if not query_terms:
        return 0.0

    block_tokens = tokenize(block)
    if not block_tokens:
        return 0.0

    counts: dict[str, int] = {}
    for token in block_tokens:
        counts[token] = counts.get(token, 0) + 1

    score = 0.0
    for term in query_terms:
        tf = counts.get(term, 0)
        if tf > 0:
            score += 1.0 + math.log1p(tf)

    # Mild length normalization to avoid favoring huge boilerplate blocks
    score /= math.sqrt(len(block_tokens))
    return score


@dataclass(slots=True)
class TfSelectorConfig:
    """Settings for contextual block selection."""

    top_k: int = 5
    min_score: float = 0.0
    min_block_chars: int = 40


def select_relevant_blocks(
    text: str,
    query: str,
    *,
    top_k: int = 5,
    min_score: float = 0.0,
    min_block_chars: int = 40,
) -> list[str]:
    """Return the top TF-scored paragraph blocks for a query."""
    blocks = split_paragraphs(text, min_block_chars=min_block_chars)
    if not blocks:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return blocks[:top_k]

    scored = [(block, tf_score(block, query_terms)) for block in blocks]
    scored.sort(key=lambda item: item[1], reverse=True)

    selected: list[str] = []
    for block, score in scored:
        if score < min_score:
            continue
        selected.append(block)
        if len(selected) >= top_k:
            break

    return selected


def blocks_to_snippets(
    blocks: list[str],
    url: str,
    *,
    title: Optional[str] = None,
    domain: Optional[str] = None,
) -> list[ContextSnippet]:
    """Convert text blocks into ContextSnippet records."""
    return [
        ContextSnippet(
            url=url,
            snippet=block,
            title=title,
            domain=domain,
        )
        for block in blocks
    ]


def select_context_for_query(
    full_text: str,
    query: str,
    url: str,
    *,
    title: Optional[str] = None,
    domain: Optional[str] = None,
    config: Optional[TfSelectorConfig] = None,
) -> list[ContextSnippet]:
    """Chunk, score, and return the most relevant snippets for a search query."""
    cfg = config or TfSelectorConfig()
    blocks = select_relevant_blocks(
        full_text,
        query,
        top_k=cfg.top_k,
        min_score=cfg.min_score,
        min_block_chars=cfg.min_block_chars,
    )
    return blocks_to_snippets(blocks, url, title=title, domain=domain)
