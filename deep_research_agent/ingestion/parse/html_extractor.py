"""High-performance HTML to plain text extraction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from selectolax.parser import HTMLParser

from deep_research_agent.state.schema import SourceContext

try:
    from readability import Document as ReadabilityDocument
except ImportError:  # pragma: no cover
    ReadabilityDocument = None  # type: ignore[misc, assignment]

DEFAULT_TAGS_TO_REMOVE = frozenset(
    {"script", "style", "nav", "footer", "aside", "noscript", "iframe"}
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class HtmlExtractorConfig:
    """Settings for DOM pruning and text normalization."""

    tags_to_remove: frozenset[str] = DEFAULT_TAGS_TO_REMOVE
    min_text_length: int = 80
    use_readability_fallback: bool = True


class HtmlExtractor:
    """Extract readable body text from raw HTML using selectolax with optional fallback."""

    def __init__(self, config: Optional[HtmlExtractorConfig] = None) -> None:
        self.config = config or HtmlExtractorConfig()

    def extract(self, html: str | bytes, *, url: str | None = None) -> str:
        if isinstance(html, bytes):
            text_input = html.decode("utf-8", errors="replace")
        else:
            text_input = html

        primary = self._extract_with_selectolax(text_input)
        if len(primary) >= self.config.min_text_length:
            return primary

        if self.config.use_readability_fallback and ReadabilityDocument is not None:
            fallback = self._extract_with_readability(text_input)
            if len(fallback) >= self.config.min_text_length:
                return fallback
            if len(fallback) > len(primary):
                return fallback

        return primary

    def _extract_with_selectolax(self, html: str) -> str:
        tree = HTMLParser(html)
        for tag in self.config.tags_to_remove:
            for node in tree.css(tag):
                node.decompose()

        body = tree.body
        if body is not None:
            raw = body.text(separator=" ")
        else:
            raw = tree.text(separator=" ")

        return self._normalize_text(raw)

    def _extract_with_readability(self, html: str) -> str:
        if ReadabilityDocument is None:
            return ""
        try:
            doc = ReadabilityDocument(html)
            summary = doc.summary() or ""
            if not summary:
                return ""
            inner = HTMLParser(summary)
            return self._normalize_text(inner.text(separator=" "))
        except Exception:
            return ""

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        collapsed = _WHITESPACE_RE.sub(" ", normalized)
        return collapsed.strip()


def domain_from_url(url: str) -> str:
    """Extract hostname from a URL for citation metadata."""
    parsed = urlparse(url)
    return parsed.netloc or ""


def to_source_context(
    turn_id: int,
    url: str,
    text: str,
    *,
    title: Optional[str] = None,
    domain: Optional[str] = None,
    context_id: int = 0,
    retrieved_at: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> SourceContext:
    """Build a SourceContext from extracted page text."""
    return SourceContext(
        context_id=context_id,
        turn_id=turn_id,
        url=url,
        title=title,
        domain=domain or domain_from_url(url),
        text_block=text,
        retrieved_at=retrieved_at,
        metadata=metadata,
    )
