"""Post-generation citation validation against authorized sources."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from deep_research_agent.state.schema import AgentState, ContextSnippet, SourceContext

MARKDOWN_CITATION_RE = re.compile(
    r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
    re.IGNORECASE,
)
PAREN_CITATION_RE = re.compile(
    r"\(([a-zA-Z0-9][-a-zA-Z0-9.]*[a-zA-Z0-9]),\s*(https?://[^)\s]+)\)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class AuthorizedSource:
    """A source the model was allowed to cite."""

    title: str
    domain: str
    url: str


@dataclass(slots=True)
class CitationRecord:
    """One citation match found in the answer text."""

    raw: str
    url: str
    domain: str
    title: str
    format: str
    is_valid: bool


@dataclass(slots=True)
class CitationReport:
    """Validation outcome for an generated answer."""

    citations: list[CitationRecord] = field(default_factory=list)
    invalid_domains: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    has_citations: bool = False

    def to_dict(self) -> dict:
        return {
            "citations": [
                {
                    "raw": c.raw,
                    "url": c.url,
                    "domain": c.domain,
                    "title": c.title,
                    "format": c.format,
                    "is_valid": c.is_valid,
                }
                for c in self.citations
            ],
            "invalid_domains": list(self.invalid_domains),
            "hallucination_flags": list(self.hallucination_flags),
            "has_citations": self.has_citations,
        }


def normalize_domain(domain: str) -> str:
    """Normalize hostnames for comparison."""
    d = domain.strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d


def domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return normalize_domain(parsed.netloc or "")


def build_authorized_sources(state: AgentState) -> list[AuthorizedSource]:
    """Collect authorized title/domain/url tuples from acquired context."""
    seen: set[str] = set()
    sources: list[AuthorizedSource] = []

    def add(url: str, title: str | None, domain: str | None) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        dom = normalize_domain(domain or domain_from_url(url))
        sources.append(
            AuthorizedSource(
                title=(title or "Untitled").strip(),
                domain=dom,
                url=url.strip(),
            )
        )

    for snippet in state.selected_snippets:
        add(snippet.url, snippet.title, snippet.domain)
    for ctx in state.source_contexts:
        add(ctx.url, ctx.title, ctx.domain)

    return sources


def validate_citations(
    answer: str,
    authorized: list[AuthorizedSource],
) -> CitationReport:
    """Scan answer text and flag citations not backed by authorized sources."""
    allowed_domains = {s.domain for s in authorized if s.domain}
    allowed_urls = {s.url for s in authorized}
    records: list[CitationRecord] = []
    invalid_domains: list[str] = []
    hallucination_flags: list[str] = []

    for match in MARKDOWN_CITATION_RE.finditer(answer):
        title, url = match.group(1).strip(), match.group(2).strip()
        domain = domain_from_url(url)
        is_valid = domain in allowed_domains or url in allowed_urls
        records.append(
            CitationRecord(
                raw=match.group(0),
                url=url,
                domain=domain,
                title=title,
                format="markdown",
                is_valid=is_valid,
            )
        )
        if not is_valid and domain and domain not in invalid_domains:
            invalid_domains.append(domain)
            hallucination_flags.append(
                f"Potential hallucination: citation domain '{domain}' not in authorized sources."
            )

    for match in PAREN_CITATION_RE.finditer(answer):
        domain_raw, url = match.group(1).strip(), match.group(2).strip()
        domain = normalize_domain(domain_raw)
        is_valid = domain in allowed_domains or url in allowed_urls
        records.append(
            CitationRecord(
                raw=match.group(0),
                url=url,
                domain=domain,
                title=domain_raw,
                format="paren",
                is_valid=is_valid,
            )
        )
        if not is_valid and domain and domain not in invalid_domains:
            invalid_domains.append(domain)
            hallucination_flags.append(
                f"Potential hallucination: citation domain '{domain}' not in authorized sources."
            )

    report = CitationReport(
        citations=records,
        invalid_domains=invalid_domains,
        hallucination_flags=hallucination_flags,
        has_citations=len(records) > 0,
    )

    if answer.strip() and not report.has_citations and len(answer) > 120:
        report.hallucination_flags.append(
            "Answer may lack required inline citations for factual claims."
        )

    return report
