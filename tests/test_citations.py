"""Tests for citation validation."""

from __future__ import annotations

import unittest

from deep_research_agent.orchestration.citations import (
    AuthorizedSource,
    build_authorized_sources,
    validate_citations,
)
from deep_research_agent.state.schema import AgentState, ContextSnippet, SourceContext


class CitationValidationTests(unittest.TestCase):
    def test_valid_markdown_citation(self) -> None:
        authorized = [
            AuthorizedSource(
                title="India GDP",
                domain="example.com",
                url="https://example.com/india",
            )
        ]
        answer = "Growth was 7% [India GDP — example.com](https://example.com/india)."
        report = validate_citations(answer, authorized)
        self.assertTrue(report.has_citations)
        self.assertEqual(len(report.citations), 1)
        self.assertTrue(report.citations[0].is_valid)
        self.assertEqual(report.invalid_domains, [])

    def test_invalid_domain_flagged(self) -> None:
        authorized = [
            AuthorizedSource(
                title="India GDP",
                domain="example.com",
                url="https://example.com/india",
            )
        ]
        answer = "Data from [Fake — evil.com](https://evil.com/report)."
        report = validate_citations(answer, authorized)
        self.assertFalse(report.citations[0].is_valid)
        self.assertIn("evil.com", report.invalid_domains)
        self.assertTrue(report.hallucination_flags)

    def test_paren_format_validated(self) -> None:
        authorized = [
            AuthorizedSource(
                title="Report",
                domain="data.gov",
                url="https://data.gov/stats",
            )
        ]
        answer = "See (data.gov, https://data.gov/stats) for details."
        report = validate_citations(answer, authorized)
        self.assertTrue(any(c.format == "paren" and c.is_valid for c in report.citations))

    def test_long_answer_without_citations_warns(self) -> None:
        answer = "x" * 150
        report = validate_citations(answer, [])
        self.assertTrue(any("lack required inline citations" in f for f in report.hallucination_flags))

    def test_build_authorized_sources_from_state(self) -> None:
        state = AgentState(session_id="s1")
        state.selected_snippets = [
            ContextSnippet(
                url="https://example.com/a",
                snippet="text",
                title="A",
                domain="example.com",
            )
        ]
        state.source_contexts = [
            SourceContext(
                context_id=1,
                turn_id=1,
                url="https://example.com/b",
                title="B",
                domain="example.com",
                text_block="block",
            )
        ]
        sources = build_authorized_sources(state)
        urls = {s.url for s in sources}
        self.assertEqual(urls, {"https://example.com/a", "https://example.com/b"})


if __name__ == "__main__":
    unittest.main()
