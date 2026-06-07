"""Tests for HTML DOM text extraction."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import patch

from deep_research_agent.ingestion.parse import (
    HtmlExtractor,
    HtmlExtractorConfig,
    domain_from_url,
    to_source_context,
)

FIXTURES = Path(__file__).parent / "fixtures"


class HtmlExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = HtmlExtractor()

    def test_strips_boilerplate_tags(self) -> None:
        html = (FIXTURES / "sample_boilerplate.html").read_text(encoding="utf-8")
        text = self.extractor.extract(html)

        self.assertIn("Retrieval augmented generation", text)
        self.assertNotIn("alert(", text)
        self.assertNotIn("Copyright 2024", text)
        self.assertNotIn("sidebar widgets", text)
        self.assertNotIn("Home", text)

    def test_normalizes_whitespace(self) -> None:
        html = "<html><body><p>Line   one</p>\n\n<p>Line two</p></body></html>"
        text = self.extractor.extract(html)
        self.assertNotIn("   ", text)
        self.assertIn("Line one", text)

    def test_readability_fallback_when_selectolax_sparse(self) -> None:
        html = (FIXTURES / "sample_sparse.html").read_text(encoding="utf-8")
        long_article = (
            "<html><body><article>"
            + ("<p>Quantum computing uses qubits for parallel computation.</p>" * 5)
            + "</article></body></html>"
        )
        extractor = HtmlExtractor(
            HtmlExtractorConfig(min_text_length=50, use_readability_fallback=True)
        )
        with patch.object(extractor, "_extract_with_selectolax", return_value="short"):
            with patch.object(
                extractor,
                "_extract_with_readability",
                return_value="Quantum computing uses qubits for parallel computation. " * 3,
            ):
                text = extractor.extract(long_article)
        self.assertIn("Quantum computing", text)
        self.assertGreater(len(text), 50)

    def test_parse_medium_fixture_quickly(self) -> None:
        html = (FIXTURES / "sample_boilerplate.html").read_text(encoding="utf-8") * 50
        start = time.perf_counter()
        for _ in range(20):
            self.extractor.extract(html)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0)

    def test_domain_from_url(self) -> None:
        self.assertEqual(domain_from_url("https://www.example.com/path"), "www.example.com")

    def test_to_source_context(self) -> None:
        ctx = to_source_context(
            1,
            "https://example.com/a",
            "body text",
            title="Example",
        )
        self.assertEqual(ctx.turn_id, 1)
        self.assertEqual(ctx.domain, "example.com")
        self.assertEqual(ctx.text_block, "body text")


if __name__ == "__main__":
    unittest.main()
