"""Tests for contextual chunking and TF selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deep_research_agent.ingestion.select import (
    select_context_for_query,
    select_relevant_blocks,
    split_paragraphs,
)
from deep_research_agent.memory import MemoryStore, TurnHistoryTracker


def _long_document() -> str:
    paragraphs = [
        "The weather in Paris remains mild during spring with occasional rain.",
        "India GDP growth reached seven percent in the latest fiscal year reports.",
        "China manufacturing output expanded while services slowed moderately.",
        "India economic survey highlights infrastructure and digital public goods.",
        "Stock markets reacted to central bank guidance on inflation targets.",
        "Tourism revenue in Europe recovered post-pandemic with regional variation.",
        "India GDP comparisons often reference purchasing power parity adjustments.",
        "Renewable energy investments accelerated across emerging economies.",
    ]
    return "\n\n".join(paragraphs)


class ChunkerTests(unittest.TestCase):
    def test_split_paragraphs(self) -> None:
        text = "First paragraph about alpha.\n\nSecond paragraph about beta."
        blocks = split_paragraphs(text, min_block_chars=10)
        self.assertEqual(len(blocks), 2)

    def test_tf_selects_query_relevant_blocks(self) -> None:
        text = _long_document()
        blocks = select_relevant_blocks(text, "India GDP growth", top_k=3)

        self.assertGreater(len(blocks), 0)
        self.assertLessEqual(len(blocks), 3)
        combined = " ".join(blocks).lower()
        self.assertIn("india", combined)
        self.assertIn("gdp", combined)

    def test_snippets_shorter_than_full_document(self) -> None:
        text = _long_document()
        snippets = select_context_for_query(
            text,
            "India GDP",
            "https://example.com/india-gdp",
            title="India GDP",
            domain="example.com",
        )
        snippet_len = sum(len(s.snippet) for s in snippets)
        self.assertLess(snippet_len, len(text))
        self.assertGreater(len(snippets), 0)


class ChunkSelectorTrackerTests(unittest.TestCase):
    def test_record_snippets_from_selection(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            store = MemoryStore(Path(tmp.name) / "sel.db")
            session = store.create_session()
            tracker = TurnHistoryTracker(store, session["session_id"])
            tracker.begin("India vs China GDP")

            snippets = select_context_for_query(
                _long_document(),
                "India GDP growth",
                "https://example.com/india",
            )
            tracker.record_snippets(snippets)
            tracker.finalize_answer("India GDP grew faster per selected sources.")

            record = tracker.reconstruct()
            assert record is not None
            self.assertGreater(len(record.context_snippets), 0)
            self.assertTrue(
                any("india" in s.snippet.lower() for s in record.context_snippets)
            )
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
