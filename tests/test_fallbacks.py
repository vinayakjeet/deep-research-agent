"""Tests for no-evidence fallback responses."""

from __future__ import annotations

import unittest

from deep_research_agent.orchestration.fallbacks import (
    NO_EVIDENCE_MESSAGE,
    build_no_evidence_response,
)


class FallbacksTests(unittest.TestCase):
    def test_no_evidence_message_is_static(self) -> None:
        self.assertIn("verifiable evidence", NO_EVIDENCE_MESSAGE.lower())

    def test_build_includes_query_and_reason(self) -> None:
        text = build_no_evidence_response(
            "What is quantum foam?",
            reason="No search results were returned.",
        )
        self.assertIn(NO_EVIDENCE_MESSAGE, text)
        self.assertIn("What is quantum foam?", text)
        self.assertIn("No search results were returned.", text)


if __name__ == "__main__":
    unittest.main()
