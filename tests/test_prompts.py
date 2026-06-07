"""Tests for LLM system prompt content."""

from __future__ import annotations

import unittest

from deep_research_agent.llm.prompts import ANSWER_SYSTEM_PROMPT


class AnswerPromptTests(unittest.TestCase):
    def test_mandates_markdown_citation_format(self) -> None:
        self.assertIn("[Title — domain](full_url)", ANSWER_SYSTEM_PROMPT)

    def test_mandates_conflict_reporting(self) -> None:
        lower = ANSWER_SYSTEM_PROMPT.lower()
        self.assertIn("disagree", lower)
        self.assertIn("however", lower)
        self.assertIn("never average", lower)


if __name__ == "__main__":
    unittest.main()
