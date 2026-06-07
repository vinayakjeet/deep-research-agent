"""Tests for bounded source assembly and answer prompts."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from deep_research_agent.orchestration.context_assembly import (
    AnswerGenerator,
    assemble_sources,
    build_answer_messages,
)
from deep_research_agent.state.enums import AgentPhase
from deep_research_agent.state.schema import AgentState, ContextSnippet, SourceContext


class ContextAssemblyTests(unittest.TestCase):
    def test_assemble_sources_uses_snippets(self) -> None:
        snippets = [
            ContextSnippet(
                url="https://example.com/a",
                snippet="India GDP grew by 7 percent.",
                title="India GDP",
                domain="example.com",
            )
        ]
        block = assemble_sources(snippets)
        self.assertIn("<source_1>", block)
        self.assertIn("</source_1>", block)
        self.assertIn("India GDP grew", block)
        self.assertNotIn("<source_2>", block)

    def test_fallback_to_source_context(self) -> None:
        contexts = [
            SourceContext(
                context_id=1,
                turn_id=1,
                url="https://example.com/b",
                title="China GDP",
                domain="example.com",
                text_block="China GDP grew by 5 percent in the latest report.",
            )
        ]
        block = assemble_sources([], source_contexts=contexts)
        self.assertIn("China GDP grew", block)

    def test_build_answer_messages_includes_sources(self) -> None:
        msgs = build_answer_messages(
            "What is India GDP?",
            "<source_1>\nContent:\n7%\n</source_1>",
            conversation_summary="Prior: user asked about GDP",
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("<source_1>", msgs[1]["content"])


class AnswerGeneratorTests(unittest.TestCase):
    def test_generate_requires_sources(self) -> None:
        gen = AnswerGenerator(MagicMock())
        state = AgentState(session_id="s1", phase=AgentPhase.ANSWERING)
        with self.assertRaises(ValueError):
            gen.generate("query", state)

    def test_generate_calls_gemini(self) -> None:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "India grew faster [India GDP — example.com]"
        gen = AnswerGenerator(mock_client)
        state = AgentState(session_id="s1", phase=AgentPhase.ANSWERING)
        state.selected_snippets = [
            ContextSnippet(
                url="https://example.com/a",
                snippet="India GDP 7%",
                title="India GDP",
                domain="example.com",
            )
        ]
        answer = gen.generate("Compare GDP", state)
        self.assertIn("India", answer)
        mock_client.generate_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
