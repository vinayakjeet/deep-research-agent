"""Tests for context window distillation."""

from __future__ import annotations

import unittest

from deep_research_agent.context import (
    BuiltContext,
    ContextBuilder,
    ContextBuilderConfig,
    estimate_tokens,
)
from deep_research_agent.state.enums import MessageRole
from deep_research_agent.state.schema import (
    AgentState,
    ContextSnippet,
    MessageRecord,
    SourceContext,
    TurnRecord,
)


def _msg(index: int, role: MessageRole, content: str) -> MessageRecord:
    return MessageRecord(
        message_id=index,
        session_id="s1",
        role=role,
        content=content,
        created_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def _long_text(repeats: int) -> str:
    return ("Long research context sentence. " * repeats).strip()


class TokenEstimateTests(unittest.TestCase):
    def test_estimate_tokens_empty(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_estimate_tokens_heuristic(self) -> None:
        text = "a" * 40
        self.assertEqual(estimate_tokens(text, chars_per_token=4.0), 10)


class ContextBuilderTests(unittest.TestCase):
    def test_small_history_no_summarization(self) -> None:
        builder = ContextBuilder(
            ContextBuilderConfig(max_tokens=4_000, reserved_tokens_for_web=1_000)
        )
        messages = [
            _msg(1, MessageRole.USER, "Hello"),
            _msg(2, MessageRole.ASSISTANT, "Hi there"),
        ]
        built = builder.build(
            current_query="What is RAG?",
            messages=messages,
            source_contexts=[],
            selected_snippets=[],
        )
        self.assertFalse(built.was_summarized)
        self.assertIsNone(built.conversation_summary)
        self.assertEqual(len(built.messages), 2)
        self.assertLessEqual(built.token_count, 4_000)

    def test_overflow_triggers_summary_and_keeps_recent(self) -> None:
        captured: list[list[MessageRecord]] = []

        def fake_summarizer(dropped: list[MessageRecord]) -> str:
            captured.append(list(dropped))
            return "Earlier topics: GDP and inflation."

        builder = ContextBuilder(
            ContextBuilderConfig(
                max_tokens=500,
                reserved_tokens_for_web=200,
                min_recent_messages=2,
                chars_per_token=4.0,
            ),
            summarizer=fake_summarizer,
        )

        messages = [
            _msg(i, MessageRole.USER if i % 2 else MessageRole.ASSISTANT, _long_text(20))
            for i in range(1, 11)
        ]
        built = builder.build(
            current_query="Follow-up question",
            messages=messages,
            source_contexts=[],
            selected_snippets=[],
        )

        self.assertTrue(built.was_summarized)
        self.assertEqual(built.conversation_summary, "Earlier topics: GDP and inflation.")
        self.assertGreater(built.dropped_message_count, 0)
        self.assertGreaterEqual(len(built.messages), 2)
        self.assertEqual(built.messages[-1]["content"], messages[-1].content)
        self.assertEqual(len(captured), 1)
        self.assertGreater(len(captured[0]), 0)

    def test_web_context_always_preserved(self) -> None:
        builder = ContextBuilder(
            ContextBuilderConfig(
                max_tokens=400,
                reserved_tokens_for_web=250,
                min_recent_messages=2,
            )
        )
        source = SourceContext(
            context_id=1,
            turn_id=1,
            url="https://example.com/article",
            title="Article",
            domain="example.com",
            text_block=_long_text(80),
        )
        snippet = ContextSnippet(
            url="https://example.com/article",
            snippet=_long_text(40),
            title="Article",
            domain="example.com",
        )
        messages = [
            _msg(i, MessageRole.USER if i % 2 else MessageRole.ASSISTANT, _long_text(30))
            for i in range(1, 9)
        ]

        built = builder.build(
            current_query="Summarize sources",
            messages=messages,
            source_contexts=[source],
            selected_snippets=[snippet],
        )

        self.assertEqual(len(built.source_contexts), 1)
        self.assertEqual(built.source_contexts[0]["text_block"], source.text_block)
        self.assertEqual(len(built.selected_snippets), 1)
        self.assertEqual(built.selected_snippets[0]["snippet"], snippet.snippet)

    def test_build_from_agent_state(self) -> None:
        state = AgentState(
            session_id="s1",
            current_query="What is RAG?",
            messages=[_msg(1, MessageRole.USER, "What is RAG?")],
            source_contexts=[
                SourceContext(
                    context_id=1,
                    turn_id=1,
                    url="https://example.com",
                    text_block="RAG combines retrieval with generation.",
                )
            ],
            selected_snippets=[
                ContextSnippet(
                    url="https://example.com",
                    snippet="RAG combines retrieval with generation.",
                )
            ],
        )
        built = ContextBuilder().build_from_state(state)
        self.assertEqual(built.current_query, "What is RAG?")
        self.assertEqual(len(built.source_contexts), 1)

    def test_prior_turn_summaries_included(self) -> None:
        builder = ContextBuilder()
        turns = [
            TurnRecord(
                turn_id=1,
                session_id="s1",
                user_query="First question",
                created_at="t1",
                final_answer="First answer",
            ),
            TurnRecord(
                turn_id=2,
                session_id="s1",
                user_query="Second question",
                created_at="t2",
                final_answer="Second answer",
            ),
        ]
        built = builder.build(
            current_query="Third question",
            messages=[],
            source_contexts=[],
            selected_snippets=[],
            prior_turns=turns,
        )
        self.assertEqual(len(built.prior_turn_summaries), 2)
        self.assertIn("First question", built.prior_turn_summaries[0])

    def test_prompt_dict_includes_summary_as_system_message(self) -> None:
        builder = ContextBuilder(
            ContextBuilderConfig(max_tokens=300, reserved_tokens_for_web=100),
            summarizer=lambda dropped: "Compressed history",
        )
        messages = [
            _msg(1, MessageRole.USER, _long_text(25)),
            _msg(2, MessageRole.ASSISTANT, _long_text(25)),
            _msg(3, MessageRole.USER, "Recent question"),
            _msg(4, MessageRole.ASSISTANT, "Recent answer"),
        ]
        built = builder.build(
            current_query="Recent question",
            messages=messages,
            source_contexts=[],
            selected_snippets=[],
        )
        payload = built.to_prompt_dict()
        self.assertTrue(built.was_summarized or built.conversation_summary)
        if built.conversation_summary:
            self.assertEqual(payload["conversation"][0]["role"], "system")
            self.assertEqual(payload["conversation"][0]["content"], built.conversation_summary)

    def test_respects_max_token_budget(self) -> None:
        builder = ContextBuilder(
            ContextBuilderConfig(
                max_tokens=600,
                reserved_tokens_for_web=200,
                prompt_overhead_tokens=50,
            )
        )
        built = builder.build(
            current_query="Q",
            messages=[
                _msg(i, MessageRole.USER, _long_text(15))
                for i in range(1, 8)
            ],
            source_contexts=[
                SourceContext(
                    context_id=1,
                    turn_id=1,
                    url="https://example.com",
                    text_block=_long_text(10),
                )
            ],
            selected_snippets=[],
        )
        self.assertLessEqual(built.token_count, 600)


if __name__ == "__main__":
    unittest.main()
