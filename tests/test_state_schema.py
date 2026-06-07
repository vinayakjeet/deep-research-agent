"""Tests for typed state schema and serialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.state.adapters import load_agent_state, load_turn_record
from deep_research_agent.state.enums import AgentPhase, MessageRole
from deep_research_agent.state.schema import (
    AgentState,
    ContextSnippet,
    MessageRecord,
    ResearchPlan,
    SearchResult,
    SessionRecord,
    SourceContext,
    TurnRecord,
)
from deep_research_agent.state.serialization import (
    agent_state_from_json,
    agent_state_to_json,
    from_json,
    to_dict,
    to_json,
)


class EnumTests(unittest.TestCase):
    def test_agent_phase_terminal_states(self) -> None:
        self.assertFalse(AgentPhase.PLANNING.is_terminal())
        self.assertTrue(AgentPhase.COMPLETE.is_terminal())
        self.assertTrue(AgentPhase.ERROR.is_terminal())

    def test_message_role_validation(self) -> None:
        self.assertEqual(MessageRole.from_str("user"), MessageRole.USER)
        with self.assertRaises(ValueError):
            MessageRole.from_str("invalid")


class SerializationTests(unittest.TestCase):
    def test_research_plan_round_trip(self) -> None:
        plan = ResearchPlan(
            summary="Compare GDP figures",
            search_queries=["India GDP", "China GDP"],
            steps=["search", "compare", "answer"],
        )
        restored = from_json(ResearchPlan, to_json(plan))
        self.assertEqual(restored.summary, plan.summary)
        self.assertEqual(restored.search_queries, plan.search_queries)

    def test_search_result_round_trip(self) -> None:
        result = SearchResult(title="T", url="https://x.com", snippet="text", score=0.8)
        restored = from_json(SearchResult, to_json(result))
        self.assertEqual(restored.score, 0.8)

    def test_agent_state_json_round_trip(self) -> None:
        state = AgentState(
            session_id="abc",
            phase=AgentPhase.SEARCHING,
            current_query="What is RAG?",
            plan=ResearchPlan(summary="Research RAG", search_queries=["RAG definition"]),
            search_results=[
                SearchResult(title="RAG", url="https://example.com", snippet="...")
            ],
            selected_snippets=[
                ContextSnippet(url="https://example.com", snippet="RAG is...", domain="example.com")
            ],
            messages=[
                MessageRecord(
                    message_id=1,
                    session_id="abc",
                    role=MessageRole.USER,
                    content="What is RAG?",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ],
        )
        payload = agent_state_to_json(state)
        restored = agent_state_from_json(payload)
        self.assertEqual(restored.phase, AgentPhase.SEARCHING)
        self.assertEqual(restored.plan.search_queries, ["RAG definition"])
        self.assertEqual(restored.search_results[0].url, "https://example.com")
        self.assertEqual(restored.messages[0].role, MessageRole.USER)

    def test_prompt_payload_is_json_serializable(self) -> None:
        state = AgentState(
            session_id="abc",
            phase=AgentPhase.ANSWERING,
            current_query="test",
            plan=ResearchPlan(summary="s", search_queries=["q"]),
        )
        payload = state.to_prompt_payload()
        encoded = json.dumps(payload)
        self.assertIn("conversation", encoded)


class StoreRowConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "state_test.db")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_turn_record_from_store_without_data_loss(self) -> None:
        session = self.store.create_session(metadata={"client": "test"})
        sid = session["session_id"]
        turn = self.store.create_turn(
            sid,
            user_query="Compare GDP",
            search_queries=["India GDP", "China GDP"],
            plan={"summary": "Compare", "search_queries": ["India GDP"], "steps": ["search"]},
        )
        self.store.add_context(
            turn_id=turn["turn_id"],
            url="https://example.com/india",
            title="India GDP",
            domain="example.com",
            text_block="Growth 7%",
            metadata={"score": 0.9},
        )
        self.store.update_turn(
            turn["turn_id"],
            urls_opened=["https://example.com/india"],
            context_snippets=[{"url": "https://example.com/india", "snippet": "7%"}],
            final_answer="India grew 7%",
        )

        record = load_turn_record(self.store, turn["turn_id"])
        assert record is not None
        self.assertEqual(record.user_query, "Compare GDP")
        self.assertEqual(record.search_queries, ["India GDP", "China GDP"])
        assert record.plan is not None
        self.assertEqual(record.plan.summary, "Compare")
        self.assertEqual(len(record.contexts), 1)
        self.assertEqual(record.contexts[0].metadata, {"score": 0.9})
        self.assertEqual(record.context_snippets[0].snippet, "7%")

    def test_agent_state_from_reconstructed_session(self) -> None:
        session = self.store.create_session()
        sid = session["session_id"]
        self.store.add_message(sid, "user", "Hello")
        turn = self.store.create_turn(sid, user_query="Hello", final_answer="Hi there")
        self.store.add_message(sid, "assistant", "Hi there")

        state = load_agent_state(self.store, sid)
        assert state is not None
        self.assertEqual(state.session_id, sid)
        self.assertEqual(state.phase, AgentPhase.COMPLETE)
        self.assertEqual(state.current_query, "Hello")
        self.assertEqual(state.final_answer, "Hi there")
        self.assertEqual(state.turn_id, turn["turn_id"])
        self.assertEqual(len(state.messages), 2)
        self.assertEqual(state.messages[0].role, MessageRole.USER)
        self.assertEqual(state.messages[1].role, MessageRole.ASSISTANT)

    def test_session_record_from_store_row(self) -> None:
        row = self.store.create_session(metadata={"k": "v"})
        record = SessionRecord.from_store_row(row)
        self.assertEqual(record.metadata, {"k": "v"})


class AgentStateWorkflowTests(unittest.TestCase):
    def test_phase_transitions(self) -> None:
        state = AgentState(session_id="s1")
        self.assertEqual(state.phase, AgentPhase.START)

        state.advance_to(AgentPhase.PLANNING)
        state.advance_to(AgentPhase.SEARCHING)
        state.advance_to(AgentPhase.ACQUIRING)
        state.advance_to(AgentPhase.ANSWERING)
        state.advance_to(AgentPhase.COMPLETE)

        self.assertTrue(state.is_terminal())

    def test_from_session_factory(self) -> None:
        session = SessionRecord(
            session_id="s1",
            created_at="t1",
            updated_at="t1",
        )
        state = AgentState.from_session(session, phase=AgentPhase.PLANNING)
        self.assertEqual(state.session_id, "s1")
        self.assertEqual(state.phase, AgentPhase.PLANNING)


if __name__ == "__main__":
    unittest.main()
