"""Tests for the SQLite episodic memory store."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from deep_research_agent.memory.store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_memory.db"
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_database_created_with_wal_mode(self) -> None:
        self.assertTrue(self.db_path.exists())
        conn = sqlite3.connect(self.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode.lower(), "wal")

    def test_session_crud(self) -> None:
        session = self.store.create_session(metadata={"source": "test"})
        session_id = session["session_id"]

        fetched = self.store.get_session(session_id)
        assert fetched is not None
        self.assertEqual(fetched["metadata"], {"source": "test"})

        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertTrue(self.store.delete_session(session_id))
        self.assertIsNone(self.store.get_session(session_id))

    def test_conversation_messages_ordered(self) -> None:
        session = self.store.create_session()
        sid = session["session_id"]

        self.store.add_message(sid, "user", "What is quantum computing?")
        self.store.add_message(sid, "assistant", "Quantum computing uses qubits...")

        messages = self.store.get_messages(sid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

    def test_turn_and_context_audit_trail(self) -> None:
        session = self.store.create_session()
        sid = session["session_id"]

        turn = self.store.create_turn(
            session_id=sid,
            user_query="Compare GDP of India and China",
            search_queries=["India GDP 2024", "China GDP 2024"],
            plan={"steps": ["search", "compare"]},
        )
        turn_id = turn["turn_id"]

        self.store.add_context(
            turn_id=turn_id,
            url="https://example.com/india-gdp",
            title="India GDP",
            domain="example.com",
            text_block="India GDP grew by 7%.",
            metadata={"score": 0.92},
        )
        self.store.add_context(
            turn_id=turn_id,
            url="https://example.com/china-gdp",
            title="China GDP",
            domain="example.com",
            text_block="China GDP grew by 5%.",
        )

        updated = self.store.update_turn(
            turn_id,
            urls_opened=[
                "https://example.com/india-gdp",
                "https://example.com/china-gdp",
            ],
            context_snippets=[
                {"url": "https://example.com/india-gdp", "snippet": "7%"},
                {"url": "https://example.com/china-gdp", "snippet": "5%"},
            ],
            final_answer="India grew faster than China per cited sources.",
        )
        assert updated is not None
        self.assertEqual(len(updated["urls_opened"]), 2)
        self.assertIn("India grew faster", updated["final_answer"])

        history = self.store.reconstruct_turn_history(turn_id)
        assert history is not None
        self.assertEqual(len(history["contexts"]), 2)
        self.assertEqual(history["contexts"][0]["metadata"], {"score": 0.92})

    def test_session_reconstruction(self) -> None:
        session = self.store.create_session()
        sid = session["session_id"]
        self.store.add_message(sid, "user", "Hello")
        turn = self.store.create_turn(sid, user_query="Hello", final_answer="Hi")
        self.store.add_context(turn["turn_id"], url="https://example.com", title="Ex")

        full = self.store.reconstruct_session(sid)
        assert full is not None
        self.assertEqual(len(full["messages"]), 1)
        self.assertEqual(len(full["turns"]), 1)
        self.assertEqual(len(full["turns"][0]["contexts"]), 1)

    def test_row_factory_returns_dict_like_rows(self) -> None:
        session = self.store.create_session()
        msg = self.store.add_message(session["session_id"], "user", "test")
        self.assertIn("message_id", msg)
        self.assertIn("content", msg)
        self.assertEqual(msg["content"], "test")

    def test_foreign_key_cascade_on_session_delete(self) -> None:
        session = self.store.create_session()
        sid = session["session_id"]
        turn = self.store.create_turn(sid, user_query="q")
        self.store.add_context(turn["turn_id"], url="https://x.com")
        self.store.add_message(sid, "user", "m")

        self.store.delete_session(sid)

        conn = sqlite3.connect(self.db_path)
        turn_count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        ctx_count = conn.execute("SELECT COUNT(*) FROM contexts").fetchone()[0]
        conn.close()

        self.assertEqual(turn_count, 0)
        self.assertEqual(msg_count, 0)
        self.assertEqual(ctx_count, 0)

    def test_json_fields_round_trip(self) -> None:
        session = self.store.create_session(metadata={"tags": ["a", "b"]})
        sid = session["session_id"]
        turn = self.store.create_turn(
            sid,
            user_query="test",
            search_queries=["q1", "q2"],
            context_snippets=[{"id": 1, "text": "snippet"}],
        )
        self.assertEqual(turn["search_queries"], ["q1", "q2"])
        self.assertEqual(turn["context_snippets"][0]["text"], "snippet")


if __name__ == "__main__":
    unittest.main()
