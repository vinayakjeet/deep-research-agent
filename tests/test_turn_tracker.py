"""Tests for the comprehensive turn history tracker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deep_research_agent.memory import MemoryStore, TurnHistoryTracker
from deep_research_agent.state.schema import ContextSnippet


class TurnHistoryTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "tracker.db")
        self.session = self.store.create_session()
        self.session_id = self.session["session_id"]

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_full_turn_lifecycle_audit_trail(self) -> None:
        tracker = TurnHistoryTracker(self.store, self.session_id)

        turn_id = tracker.begin("Compare GDP of India and China")
        self.assertIsNotNone(tracker.root_transaction_id)

        plan_tx = tracker.record_plan(
            {"summary": "Compare recent GDP growth", "steps": ["search", "compare"]},
            search_queries=["India GDP 2024", "China GDP 2024"],
        )
        india_tx = tracker.record_url_extraction(
            url="https://example.com/india-gdp",
            title="India GDP",
            domain="example.com",
            text_block="India GDP grew by 7%.",
            metadata={"score": 0.92},
        )
        china_tx = tracker.record_url_extraction(
            url="https://example.com/china-gdp",
            title="China GDP",
            domain="example.com",
            text_block="China GDP grew by 5%.",
        )
        snippet_tx = tracker.record_snippets(
            [
                ContextSnippet(
                    url="https://example.com/india-gdp",
                    snippet="7%",
                    title="India GDP",
                ),
                {"url": "https://example.com/china-gdp", "snippet": "5%"},
            ]
        )
        answer_tx = tracker.finalize_answer(
            "India grew faster than China per cited sources."
        )

        tx_ids = {plan_tx, india_tx, china_tx, snippet_tx, answer_tx}
        self.assertEqual(len(tx_ids), 5)

        audit = tracker.to_audit_dict()
        self.assertEqual(audit["turn_id"], turn_id)
        self.assertEqual(audit["user_query"], "Compare GDP of India and China")
        self.assertEqual(audit["search"]["queries"], ["India GDP 2024", "China GDP 2024"])
        self.assertEqual(len(audit["search"]["urls_accessed"]), 2)
        self.assertEqual(len(audit["extraction"]["contexts"]), 2)
        self.assertEqual(len(audit["extraction"]["snippets"]), 2)
        self.assertIn("India grew faster", audit["response"]["final_answer"])
        self.assertEqual(len(audit["events"]), 6)

        for event in audit["events"]:
            self.assertIn("transaction_id", event)
            self.assertIn("event_type", event)
            self.assertIn("timestamp", event)

        self.assertTrue(tracker.is_closed)
        with self.assertRaises(RuntimeError):
            tracker.record_search_queries(["late query"])

    def test_reconstruct_turn_from_sqlite(self) -> None:
        tracker = TurnHistoryTracker(self.store, self.session_id)
        turn_id = tracker.begin("What is RAG?")
        tracker.record_search_queries(["retrieval augmented generation"])
        tracker.record_url_extraction(
            url="https://example.com/rag",
            text_block="RAG combines retrieval with generation.",
            title="RAG Overview",
        )
        tracker.finalize_answer("RAG augments LLMs with retrieved documents.")

        reconstructed = tracker.reconstruct()
        assert reconstructed is not None
        self.assertEqual(reconstructed.turn_id, turn_id)
        self.assertEqual(reconstructed.user_query, "What is RAG?")
        self.assertEqual(reconstructed.search_queries, ["retrieval augmented generation"])
        self.assertEqual(len(reconstructed.contexts), 1)
        self.assertEqual(reconstructed.contexts[0].text_block, "RAG combines retrieval with generation.")
        self.assertIn("RAG augments", reconstructed.final_answer or "")
        assert reconstructed.audit_trail is not None
        self.assertEqual(reconstructed.audit_trail["turn_id"], turn_id)

    def test_hydrate_tracker_from_persisted_turn(self) -> None:
        tracker = TurnHistoryTracker(self.store, self.session_id)
        turn_id = tracker.begin("Hydration test")
        tracker.record_url_extraction(
            url="https://example.com/a",
            text_block="Alpha content.",
        )
        tracker.finalize_answer("Done.")

        loaded = TurnHistoryTracker.from_persisted_turn(self.store, turn_id)
        assert loaded is not None
        self.assertEqual(loaded.turn_id, turn_id)
        self.assertTrue(loaded.is_closed)
        audit = loaded.to_audit_dict()
        self.assertEqual(audit["user_query"], "Hydration test")
        self.assertEqual(len(audit["extraction"]["contexts"]), 1)

    def test_contexts_table_stores_extraction_metadata(self) -> None:
        tracker = TurnHistoryTracker(self.store, self.session_id)
        turn_id = tracker.begin("Metadata test")
        tx = tracker.record_url_extraction(
            url="https://example.com/meta",
            text_block="Stored in contexts table.",
        )

        contexts = self.store.get_contexts_for_turn(turn_id)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["url"], "https://example.com/meta")
        self.assertEqual(contexts[0]["text_block"], "Stored in contexts table.")
        self.assertEqual(contexts[0]["metadata"]["transaction_id"], tx)

    def test_store_reconstruct_session_includes_audit_trail(self) -> None:
        tracker = TurnHistoryTracker(self.store, self.session_id)
        tracker.begin("Session rebuild")
        tracker.finalize_answer("Answer.")

        session = self.store.reconstruct_session(self.session_id)
        assert session is not None
        self.assertEqual(len(session["turns"]), 1)
        self.assertIsNotNone(session["turns"][0]["audit_trail"])
        self.assertEqual(session["turns"][0]["audit_trail"]["user_query"], "Session rebuild")


if __name__ == "__main__":
    unittest.main()
