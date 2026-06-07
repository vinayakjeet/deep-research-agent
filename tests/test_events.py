"""Tests for streaming status events."""

from __future__ import annotations

import unittest

from deep_research_agent.orchestration.events import (
    no_evidence_event,
    phase_complete,
    phase_start,
    terminal_event,
)
from deep_research_agent.state.enums import AgentPhase


class EventsTests(unittest.TestCase):
    def test_phase_start_shape(self) -> None:
        event = phase_start(AgentPhase.PLANNING)
        self.assertEqual(event["event"], "phase_start")
        self.assertEqual(event["phase"], "planning")
        self.assertIn("status", event)
        self.assertIn("message", event)

    def test_terminal_includes_final_answer(self) -> None:
        event = terminal_event(
            final_answer="Done.",
            session_id="abc",
            turn_id=1,
            phases_executed=["planning"],
        )
        self.assertEqual(event["event"], "complete")
        self.assertEqual(event["final_answer"], "Done.")
        self.assertEqual(event["details"]["session_id"], "abc")

    def test_no_evidence_event(self) -> None:
        event = no_evidence_event(reason="empty", user_query="q")
        self.assertEqual(event["status"], "NoEvidence")
        self.assertEqual(event["details"]["reason"], "empty")


if __name__ == "__main__":
    unittest.main()
