"""Tests for evaluation harness modules."""

from __future__ import annotations

import unittest

from deep_research_agent.eval.behavior import score_behavior
from deep_research_agent.eval.metrics import score_retrieval
from deep_research_agent.eval.streaming import score_streaming


class TestEvalMetrics(unittest.TestCase):
    def test_retrieval_f1_perfect_match(self) -> None:
        score = score_retrieval(
            ["https://en.wikipedia.org/wiki/Thimphu"],
            ["https://en.wikipedia.org/wiki/Thimphu"],
            ["wikipedia.org"],
        )
        self.assertEqual(score.f1, 1.0)
        self.assertEqual(score.recall, 1.0)

    def test_retrieval_skips_empty_ground_truth(self) -> None:
        score = score_retrieval(["https://example.com"], [], [])
        self.assertTrue(score.skipped)

    def test_streaming_phase_alias(self) -> None:
        events = [
            {"event": "phase_start", "phase": "planning"},
            {"event": "phase_start", "phase": "searching"},
            {"event": "phase_start", "phase": "acquiring"},
            {"event": "phase_start", "phase": "answering"},
        ]
        score = score_streaming(
            events,
            ["planning", "searching", "fetching", "generating"],
        )
        self.assertTrue(score.passed)
        self.assertEqual(score.missing, [])

    def test_refuse_insufficient_evidence(self) -> None:
        score = score_behavior(
            answer="I could not locate verifiable evidence on the open web.",
            expected_behavior="refuse_insufficient_evidence",
            expected_facts=["not publicly available"],
            ended_no_evidence=True,
        )
        self.assertTrue(score.passed)

    def test_must_not_know_isolation(self) -> None:
        score = score_behavior(
            answer="I do not have information about where you work.",
            expected_behavior="must_not_know",
            expected_facts=["unknown"],
            forbidden_terms=["NCBS", "Bangalore"],
        )
        self.assertTrue(score.passed)


if __name__ == "__main__":
    unittest.main()
