"""Tests for research planning and JSON parsing."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from deep_research_agent.llm.errors import PlanningError
from deep_research_agent.orchestration.planner import Planner, parse_plan_response


class ParsePlanResponseTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        raw = '{"summary": "Compare GDP", "search_queries": ["India GDP", "China GDP"], "steps": ["search", "compare"]}'
        plan = parse_plan_response(raw)
        self.assertEqual(plan.summary, "Compare GDP")
        self.assertEqual(len(plan.search_queries), 2)

    def test_markdown_wrapped_json(self) -> None:
        raw = """Here is your plan:
```json
{"summary": "RAG research", "search_queries": ["RAG definition"], "steps": ["define"]}
```
"""
        plan = parse_plan_response(raw)
        self.assertEqual(plan.search_queries, ["RAG definition"])

    def test_empty_queries_raises(self) -> None:
        raw = '{"summary": "x", "search_queries": [], "steps": []}'
        with self.assertRaises(PlanningError):
            parse_plan_response(raw)


class PlannerTests(unittest.TestCase):
    def test_plan_calls_client(self) -> None:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = (
            '{"summary": "GDP", "search_queries": ["India GDP 2024"], "steps": ["search"]}'
        )
        planner = Planner(mock_client)
        plan = planner.plan("Compare India and China GDP")
        self.assertEqual(plan.search_queries, ["India GDP 2024"])
        mock_client.generate_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
