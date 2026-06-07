"""Tests for JSON extraction from LLM responses."""

from __future__ import annotations

import unittest

from deep_research_agent.llm.json_extract import extract_json_object, repair_json


class JsonExtractTests(unittest.TestCase):
    def test_direct_json(self) -> None:
        raw = '{"summary": "s", "search_queries": ["a"], "steps": ["b"]}'
        data = extract_json_object(raw)
        self.assertEqual(data["summary"], "s")
        self.assertEqual(data["search_queries"], ["a"])

    def test_markdown_fence(self) -> None:
        raw = """Here is the plan:
```json
{"summary": "Compare GDP", "search_queries": ["India GDP", "China GDP"], "steps": ["search"]}
```
"""
        data = extract_json_object(raw)
        self.assertEqual(len(data["search_queries"]), 2)

    def test_braced_object_with_preamble(self) -> None:
        raw = 'Sure! {"summary": "x", "search_queries": ["q1"], "steps": ["s1"]} Hope that helps.'
        data = extract_json_object(raw)
        self.assertEqual(data["search_queries"], ["q1"])

    def test_repair_trailing_comma(self) -> None:
        broken = '{"summary": "x", "search_queries": ["q"], "steps": [],}'
        repaired = repair_json(broken)
        data = extract_json_object(repaired)
        self.assertEqual(data["steps"], [])

    def test_raises_when_no_json(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("No structured data here.")


if __name__ == "__main__":
    unittest.main()
