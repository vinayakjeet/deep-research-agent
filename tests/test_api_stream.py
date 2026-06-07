"""Tests for FastAPI SSE research streaming."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from deep_research_agent.api.app import create_app
from deep_research_agent.memory import MemoryStore


class ApiStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "api.db"
        self.app = create_app(db_path=self.db_path)
        self._client_ctx = TestClient(self.app)
        self.client = self._client_ctx.__enter__()

    def tearDown(self) -> None:
        self._client_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_create_session(self) -> None:
        resp = self.client.post("/sessions")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("session_id", resp.json())

    def test_research_stream_sse(self) -> None:
        session_id = self.client.post("/sessions").json()["session_id"]

        def fake_stream(sid: str, query: str):
            yield {"event": "phase_start", "phase": "planning", "status": "Planning", "message": "Planning...", "details": {}}
            yield {
                "event": "complete",
                "phase": "complete",
                "status": "Complete",
                "message": "done",
                "details": {"session_id": sid, "turn_id": 1, "phases_executed": ["planning"]},
                "final_answer": "Test answer.",
            }

        with patch(
            "deep_research_agent.api.app.ResearchOrchestrator.run_stream",
            side_effect=fake_stream,
        ):
            resp = self.client.post(
                "/research/stream",
                json={"session_id": session_id, "query": "test"},
            )

        self.assertEqual(resp.status_code, 200)
        lines = [ln for ln in resp.text.strip().split("\n") if ln.startswith("data: ")]
        self.assertGreaterEqual(len(lines), 2)
        last = json.loads(lines[-1].removeprefix("data: "))
        self.assertEqual(last["event"], "complete")
        self.assertEqual(last["final_answer"], "Test answer.")


if __name__ == "__main__":
    unittest.main()
