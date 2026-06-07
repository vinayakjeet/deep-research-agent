"""Tests for the Gemini REST client."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import requests

from deep_research_agent.llm import GeminiClient, GeminiClientConfig, LLMClientError


class GeminiClientTests(unittest.TestCase):
    def _response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": text}],
                    }
                }
            ]
        }
        return resp

    @patch("deep_research_agent.llm.gemini_client.requests.post")
    def test_generate_text_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = self._response("Hello from Gemini")
        client = GeminiClient(GeminiClientConfig(api_key="test-key", model="gemini-2.0-flash"))
        result = client.generate_text(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        self.assertEqual(result, "Hello from Gemini")
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertIn("systemInstruction", payload)
        self.assertEqual(payload["contents"][0]["role"], "user")

    @patch("deep_research_agent.llm.gemini_client.requests.post")
    def test_retries_on_429(self, mock_post: MagicMock) -> None:
        ok = self._response("recovered")
        err_resp = MagicMock(status_code=429)
        http_err = requests.HTTPError("429", response=err_resp)

        call_count = {"n": 0}

        def side_effect(*_a, **_k):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise http_err
            return ok

        mock_post.side_effect = side_effect
        client = GeminiClient(
            GeminiClientConfig(api_key="test-key", max_retries=2),
        )
        with patch("deep_research_agent.ingestion.retry.time.sleep"):
            result = client.generate_text([{"role": "user", "content": "Hi"}])
        self.assertEqual(result, "recovered")
        self.assertEqual(mock_post.call_count, 2)

    def test_missing_api_key_raises(self) -> None:
        client = GeminiClient(GeminiClientConfig(api_key=""))
        with self.assertRaises(LLMClientError):
            client.generate_text([{"role": "user", "content": "Hi"}])


if __name__ == "__main__":
    unittest.main()
