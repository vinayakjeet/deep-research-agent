"""Google Gemini REST API client (raw HTTP, no SDK)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import requests

from deep_research_agent.ingestion.retry import retry_with_backoff
from deep_research_agent.llm.errors import LLMClientError

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass(slots=True)
class GeminiClientConfig:
    """Configuration for Gemini generateContent calls."""

    api_key: str = ""
    model: str = "gemini-3.5-flash"
    timeout: float = 60.0
    max_retries: int = 3
    temperature: float = 0.2
    max_output_tokens: int = 8192


class GeminiClient:
    """Minimal Gemini chat client using the REST generateContent endpoint."""

    provider_name = "gemini"

    def __init__(self, config: Optional[GeminiClientConfig] = None) -> None:
        cfg = config or GeminiClientConfig()
        self.api_key = cfg.api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = cfg.model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self.timeout = cfg.timeout
        self.max_retries = cfg.max_retries
        self.temperature = cfg.temperature
        self.max_output_tokens = cfg.max_output_tokens

    def generate_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
    ) -> str:
        """Send messages and return the model text response."""
        if not self.api_key:
            raise LLMClientError(
                "GEMINI_API_KEY is not set",
                provider=self.provider_name,
            )

        payload = self._build_payload(messages, temperature=temperature)
        url = f"{GEMINI_BASE_URL}/{self.model}:generateContent"

        def _request() -> requests.Response:
            response = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise requests.HTTPError(
                    f"Gemini HTTP {response.status_code}",
                    response=response,
                )
            return response

        def _status_from_exc(exc: Exception) -> int | None:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                return int(exc.response.status_code)
            return None

        try:
            response = retry_with_backoff(
                _request,
                max_retries=self.max_retries,
                get_status_code=_status_from_exc,
            )
        except requests.HTTPError as exc:
            status = _status_from_exc(exc)
            raise LLMClientError(
                f"Gemini request failed: HTTP {status}",
                provider=self.provider_name,
                status_code=status,
                cause=exc,
            ) from exc
        except requests.RequestException as exc:
            raise LLMClientError(
                f"Gemini request failed: {exc}",
                provider=self.provider_name,
                cause=exc,
            ) from exc

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise LLMClientError(
                "Gemini returned invalid JSON",
                provider=self.provider_name,
                cause=exc,
            ) from exc

        return self._extract_text(data)

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float],
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": text}],
                }
            )

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}],
            }
        return payload

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMClientError(
                "Gemini returned no candidates",
                provider="gemini",
            )
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))
        if not texts:
            raise LLMClientError(
                "Gemini returned empty text",
                provider="gemini",
            )
        return "\n".join(texts).strip()
