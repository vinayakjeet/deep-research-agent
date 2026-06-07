"""Serper (Google Search) API client."""

from __future__ import annotations

import os
from typing import Any

import requests

from deep_research_agent.ingestion.errors import SearchProviderError
from deep_research_agent.ingestion.retry import retry_with_backoff
from deep_research_agent.ingestion.search.base import normalize_result
from deep_research_agent.state.schema import SearchResult

SERPER_SEARCH_URL = "https://google.serper.dev/search"


class SerperSearchProvider:
    """Search provider backed by the Serper REST API."""

    provider_name = "serper"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise SearchProviderError(
                "SERPER_API_KEY is not set",
                provider=self.provider_name,
            )
        if not query.strip():
            return []

        payload = {"q": query.strip(), "num": max_results}
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        def _request() -> requests.Response:
            response = requests.post(
                SERPER_SEARCH_URL,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise requests.HTTPError(
                    f"Serper HTTP {response.status_code}",
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
            raise SearchProviderError(
                f"Serper search failed: HTTP {status}",
                provider=self.provider_name,
                status_code=status,
                cause=exc,
            ) from exc
        except requests.RequestException as exc:
            raise SearchProviderError(
                f"Serper search request failed: {exc}",
                provider=self.provider_name,
                cause=exc,
            ) from exc

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise SearchProviderError(
                "Serper returned invalid JSON",
                provider=self.provider_name,
                cause=exc,
            ) from exc

        results: list[SearchResult] = []
        for item in data.get("organic") or []:
            if not isinstance(item, dict):
                continue
            normalized = {
                "title": item.get("title", ""),
                "url": item.get("link", item.get("url", "")),
                "snippet": item.get("snippet", ""),
                "score": item.get("position"),
            }
            if normalized["url"]:
                results.append(normalize_result(normalized))
        return results[:max_results]
