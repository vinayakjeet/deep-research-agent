"""Concurrent async HTTP page fetcher."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from deep_research_agent.ingestion.fetch.models import FetchResult, FetchStatus

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; DeepResearchAgent/1.0; +https://github.com/local)"
)


@dataclass(slots=True)
class PageFetcherConfig:
    """Configuration for page fetching."""

    timeout_sec: float = 15.0
    max_concurrent: int = 8
    max_retries: int = 1
    user_agent: str = DEFAULT_USER_AGENT
    max_body_bytes: int = 2_000_000


class PageFetcher:
    """Fetches multiple URLs concurrently with timeouts and error classification."""

    def __init__(self, config: Optional[PageFetcherConfig] = None) -> None:
        self.config = config or PageFetcherConfig()

    async def fetch_one(
        self,
        url: str,
        session: aiohttp.ClientSession,
    ) -> FetchResult:
        cfg = self.config
        start = time.perf_counter()
        last_error: Optional[str] = None

        for attempt in range(cfg.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=cfg.timeout_sec)
                async with session.get(
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                ) as response:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    status_code = response.status

                    if status_code >= 400:
                        return FetchResult(
                            url=url,
                            status=FetchStatus.HTTP_ERROR,
                            status_code=status_code,
                            error=f"HTTP {status_code}",
                            elapsed_ms=elapsed_ms,
                        )

                    raw = await response.read()
                    if len(raw) > cfg.max_body_bytes:
                        raw = raw[: cfg.max_body_bytes]

                    charset = response.charset or "utf-8"
                    try:
                        html = raw.decode(charset, errors="replace")
                    except LookupError:
                        html = raw.decode("utf-8", errors="replace")

                    return FetchResult(
                        url=url,
                        status=FetchStatus.OK,
                        status_code=status_code,
                        html=html,
                        elapsed_ms=elapsed_ms,
                    )

            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
                last_error = f"Timeout after {cfg.timeout_sec}s"
                if attempt >= cfg.max_retries:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return FetchResult(
                        url=url,
                        status=FetchStatus.TIMEOUT,
                        error=last_error,
                        elapsed_ms=elapsed_ms,
                    )
            except aiohttp.ClientError as exc:
                last_error = str(exc)
                if attempt >= cfg.max_retries:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return FetchResult(
                        url=url,
                        status=FetchStatus.NETWORK_ERROR,
                        error=last_error,
                        elapsed_ms=elapsed_ms,
                    )
            except Exception as exc:
                last_error = str(exc)
                if attempt >= cfg.max_retries:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return FetchResult(
                        url=url,
                        status=FetchStatus.NETWORK_ERROR,
                        error=last_error,
                        elapsed_ms=elapsed_ms,
                    )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return FetchResult(
            url=url,
            status=FetchStatus.NETWORK_ERROR,
            error=last_error or "Unknown fetch error",
            elapsed_ms=elapsed_ms,
        )

    async def fetch_many(self, urls: list[str]) -> list[FetchResult]:
        if not urls:
            return []

        cfg = self.config
        headers = {"User-Agent": cfg.user_agent}
        semaphore = asyncio.Semaphore(cfg.max_concurrent)

        async with aiohttp.ClientSession(headers=headers) as session:

            async def _bounded_fetch(url: str) -> FetchResult:
                async with semaphore:
                    return await self.fetch_one(url, session)

            return list(await asyncio.gather(*[_bounded_fetch(u) for u in urls]))

    def fetch_many_sync(self, urls: list[str]) -> list[FetchResult]:
        """Synchronous entry point for callers without an event loop."""
        return asyncio.run(self.fetch_many(urls))
