"""Retrieval metrics: precision, recall, F1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from deep_research_agent.orchestration.citations import normalize_domain


@dataclass(slots=True)
class RetrievalScore:
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    true_positives: int
    retrieved_count: int
    ground_truth_count: int
    matched_urls: list[str]
    matched_domains: list[str]
    skipped: bool = False
    skip_reason: str = ""


def _domain_from_url(url: str) -> str:
    parsed = urlparse(url.strip())
    return normalize_domain(parsed.netloc or "")


def _url_matches_gt(url: str, ground_truth_urls: list[str], acceptable_domains: list[str]) -> bool:
    url = url.strip()
    if not url:
        return False
    norm_domains = {normalize_domain(d) for d in acceptable_domains if d}
    url_domain = _domain_from_url(url)
    if url_domain and url_domain in norm_domains:
        return True
    for gt in ground_truth_urls:
        gt = gt.strip()
        if not gt:
            continue
        if url.rstrip("/") == gt.rstrip("/"):
            return True
        if _domain_from_url(gt) == url_domain and url_domain:
            return True
    return False


def score_retrieval(
    retrieved_urls: list[str],
    ground_truth_urls: list[str],
    acceptable_domains: list[str],
) -> RetrievalScore:
    """Compute P/R/F1. Skip when ground_truth_urls is empty."""
    gt = [u for u in ground_truth_urls if u and u.strip()]
    if not gt:
        return RetrievalScore(
            precision=None,
            recall=None,
            f1=None,
            true_positives=0,
            retrieved_count=len(set(retrieved_urls)),
            ground_truth_count=0,
            matched_urls=[],
            matched_domains=[],
            skipped=True,
            skip_reason="empty ground_truth_urls",
        )

    seen_retrieved: list[str] = []
    for u in retrieved_urls:
        if u and u not in seen_retrieved:
            seen_retrieved.append(u)

    matched: list[str] = []
    matched_domains: list[str] = []
    for url in seen_retrieved:
        if _url_matches_gt(url, gt, acceptable_domains):
            matched.append(url)
            dom = _domain_from_url(url)
            if dom and dom not in matched_domains:
                matched_domains.append(dom)

    tp = len(matched)
    retrieved_count = len(seen_retrieved)
    gt_count = len(gt)

    precision = tp / retrieved_count if retrieved_count else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return RetrievalScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        retrieved_count=retrieved_count,
        ground_truth_count=gt_count,
        matched_urls=matched,
        matched_domains=matched_domains,
    )
