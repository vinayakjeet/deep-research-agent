"""Hardcoded responses when research cannot find verifiable evidence."""

from __future__ import annotations

NO_EVIDENCE_MESSAGE = (
    "I could not locate verifiable evidence on the open web to answer your question reliably. "
    "No answer has been synthesized from unavailable or insufficient sources."
)


def build_no_evidence_response(user_query: str, *, reason: str) -> str:
    """Return a professional no-evidence message without calling the LLM."""
    query = user_query.strip() or "your question"
    return (
        f"{NO_EVIDENCE_MESSAGE}\n\n"
        f"Question: {query}\n"
        f"Reason: {reason.strip()}\n\n"
        "You may try rephrasing the query, narrowing the topic, or providing more specific keywords."
    )
