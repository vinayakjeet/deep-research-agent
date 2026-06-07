"""Assemble bounded source context for answer generation."""

from __future__ import annotations

from typing import Optional

from deep_research_agent.llm.gemini_client import GeminiClient
from deep_research_agent.llm.prompts import ANSWER_SYSTEM_PROMPT
from deep_research_agent.state.schema import AgentState, ContextSnippet, SourceContext

DEFAULT_MAX_SNIPPET_CHARS = 1200
DEFAULT_FALLBACK_TEXT_CHARS = 800


def assemble_sources(
    snippets: list[ContextSnippet],
    *,
    source_contexts: Optional[list[SourceContext]] = None,
    max_sources: int = 10,
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
) -> str:
    """
    Build delimited source blocks for the answer prompt.

    Prefers TF-selected snippets; falls back to truncated text_block per URL.
    """
    blocks: list[str] = []
    used_urls: set[str] = set()

    for index, snippet in enumerate(snippets[:max_sources], start=1):
        content = snippet.snippet.strip()
        if len(content) > max_snippet_chars:
            content = content[: max_snippet_chars - 3] + "..."
        title = snippet.title or "Untitled"
        domain = snippet.domain or ""
        blocks.append(
            "\n".join(
                [
                    f"<source_{index}>",
                    f"Title: {title}",
                    f"URL: {snippet.url}",
                    f"Domain: {domain}",
                    "Content:",
                    content,
                    f"</source_{index}>",
                ]
            )
        )
        used_urls.add(snippet.url)

    if len(blocks) < max_sources and source_contexts:
        next_index = len(blocks) + 1
        for ctx in source_contexts:
            if ctx.url in used_urls:
                continue
            if next_index > max_sources:
                break
            text = (ctx.text_block or "").strip()
            if not text:
                continue
            if len(text) > DEFAULT_FALLBACK_TEXT_CHARS:
                text = text[: DEFAULT_FALLBACK_TEXT_CHARS - 3] + "..."
            blocks.append(
                "\n".join(
                    [
                        f"<source_{next_index}>",
                        f"Title: {ctx.title or 'Untitled'}",
                        f"URL: {ctx.url}",
                        f"Domain: {ctx.domain or ''}",
                        "Content:",
                        text,
                        f"</source_{next_index}>",
                    ]
                )
            )
            used_urls.add(ctx.url)
            next_index += 1

    return "\n\n".join(blocks)


def build_answer_messages(
    user_query: str,
    sources_block: str,
    *,
    conversation_summary: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build Gemini messages for the answer phase."""
    user_parts = [
        f"User question:\n{user_query.strip()}",
        "Use only the following sources:",
        sources_block,
    ]
    if conversation_summary and conversation_summary.strip():
        user_parts.insert(1, f"Prior conversation summary:\n{conversation_summary.strip()}")

    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


class AnswerGenerator:
    """Produces final grounded answers from assembled sources."""

    def __init__(self, client: Optional[GeminiClient] = None) -> None:
        self.client = client or GeminiClient()

    def generate(
        self,
        user_query: str,
        state: AgentState,
        *,
        conversation_summary: Optional[str] = None,
    ) -> str:
        sources_block = assemble_sources(
            state.selected_snippets,
            source_contexts=state.source_contexts,
        )
        if not sources_block.strip():
            raise ValueError("No sources available for answer generation")

        messages = build_answer_messages(
            user_query,
            sources_block,
            conversation_summary=conversation_summary,
        )
        return self.client.generate_text(messages)
