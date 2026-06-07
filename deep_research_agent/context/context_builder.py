"""Context window distillation with sliding-window history and summarization."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from deep_research_agent.context.tokens import estimate_object_tokens, estimate_tokens
from deep_research_agent.state.enums import MessageRole
from deep_research_agent.state.schema import (
    AgentState,
    ContextSnippet,
    MessageRecord,
    SourceContext,
    TurnRecord,
)

SummarizeFn = Callable[[list[MessageRecord]], str]


def default_extractive_summary(messages: list[MessageRecord]) -> str:
    """Deterministic fallback when no LLM summarizer is configured."""
    if not messages:
        return ""
    lines = ["Summary of earlier conversation:"]
    for msg in messages:
        prefix = msg.role.value.capitalize()
        excerpt = msg.content.strip().replace("\n", " ")
        if len(excerpt) > 240:
            excerpt = excerpt[:237] + "..."
        lines.append(f"- {prefix}: {excerpt}")
    return "\n".join(lines)


@dataclass(slots=True)
class ContextBuilderConfig:
    """Budget and behavior settings for prompt assembly."""

    max_tokens: int = 8_000
    reserved_tokens_for_web: int = 4_000
    min_recent_messages: int = 2
    chars_per_token: float = 4.0
    prompt_overhead_tokens: int = 200


@dataclass(slots=True)
class BuiltContext:
    """Distilled payload ready for LLM prompt injection."""

    current_query: str
    messages: list[dict[str, str]] = field(default_factory=list)
    conversation_summary: Optional[str] = None
    source_contexts: list[dict[str, Any]] = field(default_factory=list)
    selected_snippets: list[dict[str, Any]] = field(default_factory=list)
    plan: Optional[dict[str, Any]] = None
    search_queries: list[str] = field(default_factory=list)
    prior_turn_summaries: list[str] = field(default_factory=list)
    token_count: int = 0
    was_summarized: bool = False
    dropped_message_count: int = 0

    def to_prompt_dict(self) -> dict[str, Any]:
        conversation: list[dict[str, str]] = []
        if self.conversation_summary:
            conversation.append(
                {"role": "system", "content": self.conversation_summary}
            )
        conversation.extend(self.messages)
        return {
            "current_query": self.current_query,
            "conversation": conversation,
            "prior_turn_summaries": self.prior_turn_summaries,
            "plan": self.plan,
            "search_queries": self.search_queries,
            "selected_snippets": self.selected_snippets,
            "source_contexts": self.source_contexts,
        }

    def to_prompt_string(self) -> str:
        return json.dumps(self.to_prompt_dict(), ensure_ascii=False, indent=2)


class ContextBuilder:
    """
    Builds bounded LLM context from agent state.

    Web research material (source contexts and selected snippets) is always
    preserved in full. Conversation history uses a sliding window with an
    optional summarization fallback for dropped older messages.
    """

    def __init__(
        self,
        config: Optional[ContextBuilderConfig] = None,
        summarizer: Optional[SummarizeFn] = None,
    ) -> None:
        self.config = config or ContextBuilderConfig()
        self.summarizer = summarizer or default_extractive_summary

    def build_from_state(self, state: AgentState) -> BuiltContext:
        return self.build(
            current_query=state.current_query or "",
            messages=state.messages,
            source_contexts=state.source_contexts,
            selected_snippets=state.selected_snippets,
            plan=state.plan.to_dict() if state.plan else None,
            search_queries=list(state.search_queries),
        )

    def build(
        self,
        *,
        current_query: str,
        messages: list[MessageRecord],
        source_contexts: list[SourceContext],
        selected_snippets: list[ContextSnippet],
        plan: Optional[dict[str, Any]] = None,
        search_queries: Optional[list[str]] = None,
        prior_turns: Optional[list[TurnRecord]] = None,
    ) -> BuiltContext:
        cfg = self.config

        protected_contexts = [self._format_source_context(ctx) for ctx in source_contexts]
        protected_snippets = [snippet.to_dict() for snippet in selected_snippets]
        protected_query = current_query.strip()
        protected_plan = plan
        protected_search = list(search_queries or [])

        protected_tokens = (
            estimate_tokens(protected_query, chars_per_token=cfg.chars_per_token)
            + estimate_object_tokens(protected_contexts, chars_per_token=cfg.chars_per_token)
            + estimate_object_tokens(protected_snippets, chars_per_token=cfg.chars_per_token)
            + estimate_object_tokens(protected_plan, chars_per_token=cfg.chars_per_token)
            + estimate_object_tokens(protected_search, chars_per_token=cfg.chars_per_token)
        )

        web_budget = max(cfg.reserved_tokens_for_web, protected_tokens)
        history_budget = max(
            0,
            cfg.max_tokens - web_budget - cfg.prompt_overhead_tokens,
        )

        recent_messages, dropped, summary = self._distill_conversation(
            messages,
            token_budget=history_budget,
        )

        prior_turn_summaries = self._compact_prior_turns(prior_turns or [])

        built = BuiltContext(
            current_query=protected_query,
            messages=recent_messages,
            conversation_summary=summary,
            source_contexts=protected_contexts,
            selected_snippets=protected_snippets,
            plan=protected_plan,
            search_queries=protected_search,
            prior_turn_summaries=prior_turn_summaries,
            was_summarized=summary is not None,
            dropped_message_count=len(dropped),
        )
        built.token_count = estimate_object_tokens(
            built.to_prompt_dict(),
            chars_per_token=cfg.chars_per_token,
        )

        if built.token_count > cfg.max_tokens:
            built = self._enforce_hard_cap(built)

        return built

    def _distill_conversation(
        self,
        messages: list[MessageRecord],
        *,
        token_budget: int,
    ) -> tuple[list[dict[str, str]], list[MessageRecord], Optional[str]]:
        if not messages:
            return [], [], None

        cfg = self.config
        window: deque[MessageRecord] = deque()

        for message in reversed(messages):
            candidate = [message, *window]
            candidate_payload = [
                {"role": msg.role.value, "content": msg.content} for msg in candidate
            ]
            candidate_tokens = estimate_object_tokens(
                candidate_payload,
                chars_per_token=cfg.chars_per_token,
            )
            if candidate_tokens <= token_budget or len(window) < cfg.min_recent_messages:
                window.appendleft(message)
            else:
                break

        kept = list(window)
        dropped_count = len(messages) - len(kept)
        dropped = messages[:dropped_count] if dropped_count > 0 else []

        summary: Optional[str] = None
        if dropped:
            summary = self.summarizer(dropped).strip() or None

        recent = [{"role": msg.role.value, "content": msg.content} for msg in kept]
        return recent, dropped, summary

    def _compact_prior_turns(self, turns: list[TurnRecord]) -> list[str]:
        summaries: list[str] = []
        for turn in turns[-3:]:
            answer_preview = (turn.final_answer or "").strip()
            if len(answer_preview) > 180:
                answer_preview = answer_preview[:177] + "..."
            summaries.append(
                f"Turn {turn.turn_id} — Q: {turn.user_query} | A: {answer_preview or '[pending]'}"
            )
        return summaries

    def _enforce_hard_cap(self, built: BuiltContext) -> BuiltContext:
        """Last-resort trim that never touches web research payloads."""
        cfg = self.config
        trimmed_messages = list(built.messages)

        while trimmed_messages:
            built.messages = trimmed_messages
            built.token_count = estimate_object_tokens(
                built.to_prompt_dict(),
                chars_per_token=cfg.chars_per_token,
            )
            if built.token_count <= cfg.max_tokens:
                break
            trimmed_messages.pop(0)

        built.messages = trimmed_messages

        if built.conversation_summary:
            summary = built.conversation_summary
            while summary:
                built.conversation_summary = summary
                built.token_count = estimate_object_tokens(
                    built.to_prompt_dict(),
                    chars_per_token=cfg.chars_per_token,
                )
                if built.token_count <= cfg.max_tokens:
                    break
                summary = summary[: max(0, len(summary) - 200)].rstrip()
            if not summary:
                built.conversation_summary = None

        built.token_count = estimate_object_tokens(
            built.to_prompt_dict(),
            chars_per_token=cfg.chars_per_token,
        )
        return built

    @staticmethod
    def _format_source_context(ctx: SourceContext) -> dict[str, Any]:
        return {
            "url": ctx.url,
            "title": ctx.title,
            "domain": ctx.domain,
            "text_block": ctx.text_block,
            "retrieved_at": ctx.retrieved_at,
        }
