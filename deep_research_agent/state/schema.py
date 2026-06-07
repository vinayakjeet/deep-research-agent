"""Typed dataclasses mapping persistence records and runtime agent state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from deep_research_agent.state.enums import AgentPhase, MessageRole


class PlanStepDict(TypedDict, total=False):
    """Structured plan step emitted by the planning phase."""

    description: str
    search_query: str


class ContextSnippetDict(TypedDict, total=False):
    """Selected snippet persisted on a turn record."""

    url: str
    snippet: str
    title: str
    domain: str


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_store_row(cls, row: dict[str, Any]) -> SessionRecord:
        return cls(
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=row.get("metadata"),
        )


@dataclass(slots=True)
class MessageRecord:
    message_id: int
    session_id: str
    role: MessageRole
    content: str
    created_at: str

    @classmethod
    def from_store_row(cls, row: dict[str, Any]) -> MessageRecord:
        return cls(
            message_id=int(row["message_id"]),
            session_id=row["session_id"],
            role=MessageRole.from_str(row["role"]),
            content=row["content"],
            created_at=row["created_at"],
        )


@dataclass(slots=True)
class ResearchPlan:
    """Structured output from the planning phase."""

    summary: str
    search_queries: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Optional[ResearchPlan]:
        if not data:
            return None
        return cls(
            summary=str(data.get("summary", "")),
            search_queries=list(data.get("search_queries", [])),
            steps=list(data.get("steps", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "search_queries": self.search_queries,
            "steps": self.steps,
        }


@dataclass(slots=True)
class SearchResult:
    """Normalized search provider result."""

    title: str
    url: str
    snippet: str
    score: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        score = data.get("score")
        return cls(
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            snippet=str(data.get("snippet", data.get("content", ""))),
            score=float(score) if score is not None else None,
        )


@dataclass(slots=True)
class ContextSnippet:
    """Snippet selected for LLM context from a fetched source."""

    url: str
    snippet: str
    title: Optional[str] = None
    domain: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextSnippet:
        return cls(
            url=str(data["url"]),
            snippet=str(data.get("snippet", data.get("text", ""))),
            title=data.get("title"),
            domain=data.get("domain"),
        )

    def to_dict(self) -> ContextSnippetDict:
        payload: ContextSnippetDict = {
            "url": self.url,
            "snippet": self.snippet,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.domain is not None:
            payload["domain"] = self.domain
        return payload


@dataclass(slots=True)
class SourceContext:
    """Fetched page content linked to a research turn."""

    context_id: int
    turn_id: int
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None
    text_block: Optional[str] = None
    retrieved_at: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_store_row(cls, row: dict[str, Any]) -> SourceContext:
        return cls(
            context_id=int(row["context_id"]),
            turn_id=int(row["turn_id"]),
            url=row["url"],
            title=row.get("title"),
            domain=row.get("domain"),
            text_block=row.get("text_block"),
            retrieved_at=row.get("retrieved_at"),
            metadata=row.get("metadata"),
        )


@dataclass(slots=True)
class TurnRecord:
    """Audit record for one user query research cycle."""

    turn_id: int
    session_id: str
    user_query: str
    created_at: str
    search_queries: list[str] = field(default_factory=list)
    urls_opened: list[str] = field(default_factory=list)
    context_snippets: list[ContextSnippet] = field(default_factory=list)
    final_answer: Optional[str] = None
    plan: Optional[ResearchPlan] = None
    contexts: list[SourceContext] = field(default_factory=list)
    audit_trail: Optional[dict[str, Any]] = None

    @classmethod
    def from_store_row(cls, row: dict[str, Any]) -> TurnRecord:
        raw_snippets = row.get("context_snippets") or []
        snippets = [ContextSnippet.from_dict(item) for item in raw_snippets]
        raw_plan = row.get("plan")
        plan = ResearchPlan.from_dict(raw_plan) if isinstance(raw_plan, dict) else None
        nested_contexts = row.get("contexts") or []

        return cls(
            turn_id=int(row["turn_id"]),
            session_id=row["session_id"],
            user_query=row["user_query"],
            created_at=row["created_at"],
            search_queries=list(row.get("search_queries") or []),
            urls_opened=list(row.get("urls_opened") or []),
            context_snippets=snippets,
            final_answer=row.get("final_answer"),
            plan=plan,
            contexts=[SourceContext.from_store_row(item) for item in nested_contexts],
            audit_trail=row.get("audit_trail"),
        )


@dataclass(slots=True)
class AgentState:
    """Runtime workflow state for the orchestration loop."""

    session_id: str
    phase: AgentPhase = AgentPhase.START
    current_query: Optional[str] = None
    turn_id: Optional[int] = None
    plan: Optional[ResearchPlan] = None
    search_queries: list[str] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    urls_opened: list[str] = field(default_factory=list)
    source_contexts: list[SourceContext] = field(default_factory=list)
    selected_snippets: list[ContextSnippet] = field(default_factory=list)
    final_answer: Optional[str] = None
    messages: list[MessageRecord] = field(default_factory=list)
    error_message: Optional[str] = None

    def advance_to(self, phase: AgentPhase) -> None:
        self.phase = phase

    def is_terminal(self) -> bool:
        return self.phase.is_terminal()

    @classmethod
    def from_session(
        cls,
        session: SessionRecord,
        *,
        messages: list[MessageRecord] | None = None,
        phase: AgentPhase = AgentPhase.START,
    ) -> AgentState:
        return cls(
            session_id=session.session_id,
            phase=phase,
            messages=list(messages or []),
        )

    @classmethod
    def from_store_session(cls, row: dict[str, Any]) -> AgentState:
        session = SessionRecord.from_store_row(row)
        raw_messages = row.get("messages") or []
        messages = [MessageRecord.from_store_row(item) for item in raw_messages]
        turns = row.get("turns") or []
        state = cls.from_session(session, messages=messages)

        if turns:
            latest = TurnRecord.from_store_row(turns[-1])
            state.turn_id = latest.turn_id
            state.current_query = latest.user_query
            state.plan = latest.plan
            state.search_queries = list(latest.search_queries)
            state.urls_opened = list(latest.urls_opened)
            state.selected_snippets = list(latest.context_snippets)
            state.source_contexts = list(latest.contexts)
            state.final_answer = latest.final_answer
            state.phase = (
                AgentPhase.COMPLETE if latest.final_answer else AgentPhase.START
            )
        return state

    def to_prompt_payload(self) -> dict[str, Any]:
        """Compact dict suitable for LLM prompt injection."""
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "current_query": self.current_query,
            "plan": self.plan.to_dict() if self.plan else None,
            "search_queries": self.search_queries,
            "selected_snippets": [snippet.to_dict() for snippet in self.selected_snippets],
            "source_contexts": [
                {
                    "url": ctx.url,
                    "title": ctx.title,
                    "domain": ctx.domain,
                    "text_block": ctx.text_block,
                }
                for ctx in self.source_contexts
            ],
            "conversation": [
                {"role": msg.role.value, "content": msg.content}
                for msg in self.messages
            ],
        }
