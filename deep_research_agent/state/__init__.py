"""Typed state schema for the research agent."""

from deep_research_agent.state.enums import AgentPhase, MessageRole
from deep_research_agent.state.schema import (
    AgentState,
    ContextSnippet,
    MessageRecord,
    ResearchPlan,
    SearchResult,
    SessionRecord,
    SourceContext,
    TurnRecord,
)
from deep_research_agent.state.adapters import (
    load_agent_state,
    load_messages,
    load_session_record,
    load_turn_record,
)
from deep_research_agent.state.serialization import (
    agent_state_from_json,
    agent_state_to_json,
    from_json,
    to_json,
)

__all__ = [
    "AgentPhase",
    "MessageRole",
    "AgentState",
    "ContextSnippet",
    "MessageRecord",
    "ResearchPlan",
    "SearchResult",
    "SessionRecord",
    "SourceContext",
    "TurnRecord",
    "agent_state_from_json",
    "agent_state_to_json",
    "from_json",
    "to_json",
    "load_agent_state",
    "load_messages",
    "load_session_record",
    "load_turn_record",
]
