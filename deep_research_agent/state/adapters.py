"""Bridge MemoryStore rows and typed state objects."""

from __future__ import annotations

from typing import Optional

from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.state.schema import (
    AgentState,
    MessageRecord,
    SessionRecord,
    TurnRecord,
)


def load_session_record(store: MemoryStore, session_id: str) -> Optional[SessionRecord]:
    row = store.get_session(session_id)
    return SessionRecord.from_store_row(row) if row else None


def load_messages(store: MemoryStore, session_id: str) -> list[MessageRecord]:
    return [MessageRecord.from_store_row(row) for row in store.get_messages(session_id)]


def load_turn_record(store: MemoryStore, turn_id: int) -> Optional[TurnRecord]:
    row = store.reconstruct_turn_history(turn_id)
    return TurnRecord.from_store_row(row) if row else None


def load_agent_state(store: MemoryStore, session_id: str) -> Optional[AgentState]:
    row = store.reconstruct_session(session_id)
    return AgentState.from_store_session(row) if row else None
