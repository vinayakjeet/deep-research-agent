"""Immutable audit trail tracker for research turn lifecycle."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.state.schema import ContextSnippet, TurnRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_transaction_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One append-only micro-step in a turn audit trail."""

    transaction_id: str
    event_type: str
    timestamp: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class TurnHistoryTracker:
    """
    Records an immutable, append-only audit trail for one research turn.

    Each orchestration micro-step receives a distinct transaction_id and is
    persisted to SQLite (turn row + contexts table for DOM excerpts).
    """

    def __init__(self, store: MemoryStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id
        self._turn_id: Optional[int] = None
        self._root_transaction_id: Optional[str] = None
        self._user_query: str = ""
        self._events: list[AuditEvent] = []
        self._search_queries: list[str] = []
        self._urls_opened: list[str] = []
        self._context_refs: list[dict[str, Any]] = []
        self._snippets: list[dict[str, Any]] = []
        self._plan: Optional[dict[str, Any]] = None
        self._final_answer: Optional[str] = None
        self._closed = False

    @property
    def turn_id(self) -> Optional[int]:
        return self._turn_id

    @property
    def root_transaction_id(self) -> Optional[str]:
        return self._root_transaction_id

    @property
    def is_closed(self) -> bool:
        return self._closed

    def begin(self, user_query: str) -> int:
        """Start a new turn audit trail and return its turn_id."""
        if self._turn_id is not None:
            raise RuntimeError("Turn already started on this tracker instance")

        self._user_query = user_query.strip()
        if not self._user_query:
            raise ValueError("user_query must be non-empty")

        root_tx = _new_transaction_id()
        self._root_transaction_id = root_tx

        turn = self._store.create_turn(
            session_id=self._session_id,
            user_query=self._user_query,
        )
        self._turn_id = int(turn["turn_id"])

        self._append_event("turn_started", root_tx, {"user_query": self._user_query})
        self._persist_audit_trail()
        return self._turn_id

    def record_plan(
        self,
        plan: dict[str, Any],
        search_queries: Optional[list[str]] = None,
    ) -> str:
        """Log the planning phase and generated search strings."""
        self._ensure_open()

        queries = list(search_queries or plan.get("search_queries") or [])
        self._plan = deepcopy(plan)
        self._search_queries = queries

        tx = _new_transaction_id()
        self._append_event(
            "plan_recorded",
            tx,
            {"plan": plan, "search_queries": queries},
        )
        self._store.update_turn(
            self._turn_id,
            plan=plan,
            search_queries=queries,
        )
        self._persist_audit_trail()
        return tx

    def record_search_queries(self, queries: list[str]) -> str:
        """Log search strings without a full plan payload."""
        self._ensure_open()

        self._search_queries = list(queries)
        tx = _new_transaction_id()
        self._append_event(
            "search_queries_recorded",
            tx,
            {"search_queries": queries},
        )
        self._store.update_turn(self._turn_id, search_queries=queries)
        self._persist_audit_trail()
        return tx

    def record_fetch_outcome(
        self,
        url: str,
        *,
        fetch_status: str,
        status_code: int | None = None,
        error: str | None = None,
        elapsed_ms: float | None = None,
    ) -> str:
        """Log a page fetch attempt (success or failure) without storing full HTML."""
        self._ensure_open()

        event_type = "fetch_ok" if fetch_status == "ok" else "fetch_failed"
        tx = _new_transaction_id()
        self._append_event(
            event_type,
            tx,
            {
                "url": url,
                "fetch_status": fetch_status,
                "status_code": status_code,
                "error": error,
                "elapsed_ms": elapsed_ms,
            },
        )
        if fetch_status == "ok" and url not in self._urls_opened:
            self._urls_opened.append(url)
            self._store.update_turn(self._turn_id, urls_opened=list(self._urls_opened))
        self._persist_audit_trail()
        return tx

    def record_url_extraction(
        self,
        *,
        url: str,
        text_block: str,
        title: Optional[str] = None,
        domain: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Persist a DOM text excerpt to the contexts table."""
        self._ensure_open()

        tx = _new_transaction_id()
        ctx_metadata: dict[str, Any] = {
            "transaction_id": tx,
            "event_type": "url_extraction",
        }
        if metadata:
            ctx_metadata.update(metadata)

        ctx = self._store.add_context(
            turn_id=self._turn_id,
            url=url,
            title=title,
            domain=domain,
            text_block=text_block,
            metadata=ctx_metadata,
        )

        if url not in self._urls_opened:
            self._urls_opened.append(url)

        ref = {
            "context_id": ctx["context_id"],
            "url": url,
            "title": title,
            "domain": domain,
            "transaction_id": tx,
            "text_preview": (text_block or "")[:200],
        }
        self._context_refs.append(ref)

        self._append_event(
            "url_extraction",
            tx,
            {
                "context_id": ctx["context_id"],
                "url": url,
                "title": title,
                "domain": domain,
                "text_block_length": len(text_block or ""),
            },
        )
        self._store.update_turn(self._turn_id, urls_opened=list(self._urls_opened))
        self._persist_audit_trail()
        return tx

    def record_snippets(
        self,
        snippets: list[ContextSnippet | dict[str, Any]],
    ) -> str:
        """Log selected textual excerpts chosen for prompt injection."""
        self._ensure_open()

        normalized: list[dict[str, Any]] = []
        for item in snippets:
            if isinstance(item, ContextSnippet):
                normalized.append(item.to_dict())
            else:
                normalized.append(dict(item))

        self._snippets = normalized
        tx = _new_transaction_id()
        self._append_event("snippets_selected", tx, {"snippets": normalized})
        self._store.update_turn(self._turn_id, context_snippets=normalized)
        self._persist_audit_trail()
        return tx

    def record_citation_anomalies(self, report: dict[str, Any]) -> str:
        """Log citation validation results from post-answer checks."""
        self._ensure_open()
        tx = _new_transaction_id()
        self._append_event("citation_anomaly", tx, report)
        self._persist_audit_trail()
        return tx

    def finalize_answer(self, final_answer: str) -> str:
        """Commit the final synthesized response and close the audit trail."""
        self._ensure_open()

        tx = _new_transaction_id()
        self._final_answer = final_answer
        self._append_event("answer_finalized", tx, {"final_answer": final_answer})

        self._store.update_turn(
            self._turn_id,
            final_answer=final_answer,
            context_snippets=list(self._snippets),
            urls_opened=list(self._urls_opened),
            search_queries=list(self._search_queries),
            plan=self._plan,
        )
        self._closed = True
        self._persist_audit_trail()
        return tx

    def to_audit_dict(self) -> dict[str, Any]:
        """Return a deep copy of the nested audit trail structure."""
        return deepcopy(self._build_audit_dict())

    def reconstruct(self) -> Optional[TurnRecord]:
        """Rebuild the typed turn record from SQLite, including contexts."""
        if self._turn_id is None:
            return None
        row = self._store.reconstruct_turn_history(self._turn_id)
        return TurnRecord.from_store_row(row) if row else None

    @classmethod
    def from_persisted_turn(
        cls,
        store: MemoryStore,
        turn_id: int,
    ) -> Optional[TurnHistoryTracker]:
        """Hydrate a read-only tracker view from an existing SQLite turn."""
        row = store.reconstruct_turn_history(turn_id)
        if row is None:
            return None

        tracker = cls(store, row["session_id"])
        tracker._turn_id = int(row["turn_id"])
        tracker._user_query = row["user_query"]
        tracker._search_queries = list(row.get("search_queries") or [])
        tracker._urls_opened = list(row.get("urls_opened") or [])
        tracker._snippets = list(row.get("context_snippets") or [])
        tracker._plan = row.get("plan")
        tracker._final_answer = row.get("final_answer")
        tracker._closed = tracker._final_answer is not None

        audit = row.get("audit_trail") or {}
        tracker._root_transaction_id = audit.get("root_transaction_id")
        for event in audit.get("events") or []:
            tracker._events.append(
                AuditEvent(
                    transaction_id=event["transaction_id"],
                    event_type=event["event_type"],
                    timestamp=event["timestamp"],
                    payload=event.get("payload") or {},
                )
            )
        tracker._context_refs = list(
            (audit.get("extraction") or {}).get("contexts") or []
        )
        if not tracker._context_refs and row.get("contexts"):
            for ctx in row["contexts"]:
                meta = ctx.get("metadata") or {}
                tracker._context_refs.append(
                    {
                        "context_id": ctx["context_id"],
                        "url": ctx["url"],
                        "title": ctx.get("title"),
                        "domain": ctx.get("domain"),
                        "transaction_id": meta.get("transaction_id"),
                        "text_preview": (ctx.get("text_block") or "")[:200],
                    }
                )
        return tracker

    def _build_audit_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self._turn_id,
            "session_id": self._session_id,
            "root_transaction_id": self._root_transaction_id,
            "user_query": self._user_query,
            "events": [event.to_dict() for event in self._events],
            "search": {
                "queries": list(self._search_queries),
                "urls_accessed": list(self._urls_opened),
            },
            "extraction": {
                "contexts": list(self._context_refs),
                "snippets": list(self._snippets),
            },
            "response": {
                "final_answer": self._final_answer,
            },
        }

    def _append_event(
        self,
        event_type: str,
        transaction_id: str,
        payload: dict[str, Any],
    ) -> None:
        self._events.append(
            AuditEvent(
                transaction_id=transaction_id,
                event_type=event_type,
                timestamp=_utc_now_iso(),
                payload=deepcopy(payload),
            )
        )

    def _persist_audit_trail(self) -> None:
        assert self._turn_id is not None
        self._store.save_audit_trail(self._turn_id, self._build_audit_dict())

    def _ensure_open(self) -> None:
        if self._turn_id is None:
            raise RuntimeError("Call begin() before recording turn events")
        if self._closed:
            raise RuntimeError("Turn is finalized; audit trail is immutable")
