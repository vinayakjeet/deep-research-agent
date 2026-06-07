"""Native SQLite episodic memory store for sessions, turns, and contexts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator, Optional


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row into a plain column-name dict."""
    return {key: row[key] for key in row.keys()}


def _json_dumps(value: Any) -> str:
    """Serialize a Python value to a JSON string, preserving Unicode."""
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any = None) -> Any:
    """Parse a JSON string, or return default when the value is None."""
    if value is None:
        return default
    return json.loads(value)


class MemoryStore:
    """Persistent relational store for research agent session state."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path) -> None:
        """Bind to a SQLite database file and ensure schema tables exist."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a connection, commit on success, rollback on error, then close."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables, indexes, and record schema version on first run."""
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    search_queries_json TEXT,
                    urls_opened_json TEXT,
                    context_snippets_json TEXT,
                    final_answer TEXT,
                    plan_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts (
                    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    domain TEXT,
                    text_block TEXT,
                    retrieved_at TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (turn_id) REFERENCES turns (turn_id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_contexts_turn ON contexts (turn_id)"
            )

            self._apply_migrations(conn)

            conn.execute(
                """
                INSERT OR IGNORE INTO schema_meta (key, value)
                VALUES ('schema_version', ?)
                """,
                (str(self.SCHEMA_VERSION),),
            )

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply additive schema migrations for existing databases."""
        turn_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(turns)").fetchall()
        }
        if "audit_trail_json" not in turn_columns:
            conn.execute("ALTER TABLE turns ADD COLUMN audit_trail_json TEXT")

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Insert a new session row and return its parsed record."""
        sid = session_id or str(uuid.uuid4())
        now = _utc_now_iso()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (sid, now, now, _json_dumps(metadata) if metadata else None),
            )
        return self.get_session(sid)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Fetch one session by ID, or None if it does not exist."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        data = _row_to_dict(row)
        data["metadata"] = _json_loads(data.pop("metadata_json"))
        return data

    def list_sessions(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return sessions ordered by most recently updated, with pagination."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            data = _row_to_dict(row)
            data["metadata"] = _json_loads(data.pop("metadata_json"))
            results.append(data)
        return results

    def touch_session(self, session_id: str) -> None:
        """Bump a session's updated_at timestamp to the current UTC time."""
        now = _utc_now_iso()
        with self._connection() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its related rows; return True if one was removed."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Conversation messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append a chat message to a session and refresh its updated_at."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid role: {role}")
        ts = created_at or _utc_now_iso()
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, ts),
            )
            message_id = cursor.lastrowid
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (ts, session_id),
            )
        return self.get_message(int(message_id))  # type: ignore[return-value]

    def get_message(self, message_id: int) -> Optional[dict[str, Any]]:
        """Fetch one message by ID, or None if it does not exist."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return a session's messages in chronological order, optionally capped."""
        query = """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC, message_id ASC
        """
        params: list[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Research turns
    # ------------------------------------------------------------------

    def create_turn(
        self,
        session_id: str,
        user_query: str,
        search_queries: Optional[list[str]] = None,
        urls_opened: Optional[list[str]] = None,
        context_snippets: Optional[list[dict[str, Any]]] = None,
        final_answer: Optional[str] = None,
        plan: Optional[dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """Insert a research turn row and return its parsed record."""
        ts = created_at or _utc_now_iso()
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO turns (
                    session_id, user_query, search_queries_json, urls_opened_json,
                    context_snippets_json, final_answer, plan_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_query,
                    _json_dumps(search_queries) if search_queries else None,
                    _json_dumps(urls_opened) if urls_opened else None,
                    _json_dumps(context_snippets) if context_snippets else None,
                    final_answer,
                    _json_dumps(plan) if plan else None,
                    ts,
                ),
            )
            turn_id = int(cursor.lastrowid)
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (ts, session_id),
            )
        return self.get_turn(turn_id)  # type: ignore[return-value]

    def get_turn(self, turn_id: int) -> Optional[dict[str, Any]]:
        """Fetch one turn by ID with JSON fields deserialized, or None."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
        if row is None:
            return None
        return self._deserialize_turn(_row_to_dict(row))

    def get_turns(self, session_id: str) -> list[dict[str, Any]]:
        """Return all turns for a session in chronological order."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM turns
                WHERE session_id = ?
                ORDER BY created_at ASC, turn_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._deserialize_turn(_row_to_dict(row)) for row in rows]

    def update_turn(
        self,
        turn_id: int,
        *,
        search_queries: Optional[list[str]] = None,
        urls_opened: Optional[list[str]] = None,
        context_snippets: Optional[list[dict[str, Any]]] = None,
        final_answer: Optional[str] = None,
        plan: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Patch only the provided turn fields and return the updated record."""
        fields: list[str] = []
        values: list[Any] = []

        if search_queries is not None:
            fields.append("search_queries_json = ?")
            values.append(_json_dumps(search_queries))
        if urls_opened is not None:
            fields.append("urls_opened_json = ?")
            values.append(_json_dumps(urls_opened))
        if context_snippets is not None:
            fields.append("context_snippets_json = ?")
            values.append(_json_dumps(context_snippets))
        if final_answer is not None:
            fields.append("final_answer = ?")
            values.append(final_answer)
        if plan is not None:
            fields.append("plan_json = ?")
            values.append(_json_dumps(plan))

        if not fields:
            return self.get_turn(turn_id)

        values.append(turn_id)
        with self._connection() as conn:
            conn.execute(
                f"UPDATE turns SET {', '.join(fields)} WHERE turn_id = ?",
                values,
            )
        return self.get_turn(turn_id)

    def save_audit_trail(self, turn_id: int, audit_trail: dict[str, Any]) -> None:
        """Persist the nested turn audit trail JSON for a turn."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE turns SET audit_trail_json = ? WHERE turn_id = ?",
                (_json_dumps(audit_trail), turn_id),
            )

    def _deserialize_turn(self, data: dict[str, Any]) -> dict[str, Any]:
        """Rename JSON columns on a turn row and parse them into Python values."""
        data["search_queries"] = _json_loads(data.pop("search_queries_json"), [])
        data["urls_opened"] = _json_loads(data.pop("urls_opened_json"), [])
        data["context_snippets"] = _json_loads(data.pop("context_snippets_json"), [])
        data["plan"] = _json_loads(data.pop("plan_json"))
        data["audit_trail"] = _json_loads(data.pop("audit_trail_json"))
        return data

    # ------------------------------------------------------------------
    # Context blocks (per-turn fetched source material)
    # ------------------------------------------------------------------

    def add_context(
        self,
        turn_id: int,
        url: str,
        title: Optional[str] = None,
        domain: Optional[str] = None,
        text_block: Optional[str] = None,
        retrieved_at: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Insert one fetched source context block for a turn."""
        ts = retrieved_at or _utc_now_iso()
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO contexts (
                    turn_id, url, title, domain, text_block, retrieved_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    url,
                    title,
                    domain,
                    text_block,
                    ts,
                    _json_dumps(metadata) if metadata else None,
                ),
            )
            context_id = int(cursor.lastrowid)
        return self.get_context(context_id)  # type: ignore[return-value]

    def add_contexts_batch(
        self,
        turn_id: int,
        contexts: Iterator[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert multiple context blocks for a turn in a single transaction."""
        inserted: list[dict[str, Any]] = []
        with self._connection() as conn:
            for ctx in contexts:
                ts = ctx.get("retrieved_at") or _utc_now_iso()
                metadata = ctx.get("metadata")
                cursor = conn.execute(
                    """
                    INSERT INTO contexts (
                        turn_id, url, title, domain, text_block, retrieved_at, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_id,
                        ctx["url"],
                        ctx.get("title"),
                        ctx.get("domain"),
                        ctx.get("text_block"),
                        ts,
                        _json_dumps(metadata) if metadata else None,
                    ),
                )
                context_id = int(cursor.lastrowid)
                row = conn.execute(
                    "SELECT * FROM contexts WHERE context_id = ?",
                    (context_id,),
                ).fetchone()
                if row:
                    data = _row_to_dict(row)
                    data["metadata"] = _json_loads(data.pop("metadata_json"))
                    inserted.append(data)
        return inserted

    def get_context(self, context_id: int) -> Optional[dict[str, Any]]:
        """Fetch one context block by ID, or None if it does not exist."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM contexts WHERE context_id = ?",
                (context_id,),
            ).fetchone()
        if row is None:
            return None
        data = _row_to_dict(row)
        data["metadata"] = _json_loads(data.pop("metadata_json"))
        return data

    def get_contexts_for_turn(self, turn_id: int) -> list[dict[str, Any]]:
        """Return all source context blocks attached to a turn."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM contexts
                WHERE turn_id = ?
                ORDER BY context_id ASC
                """,
                (turn_id,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            data = _row_to_dict(row)
            data["metadata"] = _json_loads(data.pop("metadata_json"))
            results.append(data)
        return results

    def reconstruct_turn_history(self, turn_id: int) -> Optional[dict[str, Any]]:
        """Rebuild the full audit trail for a single research turn."""
        turn = self.get_turn(turn_id)
        if turn is None:
            return None
        turn["contexts"] = self.get_contexts_for_turn(turn_id)
        return turn

    def reconstruct_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Rebuild session with messages and full turn histories."""
        session = self.get_session(session_id)
        if session is None:
            return None
        session["messages"] = self.get_messages(session_id)
        turns = self.get_turns(session_id)
        for turn in turns:
            turn["contexts"] = self.get_contexts_for_turn(turn["turn_id"])
        session["turns"] = turns
        return session
