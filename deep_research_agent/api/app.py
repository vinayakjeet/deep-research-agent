"""FastAPI application with SSE research streaming."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.orchestration.orchestrator import ResearchOrchestrator


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


class ResearchStreamRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


def create_app(*, db_path: Optional[Path] = None) -> FastAPI:
    """Build the FastAPI app (db_path override for tests)."""
    _load_dotenv()
    resolved_db = db_path or Path(
        os.environ.get("DEEP_RESEARCH_DB", "deep_research.db")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = MemoryStore(resolved_db)
        yield

    app = FastAPI(title="Deep Research Agent", version="0.1.0", lifespan=lifespan)

    @app.post("/sessions")
    def create_session() -> dict[str, str]:
        store: MemoryStore = app.state.store
        session = store.create_session()
        return {"session_id": session["session_id"]}

    @app.post("/research/stream")
    def research_stream(body: ResearchStreamRequest) -> StreamingResponse:
        store: MemoryStore = app.state.store
        if store.get_session(body.session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")

        orchestrator = ResearchOrchestrator(store)

        def event_generator():
            try:
                for event in orchestrator.run_stream(body.session_id, body.query):
                    yield f"data: {json.dumps(event)}\n\n"
            except ValueError as exc:
                payload = {"event": "error", "message": str(exc)}
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
