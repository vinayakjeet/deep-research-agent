"""CLI entry point for streaming research to stdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.orchestration.orchestrator import ResearchOrchestrator


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deep research query with status streaming.")
    parser.add_argument("query", help="Research question")
    parser.add_argument(
        "--session-id",
        help="Existing session id (creates a new session if omitted)",
    )
    parser.add_argument(
        "--db",
        default="deep_research.db",
        help="SQLite database path",
    )
    args = parser.parse_args(argv)

    _load_dotenv()
    store = MemoryStore(Path(args.db))
    session_id = args.session_id or store.create_session()["session_id"]
    orchestrator = ResearchOrchestrator(store)

    for event in orchestrator.run_stream(session_id, args.query):
        print(json.dumps(event), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
