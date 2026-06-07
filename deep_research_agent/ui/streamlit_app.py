"""Streamlit UI for deep research with live status streaming."""
import sys
sys.path.insert(0, "/mount/src/deep-research-agent")
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.orchestration.orchestrator import ResearchOrchestrator


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


@st.cache_resource
def get_store() -> MemoryStore:
    db = Path(os.environ.get("DEEP_RESEARCH_DB", "deep_research.db"))
    return MemoryStore(db)


def main() -> None:
    _load_dotenv()
    st.set_page_config(page_title="Deep Research Agent", layout="wide")
    st.title("Deep Research Agent")

    store = get_store()
    if "session_id" not in st.session_state:
        st.session_state.session_id = store.create_session()["session_id"]

    query = st.text_input("Research question", placeholder="Compare GDP growth in India and China")
    run = st.button("Run research", type="primary")

    if run and query.strip():
        orchestrator = ResearchOrchestrator(store)
        status = st.status("Running research...", expanded=True)
        final_answer = ""
        citation_report = None

        for event in orchestrator.run_stream(st.session_state.session_id, query.strip()):
            label = event.get("status") or event.get("event", "update")
            message = event.get("message", "")
            status.write(f"**{label}** — {message}")

            if event.get("event") == "complete":
                final_answer = event.get("final_answer", "")
                citation_report = event.get("details", {}).get("citation_report")

        status.update(label="Complete", state="complete", expanded=False)

        if final_answer:
            st.subheader("Answer")
            st.markdown(final_answer)

        if citation_report and citation_report.get("hallucination_flags"):
            with st.expander("Citation warnings", expanded=True):
                for flag in citation_report["hallucination_flags"]:
                    st.warning(flag)


if __name__ == "__main__":
    main()
