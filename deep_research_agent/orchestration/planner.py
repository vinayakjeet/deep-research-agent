"""Research planning via LLM with deterministic JSON extraction."""

from __future__ import annotations

from typing import Any, Optional

from deep_research_agent.llm.errors import PlanningError
from deep_research_agent.llm.gemini_client import GeminiClient
from deep_research_agent.llm.json_extract import extract_json_object
from deep_research_agent.llm.prompts import PLAN_SYSTEM_PROMPT
from deep_research_agent.state.schema import ResearchPlan


def parse_plan_response(raw: str) -> ResearchPlan:
    """Extract and validate a ResearchPlan from model output."""
    try:
        data = extract_json_object(raw)
    except ValueError as exc:
        raise PlanningError(f"Could not parse plan JSON: {exc}", cause=exc) from exc

    queries_raw = data.get("search_queries") or []
    if not isinstance(queries_raw, list):
        raise PlanningError("search_queries must be a list")

    search_queries = [str(q).strip() for q in queries_raw if str(q).strip()]
    if not search_queries:
        raise PlanningError("search_queries must be a non-empty list")

    steps_raw = data.get("steps") or []
    steps = [str(s).strip() for s in steps_raw if str(s).strip()] if isinstance(steps_raw, list) else []

    return ResearchPlan(
        summary=str(data.get("summary", "")).strip(),
        search_queries=search_queries,
        steps=steps,
    )


class Planner:
    """Generates structured research plans from user queries."""

    def __init__(self, client: Optional[GeminiClient] = None) -> None:
        self.client = client or GeminiClient()

    def plan(
        self,
        user_query: str,
        *,
        conversation_context: str = "",
    ) -> ResearchPlan:
        """Call the LLM and return a validated ResearchPlan."""
        user_content = f"User question:\n{user_query.strip()}"
        if conversation_context.strip():
            user_content += f"\n\nConversation context:\n{conversation_context.strip()}"

        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        raw = self.client.generate_text(messages)
        return parse_plan_response(raw)
