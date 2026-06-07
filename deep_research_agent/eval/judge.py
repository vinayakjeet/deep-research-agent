"""Optional LLM-as-judge for expected_facts scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from deep_research_agent.llm.gemini_client import GeminiClient
from deep_research_agent.llm.json_extract import extract_json_object


JUDGE_SYSTEM = """You are an evaluation judge for a research agent.
Score the agent answer against expected facts and expected behavior.
Respond with ONLY valid JSON:
{
  "relevancy_score": 1-5,
  "specificity_score": 1-5,
  "attribution_score": 1-5,
  "facts_matched": ["fact1", "fact2"],
  "facts_missing": ["fact3"],
  "behavior_pass": true,
  "reasoning": "brief explanation"
}
"""


@dataclass(slots=True)
class JudgeScore:
    relevancy_score: int
    specificity_score: int
    attribution_score: int
    facts_matched: list[str]
    facts_missing: list[str]
    behavior_pass: bool
    reasoning: str
    confident_hallucination: bool = False
    skipped: bool = False
    error: str = ""


def run_judge(
    *,
    query: str,
    answer: str,
    expected_facts: list[str],
    expected_behavior: str,
    ideal_answer_sketch: str = "",
    client: Optional[GeminiClient] = None,
) -> JudgeScore:
    """Call Gemini to score answer quality."""
    if not answer.strip():
        return JudgeScore(
            relevancy_score=1,
            specificity_score=1,
            attribution_score=1,
            facts_matched=[],
            facts_missing=list(expected_facts),
            behavior_pass=False,
            reasoning="Empty answer.",
        )

    user_content = (
        f"Query: {query}\n\n"
        f"Expected behavior: {expected_behavior}\n"
        f"Expected facts: {json.dumps(expected_facts)}\n"
        f"Ideal answer sketch: {ideal_answer_sketch}\n\n"
        f"Agent answer:\n{answer}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    try:
        llm = client or GeminiClient()
        raw = llm.generate_text(messages)
        data = extract_json_object(raw)
    except Exception as exc:
        return JudgeScore(
            relevancy_score=0,
            specificity_score=0,
            attribution_score=0,
            facts_matched=[],
            facts_missing=list(expected_facts),
            behavior_pass=False,
            reasoning="",
            skipped=True,
            error=str(exc),
        )

    rel = int(data.get("relevancy_score", 1))
    spec = int(data.get("specificity_score", 1))
    attr = int(data.get("attribution_score", 1))
    behavior_pass = bool(data.get("behavior_pass", False))
    confident_hallucination = spec >= 4 and attr <= 2

    return JudgeScore(
        relevancy_score=rel,
        specificity_score=spec,
        attribution_score=attr,
        facts_matched=[str(f) for f in (data.get("facts_matched") or [])],
        facts_missing=[str(f) for f in (data.get("facts_missing") or [])],
        behavior_pass=behavior_pass,
        reasoning=str(data.get("reasoning", "")).strip(),
        confident_hallucination=confident_hallucination,
    )
