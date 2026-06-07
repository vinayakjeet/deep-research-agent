#!/usr/bin/env python3
"""Run evaluation harness against dataset.json."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from deep_research_agent.eval.loader import filter_cases, load_dataset
from deep_research_agent.eval.report import aggregate_results, write_report
from deep_research_agent.eval.runner import run_case
from deep_research_agent.memory.store import MemoryStore
from deep_research_agent.orchestration.orchestrator import ResearchOrchestrator


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate deep research agent against dataset.json")
    parser.add_argument(
        "--dataset",
        default="dataset.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_run.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--db",
        default="data/eval_agent.db",
        help="SQLite database path for eval sessions",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated case IDs (e.g. TC_001,TC_021)",
    )
    parser.add_argument(
        "--labels",
        help="Comma-separated labels; cases matching any label are included",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max number of cases to run after filtering",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge calls (faster, cheaper)",
    )
    parser.add_argument(
        "--delay-secs",
        type=float,
        default=4.0,
        help="Seconds to wait between cases to avoid Gemini rate limits (default: 4)",
    )
    args = parser.parse_args(argv)

    _load_dotenv()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    data = load_dataset(dataset_path)
    cases = filter_cases(
        data["test_cases"],
        ids=_parse_csv_list(args.ids),
        labels=_parse_csv_list(args.labels),
        limit=args.limit,
    )

    if not cases:
        print("No test cases matched filters.", file=sys.stderr)
        return 1

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(db_path)
    orchestrator = ResearchOrchestrator(store)

    case_labels = {c["id"]: c.get("labels") or [] for c in cases}
    results = []

    print(f"Running {len(cases)} case(s)...", flush=True)
    for index, case in enumerate(cases, start=1):
        case_id = case.get("id", f"case_{index}")
        print(f"[{index}/{len(cases)}] {case_id}: {case.get('query', '')[:60]}...", flush=True)

        forbidden = None
        if case_id == "TC_045":
            forbidden = ["NCBS", "Bangalore", "National Centre for Biological Sciences", "Priya"]

        result = run_case(
            case,
            store=store,
            orchestrator=orchestrator,
            prior_queries=case.get("prior_turns") or [],
            run_judge_flag=not args.skip_judge,
            forbidden_terms=forbidden,
        )
        results.append(result)

        behavior_ok = result.behavior.passed if result.behavior else False
        stream_ok = result.streaming.passed if result.streaming else False
        f1 = result.retrieval.f1 if result.retrieval and result.retrieval.f1 is not None else "n/a"
        status = "ERROR" if result.error else ("PASS" if behavior_ok and stream_ok else "FAIL")
        print(f"  -> {status} | behavior={behavior_ok} streaming={stream_ok} f1={f1} ({result.duration_sec}s)", flush=True)

        if index < len(cases) and args.delay_secs > 0:
            time.sleep(args.delay_secs)

    aggregate = aggregate_results(results, case_labels)
    out = write_report(
        results,
        aggregate,
        args.output,
        metadata={
            "dataset": str(dataset_path),
            "cases_run": len(cases),
            "skip_judge": args.skip_judge,
        },
    )

    print("\n--- Summary ---")
    print(f"Cases: {aggregate.total_cases} | Errors: {aggregate.errors}")
    print(f"Behavior pass rate: {aggregate.behavior_pass_rate}")
    print(f"Streaming pass rate: {aggregate.streaming_pass_rate}")
    print(f"Retrieval mean F1: {aggregate.retrieval_mean_f1}")
    if aggregate.judge_behavior_pass_rate is not None:
        print(f"Judge behavior pass rate: {aggregate.judge_behavior_pass_rate}")
    print(f"Confident hallucinations: {aggregate.confident_hallucination_count}")
    print(f"Report: {out}")
    print(f"CSV:   {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
