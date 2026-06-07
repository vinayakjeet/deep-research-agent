# Deep Research Agent

A research agent built from scratch in Python for searching the web, reading sources, and writing answers that cite only what it actually opened.

> **Note**
>
> This project was built without any agent framework. No LangChain, no LangGraph, no CrewAI. The agent loop, orchestration, streaming, and eval harness are all written by hand. It uses the Gemini free tier by default, which caps usage at 20 requests per day.
>
> *Note: AI tools (Cursor / Claude) were used as a coding aid during development.*

The agent takes a question, searches the web for relevant pages, reads them, picks the most useful parts, and writes an answer with inline citations. If it cannot find anything useful, it says so instead of making something up.

## Features

* **Live web research**: searches with Tavily or Serper, fetches the actual pages, and reads them on every query.
* **Grounded answers only**: cites only URLs that were fetched during the turn. Citations are validated after every answer.
* **No-hallucination fallback**: if search or fetch returns nothing usable, it returns a fixed no-evidence response without ever calling the LLM.
* **Streaming status**: each phase (plan, search, fetch, select, answer) streams as a structured event. The same generator backs the CLI, the API, and the UI.
* **Session memory**: SQLite stores sessions, turns, snippets, queries, and a full audit trail so follow-up questions keep context.
* **Three ways to run**: Streamlit UI, FastAPI with SSE streaming, or a CLI.
* **Built-in evaluation**: a 50-case dataset with metrics for citation integrity, retrieval F1, behavior conformance, streaming coverage, and an LLM judge.

## How it works

The agent runs the same loop on every turn:

```
user question
     |
   PLAN  -->  SEARCH  -->  ACQUIRE  -->  SELECT  -->  ANSWER  -->  VALIDATE
  (Gemini)  (Tavily/     (aiohttp +    (TF chunks   (Gemini +    (check cited
            Serper)      readability)   + budget)    snippets)    URLs exist)
                                            |
                                     SQLite store
                              (sessions, turns, snippets,
                               search queries, audit trail)
                                            |
                              streamed status events
                          (CLI / FastAPI SSE / Streamlit)
```

* **PLAN**: Gemini reads the question and any prior conversation and produces 1 to N search queries.
* **SEARCH**: Tavily or Serper runs those queries and returns title, URL, and snippet. Results are deduplicated by URL.
* **ACQUIRE**: the top results are fetched concurrently (aiohttp, up to 8 at once) and stripped to readable text with readability-lxml.
* **SELECT**: each page is chunked and scored against the query by term-frequency overlap, trimmed to a token budget, keeping at least one chunk per domain.
* **ANSWER**: Gemini writes the answer using only the selected snippets, citing source URLs inline.
* **VALIDATE**: every cited URL is checked against the URLs actually fetched this turn. Anything not fetched is flagged.
* **FALLBACK**: if search or acquire returns nothing usable, the LLM step is skipped and a fixed no-evidence response is returned. This path cannot hallucinate by definition.

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
SEARCH_PROVIDER=tavily        # tavily or serper
TAVILY_API_KEY=...
SERPER_API_KEY=...            # only needed if using serper
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

## Usage

### Streamlit UI

```
streamlit run deep_research_agent/ui/streamlit_app.py
```

Opens in the browser. Type a question and watch the phase updates stream in as it works.

### FastAPI (SSE streaming)

```
uvicorn deep_research_agent.api.app:app --reload

# create a session
curl -X POST http://127.0.0.1:8000/sessions

# stream a research query
curl -N -X POST http://127.0.0.1:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id from above>", "query": "What happened at ISRO this year?"}'
```

### CLI (NDJSON per event)

```
python -m deep_research_agent.cli "Compare solar capacity of India and China"
```

### Python

```python
from deep_research_agent.memory import MemoryStore
from deep_research_agent.orchestration import ResearchOrchestrator

store = MemoryStore("data/research_agent.db")
session = store.create_session()
orch = ResearchOrchestrator(store)

# streaming
for event in orch.run_stream(session["session_id"], "your question"):
    print(event["status"], event.get("message"))

# or single result
result = orch.run(session["session_id"], "your question")
print(result.final_answer)
```

## Architecture

| Stage | Files |
|-------|-------|
| Orchestrator loop | `orchestration/orchestrator.py` |
| Planner | `orchestration/planner.py` |
| Search providers | `ingestion/search/tavily.py`, `ingestion/search/serper.py` |
| Page fetching | `ingestion/fetch/page_fetcher.py` |
| HTML to text | `ingestion/parse/html_extractor.py` |
| Chunk and select | `ingestion/select/chunker.py`, `ingestion/select/tf_selector.py` |
| Context budget | `context/context_builder.py`, `context/tokens.py` |
| LLM client | `llm/gemini_client.py` |
| Citations | `orchestration/citations.py` |
| Fallbacks | `orchestration/fallbacks.py` |
| Streaming events | `orchestration/events.py` |
| SQLite memory | `memory/store.py`, `memory/turn_tracker.py` |
| State schema | `state/schema.py`, `state/enums.py` |
| Evaluation | `eval/runner.py`, `eval/metrics.py`, `eval/judge.py` |

Every step is written to SQLite. The `TurnHistoryTracker` records the plan, the queries issued, the URLs opened, the snippets selected, and the final answer, all under a single `turn_id`. You can reconstruct any turn's full audit trail with `tracker.to_audit_dict()`.

## Example sessions

Simple factual question:

```
User: What is the capital of Bhutan?

[planning]   Planning: search for "capital of Bhutan"
[searching]  Querying Tavily...
[fetching]   Fetching 3 pages
[generating] Building answer from 4 snippets

Answer: Thimphu is the capital of Bhutan.
Sources: [Thimphu - Wikipedia](https://en.wikipedia.org/wiki/Thimphu)
```

Nothing found:

```
User: What did the founder of Acme Corp say at the private board meeting last week?

[no_evidence] No verifiable sources found for this query.

Answer: I wasn't able to find any public sources covering this. If this was
a private meeting, it's unlikely to be on the open web.
```

No LLM call happens on this path. The response is deterministic.

## Evaluation

`dataset.json` has 50 hand-written test cases covering single-hop factual, multi-hop, comparison, insufficient evidence, conflicting sources, false premise, India-specific queries, multi-turn conversations, and streaming contract checks.

```
# full run (needs paid Gemini or batching across days):
python evaluate.py --dataset dataset.json --delay-secs 5 \
  --output reports/full_run.json

# quick 5-case smoke test (free tier):
python evaluate.py --dataset dataset.json \
  --ids TC_001,TC_021,TC_026,TC_045,TC_050 \
  --skip-judge --output reports/pilot.json
```

What gets measured:

* **Behavior conformance**: did the agent answer when it should, refuse when it should, acknowledge conflict when it should?
* **Retrieval F1**: URL-level with domain-level fallback.
* **Citation integrity**: no citing URLs that were not opened.
* **Streaming coverage**: required events in the right order.
* **LLM judge score**: a separate Gemini call checking factual accuracy on a 1 to 5 scale.

See `EVAL_REPORT.md` for full tables.

## Limitations

* The insufficient-evidence check is loose. It triggers only when search returns zero results. It should also trigger when snippets are off-topic.
* No support for JavaScript-rendered pages. Pages that need a browser to render come back empty.
* Conflict detection is prompt-based, with no automated cross-source comparison.
* Multi-entity queries confuse the single-query planner and need explicit decomposition per entity.
* Free-tier limits mean the 50-case eval results are partial.

Next steps would be iterative re-search when the first pass is weak, and sentence-level citation mapping instead of URL-level.

## Running tests

```
python -m unittest discover -s tests -v
```

24 test modules covering memory, orchestration, ingestion, citations, streaming, the eval harness, and the API.

## Video demo

https://github.com/user-attachments/assets/4491a3e1-04df-4c05-9106-38164ceada6b
