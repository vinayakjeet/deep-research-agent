# Deep Research Agent

A research agent built from scratch in Python — no LangChain, no LangGraph, no CrewAI. I wrote the agent loop, orchestration, streaming, and eval harness myself.

The agent takes a question, searches the web for relevant pages, reads them, picks the most useful bits, and writes an answer that cites only sources it actually opened. If it can't find anything useful, it says so instead of making something up.

---

## Table of contents

1. [Design note](#design-note)
2. [Setup](#setup)
3. [How to run it](#how-to-run-it)
4. [How it works internally](#how-it-works-internally)
5. [Example sessions](#example-sessions)
6. [Evaluation](#evaluation)
7. [Limitations and what I'd fix next](#limitations-and-what-id-fix-next)
8. [Assumptions I made](#assumptions-i-made)
9. [Video demo](#video-demo)

---

## Design note

### Who this is for and what problem it solves

Researchers and analysts who need a cited, verifiable answer to a question — not a chatbot guess. The pain point is simple: generic LLMs confidently answer from training data, which may be stale or wrong, and they don't tell you where they got it. This agent fetches sources live, only uses what it finds, and explicitly lists where every claim came from.

Secondary use case: exploratory questions where you're not sure what sources exist. The agent searches, skims, and tells you what it found (or didn't find).

### What "deep research" means in this implementation

I defined it as: search, read the actual pages, pick the relevant parts, write a grounded answer — for every single query, not just first load.

Concretely the agent runs PLAN → SEARCH → ACQUIRE → SELECT → ANSWER on every turn:

- **PLAN**: given the user's question and any prior conversation, Gemini figures out 1–N search queries to issue
- **SEARCH**: Tavily (or Serper) runs those queries and returns title, URL, snippet
- **ACQUIRE**: the agent fetches the actual HTML for the top results and strips it down to readable text
- **SELECT**: chunks each page and scores them against the query using TF overlap + recency + domain diversity, trimmed to a token budget
- **ANSWER**: generates a response using only the selected snippets, citing the source URLs inline
- **FALLBACK**: if SEARCH or ACQUIRE turns up nothing useful, the agent returns a hardcoded "insufficient evidence" message without calling the LLM at all — this is how it avoids hallucinating when sources don't exist

It keeps session history in SQLite so follow-up questions can use prior context.

### Success metrics

I picked five metrics that together tell you whether the agent is actually doing research vs. just talking:

**Citation integrity** — what fraction of the URLs cited in the answer were actually fetched during that turn. If this isn't 100%, the agent cited something it never opened, which means it hallucinated a source.

**Retrieval F1** — do the URLs (or at least the domains) the agent fetched match the ground truth URLs I curated? Measures whether it found the right sources. I use domain-level fallback because exact URLs change.

**Behavior conformance** — did the agent do what the situation called for? Answering when it has evidence, refusing when it doesn't, flagging conflict when sources disagree. A lot of research eval ignores this but it's the most practically important one.

**Streaming coverage** — did all the required status events fire in the right order? The UI depends on this and it's a contract I need to hold.

**LLM judge score** — a separate Gemini call that reads the answer and checks whether it actually contains the expected facts. Catches cases where the agent cited the right URL but still gave a wrong or vague answer.

### Data flow

```
user question
     |
   PLAN  ──►  SEARCH  ──►  ACQUIRE  ──►  SELECT  ──►  ANSWER  ──►  VALIDATE
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

Module breakdown:

| Stage | Files |
|-------|-------|
| Orchestrator loop | `orchestration/orchestrator.py` |
| Planner | `orchestration/planner.py` |
| Search providers | `ingestion/search/tavily.py`, `ingestion/search/serper.py` |
| Page fetching | `ingestion/fetch/page_fetcher.py` |
| HTML → text | `ingestion/parse/html_extractor.py` |
| Chunk + select | `ingestion/select/chunker.py`, `ingestion/select/tf_selector.py` |
| Context budget | `context/context_builder.py`, `context/tokens.py` |
| LLM client | `llm/gemini_client.py` |
| Citations | `orchestration/citations.py` |
| Fallbacks | `orchestration/fallbacks.py` |
| Streaming events | `orchestration/events.py` |
| SQLite memory | `memory/store.py`, `memory/turn_tracker.py` |
| State schema | `state/schema.py`, `state/enums.py` |
| Evaluation | `eval/runner.py`, `eval/metrics.py`, `eval/judge.py` |

### Risks I ran into

**Rate limits** hit immediately when running eval — Gemini free tier is 20 requests/day. With 2+ LLM calls per case across 50 cases, the full eval can't run in one shot. I built a `--skip-judge` flag and `--delay-secs` option for batching.

**Low-quality pages** — a lot of search results are SEO junk or JS-only. `readability-lxml` + `selectolax` gets most of it but fails on SPA-style pages with no server-rendered content. I filter out extractions below a character threshold.

**Conflicting sources** — I handled this at the prompt level (instructing the model to note disagreement and cite both) rather than doing anything algorithmic. It works inconsistently.

**Context length** — long conversations can overflow the budget I set. There's an extractive summarizer fallback in `context/context_builder.py` but it's basic.

### Two things I'd add with more time

The two that would actually matter:

1. **Re-search on weak retrieval** — right now the planner runs once. If the snippets it gets back don't really answer the question, the agent still tries to write an answer anyway. A second search pass triggered when snippet quality is low would fix a lot of the F1 failures I saw.

2. **Sentence-level citations** — currently the agent cites at the URL level. It'd be much better to pin each specific claim to the chunk it came from. That also makes conflict detection tractable (you can compare claims from different sources directly).

---

## Setup

```bash
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
GEMINI_MODEL=gemini-3.5-flash
```

> The `.env.example` in this repo has example-format keys. Don't commit real keys.

---

## How to run it

### Streamlit UI

```bash
streamlit run deep_research_agent/ui/streamlit_app.py
```

Opens in browser. Type a question, watch the phase updates stream in as it works.

### FastAPI (SSE streaming)

```bash
uvicorn deep_research_agent.api.app:app --reload
```

```bash
# create a session
curl -X POST http://127.0.0.1:8000/sessions

# stream a research query
curl -N -X POST http://127.0.0.1:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id from above>", "query": "What happened at ISRO this year?"}'
```

### CLI (NDJSON per event)

```bash
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

---

## How it works internally

Each user turn follows the same steps:

1. **Plan** — `Planner` sends the question + recent conversation to Gemini and gets back a short strategy + list of search queries to issue.

2. **Search** — `IngestionPipeline` runs each query through Tavily (or Serper). Deduplicates results by URL.

3. **Acquire** — `PageFetcher` downloads the HTML for the top results concurrently (aiohttp, up to 8 at once). `HtmlExtractor` strips navigation, ads, boilerplate using readability-lxml.

4. **Select** — `select_context_for_query` splits each page into overlapping chunks, scores them by TF overlap with the query, and picks the top chunks up to the token budget. It tries to keep at least one chunk per unique domain.

5. **Answer** — `AnswerGenerator` builds the prompt from (a) a rolling summary of the conversation so far and (b) the selected snippets formatted as `[Title — domain]: snippet text`. Gemini writes the answer.

6. **Validate** — `validate_citations` checks every URL cited in the answer against the set of URLs actually fetched this turn. Anything that wasn't fetched is flagged in `result.citation_report`.

7. **Fallback** — if step 2 or 3 returns nothing usable, the whole LLM answer step is skipped and the orchestrator returns a fixed no-evidence response. This path can't hallucinate by definition.

**Persistence**: every step is written to SQLite. The `TurnHistoryTracker` records the plan, which queries were issued, which URLs were opened, which snippets were selected, and the final answer — all under a single `turn_id`. You can reconstruct any turn's full audit trail with `tracker.to_audit_dict()`.

**Streaming**: the orchestrator is a generator that yields structured event dicts at each phase transition. The same generator backs the CLI, FastAPI SSE, and Streamlit — no duplication.

---

## Example sessions

**Simple factual question**

```
User: What is the capital of Bhutan?

[planning]  Planning: search for "capital of Bhutan"
[searching] Querying Tavily...
[fetching]  Fetching 3 pages
[generating] Building answer from 4 snippets

Answer: Thimphu is the capital of Bhutan.
Sources: [Thimphu — Wikipedia](https://en.wikipedia.org/wiki/Thimphu)
```

**Multi-entity comparison**

```
User: Compare the solar energy capacity of India and China as of 2024.

The planner issues two queries — "India solar capacity 2024" and "China solar
capacity 2024" — fetches results from both, and the answer puts the figures
side by side with citations from each country's official/news sources.
```

**Nothing found**

```
User: What did the founder of Acme Corp say at the private board meeting last week?

[no_evidence] No verifiable sources found for this query.

Answer: I wasn't able to find any public sources covering this. If this was
a private meeting, it's unlikely to be on the open web. Suggested next steps:
check the company's press releases or investor filings.
```

No LLM call happens on this path — the response is deterministic.

---

## Evaluation

### Dataset

`dataset.json` — 50 hand-written test cases covering: single-hop factual, multi-hop, comparison, insufficient evidence, conflicting sources, false premise, India-specific queries, multi-turn conversations, and streaming contract checks. Each case has `expected_facts`, `ground_truth_urls`, `acceptable_domains`, and an `expected_behavior` label.

### Running it

```bash
# full run (needs paid Gemini or batching across days):
python evaluate.py --dataset dataset.json --delay-secs 5 \
  --output reports/full_run.json

# quick 5-case smoke test (free tier):
python evaluate.py --dataset dataset.json \
  --ids TC_001,TC_021,TC_026,TC_045,TC_050 \
  --skip-judge --output reports/pilot.json
```

### What gets measured

- **Behavior conformance** — did the agent answer when it should, refuse when it should, acknowledge conflict when it should?
- **Retrieval F1** — URL-level with domain-level fallback
- **Citation integrity** — no citing URLs that weren't opened
- **Streaming coverage** — required events in order
- **LLM judge score** — separate Gemini call checking factual accuracy (1–5)

### Results

See `EVAL_REPORT.md` for full tables. Short version from the 5-case pilot:

| | |
|--|--|
| Behavior pass rate | 75% (3/4 scored — TC_021 hit a transient 429) |
| Streaming coverage | 100% |
| Retrieval F1 (mean) | 0.40 |
| Judge pass rate (3 full cases) | 100% |

**What worked**: single-hop factual (TC_001 correct, citation intact), cross-session isolation (TC_045), streaming telemetry.

**What didn't**: TC_026 — the agent should have refused but instead scraped together a low-quality answer from weak snippets. TC_050 — 3-company comparison had F1 of 0.13 because the planner issued one combined query instead of one per company.

The full 50-case run hit Gemini's 20 req/day free tier limit after about 3 cases. To run it fully you need billing or ~8-case daily batches.

---

## Limitations and what I'd fix next

Things I'm aware of that aren't fixed:

- The insufficient-evidence check is too loose. Right now it triggers only when SEARCH returns zero results. It should also trigger when snippets are there but clearly off-topic.
- No JS-rendered page support. Pages that need a browser to render come back empty.
- Conflict detection is purely prompt-based — I tell the model to note disagreement, but there's no automated cross-source comparison.
- Multi-entity queries confuse the single-query planner. Needs explicit decomposition per entity.
- Free-tier eval limit means the 50-case results are partial. The methodology is sound; the execution was rate-limited.

What I'd build next: iterative re-search when the first pass retrieval is weak, and sentence-level citation mapping rather than URL-level.

---

## Assumptions I made

- **Gemini free tier**: I used `gemini-3.5-flash` because it's free and available. The tradeoff is the 20-request/day cap which bottlenecks eval. Any other API-accessible LLM would work — the client is in `llm/gemini_client.py` and the rest of the code doesn't care what model is behind it.
- **Single planning pass**: the assignment says "produce a plan", not "replan iteratively". I added one optional replan on zero search results but stopped there.
- **English only**: no multilingual handling. The INDIA_LOCAL eval cases all use English-language sources.
- **SQLite**: fine for a single-node prototype. Nothing in the code forces this — swap `memory/store.py` for Postgres and nothing else needs to change.
- **Tavily as default**: higher quality snippets in my testing vs. Serper raw results. Serper is there as a fallback.

---

## Video demo

**Link**: _paste your Loom/Drive/YouTube URL here before submitting_

Demo script (3 queries that cover the main cases):

```bash
streamlit run deep_research_agent/ui/streamlit_app.py
```

1. `What is the capital of Bhutan?` — single-hop, shows streaming phases, citation
2. `Compare solar energy capacity of India and China in 2024` — multi-query planning
3. `What did the CEO of Acme Industries say at the private board meeting yesterday?` — insufficient evidence path

---

## Running tests

```bash
python -m unittest discover -s tests -v
```

24 test modules covering memory, orchestration, ingestion, citations, streaming, eval harness, and the API.
