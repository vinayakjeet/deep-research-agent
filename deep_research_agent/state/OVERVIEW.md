# `state` Package — High-Level Overview

> **Goal of this doc:** Give you enough context to navigate the codebase without memorizing every line. Know *what* lives here, *where* to look, and *how* it connects.

For deep syntax-level detail, see [`STATE_GUIDE.md`](./STATE_GUIDE.md).

---

## What is this folder?

The `state` package defines **typed Python objects** for the research agent — sessions, messages, research turns, plans, search results, and the live workflow object the orchestrator uses.

It sits between:

```
SQLite (memory/store.py)  →  state/  →  orchestrator / context builder / LLM
     untyped dicts              typed objects
```

**One sentence:** DB rows become structured Python objects; those objects can be loaded, updated, serialized to JSON, and fed into prompts.

---

## File map — what & where

| File | What it does | When you care |
|------|--------------|---------------|
| `__init__.py` | Public exports — import from here | You need any state type or loader |
| `enums.py` | Workflow phases + chat roles | Agent step logic, message handling |
| `schema.py` | All data models (the core) | Understanding data shapes |
| `adapters.py` | Load typed objects from `MemoryStore` | Resuming sessions from DB |
| `serialization.py` | Dataclass ↔ JSON conversion | Saving/checkpointing state |

---

## Big picture

```
                    ┌─────────────────────────────────┐
                    │         schema.py               │
                    │  SessionRecord, TurnRecord,     │
                    │  AgentState, ResearchPlan, ...  │
                    └────────────┬────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
   enums.py               adapters.py           serialization.py
 AgentPhase              load from DB           to/from JSON
 MessageRole
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                          __init__.py
                      (single import surface)
```

---

## Core concepts (5-minute version)

### 1. Session
A conversation container. Has messages (chat) and turns (research logs).

### 2. Message
One chat line: user, assistant, or system. Stored in `messages` table.

### 3. Turn
One **research cycle** for a single user question — searches run, URLs opened, snippets picked, final answer. Stored in `turns` table.

### 4. Context vs snippet
| | Context (`SourceContext`) | Snippet (`ContextSnippet`) |
|---|---------------------------|----------------------------|
| **What** | Full fetched page | Small excerpt from a page |
| **Where stored** | `contexts` table | JSON on turn row |
| **Size** | Large (`text_block`) | Small (`snippet`) |

### 5. AgentState
The **live object** the orchestrator mutates while working — current phase, plan, search results, snippets, etc. Not everything on it is saved to DB (e.g. `search_results`, `error_message` are runtime-only).

### 6. AgentPhase
Where the agent is in its workflow:

```
START → PLANNING → SEARCHING → ACQUIRING → ANSWERING → COMPLETE
                                                      ↘ ERROR
```

---

## File-by-file (high level)

### `__init__.py`
**What:** Package entry point. Re-exports everything you should use.

**How to use:**
```python
from deep_research_agent.state import AgentState, load_agent_state, AgentPhase
```

Don't import from submodules unless you have a reason — start here.

---

### `enums.py`
**What:** Two enums.

| Enum | Purpose |
|------|---------|
| `AgentPhase` | Which step the agent is on (`PLANNING`, `SEARCHING`, …) |
| `MessageRole` | Chat role (`USER`, `ASSISTANT`, `SYSTEM`) |

**Key methods:**
- `AgentPhase.is_terminal()` — is the workflow done? (`COMPLETE` or `ERROR`)
- `MessageRole.from_str("user")` — DB string → typed enum

---

### `schema.py` ⭐ (main file)
**What:** All dataclass models. This is the heart of the package.

| Class | Role |
|-------|------|
| `SessionRecord` | One session (id, timestamps, metadata) |
| `MessageRecord` | One chat message |
| `TurnRecord` | Full audit of one research question |
| `ResearchPlan` | Planner output (summary, search queries, steps) |
| `SearchResult` | One search API hit (runtime only) |
| `ContextSnippet` | Small text excerpt for LLM |
| `SourceContext` | Full fetched page |
| `AgentState` | Live workflow state for the orchestrator |

**Patterns you'll see:**
- `from_store_row(dict)` — DB dict → typed object (most records)
- `from_dict(dict)` — JSON/nested dict → typed object (plan, snippets)
- `to_dict()` / `to_prompt_payload()` — object → dict for storage or LLM

**Most important class:** `AgentState` — what the agent loop reads and writes.

---

### `adapters.py`
**What:** Thin glue between `MemoryStore` and typed objects.

| Function | Returns |
|----------|---------|
| `load_session_record(store, session_id)` | `SessionRecord` or `None` |
| `load_messages(store, session_id)` | `list[MessageRecord]` |
| `load_turn_record(store, turn_id)` | `TurnRecord` with nested contexts |
| `load_agent_state(store, session_id)` | Full `AgentState` — **main resume entry point** |

**When to use:** Any time you're reading from the database and want typed objects instead of raw dicts.

```python
state = load_agent_state(store, session_id)
```

---

### `serialization.py`
**What:** Generic dataclass ↔ JSON helpers.

| Function | Direction |
|----------|-----------|
| `to_json(obj)` | dataclass → JSON string |
| `from_json(Class, payload)` | JSON → dataclass |
| `agent_state_to_json(state)` | `AgentState` → JSON |
| `agent_state_from_json(payload)` | JSON → `AgentState` |

**When to use:** Checkpointing state, logging, tests, or passing state over the wire — not needed for normal DB persistence (that's `MemoryStore`'s job).

---

## How data flows

### Starting fresh
```
create_session() → AgentState.from_session() → orchestrator loop
```

### Resuming from DB
```
load_agent_state(store, session_id)
    → reconstruct_session() in MemoryStore
    → AgentState.from_store_session()
    → hydrated AgentState (messages + latest turn data)
```

### During a research turn
```
create_turn(user_query)
    → agent fills plan, searches, fetches pages
    → update_turn(urls, snippets, final_answer)
    → add_context / add_contexts_batch for full pages
```

### Into an LLM prompt
```
AgentState.to_prompt_payload() → ContextBuilder → prompt string
```

---

## Persistence vs runtime — quick reference

| Saved to DB | Runtime only (AgentState) |
|-------------|---------------------------|
| session, messages, turns | `search_results` |
| contexts (full pages) | `error_message` |
| plan, snippets, final answer | exact `phase` mid-workflow |
| search_queries, urls_opened | |

On reload, phase is inferred: `COMPLETE` if latest turn has an answer, else `START`.

---

## Cheat sheet — "I need to…"

| Task | Go to |
|------|-------|
| Import state types | `from deep_research_agent.state import …` |
| Resume a session | `load_agent_state(store, session_id)` |
| Check workflow step | `AgentPhase` in `enums.py` |
| See all data shapes | `schema.py` |
| Load one turn with contexts | `load_turn_record(store, turn_id)` |
| Serialize state to JSON | `agent_state_to_json(state)` |
| Build LLM input from state | `state.to_prompt_payload()` |
| Understand DB ↔ typed conversion | `adapters.py` + `from_store_row` methods in `schema.py` |

---

## Relationship to other packages

```
deep_research_agent/
├── memory/          ← SQLite persistence (raw dict rows)
├── state/           ← THIS FOLDER — typed models + loaders
├── context/         ← Builds LLM prompts from AgentState (token limits, summarization)
└── (orchestrator)   ← Drives AgentPhase transitions, calls tools
```

**Rule of thumb:**
- **`memory`** = save/load bytes on disk
- **`state`** = what those bytes *mean* as Python objects
- **`context`** = squeeze that state into an LLM-sized prompt

---

## Types at a glance

```
SessionRecord
├── messages: MessageRecord[]
└── (turns loaded separately)

TurnRecord                    AgentState (runtime)
├── user_query                ├── phase: AgentPhase
├── plan: ResearchPlan        ├── plan, search_queries
├── search_queries[]          ├── search_results[]  ← not persisted
├── urls_opened[]             ├── source_contexts[]
├── context_snippets[]        ├── selected_snippets[]
├── final_answer              ├── final_answer
└── contexts: SourceContext[] ├── messages[]
                                └── error_message     ← not persisted
```

---

*Last updated for Task 1.2 (typed state schema). Pair with `memory/store.py` for persistence and `context/context_builder.py` for prompt building.*
