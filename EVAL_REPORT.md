# Evaluation Report — Deep Research Agent

**Last updated:** 2026-05-22  
**Model:** `gemini-3.5-flash`  
**Dataset:** [`dataset.json`](dataset.json) (50 cases)

---

## Full 50-case run — outcome

The full eval **ran to completion** but **could not score most cases** due to **Gemini free-tier daily request limits**.

| Run | Cases attempted | Successful | Errors | Cause |
|-----|-----------------|------------|--------|-------|
| Full run #1 (with judge) | 50 | **3** (TC_001–003) | 47 | Hit 429 after ~9 Gemini calls/case × 3 cases |
| Full run #2 (skip judge, 5s delay) | 50 | **0** | 50 | Daily quota already exhausted |

**Free-tier limit for `gemini-3.5-flash`:** **20 requests/day** per project (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`).

Each eval case uses **2 Gemini calls** (plan + answer), or **3 with judge**. So on free tier you can run roughly **6–10 cases per day**, not 50 in one batch.

---

## Best available results — 5-case pilot (`pilot_3.5flash.json`)

**Command used:**
```bash
python evaluate.py --dataset dataset.json \
  --ids TC_001,TC_021,TC_026,TC_045,TC_050 \
  --skip-judge \
  --output reports/pilot_3.5flash.json
```

| Case | Result | F1 | Notes |
|------|--------|-----|-------|
| TC_001 | **PASS** | 0.67 | Capital of Bhutan — correct |
| TC_021 | ERROR | — | Transient 429 mid-batch |
| TC_026 | FAIL | — | Should refuse; agent answered anyway |
| TC_045 | **PASS** | — | Cross-session isolation OK |
| TC_050 | **PASS** | 0.13 | Ran; weak retrieval for 3-company compare |

| Metric | Value |
|--------|-------|
| Behavior pass rate | **75%** (3/4 scored) |
| Streaming pass rate | **100%** |
| Retrieval mean F1 | **0.40** |

---

## First 3 cases from full run (with judge)

Before quota exhaustion, TC_001–003 completed with judge enabled:

| Case | Behavior | Streaming | F1 |
|------|----------|-----------|-----|
| TC_001 | PASS | PASS | 0.50 |
| TC_002 | PASS | PASS | 0.00 |
| TC_003 | PASS | PASS | 0.00 |

Judge behavior pass: 100% on these 3 cases.

---

## Root cause summary

| Issue | Detail |
|-------|--------|
| Wrong model (fixed) | Was `gemini-2.0-flash` (limit 0); now `gemini-3.5-flash` |
| Daily request cap | **20 requests/day** on free tier for 3.5-flash |
| Full 50 in one go | Needs ~100–150 Gemini calls → requires paid tier or multi-day batches |

---

## How to finish the full 50-case eval

### Option A — Run in daily batches (free tier)

~8 cases/day with `--skip-judge` (16 API calls), leave headroom:

```bash
# Day 1: cases 1–8
python evaluate.py --dataset dataset.json --skip-judge --delay-secs 10 \
  --ids TC_001,TC_002,TC_003,TC_004,TC_005,TC_006,TC_007,TC_008 \
  --output reports/batch1.json

# Day 2: cases 9–16 … and so on
```

Merge JSON reports manually or ask to add a merge script.

### Option B — Enable billing (recommended for full run)

Enable billing on the Google AI project → higher RPM/RPD limits → run:

```bash
python evaluate.py --dataset dataset.json --delay-secs 5 \
  --output reports/full_run.json
```

### Option C — Wait for quota reset

Free-tier daily quota resets on a rolling 24h window. Check: [ai.dev/rate-limit](https://ai.dev/rate-limit)

---

## Agent findings (from successful cases)

1. **Factual single-hop works** — TC_001 answered correctly with citations
2. **Insufficient evidence weak** — TC_026 failed (fabricates or answers from weak snippets)
3. **Cross-session isolation works** — TC_045 passed
4. **Multi-hop retrieval weak** — TC_050 F1 0.13 (missed per-company URLs)
5. **Streaming telemetry OK** — phase alias mapping works

---

## Files

| File | Description |
|------|-------------|
| [`reports/pilot_3.5flash.json`](reports/pilot_3.5flash.json) | Best partial eval (5 cases) |
| [`reports/full_run.json`](reports/full_run.json) | Full run attempt (mostly 429 errors) |
| [`reports/full_run.log`](reports/full_run.log) | Console log from last run |
| [`evaluate.py`](evaluate.py) | Harness CLI (`--skip-judge`, `--delay-secs`) |
