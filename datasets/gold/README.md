# Gold set — v0.2

10 cases. Regression-heavy by design: 6 cases lock in detector contracts
discovered/refined during v0.1.1, 4 cases lock in the LangGraph adapter
bugs surfaced by Item 4's real run.

## Per-case intent

| # | id | What it locks in | Runs in CI? |
|---|----|------------------|-------------|
| 1 | `step-rep-fires-at-3` | StepRepetition fires at the contract boundary (3 identical calls). | Yes |
| 2 | `step-rep-does-not-fire-at-2` | StepRepetition does NOT fire on 2 calls (still plausibly exploratory). | Yes |
| 3 | `step-rep-fan-out-allowed` | StepRepetition must not fire on legitimate fan-out (same tool, different args). | Yes |
| 4 | `no-progress-fires-on-redundant-results` | NoProgressLoop fires when 4 tool_results have identical vocabulary. | Yes |
| 5 | `no-progress-allowed-on-distinct-results` | NoProgressLoop must not fire when 4 tool_results have distinct vocabulary (the "grep across codebase" counter-example). | Yes |
| 6 | `premature-fires-on-empty-output` | A TERMINATE with no prior MESSAGE does NOT fire FM-3.1 in this harness (the empty-bypass path is in the FM-3.1 judge detector, which is disabled by `llm_detectors=False` in the harness). Locks in the current behaviour; if we later wire the bypass path to fire without an LLM call, this assertion flips. | Yes |
| 7 | `synthetic-langgraph-loop-detected` | The synthetic LangGraph demo's looping web_search must trip StepRepetition end-to-end through the LangGraph adapter. | No — requires `ANTHROPIC_API_KEY` |
| 8 | `real-langgraph-completion-emits-terminate` | Regression for v0.1.1 Bug 2a — a clean LangGraph run must produce ≥1 TERMINATE event via the adapter's synthetic emission. | No — requires `GOOGLE_API_KEY` |
| 9 | `real-langgraph-single-trace` | Regression for v0.1.1 Bug 1 — the contextvar re-entry guard must produce exactly 1 trace_id per `invoke()`. | No — requires `GOOGLE_API_KEY` |
| 10 | `real-langgraph-storage-thread-safe` | Regression for v0.1.1 Bug 1b — storage must work from LangChain's worker threads. Passing this case means no `OperationalError: no such table: events` was raised. | No — requires `GOOGLE_API_KEY` |

## How to add a case

1. Pick a real failure mode the existing cases don't cover. Use one of:
   - A v0.1.1 bug discovery whose pytest test would benefit from also being
     covered at the agent-behaviour level.
   - A detector contract boundary not currently tested (e.g. an edge case
     in the "must not fire" half).
   - A real adapter regression discovered in production.
2. Add a JSONL line at the bottom of `v0.2.jsonl` following the schema:
   ```json
   {"id": "...", "source": "...", "notes": "...",
    "agent_target": "manual" | "module.path:callable",
    "input": {...},
    "assertions": [{"type": "...", "args": {...}}],
    "requires_env": ["..."]}
   ```
3. If the case requires LLM access, set `requires_env`. Otherwise omit.
4. Add a row to the table above.
5. Run `python scripts/run_eval.py --gold datasets/gold/v0.2.jsonl` locally
   and confirm the new case behaves as expected.

## Why no synthetic-raw-loop case

The synthetic raw-loop demo (`examples/synthetic_raw_loop_demo.py`) pushes
hand-crafted events directly through `guard.emit()` — the same code path
that cases 1–6 use. Including a case that wraps the demo would be a
re-test of the same path, not new coverage. The demo is kept around as a
human-friendly walkthrough, not as a gold-set fixture.
