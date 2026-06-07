# CLAUDE.md — context for agents working on this repo

You (Claude, or any other agent) are about to work on ProcessGuard. Read
this file first. It exists so you start each session with the same
context the rest of us have, and so you don't re-derive things that have
already been decided.

---

## Project goal

ProcessGuard is a Python library that detects multi-agent **coordination
failures** at runtime, mapped to the peer-reviewed [MAST taxonomy](https://arxiv.org/abs/2503.13657)
(Cemri et al., UC Berkeley Sky Lab, 2025). The library hooks into an
agent framework (LangGraph today), normalises events to a common schema,
runs detectors over each event as it arrives, and applies a policy
(log / steer / halt / escalate) when a detector fires. The point is to
catch failures **as they happen**, not in the postmortem — observability
tools and content guardrails are blind to this category of bug.

Differentiator: every detector is grounded in a specific MAST failure
mode and ships with a five-sentence contract that states, in plain
English, what it does and doesn't catch. No vibes-based detection.

---

## Current state (v0.1.1)

- 5 detectors: FM-1.3 step repetition, FM-1.5 unaware termination,
  BEYOND-MAST no-progress tool loop, FM-2.6 reasoning-action mismatch,
  FM-3.1 premature termination. Contracts below.
- LangGraph adapter (auto-attach via `processguard.attach(graph)`).
  Two real bugs surfaced by the v0.1.1 real-run validation and fixed
  in the same release window — see `docs/real_run_findings.md`.
- SQLite trace storage, thread-safe via shared connection + lock.
- Policy engine with four actions; per-failure-mode overrides supported.
- 34 tests passing — 24 from v0.1.0 plus 10 added in v0.1.1 covering
  adapter re-entry, callback translation, TERMINATE synthesis, and
  cross-thread storage access.

Known gaps documented honestly in the README's "What doesn't work yet"
section. The short version:
- CrewAI deferred to v1.1 — experimental adapter exists, doesn't work.
- REASONING events not auto-emitted by the LangGraph adapter (provider
  callback chains have no canonical reasoning channel). FM-2.6 only
  fires if the user emits REASONING events manually.
- Two LLM-judge detectors have no measured precision/recall — audit
  in `docs/judge_audit.md`, harness proposed but not implemented.
- End-to-end TOOL_CALL/TOOL_RESULT path is unit-tested only; real-run
  validation through a tool-invoking agent remains open.

---

## The five detector contracts

These are the canonical contracts. They live in the docstring of each
detector class. Reproduced here so any future session has them in
context immediately.

### FM-1.3 — StepRepetitionDetector

Identifies when an agent has fallen into a loop of issuing the same
effective action repeatedly, with no behavioural change between
repetitions. Fires once an agent has issued the same tool call with the
same arguments enough times in close succession that further repetition
is no longer plausibly exploratory but a loop the agent cannot break
out of on its own. Smallest meaningful case: `web_search(query="X")`
three times in a row, about to be called a fourth, with no intervening
reasoning. Must not fire when the agent calls the same tool with
materially different arguments, nor on legitimate fan-out (`read_file`
across many files). Known limitation: two calls count as "the same"
only if their arguments match exactly — semantically equivalent calls
phrased differently will not be flagged.

### FM-1.5 — UnawareTerminationDetector

Identifies when an agent has lost track of its own termination
criteria. Fires when the agent has been working far longer than the
task plausibly requires AND has settled into a narrow repertoire of
actions, suggesting it has forgotten or never had a plan for when to
stop. Smallest case: a research agent asked to summarize three papers
that has called `web_search` forty times without ever producing the
summary. Must not fire on legitimately long-running tasks (an agent
given two hundred files to process that has called `read_file` forty
times on forty different paths). Known limitation: requires the agent
to have settled into a *single* dominant action — an agent alternating
between two or three tools will slip through.

### BEYOND-MAST — NoProgressLoopDetector

Identifies when recent tool calls are returning content the agent has
effectively already seen. Fires when the most recent several tool
results, taken together, have introduced essentially no new
information that wasn't already in the agent's earlier results — the
recent window is informationally redundant with what the agent
already knows. Smallest case: four `web_search` calls on distinct
sub-questions ("RAG retrieval methods", "RAG chunking", etc.) where
each result only restates the same overview paragraph. Must not fire
when the agent is making genuine progress across distinct sources that
happen to share vocabulary (a developer agent grepping a codebase
where many files contain the same API name). Known limitation: "new
information" is judged by surface vocabulary — a tool that paraphrases
the same facts with different words each time evades detection.

### FM-2.6 — ReasoningActionMismatchDetector

Identifies when an agent's stated reasoning and its next action don't
line up. Fires whenever an agent has explicitly stated an intent for
its next step and the next observable action does not carry out that
intent, as judged by an independent model reading both. Smallest case:
agent reasons *"I will delegate this to the writer agent"* and then
calls a search tool. Must not fire on faithful paraphrase — agent
reasons *"I'll look up info on X"* and then calls
`web_search(query="X recent developments")`. Known limitation: the
judge is itself an LLM and brings biases — particularly, when the
reasoning lays out a multi-step plan but only the first step is
visible in the action, the judge tends to flag mismatch even though
the action correctly executes step one.

Operational preconditions:
- Requires the `anthropic` package.
- Requires REASONING events in the trace. The LangGraph adapter does
  NOT auto-emit them. Users must emit REASONING events manually via
  `guard.emit()` from their own LLM-wrapper instrumentation.

### FM-3.1 — PrematureTerminationDetector

Identifies when an agent declares the task complete while its actual
output fails to address the goals it was given. Fires whenever an
agent has signalled termination and its final output does not, as
judged by an independent model, fully address the task as originally
stated. Smallest case: agent given *"find five recent papers on RAG,
summarize each, recommend one"* that terminates after summarizing
three papers and offering no recommendation. Must not fire when the
output is genuinely complete even if brief — agent given *"what's the
population of Tokyo?"* terminating with *"13.96 million"* has fully
addressed the task. Known limitation: the judge is biased toward
output that *reads* like a finished answer — a confidently-phrased
single paragraph can be marked COMPLETE even when the task asked for
several specific deliverables (five citations, a final recommendation)
that the paragraph silently omits.

Operational precondition: requires the `anthropic` package.

---

## Working rules

These are not stylistic preferences. They are how this repo has been
maintained and how it should continue to be maintained. Read them
before you start.

**1. When given a specific implementation directive, implement it.** Do
not propose alternative approaches, do not re-derive the rationale, do
not list pros and cons. If the directive is ambiguous, ask one
clarifying question and wait.

**2. When a test fails, do not retry the same fix with a small
variation.** Stop, summarize what failed and why, and propose at most
two genuinely different alternatives. Wait for the user's pick.

**3. Finished editing a file ≠ done.** "Tests pass" is a verified
claim, not an assumption. Run the relevant test or command and read
its actual output before declaring done.

**4. Stay in scope.** If the current task is about docstrings, do not
refactor implementations while you are there. Out-of-scope changes get
flagged separately for the user's review. Use the spawn-task affordance
or surface a note rather than silently bundling unrelated changes.

**5. Bug-fix discipline.** Fix bugs one at a time. For each bug: write
a failing test that captures the bug, fix the bug, verify the test now
passes, verify the full suite still passes, and only then move to the
next bug. The v0.1.1 adapter bug-fix commit (`c43e30c`) is the
canonical example of this pattern.

**6. Honesty over polish.** If something doesn't work, document it.
The README's "What doesn't work yet" section is load-bearing — it's
what makes the rest of the README credible. Do not delete entries
from it without measured evidence that the gap has actually closed.

**7. Per-item status posts in multi-item work.** When working through
a numbered plan (Items 1-5, etc.), post a one-line "Item N complete.
Findings: …" message between items so the user can intervene at
natural boundaries.

**8. Do not push to remotes unless explicitly authorised.** Local
commits and local tags are reversible; pushing is not. Wait for the
user to say "push" (or equivalent) before running `git push` or
`git push --tags`.

---

## Repo layout

```
processguard/
├── README.md                          # public-facing pitch + what works/doesn't
├── CLAUDE.md                          # you are here
├── pyproject.toml                     # version, deps, optional-deps groups
├── processguard/
│   ├── __init__.py                    # attach() entry point
│   ├── guard.py                       # ProcessGuard class, adapter dispatch
│   ├── core/
│   │   ├── event.py                   # AgentEvent dataclass, EventType enum
│   │   ├── storage.py                 # TraceStorage (shared SQLite conn + lock)
│   │   └── policy.py                  # PolicyEngine, PolicyAction, Detection
│   ├── detectors/
│   │   ├── step_repetition.py         # FM-1.3
│   │   ├── unaware_termination.py     # FM-1.5
│   │   ├── no_progress_loop.py        # BEYOND-MAST
│   │   ├── reasoning_action_mismatch.py  # FM-2.6 (LLM judge)
│   │   └── premature_termination.py   # FM-3.1 (LLM judge)
│   ├── adapters/
│   │   ├── base.py
│   │   └── langgraph.py               # CompiledStateGraph adapter, callback handler
│   └── experimental/
│       ├── __init__.py
│       └── crewai.py                  # NOT IN V1 — see README
├── examples/
│   ├── real_langgraph_demo.py         # the canonical v0.1.1 demo
│   ├── real_run_log.jsonl             # captured event stream from the canonical run
│   ├── real_run_status.json           # structured run summary
│   ├── synthetic_langgraph_demo.py    # rigged to fire detectors — for demo only
│   └── synthetic_raw_loop_demo.py     # rigged demo via manual guard.emit()
├── experimental/
│   └── crewai_demo.py                 # reproducer for the CrewAI adapter gap
├── docs/
│   ├── judge_audit.md                 # FM-2.6 + FM-3.1 prompts, failure cases, harness proposal
│   └── real_run_findings.md           # Item 4 findings + the three bugs it surfaced (with fixes)
└── tests/                             # 34 tests
    ├── conftest.py
    ├── test_step_repetition.py
    ├── test_unaware_termination.py
    ├── test_no_progress_loop.py
    ├── test_policy.py
    ├── test_langgraph_adapter.py      # added in v0.1.1
    └── test_storage_threading.py      # added in v0.1.1
```

---

## Quick reference: how to run things

```bash
# Test suite
python -m pytest -q

# Synthetic demo (no API key, no network)
python examples/synthetic_raw_loop_demo.py

# Real demo (needs GOOGLE_API_KEY for Gemini; set llm_detectors=False or
# also set ANTHROPIC_API_KEY for the LLM-judge detectors)
python examples/real_langgraph_demo.py
```
