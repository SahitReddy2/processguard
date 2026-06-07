# Real LangGraph run — findings

**Run date:** 2026-05-20
**Demo script:** [`examples/real_langgraph_demo.py`](../examples/real_langgraph_demo.py)
**Event log:** [`examples/real_run_log.jsonl`](../examples/real_run_log.jsonl)
**Status JSON:** [`examples/real_run_status.json`](../examples/real_run_status.json)

This is the first run of ProcessGuard against a non-synthetic LangGraph
agent. The purpose was to observe how the rule-based detectors and the
LangGraph adapter behave on a real agent (Gemini 2.5 Flash, recursion
limit 20, DuckDuckGo + in-memory fallback) given an open-ended task.
**No detector code or adapter code was changed during this session.**
Findings are reported as-observed; fixing decisions are deferred to the
user.

The two LLM-judge detectors (FM-2.6, FM-3.1) were disabled for this run
via `llm_detectors=False` because they are hardcoded against the
Anthropic SDK and no Anthropic key was available; their behaviour is
audited separately in [`judge_audit.md`](judge_audit.md).

---

## Headline findings

### Finding 1 — The LangGraph adapter double-counts the trace

The verbose console output shows a single `agent.invoke(...)` call. The
event log shows **two distinct trace IDs**, both created within sub-millisecond
of each other:

```
[processguard] trace 50f1edf6... started
[processguard] trace 7e68aba7... started
[processguard] trace 7e68aba7... ended - 1 events, 0 detections
[processguard] trace 50f1edf6... ended - 1 events, 0 detections
```

And the JSONL has two events with **identical content, sub-millisecond
apart**, each tagged with a different trace ID:

```
{"ts": 1780798437.1404178, "trace_id": "7e68aba7-...", "span_id": "step-1", "event_type": "message", ...}
{"ts": 1780798437.1407053, "trace_id": "50f1edf6-...", "span_id": "step-1", "event_type": "message", ...}
```

**Hypothesis (not investigated, do not act on this yet):** the adapter
wraps both `graph.invoke` and `graph.stream`. If the upstream LangGraph
implementation of `invoke` internally consumes `stream`, both wrappers
fire on a single user-level call, creating two independent trace IDs and
two independent callback handlers. Each handler then independently
records the LLM response, hence the duplicated event with near-identical
timestamps.

If confirmed, the consequence is: **every detection that depends on a
trace-scoped state machine (StepRepetition, NoProgressLoop,
UnawareTermination) would see only half the events on a given run,
split across two traces, and would fail to fire on legitimate failure
patterns** because the sliding windows never accumulate enough
fingerprints in any single trace key.

This is exactly the kind of bug Item 4 was built to find. It would not
show up on the synthetic demos because those scripts use `agent.invoke`
deliberately in a way that hides the issue — the agent always finishes
in 0–1 tool calls, so the double-trace bug doesn't have time to corrupt
detector windows.

### Finding 2 — The adapter captured far fewer events than the agent produced

The agent ran to completion and produced a multi-paragraph answer. The
event log contains **two events total**, both `message`, both with the
same 200-char prefix of the agent's final answer. Specifically:

- Expected events for a single-turn answer (no tool calls): at least 1
  `message` event, ideally 1 `terminate` event.
- Actual events: 2 `message` events (one per duplicate trace, see Finding 1),
  zero `terminate` events.

The adapter's callback handler defines `on_agent_finish` and `on_llm_end`
handlers, but no `terminate` event was emitted. Either `on_agent_finish`
is not firing on a Gemini-driven LangGraph run, or `create_react_agent`
in current LangGraph versions does not surface an `AgentFinish` event
through the callback chain. Reasoning events (`REASONING` type) were
also entirely absent, which means `ReasoningActionMismatchDetector`
would have no input to act on even if it were enabled.

**Open question:** which callback chain actually fires for
`langgraph.prebuilt.create_react_agent` in v0.2+? The adapter was
designed against an older callback contract and may need an update to
hook the modern message-graph events directly rather than going through
LangChain's AgentExecutor-era callbacks.

### Finding 3 — The agent did not exercise any tool

`web_search` was wired in correctly and is invokable; the global counter
in the tool ended at zero. Gemini 2.5 Flash chose to answer the task
("compare RAG, fine-tuning, prompt engineering for internal docs QA")
from its training priors rather than searching. This is a legitimate
agent decision — the topic is squarely in its pre-training distribution
— and per the v0.1.1 plan's "do not retry with leading prompts to force
a detector to fire" rule, it is being reported as the actual outcome
rather than worked around.

**Consequence for Item 4's coverage goal:** the rule-based detectors
(StepRepetition, NoProgressLoop, UnawareTermination) all depend on
TOOL_CALL or TOOL_RESULT events. With zero tool calls, they have nothing
to observe. Their non-firing on this run is **not evidence they work
correctly** — it is the absence of an opportunity to fire. This run
therefore does not validate the detectors; it validates the adapter,
and the adapter has Findings 1 and 2.

---

## Event count by type

From the JSONL:

| Event type    | Count | Notes |
|---------------|-------|-------|
| `message`     | 2     | Duplicated across two trace IDs (Finding 1) |
| `tool_call`   | 0     | Agent did not call any tool (Finding 3) |
| `tool_result` | 0     | No tool calls means no results |
| `reasoning`   | 0     | Adapter did not emit any reasoning events (Finding 2) |
| `terminate`   | 0     | Adapter did not emit a terminate event (Finding 2) |
| **Total**     | **2** | — |

The status JSON's `final_output_length: 1` is itself a small bug in the
demo script (Gemini returns content as a list of content blocks; my
code `len(final_content)` measures the list length, not the text length).
The actual final answer is several paragraphs. This does not affect any
ProcessGuard claim, just the cosmetics of the status capture.

---

## Per-detector analysis

### StepRepetitionDetector (FM-1.3)
- **Fired:** No.
- **Should have fired:** No — zero tool calls means no repetition is
  possible.
- **Could have fired if the adapter were correct:** No — even with
  proper event capture, the agent only made one model call.
- **Verdict:** untested by this run.

### UnawareTerminationDetector (FM-1.5)
- **Fired:** No.
- **Should have fired:** No — the agent ended after one step, well under
  any reasonable step budget.
- **Verdict:** untested by this run.

### NoProgressLoopDetector (BEYOND-MAST)
- **Fired:** No.
- **Should have fired:** No — zero tool results means no novelty signal
  to evaluate.
- **Verdict:** untested by this run.

### ReasoningActionMismatchDetector (FM-2.6)
- **Disabled** for this run via `llm_detectors=False` (no Anthropic key).
- Would not have fired anyway: no REASONING events were captured by the
  adapter (Finding 2), so there is no buffered reasoning to compare any
  action against.

### PrematureTerminationDetector (FM-3.1)
- **Disabled** for this run via `llm_detectors=False`.
- Would not have fired anyway: no TERMINATE event was captured by the
  adapter (Finding 2), so the detector would never have been triggered.

**Net:** the run validated zero detectors. It validated that the
adapter is wired into the LangGraph callback chain (we got two message
events out, so something is firing), and exposed two adapter bugs in
the process.

---

## Was the task right?

Yes and no. The task ("compare RAG, fine-tuning, prompt engineering for
internal docs QA, recommend one") was open-ended enough that an agent
that did not know the answer would have searched; it just happens that
Gemini 2.5 Flash *does* know the answer well enough to skip searching.
A more search-forcing task would be one where the model has no priors:
asking about a specific company's internal practices, a very recent
event, a niche library's API, etc. **This does not warrant changing the
task in this session** — Finding 1 and Finding 2 dominate, and they
would still apply on a search-heavy task. Re-running with a different
task is a Future Item, not part of v0.1.1.

---

## DDG status

- DuckDuckGo package was installed and importable (`ddg_available: true`).
- DDG was never reached because the agent did not invoke the tool
  (`ddg_used: false`, `total_search_calls: 0`).
- The in-memory fallback was therefore also never engaged.

We learned nothing about DDG quality or rate-limiting behaviour from
this run. That's a known gap; the fallback path will be exercised
whenever a future run actually drives the agent through `web_search`.

---

## Other observations

- **Deprecation warning:** `langgraph.prebuilt.create_react_agent` is
  deprecated in LangGraph v1.0 and is moving to
  `langchain.agents.create_agent`. Not a blocker; worth noting for v1.1
  if the adapter is rewritten to hit the new entrypoint directly.
- **Gemini message content shape:** Gemini returns the final message's
  `.content` as a list of content blocks (`[{'type': 'text', 'text': '...'}]`)
  rather than a plain string. The adapter's `on_llm_end` does
  `response.generations[0][0].text` which appears to handle this, but
  downstream consumers (like the demo script's `final_content`
  length-measurement) need to be aware.
- **Cost:** two API calls to Gemini 2.5 Flash, well under $0.01.
  Anthropic spend: $0.

---

## Decisions the user has to make

These are listed because the rules say "do not edit detectors based on
this finding yet."

1. **Adapter double-trace bug (Finding 1):** ship v0.1.1 with this as a
   known issue documented in the README, or fix before tagging? My
   instinct is **document and ship** — the fix needs careful testing
   against multiple LangGraph entrypoints and that work is its own
   session, but the bug is severe enough that running v0.1.1 against a
   real multi-step agent would produce broken detection. So the doc has
   to be explicit: "v0.1.1 LangGraph adapter is known to double-count
   traces on agents whose `invoke` internally consumes `stream`; rule-based
   detectors will not fire reliably on such agents until v1.1."

2. **Adapter event coverage (Finding 2):** same call. The adapter is
   missing TERMINATE and REASONING events for current LangGraph
   ReAct agents. Either fix now (substantial work) or document and
   defer.

3. **Re-run with a search-forcing task (Finding 3):** worth doing once
   Finding 1 and 2 are addressed, but doing it before fixing the
   adapter would just produce a longer log with the same bugs. Defer
   to after the adapter fix.

4. **Demo script `final_output_length` cosmetic bug:** trivial, fix
   inline whenever the script is next touched.

---

## What this run actually tells us about ProcessGuard

The honest summary: **v0.1's LangGraph adapter, as shipped, does not
emit a clean enough event stream for the detectors to operate on a real
agent.** The detectors themselves are not implicated by this run because
they never got fed correctly. The adapter is the binding layer that
makes the rest of the library useful, and that binding layer needs
work before any precision/recall claim about the detectors against real
agents can be defended.

This is exactly the kind of finding the v0.1.1 direction-correction
pass was built to surface. v0.1.0 with its synthetic demos *looked*
fine. v0.1.1 with one real run reveals the adapter gap. That's the
loop working as intended.
