# processguard

**Runtime detection for the coordination failures that kill 41–87% of multi-agent runs.**

```
pip install processguard
```

---

## The problem

Your observability stack tells you tokens used and latency. It does not tell you:

- Your researcher agent searched the same query 47 times
- Your writer agent terminated before reading the researcher's output
- Your orchestrator said "I will delegate to agent B" then called agent A

These are coordination failures — the category of bug that existing tools are blind to. They map to a peer-reviewed taxonomy from UC Berkeley: [MAST](https://arxiv.org/abs/2503.13657) (Multi-Agent System Failure Taxonomy), 14 distinct failure modes validated against 1,600+ annotated traces.

ProcessGuard is a runtime middleware that detects MAST failure modes **as they happen**, not in the postmortem.

---

## Quickstart

```python
import processguard

# LangGraph (the only auto-attach target in v1)
processguard.attach(graph)
result = graph.invoke({"messages": [HumanMessage(content="Research RAG")]})
```

One line. No SaaS dependency. SQLite by default.

> CrewAI support is deferred to v1.1. An experimental adapter exists at
> `processguard.experimental.crewai` but does not reliably capture tool-call
> events on current CrewAI versions — most detectors will not fire. See
> [`experimental/crewai_demo.py`](experimental/crewai_demo.py) for the
> reproducer.

---

## What gets detected

| ID | Failure mode | How |
|----|-------------|-----|
| FM-1.3 | Step repetition | Fingerprint (tool, args) + sliding window |
| FM-1.5 | Unaware of termination | Step budget + convergence heuristic |
| FM-2.6 | Reasoning-action mismatch | LLM judge: stated intent vs. taken action |
| FM-3.1 | Premature termination | LLM judge: original goals vs. final output |
| BEYOND-MAST | No-progress tool loop | Entity novelty across last N tool results |

V2 adds: FM-2.3 task derailment (cosine drift), FM-3.3 incorrect verification, FM-1.4 loss of history.

---

## Policy: what happens when a failure is detected

```python
from processguard import ProcessGuard, PolicyAction, PolicyConfig

guard = ProcessGuard(
    default_policy=PolicyAction.LOG,          # just print (default)
    # or:
    # default_policy=PolicyAction.STEER,      # inject a corrective message
    # default_policy=PolicyAction.HALT,       # raise ProcessGuardError
    # default_policy=PolicyAction.ESCALATE,   # call your callback
)

# Per-failure-mode overrides
guard.policy.policies["FM-1.3"] = PolicyConfig(
    action=PolicyAction.STEER,
    steer_message="You are looping. Change strategy.",
)
guard.policy.policies["FM-3.1"] = PolicyConfig(
    action=PolicyAction.HALT,
)

guard.attach(graph)
```

---

## Manual instrumentation (any framework)

```python
from processguard import ProcessGuard
from processguard.core.event import AgentEvent, EventType

guard = ProcessGuard()

# emit events yourself
guard.emit(AgentEvent(
    trace_id="run-001",
    span_id="step-1",
    event_type=EventType.TOOL_CALL,
    agent_name="researcher",
    tool_name="web_search",
    tool_args={"query": "RAG 2026"},
))
```

---

## Architecture

```
YOUR AGENT (LangGraph / raw loop)
        │
   ADAPTER  ← normalizes events to OTEL-compatible AgentEvent schema
        │
   DETECTORS ← pure-Python or small LLM calls, one per MAST failure mode
        │
   POLICY ENGINE ← log / steer / halt / escalate
        │
   STORAGE  ← SQLite (default), swap for Postgres / Neo4j in v2
```

---

## Roadmap

- **V1** (now): 5 detectors, LangGraph adapter, SQLite storage
- **V1.1**: working CrewAI adapter (current one is experimental — see above)
- **V2**: 4 more detectors (derailment, verification), AutoGen + OpenAI Agents SDK adapters, optional hosted dashboard
- **Later**: MCP server interface, graph DB trace storage, auto-tuned thresholds

---

## Research basis

[Cemri, Pan, Yang et al. — MAST: Multi-Agent System Failure Taxonomy (arXiv 2503.13657)](https://arxiv.org/abs/2503.13657)  
UC Berkeley Sky Lab, 2025. Adopted by IBM Research IT-Bench.

---

## License

MIT
