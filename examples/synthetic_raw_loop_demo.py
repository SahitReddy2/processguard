"""
SYNTHETIC raw-loop demo — constructed to trigger StepRepetition and
NoProgressLoop via hand-crafted AgentEvents pushed through the manual
emit API.

This is NOT a real-world run. Every event is hand-written to put the
detectors into firing states; the script exists to show the manual-emit
API surface for frameworks ProcessGuard doesn't have an adapter for,
and to provide a deterministic detector demo with zero external
dependencies (no LLM, no network).

For a non-synthetic run with real model decisions, see
`examples/real_langgraph_demo.py` and the findings in
`docs/real_run_findings.md`.

No external dependencies required (uses :memory: SQLite, no LLM detectors).

Run:
    python examples/synthetic_raw_loop_demo.py
"""

import uuid
import processguard
from processguard import ProcessGuard, PolicyAction, PolicyConfig
from processguard.core.event import AgentEvent, EventType
from processguard.detectors.step_repetition import StepRepetitionDetector
from processguard.detectors.no_progress_loop import NoProgressLoopDetector


def simulate_looping_agent():
    guard = ProcessGuard(
        detectors=[
            StepRepetitionDetector(window=5, threshold=3),
            NoProgressLoopDetector(window=4, novelty_threshold=0.05),
        ],
        default_policy=PolicyAction.STEER,
        db_path=":memory:",
        llm_detectors=False,
    )

    trace_id = str(uuid.uuid4())
    guard._on_trace_start(trace_id, "Research RAG 2026")

    print("\n=== ProcessGuard Raw Loop Demo ===\n")

    queries = [
        "RAG",
        "RAG 2026",
        "RAG latest",
        "RAG newest",          # repetition threshold hit here
        "RAG most recent",
        "RAG updated 2026",
    ]

    steer_injected = False
    for i, query in enumerate(queries, 1):
        print(f"[step {i:02d}] web_search({query!r})")

        # emit tool call
        call_event = AgentEvent(
            trace_id   = trace_id,
            span_id    = f"step-{i}a",
            event_type = EventType.TOOL_CALL,
            agent_name = "researcher",
            tool_name  = "web_search",
            tool_args  = {"query": query},
        )
        steers = guard.emit(call_event)

        # emit (stale) tool result
        result_event = AgentEvent(
            trace_id    = trace_id,
            span_id     = f"step-{i}b",
            event_type  = EventType.TOOL_RESULT,
            agent_name  = "researcher",
            tool_result = "Generic content about the topic. No new information.",
        )
        steers += guard.emit(result_event)

        if steers and not steer_injected:
            steer_injected = True
            print(f"  -> steer injected: \"{steers[0]}\"")
            print(f"  [step {i+1:02d}] read_paper(url='https://arxiv.org/abs/2503.13657')")
            print(f"  [step {i+2:02d}] writer.draft(...)")
            print(f"  [step {i+3:02d}] terminate (verified)")
            break

    guard._on_trace_end(trace_id, None)
    print(f"\nTotal detections: {len(guard.policy.detections)}")


if __name__ == "__main__":
    simulate_looping_agent()
