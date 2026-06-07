"""
SYNTHETIC LangGraph demo — constructed to trigger FM-1.3 (step repetition)
and the no-progress-loop detector via a tool that always returns the same
useless string.

This is NOT a real-world run. The agent's behavior here is engineered:
the `web_search` tool returns identical content on every call, which
forces the agent to either repeat its query (catches StepRepetition) or
loop without progress (catches NoProgressLoop). The detectors firing in
this script proves wiring, not correctness against organic agent
behavior.

For a non-synthetic run with real model decisions and real search
results, see `examples/real_langgraph_demo.py` — and read
`docs/real_run_findings.md` for what that run revealed about the
LangGraph adapter (spoiler: it has known bugs as of v0.1.1).

Requires:
    pip install processguard[langgraph] langchain-anthropic

Run:
    python examples/synthetic_langgraph_demo.py
"""

from __future__ import annotations

import os
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import processguard
from processguard import PolicyAction, PolicyConfig


# ── fake tool that never makes progress (simulates the demo in the deck) ─────

_search_count = 0

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    global _search_count
    _search_count += 1
    print(f"  [tool] web_search({query!r})  call #{_search_count}")
    # always return the same useless result to trigger no-progress detector
    return "Results: some generic content about the topic. Try again for more specific results."


# ── minimal LangGraph agent ───────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools([web_search])

    def call_model(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def call_tools(state: State):
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            result = web_search.invoke(call["args"])
            results.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: State):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", call_tools)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# ── harness-callable entry point ──────────────────────────────────────────────

def run_with_task(task: str = "Research RAG architectures in 2026. Give me a summary.") -> "processguard.ProcessGuard":
    """Build the graph, attach ProcessGuard, run the task, return the guard.
    Used both by main() and by the eval harness — keeps the demo and the
    harness from drifting apart."""
    graph = build_graph()
    guard = processguard.attach(
        graph,
        default_policy = PolicyAction.STEER,
        db_path        = ":memory:",
        llm_detectors  = False,
        verbose        = False,
    )
    guard.policy.policies["BEYOND-MAST"] = PolicyConfig(action=PolicyAction.STEER)

    try:
        graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": 15},
        )
    except Exception:
        # Steer-then-halt scenarios raise; the events are still in the guard.
        pass
    return guard


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== ProcessGuard LangGraph Demo ===")
    print("Task: Research RAG architectures in 2026\n")
    guard = run_with_task()
    print(f"\nDetections: {len(guard.policy.detections)}")
    for d in guard.policy.detections:
        print(f"  {d.failure_mode} {d.failure_name} (conf={d.confidence:.2f})")


if __name__ == "__main__":
    main()
