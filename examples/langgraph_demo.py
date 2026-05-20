"""
LangGraph demo — FM-1.3 step repetition caught at runtime.

Requires:
    pip install processguard[langgraph] langchain-anthropic

Run:
    python examples/langgraph_demo.py
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    graph = build_graph()

    # Attach processguard — STEER on step repetition so the agent can recover
    guard = processguard.attach(
        graph,
        default_policy=PolicyAction.STEER,
        db_path=":memory:",
    )
    guard.policy.policies["BEYOND-MAST"] = PolicyConfig(action=PolicyAction.STEER)

    print("\n=== ProcessGuard LangGraph Demo ===")
    print("Task: Research RAG architectures in 2026\n")

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content="Research RAG architectures in 2026. Give me a summary.")]},
        )
        final = result["messages"][-1]
        print(f"\n--- Final output ---\n{final.content}")
    except Exception as e:
        print(f"\n[halted] {e}")

    print(f"\nDetections: {len(guard.policy.detections)}")
    for d in guard.policy.detections:
        print(f"  {d.failure_mode} {d.failure_name} (conf={d.confidence:.2f})")


if __name__ == "__main__":
    main()
