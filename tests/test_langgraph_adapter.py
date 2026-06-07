"""
LangGraph adapter regression tests.

These tests use a mock graph object that mimics CompiledStateGraph's
shape and behaviour just enough to exercise the adapter without needing
real LangGraph or a real LLM.
"""
from __future__ import annotations

import pytest

from processguard import ProcessGuard, PolicyAction
from processguard.adapters.langgraph import LangGraphAdapter, _PGCallbackHandler
from processguard.core.event import EventType


class _MockCompiledStateGraph:
    """
    Mock that mimics the LangGraph behaviour where graph.invoke() internally
    consumes graph.stream(). This is the shape that triggers the double-trace
    bug surfaced by Item 4's real LangGraph run.
    """
    __name__ = "CompiledStateGraph"   # not strictly needed; adapter probes type().__name__

    def __init__(self):
        self.invoke_calls = 0
        self.stream_calls = 0

    def invoke(self, input, config=None, **kwargs):
        self.invoke_calls += 1
        # invoke internally consumes stream — this is the LangGraph pattern that
        # the adapter must NOT double-trace.
        for _chunk in self.stream(input, config=config, **kwargs):
            pass
        return {"output": "done"}

    def stream(self, input, config=None, **kwargs):
        self.stream_calls += 1
        yield {"step": 1}


def _make_graph_with_correct_classname() -> _MockCompiledStateGraph:
    """The adapter checks type(obj).__name__ — give the instance a class name
    that matches the LangGraph allow-list."""
    cls = type("CompiledStateGraph", (_MockCompiledStateGraph,), {})
    return cls()


def _count_trace_starts(guard: ProcessGuard) -> list[str]:
    """Patch guard._on_trace_start to record every call."""
    starts: list[str] = []
    orig = guard._on_trace_start

    def _record(trace_id, input_data):
        starts.append(trace_id)
        return orig(trace_id, input_data)

    guard._on_trace_start = _record
    return starts


def test_invoke_does_not_double_trace_when_it_consumes_stream():
    """Regression for the bug found in Item 4: graph.invoke() internally
    consumes graph.stream(); the adapter previously wrapped both and produced
    two trace IDs for a single user-level invoke."""
    graph = _make_graph_with_correct_classname()
    guard  = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    starts = _count_trace_starts(guard)

    LangGraphAdapter(guard).attach(graph)
    graph.invoke({"input": "x"})

    assert len(starts) == 1, (
        f"Adapter created {len(starts)} traces for a single invoke; expected 1."
    )


def test_stream_called_directly_creates_one_trace():
    """Calling graph.stream() directly should still produce exactly one trace
    (the fix for the invoke-consumes-stream case must not break the direct
    stream path)."""
    graph = _make_graph_with_correct_classname()
    guard  = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    starts = _count_trace_starts(guard)

    LangGraphAdapter(guard).attach(graph)
    for _ in graph.stream({"input": "x"}):
        pass

    assert len(starts) == 1, (
        f"Direct stream produced {len(starts)} traces; expected 1."
    )


def test_two_separate_invokes_create_two_traces():
    """Two distinct user-level invoke() calls should produce two distinct
    traces. The re-entry guard must reset between top-level calls."""
    graph = _make_graph_with_correct_classname()
    guard  = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    starts = _count_trace_starts(guard)

    LangGraphAdapter(guard).attach(graph)
    graph.invoke({"input": "first"})
    graph.invoke({"input": "second"})

    assert len(starts) == 2, (
        f"Two top-level invokes produced {len(starts)} traces; expected 2."
    )
    assert starts[0] != starts[1], "Two invokes must produce distinct trace IDs."


# ── callback handler translation tests ───────────────────────────────────────

def test_callback_handler_on_tool_start_emits_tool_call_event():
    """When LangChain fires on_tool_start, the adapter's handler should
    translate that into a TOOL_CALL AgentEvent stored under the right
    trace_id with the right tool_name and parsed args."""
    guard = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    handler = _PGCallbackHandler(guard, trace_id="trace-cb-1")

    handler.on_tool_start(
        serialized={"name": "web_search"},
        input_str='{"query": "RAG architectures"}',
        metadata={"agent_name": "researcher"},
    )

    events = guard.storage.get_trace("trace-cb-1")
    tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name  == "web_search"
    assert tool_calls[0].tool_args  == {"query": "RAG architectures"}
    assert tool_calls[0].agent_name == "researcher"


def test_callback_handler_on_tool_end_emits_tool_result_event():
    guard = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    handler = _PGCallbackHandler(guard, trace_id="trace-cb-2")

    handler.on_tool_end(output="some search results", run_id=None)

    events = guard.storage.get_trace("trace-cb-2")
    tool_results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
    assert len(tool_results) == 1
    assert tool_results[0].tool_result == "some search results"


def test_invoke_emits_terminate_event_on_clean_completion():
    """Regression for Bug 2 (TERMINATE half): modern LangGraph's
    CompiledStateGraph does not surface an AgentFinish callback, so the
    adapter must synthesize a TERMINATE event when its own invoke wrapper
    returns cleanly. Without this, FM-3.1 PrematureTermination has nothing
    to fire on, even when the agent's run did legitimately end."""
    graph  = _make_graph_with_correct_classname()
    guard  = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    starts = _count_trace_starts(guard)

    LangGraphAdapter(guard).attach(graph)
    graph.invoke({"input": "x"})

    assert len(starts) == 1
    trace_id = starts[0]
    events   = guard.storage.get_trace(trace_id)
    terminates = [e for e in events if e.event_type == EventType.TERMINATE]
    assert len(terminates) >= 1, (
        f"Expected ≥1 TERMINATE event after a clean invoke; "
        f"got event types: {[e.event_type.value for e in events]}"
    )


def test_stream_emits_terminate_event_on_clean_exhaustion():
    """Same regression as above for the stream entry-point."""
    graph  = _make_graph_with_correct_classname()
    guard  = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    starts = _count_trace_starts(guard)

    LangGraphAdapter(guard).attach(graph)
    for _ in graph.stream({"input": "x"}):
        pass

    assert len(starts) == 1
    trace_id = starts[0]
    events   = guard.storage.get_trace(trace_id)
    terminates = [e for e in events if e.event_type == EventType.TERMINATE]
    assert len(terminates) >= 1, (
        f"Expected ≥1 TERMINATE event after stream exhaustion; "
        f"got event types: {[e.event_type.value for e in events]}"
    )
