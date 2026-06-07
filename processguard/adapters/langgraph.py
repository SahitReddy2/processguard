from __future__ import annotations

import contextvars
import uuid
from typing import Any, Optional

from .base import BaseAdapter
from ..core.event import AgentEvent, EventType


# Re-entry guard: LangGraph's CompiledStateGraph.invoke() internally consumes
# its own .stream(); without this guard, both the invoke and stream wrappers
# below would fire on a single user-level call, creating two trace IDs and
# duplicating every event. Item 4's real run made this concrete. See
# tests/test_langgraph_adapter.py for the regression.
_PG_TRACE_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_pg_trace_depth", default=0
)


class LangGraphAdapter(BaseAdapter):
    """
    Adapter for LangGraph compiled graphs.

    Wraps graph.invoke / graph.stream to inject a LangChain callback handler
    that converts tool calls, tool results, and LLM outputs into AgentEvents.

    Usage (automatic via processguard.attach):
        import processguard
        processguard.attach(graph)          # patches graph.invoke / graph.stream
        result = graph.invoke({"messages": [...]})
    """

    def __init__(self, guard):
        self.guard = guard

    def attach(self, graph):
        self._patch(graph)
        return graph

    # ── patching ─────────────────────────────────────────────────────────────

    def _patch(self, graph):
        guard = self.guard
        _orig_invoke = graph.invoke
        _orig_stream = graph.stream

        def _invoke(input, config=None, **kwargs):
            # Re-entry guard: if we're already inside a top-level wrapped call
            # (which happens when invoke internally consumes our wrapped stream),
            # pass straight through to the original without starting a new trace.
            if _PG_TRACE_DEPTH.get() > 0:
                return _orig_invoke(input, config=config, **kwargs)

            trace_id, config = _new_trace(guard, config, input)
            token = _PG_TRACE_DEPTH.set(1)
            guard._on_trace_start(trace_id, input)
            try:
                result = _orig_invoke(input, config=config, **kwargs)
                # Modern LangGraph CompiledStateGraph does not fire an
                # AgentFinish-style callback, so we synthesize TERMINATE on
                # clean return ourselves. Without this, FM-3.1 has nothing to
                # fire on. See tests/test_langgraph_adapter.py.
                _emit_terminate(guard, trace_id, result)
                guard._on_trace_end(trace_id, result)
                return result
            except Exception as e:
                guard._on_trace_error(trace_id, e)
                raise
            finally:
                _PG_TRACE_DEPTH.reset(token)

        def _stream(input, config=None, **kwargs):
            # Same re-entry guard as _invoke — applied at the top of the
            # generator so an inner call yields chunks straight from the
            # original without a second trace.
            if _PG_TRACE_DEPTH.get() > 0:
                yield from _orig_stream(input, config=config, **kwargs)
                return

            trace_id, config = _new_trace(guard, config, input)
            token = _PG_TRACE_DEPTH.set(1)
            guard._on_trace_start(trace_id, input)
            try:
                for chunk in _orig_stream(input, config=config, **kwargs):
                    yield chunk
                # Synthesized TERMINATE — fires only on clean exhaustion.
                # An early break raises GeneratorExit (a BaseException, not an
                # Exception), so we skip TERMINATE in that path; same goes for
                # any exception raised mid-stream (caught below).
                _emit_terminate(guard, trace_id, None)
                guard._on_trace_end(trace_id, None)
            except Exception as e:
                guard._on_trace_error(trace_id, e)
                raise
            finally:
                _PG_TRACE_DEPTH.reset(token)

        graph.invoke = _invoke
        graph.stream = _stream


# ── helpers ───────────────────────────────────────────────────────────────────

def _new_trace(guard, config, input_data):
    trace_id = str(uuid.uuid4())
    cb = _PGCallbackHandler(guard, trace_id)
    config = dict(config or {})
    config.setdefault("callbacks", [])
    config["callbacks"] = list(config["callbacks"]) + [cb]
    return trace_id, config


def _emit_terminate(guard, trace_id: str, result: Any):
    """Synthesize a TERMINATE event at the end of a clean run. Modern
    LangGraph CompiledStateGraph does not surface an AgentFinish callback,
    so without this the FM-3.1 PrematureTermination detector would never
    be triggered."""
    content = _stringify_result(result)
    event = AgentEvent(
        trace_id   = trace_id,
        span_id    = "terminate",
        event_type = EventType.TERMINATE,
        agent_name = "agent",
        content    = content,
    )
    guard._emit(event)


def _stringify_result(result: Any) -> str:
    """Best-effort extraction of the agent's final answer text from a
    LangGraph result dict. Handles the common {"messages": [...]} state
    shape, including Gemini's list-of-content-blocks message format."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result[:2000]
    if isinstance(result, dict):
        msgs = result.get("messages")
        if msgs:
            last = msgs[-1]
            content = (
                getattr(last, "content", None)
                or (last.get("content") if isinstance(last, dict) else None)
            )
            if content:
                if isinstance(content, list):
                    # Gemini-style content blocks: [{"type": "text", "text": "..."}, ...]
                    parts = []
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            parts.append(blk.get("text", ""))
                        elif isinstance(blk, dict):
                            parts.append(str(blk))
                        else:
                            parts.append(str(blk))
                    return "\n".join(p for p in parts if p)[:2000]
                return str(content)[:2000]
    return str(result)[:2000]


class _PGCallbackHandler:
    """
    Minimal LangChain BaseCallbackHandler-compatible object.
    Translates LangChain callback events into processguard AgentEvents.
    We avoid inheriting from BaseCallbackHandler to keep langchain optional.
    """

    def __init__(self, guard, trace_id: str):
        self.guard    = guard
        self.trace_id = trace_id
        self._step    = 0
        # LangChain expects these attributes on a callback handler
        self.ignore_llm          = False
        self.ignore_chain        = True
        self.ignore_agent        = False
        self.ignore_retriever    = True
        self.ignore_chat_model   = False

    # ── required by LangChain callback manager ───────────────────────────────

    def raise_error(self) -> bool:
        return False

    # ── tool events ──────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        run_id=None,
        parent_run_id=None,
        tags=None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        tool_name = serialized.get("name", "unknown_tool")
        try:
            import json
            tool_args = json.loads(input_str)
        except Exception:
            tool_args = {"input": input_str}

        agent_name = (metadata or {}).get("agent_name", "agent")
        self._emit(EventType.TOOL_CALL, agent_name, tool_name=tool_name, tool_args=tool_args)

    def on_tool_end(self, output: str, run_id=None, parent_run_id=None, **kwargs):
        self._emit(EventType.TOOL_RESULT, "agent", tool_result=str(output))

    def on_tool_error(self, error, run_id=None, **kwargs):
        self._emit(EventType.TOOL_RESULT, "agent", tool_result=f"ERROR: {error}")

    # ── agent events ──────────────────────────────────────────────────────────

    def on_agent_action(self, action, run_id=None, **kwargs):
        tool_name  = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", {})
        if isinstance(tool_input, str):
            tool_input = {"input": tool_input}
        self._emit(EventType.TOOL_CALL, "agent", tool_name=tool_name, tool_args=tool_input)

    def on_agent_finish(self, finish, run_id=None, **kwargs):
        rv = getattr(finish, "return_values", {})
        self._emit(EventType.TERMINATE, "agent", content=str(rv))

    # ── LLM events ────────────────────────────────────────────────────────────

    def on_llm_end(self, response, run_id=None, **kwargs):
        try:
            text = response.generations[0][0].text
            if text.strip():
                self._emit(EventType.MESSAGE, "agent", content=text)
        except Exception:
            pass

    # ── stubs (LangChain callback manager calls these) ────────────────────────

    def on_llm_start(self, *a, **kw):           pass
    def on_llm_new_token(self, *a, **kw):       pass
    def on_llm_error(self, *a, **kw):           pass
    def on_chain_start(self, *a, **kw):         pass
    def on_chain_end(self, *a, **kw):           pass
    def on_chain_error(self, *a, **kw):         pass
    def on_text(self, *a, **kw):               pass
    def on_retry(self, *a, **kw):              pass
    def on_chat_model_start(self, *a, **kw):   pass

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(self, event_type: EventType, agent_name: str, **kwargs):
        self._step += 1
        event = AgentEvent(
            trace_id   = self.trace_id,
            span_id    = f"step-{self._step}",
            event_type = event_type,
            agent_name = agent_name,
            **kwargs,
        )
        self.guard._emit(event)
