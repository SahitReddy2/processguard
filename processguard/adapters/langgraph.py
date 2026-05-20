from __future__ import annotations

import uuid
from typing import Any, Optional

from .base import BaseAdapter
from ..core.event import AgentEvent, EventType


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
            trace_id, config = _new_trace(guard, config, input)
            guard._on_trace_start(trace_id, input)
            try:
                result = _orig_invoke(input, config=config, **kwargs)
                guard._on_trace_end(trace_id, result)
                return result
            except Exception as e:
                guard._on_trace_error(trace_id, e)
                raise

        def _stream(input, config=None, **kwargs):
            trace_id, config = _new_trace(guard, config, input)
            guard._on_trace_start(trace_id, input)
            try:
                for chunk in _orig_stream(input, config=config, **kwargs):
                    yield chunk
                guard._on_trace_end(trace_id, None)
            except Exception as e:
                guard._on_trace_error(trace_id, e)
                raise

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
