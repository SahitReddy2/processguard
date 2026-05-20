from __future__ import annotations

import uuid
from typing import Any, Optional

from .base import BaseAdapter
from ..core.event import AgentEvent, EventType


class CrewAIAdapter(BaseAdapter):
    """
    Adapter for CrewAI Crew objects.

    Wraps crew.kickoff() and injects a step_callback that converts each
    agent step into AgentEvents.  Also wraps each agent's tool calls by
    monkey-patching the crew's tool list when possible.

    Usage (automatic via processguard.attach):
        import processguard
        processguard.attach(crew)
        result = crew.kickoff(inputs={"topic": "RAG"})
    """

    def __init__(self, guard):
        self.guard    = guard
        self._trace_id: Optional[str] = None

    def attach(self, crew):
        self._patch_kickoff(crew)
        return crew

    # ── patching ─────────────────────────────────────────────────────────────

    def _patch_kickoff(self, crew):
        guard   = self.guard
        adapter = self

        _orig = crew.kickoff

        def _kickoff(inputs: Optional[dict] = None):
            trace_id = str(uuid.uuid4())
            adapter._trace_id = trace_id

            # capture original step callback (user may have set one)
            _user_step_cb = getattr(crew, "step_callback", None)

            def _step_cb(step_output):
                adapter._on_step(step_output, trace_id)
                if _user_step_cb:
                    _user_step_cb(step_output)

            crew.step_callback = _step_cb
            guard._on_trace_start(trace_id, inputs or {})

            try:
                result = _orig(inputs=inputs)
                guard._on_trace_end(trace_id, result)
                return result
            except Exception as e:
                guard._on_trace_error(trace_id, e)
                raise

        crew.kickoff = _kickoff

    # ── step handler ─────────────────────────────────────────────────────────

    def _on_step(self, step_output: Any, trace_id: str):
        """
        Convert a CrewAI step output into AgentEvents.

        CrewAI's step_callback receives different objects depending on version:
        - v0.80+  : TaskOutput or AgentFinish-like with .result / .output
        - Older   : LangChain AgentAction / AgentFinish

        We handle both gracefully.
        """
        agent_name = _agent_name(step_output)

        # ── tool call (intermediate step) ──────────────────────────────────
        action = getattr(step_output, "action", None)
        if action:
            tool_name  = getattr(action, "tool", None)
            tool_input = getattr(action, "tool_input", {})
            if isinstance(tool_input, str):
                tool_input = {"input": tool_input}

            if tool_name:
                self._emit(trace_id, EventType.TOOL_CALL, agent_name,
                           tool_name=tool_name, tool_args=tool_input)

        # ── tool result / observation ──────────────────────────────────────
        observation = getattr(step_output, "observation", None)
        if observation:
            self._emit(trace_id, EventType.TOOL_RESULT, agent_name,
                       tool_result=str(observation))

        # ── final output / message ─────────────────────────────────────────
        output = (
            getattr(step_output, "output", None)
            or getattr(step_output, "result", None)
            or getattr(step_output, "return_values", {})
        )
        if output:
            text = output if isinstance(output, str) else str(output)
            self._emit(trace_id, EventType.MESSAGE, agent_name, content=text)

        # ── termination marker ─────────────────────────────────────────────
        if _is_finish(step_output):
            self._emit(trace_id, EventType.TERMINATE, agent_name,
                       content=str(output or ""))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _emit(self, trace_id: str, event_type: EventType, agent_name: str, **kwargs):
        event = AgentEvent(
            trace_id   = trace_id,
            span_id    = str(uuid.uuid4()),
            event_type = event_type,
            agent_name = agent_name,
            **kwargs,
        )
        self.guard._emit(event)


# ── module-level helpers ──────────────────────────────────────────────────────

def _agent_name(step_output: Any) -> str:
    agent = getattr(step_output, "agent", None)
    if agent:
        return getattr(agent, "role", None) or getattr(agent, "name", "agent")
    return "agent"


def _is_finish(step_output: Any) -> bool:
    cls_name = type(step_output).__name__.lower()
    return "finish" in cls_name or "output" in cls_name
