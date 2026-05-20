from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .base import BaseDetector
from ..core.event import AgentEvent, EventType
from ..core.policy import Detection


class UnawareTerminationDetector(BaseDetector):
    """
    FM-1.5 — Unaware of Termination.

    Fires when an agent exceeds its step budget AND the last N tool calls are all
    the same type (converged / stuck), indicating it is looping without a plan to stop.
    """

    failure_mode = "FM-1.5"
    failure_name = "unaware_termination"

    def __init__(self, step_budget: int = 30, convergence_window: int = 5):
        self.step_budget = step_budget
        self.convergence_window = convergence_window
        self._steps:      dict[str, int]        = defaultdict(int)
        self._tool_hist:  dict[str, list[str]]  = defaultdict(list)
        self._fired:      set[str]              = set()

    def observe(self, event: AgentEvent) -> Optional[Detection]:
        if event.event_type not in (EventType.TOOL_CALL, EventType.MESSAGE):
            return None

        key = f"{event.trace_id}:{event.agent_name}"
        self._steps[key] += 1

        if event.event_type == EventType.TOOL_CALL and event.tool_name:
            self._tool_hist[key].append(event.tool_name)

        if self._steps[key] <= self.step_budget or key in self._fired:
            return None

        recent = self._tool_hist[key][-self.convergence_window:]
        if len(recent) >= self.convergence_window and len(set(recent)) == 1:
            self._fired.add(key)
            return Detection(
                failure_mode=self.failure_mode,
                failure_name=self.failure_name,
                trace_id=event.trace_id,
                agent_name=event.agent_name,
                confidence=0.8,
                evidence={
                    "step_count":         self._steps[key],
                    "budget":             self.step_budget,
                    "converged_tool":     recent[-1],
                    "convergence_window": self.convergence_window,
                },
                steer_message=(
                    "You have exceeded the step budget without terminating. "
                    "Summarize your findings and end the task."
                ),
            )

        return None

    def reset(self, trace_id: str):
        for d in (self._steps, self._tool_hist):
            for k in [k for k in d if k.startswith(f"{trace_id}:")]:
                del d[k]
        for k in [k for k in self._fired if k.startswith(f"{trace_id}:")]:
            self._fired.discard(k)
