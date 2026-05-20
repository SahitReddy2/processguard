from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .base import BaseDetector
from ..core.event import AgentEvent, EventType
from ..core.policy import Detection


class UnawareTerminationDetector(BaseDetector):
    """
    FM-1.5 — Unaware of Termination.

    Identifies when an agent has lost track of its own termination criteria —
    it keeps acting past the point where a competent execution of the task
    would have produced its final answer and stopped.

    Fires when an agent has been working far longer than the task plausibly
    requires AND has settled into a narrow repertoire of actions, suggesting
    it has forgotten or never had a plan for when to declare the task done.

    Smallest meaningful case: a research agent asked to "summarize the top
    three papers on RAG" that has called web_search forty times in a row
    without ever producing a summary and is about to call it again.

    Must not fire on a legitimately long-running task where the work itself
    is genuinely large — a research agent given "summarize every paper cited
    in this 80-page review article" that has called fetch_paper sixty times
    because there really are sixty citations, and is steadily processing each
    one.

    Known limitation: an agent that has clearly lost track of when to stop
    but is alternating between two or three tools rather than fixating on
    one will slip through — the detector requires the agent to have settled
    into a single dominant action, not just a narrow set of them.
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
