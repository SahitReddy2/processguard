from __future__ import annotations

from typing import Any, Optional

from .base import Evaluator, EvalResult, EvalStatus
from .registry import register
from ..core.event import AgentEvent, EventType
from ..core.policy import Detection


# ── tool-call assertions ─────────────────────────────────────────────────────

@register
class AssertToolCalled(Evaluator):
    """Passes iff the agent emitted at least `min_times` TOOL_CALL events with
    the given tool_name."""

    assertion_type = "AssertToolCalled"

    def __init__(self, tool_name: str, min_times: int = 1):
        self.tool_name = tool_name
        self.min_times = min_times

    def check(self, events: list[AgentEvent], detections: list[Detection]) -> EvalResult:
        hits = sum(
            1 for e in events
            if e.event_type == EventType.TOOL_CALL and e.tool_name == self.tool_name
        )
        passed = hits >= self.min_times
        return EvalResult(
            assertion_type = self.assertion_type,
            status         = EvalStatus.PASSED if passed else EvalStatus.FAILED,
            message        = (
                f"tool '{self.tool_name}' called {hits} time(s); "
                f"required ≥ {self.min_times}"
            ),
            evidence       = {"tool_name": self.tool_name, "hits": hits, "min_times": self.min_times},
        )


# ── step-budget assertion ────────────────────────────────────────────────────

@register
class AssertWithinStepBudget(Evaluator):
    """Passes iff the total event count for the run is ≤ `max_events`."""

    assertion_type = "AssertWithinStepBudget"

    def __init__(self, max_events: int):
        self.max_events = max_events

    def check(self, events: list[AgentEvent], detections: list[Detection]) -> EvalResult:
        n = len(events)
        passed = n <= self.max_events
        return EvalResult(
            assertion_type = self.assertion_type,
            status         = EvalStatus.PASSED if passed else EvalStatus.FAILED,
            message        = f"{n} event(s); budget {self.max_events}",
            evidence       = {"event_count": n, "max_events": self.max_events},
        )


# ── detector-firing assertions ───────────────────────────────────────────────

@register
class AssertDetectorFired(Evaluator):
    """Passes iff at least one Detection in `detections` has the given
    failure_mode."""

    assertion_type = "AssertDetectorFired"

    def __init__(self, failure_mode: str):
        self.failure_mode = failure_mode

    def check(self, events: list[AgentEvent], detections: list[Detection]) -> EvalResult:
        matching = [d for d in detections if d.failure_mode == self.failure_mode]
        passed   = len(matching) >= 1
        return EvalResult(
            assertion_type = self.assertion_type,
            status         = EvalStatus.PASSED if passed else EvalStatus.FAILED,
            message        = (
                f"{self.failure_mode} fired {len(matching)} time(s)"
                if passed
                else f"{self.failure_mode} did not fire "
                     f"(got: {sorted({d.failure_mode for d in detections}) or 'no detections'})"
            ),
            evidence       = {
                "failure_mode":      self.failure_mode,
                "matching_count":    len(matching),
                "all_modes_fired":   sorted({d.failure_mode for d in detections}),
            },
        )


@register
class AssertDetectorDidNotFire(Evaluator):
    """Passes iff no Detection in `detections` has the given failure_mode.
    Used for 'must not fire' contract checks (fan-out patterns, etc.)."""

    assertion_type = "AssertDetectorDidNotFire"

    def __init__(self, failure_mode: str):
        self.failure_mode = failure_mode

    def check(self, events: list[AgentEvent], detections: list[Detection]) -> EvalResult:
        matching = [d for d in detections if d.failure_mode == self.failure_mode]
        passed   = len(matching) == 0
        return EvalResult(
            assertion_type = self.assertion_type,
            status         = EvalStatus.PASSED if passed else EvalStatus.FAILED,
            message        = (
                f"{self.failure_mode} correctly did not fire"
                if passed
                else f"{self.failure_mode} fired unexpectedly "
                     f"({len(matching)} time(s))"
            ),
            evidence       = {
                "failure_mode":      self.failure_mode,
                "matching_count":    len(matching),
            },
        )


# ── event-shape assertions ───────────────────────────────────────────────────

@register
class AssertEventCountByType(Evaluator):
    """Passes iff the count of events with the given event_type falls within
    [min, max] (inclusive both sides). Used for adapter regressions like 'a
    completed run must produce ≥ 1 TERMINATE event'."""

    assertion_type = "AssertEventCountByType"

    def __init__(self, event_type: str, min: int = 0, max: Optional[int] = None):
        # event_type is a string here (e.g. "terminate") for JSON-friendliness.
        self.event_type = event_type
        self.min        = min
        self.max        = max

    def check(self, events: list[AgentEvent], detections: list[Detection]) -> EvalResult:
        count = sum(1 for e in events if e.event_type.value == self.event_type)
        upper_ok = (self.max is None) or (count <= self.max)
        lower_ok = count >= self.min
        passed   = lower_ok and upper_ok

        if passed:
            msg = f"event_type '{self.event_type}' count = {count} (within [{self.min}, {self.max}])"
        elif not lower_ok:
            msg = f"event_type '{self.event_type}' count = {count}; required ≥ {self.min}"
        else:
            msg = f"event_type '{self.event_type}' count = {count}; required ≤ {self.max}"

        return EvalResult(
            assertion_type = self.assertion_type,
            status         = EvalStatus.PASSED if passed else EvalStatus.FAILED,
            message        = msg,
            evidence       = {
                "event_type": self.event_type,
                "count":      count,
                "min":        self.min,
                "max":        self.max,
            },
        )


@register
class AssertSingleTraceId(Evaluator):
    """Passes iff all events in the collected stream share exactly one
    trace_id. Direct regression test for the v0.1.1 LangGraph adapter
    double-trace bug (Bug 1)."""

    assertion_type = "AssertSingleTraceId"

    def check(self, events: list[AgentEvent], detections: list[Detection]) -> EvalResult:
        trace_ids = {e.trace_id for e in events}
        n = len(trace_ids)
        passed = n == 1
        return EvalResult(
            assertion_type = self.assertion_type,
            status         = EvalStatus.PASSED if passed else EvalStatus.FAILED,
            message        = (
                f"exactly 1 trace_id observed"
                if passed
                else f"{n} trace_ids observed: {sorted(trace_ids)}"
            ),
            evidence       = {"trace_id_count": n, "trace_ids": sorted(trace_ids)},
        )
