from __future__ import annotations

import importlib
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..core.event   import AgentEvent, EventType
from ..core.policy  import Detection
from ..evaluators   import get as get_evaluator
from ..evaluators.base import EvalResult, EvalStatus
from ..guard        import ProcessGuard
from .eval_case     import EvalCase


class CaseStatus(str, Enum):
    PASSED  = "passed"
    FAILED  = "failed"
    SKIPPED = "skipped"
    ERROR   = "error"


@dataclass
class CaseResult:
    """The harness's verdict on one EvalCase."""
    case_id:          str
    status:           CaseStatus
    assertion_results: list[EvalResult] = field(default_factory=list)
    skip_reason:      Optional[str]      = None
    error_message:    Optional[str]      = None
    elapsed_seconds:  float              = 0.0
    event_count:      int                = 0
    detection_count:  int                = 0


class Harness:
    """Runs a list of EvalCases and collects per-case results.

    Cases with `requires_env` env vars missing are SKIPPED (not failed).
    Cases that complete are PASSED iff every assertion passes; otherwise
    FAILED. Cases that raise during agent execution are ERROR.

    Each case runs in an isolated `:memory:` SQLite ProcessGuard instance —
    one case's events cannot leak into another's.
    """

    def __init__(self, cases: list[EvalCase], *, verbose: bool = False):
        self.cases   = cases
        self.verbose = verbose

    # ── public ───────────────────────────────────────────────────────────────

    def run(self) -> list[CaseResult]:
        results: list[CaseResult] = []
        for case in self.cases:
            r = self._run_case(case)
            results.append(r)
            if self.verbose:
                print(f"  [{r.status.value.upper():7s}] {case.id}")
        return results

    # ── per-case ─────────────────────────────────────────────────────────────

    def _run_case(self, case: EvalCase) -> CaseResult:
        # 1. Skip if any required env var is missing.
        missing = [v for v in case.requires_env if not os.getenv(v)]
        if missing:
            return CaseResult(
                case_id     = case.id,
                status      = CaseStatus.SKIPPED,
                skip_reason = f"missing env: {', '.join(missing)}",
            )

        # 2. Resolve assertions up front so unknown types fail fast.
        try:
            evaluators = [
                get_evaluator(a.type, a.args) for a in case.assertions
            ]
        except Exception as e:
            return CaseResult(
                case_id       = case.id,
                status        = CaseStatus.ERROR,
                error_message = f"assertion resolution: {type(e).__name__}: {e}",
            )

        # 3. Build an isolated guard and run the agent / push manual events.
        guard = ProcessGuard(
            db_path        = ":memory:",
            llm_detectors  = False,   # eval harness keeps to deterministic detectors
            verbose        = False,
        )

        t0 = time.perf_counter()
        try:
            if case.agent_target == "manual":
                self._run_manual(guard, case)
            else:
                self._run_module_target(guard, case)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return CaseResult(
                case_id         = case.id,
                status          = CaseStatus.ERROR,
                error_message   = f"agent run: {type(e).__name__}: {e}",
                elapsed_seconds = elapsed,
            )
        elapsed = time.perf_counter() - t0

        # 4. Collect events + detections for assertion application.
        trace_ids = {e.trace_id for e in self._all_events(guard)}
        events    = self._all_events(guard)
        detections = list(guard.policy.detections)

        # 5. Apply each assertion.
        assertion_results = [ev.check(events, detections) for ev in evaluators]
        all_passed = all(r.status == EvalStatus.PASSED for r in assertion_results)

        return CaseResult(
            case_id           = case.id,
            status            = CaseStatus.PASSED if all_passed else CaseStatus.FAILED,
            assertion_results = assertion_results,
            elapsed_seconds   = elapsed,
            event_count       = len(events),
            detection_count   = len(detections),
        )

    # ── manual event push path ────────────────────────────────────────────────

    def _run_manual(self, guard: ProcessGuard, case: EvalCase):
        trace_id = str(uuid.uuid4())
        for i, ev_dict in enumerate(case.input.get("events", [])):
            event = _build_event(ev_dict, trace_id=trace_id, step_idx=i)
            guard.emit(event)

    # ── module-callable path ──────────────────────────────────────────────────

    def _run_module_target(self, guard: ProcessGuard, case: EvalCase):
        """Import `module:callable`, call it with the case's input. The
        callable is expected to return its own ProcessGuard; we then copy
        its events into the harness's isolated guard.

        Why copy: we want every case run in the harness to use a fresh
        :memory: guard for isolation, but the demo scripts construct their
        own guard. So we run the demo, then transfer its event stream into
        ours so assertions act on a clean slate."""
        target_module, _, target_attr = case.agent_target.partition(":")
        if not target_module or not target_attr:
            raise ValueError(
                f'agent_target must be "module.path:callable_name", got '
                f'"{case.agent_target}"'
            )
        mod = importlib.import_module(target_module)
        fn  = getattr(mod, target_attr, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"{target_module} has no callable '{target_attr}'"
            )

        kwargs = dict(case.input)
        external_guard: ProcessGuard = fn(**kwargs)

        # transfer events + detections
        for event in self._all_events(external_guard):
            guard.emit(event)
        for d in external_guard.policy.detections:
            guard.policy.detections.append(d)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _all_events(guard: ProcessGuard) -> list[AgentEvent]:
        """Return every event from every trace currently in this guard's
        storage. Per-case isolation means there's typically one trace per
        case, but we don't enforce that — assertions can introspect."""
        # Use the underlying connection directly because TraceStorage.get_trace
        # requires a trace_id; we want all of them.
        with guard.storage._lock:
            rows = guard.storage._conn.execute(
                "SELECT * FROM events ORDER BY timestamp"
            ).fetchall()
        return [guard.storage._row(r) for r in rows]


# ── event-dict → AgentEvent ──────────────────────────────────────────────────

def _build_event(d: dict[str, Any], trace_id: str, step_idx: int) -> AgentEvent:
    """Materialise an AgentEvent from a dict in the case's input.events list.

    Required: 'type' (string matching an EventType value).
    Optional: agent_name, tool_name, tool_args, tool_result, content.
    trace_id and span_id are auto-filled."""
    raw_type = d.get("type") or d.get("event_type")
    if not raw_type:
        raise ValueError(f"manual event dict missing 'type' field: {d!r}")
    try:
        event_type = EventType(raw_type)
    except ValueError as e:
        valid = sorted([et.value for et in EventType])
        raise ValueError(
            f"unknown event type '{raw_type}'; valid: {valid}"
        ) from e

    return AgentEvent(
        trace_id    = trace_id,
        span_id     = d.get("span_id") or f"manual-{step_idx + 1}",
        event_type  = event_type,
        agent_name  = d.get("agent_name", "agent"),
        tool_name   = d.get("tool_name"),
        tool_args   = d.get("tool_args"),
        tool_result = d.get("tool_result"),
        content     = d.get("content"),
    )
