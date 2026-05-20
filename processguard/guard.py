from __future__ import annotations

from typing import Optional

from .core.event import AgentEvent, EventType
from .core.storage import TraceStorage
from .core.policy import PolicyEngine, PolicyAction, PolicyConfig, Detection
from .detectors.base import BaseDetector
from .detectors.step_repetition import StepRepetitionDetector
from .detectors.unaware_termination import UnawareTerminationDetector
from .detectors.no_progress_loop import NoProgressLoopDetector
from .detectors.reasoning_action_mismatch import ReasoningActionMismatchDetector
from .detectors.premature_termination import PrematureTerminationDetector


class ProcessGuard:
    """
    Runtime middleware that detects MAST multi-agent failure modes as they happen.

    Quickstart::

        import processguard
        processguard.attach(crew)          # CrewAI
        processguard.attach(graph)         # LangGraph
        result = crew.kickoff(...)

    Custom policy::

        guard = ProcessGuard(default_policy=PolicyAction.STEER)
        guard.policy.policies["FM-1.3"] = PolicyConfig(action=PolicyAction.HALT)
        guard.attach(graph)
    """

    def __init__(
        self,
        detectors: Optional[list[BaseDetector]] = None,
        policy: Optional[PolicyEngine] = None,
        default_policy: PolicyAction = PolicyAction.LOG,
        db_path: str = "processguard.db",
        llm_detectors: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            detectors:      Custom detector list. Defaults to all 5 V1 detectors.
            policy:         Custom PolicyEngine instance.
            default_policy: Default action when no per-mode policy is set.
            db_path:        SQLite file path. Use ":memory:" for tests.
            llm_detectors:  Include FM-2.6 and FM-3.1 (require anthropic). True by default.
            verbose:        Print trace start/end and detection summaries.
        """
        self.storage = TraceStorage(db_path)
        self.policy  = policy or PolicyEngine(default=default_policy)
        self.verbose = verbose

        if detectors is not None:
            self.detectors = detectors
        else:
            self.detectors: list[BaseDetector] = [
                StepRepetitionDetector(),
                UnawareTerminationDetector(),
                NoProgressLoopDetector(),
            ]
            if llm_detectors:
                self.detectors += [
                    ReasoningActionMismatchDetector(),
                    PrematureTerminationDetector(),
                ]

    # ── public API ────────────────────────────────────────────────────────────

    def attach(self, framework_object):
        """
        Hook ProcessGuard into a framework object in-place.
        Supports: CrewAI Crew, LangGraph CompiledGraph / CompiledStateGraph.
        Returns the same object (methods are patched).
        """
        obj_type = type(framework_object).__name__

        if _is_crewai(framework_object):
            from .adapters.crewai import CrewAIAdapter
            CrewAIAdapter(self).attach(framework_object)

        elif _is_langgraph(framework_object):
            from .adapters.langgraph import LangGraphAdapter
            LangGraphAdapter(self).attach(framework_object)

        else:
            raise TypeError(
                f"processguard.attach() does not recognise '{obj_type}'. "
                "Supported: CrewAI Crew, LangGraph CompiledGraph. "
                "For other frameworks call guard.emit() directly."
            )

        if self.verbose:
            names = [d.failure_name for d in self.detectors]
            print(f"[processguard] attached to {obj_type} — detectors: {', '.join(names)}")

        return framework_object

    def emit(self, event: AgentEvent) -> list[str]:
        """
        Manually emit an event (use for unsupported frameworks or raw loops).
        Returns any steer messages produced.
        """
        return self._emit(event)

    # ── adapter callbacks ────────────────────────────────────────────────────

    def _emit(self, event: AgentEvent) -> list[str]:
        self.storage.save(event)
        steers: list[str] = []
        for detector in self.detectors:
            detection = detector.observe(event)
            if detection:
                steer = self.policy.handle(detection)
                if steer:
                    steers.append(steer)
        return steers

    def _on_trace_start(self, trace_id: str, input_data):
        for detector in self.detectors:
            detector.reset(trace_id)
        # propagate task description to detectors that need it
        task = _extract_task(input_data)
        if task:
            for d in self.detectors:
                if hasattr(d, "set_task"):
                    d.set_task(trace_id, task)
        if self.verbose:
            print(f"[processguard] trace {trace_id[:8]}... started")

    def _on_trace_end(self, trace_id: str, _output):
        if self.verbose:
            n    = self.storage.count_events(trace_id)
            hits = len(self.policy.detections)
            print(f"[processguard] trace {trace_id[:8]}... ended - {n} events, {hits} detections")

    def _on_trace_error(self, trace_id: str, error: Exception):
        if self.verbose:
            print(f"[processguard] trace {trace_id[:8]}... error: {type(error).__name__}: {error}")


# ── type helpers ──────────────────────────────────────────────────────────────

def _is_crewai(obj) -> bool:
    try:
        from crewai import Crew
        return isinstance(obj, Crew)
    except ImportError:
        return type(obj).__name__ == "Crew"


_LANGGRAPH_TYPES = frozenset({
    "CompiledGraph",
    "CompiledStateGraph",
    "CompiledMessageGraph",
    "CompiledGraphWithCheckpointer",
})


def _is_langgraph(obj) -> bool:
    return type(obj).__name__ in _LANGGRAPH_TYPES


def _extract_task(input_data) -> Optional[str]:
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ("task", "input", "goal", "objective", "query", "topic"):
            if key in input_data:
                return str(input_data[key])
        msgs = input_data.get("messages", [])
        if msgs and isinstance(msgs, list):
            first = msgs[0]
            content = first.get("content") if isinstance(first, dict) else getattr(first, "content", None)
            if content:
                return str(content)
    return None
