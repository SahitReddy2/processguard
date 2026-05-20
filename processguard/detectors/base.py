from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..core.event import AgentEvent
from ..core.policy import Detection


class BaseDetector(ABC):
    """Base class for all MAST failure-mode detectors."""

    failure_mode: str = ""
    failure_name: str = ""

    @abstractmethod
    def observe(self, event: AgentEvent) -> Optional[Detection]:
        """Called for every event. Returns Detection if a failure mode fires, else None."""
        ...

    def reset(self, trace_id: str):
        """Clear per-trace state. Called at the start of each new run."""
        pass
