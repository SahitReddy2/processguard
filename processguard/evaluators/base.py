from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..core.event import AgentEvent
from ..core.policy import Detection


class EvalStatus(str, Enum):
    PASSED  = "passed"
    FAILED  = "failed"
    ERROR   = "error"     # the evaluator itself raised


@dataclass
class EvalResult:
    """One assertion's verdict against one case's collected trace."""
    assertion_type: str
    status:         EvalStatus
    message:        str
    evidence:       dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """
    Deterministic evaluator. Receives the full collected event stream and
    detection list for one EvalCase run, returns one EvalResult.

    Subclasses MUST set `assertion_type` to the registry key (the same
    string used in the case JSONL's `"type"` field). Subclasses MUST NOT
    raise on a normal failure; raising should be reserved for evaluator
    bugs.
    """

    assertion_type: str = ""

    @abstractmethod
    def check(
        self,
        events:     list[AgentEvent],
        detections: list[Detection],
    ) -> EvalResult: ...

    @classmethod
    def from_args(cls, args: Optional[dict[str, Any]] = None) -> "Evaluator":
        """Instantiate from a dict of args (as parsed from the case JSONL).
        Default implementation forwards args as kwargs."""
        return cls(**(args or {}))
