from __future__ import annotations

import json
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    TOOL_CALL   = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE     = "message"
    REASONING   = "reasoning"
    TERMINATE   = "terminate"
    STEER       = "steer"   # injected by processguard


@dataclass
class AgentEvent:
    """
    Normalized event schema (OpenTelemetry-compatible span).
    All framework adapters normalize their native events into this shape.
    """
    trace_id:       str           # run / session ID
    span_id:        str           # step ID within the run
    event_type:     EventType
    agent_name:     str

    timestamp:      float                  = field(default_factory=time.time)
    event_id:       str                    = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str]          = None

    # tool_call / tool_result
    tool_name:   Optional[str]             = None
    tool_args:   Optional[dict[str, Any]]  = None
    tool_result: Optional[str]             = None

    # message / reasoning / steer
    content: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> Optional[str]:
        """Canonical fingerprint for tool calls used by step-repetition detection."""
        if self.event_type != EventType.TOOL_CALL or not self.tool_name:
            return None
        canon = json.dumps(self.tool_args or {}, sort_keys=True)
        return f"{self.tool_name}:{canon}"
