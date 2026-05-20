from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class PolicyAction(str, Enum):
    LOG       = "log"
    STEER     = "steer"
    HALT      = "halt"
    ESCALATE  = "escalate"


@dataclass
class Detection:
    """A failure-mode detection result emitted by a detector."""
    failure_mode:  str           # e.g. "FM-1.3"
    failure_name:  str           # e.g. "step_repetition"
    trace_id:      str
    agent_name:    str
    confidence:    float         # 0.0 – 1.0
    evidence:      dict[str, Any] = field(default_factory=dict)
    steer_message: Optional[str]  = None


class ProcessGuardError(RuntimeError):
    """Raised when the HALT policy fires."""
    def __init__(self, detection: Detection):
        self.detection = detection
        super().__init__(
            f"[processguard] {detection.failure_mode} {detection.failure_name} "
            f"detected in agent '{detection.agent_name}' — run halted.\n"
            f"Evidence: {detection.evidence}"
        )


@dataclass
class PolicyConfig:
    """Per-failure-mode policy override."""
    action:       PolicyAction                        = PolicyAction.LOG
    steer_message: Optional[str]                     = None
    on_detection: Optional[Callable[[Detection], None]] = None
    enabled:      bool                               = True


class PolicyEngine:
    """Decides what to do when a detector fires."""

    _DEFAULT_STEER: dict[str, str] = {
        "FM-1.3": (
            "You are repeating the same action. Change strategy — "
            "try a different tool or different arguments."
        ),
        "FM-1.5": (
            "You have exceeded the step budget without completing the task. "
            "Summarize your findings and terminate."
        ),
        "FM-2.6": (
            "Your stated intent does not match the action you took. "
            "Review your plan and correct course."
        ),
        "FM-3.1": (
            "The original task goals have not been fully addressed. "
            "Continue working before terminating."
        ),
        "no_progress_loop": (
            "Your recent tool calls are returning no new information. "
            "Switch tools or change your approach."
        ),
    }

    def __init__(
        self,
        policies: Optional[dict[str, PolicyConfig]] = None,
        default: PolicyAction = PolicyAction.LOG,
    ):
        self.policies: dict[str, PolicyConfig] = policies or {}
        self.default_action = default
        self._log: list[Detection] = []

    def handle(self, detection: Detection) -> Optional[str]:
        """
        Process a detection. Returns a steer message string if action is STEER, else None.
        Raises ProcessGuardError if action is HALT.
        """
        policy = (
            self.policies.get(detection.failure_mode)
            or self.policies.get(detection.failure_name)
        )

        if policy and not policy.enabled:
            return None

        action = policy.action if policy else self.default_action

        self._log.append(detection)
        self._print(detection, action)

        if action == PolicyAction.HALT:
            if policy and policy.on_detection:
                policy.on_detection(detection)
            raise ProcessGuardError(detection)

        if action == PolicyAction.STEER:
            if policy and policy.on_detection:
                policy.on_detection(detection)
            return (
                (policy.steer_message if policy else None)
                or detection.steer_message
                or self._DEFAULT_STEER.get(detection.failure_mode)
                or self._DEFAULT_STEER.get(detection.failure_name)
                or "Adjust your approach."
            )

        if action == PolicyAction.ESCALATE:
            if policy and policy.on_detection:
                policy.on_detection(detection)
            return None

        # LOG: optionally notify but take no other action
        if policy and policy.on_detection:
            policy.on_detection(detection)

        return None

    @property
    def detections(self) -> list[Detection]:
        return list(self._log)

    def _print(self, d: Detection, action: PolicyAction):
        print(
            f"  [!] {d.failure_mode} {d.failure_name} detected "
            f"(confidence={d.confidence:.2f}, action={action.value})"
        )
        for k, v in d.evidence.items():
            print(f"      {k}: {v}")
