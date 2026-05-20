from __future__ import annotations

import re
from typing import Optional

from .base import BaseDetector
from ..core.event import AgentEvent, EventType
from ..core.policy import Detection


class PrematureTerminationDetector(BaseDetector):
    """
    FM-3.1 — Premature Termination.

    When a TERMINATE event arrives, asks a small LLM judge whether the agent's
    last output actually addresses the original task goals.
    Requires the `anthropic` package.
    """

    failure_mode = "FM-3.1"
    failure_name = "premature_termination"

    def __init__(
        self,
        task_description: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        confidence_floor: float = 0.5,
    ):
        self.task_description = task_description
        self.model = model
        self.confidence_floor = confidence_floor
        self._tasks:       dict[str, str] = {}   # trace_id -> task text
        self._last_output: dict[str, str] = {}   # (trace:agent) -> latest message
        self._client = None

    # ── called by ProcessGuard before a run when task is known ──────────────

    def set_task(self, trace_id: str, task: str):
        self._tasks[trace_id] = task

    # ── public ──────────────────────────────────────────────────────────────

    def observe(self, event: AgentEvent) -> Optional[Detection]:
        key = f"{event.trace_id}:{event.agent_name}"

        if event.event_type == EventType.MESSAGE and event.content:
            self._last_output[key] = event.content
            return None

        if event.event_type == EventType.TERMINATE:
            task   = self._tasks.get(event.trace_id) or self.task_description or ""
            output = self._last_output.get(key, "")
            if not task:
                return None
            return self._judge(event, task, output)

        return None

    def reset(self, trace_id: str):
        self._tasks.pop(trace_id, None)
        for k in [k for k in self._last_output if k.startswith(f"{trace_id}:")]:
            del self._last_output[k]

    # ── private ─────────────────────────────────────────────────────────────

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _judge(self, event: AgentEvent, task: str, output: str) -> Optional[Detection]:
        if not output:
            return Detection(
                failure_mode=self.failure_mode,
                failure_name=self.failure_name,
                trace_id=event.trace_id,
                agent_name=event.agent_name,
                confidence=0.9,
                evidence={"task": task[:200], "output": "(empty — no output produced)"},
                steer_message=(
                    "You terminated without producing output. "
                    "Return to the task and complete it."
                ),
            )

        try:
            resp = self._client_lazy().messages.create(
                model=self.model,
                max_tokens=120,
                messages=[{
                    "role": "user",
                    "content": (
                        "Has this output fully addressed the original task?\n\n"
                        f"TASK: {task[:400]}\n\n"
                        f"OUTPUT: {output[:400]}\n\n"
                        "Reply: COMPLETE or INCOMPLETE, then confidence 0-10, then one sentence why."
                    ),
                }],
            )
            text = resp.content[0].text.strip().upper()
            if "INCOMPLETE" not in text:
                return None

            m = re.search(r"(\d+)", text)
            conf = int(m.group(1)) / 10.0 if m else 0.7
            if conf < self.confidence_floor:
                return None

            return Detection(
                failure_mode=self.failure_mode,
                failure_name=self.failure_name,
                trace_id=event.trace_id,
                agent_name=event.agent_name,
                confidence=conf,
                evidence={
                    "task":           task[:200],
                    "output_preview": output[:200],
                    "judge_verdict":  text[:200],
                },
                steer_message=(
                    "The original task goals have not been fully addressed. "
                    "Continue working before terminating."
                ),
            )
        except Exception:
            return None
