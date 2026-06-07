from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from .base import BaseDetector
from ..core.event import AgentEvent, EventType
from ..core.policy import Detection


class ReasoningActionMismatchDetector(BaseDetector):
    """
    FM-2.6 — Reasoning-Action Mismatch.

    Identifies when an agent's stated reasoning and its next action do not
    line up — the agent has just declared an intent and then does something
    materially different from what it said it would do.

    Fires whenever an agent has explicitly stated an intent for its next step
    and the next observable action does not carry out that intent, as judged
    by an independent model reading both.

    Smallest meaningful case: an agent that reasons "I will delegate this to
    the writer agent because the question is stylistic, not factual" and then
    calls a search tool instead of invoking the writer agent.

    Must not fire when the action is a faithful paraphrase or refinement of
    the stated intent — an agent that reasons "I'll look up information on X"
    and then calls web_search(query="X recent developments") is consistent
    with its plan, just more specific.

    Known limitation: the judge is itself an LLM and brings its own biases —
    in particular, when the agent's reasoning lays out a multi-step plan
    ("first I'll search, then read the top result, then summarize") but only
    the first step is visible in the action, the judge tends to flag this as
    a mismatch even though the action correctly executes step one.

    Requires the `anthropic` package.

    Also requires REASONING events to be present in the trace. The
    LangGraph adapter does NOT auto-emit these (provider callback chains
    don't expose a canonical reasoning channel that's consistent across
    Claude / Gemini / GPT). Users who want this detector to fire on a
    LangGraph run must emit REASONING events themselves via guard.emit()
    from their own LLM-wrapper instrumentation. See docs/real_run_findings.md.
    """

    failure_mode = "FM-2.6"
    failure_name = "reasoning_action_mismatch"

    def __init__(self, model: str = "claude-haiku-4-5-20251001", confidence_floor: float = 0.5):
        self.model = model
        self.confidence_floor = confidence_floor
        self._pending: dict[str, str] = {}   # (trace:agent) -> last reasoning text
        self._client = None

    # ── public ──────────────────────────────────────────────────────────────

    def observe(self, event: AgentEvent) -> Optional[Detection]:
        key = f"{event.trace_id}:{event.agent_name}"

        if event.event_type == EventType.REASONING and event.content:
            self._pending[key] = event.content
            return None

        if event.event_type == EventType.TOOL_CALL and key in self._pending:
            reasoning = self._pending.pop(key)
            action_desc = f"Called tool '{event.tool_name}' with args {event.tool_args}"
            return self._judge(event, reasoning, action_desc)

        return None

    def reset(self, trace_id: str):
        for k in [k for k in self._pending if k.startswith(f"{trace_id}:")]:
            del self._pending[k]

    # ── private ─────────────────────────────────────────────────────────────

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _judge(self, event: AgentEvent, reasoning: str, action: str) -> Optional[Detection]:
        try:
            resp = self._client_lazy().messages.create(
                model=self.model,
                max_tokens=80,
                messages=[{
                    "role": "user",
                    "content": (
                        "Does this action match the stated reasoning?\n\n"
                        f"REASONING: {reasoning[:500]}\n\n"
                        f"ACTION: {action}\n\n"
                        "Reply with exactly: MATCH or MISMATCH, then a confidence score 0-10."
                    ),
                }],
            )
            text = resp.content[0].text.strip().upper()
            if "MISMATCH" not in text:
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
                    "reasoning_preview": reasoning[:200],
                    "action":            action,
                    "judge_verdict":     text[:100],
                },
                steer_message=(
                    "Your action does not match your stated plan. "
                    "Review your reasoning and correct course."
                ),
            )
        except Exception:
            return None
