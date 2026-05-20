from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from .base import BaseDetector
from ..core.event import AgentEvent, EventType
from ..core.policy import Detection


class StepRepetitionDetector(BaseDetector):
    """
    FM-1.3 — Step Repetition.

    Fingerprints each tool call as (tool_name, canonical_args).
    Fires when the same fingerprint appears >= threshold times inside a sliding window.
    After firing it resets the fire-lock so a steer that doesn't change behaviour
    will fire again, giving the policy engine another chance to escalate.
    """

    failure_mode = "FM-1.3"
    failure_name = "step_repetition"

    def __init__(self, window: int = 5, threshold: int = 3):
        self.window = window
        self.threshold = threshold
        # (trace_id, agent_name) -> sliding deque of fingerprints
        self._windows: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        # track which (trace:agent:fingerprint) combos have already fired
        self._fired: set[str] = set()

    def observe(self, event: AgentEvent) -> Optional[Detection]:
        if event.event_type != EventType.TOOL_CALL:
            return None

        fp = event.fingerprint()
        if not fp:
            return None

        key      = f"{event.trace_id}:{event.agent_name}"
        fire_key = f"{key}:{fp}"

        self._windows[key].append(fp)
        window_list = list(self._windows[key])
        count = window_list.count(fp)

        # When the agent switches to a different fingerprint, clear all fire-locks
        # for this key so a returning loop can fire again.
        locked_fps = {fk[len(key) + 1:] for fk in self._fired if fk.startswith(key + ":")}
        if locked_fps and fp not in locked_fps:
            for fk in [fk for fk in self._fired if fk.startswith(key + ":")]:
                self._fired.discard(fk)

        if count >= self.threshold and fire_key not in self._fired:
            self._fired.add(fire_key)
            return Detection(
                failure_mode=self.failure_mode,
                failure_name=self.failure_name,
                trace_id=event.trace_id,
                agent_name=event.agent_name,
                confidence=min(1.0, count / self.window),
                evidence={
                    "fingerprint":      fp,
                    "count_in_window":  count,
                    "window_size":      self.window,
                    "recent_calls":     window_list,
                },
                steer_message=(
                    "You are repeating the same tool call. "
                    "Change strategy — try a different tool or different arguments."
                ),
            )

        return None

    def reset(self, trace_id: str):
        for k in [k for k in self._windows if k.startswith(f"{trace_id}:")]:
            del self._windows[k]
        for k in [k for k in self._fired if k.startswith(f"{trace_id}:")]:
            self._fired.discard(k)
