from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from .base import BaseDetector
from ..core.event import AgentEvent, EventType
from ..core.policy import Detection


def _entities(text: str) -> frozenset[str]:
    """Lightweight entity proxy: unique content words ≥ 4 chars."""
    words = re.findall(r"\b[A-Za-z][a-z]{3,}\b", text)
    return frozenset(w.lower() for w in words)


class NoProgressLoopDetector(BaseDetector):
    """
    BEYOND-MAST — No-Progress Tool Loop.

    Tracks entity novelty across the last N tool results.
    Fires when the average fraction of new entities per result drops below
    `novelty_threshold`, meaning the agent is getting no new information.
    """

    failure_mode = "BEYOND-MAST"
    failure_name = "no_progress_loop"

    def __init__(self, window: int = 4, novelty_threshold: float = 0.05):
        self.window = window
        self.novelty_threshold = novelty_threshold
        # (trace:agent) -> list of new-entity sets per result
        self._result_sets: dict[str, list[frozenset[str]]] = defaultdict(list)
        self._seen:        dict[str, set[str]]             = defaultdict(set)
        self._fired:       set[str]                        = set()

    def observe(self, event: AgentEvent) -> Optional[Detection]:
        if event.event_type != EventType.TOOL_RESULT or not event.tool_result:
            return None

        key      = f"{event.trace_id}:{event.agent_name}"
        entities = _entities(event.tool_result)
        new_ents = entities - self._seen[key]

        self._seen[key].update(entities)
        self._result_sets[key].append(new_ents)

        recent = self._result_sets[key][-self.window:]
        if len(recent) < self.window or key in self._fired:
            return None

        total_seen = max(len(self._seen[key]), 1)
        avg_novelty = sum(len(e) for e in recent) / (self.window * total_seen)

        if avg_novelty < self.novelty_threshold:
            self._fired.add(key)
            return Detection(
                failure_mode=self.failure_mode,
                failure_name=self.failure_name,
                trace_id=event.trace_id,
                agent_name=event.agent_name,
                confidence=min(1.0, 1.0 - avg_novelty / max(self.novelty_threshold, 1e-9)),
                evidence={
                    "avg_novelty":       round(avg_novelty, 5),
                    "novelty_threshold": self.novelty_threshold,
                    "window":            self.window,
                    "recent_new_entities": [sorted(e) for e in recent],
                },
                steer_message=(
                    "Your recent tool calls are returning no new information. "
                    "Try a different tool, different search terms, or a different approach."
                ),
            )

        # unlock after novelty recovers (steer worked)
        self._fired.discard(key)
        return None

    def reset(self, trace_id: str):
        for d in (self._result_sets, self._seen):
            for k in [k for k in d if k.startswith(f"{trace_id}:")]:
                del d[k]
        for k in [k for k in self._fired if k.startswith(f"{trace_id}:")]:
            self._fired.discard(k)
