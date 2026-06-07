from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Assertion:
    """One assertion within an EvalCase. `type` is the registry key (e.g.
    "AssertDetectorFired"); `args` are forwarded to the evaluator's
    constructor."""
    type: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCase:
    """One end-to-end test case for the eval harness.

    agent_target values:
      - "manual"                         input.events is a list of AgentEvent
                                         dicts to push through guard.emit().
      - "module.path:callable_name"      The callable is imported and called
                                         with input.task; it must return a
                                         ProcessGuard with .policy.detections
                                         and .storage populated.

    requires_env lists env vars that must be set for this case to run.
    Missing any of them causes the harness to SKIP the case (not fail it)."""
    id:           str
    source:       str
    notes:        str
    agent_target: str
    input:        dict[str, Any]
    assertions:   list[Assertion]
    requires_env: list[str] = field(default_factory=list)


# ── JSONL loader ─────────────────────────────────────────────────────────────

def load_cases(path: Path | str) -> list[EvalCase]:
    """Load EvalCases from a JSONL file. One JSON object per line."""
    path  = Path(path)
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("//") or raw.startswith("#"):
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            cases.append(_case_from_dict(d, source_hint=f"{path}:{line_no}"))
    return cases


def _case_from_dict(d: dict[str, Any], source_hint: str) -> EvalCase:
    try:
        return EvalCase(
            id           = d["id"],
            source       = d.get("source", ""),
            notes        = d.get("notes", ""),
            agent_target = d["agent_target"],
            input        = d.get("input", {}),
            assertions   = [
                Assertion(type=a["type"], args=a.get("args", {}))
                for a in d.get("assertions", [])
            ],
            requires_env = list(d.get("requires_env", [])),
        )
    except KeyError as e:
        raise ValueError(
            f"{source_hint}: missing required field {e} in case definition"
        ) from e
