"""
LLM-judge evaluator — checks whether one of the two LLM-judge detectors
(FM-2.6 ReasoningActionMismatch, FM-3.1 PrematureTermination) emits the
verdict a human labeller pre-recorded for the case.

This is the integration point for v0.3's calibration set. A calibration
case is a normal manual EvalCase (events constructed for the harness to
emit) plus exactly one `AssertJudgeVerdict` assertion carrying the
human-assigned `expected` verdict. The harness emits the events through
the guard, the judge detector observes them and either fires or doesn't,
and this evaluator translates the resulting Detection (or absence) into
PASSED / FAILED.

A "fire" verdict means the detector said the failure mode is present:
- FM-2.6 fire = MISMATCH
- FM-3.1 fire = INCOMPLETE

A "no-fire" verdict means the detector said the run is fine:
- FM-2.6 no-fire = MATCH
- FM-3.1 no-fire = COMPLETE

The evaluator records the verdict, the detection's confidence (if any),
and the judge's raw response text (extracted from the Detection's
evidence dict by the underlying detector) into the EvalResult evidence
field. The downstream calibration script (Phase B) reads these per-case
records to compute kappa, accuracy, and per-category breakdowns.
"""
from __future__ import annotations

from typing import Any, Optional

from .base     import Evaluator, EvalResult, EvalStatus
from .registry import register
from ..core.event  import AgentEvent
from ..core.policy import Detection


# Map detector failure_mode to the two verdict labels and which one means
# "the detector fired."
_FIRE_VERDICTS = {
    "FM-2.6": ("match",    "mismatch"),   # (no-fire, fire)
    "FM-3.1": ("complete", "incomplete"),
}


@register
class AssertJudgeVerdict(Evaluator):
    """
    Passes iff the named LLM-judge detector's verdict on the case matches
    `expected`.

    Args:
        detector: failure_mode string for the detector under test.
                  Must be one of: "FM-2.6", "FM-3.1".
        expected: the verdict the human labeller pre-recorded.
                  - For FM-2.6: "match" or "mismatch".
                  - For FM-3.1: "complete" or "incomplete".

    "Match" / "complete" mean the detector should NOT fire on this case;
    "mismatch" / "incomplete" mean it SHOULD fire.

    The evaluator records the detector's actual verdict, confidence, and
    raw judge response text in the EvalResult's `evidence` dict so the
    calibration script can compute per-case agreement metrics without
    re-running the judge.

    Edge cases:
    - The judge detector wraps its LLM call in a broad `except Exception:
      return None`, so network errors and malformed responses are silently
      treated as "no detection." That means the verdict captured here will
      be "match" / "complete" even when the call genuinely failed. The
      calibration script's report should surface raw judge text per-case
      so spot-inspection can distinguish a true "match" verdict from a
      silent error.
    - The detector's `confidence_floor` (default 0.5) suppresses
      low-confidence detections. If the judge returns "MISMATCH 3" the
      detection is suppressed and the verdict captured here is "match",
      which is the detector's actual production behaviour. This is
      intentional — calibration measures the deployed prompt, not the
      raw LLM output.
    """

    assertion_type = "AssertJudgeVerdict"

    def __init__(self, detector: str, expected: str):
        if detector not in _FIRE_VERDICTS:
            raise ValueError(
                f"detector must be one of {sorted(_FIRE_VERDICTS)}; "
                f"got {detector!r}"
            )
        no_fire, fire = _FIRE_VERDICTS[detector]
        if expected not in (no_fire, fire):
            raise ValueError(
                f"expected for {detector} must be one of "
                f"{[no_fire, fire]}; got {expected!r}"
            )
        self.detector = detector
        self.expected = expected
        self._fire_label    = fire
        self._no_fire_label = no_fire

    def check(
        self,
        events:     list[AgentEvent],
        detections: list[Detection],
    ) -> EvalResult:
        # Did the named detector fire on this case?
        matching = [d for d in detections if d.failure_mode == self.detector]
        fired    = bool(matching)
        verdict  = self._fire_label if fired else self._no_fire_label

        expected_to_fire = (self.expected == self._fire_label)
        passed           = (fired == expected_to_fire)

        evidence: dict[str, Any] = {
            "detector":         self.detector,
            "expected":         self.expected,
            "verdict":          verdict,
            "judge_fired":      fired,
        }
        if matching:
            d = matching[0]
            evidence["confidence"]   = d.confidence
            evidence["judge_raw"]    = d.evidence.get("judge_verdict", "")
            # Include the buffered context the detector saw (reasoning or
            # task), trimmed, so a human reviewing the report can spot-check
            # what the judge actually evaluated.
            for k in ("reasoning_preview", "action", "task", "output_preview"):
                if k in d.evidence:
                    evidence[k] = d.evidence[k]

        if passed:
            msg = f"judge verdict={verdict!r} matches expected"
        else:
            msg = (
                f"judge verdict={verdict!r}, expected={self.expected!r} — "
                f"detector {'fired' if fired else 'did not fire'} "
                f"but should have {'fired' if expected_to_fire else 'not fired'}"
            )

        return EvalResult(
            assertion_type = self.assertion_type,
            status         = EvalStatus.PASSED if passed else EvalStatus.FAILED,
            message        = msg,
            evidence       = evidence,
        )
