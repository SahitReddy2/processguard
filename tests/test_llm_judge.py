"""
AssertJudgeVerdict evaluator tests — Phase A of v0.3.

All tests run with no live Anthropic calls. The integration tests
monkey-patch the judge detector's `_client_lazy()` to return a fake
client whose `messages.create()` returns hand-crafted response objects.
That keeps `pytest` free, fast, and reproducible across environments
that don't have ANTHROPIC_API_KEY set.

The single live-API smoke test is intentionally not in this file; it
lives in the maintainer's manual checklist for Phase A success
criterion and costs ~$0.001 to run.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from processguard.core.event   import AgentEvent, EventType
from processguard.core.policy  import Detection
from processguard.evaluators   import (
    AssertJudgeVerdict,
    AssertionTypeNotRegistered,
    EvalStatus,
    get,
    registered_names,
)
from processguard.detectors.reasoning_action_mismatch import ReasoningActionMismatchDetector
from processguard.detectors.premature_termination     import PrematureTerminationDetector


# ── registry / construction ─────────────────────────────────────────────────

def test_assert_judge_verdict_is_registered():
    """The evaluator must be discoverable via the registry — same path the
    harness uses to instantiate it from a case JSONL."""
    assert "AssertJudgeVerdict" in registered_names()


def test_from_registry_with_args():
    ev = get("AssertJudgeVerdict", {"detector": "FM-2.6", "expected": "mismatch"})
    assert isinstance(ev, AssertJudgeVerdict)
    assert ev.detector == "FM-2.6"
    assert ev.expected == "mismatch"


def test_unknown_assertion_type_lists_registered_names():
    with pytest.raises(AssertionTypeNotRegistered) as exc:
        get("AssertJudgeVerdictTypo", {})
    msg = str(exc.value)
    # The registered-names listing is the core feature of the error message;
    # AssertJudgeVerdict must appear in it.
    assert "AssertJudgeVerdict" in msg


# ── argument validation ─────────────────────────────────────────────────────

def test_invalid_detector_name_raises():
    with pytest.raises(ValueError) as exc:
        AssertJudgeVerdict(detector="FM-99.9", expected="mismatch")
    assert "FM-99.9" in str(exc.value)
    assert "FM-2.6" in str(exc.value)


@pytest.mark.parametrize("expected", ["incomplete", "nope", ""])
def test_invalid_expected_for_ram_raises(expected: str):
    """RAM only allows 'match' / 'mismatch'."""
    with pytest.raises(ValueError) as exc:
        AssertJudgeVerdict(detector="FM-2.6", expected=expected)
    assert "match" in str(exc.value)
    assert "mismatch" in str(exc.value)


@pytest.mark.parametrize("expected", ["match", "nope", ""])
def test_invalid_expected_for_pt_raises(expected: str):
    """PT only allows 'complete' / 'incomplete'."""
    with pytest.raises(ValueError) as exc:
        AssertJudgeVerdict(detector="FM-3.1", expected=expected)
    assert "complete" in str(exc.value)
    assert "incomplete" in str(exc.value)


# ── verdict logic (pure, no detector / no LLM) ──────────────────────────────

def _mk_detection(mode: str, confidence: float = 0.8, judge_text: str = "MISMATCH 8") -> Detection:
    return Detection(
        failure_mode = mode,
        failure_name = "reasoning_action_mismatch" if mode == "FM-2.6" else "premature_termination",
        trace_id     = "t",
        agent_name   = "agent",
        confidence   = confidence,
        evidence     = {"judge_verdict": judge_text, "reasoning_preview": "I will X"},
    )


def test_ram_fired_when_expected_mismatch_passes():
    ev = AssertJudgeVerdict(detector="FM-2.6", expected="mismatch")
    result = ev.check(events=[], detections=[_mk_detection("FM-2.6")])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"]     == "mismatch"
    assert result.evidence["judge_fired"] is True
    assert result.evidence["judge_raw"]   == "MISMATCH 8"
    assert result.evidence["confidence"]  == 0.8


def test_ram_fired_when_expected_match_fails():
    """The judge wrongly flagged MISMATCH on a case the human labelled MATCH —
    a false positive. Must FAIL (this is the kind of disagreement the
    calibration set exists to measure)."""
    ev = AssertJudgeVerdict(detector="FM-2.6", expected="match")
    result = ev.check(events=[], detections=[_mk_detection("FM-2.6")])
    assert result.status == EvalStatus.FAILED
    assert result.evidence["verdict"] == "mismatch"


def test_ram_not_fired_when_expected_match_passes():
    """No detection means the judge said MATCH (or stayed silent). Human label
    is MATCH. Agreement → PASSED."""
    ev = AssertJudgeVerdict(detector="FM-2.6", expected="match")
    result = ev.check(events=[], detections=[])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"]     == "match"
    assert result.evidence["judge_fired"] is False
    # No detection means no judge_raw in evidence.
    assert "judge_raw" not in result.evidence


def test_ram_not_fired_when_expected_mismatch_fails():
    """The judge missed a real mismatch the human caught — a false negative.
    Must FAIL."""
    ev = AssertJudgeVerdict(detector="FM-2.6", expected="mismatch")
    result = ev.check(events=[], detections=[])
    assert result.status == EvalStatus.FAILED
    assert result.evidence["verdict"] == "match"


def test_pt_fired_when_expected_incomplete_passes():
    ev = AssertJudgeVerdict(detector="FM-3.1", expected="incomplete")
    result = ev.check(events=[], detections=[
        _mk_detection("FM-3.1", confidence=0.7, judge_text="INCOMPLETE 7 the task asked for 5 citations"),
    ])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"]    == "incomplete"
    assert result.evidence["judge_raw"]  == "INCOMPLETE 7 the task asked for 5 citations"


def test_pt_not_fired_when_expected_complete_passes():
    ev = AssertJudgeVerdict(detector="FM-3.1", expected="complete")
    result = ev.check(events=[], detections=[])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"] == "complete"


def test_detection_for_unrelated_detector_is_ignored():
    """A FM-1.3 detection in the list shouldn't affect a FM-2.6 verdict
    check. The evaluator filters by failure_mode."""
    ev = AssertJudgeVerdict(detector="FM-2.6", expected="match")
    irrelevant = Detection(
        failure_mode="FM-1.3", failure_name="step_repetition",
        trace_id="t", agent_name="agent", confidence=1.0, evidence={},
    )
    result = ev.check(events=[], detections=[irrelevant])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["judge_fired"] is False


# ── integration: mocked LLM, real detector, end-to-end ──────────────────────

@dataclass
class _FakeText:
    text: str

@dataclass
class _FakeResp:
    content: list[_FakeText]


def _patch_judge_client(monkeypatch, detector_cls, fake_reply: str):
    """Replace the detector class's `_client_lazy` with a fake whose
    messages.create returns a single content block with `fake_reply` text."""
    class _FakeClient:
        class messages:
            @staticmethod
            def create(*args, **kwargs):
                return _FakeResp(content=[_FakeText(text=fake_reply)])
    monkeypatch.setattr(detector_cls, "_client_lazy", lambda self: _FakeClient)


def test_ram_detector_pipeline_with_mocked_mismatch(monkeypatch):
    """End-to-end: REASONING then TOOL_CALL events through the real RAM
    detector, judge returns MISMATCH, AssertJudgeVerdict(expected=mismatch)
    PASSES."""
    _patch_judge_client(monkeypatch, ReasoningActionMismatchDetector, "MISMATCH 8")

    detector   = ReasoningActionMismatchDetector(confidence_floor=0.5)
    reasoning  = AgentEvent(
        trace_id="t", span_id="s1", event_type=EventType.REASONING,
        agent_name="agent", content="I will delegate to the writer agent.",
    )
    tool_call  = AgentEvent(
        trace_id="t", span_id="s2", event_type=EventType.TOOL_CALL,
        agent_name="agent", tool_name="web_search", tool_args={"q": "x"},
    )

    detector.observe(reasoning)
    detection = detector.observe(tool_call)
    assert detection is not None, "real RAM detector must fire on MISMATCH judge reply above confidence_floor"

    ev     = AssertJudgeVerdict(detector="FM-2.6", expected="mismatch")
    result = ev.check(events=[reasoning, tool_call], detections=[detection])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"]   == "mismatch"
    assert result.evidence["confidence"] == pytest.approx(0.8)
    assert "MISMATCH" in result.evidence["judge_raw"]


def test_ram_detector_pipeline_with_mocked_match(monkeypatch):
    """Same pipeline, judge returns MATCH (any non-MISMATCH reply). Detector
    suppresses the detection. AssertJudgeVerdict(expected=match) PASSES."""
    _patch_judge_client(monkeypatch, ReasoningActionMismatchDetector, "MATCH 9")

    detector  = ReasoningActionMismatchDetector(confidence_floor=0.5)
    detector.observe(AgentEvent(
        trace_id="t", span_id="s1", event_type=EventType.REASONING,
        agent_name="agent", content="I will search for X.",
    ))
    detection = detector.observe(AgentEvent(
        trace_id="t", span_id="s2", event_type=EventType.TOOL_CALL,
        agent_name="agent", tool_name="web_search", tool_args={"query": "X"},
    ))
    assert detection is None, "MATCH reply must not produce a Detection"

    ev     = AssertJudgeVerdict(detector="FM-2.6", expected="match")
    result = ev.check(events=[], detections=[])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"] == "match"


def test_ram_detector_pipeline_low_confidence_suppressed(monkeypatch):
    """If the judge returns MISMATCH with confidence below the floor, the
    detector suppresses. The verdict captured by the evaluator is 'match'
    — which IS the deployed prompt's actual behaviour, and is the right
    thing for calibration to measure."""
    _patch_judge_client(monkeypatch, ReasoningActionMismatchDetector, "MISMATCH 2")

    detector  = ReasoningActionMismatchDetector(confidence_floor=0.5)
    detector.observe(AgentEvent(
        trace_id="t", span_id="s1", event_type=EventType.REASONING,
        agent_name="agent", content="Reasoning.",
    ))
    detection = detector.observe(AgentEvent(
        trace_id="t", span_id="s2", event_type=EventType.TOOL_CALL,
        agent_name="agent", tool_name="x", tool_args={},
    ))
    assert detection is None, "below-floor confidence must suppress the detection"

    # Even though the model said MISMATCH, the deployed detector did not
    # fire, so the calibration sees this as a "match" verdict. A human who
    # labelled this case "mismatch" would correctly mark it as a judge
    # false-negative — which is precisely what we want the calibration
    # number to capture.
    ev     = AssertJudgeVerdict(detector="FM-2.6", expected="mismatch")
    result = ev.check(events=[], detections=[])
    assert result.status == EvalStatus.FAILED


def test_pt_detector_pipeline_with_mocked_incomplete(monkeypatch):
    _patch_judge_client(monkeypatch, PrematureTerminationDetector, "INCOMPLETE 8 missing citations")

    detector = PrematureTerminationDetector(confidence_floor=0.5)
    detector.set_task("t", "Find 5 citations and summarise each.")
    detector.observe(AgentEvent(
        trace_id="t", span_id="s1", event_type=EventType.MESSAGE,
        agent_name="agent", content="Here is a one-paragraph overview.",
    ))
    detection = detector.observe(AgentEvent(
        trace_id="t", span_id="s2", event_type=EventType.TERMINATE,
        agent_name="agent",
    ))
    assert detection is not None

    ev     = AssertJudgeVerdict(detector="FM-3.1", expected="incomplete")
    result = ev.check(events=[], detections=[detection])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"] == "incomplete"


def test_pt_detector_pipeline_with_mocked_complete(monkeypatch):
    _patch_judge_client(monkeypatch, PrematureTerminationDetector, "COMPLETE 9 fully addresses the task")

    detector = PrematureTerminationDetector(confidence_floor=0.5)
    detector.set_task("t", "What is 2+2?")
    detector.observe(AgentEvent(
        trace_id="t", span_id="s1", event_type=EventType.MESSAGE,
        agent_name="agent", content="4",
    ))
    detection = detector.observe(AgentEvent(
        trace_id="t", span_id="s2", event_type=EventType.TERMINATE,
        agent_name="agent",
    ))
    assert detection is None

    ev     = AssertJudgeVerdict(detector="FM-3.1", expected="complete")
    result = ev.check(events=[], detections=[])
    assert result.status == EvalStatus.PASSED


def test_pt_empty_output_bypass_fires_without_llm(monkeypatch):
    """The FM-3.1 detector has an explicit-bypass path for empty output: if
    the agent's last MESSAGE was empty when TERMINATE arrives, it emits a
    high-confidence detection without calling the LLM. The evaluator
    should treat this the same as any other 'incomplete' verdict."""
    # Patch the client to RAISE if called — proves we didn't go through it.
    def _explode(self):
        raise RuntimeError("LLM should not have been called on the empty-bypass path")
    monkeypatch.setattr(PrematureTerminationDetector, "_client_lazy", _explode)

    detector = PrematureTerminationDetector(confidence_floor=0.5)
    detector.set_task("t", "Write a 3-paragraph summary.")
    # No MESSAGE event emitted — agent went straight to TERMINATE with
    # nothing in the buffer.
    detection = detector.observe(AgentEvent(
        trace_id="t", span_id="s1", event_type=EventType.TERMINATE,
        agent_name="agent",
    ))
    assert detection is not None
    assert detection.confidence == pytest.approx(0.9)

    ev     = AssertJudgeVerdict(detector="FM-3.1", expected="incomplete")
    result = ev.check(events=[], detections=[detection])
    assert result.status == EvalStatus.PASSED
    assert result.evidence["verdict"] == "incomplete"
