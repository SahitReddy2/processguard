"""Unit tests for each deterministic evaluator. Each evaluator gets one
PASSED case and one FAILED case so the inversion is covered."""
from __future__ import annotations

import uuid
from typing import Iterable

import pytest

from processguard.core.event  import AgentEvent, EventType
from processguard.core.policy import Detection
from processguard.evaluators  import (
    AssertDetectorDidNotFire,
    AssertDetectorFired,
    AssertEventCountByType,
    AssertSingleTraceId,
    AssertToolCalled,
    AssertWithinStepBudget,
)
from processguard.evaluators.base import EvalStatus


def _evt(event_type: EventType, **kwargs) -> AgentEvent:
    return AgentEvent(
        trace_id   = kwargs.pop("trace_id", "trace-1"),
        span_id    = kwargs.pop("span_id", str(uuid.uuid4())),
        event_type = event_type,
        agent_name = kwargs.pop("agent_name", "agent"),
        **kwargs,
    )


def _det(failure_mode: str) -> Detection:
    return Detection(
        failure_mode = failure_mode,
        failure_name = failure_mode.lower().replace(".", "_").replace("-", "_"),
        trace_id     = "trace-1",
        agent_name   = "agent",
        confidence   = 0.9,
        evidence     = {},
    )


# ── AssertToolCalled ─────────────────────────────────────────────────────────

def test_tool_called_passes_when_tool_was_invoked():
    events = [_evt(EventType.TOOL_CALL, tool_name="web_search", tool_args={"q": "x"})]
    r = AssertToolCalled(tool_name="web_search").check(events, [])
    assert r.status == EvalStatus.PASSED


def test_tool_called_fails_when_tool_was_not_invoked():
    events = [_evt(EventType.TOOL_CALL, tool_name="other_tool")]
    r = AssertToolCalled(tool_name="web_search").check(events, [])
    assert r.status == EvalStatus.FAILED


def test_tool_called_respects_min_times():
    events = [
        _evt(EventType.TOOL_CALL, tool_name="web_search"),
        _evt(EventType.TOOL_CALL, tool_name="web_search"),
    ]
    assert AssertToolCalled(tool_name="web_search", min_times=2).check(events, []).status == EvalStatus.PASSED
    assert AssertToolCalled(tool_name="web_search", min_times=3).check(events, []).status == EvalStatus.FAILED


# ── AssertWithinStepBudget ───────────────────────────────────────────────────

def test_step_budget_passes_when_under():
    events = [_evt(EventType.MESSAGE)] * 3
    assert AssertWithinStepBudget(max_events=5).check(events, []).status == EvalStatus.PASSED


def test_step_budget_fails_when_over():
    events = [_evt(EventType.MESSAGE)] * 10
    assert AssertWithinStepBudget(max_events=5).check(events, []).status == EvalStatus.FAILED


# ── AssertDetectorFired ──────────────────────────────────────────────────────

def test_detector_fired_passes_when_present():
    dets = [_det("FM-1.3")]
    assert AssertDetectorFired(failure_mode="FM-1.3").check([], dets).status == EvalStatus.PASSED


def test_detector_fired_fails_when_absent():
    dets = [_det("FM-1.5")]
    r = AssertDetectorFired(failure_mode="FM-1.3").check([], dets)
    assert r.status == EvalStatus.FAILED
    # diagnostic message should show what DID fire
    assert "FM-1.5" in r.message


# ── AssertDetectorDidNotFire ─────────────────────────────────────────────────

def test_detector_did_not_fire_passes_when_absent():
    dets = [_det("FM-1.5")]
    r = AssertDetectorDidNotFire(failure_mode="FM-1.3").check([], dets)
    assert r.status == EvalStatus.PASSED


def test_detector_did_not_fire_fails_when_present():
    dets = [_det("FM-1.3")]
    r = AssertDetectorDidNotFire(failure_mode="FM-1.3").check([], dets)
    assert r.status == EvalStatus.FAILED


# ── AssertEventCountByType ───────────────────────────────────────────────────

def test_event_count_passes_in_range():
    events = [_evt(EventType.TERMINATE)] * 2
    r = AssertEventCountByType(event_type="terminate", min=1, max=3).check(events, [])
    assert r.status == EvalStatus.PASSED


def test_event_count_fails_below_min():
    events = []
    r = AssertEventCountByType(event_type="terminate", min=1).check(events, [])
    assert r.status == EvalStatus.FAILED


def test_event_count_fails_above_max():
    events = [_evt(EventType.TERMINATE)] * 4
    r = AssertEventCountByType(event_type="terminate", min=0, max=2).check(events, [])
    assert r.status == EvalStatus.FAILED


# ── AssertSingleTraceId ──────────────────────────────────────────────────────

def test_single_trace_passes_when_one_id():
    events = [_evt(EventType.MESSAGE, trace_id="t-1"), _evt(EventType.MESSAGE, trace_id="t-1")]
    assert AssertSingleTraceId().check(events, []).status == EvalStatus.PASSED


def test_single_trace_fails_when_multiple_ids():
    events = [_evt(EventType.MESSAGE, trace_id="t-1"), _evt(EventType.MESSAGE, trace_id="t-2")]
    r = AssertSingleTraceId().check(events, [])
    assert r.status == EvalStatus.FAILED
    assert "t-1" in r.message and "t-2" in r.message
