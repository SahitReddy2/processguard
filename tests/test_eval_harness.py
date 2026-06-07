"""End-to-end tests for the Harness: case loading, manual event push, SKIP
behaviour for missing env vars, ERROR for assertion-resolution failure."""
from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent

import pytest

from processguard.harness import (
    Assertion,
    EvalCase,
    Harness,
    load_cases,
)
from processguard.harness.runner import CaseStatus


def _manual_case(case_id: str, events: list[dict], assertions: list[dict],
                 requires_env: list[str] | None = None) -> EvalCase:
    return EvalCase(
        id           = case_id,
        source       = "test",
        notes        = "",
        agent_target = "manual",
        input        = {"events": events},
        assertions   = [Assertion(type=a["type"], args=a.get("args", {})) for a in assertions],
        requires_env = requires_env or [],
    )


# ── manual-event path ────────────────────────────────────────────────────────

def test_manual_case_passes_when_detector_fires():
    case = _manual_case(
        case_id    = "smoke-1",
        events     = [
            {"type": "tool_call", "tool_name": "web_search", "tool_args": {"q": "x"}}
        ] * 3,
        assertions = [{"type": "AssertDetectorFired", "args": {"failure_mode": "FM-1.3"}}],
    )
    [result] = Harness([case]).run()
    assert result.status == CaseStatus.PASSED
    assert result.detection_count >= 1
    assert all(a.status.value == "passed" for a in result.assertion_results)


def test_manual_case_fails_when_assertion_fails():
    case = _manual_case(
        case_id    = "smoke-2",
        events     = [
            {"type": "tool_call", "tool_name": "web_search", "tool_args": {"q": "x"}}
        ],
        assertions = [{"type": "AssertDetectorFired", "args": {"failure_mode": "FM-1.3"}}],
    )
    [result] = Harness([case]).run()
    assert result.status == CaseStatus.FAILED


# ── SKIP behaviour ───────────────────────────────────────────────────────────

def test_case_with_missing_env_is_skipped_not_failed(monkeypatch):
    monkeypatch.delenv("FAKE_KEY_XYZ", raising=False)
    case = _manual_case(
        case_id      = "needs-env",
        events       = [],
        assertions   = [{"type": "AssertDetectorFired", "args": {"failure_mode": "FM-1.3"}}],
        requires_env = ["FAKE_KEY_XYZ"],
    )
    [result] = Harness([case]).run()
    assert result.status == CaseStatus.SKIPPED
    assert "FAKE_KEY_XYZ" in (result.skip_reason or "")


def test_case_with_env_present_does_not_skip(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_XYZ", "anything")
    case = _manual_case(
        case_id      = "needs-env-set",
        events       = [{"type": "tool_call", "tool_name": "x", "tool_args": {"q": "x"}}] * 3,
        assertions   = [{"type": "AssertDetectorFired", "args": {"failure_mode": "FM-1.3"}}],
        requires_env = ["FAKE_KEY_XYZ"],
    )
    [result] = Harness([case]).run()
    assert result.status == CaseStatus.PASSED


# ── ERROR behaviour ──────────────────────────────────────────────────────────

def test_unknown_assertion_type_yields_error_not_failed():
    case = _manual_case(
        case_id    = "bad-assertion",
        events     = [],
        assertions = [{"type": "AssertDefinitelyNotARealEvaluator"}],
    )
    [result] = Harness([case]).run()
    assert result.status == CaseStatus.ERROR
    assert "AssertDefinitelyNotARealEvaluator" in (result.error_message or "")


# ── JSONL loader ─────────────────────────────────────────────────────────────

def test_load_cases_parses_valid_jsonl(tmp_path: Path):
    p = tmp_path / "cases.jsonl"
    p.write_text(dedent("""\
        {"id": "a", "agent_target": "manual", "input": {"events": []}, "assertions": [], "source": "test", "notes": ""}
        {"id": "b", "agent_target": "manual", "input": {"events": []}, "assertions": [], "source": "test", "notes": ""}
    """), encoding="utf-8")
    cases = load_cases(p)
    assert [c.id for c in cases] == ["a", "b"]


def test_load_cases_skips_comments_and_blank_lines(tmp_path: Path):
    p = tmp_path / "cases.jsonl"
    p.write_text(dedent("""\

        # a comment
        // another comment
        {"id": "real", "agent_target": "manual", "input": {"events": []}, "assertions": [], "source": "", "notes": ""}

    """), encoding="utf-8")
    cases = load_cases(p)
    assert len(cases) == 1 and cases[0].id == "real"


def test_load_cases_reports_invalid_json_with_line_number(tmp_path: Path):
    p = tmp_path / "cases.jsonl"
    p.write_text('{"id": "ok", "agent_target": "manual", "input": {"events": []}, "assertions": [], "source": "", "notes": ""}\n{this is not json}\n', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_cases(p)
    msg = str(excinfo.value)
    assert ":2:" in msg  # line number is in the error


def test_load_cases_reports_missing_required_field(tmp_path: Path):
    p = tmp_path / "cases.jsonl"
    p.write_text('{"agent_target": "manual"}\n', encoding="utf-8")  # no 'id'
    with pytest.raises(ValueError) as excinfo:
        load_cases(p)
    assert "id" in str(excinfo.value)
