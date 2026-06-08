"""
--repeat / pass^k plumbing tests.

These exercise the Harness's repeat parameter on deterministic manual cases.
True flakiness (a case that passes some attempts and fails others) is
hard to provoke without nondeterminism we control; the unit tests cover
the deterministic edges and the math:

- repeat=1: behaviour unchanged from before; no `attempts` list populated.
- repeat=N on a passing case: aggregate PASSED, attempts_passed == N.
- repeat=N on a failing case: aggregate FAILED, attempts_passed == 0.
- pass^k display string formats correctly.
- Markdown report adds the column when k>1, hides it when k=1.

A real flaky run (LLM nondeterminism, network blips) would show
intermediate fractions like 3/5; that surface is observable in the
markdown and JSON via the same per-attempt list.
"""
from __future__ import annotations

import json

import pytest

from processguard.harness          import Harness, EvalReport, render_markdown
from processguard.harness.eval_case import EvalCase, Assertion
from processguard.harness.runner   import CaseStatus


def _passing_case(case_id: str = "p1") -> EvalCase:
    """A manual case that always passes: emits 1 tool_call, the only
    assertion checks the call happened ≥1 time."""
    return EvalCase(
        id            = case_id,
        source        = "test",
        notes         = "",
        agent_target  = "manual",
        input         = {"events": [
            {"type": "tool_call", "tool_name": "web_search", "tool_args": {"q": "x"}},
        ]},
        assertions    = [
            Assertion(type="AssertToolCalled", args={"tool_name": "web_search"}),
        ],
    )


def _failing_case(case_id: str = "f1") -> EvalCase:
    """A manual case that always fails: emits 1 tool_call to tool A, asserts
    that tool B was called."""
    return EvalCase(
        id            = case_id,
        source        = "test",
        notes         = "",
        agent_target  = "manual",
        input         = {"events": [
            {"type": "tool_call", "tool_name": "tool_a", "tool_args": {}},
        ]},
        assertions    = [
            Assertion(type="AssertToolCalled", args={"tool_name": "tool_b"}),
        ],
    )


# ── repeat=1 unchanged ──────────────────────────────────────────────────────

def test_repeat_one_does_not_populate_attempts_list():
    h = Harness([_passing_case()], repeat=1)
    [r] = h.run()
    assert r.status == CaseStatus.PASSED
    assert r.attempts == []           # k=1 keeps the old shape
    assert r.attempts_total  == 1
    assert r.attempts_passed == 1
    assert r.pass_at_k == ""          # no decoration in the k=1 report


def test_repeat_one_is_default():
    h = Harness([_passing_case()])
    [r] = h.run()
    assert r.attempts == []


def test_invalid_repeat_raises():
    with pytest.raises(ValueError):
        Harness([_passing_case()], repeat=0)
    with pytest.raises(ValueError):
        Harness([_passing_case()], repeat=-1)


# ── repeat=N on deterministic cases ─────────────────────────────────────────

def test_repeat_three_passing_case_runs_three_times_all_pass():
    h = Harness([_passing_case()], repeat=3)
    [r] = h.run()
    assert r.status            == CaseStatus.PASSED
    assert r.attempts_total    == 3
    assert r.attempts_passed   == 3
    assert r.pass_at_k         == "3/3"
    assert all(a.status == CaseStatus.PASSED for a in r.attempts)


def test_repeat_three_failing_case_runs_three_times_all_fail():
    h = Harness([_failing_case()], repeat=3)
    [r] = h.run()
    assert r.status            == CaseStatus.FAILED
    assert r.attempts_total    == 3
    assert r.attempts_passed   == 0
    assert r.pass_at_k         == "0/3"


def test_repeat_five_on_mixed_set_runs_each_case_five_times():
    h = Harness([_passing_case("p1"), _failing_case("f1")], repeat=5)
    results = h.run()
    p, f = results
    assert p.attempts_total == 5
    assert p.attempts_passed == 5
    assert f.attempts_total == 5
    assert f.attempts_passed == 0


# ── elapsed_seconds aggregation ─────────────────────────────────────────────

def test_aggregate_elapsed_is_sum_of_attempts():
    h = Harness([_passing_case()], repeat=4)
    [r] = h.run()
    individually = sum(a.elapsed_seconds for a in r.attempts)
    assert r.elapsed_seconds == pytest.approx(individually)


# ── pass^k in report ────────────────────────────────────────────────────────

def test_markdown_adds_pass_at_k_column_when_k_gt_one():
    h = Harness([_passing_case(), _failing_case()], repeat=2)
    results = h.run()
    report = EvalReport(case_results=results)

    md = render_markdown(report)
    assert "pass^2" in md          # header summary line mentions k
    assert "| pass^2 |" in md      # column header
    assert "2/2" in md             # passing case fraction
    assert "0/2" in md             # failing case fraction


def test_markdown_omits_pass_at_k_column_when_k_is_one():
    h = Harness([_passing_case()], repeat=1)
    results = h.run()
    report = EvalReport(case_results=results)

    md = render_markdown(report)
    assert "pass^" not in md
    assert "| pass" not in md


# ── JSON report ─────────────────────────────────────────────────────────────

def test_json_report_includes_attempts_when_k_gt_one():
    h = Harness([_passing_case()], repeat=3)
    results = h.run()
    report = EvalReport(case_results=results)
    payload = json.loads(report.to_json())

    case = payload["cases"][0]
    assert case["attempts_passed"] == 3
    assert case["attempts_total"]  == 3
    assert len(case["attempts"])   == 3
    for a in case["attempts"]:
        assert a["status"] == "passed"


def test_json_report_keeps_attempts_empty_for_k_eq_one():
    h = Harness([_passing_case()], repeat=1)
    results = h.run()
    report = EvalReport(case_results=results)
    payload = json.loads(report.to_json())

    case = payload["cases"][0]
    assert case["attempts_passed"] == 1
    assert case["attempts_total"]  == 1
    assert case["attempts"]        == []        # back-compat shape


# ── headline pass^k count in header ─────────────────────────────────────────

def test_header_reports_aggregate_pass_at_k():
    """Two passing cases + one failing case, repeated 3x. Aggregate should
    be '2/3' cases passed all k attempts."""
    cases = [_passing_case("p1"), _passing_case("p2"), _failing_case("f1")]
    h = Harness(cases, repeat=3)
    results = h.run()
    report  = EvalReport(case_results=results)

    md = render_markdown(report)
    # 2 cases passed all 3 attempts, 1 failed → "pass^3: 2/3"
    assert "pass^3: 2/3" in md
