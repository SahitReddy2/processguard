"""
Tests for the calibration metric computations. All pure functions over
hand-built observation lists — no LLM, no I/O. These cover the numbers
that go into docs/calibration.md, so getting them wrong silently would
be the worst kind of bug.
"""
from __future__ import annotations

import pytest

from processguard.harness.calibration import (
    Observation,
    ConfusionMatrix,
    build_matrix,
    compute_metrics,
    verdict_counts,
)


def _obs(case_id: str, expected: bool, observed: bool, category: str | None = None) -> Observation:
    return Observation(
        case_id        = case_id,
        detector       = "FM-2.6",
        expected_fire  = expected,
        observed_fire = observed,
        judge_raw     = "",
        category      = category,
    )


# ── confusion-matrix counts ─────────────────────────────────────────────────

def test_empty_observation_list_yields_zero_matrix():
    m = build_matrix([])
    assert m.tp == m.fp == m.fn == m.tn == 0
    assert m.n == 0
    assert m.accuracy == 0.0


def test_all_true_positives():
    obs = [_obs(f"c{i}", True, True) for i in range(5)]
    m = build_matrix(obs)
    assert m.tp == 5
    assert m.fp == m.fn == m.tn == 0
    assert m.n == 5
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall    == 1.0
    assert m.f1        == 1.0


def test_all_true_negatives():
    obs = [_obs(f"c{i}", False, False) for i in range(5)]
    m = build_matrix(obs)
    assert m.tn == 5
    assert m.tp == m.fp == m.fn == 0
    assert m.accuracy == 1.0
    # precision/recall/F1 are 0 by convention when there are no positives
    assert m.precision == 0.0
    assert m.recall    == 0.0
    assert m.f1        == 0.0


def test_mixed_outcomes_count_into_correct_cells():
    obs = [
        _obs("a", True,  True),    # tp
        _obs("b", True,  False),   # fn
        _obs("c", False, True),    # fp
        _obs("d", False, False),   # tn
        _obs("e", True,  True),    # tp
    ]
    m = build_matrix(obs)
    assert m.tp == 2
    assert m.fn == 1
    assert m.fp == 1
    assert m.tn == 1
    assert m.n  == 5
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.precision == pytest.approx(2 / 3)      # 2 tp / (2 tp + 1 fp)
    assert m.recall    == pytest.approx(2 / 3)      # 2 tp / (2 tp + 1 fn)
    f1_expected = 2 * (2/3) * (2/3) / ((2/3) + (2/3))
    assert m.f1 == pytest.approx(f1_expected)


# ── Cohen's kappa ───────────────────────────────────────────────────────────

def test_kappa_perfect_agreement_is_one():
    obs = [_obs(f"c{i}", True if i % 2 == 0 else False,
                          True if i % 2 == 0 else False)
           for i in range(10)]
    m = build_matrix(obs)
    assert m.cohen_kappa == pytest.approx(1.0)


def test_kappa_chance_agreement_is_zero():
    """When observed agreement equals chance agreement, kappa is 0. For a
    50/50 marginal distribution with random independence, p_o = p_e = 0.5,
    so kappa = 0."""
    # Construct a 2x2 with p_o = 0.5 and balanced marginals:
    # 2 tp, 2 tn, 2 fp, 2 fn → tp+tn = 4/8 = 0.5
    obs = (
        [_obs(f"a{i}", True,  True)  for i in range(2)] +
        [_obs(f"b{i}", True,  False) for i in range(2)] +
        [_obs(f"c{i}", False, True)  for i in range(2)] +
        [_obs(f"d{i}", False, False) for i in range(2)]
    )
    m = build_matrix(obs)
    # Marginals: expected=fire 4/8 = 0.5; observed=fire 4/8 = 0.5
    # p_e = 0.5*0.5 + 0.5*0.5 = 0.5
    # p_o = (2+2)/8 = 0.5
    # κ = (0.5 - 0.5) / (1 - 0.5) = 0
    assert m.cohen_kappa == pytest.approx(0.0)


def test_kappa_worse_than_chance_is_negative():
    """If the observer disagrees more than chance would predict, kappa is
    negative. All TP and TN swap with FP and FN."""
    obs = (
        [_obs(f"a{i}", True,  False) for i in range(4)] +
        [_obs(f"b{i}", False, True)  for i in range(4)]
    )
    m = build_matrix(obs)
    # p_o = 0/8 = 0; marginals: expected=fire 4/8=0.5, observed=fire 4/8=0.5
    # p_e = 0.5; κ = (0 - 0.5) / (1 - 0.5) = -1.0
    assert m.cohen_kappa == pytest.approx(-1.0)


def test_kappa_uneven_marginals():
    """Known-result spot-check. 10 cases, 8 expected fire, 7 observed fire.
    Confusion: tp=6, fn=2, fp=1, tn=1.
    p_o = (6+1)/10 = 0.7
    p_e_pos = (8/10) * (7/10) = 0.56
    p_e_neg = (2/10) * (3/10) = 0.06
    p_e = 0.62
    κ = (0.7 - 0.62) / (1 - 0.62) = 0.08 / 0.38 ≈ 0.2105."""
    obs = (
        [_obs(f"a{i}", True, True)   for i in range(6)] +  # 6 tp
        [_obs(f"b{i}", True, False)  for i in range(2)] +  # 2 fn
        [_obs("c1",     False, True)] +                     # 1 fp
        [_obs("d1",     False, False)]                      # 1 tn
    )
    m = build_matrix(obs)
    assert m.tp == 6 and m.fn == 2 and m.fp == 1 and m.tn == 1
    assert m.cohen_kappa == pytest.approx(0.08 / 0.38, rel=1e-4)


def test_kappa_zero_observations_returns_zero():
    m = ConfusionMatrix()
    assert m.cohen_kappa == 0.0


# ── per-category breakdown ──────────────────────────────────────────────────

def test_per_category_groups_correctly():
    obs = [
        _obs("a", True,  True,  category="multi-step-plan"),
        _obs("b", True,  False, category="multi-step-plan"),
        _obs("c", False, False, category="semantic-drift"),
        _obs("d", False, False, category="semantic-drift"),
        _obs("e", True,  True,  category="semantic-drift"),
    ]
    metrics = compute_metrics(obs, detector="FM-2.6")

    assert metrics.matrix.n == 5
    assert sorted(metrics.by_category) == ["multi-step-plan", "semantic-drift"]

    msp = metrics.by_category["multi-step-plan"]
    assert msp.matrix.tp == 1
    assert msp.matrix.fn == 1
    assert msp.matrix.fp == 0
    assert msp.matrix.tn == 0
    assert sorted(msp.case_ids) == ["a", "b"]

    sd = metrics.by_category["semantic-drift"]
    assert sd.matrix.tp == 1
    assert sd.matrix.fn == 0
    assert sd.matrix.fp == 0
    assert sd.matrix.tn == 2
    assert sorted(sd.case_ids) == ["c", "d", "e"]


def test_per_category_omitted_when_no_categories_provided():
    obs = [_obs("a", True, True), _obs("b", False, False)]
    metrics = compute_metrics(obs, detector="FM-2.6")
    assert metrics.by_category == {}


# ── verdict_counts sanity helper ────────────────────────────────────────────

def test_verdict_counts_has_all_four_cells():
    obs = [
        _obs("a", True,  True),
        _obs("b", True,  False),
        _obs("c", False, True),
        _obs("d", False, False),
        _obs("e", True,  True),
    ]
    counts = verdict_counts(obs)
    assert counts == {
        "expected_fire_observed_fire":       2,
        "expected_fire_observed_no_fire":    1,
        "expected_no_fire_observed_fire":    1,
        "expected_no_fire_observed_no_fire": 1,
    }


# ── flip_count plumbing ─────────────────────────────────────────────────────

def test_flip_count_passes_through():
    obs = [_obs("a", True, True)]
    metrics = compute_metrics(obs, detector="FM-2.6", flip_count=3)
    assert metrics.flip_count == 3
