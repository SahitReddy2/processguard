"""
Calibration metrics for the v0.3 LLM-judge calibration set.

Pure functions over `(expected, observed)` pairs — no LLM, no I/O. The
metric computations here are unit-tested against hand-built confusion
matrices so they're verifiable independently of the live calibration
run. The runner that produces the observations lives in
`scripts/run_calibration.py`.

`expected` and `observed` are both binary: the "fire" label of the
relevant detector (mismatch for FM-2.6, incomplete for FM-3.1) is the
positive class.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Observation:
    """One case's outcome — expected (human-labelled) vs observed (deployed
    judge's behaviour). `category` is optional and is filled at report
    time by joining against `categories.jsonl`."""
    case_id:        str
    detector:       str        # "FM-2.6" / "FM-3.1"
    expected_fire:  bool       # human said the detector should fire
    observed_fire: bool        # the deployed detector did fire
    judge_raw:     str = ""    # raw model response text (for spot-check)
    confidence:    Optional[float] = None
    category:      Optional[str] = None   # filled at report time


@dataclass
class ConfusionMatrix:
    tp: int = 0   # expected fire, observed fire
    fp: int = 0   # expected no-fire, observed fire
    fn: int = 0   # expected fire, observed no-fire
    tn: int = 0   # expected no-fire, observed no-fire

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def cohen_kappa(self) -> float:
        """Cohen's kappa for binary classification.

        κ = (p_o - p_e) / (1 - p_e)
        where p_o = observed agreement, p_e = chance agreement.

        Returns 0 when p_e == 1 (perfect chance agreement → kappa undefined;
        0 is the conventional fallback)."""
        n = self.n
        if n == 0:
            return 0.0
        p_o = self.accuracy
        # Chance-agreement: probability that both raters say "fire" times
        # probability that both say "no-fire", summed.
        p_fire_expected   = (self.tp + self.fn) / n
        p_fire_observed   = (self.tp + self.fp) / n
        p_e = (
            p_fire_expected * p_fire_observed
            + (1 - p_fire_expected) * (1 - p_fire_observed)
        )
        if p_e == 1.0:
            return 0.0
        return (p_o - p_e) / (1 - p_e)


@dataclass
class CategoryBreakdown:
    """Per-category subset of observations + its own metrics."""
    category: str
    matrix:   ConfusionMatrix = field(default_factory=ConfusionMatrix)
    case_ids: list[str]        = field(default_factory=list)


@dataclass
class DetectorMetrics:
    """All metrics for one detector's full observation set."""
    detector:   str
    matrix:     ConfusionMatrix
    by_category: dict[str, CategoryBreakdown] = field(default_factory=dict)
    flip_count: int = 0    # post-judge-review label flips, if any


def build_matrix(observations: list[Observation]) -> ConfusionMatrix:
    m = ConfusionMatrix()
    for o in observations:
        if o.expected_fire and o.observed_fire:
            m.tp += 1
        elif o.expected_fire and not o.observed_fire:
            m.fn += 1
        elif not o.expected_fire and o.observed_fire:
            m.fp += 1
        else:
            m.tn += 1
    return m


def compute_metrics(
    observations: list[Observation],
    detector:     str,
    flip_count:   int = 0,
) -> DetectorMetrics:
    """Aggregate observations into a DetectorMetrics. Per-category breakdown
    is built if any observation has a non-None `category`."""
    overall = build_matrix(observations)

    by_cat: dict[str, CategoryBreakdown] = {}
    cat_groups: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        if o.category:
            cat_groups[o.category].append(o)
    for cat, group in cat_groups.items():
        by_cat[cat] = CategoryBreakdown(
            category = cat,
            matrix   = build_matrix(group),
            case_ids = [o.case_id for o in group],
        )

    return DetectorMetrics(
        detector    = detector,
        matrix      = overall,
        by_category = by_cat,
        flip_count  = flip_count,
    )


def verdict_counts(observations: list[Observation]) -> dict[str, int]:
    """Sanity-check helper — distribution of {expected, observed} pairs."""
    c = Counter()
    for o in observations:
        c[(o.expected_fire, o.observed_fire)] += 1
    return {
        "expected_fire_observed_fire":     c[(True, True)],
        "expected_fire_observed_no_fire":  c[(True, False)],
        "expected_no_fire_observed_fire":  c[(False, True)],
        "expected_no_fire_observed_no_fire": c[(False, False)],
    }
