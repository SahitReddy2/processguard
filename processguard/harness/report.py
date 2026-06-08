from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..evaluators.base import EvalStatus
from .runner          import CaseResult, CaseStatus


@dataclass
class EvalReport:
    """Aggregate report across a full harness run."""
    case_results:    list[CaseResult] = field(default_factory=list)
    timestamp_utc:   str              = ""
    total_seconds:   float            = 0.0
    gold_set_path:   str              = ""

    # ── derived counts ──────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.status == CaseStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.case_results if r.status == CaseStatus.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.case_results if r.status == CaseStatus.SKIPPED)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.case_results if r.status == CaseStatus.ERROR)

    @property
    def exit_code(self) -> int:
        """0 iff no FAILED and no ERROR. SKIPPED does not affect exit code."""
        return 0 if (self.failed == 0 and self.errored == 0) else 1

    @property
    def skip_reasons(self) -> dict[str, list[str]]:
        """Map of skip-reason → list of case ids. Used to surface 'why' in the
        PR comment header (e.g. 'ANTHROPIC_API_KEY not set in CI')."""
        out: dict[str, list[str]] = {}
        for r in self.case_results:
            if r.status == CaseStatus.SKIPPED and r.skip_reason:
                out.setdefault(r.skip_reason, []).append(r.case_id)
        return out

    # ── serialisation ───────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(self._json_dict(), indent=2)

    def write_json(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def _json_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "gold_set":      self.gold_set_path,
            "totals": {
                "total":   self.total,
                "passed":  self.passed,
                "failed":  self.failed,
                "skipped": self.skipped,
                "errored": self.errored,
            },
            "elapsed_seconds": self.total_seconds,
            "skip_reasons":    self.skip_reasons,
            "cases": [
                {
                    "case_id":         c.case_id,
                    "status":          c.status.value,
                    "skip_reason":     c.skip_reason,
                    "error_message":   c.error_message,
                    "elapsed_seconds": c.elapsed_seconds,
                    "event_count":     c.event_count,
                    "detection_count": c.detection_count,
                    "attempts_passed": c.attempts_passed,
                    "attempts_total":  c.attempts_total,
                    "assertions": [
                        {
                            "type":     a.assertion_type,
                            "status":   a.status.value,
                            "message":  a.message,
                            "evidence": a.evidence,
                        }
                        for a in c.assertion_results
                    ],
                    # When k>1, per-attempt detail lives here so flaky-case
                    # identification doesn't lose information. Empty list
                    # when k=1 (back-compat).
                    "attempts": [
                        {
                            "status":          a.status.value,
                            "elapsed_seconds": a.elapsed_seconds,
                            "error_message":   a.error_message,
                            "skip_reason":     a.skip_reason,
                        }
                        for a in c.attempts
                    ],
                }
                for c in self.case_results
            ],
        }


# ── markdown rendering ───────────────────────────────────────────────────────

_STATUS_EMOJI = {
    CaseStatus.PASSED:  "✅",
    CaseStatus.FAILED:  "❌",
    CaseStatus.SKIPPED: "⏭️",
    CaseStatus.ERROR:   "💥",
}


def render_markdown(report: EvalReport) -> str:
    """Markdown table suitable for printing to stdout AND for posting as a
    PR comment. The skip count is on the header line so the green badge
    can't overclaim. When the harness ran with `repeat > 1` on any case,
    an extra `pass^k` column is added showing the per-case fraction."""
    deterministic_passed = sum(
        1 for r in report.case_results
        if r.status == CaseStatus.PASSED and not _was_skip_eligible(r)
    )
    deterministic_total  = sum(
        1 for r in report.case_results
        if r.status in (CaseStatus.PASSED, CaseStatus.FAILED, CaseStatus.ERROR)
    )
    skip_count = report.skipped
    # Detect whether any case was run with k>1, in which case we surface the
    # pass^k column. A heterogeneous mix (some cases k=1, some k=N) is
    # technically possible but not produced by the current Harness; we'd
    # still display the column for cases where k>1 and leave it blank for
    # the rest, which is the right behaviour.
    show_pass_at_k = any(c.attempts for c in report.case_results)
    k_max          = max((c.attempts_total for c in report.case_results), default=1)

    header_parts = [
        f"**ProcessGuard eval gate** — {deterministic_passed}/{deterministic_total} deterministic"
    ]
    if report.failed or report.errored:
        header_parts.append("❌")
    else:
        header_parts.append("✅")
    if show_pass_at_k:
        # Aggregate pass^k across the whole gold set: cases that passed all
        # k attempts. This is the headline consistency number for the run.
        all_pass_count = sum(
            1 for c in report.case_results
            if c.attempts and c.attempts_passed == c.attempts_total
        )
        total_with_attempts = sum(1 for c in report.case_results if c.attempts)
        header_parts.append(f"• pass^{k_max}: {all_pass_count}/{total_with_attempts}")
    if skip_count:
        reasons = " • ".join(
            f"{len(ids)} case(s) SKIPPED ({reason})"
            for reason, ids in report.skip_reasons.items()
        )
        header_parts.append(f"• {reasons}")

    if show_pass_at_k:
        lines = [
            " ".join(header_parts),
            "",
            f"| Status | Case | pass^{k_max} | Notes |",
            "|--------|------|---------|-------|",
        ]
    else:
        lines = [
            " ".join(header_parts),
            "",
            "| Status | Case | Notes |",
            "|--------|------|-------|",
        ]

    for c in report.case_results:
        emoji   = _STATUS_EMOJI[c.status]
        if c.status == CaseStatus.SKIPPED:
            note = c.skip_reason or ""
        elif c.status == CaseStatus.ERROR:
            note = c.error_message or ""
        else:
            failed_asserts = [a for a in c.assertion_results if a.status != EvalStatus.PASSED]
            if failed_asserts:
                note = "; ".join(f"{a.assertion_type}: {a.message}" for a in failed_asserts)
            else:
                note = f"{c.event_count} events, {c.detection_count} detections, {c.elapsed_seconds*1000:.1f} ms"
        if show_pass_at_k:
            pak_col = c.pass_at_k or "—"
            lines.append(f"| {emoji} {c.status.value} | `{c.case_id}` | {pak_col} | {note} |")
        else:
            lines.append(f"| {emoji} {c.status.value} | `{c.case_id}` | {note} |")

    lines += [
        "",
        f"_Gold set: `{report.gold_set_path}` • run at {report.timestamp_utc} • {report.total_seconds:.2f}s total_",
    ]
    return "\n".join(lines)


def _was_skip_eligible(case_result: CaseResult) -> bool:
    """A run case that wasn't skipped — keeps the deterministic count clean
    by not including LLM-required cases when they DID run (which happens
    locally with keys set). Currently no cases are 'partially skip-eligible';
    this is a hook for future use."""
    return False
