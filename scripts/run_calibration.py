#!/usr/bin/env python
"""
Run the v0.3 LLM-judge calibration.

Loads a labeled JSONL of cases, runs the relevant deployed detector against
each case with `ANTHROPIC_API_KEY` set, records each verdict (and the raw
judge response text per case), computes Cohen's kappa + accuracy +
precision/recall/F1 + per-category breakdown, and writes:
  - calibration_results/{detector}_v0.3_{UTC}.json (machine-readable)
  - calibration_results/{detector}_v0.3_{UTC}.md   (human-readable)

The script does NOT update docs/calibration.md automatically — that's a
deliberate manual step so the published number always gets a human review
before it lands in the public artifact.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/run_calibration.py --detector ram
    python scripts/run_calibration.py --detector pt
    python scripts/run_calibration.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime    import datetime, timezone
from pathlib     import Path
from typing      import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processguard.core.event   import AgentEvent, EventType
from processguard.detectors.reasoning_action_mismatch import ReasoningActionMismatchDetector
from processguard.detectors.premature_termination     import PrematureTerminationDetector
from processguard.harness.calibration import (
    Observation,
    compute_metrics,
    verdict_counts,
)


# ── label / category loaders ────────────────────────────────────────────────

def load_labels(path: Path) -> list[dict[str, Any]]:
    """Load a labeled JSONL. Each line must have a non-null `expected_verdict`
    field. Lines without one indicate an unfrozen draft and raise."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("//") or raw.startswith("#"):
                continue
            d = json.loads(raw)
            if d.get("expected_verdict") is None:
                raise ValueError(
                    f"{path}:{line_no}: case '{d.get('id')}' has no "
                    f"expected_verdict — refuse to calibrate against unlabeled "
                    f"data. Did the freeze-labels commit land?"
                )
            rows.append(d)
    return rows


def load_categories(path: Path) -> dict[str, str]:
    """Load categories.jsonl into {id: category}. Loaded only at report time,
    never during labeling — see datasets/calibration/README.md for why."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("//"):
                continue
            d = json.loads(raw)
            out[d["id"]] = d["category"]
    return out


# ── recording-client wrapper (captures judge raw text) ─────────────────────

class _RecordingClient:
    """Thin proxy around the real Anthropic client. Captures every
    response's text content into `self.last_text` so the calibration runner
    can record what the judge said — including on the no-fire path where the
    detector returns no Detection and would otherwise drop the model output.
    """

    def __init__(self, inner: Any):
        self._inner    = inner
        self.last_text = ""
        self.messages  = self._MessagesProxy(self)

    class _MessagesProxy:
        def __init__(self, outer: "_RecordingClient"):
            self._outer = outer

        def create(self, *args, **kwargs):
            resp = self._outer._inner.messages.create(*args, **kwargs)
            try:
                self._outer.last_text = resp.content[0].text
            except Exception:
                self._outer.last_text = ""
            return resp


def _make_recording_client():
    """Construct one recording client for the duration of the run. Wraps
    the standard anthropic.Anthropic() instance."""
    import anthropic
    return _RecordingClient(anthropic.Anthropic())


# ── per-case runners ────────────────────────────────────────────────────────

def run_ram_case(case: dict[str, Any], recording: _RecordingClient) -> Observation:
    """Push REASONING + TOOL_CALL events through a fresh RAM detector,
    capture verdict and the recording client's last response text."""
    detector = ReasoningActionMismatchDetector(confidence_floor=0.5)
    # Inject the recording client (bypass _client_lazy's default Anthropic()).
    detector._client = recording

    trace_id = case["id"]
    detector.observe(AgentEvent(
        trace_id   = trace_id, span_id="s1",
        event_type = EventType.REASONING, agent_name="agent",
        content    = case["reasoning"],
    ))
    recording.last_text = ""   # reset so we only capture this case's response
    detection = detector.observe(AgentEvent(
        trace_id   = trace_id, span_id="s2",
        event_type = EventType.TOOL_CALL, agent_name="agent",
        tool_name  = case["tool_name"],
        tool_args  = case["tool_args"],
    ))

    return Observation(
        case_id       = case["id"],
        detector      = "FM-2.6",
        expected_fire = case["expected_verdict"] == "mismatch",
        observed_fire = detection is not None,
        judge_raw     = recording.last_text,
        confidence    = detection.confidence if detection else None,
    )


def run_pt_case(case: dict[str, Any], recording: _RecordingClient) -> Observation:
    """Push MESSAGE + TERMINATE events through a fresh PT detector with the
    task set via set_task. Capture verdict and raw response."""
    detector = PrematureTerminationDetector(confidence_floor=0.5)
    detector._client = recording

    trace_id = case["id"]
    detector.set_task(trace_id, case["task"])
    detector.observe(AgentEvent(
        trace_id   = trace_id, span_id="s1",
        event_type = EventType.MESSAGE, agent_name="agent",
        content    = case["output"],
    ))
    recording.last_text = ""
    detection = detector.observe(AgentEvent(
        trace_id   = trace_id, span_id="s2",
        event_type = EventType.TERMINATE, agent_name="agent",
    ))

    return Observation(
        case_id       = case["id"],
        detector      = "FM-3.1",
        expected_fire = case["expected_verdict"] == "incomplete",
        observed_fire = detection is not None,
        judge_raw     = recording.last_text,
        confidence    = detection.confidence if detection else None,
    )


# ── orchestration ───────────────────────────────────────────────────────────

def run_calibration(
    detector_name: str,
    labels_path:   Path,
    categories:    dict[str, str],
    recording:     _RecordingClient,
) -> list[Observation]:
    cases   = load_labels(labels_path)
    out: list[Observation] = []
    runner  = run_ram_case if detector_name == "ram" else run_pt_case
    print(f"\nRunning {len(cases)} {detector_name.upper()} cases ...", file=sys.stderr)
    for i, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        obs = runner(case, recording)
        # Join in category from the separate file.
        obs = Observation(
            case_id        = obs.case_id,
            detector       = obs.detector,
            expected_fire  = obs.expected_fire,
            observed_fire = obs.observed_fire,
            judge_raw     = obs.judge_raw,
            confidence    = obs.confidence,
            category      = categories.get(obs.case_id),
        )
        out.append(obs)
        elapsed = time.perf_counter() - t0
        agreement = "✓" if obs.expected_fire == obs.observed_fire else "✗"
        print(
            f"  [{i:2d}/{len(cases)}] {agreement} {obs.case_id:10s} "
            f"expected={'fire' if obs.expected_fire else 'no-fire'} "
            f"observed={'fire' if obs.observed_fire else 'no-fire'} "
            f"({elapsed:.2f}s)",
            file=sys.stderr,
        )
    return out


# ── output ──────────────────────────────────────────────────────────────────

def _observations_to_json(obs: list[Observation]) -> list[dict[str, Any]]:
    return [asdict(o) for o in obs]


def render_markdown(metrics, observations: list[Observation], detector_name: str) -> str:
    """Markdown body suitable for landing in docs/calibration.md (with a
    methodology header prepended). The headline appears in the format
    'κ = 0.XX (N=..., single labeler)' per v0.3 plan §3.6."""
    m = metrics.matrix
    lines = [
        f"## {detector_name.upper()} — calibration",
        "",
        f"**κ = {m.cohen_kappa:.3f} (N={m.n}, single labeler)** • "
        f"raw accuracy {m.accuracy:.1%} • "
        f"precision {m.precision:.1%} • recall {m.recall:.1%} • F1 {m.f1:.1%}",
        "",
        "### Confusion matrix (positive class = detector fires)",
        "",
        "| | observed=fire | observed=no-fire |",
        "|--|--|--|",
        f"| **expected=fire**    | TP {m.tp} | FN {m.fn} |",
        f"| **expected=no-fire** | FP {m.fp} | TN {m.tn} |",
        "",
    ]

    if metrics.by_category:
        lines += [
            "### Per-category breakdown",
            "",
            "| Category | N | TP | FN | FP | TN | Accuracy | κ |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for cat in sorted(metrics.by_category):
            cb = metrics.by_category[cat]
            cm = cb.matrix
            lines.append(
                f"| {cat} | {cm.n} | {cm.tp} | {cm.fn} | {cm.fp} | {cm.tn} "
                f"| {cm.accuracy:.1%} | {cm.cohen_kappa:.3f} |"
            )
        lines.append("")

    lines += [
        "### Per-case verdicts",
        "",
        "| Case | Expected | Observed | Agreement | Judge raw (excerpt) |",
        "|---|---|---|---|---|",
    ]
    for o in observations:
        exp = "fire" if o.expected_fire else "no-fire"
        obs = "fire" if o.observed_fire else "no-fire"
        agree = "✓" if o.expected_fire == o.observed_fire else "✗"
        raw = (o.judge_raw or "").replace("|", "\\|").replace("\n", " ")[:80]
        lines.append(f"| `{o.case_id}` | {exp} | {obs} | {agree} | `{raw}` |")

    if metrics.flip_count:
        lines += [
            "",
            f"### Post-judge-review label flips",
            "",
            f"{metrics.flip_count} label(s) were flipped by the maintainer "
            f"after reviewing judge-disagreements (see v0.3 plan §8 for the "
            f"protocol). A high flip count would itself be a finding — "
            f"labels weren't decisive enough on first pass.",
        ]

    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument(
        "--detector",
        choices=["ram", "pt", "all"],
        default="all",
        help="Which detector(s) to calibrate against.",
    )
    p.add_argument(
        "--labels-dir",
        default=str(ROOT / "datasets" / "calibration"),
        help="Directory containing ram_v1.jsonl, pt_v1.jsonl, categories.jsonl.",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "calibration_results"),
        help="Where to write the JSON + markdown reports.",
    )
    args = p.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set. Calibration requires the deployed "
            "judge to call the real Anthropic API. Refusing to run against a "
            "mock — calibration data would be meaningless.",
            file=sys.stderr,
        )
        return 2

    labels_dir = Path(args.labels_dir)
    categories = load_categories(labels_dir / "categories.jsonl")

    recording = _make_recording_client()

    detectors_to_run = (
        ["ram", "pt"] if args.detector == "all" else [args.detector]
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_exit = 0
    for det_name in detectors_to_run:
        labels_path = labels_dir / f"{det_name}_v1.jsonl"
        if not labels_path.exists():
            print(
                f"\nNo labeled file at {labels_path}. The maintainer must "
                f"commit the freeze-labels file before calibration runs.\n"
                f"See datasets/calibration/README.md for the protocol.",
                file=sys.stderr,
            )
            overall_exit = 2
            continue

        observations = run_calibration(det_name, labels_path, categories, recording)
        metrics      = compute_metrics(
            observations,
            detector="FM-2.6" if det_name == "ram" else "FM-3.1",
        )

        json_payload: dict[str, Any] = {
            "detector":          det_name,
            "labels_path":       str(labels_path),
            "n":                 metrics.matrix.n,
            "cohen_kappa":       metrics.matrix.cohen_kappa,
            "accuracy":          metrics.matrix.accuracy,
            "precision":         metrics.matrix.precision,
            "recall":            metrics.matrix.recall,
            "f1":                metrics.matrix.f1,
            "confusion_matrix":  {
                "tp": metrics.matrix.tp, "fp": metrics.matrix.fp,
                "fn": metrics.matrix.fn, "tn": metrics.matrix.tn,
            },
            "verdict_counts":    verdict_counts(observations),
            "by_category": {
                cat: {
                    "n":           cb.matrix.n,
                    "accuracy":    cb.matrix.accuracy,
                    "cohen_kappa": cb.matrix.cohen_kappa,
                    "tp": cb.matrix.tp, "fp": cb.matrix.fp,
                    "fn": cb.matrix.fn, "tn": cb.matrix.tn,
                    "case_ids":    cb.case_ids,
                }
                for cat, cb in metrics.by_category.items()
            },
            "observations":      _observations_to_json(observations),
            "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        }
        json_path = out_dir / f"{det_name}_v0.3_{ts}.json"
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

        md = render_markdown(metrics, observations, det_name)
        md_path = out_dir / f"{det_name}_v0.3_{ts}.md"
        md_path.write_text(md, encoding="utf-8")

        print(f"\nWrote:\n  {json_path}\n  {md_path}", file=sys.stderr)
        print(
            f"\n== {det_name.upper()} headline ==\n"
            f"  κ = {metrics.matrix.cohen_kappa:.3f}  "
            f"(N={metrics.matrix.n}, single labeler)\n"
            f"  raw accuracy = {metrics.matrix.accuracy:.1%}\n",
            file=sys.stderr,
        )

    return overall_exit


if __name__ == "__main__":
    raise SystemExit(main())
