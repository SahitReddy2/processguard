#!/usr/bin/env python
"""
Run the ProcessGuard offline eval harness against a gold-set JSONL file.

Usage:
    python scripts/run_eval.py [--gold PATH] [--output PATH] [--quiet]

Exit code 0 iff no case FAILED or ERRORed. SKIPPED cases do not affect the
exit code (a case skipped for missing API key counts as build-passing).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the repo root without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processguard.harness import (
    Harness,
    EvalReport,
    load_cases,
    render_markdown,
)


def main() -> int:
    # Force UTF-8 on stdout/stderr so emoji in the markdown report don't blow
    # up on Windows' default cp1252 console. Best-effort; falls back silently
    # if Python is too old or the streams don't support reconfigure.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument(
        "--gold",
        default=str(ROOT / "datasets" / "gold" / "v0.2.jsonl"),
        help="Path to the gold-set JSONL file (default: datasets/gold/v0.2.jsonl)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON report. Default: eval_results/v0.2_{UTC-ISO}.json",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-case stdout chatter (markdown report still prints unless --no-markdown).",
    )
    p.add_argument(
        "--no-markdown",
        action="store_true",
        help="Don't print the markdown report to stdout (use --markdown-out to write to a file instead).",
    )
    p.add_argument(
        "--markdown-out",
        default=None,
        help="Write the markdown report to this file path (in addition to stdout, unless --no-markdown).",
    )
    args = p.parse_args()

    cases = load_cases(args.gold)
    if not cases:
        print(f"No cases loaded from {args.gold}", file=sys.stderr)
        return 2

    print(f"Loaded {len(cases)} case(s) from {args.gold}", file=sys.stderr)

    t0 = time.perf_counter()
    harness  = Harness(cases, verbose=not args.quiet)
    results  = harness.run()
    elapsed  = time.perf_counter() - t0

    report = EvalReport(
        case_results  = results,
        timestamp_utc = datetime.now(timezone.utc).isoformat(),
        total_seconds = elapsed,
        gold_set_path = args.gold,
    )

    if args.output:
        json_path = Path(args.output)
    else:
        ts = report.timestamp_utc.replace(":", "-").replace(".", "-")
        json_path = ROOT / "eval_results" / f"v0.2_{ts}.json"
    report.write_json(json_path)

    markdown = render_markdown(report)
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    if not args.no_markdown:
        print()
        print(markdown)

    print(f"\nReport: {json_path}", file=sys.stderr)
    print(
        f"{report.passed} passed, {report.failed} failed, {report.skipped} skipped, "
        f"{report.errored} errored (exit {report.exit_code})",
        file=sys.stderr,
    )
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
