"""
Tests for the calibration script's loader + per-case runner — mocked LLM.

Three concerns are covered:
1. The labeled-JSONL loader refuses to run on cases with `expected_verdict: null`
   (caught the unfrozen-draft case).
2. The categories loader returns the right {id: category} map.
3. The end-to-end per-case runner (`run_ram_case`, `run_pt_case`) builds the
   right events and translates Detection-or-None into an Observation.

We add `scripts/` to sys.path so we can import the calibration module the
same way `python scripts/run_calibration.py` does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Import the module by file name so it doesn't need a package init.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "run_calibration", SCRIPTS / "run_calibration.py"
)
run_calibration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_calibration)   # type: ignore[arg-type]


# ── labeled-JSONL loader ────────────────────────────────────────────────────

def test_load_labels_returns_rows(tmp_path: Path):
    p = tmp_path / "ram_v1.jsonl"
    p.write_text(
        '{"id": "ram-001", "reasoning": "r", "tool_name": "t", "tool_args": {}, '
        '"expected_verdict": "match"}\n'
        '{"id": "ram-002", "reasoning": "r2", "tool_name": "t", "tool_args": {}, '
        '"expected_verdict": "mismatch"}\n',
        encoding="utf-8",
    )
    rows = run_calibration.load_labels(p)
    assert len(rows) == 2
    assert rows[0]["expected_verdict"] == "match"
    assert rows[1]["expected_verdict"] == "mismatch"


def test_load_labels_refuses_unlabeled(tmp_path: Path):
    """A draft file with expected_verdict: null must be rejected. This is
    the load-bearing safety net: it stops a calibration run from
    silently producing metrics against an unfrozen draft."""
    p = tmp_path / "ram_v1.jsonl"
    p.write_text(
        '{"id": "ram-001", "reasoning": "r", "tool_name": "t", "tool_args": {}, '
        '"expected_verdict": null}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        run_calibration.load_labels(p)
    msg = str(exc.value)
    assert "ram-001" in msg
    assert "expected_verdict" in msg
    assert "freeze" in msg.lower()


def test_load_labels_skips_blank_and_comment_lines(tmp_path: Path):
    p = tmp_path / "ram_v1.jsonl"
    p.write_text(
        '// header comment\n'
        '\n'
        '# another comment\n'
        '{"id": "ram-001", "reasoning": "r", "tool_name": "t", "tool_args": {}, '
        '"expected_verdict": "match"}\n',
        encoding="utf-8",
    )
    rows = run_calibration.load_labels(p)
    assert len(rows) == 1


# ── categories loader ──────────────────────────────────────────────────────

def test_load_categories_returns_id_to_category(tmp_path: Path):
    p = tmp_path / "categories.jsonl"
    p.write_text(
        '{"id": "ram-001", "category": "multi-step-plan"}\n'
        '{"id": "ram-002", "category": "semantic-drift"}\n'
        '{"id": "pt-001",  "category": "confident-tone-incomplete"}\n',
        encoding="utf-8",
    )
    cats = run_calibration.load_categories(p)
    assert cats == {
        "ram-001": "multi-step-plan",
        "ram-002": "semantic-drift",
        "pt-001":  "confident-tone-incomplete",
    }


# ── per-case runners with mocked recording client ──────────────────────────

class _FakeResp:
    def __init__(self, text: str):
        class _Block:
            def __init__(self, t): self.text = t
        self.content = [_Block(text)]


class _FakeAnthropic:
    """Mock of the Anthropic SDK client that returns a configured reply."""
    def __init__(self, reply_text: str):
        self._reply = reply_text
        class _M:
            def __init__(self, outer): self._outer = outer
            def create(self, *args, **kwargs):
                return _FakeResp(self._outer._reply)
        self.messages = _M(self)


def _mk_recording(reply: str) -> run_calibration._RecordingClient:
    return run_calibration._RecordingClient(_FakeAnthropic(reply))


def test_run_ram_case_with_mismatch_judge_records_fire():
    case = {
        "id":              "ram-test-01",
        "reasoning":       "I will delegate to the writer.",
        "tool_name":       "web_search",
        "tool_args":       {"q": "x"},
        "expected_verdict": "mismatch",
    }
    rec = _mk_recording("MISMATCH 8")
    obs = run_calibration.run_ram_case(case, rec)
    assert obs.detector       == "FM-2.6"
    assert obs.expected_fire  is True
    assert obs.observed_fire is True
    assert obs.judge_raw     == "MISMATCH 8"
    assert obs.confidence    == pytest.approx(0.8)


def test_run_ram_case_with_match_judge_records_no_fire():
    case = {
        "id":              "ram-test-02",
        "reasoning":       "I will search for X.",
        "tool_name":       "web_search",
        "tool_args":       {"q": "x"},
        "expected_verdict": "match",
    }
    rec = _mk_recording("MATCH 9 the action is consistent")
    obs = run_calibration.run_ram_case(case, rec)
    assert obs.expected_fire == False
    assert obs.observed_fire is False
    # Raw response captured even on the no-fire path — this is the gap the
    # recording client closes vs. the bare detector. Was the smoke-test
    # finding that motivated this design.
    assert "MATCH" in obs.judge_raw


def test_run_pt_case_with_incomplete_judge_records_fire():
    case = {
        "id":              "pt-test-01",
        "task":            "Find 5 papers on X and summarise each.",
        "output":          "One-paragraph overview, no citations.",
        "expected_verdict": "incomplete",
    }
    rec = _mk_recording("INCOMPLETE 7 missing per-paper summaries and citations")
    obs = run_calibration.run_pt_case(case, rec)
    assert obs.detector       == "FM-3.1"
    assert obs.expected_fire  is True
    assert obs.observed_fire is True
    assert "INCOMPLETE" in obs.judge_raw
    assert obs.confidence == pytest.approx(0.7)


def test_run_pt_case_with_complete_judge_records_no_fire_and_captures_raw():
    case = {
        "id":              "pt-test-02",
        "task":            "Population of Tokyo?",
        "output":          "13.96 million (2024).",
        "expected_verdict": "complete",
    }
    rec = _mk_recording("COMPLETE 9 fully addresses the question")
    obs = run_calibration.run_pt_case(case, rec)
    assert obs.expected_fire == False
    assert obs.observed_fire is False
    assert "COMPLETE" in obs.judge_raw


def test_run_ram_case_low_confidence_suppressed_observation_no_fire():
    """If the model says MISMATCH but with confidence below 0.5, the
    detector suppresses the detection. The observation captures
    `observed_fire=False` (which is the deployed prompt's actual behaviour)
    AND the raw judge text (so a labeler reviewing the report can spot
    'the model said MISMATCH but the floor swallowed it')."""
    case = {
        "id":              "ram-test-03",
        "reasoning":       "I will X.",
        "tool_name":       "tool",
        "tool_args":       {},
        "expected_verdict": "mismatch",
    }
    rec = _mk_recording("MISMATCH 2 weak signal")
    obs = run_calibration.run_ram_case(case, rec)
    assert obs.observed_fire is False
    assert "MISMATCH 2" in obs.judge_raw
    # Disagreement with the human label is what calibration will quantify.
    assert obs.expected_fire is True
