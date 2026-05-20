import pytest
from processguard.detectors.unaware_termination import UnawareTerminationDetector
from .conftest import tool_call


def test_fires_past_budget_with_convergence():
    # The detector fires on the first step that exceeds budget AND satisfies
    # convergence. After that it's locked (fires exactly once per trace).
    det = UnawareTerminationDetector(step_budget=5, convergence_window=3)
    results = [det.observe(tool_call("RAG")) for _ in range(11)]
    detections = [r for r in results if r is not None]
    assert len(detections) == 1
    assert detections[0].failure_mode == "FM-1.5"
    assert detections[0].evidence["step_count"] > 5


def test_no_fire_within_budget():
    det = UnawareTerminationDetector(step_budget=20, convergence_window=3)
    results = [det.observe(tool_call("RAG")) for _ in range(5)]
    assert all(r is None for r in results)


def test_no_fire_without_convergence():
    det = UnawareTerminationDetector(step_budget=5, convergence_window=3)
    tools = ["web_search", "read_paper", "write_draft", "summarise", "translate",
             "web_search", "read_paper", "write_draft"]
    results = [det.observe(tool_call("x", tool=t)) for t in tools]
    assert all(r is None for r in results)


def test_reset_clears_state():
    det = UnawareTerminationDetector(step_budget=2, convergence_window=2)
    for _ in range(4):
        det.observe(tool_call("RAG"))
    det.reset("trace-test")
    results = [det.observe(tool_call("RAG")) for _ in range(2)]
    assert all(r is None for r in results)
