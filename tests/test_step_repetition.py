import pytest
from processguard.detectors.step_repetition import StepRepetitionDetector
from .conftest import tool_call


def test_fires_at_threshold():
    det = StepRepetitionDetector(window=5, threshold=3)
    results = [det.observe(tool_call("RAG")) for _ in range(5)]
    detections = [r for r in results if r is not None]
    assert len(detections) == 1
    assert detections[0].failure_mode == "FM-1.3"
    assert detections[0].evidence["count_in_window"] >= 3


def test_no_fire_below_threshold():
    det = StepRepetitionDetector(window=5, threshold=3)
    results = [det.observe(tool_call("RAG")) for _ in range(2)]
    assert all(r is None for r in results)


def test_different_queries_dont_fire():
    det = StepRepetitionDetector(window=5, threshold=3)
    queries = ["RAG", "RAG 2026", "RAG latest", "vector db", "embeddings"]
    results = [det.observe(tool_call(q)) for q in queries]
    assert all(r is None for r in results)


def test_refire_after_continued_loop():
    """After firing once, continuing the loop should fire again."""
    det = StepRepetitionDetector(window=5, threshold=3)
    # fire once
    for _ in range(3):
        det.observe(tool_call("RAG"))
    # inject different call to reset fire-lock
    det.observe(tool_call("something else"))
    det.observe(tool_call("something else"))
    # loop again with same query — should fire again
    results = [det.observe(tool_call("RAG")) for _ in range(3)]
    detections = [r for r in results if r is not None]
    assert len(detections) == 1


def test_reset_clears_state():
    det = StepRepetitionDetector(window=5, threshold=3)
    for _ in range(3):
        det.observe(tool_call("RAG"))
    det.reset("trace-test")
    # after reset, window is empty — should need 3 new calls to fire
    results = [det.observe(tool_call("RAG")) for _ in range(2)]
    assert all(r is None for r in results)


def test_isolation_between_traces():
    det = StepRepetitionDetector(window=5, threshold=3)
    for _ in range(3):
        det.observe(tool_call("RAG", trace_id="trace-A"))
    # trace-B should be unaffected
    result = det.observe(tool_call("RAG", trace_id="trace-B"))
    assert result is None


def test_confidence_increases_with_repetition():
    det = StepRepetitionDetector(window=5, threshold=3)
    detections = []
    for _ in range(5):
        r = det.observe(tool_call("RAG"))
        if r:
            detections.append(r)
    assert detections[0].confidence > 0
    assert detections[0].confidence <= 1.0
