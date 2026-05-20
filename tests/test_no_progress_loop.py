import pytest
from processguard.detectors.no_progress_loop import NoProgressLoopDetector, _entities
from .conftest import tool_result


RICH_TEXT   = "Retrieval Augmented Generation improves factual accuracy by grounding outputs."
REPEAT_TEXT = "error timeout retry connection failed"   # same words every time


def test_fires_on_repeated_results():
    # window+1 calls needed: the 1st call discovers entities (high novelty),
    # so it contaminates the first window. The window becomes fully stale only
    # once that first call slides off — i.e. after window+1 identical calls.
    det = NoProgressLoopDetector(window=4, novelty_threshold=0.05)
    results = [det.observe(tool_result(REPEAT_TEXT)) for _ in range(5)]
    detections = [r for r in results if r is not None]
    assert len(detections) == 1
    assert detections[0].failure_mode == "BEYOND-MAST"
    assert detections[0].failure_name == "no_progress_loop"


def test_no_fire_on_novel_results():
    det = NoProgressLoopDetector(window=4, novelty_threshold=0.05)
    texts = [
        "Retrieval Augmented Generation improves factual accuracy",
        "Vector databases store embeddings for semantic search",
        "Fine-tuning adjusts model weights on domain data",
        "Prompt engineering shapes model behaviour without training",
    ]
    results = [det.observe(tool_result(t)) for t in texts]
    assert all(r is None for r in results)


def test_entity_extraction():
    ents = _entities("Retrieval Augmented Generation improves factual accuracy")
    assert "retrieval" in ents
    assert "generation" in ents


def test_reset_clears_state():
    det = NoProgressLoopDetector(window=4, novelty_threshold=0.05)
    for _ in range(4):
        det.observe(tool_result(REPEAT_TEXT))
    det.reset("trace-test")
    # after reset window is empty
    results = [det.observe(tool_result(REPEAT_TEXT)) for _ in range(3)]
    assert all(r is None for r in results)


def test_confidence_between_0_and_1():
    det = NoProgressLoopDetector(window=4, novelty_threshold=0.05)
    for _ in range(4):
        det.observe(tool_result(REPEAT_TEXT))
    detections = [r for r in [det.observe(tool_result(REPEAT_TEXT))] if r]
    if detections:
        assert 0.0 <= detections[0].confidence <= 1.0
