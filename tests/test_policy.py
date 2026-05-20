import pytest
from processguard.core.policy import (
    PolicyEngine, PolicyAction, PolicyConfig, Detection, ProcessGuardError
)


def _det(**kwargs) -> Detection:
    return Detection(
        failure_mode="FM-1.3",
        failure_name="step_repetition",
        trace_id="t1",
        agent_name="researcher",
        confidence=0.9,
        **kwargs,
    )


def test_default_log_returns_none():
    engine = PolicyEngine(default=PolicyAction.LOG)
    result = engine.handle(_det())
    assert result is None
    assert len(engine.detections) == 1


def test_steer_returns_message():
    engine = PolicyEngine(default=PolicyAction.STEER)
    msg = engine.handle(_det())
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_steer_custom_message():
    policy = PolicyConfig(action=PolicyAction.STEER, steer_message="custom steer")
    engine = PolicyEngine(policies={"FM-1.3": policy})
    msg = engine.handle(_det())
    assert msg == "custom steer"


def test_halt_raises():
    policy = PolicyConfig(action=PolicyAction.HALT)
    engine = PolicyEngine(policies={"FM-1.3": policy})
    with pytest.raises(ProcessGuardError) as exc_info:
        engine.handle(_det())
    assert exc_info.value.detection.failure_mode == "FM-1.3"


def test_disabled_policy_skips():
    policy = PolicyConfig(action=PolicyAction.HALT, enabled=False)
    engine = PolicyEngine(policies={"FM-1.3": policy})
    result = engine.handle(_det())   # should NOT raise
    assert result is None


def test_escalate_calls_callback():
    called_with = []
    policy = PolicyConfig(
        action=PolicyAction.ESCALATE,
        on_detection=lambda d: called_with.append(d),
    )
    engine = PolicyEngine(policies={"FM-1.3": policy})
    engine.handle(_det())
    assert len(called_with) == 1
    assert called_with[0].failure_mode == "FM-1.3"


def test_detection_log_accumulates():
    engine = PolicyEngine(default=PolicyAction.LOG)
    engine.handle(_det())
    engine.handle(_det())
    assert len(engine.detections) == 2


def test_steer_message_fallback_chain():
    """Detection.steer_message takes precedence over engine default."""
    engine = PolicyEngine(default=PolicyAction.STEER)
    d = _det(steer_message="detector override")
    msg = engine.handle(d)
    assert msg == "detector override"
