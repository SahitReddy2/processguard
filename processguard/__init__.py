"""
processguard — runtime detection for multi-agent coordination failures.

Quickstart:
    import processguard
    processguard.attach(graph)
    result = graph.invoke(...)
"""

from .guard import ProcessGuard
from .core.policy import PolicyAction, PolicyConfig, Detection, PolicyEngine, ProcessGuardError
from .core.event import AgentEvent, EventType

__version__ = "0.2.1"
__all__ = [
    "ProcessGuard",
    "attach",
    "PolicyAction",
    "PolicyConfig",
    "Detection",
    "PolicyEngine",
    "ProcessGuardError",
    "AgentEvent",
    "EventType",
    "__version__",
]

# Module-level convenience: processguard.attach(obj)
_default_guard: ProcessGuard | None = None


def attach(framework_object, **kwargs) -> ProcessGuard:
    """
    Attach a default ProcessGuard instance to a framework object.
    Returns the guard so you can customise the policy after the fact.

    Example::

        guard = processguard.attach(graph)
        guard.policy.policies["FM-1.3"] = PolicyConfig(action=PolicyAction.HALT)
        graph.invoke(...)
    """
    global _default_guard
    _default_guard = ProcessGuard(**kwargs)
    _default_guard.attach(framework_object)
    return _default_guard
