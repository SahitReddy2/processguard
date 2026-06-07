"""
ProcessGuard.close() + TraceStorage.close() regression tests.

Added in v0.2.1 after the real-run probe surfaced
`ResourceWarning: unclosed database` on guard GC. close() makes the
shutdown explicit and the warning goes away.
"""
from __future__ import annotations

import sqlite3
import uuid
import warnings

import pytest

from processguard import ProcessGuard
from processguard.core.event import AgentEvent, EventType


def _event() -> AgentEvent:
    return AgentEvent(
        trace_id   = "t-close",
        span_id    = str(uuid.uuid4()),
        event_type = EventType.TOOL_CALL,
        agent_name = "agent",
        tool_name  = "web_search",
        tool_args  = {"q": "x"},
    )


def test_close_releases_sqlite_connection():
    guard = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    guard.emit(_event())
    assert guard.storage.count_events("t-close") == 1

    guard.close()

    # After close, the storage connection is gone. Reading raises.
    with pytest.raises((sqlite3.ProgrammingError, AttributeError)):
        guard.storage.count_events("t-close")


def test_close_is_idempotent():
    """Calling close() twice must not raise."""
    guard = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    guard.close()
    guard.close()   # no exception


def test_context_manager_closes_on_exit():
    """`with ProcessGuard(...) as g:` should release the connection on exit."""
    with ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False) as g:
        g.emit(_event())
    # exited; the storage connection should be closed
    with pytest.raises((sqlite3.ProgrammingError, AttributeError)):
        g.storage.count_events("t-close")


def test_del_does_not_raise_on_gc():
    """The destructor must swallow any cleanup errors silently — a destructor
    that raises during GC turns into a noisy unraisable-exception warning
    that's hard to diagnose."""
    guard = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
    guard.close()                              # connection already gone
    # Trigger __del__ explicitly; must not raise even though the underlying
    # connection has already been closed.
    guard.__del__()                            # noqa: PLC2801 — intentional


def test_no_resource_warning_when_close_called():
    """When close() is called explicitly, Python's ResourceWarning machinery
    should not fire on subsequent GC of the guard."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        guard = ProcessGuard(db_path=":memory:", llm_detectors=False, verbose=False)
        guard.emit(_event())
        guard.close()
        del guard

        rw = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert not rw, f"got unexpected ResourceWarning(s): {[str(w.message) for w in rw]}"
