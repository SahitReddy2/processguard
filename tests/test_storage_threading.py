"""
TraceStorage threading regression tests.

These exist because Item 4's real run surfaced `OperationalError:
no such table: events` from inside a worker thread. The root cause:
TraceStorage used threading.local() to hold per-thread sqlite3
connections, and sqlite3.connect(":memory:") returns a NEW empty
in-memory database per connection. So the main thread's connection
held the schema and the worker thread's connection did not.

After the fix, TraceStorage shares one connection across threads.
"""
from __future__ import annotations

import threading
import uuid

from processguard.core.storage import TraceStorage
from processguard.core.event import AgentEvent, EventType


def _event(trace_id: str = "trace-A", agent_name: str = "researcher") -> AgentEvent:
    return AgentEvent(
        trace_id   = trace_id,
        span_id    = str(uuid.uuid4()),
        event_type = EventType.TOOL_CALL,
        agent_name = agent_name,
        tool_name  = "web_search",
        tool_args  = {"query": "x"},
    )


def test_memory_db_visible_from_worker_thread():
    """The smoking-gun case from the Item 4 real run: events saved from a
    worker thread must land in the same :memory: database the main thread
    initialised, so a subsequent count_events() from the main thread sees
    them."""
    storage = TraceStorage(db_path=":memory:")

    err: list[Exception] = []

    def worker():
        try:
            storage.save(_event())
        except Exception as e:
            err.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert not err, f"Worker thread raised {err[0]!r} when saving an event"
    assert storage.count_events("trace-A") == 1


def test_file_db_visible_from_worker_thread(tmp_path):
    """Same case for a file-backed database (real production path)."""
    db = tmp_path / "trace.db"
    storage = TraceStorage(db_path=str(db))

    err: list[Exception] = []

    def worker():
        try:
            storage.save(_event())
        except Exception as e:
            err.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert not err, f"Worker thread raised {err[0]!r} when saving an event"
    assert storage.count_events("trace-A") == 1


def test_many_threads_concurrent_save():
    """Sanity: a handful of threads each saving a few events should all land
    in the store with no exceptions and no lost events."""
    storage = TraceStorage(db_path=":memory:")
    errors: list[Exception] = []
    n_threads = 8
    per_thread = 5

    def worker(thread_idx: int):
        try:
            for i in range(per_thread):
                storage.save(_event(agent_name=f"agent-{thread_idx}-{i}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Got exceptions: {errors}"
    assert storage.count_events("trace-A") == n_threads * per_thread
