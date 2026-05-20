from __future__ import annotations

import json
import sqlite3
import threading
from typing import Optional

from .event import AgentEvent, EventType


class TraceStorage:
    """Thread-safe SQLite-backed event store. Default path is processguard.db."""

    _CREATE = """
        CREATE TABLE IF NOT EXISTS events (
            event_id       TEXT PRIMARY KEY,
            trace_id       TEXT NOT NULL,
            span_id        TEXT NOT NULL,
            parent_span_id TEXT,
            event_type     TEXT NOT NULL,
            agent_name     TEXT NOT NULL,
            timestamp      REAL NOT NULL,
            tool_name      TEXT,
            tool_args      TEXT,
            tool_result    TEXT,
            content        TEXT,
            metadata       TEXT
        )
    """

    def __init__(self, db_path: str = "processguard.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    # ── connection ─────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.conn

    def _init_db(self):
        c = self._conn()
        c.execute(self._CREATE)
        c.execute("CREATE INDEX IF NOT EXISTS idx_trace ON events(trace_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_agent ON events(trace_id, agent_name)")
        c.commit()

    # ── write ───────────────────────────────────────────────────────────────

    def save(self, event: AgentEvent):
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event.event_id, event.trace_id, event.span_id, event.parent_span_id,
                event.event_type.value, event.agent_name, event.timestamp,
                event.tool_name,
                json.dumps(event.tool_args) if event.tool_args is not None else None,
                event.tool_result,
                event.content,
                json.dumps(event.metadata) if event.metadata else None,
            ),
        )
        c.commit()

    # ── read ────────────────────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> list[AgentEvent]:
        rows = self._conn().execute(
            "SELECT * FROM events WHERE trace_id=? ORDER BY timestamp", (trace_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def get_recent(
        self,
        trace_id: str,
        n: int,
        agent_name: Optional[str] = None,
    ) -> list[AgentEvent]:
        if agent_name:
            rows = self._conn().execute(
                "SELECT * FROM events WHERE trace_id=? AND agent_name=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (trace_id, agent_name, n),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM events WHERE trace_id=? ORDER BY timestamp DESC LIMIT ?",
                (trace_id, n),
            ).fetchall()
        return list(reversed([self._row(r) for r in rows]))

    def count_events(self, trace_id: str, agent_name: Optional[str] = None) -> int:
        if agent_name:
            return self._conn().execute(
                "SELECT COUNT(*) FROM events WHERE trace_id=? AND agent_name=?",
                (trace_id, agent_name),
            ).fetchone()[0]
        return self._conn().execute(
            "SELECT COUNT(*) FROM events WHERE trace_id=?", (trace_id,)
        ).fetchone()[0]

    # ── helpers ─────────────────────────────────────────────────────────────

    def _row(self, row) -> AgentEvent:
        return AgentEvent(
            event_id=row[0],
            trace_id=row[1],
            span_id=row[2],
            parent_span_id=row[3],
            event_type=EventType(row[4]),
            agent_name=row[5],
            timestamp=row[6],
            tool_name=row[7],
            tool_args=json.loads(row[8]) if row[8] else None,
            tool_result=row[9],
            content=row[10],
            metadata=json.loads(row[11]) if row[11] else {},
        )
