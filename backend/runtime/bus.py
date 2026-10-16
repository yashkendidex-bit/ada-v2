"""Typed in-process async pub/sub with a durable SQLite tail.

Every published :class:`Event` is fanned out to subscriber queues AND
appended to a local SQLite database (WAL mode), so the bus doubles as the
event log: a restart can replay recent lifecycle transitions. Low volume by
design (lifecycle/health/policy events) — a synchronous append is plenty.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Event:
    type: str
    source: str = ""
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    schema_v: int = SCHEMA_VERSION


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runtime_events ("
        "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts REAL NOT NULL, type TEXT NOT NULL, source TEXT NOT NULL, "
        "payload TEXT NOT NULL, schema_v INTEGER NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_events_ts ON runtime_events(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_events_type ON runtime_events(type)")
    conn.commit()
    return conn


class EventBus:
    """asyncio.Queue fan-out; every event also persisted to SQLite."""

    def __init__(self, db_path):
        self._db_path = Path(db_path)
        self._conn = _connect(self._db_path)
        self._lock = threading.Lock()
        # (event_type filter or "*", queue)
        self._subs: list[tuple[str, asyncio.Queue]] = []

    def subscribe(self, event_type: str = "*") -> asyncio.Queue:
        """Return a queue that receives matching events, in publish order."""
        q: asyncio.Queue = asyncio.Queue()
        self._subs.append((event_type, q))
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subs = [(f, q) for f, q in self._subs if q is not queue]

    async def publish(self, event, *, source: str = "", payload: dict | None = None,
                      schema_v: int = SCHEMA_VERSION) -> Event:
        """Publish an Event (or a type string + kwargs). Returns the Event."""
        if isinstance(event, Event):
            ev = event
        else:
            ev = Event(type=str(event), source=source,
                       payload=dict(payload or {}), schema_v=schema_v)
        row = (ev.ts, ev.type, ev.source, json.dumps(ev.payload), ev.schema_v)
        with self._lock:
            self._conn.execute(
                "INSERT INTO runtime_events(ts,type,source,payload,schema_v) "
                "VALUES(?,?,?,?,?)", row)
            self._conn.commit()
        for filt, q in list(self._subs):
            if filt == "*" or filt == ev.type:
                q.put_nowait(ev)
        return ev

    def recent(self, limit: int = 100, event_type: str | None = None) -> list[Event]:
        """Replay the durable tail, oldest first."""
        with self._lock:
            if event_type:
                rows = self._conn.execute(
                    "SELECT ts,type,source,payload,schema_v FROM runtime_events "
                    "WHERE type=? ORDER BY seq ASC LIMIT ?",
                    (event_type, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT ts,type,source,payload,schema_v FROM runtime_events "
                    "ORDER BY seq ASC LIMIT ?", (limit,)).fetchall()
        out = []
        for ts, typ, src, payload, sv in rows:
            try:
                pl = json.loads(payload)
            except Exception:
                pl = {}
            out.append(Event(type=typ, source=src, payload=pl, ts=ts, schema_v=sv))
        return out

    async def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass

    def to_dict(self, ev: Event) -> dict:
        return asdict(ev)
