"""Durable checkpoints for long tasks — the lightweight LangGraph/DBOS.

Multi-step jobs (skill reps, overnight research) record each completed step
in SQLite (WAL). If the process dies mid-task, the next run calls
``resume(task_id)`` and skips steps that already finished instead of
re-doing (or half-re-doing) them.

Contract:
  checkpoint(task_id, step, payload, db_path)  — record a finished step
  resume(task_id, db_path)                     — {step: payload} in step order
  clear(task_id, db_path)                      — drop a task's checkpoints
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def _connect(db_path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS checkpoints ("
        "task_id TEXT NOT NULL, step TEXT NOT NULL, "
        "payload TEXT NOT NULL, ts REAL NOT NULL, "
        "PRIMARY KEY (task_id, step))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON checkpoints(task_id)")
    conn.commit()
    return conn


def checkpoint(task_id: str, step: str, payload: dict, db_path) -> None:
    """Record that ``step`` of ``task_id`` finished, with its result."""
    conn = _connect(db_path)
    try:
        with _lock:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints(task_id, step, payload, ts) "
                "VALUES(?,?,?,?)",
                (str(task_id), str(step), json.dumps(dict(payload or {})),
                 time.time()))
            conn.commit()
    finally:
        conn.close()


def resume(task_id: str, db_path) -> dict:
    """Return {step: payload} for a task, in the order steps completed."""
    conn = _connect(db_path)
    try:
        with _lock:
            rows = conn.execute(
                "SELECT step, payload FROM checkpoints WHERE task_id=? ORDER BY ts",
                (str(task_id),)).fetchall()
    finally:
        conn.close()
    out: dict = {}
    for step, payload in rows:
        try:
            out[step] = json.loads(payload)
        except Exception:
            out[step] = {}
    return out


def clear(task_id: str, db_path) -> None:
    """Drop all checkpoints for a task (call when it completes or aborts)."""
    conn = _connect(db_path)
    try:
        with _lock:
            conn.execute("DELETE FROM checkpoints WHERE task_id=?", (str(task_id),))
            conn.commit()
    finally:
        conn.close()
