"""Unified perception timeline — event-sourced system of record (Feature 4).

One SQLite DB (ada_events.db, WAL mode), one append-only `events` table.
Every perception/action is an event; projections (daily_digest, episodes)
are pure functions over the event stream — rebuild by replay, never by
mutation. Corrections are new compensating events; rows are never updated.
The only sanctioned deletion is TTL compaction, which replaces raw rows
with deterministic summaries.

Retention honors observe tiers (backend/observe.py):
  tier 0 (OFF)      — writes go to the RAM ring only, never the DB
  tier 1 (CAPTIONS) — captions/text kept per per-source TTLs (default)
  tier 2 (DETAIL)   — longer TTLs, only on Yash's explicit ask

SQLite + stdlib only. FTS5 (built into Python's sqlite3) for keyword
search. No model downloads, no LLM — summaries are template-based and
deterministic.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_V = 1

# Default per-source TTLs (days). Voice transcripts 90d, screen text 30d,
# metadata-style records longer. Tier 2 extends everything.
DEFAULT_TTLS = {
    "voice": 90,      # transcripts
    "screen": 30,     # captions / OCR text
    "sensor": 30,     # ring-framed descriptors
    "message": 180,   # user/assistant messages
    "tool": 365,      # tool call metadata
    "agent": 365,     # interjections, decisions
    "system": 365,    # lifecycle ticks
}
TIER_TTL_MULT = {0: 0.0, 1: 1.0, 2: 4.0}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'\-]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "you", "your",
    "are", "was", "were", "has", "have", "had", "not", "but", "all",
    "she", "his", "her", "him", "its", "our", "their", "will", "would",
    "can", "just", "now", "then", "there", "here", "what", "when",
    "which", "who", "how", "said", "say", "about", "into", "over",
    "also", "very", "more", "than", "been", "did", "out", "one",
})


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


class PerceptionTimeline:
    """Append-only event timeline. Open with a path; use module-level
    init_timeline()/emit() so writers (server.py, ada.py, ...) need no
    plumbing and stay safe when the timeline isn't initialized."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "seq INTEGER PRIMARY KEY, "
                "ts_utc REAL NOT NULL, source TEXT NOT NULL, type TEXT NOT NULL, "
                "payload_json TEXT NOT NULL, schema_v INTEGER NOT NULL, "
                "session_id TEXT NOT NULL)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_src ON events(source, type)")
            # FTS5 over type + payload for keyword search. Contentless —
            # rows are managed explicitly (insert on emit, delete on compact)
            # so compaction can't leave a stale index behind.
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS events_fts "
                "USING fts5(type, payload_json, content='', tokenize='unicode61')")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS summaries ("
                "id INTEGER PRIMARY KEY, window_start REAL NOT NULL, "
                "window_end REAL NOT NULL, source TEXT NOT NULL, type TEXT NOT NULL, "
                "event_count INTEGER NOT NULL, summary TEXT NOT NULL, "
                "schema_v INTEGER NOT NULL)")
        self.session_id = uuid.uuid4().hex[:12]
        self._tier = 1
        self._ttls = dict(DEFAULT_TTLS)

    # ── retention tiers ─────────────────────────────────────────────────
    def set_retention_tier(self, tier: int):
        """0 = ring only (emit becomes a no-op), 1 = default TTLs,
        2 = extended TTLs on explicit ask."""
        self._tier = max(0, min(2, int(tier)))

    @property
    def retention_tier(self) -> int:
        return self._tier

    def set_ttl(self, source: str, days: float):
        self._ttls[source] = max(0.0, float(days))

    def ttl_for(self, source: str) -> float:
        """Effective TTL in days for a source at the current tier."""
        base = self._ttls.get(source, 365.0)
        return base * TIER_TTL_MULT[self._tier]

    # ── append-only writes ──────────────────────────────────────────────
    def emit(self, source: str, type: str, payload: dict | None = None,
             session_id: str | None = None, ts: float | None = None) -> int | None:
        """Append one event. Returns the seq, or None at tier 0 (ring only)."""
        if self._tier == 0:
            return None
        payload = dict(payload or {})
        row = {
            "ts_utc": float(ts if ts is not None else time.time()),
            "source": str(source)[:32],
            "type": str(type)[:64],
            "payload_json": json.dumps(payload, ensure_ascii=True)[:8000],
            "schema_v": SCHEMA_V,
            "session_id": (session_id or self.session_id)[:32],
        }
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO events(ts_utc, source, type, payload_json, "
                "schema_v, session_id) VALUES(?,?,?,?,?,?)",
                (row["ts_utc"], row["source"], row["type"],
                 row["payload_json"], row["schema_v"], row["session_id"]))
            seq = cur.lastrowid
            self._conn.execute(
                "INSERT INTO events_fts(rowid, type, payload_json) "
                "VALUES(?,?,?)", (seq, row["type"], row["payload_json"]))
        return seq

    def correct(self, source: str, type: str, payload: dict,
                supersedes: int | None = None, **kw) -> int | None:
        """Compensating event — the event-sourcing way to 'fix' a row.
        Never updates; appends a new event linked to the old seq."""
        payload = dict(payload)
        if supersedes is not None:
            payload["supersedes_seq"] = supersedes
        return self.emit(source, f"{type}.corrected", payload, **kw)

    # ── reads ───────────────────────────────────────────────────────────
    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def query(self, ts_from: float | None = None, ts_to: float | None = None,
              source: str | None = None, type: str | None = None,
              text: str | None = None, limit: int = 200) -> list[dict]:
        """Filter by time window, source, type, and/or FTS5 keyword search."""
        sql = "SELECT e.* FROM events e"
        where, args = [], []
        if text:
            sql += " JOIN events_fts f ON f.rowid = e.seq"
            where.append("events_fts MATCH ?")
            args.append(text)
        if ts_from is not None:
            where.append("e.ts_utc >= ?")
            args.append(float(ts_from))
        if ts_to is not None:
            where.append("e.ts_utc <= ?")
            args.append(float(ts_to))
        if source:
            where.append("e.source = ?")
            args.append(source)
        if type:
            where.append("e.type = ?")
            args.append(type)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY e.ts_utc ASC, e.seq ASC LIMIT ?"
        args.append(int(limit))
        rows = self._conn.execute(sql, args).fetchall()
        return [self._row(d) for d in rows]

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["payload"] = json.loads(d["payload_json"])
        except Exception:
            d["payload"] = {}
        return d

    # ── projections (pure over the event stream) ────────────────────────
    def daily_digest(self, date: str | datetime) -> dict:
        """All events for one UTC date, grouped by (source, type) —
        the morning-briefing projection."""
        if isinstance(date, datetime):
            day = date.strftime("%Y-%m-%d")
        else:
            day = str(date)[:10]
        start = datetime.strptime(day, "%Y-%m-%d").replace(
            tzinfo=timezone.utc).timestamp()
        events = self.query(ts_from=start, ts_to=start + 86400, limit=100000)
        groups: dict[tuple[str, str], list[dict]] = {}
        for e in events:
            groups.setdefault((e["source"], e["type"]), []).append(e)
        lines = []
        for (source, type_), evs in sorted(groups.items()):
            span = f"{_utc(evs[0]['ts_utc'])} → {_utc(evs[-1]['ts_utc'])}"
            lines.append(f"{len(evs):>4} × {source}.{type_}  ({span})")
        sums = self._conn.execute(
            "SELECT * FROM summaries WHERE window_start >= ? AND window_end <= ? "
            "ORDER BY window_start ASC", (start, start + 86400)).fetchall()
        return {
            "date": day,
            "event_count": len(events),
            "groups": lines,
            "compacted_summaries": [s["summary"] for s in sums],
        }

    def episodes(self) -> list[dict]:
        """Session-grouped episodes — the memory-retrieval projection."""
        rows = self._conn.execute(
            "SELECT session_id, MIN(ts_utc) s0, MAX(ts_utc) s1, COUNT(*) n, "
            "GROUP_CONCAT(DISTINCT source) srcs, GROUP_CONCAT(DISTINCT type) types "
            "FROM events GROUP BY session_id ORDER BY s0 ASC").fetchall()
        out = []
        for r in rows:
            out.append({
                "session_id": r["session_id"],
                "start_utc": _utc(r["s0"]),
                "end_utc": _utc(r["s1"]),
                "event_count": r["n"],
                "sources": sorted((r["srcs"] or "").split(",")),
                "types": sorted((r["types"] or "").split(",")),
            })
        return out

    # ── retention / compaction ──────────────────────────────────────────
    @staticmethod
    def _summarize(group: list[dict]) -> str:
        """Deterministic 1–2 line extractive summary. No LLM, stdlib only."""
        n = len(group)
        s0, s1 = _utc(group[0]["ts_utc"]), _utc(group[-1]["ts_utc"])
        words = Counter()
        for e in group:
            for w in _WORD_RE.findall(
                    json.dumps(e["payload"], ensure_ascii=False).lower()):
                if w not in _STOPWORDS:
                    words[w] += 1
        top = ", ".join(w for w, _ in words.most_common(3)) or "—"
        return (f"{n} {group[0]['source']}.{group[0]['type']} events, "
                f"{s0} → {s1} UTC — about: {top}.")

    def compact(self, older_than_ts: float, keep_summary: bool = True,
                source: str | None = None) -> dict:
        """Summarize events older than older_than_ts into `summaries`
        (grouped by source × type × UTC day), then delete the raw rows
        (events + FTS). Returns counts. Optional source filter so
        compact_ttl() can apply each source's own TTL."""
        older_than_ts = float(older_than_ts)
        rows = self.query(ts_to=older_than_ts, source=source, limit=1_000_000)
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for e in rows:
            day = datetime.fromtimestamp(e["ts_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
            groups.setdefault((e["source"], e["type"], day), []).append(e)
        seqs = [e["seq"] for e in rows]
        summaries_written = 0
        with self._conn:
            for (source, type_, day), evs in sorted(groups.items()):
                evs.sort(key=lambda e: (e["ts_utc"], e["seq"]))
                if keep_summary:
                    self._conn.execute(
                        "INSERT INTO summaries(window_start, window_end, source, "
                        "type, event_count, summary, schema_v) VALUES(?,?,?,?,?,?,?)",
                        (evs[0]["ts_utc"], evs[-1]["ts_utc"], source, type_,
                         len(evs), self._summarize(evs), SCHEMA_V))
                    summaries_written += 1
            if seqs:
                self._conn.execute(
                    f"DELETE FROM events WHERE seq IN ({','.join('?' * len(seqs))})",
                    seqs)
                # Contentless FTS5 tables delete rows via the special
                # 'delete' command, not DELETE.
                for seq in seqs:
                    self._conn.execute(
                        "INSERT INTO events_fts(events_fts, rowid, type, "
                        "payload_json) VALUES('delete', ?, '', '')", (seq,))
        return {"raw_deleted": len(seqs), "summaries_written": summaries_written}

    def compact_ttl(self, now: float | None = None) -> dict:
        """Nightly-compaction entry point: apply per-source TTLs at the
        current tier. Tier 0: nothing to compact (nothing was written)."""
        now = float(now if now is not None else time.time())
        total = {"raw_deleted": 0, "summaries_written": 0}
        for source, days in self._ttls.items():
            ttl_s = days * TIER_TTL_MULT[self._tier] * 86400
            if ttl_s <= 0:
                continue
            # Each source compacts at its own cutoff so other sources
            # keep their full TTL.
            r = self.compact(now - ttl_s, keep_summary=True, source=source)
            total["raw_deleted"] += r["raw_deleted"]
            total["summaries_written"] += r["summaries_written"]
        return total

    # ── trust console: export + wipe ────────────────────────────────────
    def export_all(self, path: str | Path) -> dict:
        """One-button export: every event + summary as JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_events = n_sums = 0
        with open(path, "w", encoding="utf-8") as fh:
            for e in self.query(limit=10_000_000):
                fh.write(json.dumps({"record": "event", **e}, ensure_ascii=True) + "\n")
                n_events += 1
            for s in self._conn.execute(
                    "SELECT * FROM summaries ORDER BY window_start ASC"):
                d = dict(s)
                fh.write(json.dumps({"record": "summary", **d}, ensure_ascii=True) + "\n")
                n_sums += 1
        return {"path": str(path), "events": n_events, "summaries": n_sums}

    def delete_everything(self) -> dict:
        """One-button wipe: events + FTS index + summaries. Trust-console."""
        with self._conn:
            n_events = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            n_sums = self._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            self._conn.execute("DELETE FROM events")
            # Contentless FTS5: rebuild the index table instead of DELETE.
            self._conn.execute("DROP TABLE events_fts")
            self._conn.execute(
                "CREATE VIRTUAL TABLE events_fts "
                "USING fts5(type, payload_json, content='', tokenize='unicode61')")
            self._conn.execute("DELETE FROM summaries")
        self._conn.execute("VACUUM")  # outside the transaction
        return {"events_deleted": n_events, "summaries_deleted": n_sums}

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ── module singleton: writers call emit() without plumbing ────────────────
_INSTANCE: PerceptionTimeline | None = None


def init_timeline(db_path: str | Path) -> PerceptionTimeline:
    global _INSTANCE
    _INSTANCE = PerceptionTimeline(db_path)
    return _INSTANCE


def get_timeline() -> PerceptionTimeline | None:
    return _INSTANCE


def set_retention_tier(tier: int):
    if _INSTANCE is not None:
        _INSTANCE.set_retention_tier(tier)


def emit(source: str, type: str, payload: dict | None = None, **kw) -> int | None:
    """Writer entry point. Safe no-op when the timeline isn't initialized
    or retention is tier 0 — writers must never break on this call."""
    tl = _INSTANCE
    if tl is None:
        return None
    try:
        return tl.emit(source, type, payload, **kw)
    except Exception as e:
        print(f"[TIMELINE] emit failed: {type(e).__name__}: {str(e)[:100]}")
        return None
