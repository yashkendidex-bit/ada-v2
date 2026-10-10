"""Storage layer for ADA's memory.

Design rules, all of which come from a specific failure in the old `MemoryBank`:

1. **Additive migrations only.** `ada_kernel/ada.db` is live and not backed by git.
   Every schema change is an `ALTER TABLE ADD COLUMN` guarded by a `PRAGMA
   table_info` check, or a `CREATE TABLE IF NOT EXISTS`. Nothing is ever dropped or
   rewritten, so the 27 existing memories keep working.

2. **The episodic store is the substrate; everything else is derived.** The old code
   had `recall_entries` (conversations) and `episodes` (one-line summaries) as two
   half-populated stores. Here `recall_entries` is the single lossless L0 log, and
   summaries are a *field on the row* rather than a second table. That removes the
   split-brain that made `episodes` 27 rows of duplicated content.

3. **Facts are bi-temporal.** `t_valid`/`t_invalid` are event time; `t_created`/
   `t_expired` are transaction time. Updating a fact invalidates the old row instead
   of DELETE-ing it, which is what made the old `add_fact` lossy.

4. **Nothing is deleted for being old.** `fidelity_level` degrades a memory from
   full text to gist to tombstone. Deletion happens only on an explicit user request.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

# ── fidelity ladder (see ADA_MEMORY_ARCHITECTURE.md §4.5) ────────────────────
L0_FULL = 0        # complete episodic record
L2_SUMMARY = 2     # condensed
L3_GIST = 3        # one clause
L5_TOMBSTONE = 5   # only the fact that a memory existed

FIDELITY_LABEL = {
    L0_FULL: "full", L2_SUMMARY: "summary", L3_GIST: "gist", L5_TOMBSTONE: "tombstone",
}

# Columns added to `recall_entries` to make it a real episodic store.
# `entities` is new here — the legacy schema only had a freeform `tags` column, which
# nothing ever wrote meaningfully.
_EPISODE_COLUMNS = [
    ("summary", "TEXT"),
    ("gist", "TEXT"),
    ("entities", "TEXT"),
    ("fidelity_level", "INTEGER DEFAULT 0"),
    ("ts_event", "REAL"),
    ("importance", "INTEGER DEFAULT 3"),
    ("arousal", "INTEGER DEFAULT 1"),
    ("access_count", "INTEGER DEFAULT 0"),
    ("last_access", "REAL"),
    ("stability", "REAL DEFAULT 1.0"),
    ("difficulty", "REAL DEFAULT 5.0"),
    ("pinned", "INTEGER DEFAULT 0"),
    ("session_id", "TEXT"),
    ("turn_index", "INTEGER"),
    ("embed_backend", "TEXT"),
    ("quarantined", "INTEGER DEFAULT 0"),
]

# Columns added to `core_facts` to make it bi-temporal and inspectable.
_FACT_COLUMNS = [
    ("subject", "TEXT"),
    ("predicate", "TEXT"),
    ("object", "TEXT"),
    ("confidence", "REAL DEFAULT 0.8"),
    ("pinned", "INTEGER DEFAULT 0"),
    ("t_valid", "REAL"),
    ("t_invalid", "REAL"),
    ("t_created", "REAL"),
    ("t_expired", "REAL"),
    ("superseded_by", "INTEGER"),
    ("source_episode_id", "INTEGER"),
    ("activation_strength", "REAL DEFAULT 1.0"),
    ("embedding", "BLOB"),
    ("updated", "REAL"),
]

_EDGE_COLUMNS = [
    ("src_id", "INTEGER"),
    ("dst_id", "INTEGER"),
    ("t_valid", "REAL"),
    ("t_invalid", "REAL"),
    ("t_created", "REAL"),
    ("t_expired", "REAL"),
    ("confidence", "REAL DEFAULT 0.7"),
    ("source_episode_id", "INTEGER"),
    ("weight", "REAL DEFAULT 1.0"),
]

_REFLECTION_COLUMNS = [
    ("level", "INTEGER DEFAULT 1"),
    ("cites", "TEXT"),
    ("embedding", "BLOB"),
    ("importance", "INTEGER DEFAULT 5"),
    ("pinned", "INTEGER DEFAULT 0"),
]

_NEW_TABLES = [
    # Entities with a canonical form, so "the drone" and "the quadcopter" converge.
    """CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        canonical TEXT,
        kind TEXT,
        summary TEXT,
        embedding BLOB,
        first_seen REAL, last_seen REAL,
        mention_count INTEGER DEFAULT 1,
        salience REAL DEFAULT 0.5
    )""",
    """CREATE TABLE IF NOT EXISTS communities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, summary TEXT, members TEXT,
        embedding BLOB, updated REAL
    )""",
    # Every retrieval, so stability can be updated (retrieval practice) and so we
    # can measure whether recall is actually being used.
    """CREATE TABLE IF NOT EXISTS access_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_kind TEXT, memory_id INTEGER, ts REAL,
        query TEXT, score REAL, surfaced INTEGER DEFAULT 1
    )""",
    # Anomalies held back before they can poison the store (out-of-order arrival,
    # causal inversion, near-duplicate). Short TTL, never merged in.
    """CREATE TABLE IF NOT EXISTS quarantine (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payload TEXT, reason TEXT, ts REAL, ttl REAL
    )""",
    # Consolidation bookkeeping, so the nightly job is resumable and idempotent.
    """CREATE TABLE IF NOT EXISTS consolidation_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started REAL, finished REAL, scored INTEGER, promoted INTEGER,
        deduped INTEGER, pruned INTEGER, reflected INTEGER, decayed INTEGER,
        notes TEXT
    )""",
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_recall_ts ON recall_entries(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_recall_pinned ON recall_entries(pinned)",
    "CREATE INDEX IF NOT EXISTS idx_recall_fidelity ON recall_entries(fidelity_level)",
    "CREATE INDEX IF NOT EXISTS idx_facts_key ON core_facts(key)",
    "CREATE INDEX IF NOT EXISTS idx_facts_live ON core_facts(t_invalid)",
    "CREATE INDEX IF NOT EXISTS idx_edges_ab ON entity_edges(a, b)",
    "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
    "CREATE INDEX IF NOT EXISTS idx_access_mem ON access_log(memory_kind, memory_id)",
]


class Store:
    """Owns the SQLite connection and the schema. Thread-safe via a single lock.

    SQLite with `check_same_thread=False` plus an explicit lock is the right shape
    here: writes are batched and short, reads dominate, and a single writer avoids
    the `database is locked` churn that a pool would introduce under asyncio.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self.migrate()

    # ── schema ───────────────────────────────────────────────────────────────
    def migrate(self) -> dict:
        """Idempotent. Safe to call on every boot."""
        report = {"added_columns": [], "added_tables": [], "added_indexes": []}
        with self._lock:
            c = self._conn.cursor()
            # Base tables (no-ops on an existing database).
            c.execute("CREATE TABLE IF NOT EXISTS recall_entries (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      "kind TEXT, content TEXT, tags TEXT, ts REAL, embedding BLOB)")
            c.execute("CREATE TABLE IF NOT EXISTS core_facts (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      "key TEXT, value TEXT, source TEXT, ts REAL, importance INTEGER DEFAULT 1)")
            c.execute("CREATE TABLE IF NOT EXISTS entity_edges (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      "a TEXT, b TEXT, rel TEXT, ts REAL)")
            c.execute("CREATE TABLE IF NOT EXISTS notebook (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      "section TEXT UNIQUE, content TEXT, updated REAL)")
            c.execute("CREATE TABLE IF NOT EXISTS reflections (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      "ts REAL, text TEXT)")

            for table, cols in (("recall_entries", _EPISODE_COLUMNS),
                                ("core_facts", _FACT_COLUMNS),
                                ("entity_edges", _EDGE_COLUMNS),
                                ("reflections", _REFLECTION_COLUMNS)):
                existing = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
                for col, decl in cols:
                    if col not in existing:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                        report["added_columns"].append(f"{table}.{col}")

            for ddl in _NEW_TABLES:
                name = ddl.split("EXISTS", 1)[1].split("(", 1)[0].strip()
                c.execute(ddl)
                report["added_tables"].append(name)

            # Backfill: rows written before ts_event existed should not read as epoch 0.
            c.execute("UPDATE recall_entries SET ts_event = ts WHERE ts_event IS NULL")
            c.execute("UPDATE recall_entries SET last_access = ts WHERE last_access IS NULL")
            c.execute("UPDATE core_facts SET t_valid = ts WHERE t_valid IS NULL")
            c.execute("UPDATE core_facts SET t_created = ts WHERE t_created IS NULL")
            c.execute("UPDATE core_facts SET updated = ts WHERE updated IS NULL")

            # Lexical channel.
            #
            # This is a PLAIN fts5 table, not an external-content one. The external
            # content design (`content='recall_entries'`) looked tidier — the text is
            # stored once — but it makes updates treacherous: FTS5's 'delete' command
            # must be given the exact old values, and issuing it for a row that was
            # never indexed corrupts the index ("database disk image is malformed"),
            # which then breaks every subsequent search. Nor can it use UPSERT, since
            # virtual tables do not support ON CONFLICT.
            #
            # Storing the text a second time is the cheaper mistake. It costs some
            # disk and buys DELETE-then-INSERT semantics that always work.
            existing = c.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='recall_fts'"
            ).fetchone()
            if existing and "content=" in (existing["sql"] or ""):
                # Old external-content layout: derived data, safe to rebuild.
                c.execute("DROP TABLE recall_fts")
                existing = None
            try:
                c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS recall_fts USING fts5("
                          "content, summary, tokenize='porter unicode61')")
                report["added_tables"].append("recall_fts")
                # Backfill. The table is created empty, so without this every existing
                # memory is invisible to the lexical channel — which is exactly the
                # silent-degradation failure this subsystem exists to avoid.
                indexed = c.execute("SELECT COUNT(*) FROM recall_fts").fetchone()[0]
                total = c.execute("SELECT COUNT(*) FROM recall_entries").fetchone()[0]
                if indexed < total:
                    c.execute("INSERT INTO recall_fts(rowid, content, summary) "
                              "SELECT id, content, summary FROM recall_entries")
                    report["notes"] = (f"backfilled {total - indexed} rows into recall_fts")
            except sqlite3.OperationalError as e:
                # FTS5 missing on some builds. Retrieval degrades to vector-only and
                # says so, rather than silently losing the lexical channel.
                report["notes"] = f"FTS5 unavailable: {e}"

            for ddl in _INDEXES:
                c.execute(ddl)
            self._conn.commit()
        return report

    def fts_available(self) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recall_fts'"
            ).fetchone()
        return row is not None

    # ── low-level helpers ────────────────────────────────────────────────────
    def execute(self, sql: str, args: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(args))
            self._conn.commit()
            return cur

    def query(self, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).fetchall()

    def one(self, sql: str, args: Iterable = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).fetchone()

    def executemany(self, sql: str, rows: Iterable) -> None:
        with self._lock:
            self._conn.executemany(sql, list(rows))
            self._conn.commit()

    def transaction(self):
        """Context manager for multi-statement atomic work."""
        return _Transaction(self)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ── FTS sync ─────────────────────────────────────────────────────────────
    def fts_index(self, rowid: int) -> None:
        """Index one row, replacing any previous entry for it."""
        if not self.fts_available():
            return
        with self._lock:
            self._conn.execute("DELETE FROM recall_fts WHERE rowid=?", (rowid,))
            self._conn.execute(
                "INSERT INTO recall_fts(rowid, content, summary) "
                "SELECT id, content, summary FROM recall_entries WHERE id=?", (rowid,))
            self._conn.commit()

    def fts_rebuild(self) -> None:
        """Re-index everything. Used after bulk writes and by the consolidation job."""
        if not self.fts_available():
            return
        with self._lock:
            self._conn.execute("DELETE FROM recall_fts")
            self._conn.execute(
                "INSERT INTO recall_fts(rowid, content, summary) "
                "SELECT id, content, summary FROM recall_entries")
            self._conn.commit()

    def fts_search(self, query: str, limit: int = 30) -> list[tuple[int, float]]:
        """BM25 lexical search. Returns [(rowid, magnitude)].

        FTS5's ``bm25()`` returns *negative* numbers where more negative means a
        better match. The previous version scored ``1/(1+|bm25|)``, which inverted
        the ranking — the best match got the lowest score. We return ``-bm25`` so
        larger is better, and let the caller decide how to normalise: the magnitude
        is not comparable across queries, so it is only safe to use for *ordering*.
        """
        if not self.fts_available():
            return []
        safe = _fts_query(query)
        if not safe:
            return []
        try:
            rows = self.query(
                "SELECT rowid, bm25(recall_fts) AS rank FROM recall_fts "
                "WHERE recall_fts MATCH ? ORDER BY rank LIMIT ?", (safe, limit))
        except sqlite3.OperationalError:
            return []
        return [(int(r["rowid"]), -float(r["rank"])) for r in rows]


class _Transaction:
    def __init__(self, store: Store):
        self.store = store

    def __enter__(self):
        self.store._lock.acquire()
        self.store._conn.execute("BEGIN")
        return self.store

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.store._conn.commit()
            else:
                self.store._conn.rollback()
        finally:
            self.store._lock.release()
        return False


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    FTS5 treats a lot of punctuation as syntax, so a raw user query raises
    OperationalError. Quoting each token makes it a phrase match and neutralises
    operators. Empty result means "nothing to search for", not "match everything".
    """
    import re
    toks = [t for t in re.findall(r"[A-Za-z0-9_]+", str(text)) if len(t) > 1]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"' for t in toks[:24])


def now() -> float:
    return time.time()


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def loads(text: Any, default=None):
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default
