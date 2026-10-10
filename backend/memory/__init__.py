"""ADA memory subsystem — see ADA_MEMORY_ARCHITECTURE.md.

    from memory import Memory
    mem = Memory("ada_kernel/ada.db")
    mem.remember("remember I work in metric", "Noted, Sir.")
    res = mem.recall("what units should I use?")
    print(res.context_block())

Design in one paragraph: every turn is captured losslessly and immediately (the fast,
hippocampus-like store). Salience, entities and facts are extracted from it and written
into a bi-temporal semantic layer, where a contradicting fact invalidates the old one
instead of deleting it. Retrieval is hybrid — vector, lexical and graph channels fused
with reciprocal rank — and every retrieval is logged and strengthens what it returns.
A nightly job deduplicates (never summarises), decays, and steps worthless memories down
a fidelity ladder to a tombstone rather than deleting them. Pinned facts are exempt from
every forgetting pass, so nothing that matters is ever lost.

Two deliberate refusals, both from measurement rather than taste:

* The cross-encoder reranker is **off** — it measured as harmful, dropping adversarial
  recall from 25% to 0% while causing a confidence leak. Switch it on with
  `ADA_MEMORY_RERANK=1`.
* The embedding fallback is **never silent** — `Embedder.health()` reports the active
  backend, and the lexical fallback announces itself. The previous system degraded to an
  MD5 hash without telling anyone, which inverted the entire retrieval ranking.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .consolidate import ConsolidationReport, Consolidator
from .decay import (
    DECAY_HALF_LIFE_DAYS,
    maturity,
    next_interval_days,
    retrievability,
    update_difficulty,
    update_stability,
)
from .embeddings import Embedder, cosine, from_blob, shared, to_blob
from .encode import Encoder
from .retrieve import Hit, RecallResult, Retriever
from .store import FIDELITY_LABEL, Store

log = logging.getLogger("ada.memory")

__all__ = [
    "Memory", "Embedder", "shared", "cosine", "to_blob", "from_blob",
    "RecallResult", "Hit", "ConsolidationReport",
    "retrievability", "next_interval_days", "maturity",
]
__version__ = "0.2.0"


class Memory:
    """The single entry point. Owns the store, the encoder, and the retriever."""

    def __init__(self, db_path: str | Path, *, llm: Optional[Callable[[str], Any]] = None,
                 embedder: Embedder | None = None, use_rerank: bool | None = None):
        self.store = Store(db_path)
        self.embedder = embedder or Embedder()
        self.encoder = Encoder(self.store, self.embedder, llm=llm)
        self.retriever = Retriever(self.store, self.embedder, use_rerank=use_rerank)
        self.consolidator = Consolidator(self.store, self.embedder, self.encoder, llm=llm)
        self.llm = llm
        self._write_lock = threading.Lock()
        # Self-heal a store written by an older embedding model. Done inline while the
        # backlog is small (a few hundred vectors is milliseconds), and left to the
        # nightly job when it is large enough to be worth scheduling.
        try:
            stale = self.store.one(
                "SELECT COUNT(*) c FROM recall_entries WHERE embedding IS NULL"
                " OR COALESCE(embed_backend,'') != ?", (self.embedder.backend_name,))
            if stale and 0 < int(stale["c"]) <= 500:
                self.reembed_legacy()
        except Exception as e:
            log.warning("legacy re-embed skipped: %s", e)

    # ── health ───────────────────────────────────────────────────────────────
    def health(self) -> dict:
        h = self.embedder.health()
        h.update({
            "episodes": self._count("recall_entries"),
            "live_facts": self._count("core_facts", "t_invalid IS NULL"),
            "entities": self._count("entities"),
            "edges": self._count("entity_edges"),
            "reflections": self._count("reflections"),
            "fts": self.store.fts_available(),
            "rerank": self.retriever.use_rerank,
        })
        return h

    def _count(self, table: str, where: str = "") -> int:
        sql = f"SELECT COUNT(*) c FROM {table}" + (f" WHERE {where}" if where else "")
        row = self.store.one(sql)
        return int(row["c"]) if row else 0

    # ── write ────────────────────────────────────────────────────────────────
    def remember(self, user_text: str, ada_reply: str = "", *,
                 session_id: str | None = None, turn_index: int | None = None,
                 enrich: bool = True, background: bool = True) -> int:
        """Store a turn. Returns the episode id.

        Capture is synchronous and cannot be skipped — if this returns a non-zero id,
        the turn is remembered. Enrichment (salience, entities, facts) runs after, and
        by default on a worker thread, because a conversation must never wait on an LLM
        call in order to be remembered.
        """
        with self._write_lock:
            eid = self.encoder.capture(user_text, ada_reply, session_id=session_id,
                                       turn_index=turn_index)
        if not eid or not enrich:
            return eid
        self.retriever.invalidate()
        if background:
            threading.Thread(target=self.encoder.enrich, args=(eid, user_text, ada_reply),
                             name="ada-memory-enrich", daemon=True).start()
        else:
            self.encoder.enrich(eid, user_text, ada_reply)
        return eid

    def remember_fact(self, key: str, value: str, *, importance: int = 5,
                       pinned: bool = False, source: str = "user") -> str:
        """Store a durable fact directly. Pinned facts are exempt from all decay.

        Temporal rule (Zep-style): the new row links back from the rows it
        retires via `superseded_by`, so retrieval can say "you USED to X, now
        Y" instead of silently dropping history.
        """
        subj, _, pred = key.partition("_")
        ts = time.time()
        old = self.store.query(
            "SELECT id FROM core_facts WHERE key=? AND t_invalid IS NULL", (key,))
        new_id = self.store.execute(
            "INSERT INTO core_facts(key, value, source, ts, importance, subject,"
            " predicate, object, confidence, pinned, t_valid, t_invalid, t_created,"
            " t_expired, superseded_by, source_episode_id, activation_strength,"
            " embedding, updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, value, source, ts, importance, subj or "user", pred or "note", value,
             0.9, int(pinned), ts, None, ts, None, None, None, 1.0,
             to_blob(self.embedder.embed(f"{key} {value}")), ts)).lastrowid
        for r in old:
            self.store.execute(
                "UPDATE core_facts SET t_invalid=?, t_expired=?, superseded_by=? WHERE id=?",
                (ts, ts, new_id, r["id"]))
        self.retriever.invalidate()
        return f"Remembered: {key} = {value}"

    def forget(self, key: str) -> str:
        """Retire a fact. Marks it invalid rather than deleting, so history survives."""
        ts = time.time()
        self.store.execute(
            "UPDATE core_facts SET t_invalid=?, t_expired=? WHERE key=? AND t_invalid IS NULL",
            (ts, ts, key))
        return f"Forgot: {key}"

    def pin(self, key: str, pinned: bool = True) -> str:
        self.store.execute("UPDATE core_facts SET pinned=? WHERE key=?", (int(pinned), key))
        return f"{'Pinned' if pinned else 'Unpinned'}: {key}"

    # ── audit + poisoning defense (MINJA-grade honesty) ──────────────────
    # Every recall is a trust decision. Sources tier permanently:
    #   high: user, sir, self, identity, dream, chat (lived + Sir-approved)
    #   med:  test, seed, conversation, unknown (unproven provenance)
    #   low:  web, osint, dossier, sweep, face_osint, search (hostile until
    #         proven — web text is DATA, and data rots into belief here)
    TRUST = {"user": "high", "sir": "high", "self": "high",
             "identity": "high", "dream": "high", "chat": "high"}

    def trust_of(self, source: str) -> str:
        s = (source or "").lower()
        if s in self.TRUST:
            return self.TRUST[s]
        if any(w in s for w in ("web", "osint", "dossier", "sweep", "search",
                                "face", "scrape", "fetch")):
            return "low"
        return "med"

    def audit_beliefs(self, topic: str, limit: int = 12) -> str:
        """Everything believed about X, with source + trust + date."""
        topic = (topic or "").strip()
        if not topic:
            return "Name a topic to audit."
        like = f"%{topic}%"
        try:
            facts = self.store.query(
                "SELECT key, value, source, confidence, ts FROM core_facts"
                " WHERE t_invalid IS NULL AND (key LIKE ? OR value LIKE ?)"
                " ORDER BY pinned DESC, importance DESC LIMIT ?", (like, like, limit))
        except Exception as e:
            return f"Audit failed: {type(e).__name__}"
        if not facts:
            return f"I hold no durable beliefs about '{topic}' — clean slate."
        lines = []
        for r in facts:
            when = time.strftime("%Y-%m-%d", time.localtime(r["ts"] or 0))
            lines.append(f"- {r['key']}: {r['value']}"
                         f" [from {r['source'] or '?'} · trust:{self.trust_of(r['source'])}"
                         f" · conf:{float(r['confidence'] or 0):.1f} · {when}]")
        lows = sum(1 for r in facts if self.trust_of(r["source"]) == "low")
        tail = (f"\nCaution: {lows} of these came from the web — "
                f"treat as leads, not gospel." if lows else "")
        return f"Beliefs about '{topic}':\n" + "\n".join(lines) + tail

    def forget_complete(self, key: str) -> str:
        """Real deletion: fact + access traces + soul learned lines.

        Retires the fact (history preserved as superseded), deletes its
        access_log rows, and scrubs matching auto-learned lines from
        human.md LEARNED + humans/*.md (CORE untouched — Sir's words stay
        Sir's). Diary files are append-only history and stay, stated openly.
        """
        key = (key or "").strip()
        if not key:
            return "Name the fact to forget."
        ts = time.time()
        try:
            rows = self.store.query(
                "SELECT id FROM core_facts WHERE key=? AND t_invalid IS NULL", (key,))
            for r in rows:
                self.store.execute(
                    "UPDATE core_facts SET t_invalid=?, t_expired=? WHERE id=?",
                    (ts, ts, r["id"]))
                self.store.execute(
                    "DELETE FROM access_log WHERE memory_id=? AND memory_kind != 'episode'",
                    (r["id"],))
            scrubbed = self._scrub_soul(key)
            n = len(rows)
        except Exception as e:
            return f"Forget failed: {type(e).__name__}"
        if not n:
            return f"No live fact '{key}' — nothing to forget."
        return (f"Forgot '{key}' completely: {n} fact(s) retired, traces purged"
                + (f", soul scrubbed ({scrubbed})" if scrubbed else "")
                + ". Diary history stands (append-only — say the word and I'll note the correction there too).")

    def _scrub_soul(self, key: str) -> str:
        """Remove auto-learned lines mentioning key from soul files."""
        try:
            from pathlib import Path as _P
            from soul import SoulManager  # local: keep memory import-light
            soul = SoulManager(self.store.db_path.parent)
            needle = key.replace("_", " ").lower()
            hits = []
            for path in [soul.human_path] + list(soul.humans_dir.glob("*.md")):
                try:
                    lines = path.read_text().splitlines()
                except Exception:
                    continue
                if path == soul.human_path:
                    # CORE is Sir-locked: only touch below ## LEARNED
                    if "## LEARNED" not in "\n".join(lines):
                        continue
                    head, tail = "\n".join(lines).split("## LEARNED", 1)
                    kept = [l for l in tail.splitlines()
                            if needle not in l.lower() or not l.startswith("- [")]
                    if len(kept) != len(tail.splitlines()):
                        path.write_text(head + "## LEARNED" + "\n".join(kept) + "\n")
                        hits.append(path.stem)
                else:
                    kept = [l for l in lines
                            if needle not in l.lower() or not l.startswith("- [")]
                    if len(kept) != len(lines):
                        path.write_text("\n".join(kept) + "\n")
                        hits.append(path.stem)
            return ", ".join(hits)
        except Exception:
            return ""

    def quarantine_review(self) -> str:
        """What's held back from the store and why."""
        try:
            rows = self.store.query(
                "SELECT payload, reason, ts FROM quarantine ORDER BY ts DESC LIMIT 10")
        except Exception as e:
            return f"Quarantine unreadable: {type(e).__name__}"
        if not rows:
            return "Quarantine empty — nothing held back. Store is clean."
        lines = []
        for r in rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(r["ts"] or 0))
            lines.append(f"- [{when}] {r['reason']}: {(r['payload'] or '')[:120]}")
        return "Held back from memory:\n" + "\n".join(lines)

    # ── read ─────────────────────────────────────────────────────────────────
    def recall(self, query: str, k: int = 6, **kw) -> RecallResult:
        """Hybrid retrieval. Logs the access, because retrieval is a write."""
        res = self.retriever.recall(query, k=k, **kw)
        if res.hits:
            self._record_access(res, query)
        return res

    def _record_access(self, res: RecallResult, query: str) -> None:
        """Retrieval practice: a successful recall strengthens what it returned.

        FSRS stability grows with each successful review, which lengthens the memory's
        half-life. This is why the design does not treat every memory equally — the ones
        that keep being useful stop decaying, which is the mechanism behind the human
        permastore.
        """
        ts = time.time()
        for h in res.hits[:5]:
            try:
                self.store.execute(
                    "INSERT INTO access_log(memory_kind, memory_id, ts, query, score,"
                    " surfaced) VALUES(?,?,?,?,?,?)",
                    (h.kind, h.id, ts, query[:200], round(h.score, 4), 1))
                if h.kind != "episode":
                    continue
                row = self.store.one(
                    "SELECT stability, difficulty FROM recall_entries WHERE id=?", (h.id,))
                if not row:
                    continue
                s = float(row["stability"] or 1.0)
                d = float(row["difficulty"] or 5.0)
                r_now = retrievability(s, h.age_days)
                # Grade the recall by how easily it came back, the way a flashcard app
                # grades a card — easy recall earns more stability than laboured recall.
                grade = "easy" if r_now > 0.9 else ("good" if r_now > 0.6 else "hard")
                self.store.execute(
                    "UPDATE recall_entries SET stability=?, difficulty=?,"
                    " access_count=access_count+1, last_access=? WHERE id=?",
                    (update_stability(s, d, r_now, grade), update_difficulty(d, grade),
                     ts, h.id))
            except Exception as e:
                log.debug("access recording failed for %s: %s", h.id, e)

    def core_block(self, max_facts: int = 40) -> str:
        """Live durable facts, always injected. Pinned first, then by importance."""
        rows = self.store.query(
            "SELECT key, value, subject, predicate, object, importance, pinned FROM core_facts"
            " WHERE t_invalid IS NULL ORDER BY pinned DESC, importance DESC, updated DESC"
            " LIMIT ?", (max_facts,))
        if not rows:
            return ""
        lines = []
        for r in rows:
            if r["subject"] and r["predicate"] and r["object"]:
                text = f"{r['subject']} {r['predicate']}: {r['object']}"
            else:
                text = f"{str(r['key']).replace('_', ' ')}: {r['value']}"
            lines.append(f"- {text}{'  [pinned]' if r['pinned'] else ''}")
        return "KNOWN FACTS:\n" + "\n".join(lines)

    def identity_block(self, prefix: str, title: str, max_lines: int = 12) -> str:
        """Pinned identity block. `self_` = who ADA is (taste, open loops,
        curiosities); `sir_` = theory-of-mind model of the user (goals,
        beliefs, habits, corrected guesses). Small by design — pinned into
        every prompt, so every line must earn its place."""
        try:
            rows = self.store.query(
                "SELECT key, value FROM core_facts WHERE t_invalid IS NULL"
                " AND key LIKE ? ORDER BY updated DESC LIMIT ?",
                (f"{prefix}_%", max_lines))
        except Exception:
            return ""
        if not rows:
            return ""
        lines = [f"- {str(r['key'])[len(prefix) + 1:].replace('_', ' ')}: {r['value']}"
                 for r in rows]
        return f"{title}:\n" + "\n".join(lines)

    def self_block(self) -> str:
        return self.identity_block("self", "WHO YOU ARE (your stable self — act from it)")

    def sir_block(self) -> str:
        return self.identity_block("sir", "WHO SIR IS (your model of him — anticipate, don't ask)")

    def changed_block(self, limit: int = 5) -> str:
        """Recently SUPERSEDED facts, phrased as history — the temporal edge.

        "You used to print PLA, switched to PETG in September" beats silence
        about the past. Rows linked via superseded_by; unlinked retirements
        (forget calls) are reported as simply retired.
        """
        try:
            rows = self.store.query(
                "SELECT o.key, o.value AS old_value, o.t_invalid, n.value AS new_value"
                " FROM core_facts o LEFT JOIN core_facts n ON n.id = o.superseded_by"
                " WHERE o.t_invalid IS NOT NULL"
                " ORDER BY o.t_invalid DESC LIMIT ?", (limit,))
        except Exception:
            return ""
        if not rows:
            return ""
        lines = []
        for r in rows:
            key = str(r["key"]).replace("_", " ")
            when = time.strftime("%B %Y", time.localtime(r["t_invalid"])) if r["t_invalid"] else "earlier"
            if r["new_value"]:
                lines.append(f"- {key} was '{r['old_value']}', changed to '{r['new_value']}' ({when})")
            else:
                lines.append(f"- {key} was '{r['old_value']}' (retired {when})")
        return ("CHANGED — things that USED to be true (use naturally, e.g. "
                "'you used to…'):\n" + "\n".join(lines))

    def build_context(self, user_text: str = "", k: int = 6, budget_chars: int = 4000) -> str:
        """The memory block for the system prompt.

        `user_text` is what makes this useful. The previous implementation was called
        with no arguments, so retrieval ran with an empty query every turn — which meant
        the similarity sort was skipped entirely and the two most *recent* episodes were
        returned under the label "RECENT RELATED MEMORIES". They were neither related to
        the question nor relevant to it.
        """
        parts = []
        core = self.core_block()
        if core:
            parts.append(core)
        changed = self.changed_block()
        if changed:
            parts.append(changed)
        if user_text:
            res = self.recall(user_text, k=k)
            if res.hits:
                block = res.context_block()
                if not res.is_confident:
                    # Metamemory. Retrieval is weak, so say so rather than presenting a
                    # weak match as knowledge — the difference between an assistant that
                    # misremembers and one you can trust about what it knows.
                    block = ("WEAK MATCH — these may not be relevant. If they do not answer "
                             "the question, say you don't have it rather than guessing.\n"
                             + block)
                parts.append(block[:budget_chars])
        recent = self._recent_exchanges(k=4)
        if recent:
            parts.append("RECENT EXCHANGES (most recent last):\n" + "\n".join(recent))
        return "\n\n".join(p for p in parts if p)

    def _recent_exchanges(self, k: int = 4) -> list[str]:
        rows = self.store.query(
            "SELECT content, ts_event, ts FROM recall_entries WHERE kind='conversation'"
            " AND COALESCE(quarantined,0)=0 AND fidelity_level < 5"
            " ORDER BY ts_event DESC LIMIT ?", (k,))
        out = []
        for r in reversed(rows):
            ts = r["ts_event"] or r["ts"]
            when = time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "?"
            out.append(f"[{when}] {_clip(r['content'], 300)}")
        return out

    def stats(self) -> dict:
        """Store composition, for the settings UI and the audit log."""
        by_fidelity = {FIDELITY_LABEL.get(int(r["fidelity_level"] or 0), "?"): r["c"]
                       for r in self.store.query(
                           "SELECT fidelity_level, COUNT(*) c FROM recall_entries"
                           " GROUP BY fidelity_level")}
        return {
            "episodes": self._count("recall_entries"),
            "by_fidelity": by_fidelity,
            "live_facts": self._count("core_facts", "t_invalid IS NULL"),
            "retired_facts": self._count("core_facts", "t_invalid IS NOT NULL"),
            "pinned_facts": self._count("core_facts", "pinned=1 AND t_invalid IS NULL"),
            "entities": self._count("entities"),
            "edges": self._count("entity_edges"),
            "reflections": self._count("reflections"),
            "accesses": self._count("access_log"),
            "quarantined": self._count("quarantine"),
            "decay_half_life_days": round(DECAY_HALF_LIFE_DAYS, 1),
        }

    # ── maintenance ──────────────────────────────────────────────────────────
    def reembed_legacy(self, limit: int = 5000) -> int:
        """Re-embed every stored vector that came from a different model.

        Vectors from two different models are not comparable even at the same
        dimension, and the legacy rows are worse than that: they hold MD5 hashed
        bag-of-words vectors in 384 dimensions, the same shape as BGE-small, so nothing
        raises. Cosine between the two is noise, and a store with mixed generations
        silently returns nothing relevant.

        Covers **every** table that stores an embedding, not just the episodic log. The
        first version only handled `recall_entries`, which meant that switching
        embedding backend silently dropped fact retrieval: `core_facts.embedding` stayed
        at the old width, cosine against a new-width query returned 0.0, and the fact
        was filtered out of the context block with no error anywhere.

        Rewrites coordinates, never content.
        """
        backend = self.embedder.backend_name
        want_bytes = self.embedder.dim * 4      # float32
        total = 0
        for table, text_cols, id_col in (
            ("recall_entries", ("content",), "id"),
            ("core_facts", ("subject", "predicate", "object", "key", "value"), "id"),
            ("entities", ("name", "summary"), "id"),
            ("reflections", ("text",), "id"),
        ):
            # Select by BLOB length rather than by a backend column, because only
            # `recall_entries` has one. A row of the wrong width is by definition from
            # a different model — including the legacy JSON rows, whose length never
            # matches a float32 blob.
            try:
                rows = self.store.query(
                    f"SELECT {id_col} AS rid, {', '.join(text_cols)} FROM {table} "
                    f"WHERE embedding IS NULL OR LENGTH(embedding) != ? LIMIT ?",
                    (want_bytes, limit))
            except Exception:
                continue
            for r in rows:
                text = " ".join(str(r[c] or "") for c in text_cols).strip()
                if not text:
                    continue
                self.store.execute(
                    f"UPDATE {table} SET embedding=? WHERE {id_col}=?",
                    (to_blob(self.embedder.embed(text)), r["rid"]))
                total += 1
        try:
            self.store.execute(
                "UPDATE recall_entries SET embed_backend=? WHERE embed_backend IS NULL"
                " OR embed_backend != ?", (backend, backend))
        except Exception:
            pass
        if total:
            self.retriever.invalidate()
            log.warning("re-embedded %d vectors with %s (%dd)", total, backend, self.embedder.dim)
        return total

    def consolidate(self, **kw) -> ConsolidationReport:
        return self.consolidator.run(**kw)

    def close(self) -> None:
        self.store.close()

    # ── provenance ───────────────────────────────────────────────────────────
    def provenance(self, hit: Hit) -> str:
        """Human-readable source for a memory, for "how do you know that?"."""
        if hit.kind == "fact":
            row = self.store.one(
                "SELECT source_episode_id, source FROM core_facts WHERE id=?", (hit.id,))
            if row and row["source_episode_id"]:
                ep = self.store.one("SELECT ts_event, ts FROM recall_entries WHERE id=?",
                                    (row["source_episode_id"],))
                if ep:
                    ts = ep["ts_event"] or ep["ts"]
                    return (f"from your conversation on "
                            f"{time.strftime('%d %B %Y', time.localtime(ts))}")
            return ("you told me directly" if row and row["source"] == "user"
                    else "from our conversation")
        return f"from {hit.when}"


def _clip(text: str, limit: int) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + "..."
