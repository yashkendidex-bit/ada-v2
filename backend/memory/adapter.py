"""Compatibility surface for `AgentKernel`.

`agent_kernel.MemoryBank` was the old implementation: five tables, an MD5-hash
embedding fallback, a full-table cosine scan, and six regexes for fact extraction.
This module keeps the *names* that the kernel's tools already call, so no tool
signature has to change, while every one of them now routes into the real memory
subsystem.

The rule here is that no method lies. `recall_episodes(query)` genuinely sorts by
relevance now instead of returning the two newest rows; `add_fact` genuinely preserves
the value it replaces instead of deleting it. Where the old signature cannot express
the new behaviour (a list of formatted strings cannot carry provenance), the method
returns the old shape but the underlying data has the richer one, retrievable through
`Memory.recall`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import Memory

log = logging.getLogger("ada.memory.adapter")


class KernelMemory:
    """Duck-types the old `MemoryBank` API over the new `Memory`."""

    def __init__(self, db_path: Path, llm: Optional[Callable[[str], Any]] = None):
        self._mem = Memory(db_path, llm=llm)
        self.db_path = Path(db_path)

    # ── pass-throughs ────────────────────────────────────────────────────────
    def __getattr__(self, name):
        # Anything not shimmed below (stats, health, consolidate, provenance, store…)
        # resolves to the real implementation rather than silently failing.
        return getattr(self._mem, name)

    @property
    def memory(self) -> Memory:
        return self._mem

    # ── core facts ───────────────────────────────────────────────────────────
    def add_fact(self, key, value, source="user", importance=1):
        return self._mem.remember_fact(str(key), str(value), importance=max(1, min(10, int(importance or 3))),
                                       source=source)

    def forget_fact(self, key):
        return self._mem.forget(key)

    def core_block(self, max_facts=40):
        return self._mem.core_block(max_facts=max_facts)

    # ── archival / recall ────────────────────────────────────────────────────
    def add_recall(self, content, kind="conversation", tags=None, embed=True):
        """Used by the document indexer. Documents are episodes like anything else."""
        eid = self._mem.encoder.capture(str(content), "", kind=kind)
        self._mem.retriever.invalidate()
        return eid

    def search_recall(self, query, k=5, since=None, kind=None):
        """Old signature returned row dicts. Kept, but relevance-ranked and hybrid.

        The old implementation loaded every row and did a Python cosine loop with an
        MD5 bag-of-words vector. This runs the four-channel retriever.
        """
        res = self._mem.recall(query, k=k)
        out = []
        for h in res.hits:
            if kind and h.kind != kind:
                continue
            if since is not None and (h.ts or 0) < since:
                continue
            out.append({
                "id": h.id,
                "kind": h.kind,
                "content": h.content,
                "ts": h.ts,
                "score": round(h.score, 4),
                "confidence": round(res.confidence, 4),
                "channels": h.channels,
            })
        # Re-sort after filtering: the retriever appends fact and reflection hits after
        # the episodes, so a filtered list is no longer in score order.
        out.sort(key=lambda d: -d["score"])
        return out[:k]

    def recall_episodes(self, query=None, k=3):
        """Old signature returned formatted strings. Now genuinely query-ranked.

        The old version took `LIMIT 200 ORDER BY ts DESC` and only sorted by similarity
        `if query:` — and `build_context` called it with an empty query, so in practice
        it returned the two newest rows labelled "related memories".
        """
        if not query:
            rows = self._mem.store.query(
                "SELECT content, ts_event, ts FROM recall_entries"
                " WHERE kind='conversation' AND COALESCE(quarantined,0)=0"
                " AND fidelity_level < 5 ORDER BY ts_event DESC LIMIT ?", (k,))
            return [_fmt(r["content"], r["ts_event"] or r["ts"]) for r in rows]
        res = self._mem.recall(query, k=k, include_facts=False, include_reflections=False)
        return [_fmt(h.content, h.ts) for h in res.hits]

    def recent_exchanges(self, k=8):
        return self._mem._recent_exchanges(k=k)

    def add_episode(self, summary, entities=None):
        """Episodes are no longer a separate table — they are the same row, summarised.

        The old design wrote the conversation to `recall_entries` and a truncated copy
        to `episodes`, which is why the two drifted (27 rows each, different content).
        """
        return self._mem.encoder.capture(str(summary), "", kind="conversation")

    def remember_context(self, user_text, ada_reply):
        """The per-turn write. Capture is synchronous; enrichment runs off-thread."""
        return self._mem.remember(user_text, ada_reply, background=True)

    def related_entities(self, entity, k=6):
        rows = self._mem.store.query(
            "SELECT a, b, rel FROM entity_edges WHERE (a=? OR b=?) AND t_invalid IS NULL"
            " ORDER BY weight DESC, ts DESC LIMIT ?", (entity, entity, k))
        return [f"{r['a']} <{r['rel']}> {r['b']}" for r in rows]

    # ── notebook ─────────────────────────────────────────────────────────────
    def notebook_read(self):
        rows = self._mem.store.query("SELECT section, content FROM notebook ORDER BY section")
        return {r["section"]: r["content"] for r in rows}

    def notebook_write(self, section, content):
        self._mem.store.execute(
            "INSERT INTO notebook(section, content, updated) VALUES(?,?,?)"
            " ON CONFLICT(section) DO UPDATE SET content=excluded.content,"
            " updated=excluded.updated",
            (section, content, time.time()))
        return f"Notebook section '{section}' updated."

    # ── introspection ────────────────────────────────────────────────────────
    def health(self):
        return self._mem.health()

    def stats(self):
        return self._mem.stats()

    def consolidate(self, **kw):
        return self._mem.consolidate(**kw)


def _fmt(content: str, ts: float | None) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "unknown time"
    text = (content or "").strip().replace("\n", " ")
    if len(text) > 400:
        text = text[:400].rsplit(" ", 1)[0] + "..."
    return f"[{when}] {text}"
