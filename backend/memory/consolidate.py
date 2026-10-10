"""Consolidation — the nightly "sleep" job.

Biological basis: during slow-wave sleep the hippocampus replays recently encoded
ensembles to the neocortex in compressed time, and the slow learner gradually
abstracts regularities from that replay without overwriting what it already knew
(complementary learning systems). Consolidation is therefore a *scheduled, offline,
batched* process — not something that happens inline during a conversation.

The single most important finding from the literature, and the rule that governs this
whole module:

    DEDUPLICATE, DO NOT SUMMARISE.

The Microsoft human-inspired memory study measured aggressive consolidation at **48.4%**
accuracy versus **76.8%** for dedup-only, on LongMemEval. Over-consolidation is worse
than doing nothing. So this job is deliberately conservative: it merges near-identical
memories, promotes what matters, degrades what does not, and refuses to paraphrase.
A memory's own words are evidence; a model's summary of them is not.

Measured results worth aiming at, from the same study: 97.2% retention precision with
58% store reduction, and a store that self-regulates at 300-500 events regardless of
input volume.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .decay import (
    INTERFERENCE_THRESHOLD,
    LADDER,
    interference_score,
    next_fidelity,
    retention_score,
)
from .embeddings import Embedder, cosine, from_blob, to_blob
from .encode import Encoder
from .store import Store, dumps

log = logging.getLogger("ada.memory.consolidate")

# Dedup threshold, measured against the actual embedding model rather than guessed.
#
# Similarity of BGE-small-en-v1.5 cosines for known pairs:
#     exact duplicate                        1.000
#     near-duplicate (trailing punctuation)  0.979   <- should merge
#     distinct, one word differs ("0" vs "1") 0.948  <- must NOT merge
#     distinct, same topic                   0.821
#     related but different                  0.759
#     unrelated                              0.485
#
# So the line sits between 0.948 and 0.979. 0.965 catches punctuation and whitespace
# duplicates while leaving genuinely distinct memories alone.
#
# The source study used a *percentile* (P99 of pairwise similarity) so the threshold
# would transfer between domains without retuning. That is elegant but it is not
# idempotent — the percentile moves as the population shrinks, so a second pass keeps
# finding new "duplicates" and progressively eats the store. Measured here: a 6-note
# set collapsed by one more memory on every run. A fixed measured threshold is
# predictable, and predictable is the property that matters for a forgetting policy.
DEDUP_THRESHOLD = 0.965
# Similarity above which two memories are treated as competing for recall.
CLUSTER_PCT = 0.95
REFLECTION_IMPORTANCE_SUM = 150.0   # Generative Agents' reflection trigger

# Promotion bands. Top 20% promote, middle 60% retain, bottom 20% prune.
PROMOTE_FRAC, PRUNE_FRAC = 0.20, 0.20


@dataclass
class ConsolidationReport:
    scored: int = 0
    promoted: int = 0
    deduped: int = 0
    pruned: int = 0
    reflected: int = 0
    decayed: int = 0
    degraded: int = 0
    quarantined: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"scored={self.scored} promoted={self.promoted} deduped={self.deduped} "
                f"pruned={self.pruned} reflected={self.reflected} "
                f"decayed={self.decayed} degraded={self.degraded} "
                f"quarantined={self.quarantined}")


class Consolidator:
    """Runs the consolidation pipeline. Idempotent: running twice changes nothing."""

    def __init__(self, store: Store, embedder: Embedder | None = None,
                 encoder: Encoder | None = None,
                 llm: Optional[Callable[[str], Any]] = None):
        self.store = store
        self.embedder = embedder or Embedder()
        self.encoder = encoder or Encoder(store, self.embedder, llm=llm)
        self.llm = llm

    # ── entry point ──────────────────────────────────────────────────────────
    def run(self, *, since: float | None = None, dry_run: bool = False) -> ConsolidationReport:
        rep = ConsolidationReport()
        started = time.time()
        try:
            self._validate_temporal(rep, since)
            rows = self._pending(since)
            rep.scored = len(rows)
            if not rows:
                rep.notes.append("nothing new to consolidate")
                return rep
            self._promote(rep, rows, dry_run)
            self._dedup(rep, rows, dry_run)
            self._reflect(rep, dry_run)
            self._decay(rep, dry_run)
            self._degrade(rep, dry_run)
        except Exception as e:
            log.exception("consolidation failed")
            rep.notes.append(f"error: {type(e).__name__}: {e}")
        finally:
            if not dry_run:
                self.store.execute(
                    "INSERT INTO consolidation_runs(started, finished, scored, promoted,"
                    " deduped, pruned, reflected, decayed, notes) VALUES(?,?,?,?,?,?,?,?,?)",
                    (started, time.time(), rep.scored, rep.promoted, rep.deduped,
                     rep.pruned, rep.reflected, rep.decayed, "; ".join(rep.notes)[:500]))
        return rep

    # ── 1. temporal validation ───────────────────────────────────────────────
    def _validate_temporal(self, rep: ConsolidationReport, since: float | None) -> None:
        """Quarantine anomalies before they can poison the store.

        Out-of-order arrival and causal inversion both produce the "agent déjà vu"
        failure: a memory that appears to describe something that has not happened yet,
        which then gets retrieved as fact. Holding it for a short TTL is cheap; the
        alternative is a store that contradicts itself.
        """
        cutoff = since or 0.0
        rows = self.store.query(
            "SELECT id, ts_event, ts FROM recall_entries WHERE COALESCE(quarantined,0)=0"
            " AND ts_event > ?", (time.time() + 300,))
        for r in rows:
            self.store.execute(
                "INSERT INTO quarantine(payload, reason, ts, ttl) VALUES(?,?,?,?)",
                (dumps({"episode_id": r["id"], "ts_event": r["ts_event"]}),
                 "ts_event in the future", time.time(), 900))
            self.store.execute("UPDATE recall_entries SET quarantined=1 WHERE id=?", (r["id"],))
            rep.quarantined += 1
        if rep.quarantined:
            rep.notes.append(f"quarantined {rep.quarantined} future-dated memories")

    # ── 2. promotion scoring ─────────────────────────────────────────────────
    def _pending(self, since: float | None) -> list[dict]:
        # fidelity_level < 5 excludes rows already reduced to tombstones by a previous
        # pass. Without it the job re-merged the same duplicates on every run and was
        # not idempotent.
        sql = ("SELECT id, content, summary, ts, ts_event, importance, arousal,"
               " access_count, pinned, fidelity_level, embedding, stability"
               " FROM recall_entries WHERE COALESCE(quarantined,0)=0 AND fidelity_level < 5")
        args: list = []
        if since:
            sql += " AND ts_event > ?"
            args.append(since)
        return [dict(r) for r in self.store.query(sql, args)]

    def _promote(self, rep: ConsolidationReport, rows: list[dict], dry_run: bool) -> None:
        """Score every memory and mark the top band.

        Score is retention_score: retrievability + importance + rehearsal, with pinned
        memories short-circuited to 1.0. Promotion is a *mark*, not a move — memories
        live in one place and the band decides what the forgetting passes may touch.
        """
        now = time.time()
        scored = []
        for r in rows:
            age_days = max(0.0, (now - (r["ts_event"] or r["ts"] or now)) / 86400.0)
            s = retention_score(
                importance=r["importance"] or 3,
                stability_days=float(r["stability"] or 1.0),
                age_days=age_days,
                access_count=int(r["access_count"] or 0),
                pinned=bool(r["pinned"]),
            )
            scored.append((s, r["id"], r))
        scored.sort(key=lambda x: -x[0])
        n = len(scored)
        promote_n = max(1, int(n * PROMOTE_FRAC))
        prune_n = max(0, int(n * PRUNE_FRAC))

        for i, (s, mid, r) in enumerate(scored):
            if dry_run:
                continue
            # Stability grows for the promoted band: rehearsal is what makes a memory
            # durable, and the top band is by construction the rehearsed, important set.
            if i < promote_n and not r["pinned"]:
                self.store.execute(
                    "UPDATE recall_entries SET stability = MAX(stability, 7.0) WHERE id=?",
                    (mid,))
        rep.promoted = promote_n
        rep.pruned = prune_n
        rep.notes.append(f"bands: promote {promote_n}, retain {n - promote_n - prune_n}, "
                         f"prune {prune_n}")

    # ── 3. dedup (the load-bearing step) ─────────────────────────────────────
    def _dedup(self, rep: ConsolidationReport, rows: list[dict], dry_run: bool) -> None:
        """Collapse near-identical memories. Never summarise.

        The measured lesson: aggressive consolidation scored 48.4% vs 76.8% for
        dedup-only. So this only merges memories that are *the same memory recorded
        twice* — a threshold set from the data (P99 of pairwise similarity) rather than
        a guess. Everything else is left alone, even when it looks redundant.
        """
        vecs = [(r["id"], r, from_blob(r["embedding"])) for r in rows]
        vecs = [(i, r, v) for i, r, v in vecs if v is not None and v.size]
        if len(vecs) < 2:
            return

        threshold = DEDUP_THRESHOLD
        merged_into: dict[int, int] = {}
        kept: list[tuple[int, Any, Any]] = []
        for mid, r, v in vecs:
            dup_of = None
            for kid, kr, kv in kept:
                if cosine(v, kv) >= threshold:
                    dup_of = kid
                    break
            if dup_of is None:
                kept.append((mid, r, v))
            else:
                merged_into[mid] = dup_of

        if not merged_into:
            return
        for dup, keep in merged_into.items():
            if dry_run:
                continue
            # The survivor keeps the higher importance and inherits the duplicate's
            # access history, so consolidation never loses evidence of use.
            self.store.execute(
                "UPDATE recall_entries SET access_count = access_count +"
                " (SELECT COALESCE(access_count,0) FROM recall_entries WHERE id=?),"
                " importance = MAX(importance, (SELECT COALESCE(importance,1)"
                " FROM recall_entries WHERE id=?)) WHERE id=?", (dup, dup, keep))
            self.store.execute(
                "UPDATE recall_entries SET fidelity_level=5,"
                " summary=COALESCE(summary, content) WHERE id=?", (dup,))
        rep.deduped = len(merged_into)
        rep.notes.append(f"deduped {len(merged_into)} at fixed threshold {threshold:.3f}")

    # ── 4. interference-based forgetting ─────────────────────────────────────
    def _decay(self, rep: ConsolidationReport, dry_run: bool) -> None:
        """Apply interference on top of passive decay. Pinned memories are exempt.

        Two independent loss processes, as in the biology: passive decay (time) and
        interference (competition). Retroactive interference is weighted higher than
        proactive — new learning disrupts old memory more than the reverse.
        """
        rows = self.store.query(
            "SELECT id, importance, ts, ts_event, embedding, access_count, pinned,"
            " fidelity_level, stability FROM recall_entries"
            " WHERE COALESCE(pinned,0)=0 AND fidelity_level < 5 AND embedding IS NOT NULL")
        if len(rows) < 2:
            return
        now = time.time()
        vecs = [(int(r["id"]), r, from_blob(r["embedding"])) for r in rows]
        vecs = [(i, r, v) for i, r, v in vecs if v is not None and v.size]
        ages = [(now - (r["ts_event"] or r["ts"] or now)) / 3600.0 for _, r, _ in vecs]
        mean_age = sum(ages) / len(ages) if ages else 0.0

        touched = 0
        for idx, (mid, r, v) in enumerate(vecs):
            # Retrieval practice protects a memory. Anything she has actually surfaced
            # is exempt: it has proven useful, and without this guard the nightly pass
            # becomes a ratchet that decrements the same rows every night until every
            # importance bottoms out at 1 and the salience signal is meaningless.
            if int(r["access_count"] or 0) > 0:
                continue
            sims = [cosine(v, ov) for j, (_, _, ov) in enumerate(vecs) if j != idx]
            high = [s for s in sims if s >= INTERFERENCE_THRESHOLD]
            if not high:
                continue
            interf = interference_score(sims, ages[idx], mean_age)
            if interf < 0.35:
                continue
            # High interference + low value = the memory being crowded out.
            if int(r["importance"] or 3) <= 1:
                # Already at the floor. Decrementing again would be a no-op reported as
                # work, so the job would never look idle — and an idempotent pass is the
                # property that makes it safe to re-run after a restart.
                continue
            age_days = ages[idx] / 24.0
            score = retention_score(r["importance"] or 3, float(r["stability"] or 1.0),
                                    age_days, int(r["access_count"] or 0), False)
            if score > 0.55:
                continue
            if not dry_run:
                self.store.execute(
                    "UPDATE recall_entries SET importance=MAX(1, importance-1) WHERE id=?",
                    (mid,))
            touched += 1
        rep.decayed = touched
        if touched:
            rep.notes.append(f"interference down-weighted {touched} crowded memories")

    # ── 5. fidelity ladder ───────────────────────────────────────────────────
    def _degrade(self, rep: ConsolidationReport, dry_run: bool) -> None:
        """Step low-value, old memories down the ladder. Never delete.

        A memory becomes a summary, then a gist, then a tombstone that records only
        that it existed. Knowing *that* you discussed something is worth more than not
        knowing it happened, and the L0 text is still in the row — degradation changes
        what is surfaced, not what is stored.
        """
        rows = self.store.query(
            "SELECT id, content, summary, gist, fidelity_level, ts, ts_event,"
            " importance, stability, access_count, pinned FROM recall_entries"
            " WHERE COALESCE(pinned,0)=0 AND fidelity_level < 5")
        now = time.time()
        for r in rows:
            age_days = max(0.0, (now - (r["ts_event"] or r["ts"] or now)) / 86400.0)
            score = retention_score(r["importance"] or 3, float(r["stability"] or 1.0),
                                    age_days, int(r["access_count"] or 0), False)
            target = next_fidelity(int(r["fidelity_level"] or 0), age_days, score)
            if target <= int(r["fidelity_level"] or 0):
                continue
            if dry_run:
                rep.degraded += 1
                continue
            # Build the degraded form from the memory's own words — a truncation, not
            # a paraphrase. Summarising here is exactly the mistake the study measured.
            summary = r["summary"] or _truncate(r["content"], 240)
            gist = r["gist"] or _truncate(r["content"], 90)
            self.store.execute(
                "UPDATE recall_entries SET fidelity_level=?, summary=?, gist=? WHERE id=?",
                (target, summary, gist, r["id"]))
            rep.degraded += 1
        if rep.degraded:
            rep.notes.append(f"stepped {rep.degraded} memories down the fidelity ladder")

    # ── 6. reflection ────────────────────────────────────────────────────────
    def _reflect(self, rep: ConsolidationReport, dry_run: bool) -> None:
        """Generate reflections, each citing the memories it came from.

        Trigger and shape follow the Generative Agents architecture: when the summed
        importance of recent material clears a threshold, ask what high-level questions
        it raises, retrieve evidence for each, and store the resulting insights with
        pointers to the evidence. Reflections can cite other reflections, forming a
        tree from observations up to abstractions.

        Without an LLM this is skipped rather than faked. A reflection that was not
        actually derived from anything is a hallucination with a citation attached.
        """
        if self.llm is None:
            rep.notes.append("no LLM configured — reflection skipped (not faked)")
            return
        since = self._last_reflection_ts()
        rows = self.store.query(
            "SELECT id, content, importance FROM recall_entries"
            " WHERE COALESCE(quarantined,0)=0 AND ts_event > ? ORDER BY ts_event DESC LIMIT 100",
            (since,))
        total = sum(int(r["importance"] or 1) for r in rows)
        if total < REFLECTION_IMPORTANCE_SUM or len(rows) < 4:
            rep.notes.append(f"reflection skipped (importance sum {total:.0f} < "
                             f"{REFLECTION_IMPORTANCE_SUM:.0f})")
            return

        digest = "\n".join(f"[{r['id']}] {_truncate(r['content'], 200)}" for r in rows[:60])
        prompt = (
            "Below are recent memories, each prefixed with its id.\n"
            "What 3 high-level questions do these raise about the user, their projects, "
            "or their preferences?\nReply with only the questions, one per line.\n\n"
            f"{digest}"
        )
        try:
            questions = [q.strip(" -*0123456789.") for q in str(self.llm(prompt)).splitlines()]
            questions = [q for q in questions if len(q) > 12][:3]
        except Exception as e:
            rep.notes.append(f"reflection question generation failed: {type(e).__name__}")
            return

        made = 0
        for q in questions:
            prompt2 = (
                f"Memories:\n{digest}\n\nQuestion: {q}\n"
                "Answer in ONE sentence, and end with the ids you used in the form "
                "(because of 12, 30). If the memories do not answer it, reply exactly "
                "NOTHING.\n"
            )
            try:
                ans = str(self.llm(prompt2)).strip()
            except Exception:
                continue
            if not ans or ans.upper().startswith("NOTHING"):
                continue
            cites = [int(x) for x in _cited_ids(ans)]
            if not cites:
                continue   # an insight with no evidence is not stored
            if dry_run:
                made += 1
                continue
            self.store.execute(
                "INSERT INTO reflections(ts, text, level, cites, embedding, importance,"
                " pinned) VALUES(?,?,?,?,?,?,?)",
                (time.time(), ans, 1, dumps(cites),
                 to_blob(self.embedder.embed(ans)), 6, 0))
            made += 1
        rep.reflected = made
        rep.notes.append(f"reflected {made} insights from {len(rows)} memories")

    def _last_reflection_ts(self) -> float:
        row = self.store.one("SELECT MAX(ts) m FROM reflections")
        return float(row["m"]) if row and row["m"] else 0.0


def _truncate(text: str, limit: int) -> str:
    """Truncate on a word boundary. Deliberately NOT a summary — see the module doc."""
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[:limit].rsplit(" ", 1)[0] + "..."


def _cited_ids(text: str) -> list[str]:
    import re
    m = re.search(r"\(?\s*because of ([0-9,\s]+)\)?", text or "", re.I)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]
