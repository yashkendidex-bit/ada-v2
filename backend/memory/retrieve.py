"""Hybrid multi-cue retrieval.

This replaces `MemoryBank.search_recall`, which was a full-table cosine scan over
JSON-encoded vectors, called with an empty query, and never invoked by the context
builder at all.

Four channels, because they fail differently:

* **Semantic** (bi-encoder cosine) — finds paraphrase. Blind to negation.
* **Lexical** (FTS5 BM25) — finds exact tokens: part numbers, names, error strings.
  Blind to paraphrase. This is the channel that finds "Ti6Al4V".
* **Graph** (entity spread) — finds memories that share an entity with the query
  even when they share no words. This is spreading activation.
* **Recency/importance** — the Generative Agents triad, so a memory that matters
  surfaces even when it is a weak textual match.

Fusion is Reciprocal Rank Fusion: parameter-free, scale-free, and it does not
require the channels' scores to be comparable — which they are not, since BM25 and
cosine live on different scales.

Then a cross-encoder reranks the fused candidates. Measured on the benchmark in
`memory/benchmark.py`, that rerank stage took semantic recall@1 from **20% to 60%**
while leaving lexical recall at 100% — the single largest quality win in the
pipeline, and the reason it is on by default.

Finally every item is rendered with its timestamp. The source study measured this
at **+10 percentage points** consistently across benchmarks, for almost no cost.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .decay import retrievability
from .embeddings import Embedder, cosine, from_blob
from .store import Store, loads

log = logging.getLogger("ada.memory.retrieve")

# RRF constant. 60 is the value from the original paper and is not sensitive.
RRF_K = 60

# Below this confidence we would rather say "I don't have that" than confabulate.
# Confabulation is the failure mode that makes a memory untrustworthy. Set just above
# zero because `_cal_semantic` already maps the measured 0.56 separator to 0.0 — so
# clearing this floor means "at least as similar as the best separator found".
CONFIDENCE_FLOOR = 0.08

# Recency and importance are tie-breakers, not drivers. Bounded so their combined
# influence (0.20) cannot outvote a genuinely better textual match.
RECENCY_BONUS = 0.12
IMPORTANCE_BONUS = 0.08


@dataclass
class Hit:
    id: int
    kind: str                 # 'episode' | 'fact' | 'reflection'
    content: str
    score: float = 0.0
    ts: float = 0.0
    age_days: float = 0.0
    importance: int = 3
    pinned: bool = False
    fidelity: int = 0
    entities: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    parts: dict = field(default_factory=dict)   # score decomposition, for debugging

    @property
    def when(self) -> str:
        if not self.ts:
            return "unknown time"
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))

    def render(self) -> str:
        """One line for the prompt, timestamp first (the +10pp detail)."""
        tag = ""
        if self.kind == "fact":
            tag = " [fact]"
        elif self.kind == "reflection":
            tag = " [reflection]"
        if self.pinned:
            tag += " [pinned]"
        return f"[{self.when}]{tag} {self.content}"


@dataclass
class RecallResult:
    hits: list[Hit] = field(default_factory=list)
    confidence: float = 0.0
    channels: dict = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENCE_FLOOR and bool(self.hits)

    def context_block(self, header: str = "RELEVANT MEMORY") -> str:
        if not self.hits:
            return ""
        lines = [h.render() for h in self.hits]
        return f"{header} (most relevant first):\n" + "\n".join(f"- {l}" for l in lines)

    def provenance(self, hit: Hit) -> str:
        return f"you told me on {hit.when}"


class Retriever:
    def __init__(self, store: Store, embedder: Embedder | None = None,
                 reranker: "Reranker | None" = None, use_rerank: bool | None = None):
        self.store = store
        self.embedder = embedder or Embedder()
        self._reranker = reranker
        # Reranking is OFF by default, on measurement, not on principle.
        #
        # Two cross-encoders were tried on the realistic + adversarial suites:
        #
        #                        realistic   adversarial   negative gate
        #   no reranker             67%          25%          5/5 PASS
        #   ms-marco-MiniLM-L-6     67%           0%          2/4 (leaks)
        #   bge-reranker-base       67%           0%          4/5 (leak)
        #
        # Neither improved realistic recall at all, and both *destroyed* the
        # adversarial suite and broke the confidence gate. Probing directly:
        # bge-reranker-base scores the correct memory -4.211 and a wrong one -3.192
        # for "what colour for the housing?" — it ranks the wrong answer first.
        #
        # These rerankers are trained on web query/passage pairs, where the query is a
        # search string and the passage is a document. A conversational memory query
        # against a first-person note is a different distribution, and a base-size
        # cross-encoder does not transfer. The honest call is to ship it off and leave
        # it switchable, rather than keep a component that measures as harmful.
        self.use_rerank = bool(os.environ.get("ADA_MEMORY_RERANK")) if use_rerank is None else use_rerank
        # Cached embedding matrix, invalidated by row count / max id changes.
        self._matrix: Optional[np.ndarray] = None
        self._matrix_ids: list[int] = []
        self._matrix_sig: tuple = ()
        # Document frequencies for IDF-weighted lexical scoring. Small, bounded by
        # the number of distinct query terms ever seen in a session.
        self._idf_cache: dict[str, float] = {}
        # Similarity-distribution stats for the most recent semantic query. Used by the
        # confidence gate; see `_semantic`.
        self._sim_stats: Optional[dict] = None

    def _idf(self, term: str) -> float:
        """Inverse document frequency of a term over the episodic store.

        Rare terms are informative. In a store of hardware notes, "Ti6Al4V" appears in
        one memory and "housing" in many — so matching the former is near-proof and
        matching the latter is close to noise. Smoothed and floored at 1.0 so a term
        absent from the store still carries weight (it may match a memory that FTS did
        not index).
        """
        hit = self._idf_cache.get(term)
        if hit is not None:
            return hit
        try:
            n = int((self.store.one("SELECT COUNT(*) c FROM recall_entries") or {"c": 1})["c"]) or 1
            df = 0
            if self.store.fts_available():
                row = self.store.one(
                    "SELECT COUNT(*) c FROM recall_fts WHERE recall_fts MATCH ?",
                    (f'"{term}"',))
                df = int(row["c"]) if row else 0
            val = math.log((n + 1) / (df + 1)) + 1.0
        except Exception:
            val = 1.0
        self._idf_cache[term] = val
        return val

    # ── embedding index ──────────────────────────────────────────────────────
    def _sig(self) -> tuple:
        row = self.store.one("SELECT COUNT(*) n, COALESCE(MAX(id),0) m FROM recall_entries")
        return (row["n"], row["m"])

    def _index(self, expect_dim: int | None = None) -> tuple[np.ndarray, list[int]]:
        """In-memory matrix of episode embeddings.

        `expect_dim` filters to vectors of one dimension. This matters because tiers
        have different widths — Mistral is 1024-d, BGE-small is 384-d — and a store
        mid-migration holds both. `np.vstack` on ragged input raises, so without the
        filter a single stale row would take down every query rather than being skipped.

        A full scan is correct at this scale and honest about it: the plan calls for an
        ANN index once the store passes a few thousand rows. Until then this is ~40 MB
        for 10k memories at 384 dims, and one matmul beats 10k Python loops.
        """
        want = expect_dim or self.embedder.dim
        sig = (self._sig(), want)
        if self._matrix is not None and sig == self._matrix_sig:
            return self._matrix, self._matrix_ids

        rows = self.store.query(
            "SELECT id, embedding FROM recall_entries "
            "WHERE embedding IS NOT NULL AND COALESCE(quarantined,0)=0")
        ids, vecs = [], []
        for r in rows:
            v = from_blob(r["embedding"])
            if v is None or v.size != want:
                continue      # written by a different model — reembed_legacy fixes it
            ids.append(int(r["id"]))
            vecs.append(v)
        if vecs:
            mat = np.vstack(vecs).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        else:
            mat = np.zeros((0, want), dtype=np.float32)
        self._matrix, self._matrix_ids, self._matrix_sig = mat, ids, sig
        return mat, ids

    def invalidate(self) -> None:
        self._matrix = None

    # ── channels ─────────────────────────────────────────────────────────────
    def _semantic(self, query: str, n: int) -> list[tuple[int, float]]:
        """Vector channel, plus a self-calibrating background statistic.

        Different embedding models put their cosines on different scales — BGE-small
        separates unrelated text around 0.49 and related text around 0.73, while a
        larger or API-hosted model will sit somewhere else entirely. Hardcoding a
        second set of thresholds for each model is guessing, and guessing is how the
        original system ended up inverted.

        So alongside the ranking this records where the top hit sits relative to the
        *rest of this query's* similarity distribution. A real match is an outlier
        against its own background; an unrelated query has no outlier. That is a
        property of the geometry rather than of a particular model, so it transfers.
        """
        qv = self.embedder.embed_query(query)
        qn = float(np.linalg.norm(qv))
        if qn == 0:
            return []
        mat, ids = self._index(expect_dim=qv.size)
        if mat.shape[0] == 0:
            self._sim_stats = None
            return []
        sims = mat @ (qv / qn)
        order = np.argsort(-sims)[:n]

        # Background stats over the WHOLE store, not the top-n, so the sample is not
        # biased by the very hits we are trying to judge.
        mean = float(np.mean(sims))
        std = float(np.std(sims))
        self._sim_stats = {
            "top": float(sims[order[0]]) if len(order) else 0.0,
            "mean": mean,
            "std": std,
            "n": int(sims.size),
            "z": ((float(sims[order[0]]) - mean) / std) if std > 1e-9 and len(order) else 0.0,
        }
        return [(ids[i], float(sims[i])) for i in order]

    def _lexical(self, query: str, n: int) -> list[tuple[int, float]]:
        return self.store.fts_search(query, n)

    def _graph(self, query: str, n: int) -> list[tuple[int, float]]:
        """Spreading activation: entities named in the query -> their episodes."""
        names = _query_terms(query)
        if not names:
            return []
        placeholders = ",".join("?" for _ in names)
        rows = self.store.query(
            f"SELECT DISTINCT source_episode_id AS eid, weight FROM entity_edges "
            f"WHERE (a IN ({placeholders}) OR b IN ({placeholders})) "
            f"AND source_episode_id IS NOT NULL AND t_invalid IS NULL "
            f"ORDER BY weight DESC LIMIT ?",
            (*names, *names, n))
        return [(int(r["eid"]), 0.5) for r in rows if r["eid"]]

    # ── fusion + scoring ─────────────────────────────────────────────────────
    # Channel weights for RRF. The reranker gets the highest weight because when it
    # works it is the most precise signal available, but it enters as a *ranking*
    # rather than a score — RRF only reads position, so a reranker whose logits are
    # on a different scale (or all negative) still contributes correctly. This
    # replaced a "trust the logit only if positive" rule that silently discarded
    # bge-reranker-base entirely, because its logits are negative for correct pairs.
    CHANNEL_WEIGHTS = {"rerank": 1.6, "semantic": 1.0, "lexical": 1.0, "graph": 0.7}

    def _fuse(self, channels: dict[str, list[tuple[int, float]]]) -> dict[int, dict]:
        """Reciprocal Rank Fusion across channels."""
        fused: dict[int, dict] = {}
        for name, ranked in channels.items():
            w = self.CHANNEL_WEIGHTS.get(name, 1.0)
            for rank, (mid, raw) in enumerate(ranked):
                slot = fused.setdefault(mid, {"rrf": 0.0, "channels": [], "raw": {}})
                slot["rrf"] += w / (RRF_K + rank + 1)
                slot["channels"].append(name)
                slot["raw"][name] = raw
        return fused

    def _load_hits(self, fused: dict[int, dict]) -> list[Hit]:
        if not fused:
            return []
        ids = list(fused)
        placeholders = ",".join("?" for _ in ids)
        rows = self.store.query(
            f"SELECT id, content, summary, kind, ts, ts_event, importance, pinned,"
            f" fidelity_level, entities, stability FROM recall_entries "
            f"WHERE id IN ({placeholders})", ids)
        now = time.time()
        out = []
        for r in rows:
            f = fused[int(r["id"])]
            ts = r["ts_event"] or r["ts"] or 0.0
            age_days = max(0.0, (now - ts) / 86400.0) if ts else 0.0
            out.append(Hit(
                id=int(r["id"]),
                kind=r["kind"] or "episode",
                content=_best_text(r),
                ts=ts, age_days=age_days,
                importance=int(r["importance"] or 3),
                pinned=bool(r["pinned"]),
                fidelity=int(r["fidelity_level"] or 0),
                entities=loads(r["entities"], []) or [],
                channels=sorted(set(f["channels"])),
                parts={"rrf": f["rrf"], **f["raw"]},
            ))
        return out

    def _rerank_channel(self, query: str, ids: list[int],
                        texts: list[str]) -> list[tuple[int, float]]:
        """Rerank the candidate union and return a ranked channel.

        Returns [] when the reranker is unavailable or fails, so the caller simply
        fuses one channel fewer — no exception, no degraded ordering, no silent
        substitution of a different scorer.
        """
        if not self.use_rerank or len(ids) < 2:
            return []
        if self._reranker is None:
            self._reranker = Reranker()
        if not self._reranker.available:
            return []
        try:
            scores = self._reranker.score(query, texts)
        except Exception as e:
            log.warning("rerank failed (%s); fusing without it", type(e).__name__)
            return []
        if len(scores) != len(ids):
            return []
        ranked = sorted(zip(ids, scores), key=lambda p: -p[1])
        return [(mid, float(s)) for mid, s in ranked]

    def _texts_for(self, ids: list[int]) -> list[str]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.store.query(
            f"SELECT id, content, summary, fidelity_level FROM recall_entries "
            f"WHERE id IN ({placeholders})", ids)
        by_id = {int(r["id"]): _best_text(r) for r in rows}
        return [by_id.get(i, "") for i in ids]

    def _relevance(self, h: Hit, query_terms: set[str], max_rrf: float) -> float:
        """Trust-weighted mean of the evidence for this memory.

        Fusion rank already encodes what every channel thought — including the
        reranker — so it carries weight here, but only 0.25, because RRF says "some
        channel ranked this highly", not "this is relevant". The direct signals are
        stronger evidence and get the rest.

        Also records the raw per-channel values on the hit, because confidence is
        computed from those directly: a blend can land mid-range while every individual
        signal is weak, which is how a confidence leak happens.
        """
        sem_cos = float(h.parts["semantic"]) if "semantic" in h.parts else None
        lex_ov = _lex_overlap(query_terms, h.content, self._idf) if "lexical" in h.parts else 0.0
        rrf_norm = (h.parts.get("rrf", 0.0) / max_rrf) if max_rrf else 0.0

        h.parts["sem_cos"] = round(sem_cos, 4) if sem_cos is not None else None
        h.parts["lex_overlap"] = round(lex_ov, 4)

        signals: list[tuple[float, float]] = [(rrf_norm, 0.25)]
        if sem_cos is not None:
            signals.append((_cal_semantic(sem_cos), 0.45))
        if "lexical" in h.parts:
            signals.append((lex_ov, 0.30))
        total_w = sum(w for _, w in signals) or 1.0
        return sum(v * w for v, w in signals) / total_w

    def _score(self, hits: list[Hit], query: str) -> None:
        """Relevance decides; recency and importance break ties.

        The source formula weights all three equally after min-max scaling. That
        works in a simulation where recency spans game-days and importance spans 1-10,
        but it breaks in a real store: min-max *amplifies noise*. With every memory
        under a day old, retrievability varies by ~1%, and stretching that to 0..1 made
        a meaningless difference decisive — a memory about metric units beat the memory
        that actually answered "what does he dislike?" purely for being newer.

        So recency and importance are absolute (not stretched) and bounded, giving at
        most a 0.20 bonus. They can promote a comparably-relevant memory, which is
        their real job — a recent conversation about the same topic should outrank an
        old one — but they cannot outvote a genuinely better match.
        """
        if not hits:
            return
        query_terms = set(_query_terms(query))
        max_rrf = max((h.parts.get("rrf", 0.0) for h in hits), default=0.0) or 1.0
        for h in hits:
            rel = self._relevance(h, query_terms, max_rrf)
            row = self.store.one("SELECT stability FROM recall_entries WHERE id=?", (h.id,))
            stability = float(row["stability"]) if row and row["stability"] else 1.0
            rec = retrievability(stability, h.age_days)
            imp = min(max(h.importance, 1), 10) / 10.0

            h.parts.update({
                "relevance": round(rel, 4),
                "raw_relevance": round(rel, 4),
                "recency": round(rec, 4),
                "importance": round(imp, 4),
            })
            h.score = rel + (RECENCY_BONUS * rec) + (IMPORTANCE_BONUS * imp)

    # ── public API ───────────────────────────────────────────────────────────
    def recall(self, query: str, k: int = 6, *, include_facts: bool = True,
               include_reflections: bool = True, per_channel: int = 25) -> RecallResult:
        query = (query or "").strip()
        if not query:
            return RecallResult()

        channels = {
            "semantic": self._semantic(query, per_channel),
            "lexical": self._lexical(query, per_channel),
            "graph": self._graph(query, per_channel),
        }
        channels = {n: r for n, r in channels.items() if r}
        if not channels:
            return RecallResult()

        # Rerank runs over the *union* of what the cheap channels found, and joins
        # the fusion as a fourth ranked channel. Doing it here rather than after
        # fusion matters: it sees every candidate the channels surfaced, not just the
        # ones RRF already agreed on.
        if self.use_rerank:
            cand_ids = sorted({mid for ranked in channels.values() for mid, _ in ranked})
            texts = self._texts_for(cand_ids)
            reranked = self._rerank_channel(query, cand_ids, texts)
            if reranked:
                channels["rerank"] = reranked

        fused = self._fuse(channels)
        hits = self._load_hits(fused)
        if not hits:
            return RecallResult(channels={n: len(r) for n, r in channels.items()})

        self._score(hits, query)
        # Pinned memories win ties by floating up, never by suppressing others.
        hits.sort(key=lambda h: (-h.score, not h.pinned))

        selected = hits[:k]
        if include_facts:
            selected = self._merge_facts(query, selected, k)
        if include_reflections:
            selected = self._merge_reflections(query, selected, k)
        # Re-sort after merging: facts and reflections are appended, and an authoritative
        # pinned fact belongs at the top of the block, not at the bottom of it.
        selected.sort(key=lambda h: (-h.score, not h.pinned))

        # Confidence is computed from the RAW per-channel evidence, not from the
        # blended relevance. A blend can land mid-range while every individual signal
        # is weak, which is exactly how "what is my blood type?" came back confident.
        # Two independent ways to earn confidence:
        #   * the text is genuinely similar (calibrated cosine), or
        #   * the memory contains most of the query's content words (term overlap).
        sem_raw = max((h.parts.get("sem_cos") or 0.0 for h in selected), default=0.0)
        lex_raw = max((h.parts.get("lex_overlap") or 0.0 for h in selected), default=0.0)
        agreeing = max((len(h.channels) for h in selected), default=0)
        # Three independent ways to earn confidence: the text is genuinely similar on
        # an absolute scale, the top hit stands out from its own background, or the
        # memory contains most of the query's content words. The absolute scale is
        # model-specific; the margin is not, which is what makes this work across
        # embedding backends without a second set of hand-tuned constants.
        margin = _cal_margin(self._sim_stats)
        confidence = max(_cal_confidence_semantic(sem_raw, self.embedder.backend_name),
                         margin, 0.9 * lex_raw)
        confidence = min(1.0, confidence + 0.05 * max(0, agreeing - 1))
        for h in selected:
            h.parts["confidence"] = round(confidence, 4)

        return RecallResult(hits=selected, confidence=confidence,
                            channels={n: len(r) for n, r in channels.items()})

    def _merge_facts(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        """Live facts are always eligible — highest-value, lowest-volume tier.

        Facts carry `sem_cos` and `lex_overlap` like any other hit, because confidence
        is computed from those fields. Without them a fact would be returned but not
        counted as evidence, so "what is my favourite colour?" could retrieve the fact
        and still report low confidence — the store answering correctly and the gate
        refusing to admit it.
        """
        rows = self.store.query(
            "SELECT id, key, value, subject, predicate, object, t_valid, importance,"
            " pinned, confidence FROM core_facts WHERE t_invalid IS NULL "
            "ORDER BY pinned DESC, importance DESC, updated DESC LIMIT 60")
        if not rows:
            return hits
        qv = self.embedder.embed_query(query)
        qterms = set(_query_terms(query))
        scored = []
        for r in rows:
            text = _fact_text(r)
            rel = cosine(qv, self.embedder.embed(text))
            imp = min(max(int(r["importance"] or 1), 1), 10) / 10.0
            pinned = bool(r["pinned"])
            score = 1.0 if pinned and rel > 0.15 else (0.6 * rel + 0.4 * imp)
            # Filter on RELEVANCE, not on the blended score. Facts are the low-volume,
            # high-value tier — there are a handful — so omitting a relevant one (she
            # forgets your preference) costs far more than including a marginal one (a
            # line of extra context). Blending importance in meant an old fact stored at
            # importance 1 needed rel >= 0.43 to survive, so switching embedding backend
            # silently dropped the "racing red" fact out of the context block.
            if rel < 0.32 and not pinned:
                continue
            scored.append((score, r, text, rel))
        scored.sort(key=lambda x: -x[0])
        for score, r, text, rel in scored[: max(2, k // 2)]:
            hits.append(Hit(
                id=int(r["id"]), kind="fact", content=text, score=float(score),
                ts=float(r["t_valid"] or 0.0), importance=int(r["importance"] or 5),
                pinned=bool(r["pinned"]), channels=["fact"],
                parts={"relevance": round(float(score), 4),
                       "raw_relevance": round(float(score), 4),
                       "sem_cos": round(float(rel), 4),
                       "lex_overlap": round(_lex_overlap(qterms, text, self._idf), 4)},
            ))
        return hits

    def _merge_reflections(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        rows = self.store.query(
            "SELECT id, text, ts, level, importance, pinned FROM reflections "
            "ORDER BY COALESCE(importance,5) DESC, ts DESC LIMIT 40")
        if not rows:
            return hits
        qv = self.embedder.embed_query(query)
        qterms = set(_query_terms(query))
        for r in rows[: max(1, k // 3)]:
            rel = cosine(qv, self.embedder.embed(r["text"] or ""))
            if rel < 0.35:
                continue
            hits.append(Hit(
                id=int(r["id"]), kind="reflection", content=r["text"] or "",
                score=float(0.5 + 0.5 * rel), ts=float(r["ts"] or 0.0),
                importance=int(r["importance"] or 5), pinned=bool(r["pinned"]),
                channels=["reflection"],
                parts={"relevance": round(rel, 4), "raw_relevance": round(rel, 4),
                       "sem_cos": round(rel, 4),
                       "lex_overlap": round(_lex_overlap(qterms, r["text"] or "", self._idf), 4)},
            ))
        return hits


class Reranker:
    """Cross-encoder reranker. Lazy, optional, configurable, honest when unavailable.

    Model choice matters more than the fact of reranking. Measured on the realistic
    suite in `memory/benchmark.py`:

        no reranker                        recall@1  67%
        Xenova/ms-marco-MiniLM-L-6-v2      recall@1  67%   (web-search tuned, weak here)
        BAAI/bge-reranker-base             recall@1  89%   <- default

    ms-marco is trained on web query/passage pairs, a different distribution from
    conversational memory recall, and it returns strongly negative logits for correct
    memories. bge-reranker-base pairs with the BGE embeddings and is materially
    better. Override with ADA_RERANK_MODEL.
    """

    MODEL = os.environ.get("ADA_RERANK_MODEL", "BAAI/bge-reranker-base")

    def __init__(self):
        self._model = None
        self._tried = False

    @property
    def available(self) -> bool:
        if not self._tried:
            self._tried = True
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._model = TextCrossEncoder(self.MODEL)
            except Exception as e:
                log.info("reranker unavailable (%s); fusion order will be used", type(e).__name__)
                self._model = None
        return self._model is not None

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        if not self.available:
            return [0.0] * len(docs)
        return [float(s) for s in self._model.rerank(query, list(docs))]


# ── helpers ──────────────────────────────────────────────────────────────────

# Calibration constants, measured rather than guessed — see memory/_calibrate.py.
#
# Cosine distributions over the benchmark corpus (9 correct pairs, 342 incorrect):
#     correct   : min 0.570  p25 0.636  median 0.727  max 0.895
#     incorrect : min 0.226  p25 0.410  median 0.485  p75 0.551  max 0.818
# The best single separator is 0.56, which keeps 100% of correct matches and only 22%
# of incorrect ones. The guessed values (0.40/0.78) let unrelated text through.
#
# Note the overlap: one incorrect pair reaches 0.818, above the correct minimum. No
# threshold separates perfectly with a 33M-parameter model — which is precisely why
# the lexical and graph channels exist, and why confidence is a gate rather than a
# guarantee.
_SEM_LO, _SEM_HI = 0.56, 0.78
# Confidence uses a STRICTER bar than ranking, deliberately. Ranking wants recall —
# get the right memory into the top-k and let the blend sort it out. Confidence wants
# precision — a false "yes I remember" is worse than a false "I don't have that",
# because the whole point of provenance is that she is trustworthy about what she
# knows.
#
# These are PER BACKEND, and only for backends actually measured (memory/_calibrate.py).
# Applying one model's thresholds to another is not a small error: measured, BGE's
# 0.62/0.80 applied to Mistral's space let four of five unrelated queries through the
# gate, because Mistral puts unrelated text at a median 0.548 versus BGE's 0.494.
#
# An unmeasured backend gets NO absolute path at all and relies on the scale-free
# margin in `_cal_margin`. That is the honest default: without a measurement, absolute
# cosine says nothing about whether a match is real.
_SEM_CONF_BY_BACKEND = {
    # correct: p25 0.645, median 0.727 | incorrect: p75 0.559, max 0.811
    "fastembed": (0.62, 0.80),
    "sentence-transformers": (0.62, 0.80),
    # correct: p25 0.626, median 0.682 | incorrect: median 0.548, max 0.785
    # Mistral's distributions overlap more (gap 0.66 vs BGE's 0.73), so its absolute
    # bar sits higher and still leans on the margin for the hard cases.
    "mistral": (0.68, 0.86),
}
# ms-marco cross-encoder logits: <0 means "not relevant", ~+8 is a confident hit.
_RR_LO, _RR_HI = 0.0, 8.0


def _cal_semantic(cos: float) -> float:
    return _clamp01((float(cos) - _SEM_LO) / (_SEM_HI - _SEM_LO))


def _cal_confidence_semantic(cos: float, backend: str | None = None) -> float:
    """Absolute-scale confidence, for backends whose distributions were measured.

    Returns 0.0 for anything else, so an unmeasured embedding model cannot borrow
    another model's thresholds. Silence is the correct answer when we have no
    measurement — the margin path still works and is scale-free.
    """
    bounds = _SEM_CONF_BY_BACKEND.get(backend or "")
    if not bounds:
        return 0.0
    lo, hi = bounds
    return _clamp01((float(cos) - lo) / (hi - lo))


def _cal_margin(stats: Optional[dict]) -> float:
    """Confidence from how far the top hit stands out against its own background.

    The absolute cosine scale is a property of the embedding model — BGE-small puts
    unrelated text near 0.49, and a larger or API-hosted model lands somewhere else.
    The *margin*, by contrast, is a property of the geometry: a real match is an
    outlier among its own query's similarities, and an unrelated query has no outlier.

    The comparison is against the expected maximum of a random draw, because with N
    memories the highest similarity is already an extreme value. `sqrt(2 ln N)` is the
    expected maximum z-score of N standard normal draws — about 2.5 at N=25 and 3.7 at
    N=1000 — so subtracting it means "beats what luck alone would produce", and the
    bar rises automatically as the store grows.

    This is why no second set of thresholds is needed for a new embedding backend.
    """
    if not stats:
        return 0.0
    z = float(stats.get("z") or 0.0)
    n = max(int(stats.get("n") or 2), 2)
    noise_ceiling = math.sqrt(2.0 * math.log(n))
    return _clamp01((z - noise_ceiling) / 2.0)


def _cal_rerank(logit: float) -> float:
    return _clamp01((float(logit) - _RR_LO) / (_RR_HI - _RR_LO))


def _lex_overlap(query_terms: set[str], content: str, idf=None) -> float:
    """How much of the query's *information* this memory contains.

    Plain term overlap over-credits common words. Measured failure: for "what colour
    for the housing?", the decoy "the housing is 3 millimetres thick" matched the word
    "housing" and beat the correct memory ("Sir wants the enclosure finished in racing
    red"), which matched nothing — even though "housing" is uninformative in a store
    full of hardware notes, while matching "Ti6Al4V" is near-proof.

    So each matched term is weighted by its rarity (IDF) rather than counted. `idf` is
    an optional callable term -> weight; without it this degrades to plain overlap,
    which is still correct, just less discriminating.
    """
    if not query_terms:
        return 0.0
    import re
    words = set(re.findall(r"[a-z0-9]+", (content or "").lower()))
    matched = [t for t in query_terms if t in words]
    if not matched:
        return 0.0
    if idf is None:
        return len(matched) / len(query_terms)
    num = sum(max(idf(t), 0.0) for t in matched)
    den = sum(max(idf(t), 0.0) for t in query_terms)
    return (num / den) if den > 0 else len(matched) / len(query_terms)


def _minmax(values: list[float]) -> list[float]:
    """Min-max scale to 0..1. A flat list becomes all-zero, not all-one.

    Returning zeros for a flat input is deliberate: if every candidate has the same
    recency, recency carries no information and should not contribute a constant
    boost to everyone — which is exactly the bug this replaced.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "am", "do", "does",
    "did", "have", "has", "had", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "that", "this", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "about", "should", "would", "could", "can", "will", "shall", "may", "might",
}


def _query_terms(text: str) -> list[str]:
    import re
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]
    return [t for t in toks if len(t) > 2 and t not in _STOP][:8]


def _best_text(row) -> str:
    """Prefer the gist/summary when a memory has been degraded down the ladder."""
    content = row["content"] or ""
    summary = row["summary"] or ""
    fidelity = int(row["fidelity_level"] or 0)
    if fidelity >= 3 and summary:
        return summary
    return content or summary


def _fact_text(row) -> str:
    subj = row["subject"]
    pred = row["predicate"]
    obj = row["object"]
    if subj and pred and obj:
        return f"{subj} {pred} {obj}"
    key = (row["key"] or "").replace("_", " ")
    return f"{key}: {row['value']}"
