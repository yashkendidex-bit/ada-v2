"""Routine mining — pure-stdlib frequent-sequence mining over the F4
perception timeline (Feature 3, nightly job).

Pipeline:
  1. discretize events into activity tokens per 15-min bucket
  2. PrefixSpan-style frequent-subsequence mining (gap-constrained,
     min support configurable, default 3 occurrences)
  3. 1-D clustering of occurrence start times → a (hour, support,
     confidence) routine per cluster
  4. write mined routines as edges  user —usually_does_at→ activity
     (source_episode='miner')

No LLM, no model downloads, stdlib + sqlite3 only. Deterministic:
ties break alphabetically / by bucket order, never by hash order.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

_WORD_RE = re.compile(r"[a-z][a-z0-9'\-]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "you", "your",
    "are", "was", "were", "has", "have", "had", "not", "but", "all",
    "she", "his", "her", "him", "its", "our", "their", "will", "would",
    "can", "just", "now", "then", "there", "here", "what", "when",
    "which", "who", "how", "said", "say", "about", "into", "over",
    "also", "very", "more", "than", "been", "did", "out", "one",
    "ada", "sir",
})


def _find_occurrences(stream: list[tuple[int, str]],
                      pattern: tuple[str, ...],
                      max_gap: int) -> list[int]:
    """Start bucket indices of every occurrence of `pattern` as a
    subsequence, with consecutive tokens at most `max_gap` buckets apart.
    Deterministic (bucket order)."""
    out: list[int] = []

    def dfs(p_i: int, s_i: int, prev_bkt: int | None, start_ts: int):
        if p_i == len(pattern):
            out.append(start_ts)
            return
        for k in range(s_i, len(stream)):
            bkt, tok = stream[k]
            if prev_bkt is not None and (bkt - prev_bkt) > max_gap:
                break
            if tok == pattern[p_i]:
                dfs(p_i + 1, k + 1, bkt,
                    start_ts if p_i else stream[k][0])

    if pattern:
        dfs(0, 0, None, 0)
    return out


def _prefixspan(stream: list[tuple[int, str]], min_support: int,
                max_gap: int, max_len: int) -> dict[tuple[str, ...], list[int]]:
    """Prefix-growth frequent-subsequence mining. Returns
    {pattern: [start bucket indices]}."""
    items = sorted({tok for _, tok in stream})
    freq: dict[tuple[str, ...], list[int]] = {}
    stack = [(t,) for t in items]
    while stack:
        pat = stack.pop()
        occ = _find_occurrences(stream, pat, max_gap)
        if len(occ) >= min_support:
            freq[pat] = occ
            if len(pat) < max_len:
                for t in items:      # pushed sorted; pop takes last
                    stack.append(pat + (t,))
    return freq


def _cluster_hours(hours: list[float], tol: float = 1.5) -> list[list[float]]:
    """Greedy 1-D clustering, sorted order, deterministic."""
    clusters: list[list[float]] = []
    for h in sorted(hours):
        if clusters and h - clusters[-1][-1] <= tol:
            clusters[-1].append(h)
        else:
            clusters.append([h])
    return clusters


class RoutineMiner:
    """Nightly job: timeline → mined routine edges in the world model."""

    def __init__(self, timeline, world, *, min_support: int = 3,
                 bucket_minutes: int = 15, max_len: int = 4,
                 max_gap_buckets: int = 4):
        self.timeline = timeline
        self.world = world
        self.min_support = max(1, int(min_support))
        self.bucket = int(bucket_minutes) * 60
        self.max_len = max(1, int(max_len))
        self.max_gap = max(1, int(max_gap_buckets))

    # ── 1. discretize ─────────────────────────────────────────────
    def token_stream(self, days_back: float = 30,
                     now: float | None = None) -> list[tuple[int, str]]:
        """[(bucket_index, activity_token)] — one token per 15-min bucket,
        the most frequent non-stopword from the bucket's events
        (type + payload words); ties break alphabetically."""
        now = now if now is not None else time.time()
        events = self.timeline.query(
            ts_from=now - days_back * 86400, ts_to=now, limit=100000)
        buckets: dict[int, Counter] = {}
        for e in events:
            bkt = int(e["ts_utc"] // self.bucket)
            blob = f"{e.get('type', '')} " + json.dumps(
                e.get("payload", {}), ensure_ascii=False)
            for w in _WORD_RE.findall(blob.lower()):
                if w not in _STOPWORDS:
                    buckets.setdefault(bkt, Counter())[w] += 1
        stream = []
        for bkt in sorted(buckets):
            counts = buckets[bkt]
            if counts:
                top = counts.most_common()
                topn = max(c for _, c in top)
                token = sorted(w for w, c in top if c == topn)[0]
                stream.append((bkt, token))
        return stream

    # ── 2–3. mine + cluster ────────────────────────────────────────
    def mine(self, days_back: float = 30,
             now: float | None = None) -> list[dict]:
        """Frequent patterns → routines [{activity, hour, support,
        confidence, pattern}]."""
        stream = self.token_stream(days_back, now)
        if not stream:
            return []
        freq = _prefixspan(stream, self.min_support, self.max_gap,
                           self.max_len)
        routines = []
        for pat, occ in freq.items():
            hours = [(b * self.bucket) % 86400 / 3600.0 for b in occ]
            for cluster in _cluster_hours(hours):
                if len(cluster) < self.min_support:
                    continue
                hour = round(sum(cluster) / len(cluster), 1)
                routines.append({
                    "activity": " ".join(pat),
                    "hour": hour,
                    "support": len(cluster),
                    "confidence": round(len(cluster) / max(len(occ), 1), 2),
                    "pattern": list(pat),
                })
        # deterministic ranking: support desc, then hour, then activity
        routines.sort(key=lambda r: (-r["support"], r["hour"], r["activity"]))
        # dedupe overlapping patterns keeping the strongest per activity+hour
        seen: dict[tuple[str, float], dict] = {}
        for r in routines:
            key = (r["activity"], round(r["hour"]))
            if key not in seen or r["support"] > seen[key]["support"]:
                seen[key] = r
        return sorted(seen.values(),
                      key=lambda r: (-r["support"], r["hour"], r["activity"]))

    # ── 4. write back ──────────────────────────────────────────────
    def run(self, days_back: float = 30,
            now: float | None = None) -> list[dict]:
        """Mine and persist. Returns the routines written."""
        routines = self.mine(days_back, now)
        for r in routines:
            self.world.upsert_mined_routine(
                r["activity"], r["hour"], r["confidence"], r["support"],
                ts=now)
        return routines


def mine_nightly(timeline_db: str | Path, world_dir: str | Path, **kw) -> list[dict]:
    """Entrypoint for the nightly cron: open timeline + world, mine, write."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from perception import PerceptionTimeline  # noqa: E402
    from world_model import WorldModel  # noqa: E402
    tl = PerceptionTimeline(timeline_db)
    world = WorldModel(world_dir)
    return RoutineMiner(tl, world, **kw).run()
