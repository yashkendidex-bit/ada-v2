"""Retrieval quality benchmark for ADA's memory.

This is Part 8 of `ADA_MEMORY_ARCHITECTURE.md` made executable: the probe set that
tells you whether retrieval works, rather than whether it runs.

    python -m memory.benchmark              # from backend/
    python -m memory.benchmark --verbose
    python -m memory.benchmark --no-rerank  # measure what the reranker contributes

It measures the **whole pipeline** — hybrid channels, fusion, rerank, scoring, gate —
not the raw embedder, because the embedder is not the product. Stage-by-stage, on the
realistic suite:

    bi-encoder only            ~20%
    + hybrid fusion (BM25/graph)
    + cross-encoder rerank
    + the scoring fixes        <- what this reports

Three suites, because they answer different questions:

* **REALISTIC** — how people actually query a memory: partial vocabulary overlap with
  the right memory, decoys about unrelated topics. This is the number that matters,
  and the one the run is gated on.
* **ADVERSARIAL** — the query shares *no* content words with the correct memory while
  every decoy *does* share one. Lexical matching is pulled toward the wrong answer, so
  only real semantics can win. **Reported, not gated.** Measured honestly: a
  33M-parameter model cannot solve these reliably, and pretending otherwise would be
  the same mistake as the silent MD5 fallback this replaced. Watch the trend here, not
  the absolute number.
* **NEGATIVE** — queries with no answer in the store. Must come back NOT confident.
  A leak here is the worst failure the system has, because it means she confabulates.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Sequence, Tuple

Case = Tuple[str, str, Sequence[str]]

# How people actually ask. Partial overlap, unrelated decoys.
REALISTIC_CASES: list[Case] = [
    ("what units should I use for drawings?",
     "Sir prefers metric units on every drawing",
     ["the drawing export failed last night",
      "the gearbox tolerance still needs checking"]),
    ("how long is the drone arm?",
     "the drone arm measures 160 millimetres",
     ["the drone flew for eleven minutes",
      "Sir asked about arm material costs"]),
    ("what material did we settle on?",
     "final decision: the drone arm is titanium Ti6Al4V",
     ["the titanium supplier is closed on Mondays",
      "Sir dislikes brass for structural work"]),
    ("was the bracket strong enough?",
     "the bracket analysis returned a safety factor of 2.1, so it passes",
     ["the bracket is 80 by 40 millimetres",
      "the bracket needed a 2 millimetre fillet"]),
    ("why did the part crack?",
     "the aluminium bracket cracked at the motor mount under load",
     ["the part numbers are in the spreadsheet",
      "I ordered three of that part"]),
    ("M8 bolt length?",
     "the M8 bolts are 45 millimetres long",
     ["the M8 nuts arrived late",
      "I ordered M6 washers by mistake"]),
    ("what does he dislike?",
     "Sir dislikes brass for structural work",
     ["Sir prefers metric units on every drawing",
      "Sir likes the titanium finish"]),
    ("what colour for the housing?",
     "Sir wants the enclosure finished in racing red",
     ["the colour profile is sRGB",
      "the housing is 3 millimetres thick"]),
    ("Ti6Al4V?",
     "final choice: titanium Ti6Al4V for the arm",
     ["aluminium 6061 was the earlier idea",
      "the titanium supplier is closed until Monday"]),
]

# Hostile by construction. Diagnostic only — see the module docstring.
ADVERSARIAL_CASES: list[Case] = [
    ("what snapped on the rig?",
     "the bracket fractured at the motor mount",
     ["the snap-fit clip went missing",
      "I snapped a photo of the rig yesterday"]),
    ("how heavy is the assembly?",
     "the finished unit weighs 142 grams",
     ["heavy rain is forecast for Tuesday",
      "Sir found the review heavy going"]),
    ("what holds the plate on?",
     "four M5 cap screws secure the mounting plate",
     ["the plate is 3 millimetres thick",
      "Sir wants the plate anodised"]),
    ("was the part too weak?",
     "the analysis returned a safety factor of 1.2, under the 2.0 requirement",
     ["the weak point is the fillet radius",
      "I felt weak after that long session"]),
]

NEGATIVE_CASES = [
    "what is the capital of Peru?",
    "who won the 1998 world cup?",
    "explain quantum chromodynamics",
    "what is my blood type?",
    "how do I reset the router?",
]

# Regression floor, NOT a target.
#
# Measured on 2026-09-15, same suite, same corpus:
#
#   backend                        recall@1   MRR     negative gate
#   mistral-embed (1024-d, API)      67%      0.781    5/5 gated
#   bge-small-en-v1.5 (384-d, local) 67%      0.744    5/5 gated
#
# So Mistral does NOT lift recall@1. It improves MRR by ~0.04, and the reason is a
# single case: "what material did we settle on?" puts the right memory at rank 2
# instead of rank 6. The three misses are identical under both models, and they share a
# cause — the query's own words ("material", "bracket", "housing") appear in the decoys
# while the correct memory uses different ones ("titanium", "safety factor",
# "enclosure"). That is a vocabulary gap, and a bigger embedding model does not close it
# at this corpus size.
#
# The real, measured reason to prefer Mistral is not retrieval quality: it is that it
# costs no RAM, which matters on an 8 GB machine running the rest of the app. Judge the
# switch on that, not on this number.
#
# The ceiling is therefore a property of the task as much as of the model. This gate
# exists to catch *regressions*, not to certify quality.
REALISTIC_FLOOR = 0.65
# Surfaced in the output so nobody mistakes 67% for a pipeline failure or a good number.
BASELINE_NOTE = ("recall@1 is task-bound at ~67% for both measured backends — "
                 "the 3 misses are vocabulary gaps, not model quality")


def _seed(store, embedder, texts: Sequence[str]) -> None:
    now = time.time()
    for i, text in enumerate(texts):
        store.execute(
            "INSERT INTO recall_entries(kind, content, ts, ts_event, importance,"
            " embedding, embed_backend, stability, difficulty, last_access)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("conversation", text, now - i * 600, now - i * 600, 5,
             embedder.embed(text).tobytes(), embedder.backend_name, 1.0, 5.0, now))
    store.execute("INSERT INTO recall_fts(recall_fts) VALUES('rebuild')")


def _run_suite(retriever, cases: Sequence[Case], k: int = 5):
    hits, mrr, detail = 0, 0.0, []
    for query, correct, _ in cases:
        res = retriever.recall(query, k=k)
        ranked = [h.content for h in res.hits]
        rank = ranked.index(correct) if correct in ranked else len(ranked)
        hits += rank == 0
        mrr += 1.0 / (rank + 1)
        detail.append((query, rank, ranked[0] if ranked else "(none)"))
    n = len(cases)
    return hits / n, mrr / n, detail


def main() -> int:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rerank", action="store_true",
                    help="force the cross-encoder on (off by default — it measures as harmful)")
    ap.add_argument("--no-rerank", action="store_true",
                    help="force the cross-encoder off (this is already the default)")
    args = ap.parse_args()

    from memory.embeddings import Embedder
    from memory.retrieve import Retriever
    from memory.store import Store

    embedder = Embedder()
    health = embedder.health()
    print(f"embedder  : {health['backend']} (semantic={health['semantic']}, dim={health['dim']})")
    if health["model"]:
        print(f"model     : {health['model']}")
    if not health["semantic"]:
        print("WARNING   : lexical fallback active — semantic recall will be poor.")
    print()

    tmp = Path(tempfile.mkdtemp()) / "bench.db"
    store = Store(tmp)

    corpus: list[str] = []
    for cases in (REALISTIC_CASES, ADVERSARIAL_CASES):
        for _, correct, decoys in cases:
            corpus.append(correct)
            corpus.extend(decoys)
    # NEGATIVE_CASES are deliberately NOT seeded. Seeding them made each negative
    # query an exact match for itself and reported a confidence leak that was an
    # artefact of the test, not of the retriever.
    _seed(store, embedder, corpus)

    # None means "use the Retriever's own default", so this harness cannot silently
    # disagree with what ships.
    use_rerank = True if args.rerank else (False if args.no_rerank else None)
    retriever = Retriever(store, embedder, use_rerank=use_rerank)
    print(f"reranker  : {'on' if retriever.use_rerank else 'off'}"
          f"{'' if use_rerank is not None else ' (default)'}")
    print()

    ok = True
    r1, mrr, detail = _run_suite(retriever, REALISTIC_CASES)
    verdict = "PASS" if r1 >= REALISTIC_FLOOR else "FAIL"
    ok &= r1 >= REALISTIC_FLOOR
    print(f"{'REALISTIC (gated)':30s} recall@1 {r1:>5.0%}  MRR {mrr:.3f}  "
          f"[{verdict}, floor {REALISTIC_FLOOR:.0%}]")
    if args.verbose:
        for q, rank, top in detail:
            print(f"      {'ok ' if rank == 0 else f'#{rank+1}':4s} {q!r} -> {top[:52]}")

    r1a, mrra, detaila = _run_suite(retriever, ADVERSARIAL_CASES)
    print(f"{'ADVERSARIAL (diagnostic)':30s} recall@1 {r1a:>5.0%}  MRR {mrra:.3f}  "
          f"[reported, not gated]")
    if args.verbose:
        for q, rank, top in detaila:
            print(f"      {'ok ' if rank == 0 else f'#{rank+1}':4s} {q!r} -> {top[:52]}")

    leaks = [(q, res.confidence)
             for q in NEGATIVE_CASES
             if (res := retriever.recall(q, k=3)).is_confident]
    gate_ok = not leaks
    ok &= gate_ok
    print(f"{'NEGATIVE (gated)':30s} {len(NEGATIVE_CASES)-len(leaks)}/{len(NEGATIVE_CASES)} gated"
          f"  [{'PASS' if gate_ok else 'FAIL'}]")
    for q, c in leaks:
        print(f"      LEAK  {q!r} confidence={c:.3f}")

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    if r1 < 0.85:
        print(f"note      : {BASELINE_NOTE}")
    store.close()
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
