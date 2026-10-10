"""Smoke test for the retriever. Seeds a throwaway store and queries it.

    python memory/_smoke.py     # from backend/
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.embeddings import Embedder, to_blob          # noqa: E402
from memory.retrieve import Retriever                    # noqa: E402
from memory.store import Store                           # noqa: E402

MEMORIES = [
    ("Sir prefers metric units on every drawing", 8),
    ("the drone bracket fractured at the motor mount under load", 9),
    ("the drone arm measures 160 millimetres", 6),
    ("final decision: the drone arm is titanium Ti6Al4V", 9),
    ("Sir dislikes brass for structural work", 5),
    ("the M8 bolts are 45 millimetres long", 4),
    ("we watched a film about submarines on Friday", 2),
    ("the coffee machine in the kitchen is broken", 2),
]

QUERIES = [
    ("what units should you use for drawings?", 0),
    ("what snapped on the rig?", 1),
    ("how long is the drone arm?", 2),
    ("which metal did we settle on?", 3),
    ("what does he dislike?", 4),
    ("M8 bolt length?", 5),
    ("what did we do on Friday evening?", 6),
]


def main() -> int:
    tmp = Path(tempfile.mkdtemp()) / "smoke.db"
    store = Store(tmp)
    emb = Embedder()
    print(f"embedder: {emb.backend_name} (semantic={emb.is_semantic})")
    print(f"store   : {tmp}")
    print()

    now = time.time()
    for i, (text, imp) in enumerate(MEMORIES):
        store.execute(
            "INSERT INTO recall_entries(kind, content, ts, ts_event, importance,"
            " embedding, embed_backend, stability, difficulty, last_access)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("conversation", text, now - i * 3600, now - i * 3600, imp,
             to_blob(emb.embed(text)), emb.backend_name, 1.0, 5.0, now))
    store.execute("INSERT INTO recall_fts(recall_fts) VALUES('rebuild')")

    r = Retriever(store, emb)
    passed = 0
    for query, expected_idx in QUERIES:
        res = r.recall(query, k=3)
        top = res.hits[0] if res.hits else None
        ok = top is not None and top.content == MEMORIES[expected_idx][0]
        passed += ok
        mark = "ok " if ok else "MISS"
        print(f"{mark} {query!r}")
        print(f"      -> {top.content[:64] if top else '(nothing)'}")
        if not ok:
            print(f"      expected: {MEMORIES[expected_idx][0][:64]}")
        print(f"      confidence={res.confidence:.3f} confident={res.is_confident} "
              f"channels={res.channels}")
        if top:
            print(f"      parts={ {k: v for k, v in top.parts.items() if k != 'rrf'} }")
    print()
    print(f"recall@1: {passed}/{len(QUERIES)}")

    print()
    print("--- confidence gate: unrelated queries must NOT be confident ---")
    for q in ("what is the capital of Peru?",
              "who won the 1998 world cup?",
              "explain quantum chromodynamics"):
        res = r.recall(q, k=3)
        flag = "LEAK" if res.is_confident else "gated"
        print(f"    [{flag}] {q!r} confidence={res.confidence:.3f}"
              + (f"  top={res.hits[0].content[:44]!r}" if res.hits else "  (no hits)"))

    print()
    print("--- context block (what actually reaches the prompt) ---")
    res = r.recall("what material is the drone arm?", k=3)
    print(res.context_block())
    return 0


if __name__ == "__main__":
    sys.exit(main())
