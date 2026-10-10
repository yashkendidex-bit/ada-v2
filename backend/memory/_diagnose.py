"""Diagnose one query: dump every score component for every candidate.

    python memory/_diagnose.py "what colour for the housing?"
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.benchmark import ADVERSARIAL_CASES, REALISTIC_CASES, _seed  # noqa: E402
from memory.embeddings import Embedder  # noqa: E402
from memory.retrieve import Retriever  # noqa: E402
from memory.store import Store  # noqa: E402


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "what colour for the housing?"
    correct = None
    for cases in (REALISTIC_CASES, ADVERSARIAL_CASES):
        for q, c, _ in cases:
            if q == query:
                correct = c

    emb = Embedder()
    store = Store(Path(tempfile.mkdtemp()) / "diag.db")
    corpus = []
    for cases in (REALISTIC_CASES, ADVERSARIAL_CASES):
        for _, c, decoys in cases:
            corpus.append(c)
            corpus.extend(decoys)
    _seed(store, emb, corpus)

    r = Retriever(store, emb)
    print(f"query   : {query!r}")
    print(f"correct : {correct!r}\n")

    res = r.recall(query, k=8)
    print(f"{'rank':>4}  {'score':>6} {'rel':>6} {'sem':>6} {'cal_s':>6} {'lexIDF':>7} "
          f"{'rrf_n':>6} {'rec':>6} {'imp':>6}  content")
    for i, h in enumerate(res.hits):
        p = h.parts
        mark = "*" if h.content == correct else " "
        print(f"{i+1:>4}{mark} {h.score:>6.3f} {p.get('relevance', 0):>6.3f} "
              f"{(p.get('sem_cos') or 0):>6.3f} {p.get('relevance', 0):>6.3f} "
              f"{p.get('lex_overlap', 0):>7.3f} "
              f"{p.get('rrf', 0):>6.3f} {p.get('recency', 0):>6.3f} "
              f"{p.get('importance', 0):>6.3f}  {h.content[:46]}")
    print()
    print(f"confidence={res.confidence:.3f} confident={res.is_confident} channels={res.channels}")
    if correct and correct not in [h.content for h in res.hits]:
        print("\n!! the correct memory is not in the candidate set at all — a channel problem")
    return 0


if __name__ == "__main__":
    sys.exit(main())
