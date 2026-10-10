"""Calibrate the confidence thresholds against real score distributions.

The thresholds in retrieve.py (_SEM_LO/_SEM_HI) were guesses. This measures what
correct and incorrect matches actually score, so the confidence gate is set from data.

    python memory/_calibrate.py      # from backend/
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.benchmark import (  # noqa: E402
    ADVERSARIAL_CASES,
    NEGATIVE_CASES,
    REALISTIC_CASES,
)
from memory.embeddings import Embedder  # noqa: E402


def main() -> int:
    emb = Embedder()
    print(f"embedder: {emb.backend_name} ({emb.dim}d)\n")

    all_cases = list(REALISTIC_CASES) + list(ADVERSARIAL_CASES)
    corpus = []
    for _, correct, decoys in all_cases:
        corpus.append(correct)
        corpus.extend(decoys)

    pos, neg = [], []
    for query, correct, decoys in all_cases:
        qv = emb.embed_query(query)
        pos.append(float(qv @ emb.embed(correct)))
        for d in decoys:
            neg.append(float(qv @ emb.embed(d)))
        # cross-case documents are the realistic hard negatives
        for other in corpus:
            if other != correct and other not in decoys:
                neg.append(float(qv @ emb.embed(other)))

    for q in NEGATIVE_CASES:
        qv = emb.embed_query(q)
        for doc in corpus:
            neg.append(float(qv @ emb.embed(doc)))

    def stats(name, xs):
        xs = sorted(xs)
        print(f"{name:24s} n={len(xs):4d}  min={xs[0]:.3f}  p25={xs[len(xs)//4]:.3f}  "
              f"median={statistics.median(xs):.3f}  p75={xs[3*len(xs)//4]:.3f}  max={xs[-1]:.3f}")

    print("cosine similarity distributions")
    stats("CORRECT (should be high)", pos)
    stats("INCORRECT (should be low)", neg)
    print()

    # Find the threshold that best separates them.
    best, best_gap = None, -1
    for t in [x / 100 for x in range(30, 80)]:
        tpr = sum(1 for p in pos if p >= t) / len(pos)
        fpr = sum(1 for n in neg if n >= t) / len(neg)
        gap = tpr - fpr
        if gap > best_gap:
            best_gap, best = gap, (t, tpr, fpr)
    t, tpr, fpr = best
    print(f"best single threshold: {t:.2f}  (correct kept {tpr:.0%}, incorrect kept {fpr:.0%}, gap {best_gap:.2f})")
    print()

    # Suggest a mapping: LO = the point below which almost nothing correct lives,
    # HI = a value comfortably inside the correct distribution.
    pos_sorted = sorted(pos)
    neg_sorted = sorted(neg)
    lo = neg_sorted[int(len(neg_sorted) * 0.90)]     # 90th pct of wrong answers
    hi = pos_sorted[int(len(pos_sorted) * 0.50)]     # median of right answers
    if hi <= lo:
        hi = lo + 0.10
    print(f"suggested _SEM_LO = {lo:.2f}   (90th pct of incorrect)")
    print(f"suggested _SEM_HI = {hi:.2f}   (median of correct)")
    print(f"    -> currently 0.40 / 0.78")
    return 0


if __name__ == "__main__":
    sys.exit(main())
