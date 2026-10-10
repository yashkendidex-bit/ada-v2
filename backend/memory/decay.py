"""Decay, interference, and the fidelity ladder.

The biology this implements (ADA_MEMORY_ARCHITECTURE.md §2.8, §2.9):

* Forgetting is a **power law, not an exponential** — it decelerates, which is why
  old memories resist loss. Ebbinghaus's own fit was logarithmic, and across 200+
  retention datasets power/log forms consistently beat exponential ones.
* **Retrieval strengthens.** A successful recall does not just reset the clock, it
  slows the clock: stability grows, so the next interval is longer. That is the
  "sawtooth that flattens" in the spacing literature, and it is why two spaced
  reviews across 30 days hold ~56% versus ~0.7% unreviewed.
* **Two independent loss processes.** Decay (time) and interference (competition).
  Retroactive interference is stronger than proactive — new learning disrupts old
  memories more than the reverse — hence the 0.6 / 0.4 weights.
* **Degrade, don't delete.** A memory steps down a fidelity ladder to a tombstone
  that records only that it existed. Deleting is reserved for an explicit request.

The FSRS constants below are the published defaults, not invented:

    R = (1 + FACTOR * elapsed_days / stability) ** DECAY
        DECAY  = -0.5
        FACTOR = 19/81  ~= 0.2346      (from 0.9 ** (1/DECAY) - 1)
"""

from __future__ import annotations

import math
from typing import Iterable

# ── FSRS / DSR constants ────────────────────────────────────────────────────
FSRS_DECAY = -0.5
FSRS_FACTOR = 19.0 / 81.0          # ≈ 0.2346
DEFAULT_TARGET_RETENTION = 0.9     # recall is "due" when R falls to this

# λ from the Microsoft human-inspired memory study: half-life ≈ 29 days. They note
# this is deliberately *longer* than human biology, because the right horizon
# follows the rhythm of the domain rather than the circadian cycle. A design
# assistant works on projects over months, so 29 days is the right starting point.
DECAY_LAMBDA = 0.001               # per hour
DECAY_HALF_LIFE_DAYS = math.log(2) / DECAY_LAMBDA / 24.0   # ≈ 28.9

# Interference weights. Retroactive > proactive.
W_RETROACTIVE = 0.6
W_PROACTIVE = 0.4
INTERFERENCE_THRESHOLD = 0.55      # cosine above which two memories compete

# Fidelity ladder thresholds. Age is in days, score is 0..1.
LADDER = [
    # (min_age_days, max_score, target_fidelity)
    (120.0, 0.15, 3),   # old and low value -> gist
    (240.0, 0.10, 5),   # very old and near-worthless -> tombstone
    (60.0, 0.10, 2),    # two months old and low value -> summary
]


def retrievability(stability_days: float, elapsed_days: float) -> float:
    """Probability the memory is still recallable. FSRS power-law curve."""
    if stability_days is None or stability_days <= 0:
        stability_days = 1.0
    elapsed = max(0.0, float(elapsed_days or 0.0))
    return float((1.0 + FSRS_FACTOR * elapsed / stability_days) ** FSRS_DECAY)


def next_interval_days(stability_days: float,
                       target_retention: float = DEFAULT_TARGET_RETENTION) -> float:
    """When R will fall to `target_retention`. The inverse of `retrievability`."""
    if stability_days is None or stability_days <= 0:
        stability_days = 1.0
    target = min(max(target_retention, 0.01), 0.999)
    return float(stability_days / FSRS_FACTOR * (target ** (1.0 / FSRS_DECAY) - 1.0))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def update_stability(stability_days: float, difficulty: float,
                     retrievability_now: float, grade: str = "good") -> float:
    """Stability after a review. Retrieval is a write, not a read.

    Grade maps to the four FSRS outcomes: again / hard / good / easy. The growth
    factors are the published shape (S' = S * (1 + a * (11 - D) * S^-b * (e^(c(1-R)) - 1)))
    with conservative constants; they only need to be directionally right, because
    the exact values are meant to be fitted per-user from review history.
    """
    s = max(0.1, float(stability_days or 1.0))
    d = _clamp(float(difficulty or 5.0), 1.0, 10.0)
    r = _clamp(float(retrievability_now or 0.9), 0.01, 0.999)

    growth = math.exp(1.2 * (1.0 - r)) - 1.0          # more gain when recall was hard
    base = 1.0 + 1.6 * (11.0 - d) * (s ** -0.2) * growth
    factor = {"again": 0.35, "hard": 0.85, "good": 1.0, "easy": 1.35}.get(grade, 1.0)
    return float(_clamp(s * base * factor, 0.1, 3650.0))


def update_difficulty(difficulty: float, grade: str) -> float:
    """Difficulty drifts with performance, with mean reversion to 5."""
    d = _clamp(float(difficulty or 5.0), 1.0, 10.0)
    delta = {"again": 0.9, "hard": 0.35, "good": -0.15, "easy": -0.5}.get(grade, 0.0)
    return float(_clamp(d + delta + 0.03 * (5.0 - d), 1.0, 10.0))


def decayed_importance(importance: float, age_hours: float,
                       lam: float = DECAY_LAMBDA) -> float:
    """Exponential decay of an unconsolidated memory's importance."""
    return float(max(0.0, float(importance or 0.0)) * math.exp(-lam * max(0.0, age_hours)))


def interference_score(similarities: Iterable[float], age_hours: float,
                       mean_age_hours: float) -> float:
    """How much a memory is being crowded out by its neighbours.

    New learning disrupts old memory more than the reverse, so a memory that is
    *younger* than its competitors takes the retroactive hit and an older one takes
    the proactive hit.
    """
    sims = [s for s in similarities if s is not None and s > INTERFERENCE_THRESHOLD]
    if not sims:
        return 0.0
    retroactive = age_hours < mean_age_hours
    w = W_RETROACTIVE if retroactive else W_PROACTIVE
    return float(w * sum(sims) / len(sims))


def retention_score(importance: float, stability_days: float, age_days: float,
                    access_count: int = 0, pinned: bool = False) -> float:
    """A single 0..1 number for "how much does this memory deserve to stay intact".

    Pinned memories score 1.0 by definition — they are exempt from every pass.
    """
    if pinned:
        return 1.0
    r = retrievability(stability_days, age_days)
    imp = _clamp(float(importance or 1.0) / 10.0, 0.0, 1.0)
    # Rehearsal is evidence of value, with diminishing returns.
    rehearse = _clamp(math.log1p(max(0, access_count)) / math.log1p(50.0), 0.0, 1.0)
    return float(_clamp(0.45 * r + 0.35 * imp + 0.20 * rehearse, 0.0, 1.0))


def next_fidelity(current: int, age_days: float, score: float,
                  pinned: bool = False) -> int:
    """Step down the ladder, or stay put.

    Returns the *new* fidelity level. Monotonic: a memory never climbs back up the
    ladder, because a gist cannot be un-summarised — that is the whole point of
    keeping L0 in the first place.
    """
    if pinned:
        return current
    for min_age, max_score, target in LADDER:
        if age_days >= min_age and score <= max_score:
            return max(current, target)
    return current


def maturity(age_hours: float, half_life_h: float = 168.0, k: float = 48.0) -> float:
    """Engram maturation sigmoid. Silent (≈0.03) at birth, 0.5 at one week, >0.9 at two.

    Caveat carried over from the source study: this mechanism was never ablated, so
    treat it as a design rationale rather than a measured win. It is off unless
    explicitly enabled.
    """
    return float(1.0 / (1.0 + math.exp(-(max(0.0, age_hours) - half_life_h) / k)))
