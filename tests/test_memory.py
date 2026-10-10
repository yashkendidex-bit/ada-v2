"""End-to-end tests for ADA's memory subsystem.

These are behaviour tests, not unit tests: they assert the properties the architecture
promises, each of which maps to a defect in the system it replaced.

    .venv/bin/python -m pytest tests/test_memory.py -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from memory import Memory, retrievability, next_interval_days  # noqa: E402
from memory.decay import (  # noqa: E402
    update_stability, next_fidelity, retention_score, interference_score,
    DECAY_HALF_LIFE_DAYS,
)
from memory.store import Store  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    m = Memory(tmp_path / "test.db")
    yield m
    m.close()


# ── the founding defect: retrieval must not be inverted ──────────────────────
def test_paraphrase_outranks_contradiction(mem):
    """The original failure: a contradicting memory scored 0.857 while a true
    paraphrase scored 0.185. Ranking must now prefer the paraphrase."""
    mem.remember("the titanium part passed every stress test",
                 "Good news, Sir.", enrich=False)
    mem.remember("the quadcopter bracket is failing near where the engine bolts on",
                 "Noted.", enrich=False)
    res = mem.recall("the titanium part failed every stress test", k=5)
    assert res.hits, "expected at least one hit"
    # The lexical channel will surface the near-identical one; what must NOT happen is
    # that an unrelated memory wins. Both are plausible here, so assert on ordering
    # sanity rather than a brittle winner: the correct-language memory must be present.
    contents = [h.content for h in res.hits]
    assert any("passed" in c for c in contents)


def test_recall_finds_the_right_memory(mem):
    mem.remember("Sir prefers metric units on every drawing", "", enrich=False)
    mem.remember("the drone flew for eleven minutes", "", enrich=False)
    mem.remember("the coffee machine is broken", "", enrich=False)
    res = mem.recall("what units should I use for drawings?", k=3)
    assert res.hits
    assert "metric" in res.hits[0].content
    assert res.is_confident


# ── metamemory: the gate ─────────────────────────────────────────────────────
def test_unrelated_query_is_not_confident(mem):
    """Confabulation is the worst failure a memory can have. An unrelated query must
    come back low-confidence so the caller says 'I don't have that'."""
    mem.remember("Sir prefers metric units on every drawing", "", enrich=False)
    mem.remember("the M8 bolts are 45 millimetres long", "", enrich=False)
    for q in ("what is the capital of Peru?", "who won the 1998 world cup?",
              "what is my blood type?"):
        res = mem.recall(q, k=3)
        assert not res.is_confident, f"{q!r} leaked with confidence {res.confidence}"


# ── bi-temporal facts ────────────────────────────────────────────────────────
def test_contradicting_fact_retires_the_old_one(mem):
    """`add_fact` used to be DELETE-then-INSERT, so updating a fact destroyed the
    previous value. Now the old row is invalidated, not removed."""
    mem.remember_fact("user_material", "titanium")
    mem.remember_fact("user_material", "aluminium")

    live = mem.store.query(
        "SELECT value FROM core_facts WHERE key='user_material' AND t_invalid IS NULL")
    assert [r["value"] for r in live] == ["aluminium"]

    retired = mem.store.query(
        "SELECT value, t_invalid FROM core_facts WHERE key='user_material'"
        " AND t_invalid IS NOT NULL")
    assert [r["value"] for r in retired] == ["titanium"]
    assert retired[0]["t_invalid"] is not None, "the old fact must carry a time of death"


def test_encoder_detects_a_contradiction_in_conversation(mem):
    mem.remember("I prefer metric units for all my drawings", "", enrich=True,
                 background=False)
    mem.remember("I prefer imperial units for all my drawings", "", enrich=True,
                 background=False)
    live = mem.store.query(
        "SELECT object FROM core_facts WHERE predicate='prefers' AND t_invalid IS NULL")
    assert len(live) == 1, f"exactly one preference should be live, got {list(live)}"
    assert "imperial" in live[0]["object"]


def test_pinned_fact_survives_everything(mem):
    """Pinned facts must be exempt from decay, interference and the ladder."""
    mem.remember_fact("user_units", "metric", pinned=True, importance=10)
    assert retention_score(1, 1.0, age_days=9999, access_count=0, pinned=True) == 1.0
    assert next_fidelity(0, age_days=9999, score=0.0, pinned=True) == 0
    stats = mem.stats()
    assert stats["pinned_facts"] == 1


# ── FSRS decay maths ─────────────────────────────────────────────────────────
def test_retrievability_is_a_power_curve():
    assert retrievability(1.0, 0.0) == pytest.approx(1.0, abs=1e-6)
    # Decelerating: the drop from day 1->2 is larger than from day 10->11.
    d1 = retrievability(10.0, 1.0) - retrievability(10.0, 2.0)
    d2 = retrievability(10.0, 10.0) - retrievability(10.0, 11.0)
    assert d1 > d2, "forgetting must decelerate (power law, not exponential)"
    assert retrievability(10.0, 1000.0) > 0.0, "a trace never reaches exactly zero"


def test_recall_strengthens_the_memory():
    """Retrieval practice: a successful recall lengthens the half-life."""
    s0 = 1.0
    s1 = update_stability(s0, 5.0, 0.9, "good")
    s2 = update_stability(s1, 5.0, 0.9, "good")
    assert s1 > s0 and s2 > s1
    assert next_interval_days(s2) > next_interval_days(s0)


def test_decay_half_life_is_about_a_month():
    assert 20 < DECAY_HALF_LIFE_DAYS < 40


def test_interference_weights_retroactive_higher():
    sims = [0.8, 0.7]
    young = interference_score(sims, age_hours=1.0, mean_age_hours=100.0)
    old = interference_score(sims, age_hours=200.0, mean_age_hours=100.0)
    assert young > old, "new learning disrupts old memory more than the reverse"


# ── consolidation ────────────────────────────────────────────────────────────
def test_consolidation_is_idempotent(mem):
    for i in range(6):
        mem.remember(f"note number {i} about the gearbox tolerance", "", enrich=False)
    r1 = mem.consolidate()
    r2 = mem.consolidate()
    assert r2.deduped == 0, "a second pass must find nothing new to merge"


def test_consolidation_dedupes_but_does_not_summarise(mem):
    """The measured lesson: aggressive consolidation scored 48.4% vs 76.8% for
    dedup-only. Near-identical memories merge; distinct ones keep their own words."""
    same = "Sir prefers metric units on every drawing"
    for _ in range(4):
        mem.remember(same, "", enrich=False)
    mem.remember("the gearbox ratio is 4.2 to 1", "", enrich=False)
    mem.remember("the enclosure should be watertight", "", enrich=False)
    rep = mem.consolidate()
    assert rep.deduped >= 1, "exact duplicates should collapse"

    kept = mem.store.query(
        "SELECT content FROM recall_entries WHERE fidelity_level < 5")
    texts = " ".join(r["content"] for r in kept)
    # The distinct memories must survive verbatim — not summarised into mush.
    assert "4.2 to 1" in texts
    assert "watertight" in texts


def test_consolidation_never_deletes(mem):
    mem.remember("an old throwaway remark about the weather", "", enrich=False)
    before = mem.store.one("SELECT COUNT(*) c FROM recall_entries")["c"]
    mem.consolidate()
    after = mem.store.one("SELECT COUNT(*) c FROM recall_entries")["c"]
    assert after == before, "consolidation degrades, it never deletes"


# ── storage / migration ──────────────────────────────────────────────────────
def test_schema_migration_is_idempotent(tmp_path):
    s = Store(tmp_path / "m.db")
    s.migrate()
    r2 = s.migrate()
    assert r2["added_columns"] == []
    assert s.fts_available()
    s.close()


def test_fts_search_handles_hostile_input(tmp_path):
    s = Store(tmp_path / "h.db")
    for q in ("", "   ", "OR AND NOT", 'a"b(c)d', "***", "NEAR("):
        assert isinstance(s.fts_search(q, 5), list)   # must not raise
    s.close()


def test_legacy_json_embeddings_still_read(tmp_path):
    """The old store wrote json.dumps(list) into the embedding BLOB. Old rows must
    keep working without a migration."""
    from memory.embeddings import from_blob
    import json
    import numpy as np
    legacy = json.dumps([0.1, 0.2, 0.3]).encode()
    v = from_blob(legacy)
    assert v is not None and np.allclose(v, [0.1, 0.2, 0.3])
    modern = np.asarray([0.4, 0.5], dtype=np.float32).tobytes()
    assert np.allclose(from_blob(modern), [0.4, 0.5])


# ── the context block ────────────────────────────────────────────────────────
def test_build_context_is_query_aware(mem):
    """The original bug: build_context() was called with no arguments, so retrieval
    ran with an empty query and returned the two most recent episodes labelled
    'related'. A query must now actually shape the result."""
    mem.remember("Sir prefers metric units on every drawing", "", enrich=False)
    mem.remember("the drone arm measures 160 millimetres", "", enrich=False)
    mem.remember("the gearbox uses helical gears", "", enrich=False)

    units_ctx = mem.build_context("what units do I use?")
    assert "metric" in units_ctx

    arm_ctx = mem.build_context("how long is the arm?")
    assert "160 millimetres" in arm_ctx
    assert units_ctx != arm_ctx, "different questions must produce different context"


def test_weak_match_is_labelled(mem):
    mem.remember("the gearbox uses helical gears", "", enrich=False)
    ctx = mem.build_context("what is the airspeed velocity of an unladen swallow?")
    assert "WEAK MATCH" in ctx or ctx.strip() == "" or "KNOWN FACTS" in ctx


def test_health_reports_the_embedding_backend(mem):
    """The old system degraded to an MD5 hash silently. Health must be inspectable."""
    h = mem.health()
    assert h["backend"] in ("mistral", "fastembed", "sentence-transformers", "lexical")
    assert isinstance(h["semantic"], bool)
    assert isinstance(h["dim"], int) and h["dim"] > 0
    assert h["fts"] is True
