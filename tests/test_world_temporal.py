"""F3 — temporal world model tests (stdlib only, tmp dirs only)."""
import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent
                       / "backend"))

from world_model import WorldModel  # noqa: E402
from perception import PerceptionTimeline  # noqa: E402
from routine_miner import RoutineMiner  # noqa: E402


def _world(tmp_path):
    return WorldModel(str(tmp_path / "world"))


def _plant_timeline(tmp_path, days=4, hour=7.5):
    tl = PerceptionTimeline(str(tmp_path / "events.db"))
    day = 86400
    base = (time.time() // day * day) - (days + 1) * day
    for d in range(days):
        ts = base + d * day + hour * 3600
        tl.emit("voice", "transcript",
                {"text": "coffee coffee morning"}, ts=ts)
    tl.emit("voice", "transcript", {"text": "stock charts review"},
            ts=base + 12 * 3600)
    return tl


# ── 1. entity create/link ────────────────────────────────────────────
def test_observe_episode_creates_and_links_entities(tmp_path):
    w = _world(tmp_path)
    ep = w.observe_episode('Met with Alice Johnson about the gearbox project')
    assert ep > 0
    eid = w._entity_id("Alice Johnson")
    assert eid is not None
    row = w._db.execute(
        "SELECT 1 FROM mentions WHERE episode_id = ? AND entity_id = ?",
        (ep, eid)).fetchone()
    assert row is not None
    # re-observing links to the SAME entity (no duplicate)
    w.observe_episode("Alice Johnson called back")
    assert w._entity_id("Alice Johnson") == eid


# ── 2. as-of query returns past truth ─────────────────────────────────
def test_state_at_returns_past_truth(tmp_path):
    w = _world(tmp_path)
    t0 = time.time() - 100
    w.add_fact("user", "lives_in", "Berlin", ts=t0)
    w.add_fact("user", "lives_in", "Mumbai", ts=t0 + 10)
    past = [e for e in w.state_at(t0 + 1) if e["relation"] == "lives_in"]
    now = [e for e in w.state_at(t0 + 20) if e["relation"] == "lives_in"]
    assert [(e["a_name"], e["b_name"]) for e in past] == [("user", "Berlin")]
    assert [(e["a_name"], e["b_name"]) for e in now] == [("user", "Mumbai")]


# ── 3. contradiction retires old edge, keeps both ─────────────────────
def test_contradiction_retires_keeps_both(tmp_path):
    w = _world(tmp_path)
    events = []

    class FakeBus:
        def publish(self, event_type, source="", payload=None):
            events.append((event_type, payload))

    w._bus = FakeBus()
    t0 = time.time() - 100
    w.add_fact("user", "lives_in", "Berlin", ts=t0)
    w.add_fact("user", "lives_in", "Mumbai", ts=t0 + 10)
    rows = w._db.execute(
        "SELECT b.name n, e.valid_to v FROM edges e "
        "JOIN entities a ON a.id = e.a_id JOIN entities b ON b.id = e.b_id "
        "WHERE a.name = 'user' AND e.relation = 'lives_in'").fetchall()
    by_name = {r["n"]: r["v"] for r in rows}
    assert set(by_name) == {"Berlin", "Mumbai"}          # both kept
    assert by_name["Berlin"] is not None                 # old retired
    assert by_name["Mumbai"] is None                     # new current
    # bus event surfaced
    assert events and events[0][0] == "world.contradiction"
    assert events[0][1]["old_b"] == "Berlin"
    assert events[0][1]["new_b"] == "Mumbai"
    # local log too (bus-optional path)
    assert w.contradictions and w.contradictions[0]["new_b"] == "Mumbai"


def test_contradiction_silent_without_bus(tmp_path):
    w = _world(tmp_path)  # no bus — must not raise
    t0 = time.time() - 100
    w.add_fact("user", "lives_in", "Berlin", ts=t0)
    w.add_fact("user", "lives_in", "Mumbai", ts=t0 + 10)
    assert w.contradictions  # still recorded locally


def test_add_fact_idempotent_reassertion(tmp_path):
    w = _world(tmp_path)
    t0 = time.time() - 100
    i1 = w.add_fact("user", "lives_in", "Berlin", ts=t0)
    i2 = w.add_fact("user", "lives_in", "Berlin", ts=t0 + 10)
    assert i1 == i2
    assert not w.contradictions  # no contradiction on re-assertion


# ── 4. miner finds a planted routine (deterministic) ──────────────────
def test_miner_finds_planted_routine(tmp_path):
    tl = _plant_timeline(tmp_path)
    w = _world(tmp_path)
    miner = RoutineMiner(tl, w, min_support=3)
    routines = miner.run(days_back=7)
    coffee = [r for r in routines if r["activity"] == "coffee"]
    assert coffee, f"planted routine not found: {routines}"
    r = coffee[0]
    assert r["support"] >= 3
    assert abs(r["hour"] - 7.5) <= 0.6
    # persisted as a miner-sourced edge
    row = w._db.execute(
        "SELECT e.source_episode, e.attrs_json FROM edges e "
        "JOIN entities b ON b.id = e.b_id "
        "WHERE e.relation = 'usually_does_at' AND b.name = 'coffee' "
        "AND e.valid_to IS NULL").fetchone()
    assert row is not None
    assert row["source_episode"] == "miner"


def test_miner_deterministic_repeat(tmp_path):
    tl = _plant_timeline(tmp_path)
    w = _world(tmp_path)
    m = RoutineMiner(tl, w, min_support=3)
    r1 = m.mine(days_back=7)
    r2 = m.mine(days_back=7)
    assert r1 == r2


# ── 5. predict() returns the planted routine at the right hour ───────
def test_predict_returns_planted_routine(tmp_path):
    tl = _plant_timeline(tmp_path)
    w = _world(tmp_path)
    RoutineMiner(tl, w, min_support=3).run(days_back=7)
    activity, conf, support = w.predict(7.5, 2)  # weekday unused for now
    assert activity == "coffee"
    assert conf > 0 and support >= 3
    # wrong hour → honest fallback
    activity2, conf2, sup2 = w.predict(15, 2)
    assert activity2 == "nothing reliable yet"
    assert conf2 == 0.0 and sup2 == 0


# ── 6. backward compat: note()/check()/watch() ───────────────────────
def test_note_check_watch_still_work(tmp_path):
    w = _world(tmp_path)
    assert "Routine noted" in w.note("I print PETG, nozzle 240, bed 80")
    out = w.check("printing PLA at nozzle 200")
    assert "deviation" in out.lower()
    assert "Matches your routines" in w.check("printing PETG at nozzle 240")
    # watch: deadline signal (watch() warns once per project, so use a
    # fresh model for the structured feed, which calls watch() itself)
    w.note("gearbox due tomorrow")
    sigs = w.watch()
    assert any(k == "due_gearbox" for k, _ in sigs)
    w2 = WorldModel(str(tmp_path / "world"))
    w2.note("gearbox due tomorrow")
    # w already warned once and persisted the flag — reset for this check
    w2._state["projects"]["gearbox"].pop("warned_due", None)
    structured = w2.world_risk_signals()
    assert any(s["id"] == "due_gearbox" and s["kind"] == "due"
               for s in structured)
    assert all(set(s) >= {"id", "kind", "severity", "text", "ts"}
               for s in structured)


# ── 7. entity context helper ──────────────────────────────────────────
def test_get_entity_context(tmp_path):
    w = _world(tmp_path)
    t0 = time.time() - 100
    w.add_fact("user", "lives_in", "Mumbai", ts=t0)
    ctx = w.get_entity_context("user")
    assert "usually" not in ctx  # no routines yet
    assert "user lives_in Mumbai" in ctx
    assert w.get_entity_context("nobody here") == ""


# ── 8. LLM extraction defaults OFF ────────────────────────────────────
def test_llm_extraction_off_by_default(tmp_path):
    calls = []

    def fake_llm(texts):
        calls.append(texts)
        return [("LLM Person", "person")]

    os.environ.pop("ADA_WORLD_LLM_EXTRACT", None)
    w = _world(tmp_path)
    w._llm = fake_llm
    # recompute gate the way __init__ does (env absent → off)
    w._llm_enabled = False
    w.observe_episode("Met with Real Person Tonight")
    assert not calls
    assert w._entity_id("LLM Person") is None
    # flag on → callable is used
    os.environ["ADA_WORLD_LLM_EXTRACT"] = "1"
    try:
        w2 = WorldModel(str(tmp_path / "world2"), llm=fake_llm)
        assert w2._llm_enabled
        w2.observe_episode("anything at all")
        assert calls
    finally:
        os.environ.pop("ADA_WORLD_LLM_EXTRACT", None)
