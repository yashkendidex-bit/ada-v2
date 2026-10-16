"""Tests for the Feature 8 learned proactivity policy.

Covers: v1 heuristic send/suppress, Thompson sampling convergence,
hush/reaction logging, learned quiet hours, the daily cap, the
voice-turn rail, SignalBus push wake, and F3 world_risk_signals as a
candidate source. All state in tmp dirs; no network, no models.
"""
import asyncio
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "backend"
import sys
sys.path.insert(0, str(BACKEND_DIR))

from proactivity_policy import (
    InterruptionPolicy, PolicyStore, context_bucket,
    DEFAULT_THRESHOLD, DEFAULT_MAX_PER_DAY, REACTION_ENGAGED, REACTION_DISMISSED,
)
from proactive import ProactiveEngine
from signals import SignalBus


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "policy.db")


@pytest.fixture()
def store(db_path):
    s = PolicyStore(db_path)
    yield s
    s.close()


@pytest.fixture()
def policy(store):
    return InterruptionPolicy(store=store)


@pytest.fixture()
def engine(db_path):
    """Engine with a stub kernel; quiet hours disabled for determinism."""
    kernel = SimpleNamespace(scheduler=None, signals=None, world=None,
                             wellbeing=None, task_board_ref={})
    eng = ProactiveEngine(
        kernel, speak_fn=lambda line: True,
        db_path=db_path, quiet_hours=(0, 0),
        voice_turn_active_fn=lambda: False,
        speaker_present_fn=lambda: False)
    yield eng
    eng.policy.store.close()


def _sent_row(store, iid, stype="signal", reaction="none"):
    store.log_decision(iid, stype, {"bucket": context_bucket()}, "sent",
                       "score-ok")
    if reaction != "none":
        store.log_reaction(iid, reaction, 1.0)


# ── v1 heuristic ──────────────────────────────────────────────────────────
class TestV1:
    def test_v1_sends_when_relevance_high_and_cost_low(self, engine):
        d = engine._decide("reminder_1", "reminder", False)
        assert d["action"] == "sent"
        assert d["reason"] is None
        assert d["policy"] == "v1"
        assert d["score"] > DEFAULT_THRESHOLD

    def test_v1_suppresses_on_dismissal_streak(self, engine):
        # Five recent dismissals -> escalating suppression buries the score.
        for i in range(5):
            _sent_row(engine.policy.store, f"s{i}", "reminder",
                      REACTION_DISMISSED)
        d = engine._decide("reminder_new", "reminder", False)
        assert d["action"] == "suppressed"
        assert d["score"] <= DEFAULT_THRESHOLD

    def test_v1_suppresses_high_cost_context(self, policy):
        cost = policy.interrupt_cost({
            "focus": True, "typing": True, "calendar_busy": True,
            "minutes_since_last": 2.0})
        assert cost > 0.5
        assert policy.v1_score(0.8, {
            "focus": True, "typing": True, "calendar_busy": True,
            "minutes_since_last": 2.0}) <= DEFAULT_THRESHOLD

    def test_defaults_conservative(self, policy):
        assert policy.threshold == DEFAULT_THRESHOLD == 0.7
        assert policy.max_per_day == DEFAULT_MAX_PER_DAY == 3


# ── v2 Thompson sampling ────────────────────────────────────────────────
class TestV2:
    def test_convergence(self, store):
        """50 engaged in bucket A vs 50 dismissed in bucket B: the sampled
        send-probability must separate decisively."""
        for _ in range(50):
            store._update_bucket("signal", {"bucket": "bA"}, success=True)
        for _ in range(50):
            store._update_bucket("signal", {"bucket": "bB"}, success=False)
        aA, bA = store.bucket_samples("bA", "signal")
        aB, bB = store.bucket_samples("bB", "signal")
        assert (aA, bA) == (51.0, 1.0)
        assert (aB, bB) == (1.0, 51.0)
        random.seed(7)
        meanA = sum(random.betavariate(aA, bA) for _ in range(200)) / 200
        meanB = sum(random.betavariate(aB, bB) for _ in range(200)) / 200
        assert meanA - meanB > 0.5

    def test_v2_decides_only_with_enough_samples(self, policy):
        bucket = context_bucket()
        assert not policy.v2_decides("signal", bucket)  # 0 samples
        for _ in range(5):
            policy.store._update_bucket("signal", {"bucket": bucket},
                                        success=True)
        assert policy.v2_decides("signal", bucket)

    def test_v2_drives_decision_when_data_exists(self, engine):
        """A bucket that always engages sends even at modest relevance;
        a bucket that always dismisses suppresses even hot signals."""
        bucket = context_bucket()
        for _ in range(8):
            engine.policy.store._update_bucket("signal", {"bucket": bucket},
                                               success=True)
        d = engine._decide("k_hot", "signal", False)
        assert d["policy"] == "v2"
        assert d["action"] == "sent"


# ── reactions / hush ────────────────────────────────────────────────────
class TestReactions:
    def test_hush_logged_as_dismissal_and_updates_bucket(self, engine):
        engine.policy.store.log_decision(
            "int_h1", "signal", {"bucket": context_bucket()}, "sent", "ok")
        engine._last_sent_iid = "int_h1"
        engine._last_sent_at = time.time()
        engine.on_hush()
        row = engine.policy.store._conn.execute(
            "SELECT reaction FROM interruptions WHERE id='int_h1'").fetchone()
        assert row[0] == REACTION_DISMISSED
        assert engine._hush_until > time.time()
        alpha, beta = engine.policy.store.bucket_samples(
            context_bucket(), "signal")
        assert (alpha, beta) == (1.0, 2.0)  # the dismissal moved beta

    def test_hush_cooldown_suppresses(self, engine):
        engine.policy.store.log_decision(
            "int_h2", "signal", {"bucket": context_bucket()}, "sent", "ok")
        engine._last_sent_iid = "int_h2"
        engine._last_sent_at = time.time()
        engine.on_hush()
        d = engine._decide("k_after_hush", "reminder_urgent", False)
        assert d["action"] == "suppressed"
        assert d["reason"] == "hush-cooldown"

    def test_log_reaction_api_rejects_unknown(self, policy):
        assert policy.log_reaction("nope", "engaged") is False
        assert policy.log_reaction("nope", "bogus-reaction") is False

    def test_record_feedback_attributes_recent_send(self, engine):
        engine.policy.store.log_decision(
            "int_f1", "signal", {"bucket": context_bucket()}, "sent", "ok")
        engine._last_sent_iid = "int_f1"
        engine._last_sent_at = time.time()
        assert engine.record_feedback(REACTION_ENGAGED) is True
        row = engine.policy.store._conn.execute(
            "SELECT reaction FROM interruptions WHERE id='int_f1'").fetchone()
        assert row[0] == REACTION_ENGAGED
        # second feedback for the same send is refused (already answered)
        assert engine.record_feedback(REACTION_DISMISSED) is False


# ── rails ───────────────────────────────────────────────────────────────
class TestRails:
    def test_quiet_hour_learning(self, engine):
        """An hour with sustained high dismissal becomes soft-quiet."""
        store = engine.policy.store
        h = datetime.now().hour
        for i in range(12):
            store.log_decision(f"q{i}", "signal",
                               {"bucket": context_bucket()}, "sent", "ok")
            store.log_reaction(f"q{i}", REACTION_DISMISSED, 2.0)
            store.record_send(h)
        assert store.is_soft_quiet_hour(h)
        d = engine._decide("k_soft", "signal", False)
        assert d["action"] == "suppressed"
        assert d["reason"] == "soft-quiet-hour"

    def test_urgent_bypasses_soft_quiet(self, engine):
        store = engine.policy.store
        h = datetime.now().hour
        for i in range(12):
            store.log_decision(f"u{i}", "signal",
                               {"bucket": context_bucket()}, "sent", "ok")
            store.log_reaction(f"u{i}", REACTION_DISMISSED, 2.0)
            store.record_send(h)
        assert store.is_soft_quiet_hour(h)
        d = engine._decide("k_urg", "reminder_urgent", True)
        assert d["action"] == "sent"

    def test_max_per_day_cap(self, engine):
        engine.policy.max_per_day = 2
        d1 = engine._decide("c1", "reminder_urgent", False)
        d2 = engine._decide("c2", "reminder_urgent", False)
        assert d1["action"] == "sent" and d2["action"] == "sent"
        d3 = engine._decide("c3", "reminder_urgent", False)
        assert d3["action"] == "suppressed"
        assert d3["reason"] == "daily-cap"

    def test_voice_turn_suppresses_everything(self, db_path):
        kernel = SimpleNamespace(scheduler=None, signals=None, world=None,
                                 wellbeing=None)
        eng = ProactiveEngine(kernel, speak_fn=lambda line: True,
                              db_path=db_path, quiet_hours=(0, 0),
                              voice_turn_active_fn=lambda: True)
        try:
            d = eng._decide("v1", "reminder_urgent", True)
            assert d["action"] == "suppressed"
            assert d["reason"] == "voice-turn-active"
        finally:
            eng.policy.store.close()

    def test_speaker_present_suppresses(self, db_path):
        kernel = SimpleNamespace(scheduler=None, signals=None, world=None,
                                 wellbeing=None)
        eng = ProactiveEngine(kernel, speak_fn=lambda line: True,
                              db_path=db_path, quiet_hours=(0, 0),
                              voice_turn_active_fn=lambda: False,
                              speaker_present_fn=lambda: True)
        try:
            d = eng._decide("s1", "reminder_urgent", False)
            assert d["action"] == "suppressed"
            assert d["reason"] == "speaker-present"
        finally:
            eng.policy.store.close()

    def test_suppressed_never_calls_speak(self, db_path):
        spoken = []
        kernel = SimpleNamespace(scheduler=None, signals=None, world=None,
                                 wellbeing=None)
        eng = ProactiveEngine(kernel, speak_fn=lambda line: spoken.append(line)
                             or True,
                              db_path=db_path, quiet_hours=(0, 0),
                              voice_turn_active_fn=lambda: True)
        try:
            asyncio.run(eng._maybe_speak("v2", "hello sir", "reminder"))
            assert spoken == []
        finally:
            eng.policy.store.close()


# ── signal bus push (inverted loop) ─────────────────────────────────────
class TestSignalPush:
    def test_detection_wakes_engine(self):
        bus = SignalBus()
        kernel = SimpleNamespace(scheduler=None, signals=bus, world=None,
                                 wellbeing=None)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            eng = ProactiveEngine(kernel, speak_fn=lambda line: True,
                                  db_path=str(Path(td) / "p.db"),
                                  quiet_hours=(0, 0),
                                  voice_turn_active_fn=lambda: False)
            try:
                assert eng._subscribed_bus is bus
                eng._wake.clear()
                bus.note_error("boom")
                bus.note_error("boom")   # 2nd repeat fires the detection
                assert eng._wake.is_set()
            finally:
                eng.policy.store.close()

    def test_subscriber_exceptions_never_break_tool_path(self):
        bus = SignalBus()

        def bad(det):
            raise RuntimeError("listener exploded")

        bus.subscribe(bad)
        bus.note_error("x")   # must not raise
        bus.note_error("x")
        assert len(bus.errors) == 2

    def test_file_push_does_not_consume_announcement(self):
        bus = SignalBus()
        seen = []
        bus.subscribe(seen.append)
        bus.note_file("/tmp/model.stl")
        assert seen and seen[0] is not None          # push fired
        det = bus.file_landed()
        assert det is not None                        # still announced once
        assert bus.file_landed() is None               # then consumed


# ── F3 world_risk_signals as candidate source ───────────────────────────
class TestWorldRisk:
    def test_world_risk_signals_become_candidates(self, db_path):
        world = SimpleNamespace(world_risk_signals=lambda: [
            {"id": "pace_gearbox", "kind": "pace", "project": "gearbox",
             "severity": "warn", "text": "gearbox is slipping pace",
             "ts": time.time()}])
        kernel = SimpleNamespace(scheduler=None, signals=None, world=world,
                                 wellbeing=None, task_board_ref={})
        eng = ProactiveEngine(kernel, speak_fn=lambda line: True,
                              db_path=db_path, quiet_hours=(0, 0),
                              voice_turn_active_fn=lambda: False)
        try:
            res = eng._signal_events()
            assert res is not None
            key, line, stype = res
            assert key == "pace_gearbox"
            assert stype == "world_risk"
            assert "gearbox" in line
        finally:
            eng.policy.store.close()


# ── interruption log shape ──────────────────────────────────────────────
class TestLog:
    def test_schema_and_decision_logging(self, store):
        cols = {r[1] for r in
                store._conn.execute("PRAGMA table_info(interruptions)")}
        assert {"id", "ts", "suggestion_type", "context_json", "action",
                "reaction", "latency_s", "reason"} <= cols
        pol = InterruptionPolicy(store=store)
        d = pol.decide({"suggestion_type": "signal", "relevance": 1.0,
                        "quiet_hour": False})
        assert d["action"] == "sent"
        row = store._conn.execute(
            "SELECT action, reaction FROM interruptions WHERE id=?",
            (d["iid"],)).fetchone()
        assert row == ("sent", "none")
