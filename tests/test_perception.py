"""Tests for backend/perception.py — the unified perception timeline (F4)."""
import json
import time

import pytest

import perception
from perception import PerceptionTimeline, SCHEMA_V


@pytest.fixture()
def tl(tmp_path):
    t = PerceptionTimeline(tmp_path / "ada_events.db")
    yield t
    t.close()


def _seed(tl, n=5, source="voice", type_="voice.turn"):
    seqs = []
    for i in range(n):
        seqs.append(tl.emit(source, type_,
                            {"text": f"test utterance number {i} about orcaslicer"},
                            ts=1_700_000_000.0 + i))
    return seqs


# ── append/query round-trip ─────────────────────────────────────────────
def test_append_query_roundtrip(tl):
    seqs = _seed(tl, 3)
    assert seqs == [1, 2, 3]  # append-only, seq assigned in order
    rows = tl.query(source="voice", type="voice.turn")
    assert len(rows) == 3
    assert rows[0]["payload"]["text"].startswith("test utterance number 0")
    assert rows[0]["source"] == "voice"
    assert rows[0]["type"] == "voice.turn"
    assert rows[0]["ts_utc"] == pytest.approx(1_700_000_000.0)


def test_schema_v_on_every_event(tl):
    _seed(tl, 4)
    rows = tl.query()
    assert rows, "expected events"
    for r in rows:
        assert r["schema_v"] == SCHEMA_V
    # raw rows too
    raw = tl._conn.execute("SELECT DISTINCT schema_v FROM events").fetchall()
    assert [r[0] for r in raw] == [SCHEMA_V]


def test_query_filters(tl):
    t0 = 1_700_000_000.0
    tl.emit("voice", "voice.turn", {"text": "hello"}, ts=t0)
    tl.emit("screen", "screen.note", {"caption": "orca open"}, ts=t0 + 10)
    tl.emit("agent", "agent.interjection", {"said": "nudge"}, ts=t0 + 20)
    assert len(tl.query(source="screen")) == 1
    assert len(tl.query(type="voice.turn")) == 1
    assert len(tl.query(ts_from=t0 + 5, ts_to=t0 + 15)) == 1
    assert len(tl.query(limit=2)) == 2


# ── FTS5 search ─────────────────────────────────────────────────────────
def test_fts5_search_finds_events(tl):
    _seed(tl, 3)
    tl.emit("screen", "screen.note", {"caption": "petg profile sliced"},
            ts=1_700_000_100.0)
    hits = tl.query(text="orcaslicer")
    assert len(hits) == 3  # the voice.turn payloads mention it
    hits = tl.query(text="petg")
    assert len(hits) == 1
    assert hits[0]["type"] == "screen.note"
    hits = tl.query(text="orcaslicer", source="screen")
    assert hits == []  # filters combine
    hits = tl.query(text="zzzznothing")
    assert hits == []


# ── append-only: corrections are new events ─────────────────────────────
def test_correction_appends_not_updates(tl):
    seq = tl.emit("voice", "voice.turn", {"text": "wrong text"},
                  ts=1_700_000_000.0)
    before = tl.count()
    new = tl.correct("voice", "voice.turn", {"text": "right text"},
                     supersedes=seq, ts=1_700_000_001.0)
    assert tl.count() == before + 1
    assert new == seq + 1
    orig = tl.query(type="voice.turn")
    assert orig[0]["payload"]["text"] == "wrong text"  # original untouched
    corr = tl.query(type="voice.turn.corrected")
    assert corr[0]["payload"]["supersedes_seq"] == seq


# ── compaction ──────────────────────────────────────────────────────────
def test_compaction_deletes_raw_keeps_summaries(tl):
    old = 1_700_000_000.0
    _seed(tl, 6)
    # shift them all into the past so they're compactable
    with tl._conn:
        tl._conn.execute("UPDATE events SET ts_utc = ? WHERE ts_utc >= ?",
                         (old, 1_700_000_000.0))
    assert tl.count() == 6
    r = tl.compact(old + 3600, keep_summary=True)
    assert r["raw_deleted"] == 6
    assert r["summaries_written"] == 1  # one source×type×day group
    assert tl.count() == 0
    # summaries survive
    sums = tl._conn.execute("SELECT * FROM summaries").fetchall()
    assert len(sums) == 1
    assert "6 voice.voice.turn events" in sums[0]["summary"]
    assert sums[0]["event_count"] == 6
    # FTS index no longer finds the deleted payloads
    assert tl.query(text="orcaslicer") == []


def test_compact_ttl_per_source(tl):
    now = 1_800_000_000.0
    tl.emit("voice", "voice.turn", {"text": "old voice"}, ts=now - 100 * 86400)
    tl.emit("screen", "screen.note", {"caption": "old screen"}, ts=now - 40 * 86400)
    tl.emit("tool", "tool.call", {"tool": "remember"}, ts=now - 400 * 86400)
    r = tl.compact_ttl(now=now)
    # voice ttl 90d -> old voice (100d) compacted; screen ttl 30d -> compacted;
    # tool ttl 365d -> 400d > 365d, also compacted
    remaining = {e["type"] for e in tl.query()}
    assert "voice.turn" not in remaining
    assert "screen.note" not in remaining
    assert "tool.call" not in remaining
    assert r["raw_deleted"] == 3
    assert tl._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 3


# ── export / wipe ───────────────────────────────────────────────────────
def test_export_and_delete_everything(tl, tmp_path):
    _seed(tl, 3)
    out = tl.export_all(tmp_path / "dump.jsonl")
    assert out["events"] == tl.count() == 3
    lines = (tmp_path / "dump.jsonl").read_text().splitlines()
    assert all(json.loads(l)["record"] in ("event", "summary") for l in lines)
    ev_lines = [l for l in lines if json.loads(l)["record"] == "event"]
    assert len(ev_lines) == tl.count()
    r = tl.delete_everything()
    assert r["events_deleted"] == 3
    assert tl.count() == 0
    assert tl.query(text="orcaslicer") == []
    assert tl._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 0


# ── observe tiers ───────────────────────────────────────────────────────
def test_tier0_writes_nothing_to_db(tmp_path):
    t = PerceptionTimeline(tmp_path / "t0.db")
    t.set_retention_tier(0)
    assert t.emit("voice", "voice.turn", {"text": "should not land"}) is None
    assert t.count() == 0
    t.close()


def test_tier1_writes_and_ttl_scaling(tl):
    tl.set_retention_tier(1)
    assert tl.ttl_for("voice") == pytest.approx(90.0)
    tl.set_retention_tier(2)
    assert tl.ttl_for("voice") == pytest.approx(360.0)
    tl.set_retention_tier(1)
    assert tl.emit("voice", "voice.turn", {"text": "x"}) == 1


# ── projections ─────────────────────────────────────────────────────────
def test_projections_deterministic(tl):
    t0 = 1_700_000_000.0
    for i in range(3):
        tl.emit("voice", "voice.turn", {"text": f"hello {i}"}, ts=t0 + i,
                session_id="sess-a")
    for i in range(2):
        tl.emit("screen", "screen.note", {"caption": f"cap {i}"}, ts=t0 + 100 + i,
                session_id="sess-b")
    d1 = tl.daily_digest("2023-11-14")
    d2 = tl.daily_digest("2023-11-14")
    assert d1 == d2  # pure over the stream
    assert d1["event_count"] == 5
    assert len(d1["groups"]) == 2
    eps1 = tl.episodes()
    eps2 = tl.episodes()
    assert eps1 == eps2
    assert [e["session_id"] for e in eps1] == ["sess-a", "sess-b"]
    assert eps1[0]["event_count"] == 3
    assert eps1[1]["sources"] == ["screen"]


# ── module singleton is a safe no-op when uninitialized ─────────────────
def test_module_emit_noop_without_init():
    assert perception._INSTANCE is None or True
    # save/restore singleton
    old = perception._INSTANCE
    perception._INSTANCE = None
    try:
        assert perception.emit("voice", "voice.turn", {"text": "x"}) is None
    finally:
        perception._INSTANCE = old


# ── sensory ring bridge ─────────────────────────────────────────────────
def test_ring_bridge_flows_when_tier_ge_1(tmp_path):
    from sensory import SensoryRing
    t = PerceptionTimeline(tmp_path / "bridge.db")
    ring = SensoryRing(timeline=t)
    ring.push_audio("hello ada")
    ring.push_screen("orca slicer open")
    ring.push_frame("workshop view")
    rows = t.query(source="sensor")
    assert len(rows) == 3
    assert {r["type"] for r in rows} == {"ring.audio", "ring.screen", "ring.frame"}
    # old readers still work: ring answers from RAM, unchanged behavior
    assert "hello ada" in ring.answer_what_just("say")
    t.close()


def test_ring_bridge_quiet_at_tier0(tmp_path):
    from sensory import SensoryRing
    t = PerceptionTimeline(tmp_path / "bridge0.db")
    t.set_retention_tier(0)
    ring = SensoryRing(timeline=t)
    ring.push_audio("hello ada")
    assert t.count() == 0  # ring only
    assert "hello ada" in ring.answer_what_just("say")  # RAM read intact
    t.close()


def test_wal_mode(tmp_path):
    t = PerceptionTimeline(tmp_path / "wal.db")
    mode = t._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    t.close()
