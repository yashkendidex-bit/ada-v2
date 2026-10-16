"""Tests for Feature 7 — speaker & visitor awareness.

All audio is SYNTHESIZED (sine stacks at different fundamentals as fake
speakers); no audio files, no network, no models. All DBs in tmp dirs or
in-memory.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import speaker_id
from speaker_id import (
    MATCH_THRESHOLD, VOICE_ASK_LINE, VoiceprintStore, SpeakerManager,
    fingerprint, identify_turn, record_activity, recent_non_owner_active,
    _reset_activity,
)


# ── synthesized "speakers" ──────────────────────────────────────────────────

def synth(f0: float, dur: float = 5.0, seed: int = 0, noise: float = 0.02):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * 16000)) / 16000.0
    x = np.zeros_like(t)
    for h, a in [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]:
        x += a * np.sin(2 * np.pi * f0 * h * t + 0.3 * np.sin(2 * np.pi * 5 * t))
    x = x / np.max(np.abs(x)) * 0.9
    if noise:
        x += rng.normal(0.0, noise, x.shape)
    return x


def cos(u, v):
    return float(np.dot(u, v))


# ── 1. MFCC fingerprinting ──────────────────────────────────────────────────

def test_fingerprint_deterministic():
    x = synth(110.0, seed=1)
    a, b = fingerprint(x), fingerprint(x)
    assert a is not None and b is not None
    assert cos(a, b) == pytest.approx(1.0, abs=1e-9)


def test_fingerprint_dim():
    fp = fingerprint(synth(110.0, seed=1))
    assert fp.shape == (40,)
    assert abs(np.linalg.norm(fp) - 1.0) < 1e-9  # L2-normalized


def test_same_speaker_matches():
    a = fingerprint(synth(110.0, seed=1))
    b = fingerprint(synth(110.0, seed=2))  # different utterance, same voice
    assert cos(a, b) >= MATCH_THRESHOLD


def test_different_speakers_separate():
    a = fingerprint(synth(110.0, seed=1))
    for f0 in (190.0, 280.0, 350.0):
        b = fingerprint(synth(f0, seed=7))
        assert cos(a, b) < MATCH_THRESHOLD, f"f0={f0}"


def test_short_or_silent_audio_skipped():
    assert fingerprint(synth(110.0, dur=1.0, seed=9)) is None
    assert fingerprint(np.zeros(16000 * 5)) is None
    assert fingerprint(np.array([])) is None


# ── 2. VoiceprintStore ─────────────────────────────────────────────────────

@pytest.fixture()
def store():
    return VoiceprintStore(":memory:")


def test_unknown_is_visitor(store):
    label, score = store.identify(fingerprint(synth(110.0, seed=1)))
    assert label == "visitor"
    assert score < MATCH_THRESHOLD


def test_enroll_then_known(store):
    fp = fingerprint(synth(190.0, seed=3))
    label = store.enroll("Maya", fp, consent_event_ref="ask@1")
    assert label == "known:Maya"
    got, score = store.identify(fingerprint(synth(190.0, seed=4)))
    assert got == "known:Maya" and score >= MATCH_THRESHOLD
    # a different voice still reads as visitor
    other, _ = store.identify(fingerprint(synth(110.0, seed=1)))
    assert other == "visitor"


def test_owner_label(store):
    fp = fingerprint(synth(110.0, seed=1))
    assert store.enroll("Yash", fp, consent_event_ref="ask@0", owner=True) == "owner:Yash"
    label, score = store.identify(fingerprint(synth(110.0, seed=2)))
    assert label == "owner:Yash" and score >= MATCH_THRESHOLD
    assert store.owner_name() == "Yash"


def test_forget_deletes_and_reverts_to_visitor(store):
    store.enroll("Maya", fingerprint(synth(190.0, seed=3)))
    assert store.forget("Maya") is True
    assert store.forget("Maya") is False  # already gone
    label, _ = store.identify(fingerprint(synth(190.0, seed=4)))
    assert label == "visitor"
    assert store.list_known_voices() == []


def test_list_known_voices_names_only(store):
    store.enroll("Maya", fingerprint(synth(190.0, seed=3)), owner=False)
    store.enroll("Yash", fingerprint(synth(110.0, seed=1)), owner=True)
    people = store.list_known_voices()
    assert {(p["name"], p["role"]) for p in people} == {
        ("Maya", "known"), ("Yash", "owner")}
    # no audio, no vectors leak through the brain-readable API
    assert all(set(p) == {"person_id", "name", "role", "enrolled_ts",
                          "consent_event_ref"} for p in people)


# ── 3. identify_turn: cloud gated on env, local is primary ─────────────────

def test_identify_turn_local_without_key(store, monkeypatch):
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    label, score = identify_turn(fingerprint(synth(110.0, seed=1)), store=store)
    assert label == "visitor"


def test_identify_turn_cloud_failure_falls_back_to_local(store, monkeypatch):
    monkeypatch.setenv("PYANNOTE_API_KEY", "bogus-key")
    import httpx

    class _Boom:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k): raise httpx.ConnectError("no network")
        def get(self, *a, **k): raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "Client", _Boom)
    label, score = identify_turn(fingerprint(synth(110.0, seed=1)), store=store,
                                 audio_bytes=b"fake-webm", filename="t.webm")
    assert label == "visitor"  # local path answered, nothing raised


def test_identify_turn_never_hardcodes_key(store, monkeypatch):
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    src = Path(speaker_id.__file__).read_text()
    assert "PYANNOTE_API_KEY" in src
    # the only read of the key is via the environment
    assert src.count("os.environ.get(\"PYANNOTE_API_KEY\")") >= 1
    assert "sk-" not in src and "Bearer " not in src.replace(
        '"Bearer {api_key}"', "")


# ── 4. the voice-ask flow ───────────────────────────────────────────────────

@pytest.fixture()
def mgr():
    return SpeakerManager(VoiceprintStore(":memory:"), owner_hint="Yash")


def _feed(mgr, pcm, text, monkeypatch):
    monkeypatch.setattr(speaker_id, "decode_to_pcm16", lambda b, timeout=15.0: pcm)
    return mgr.process_turn(b"fake-bytes", "turn.webm", transcript_text=text)


def test_unknown_voice_triggers_one_ask(mgr, monkeypatch):
    out = _feed(mgr, synth(190.0, seed=3), "hello, is anyone there", monkeypatch)
    assert out["speaker_label"] == "visitor"
    assert out["voice_ask"] == VOICE_ASK_LINE
    assert mgr.get_pending_ask() == VOICE_ASK_LINE


def test_ask_rate_limited_once_per_voice_per_session(mgr, monkeypatch):
    out = _feed(mgr, synth(190.0, seed=3), "hello", monkeypatch)
    assert out["voice_ask"] == VOICE_ASK_LINE
    # the person brushes past the question -> declined, ask cleared
    out2 = _feed(mgr, synth(190.0, seed=5), "never mind", monkeypatch)
    assert out2["voice_ask"] is None and mgr.get_pending_ask() is None
    # same voice speaks again: already asked this session -> no nag
    out3 = _feed(mgr, synth(190.0, seed=6), "hey again", monkeypatch)
    assert out3["voice_ask"] is None
    # a genuinely different unknown voice still gets the ask
    out4 = _feed(mgr, synth(280.0, seed=7), "hi there", monkeypatch)
    assert out4["voice_ask"] == VOICE_ASK_LINE
    # answer with a name -> enrolled, ask cleared
    out5 = _feed(mgr, synth(280.0, seed=9), "I'm Maya", monkeypatch)
    assert out5["enrolled"] == "Maya"
    # next turn recognized as known
    out6 = _feed(mgr, synth(280.0, seed=11), "thanks", monkeypatch)
    assert out6["speaker_label"] == "known:Maya"


def test_owner_auto_enrolls_on_explicit_confirmation(mgr, monkeypatch):
    _feed(mgr, synth(110.0, seed=1), "hey ada", monkeypatch)
    out = _feed(mgr, synth(110.0, seed=2), "yes, that's me", monkeypatch)
    assert out["speaker_label"] == "owner:Yash"
    assert out["enrolled"] == "Yash"


def test_no_silent_enrollment(mgr, monkeypatch):
    # many unknown turns, no answer to the ask -> DB stays empty
    for s in range(3, 8):
        _feed(mgr, synth(190.0, seed=s), "talking away", monkeypatch)
    assert mgr.store.list_known_voices() == []


def test_voice_forget_command(mgr, monkeypatch):
    _feed(mgr, synth(190.0, seed=3), "hi", monkeypatch)
    _feed(mgr, synth(190.0, seed=4), "I'm Maya", monkeypatch)
    assert mgr.store.get("Maya") is not None
    out = _feed(mgr, synth(190.0, seed=9), "please forget that voice", monkeypatch)
    assert out["forgot"] == "Maya"
    assert mgr.store.get("Maya") is None
    out2 = _feed(mgr, synth(190.0, seed=10), "hi again", monkeypatch)
    assert out2["speaker_label"] == "visitor"


# ── 5. speaker_present() — the F8 rail ──────────────────────────────────────

def test_speaker_present_reflects_recent_non_owner_activity():
    _reset_activity()
    try:
        assert recent_non_owner_active() is False
        record_activity("owner:Yash")
        assert recent_non_owner_active() is False
        record_activity("visitor")
        assert recent_non_owner_active() is True
        record_activity("known:Maya")
        assert recent_non_owner_active() is True
        # stale activity does not count
        _reset_activity()
        record_activity("visitor", ts=time.time() - 10 * 60)
        assert recent_non_owner_active(window_minutes=5) is False
        assert recent_non_owner_active(window_minutes=15) is True
    finally:
        _reset_activity()


def test_proactivity_policy_hook_filled():
    from proactivity_policy import speaker_present as hook
    _reset_activity()
    try:
        assert hook() is False
        record_activity("known:Maya")
        assert hook() is True
    finally:
        _reset_activity()


# ── 6. never for auth ───────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", [
    "backend/authenticator.py",
    "backend/autonomy.py",
    "backend/cad_v3/gate.py",
])
def test_voice_id_never_used_for_auth(rel):
    src = (Path(__file__).parent.parent / rel).read_text()
    assert "speaker_id" not in src, f"{rel} must never touch voiceprints"
    assert "VoiceprintStore" not in src


# ── 7. trust console voiceprint panel ───────────────────────────────────────

def test_console_snapshot_voiceprints_panel(tmp_path):
    import replay
    vdb = tmp_path / "voiceprints.db"
    vs = VoiceprintStore(str(vdb))
    vs.enroll("Maya", fingerprint(synth(190.0, seed=3)), owner=False)
    adb = tmp_path / "audit.db"
    adb.touch()
    props = replay.console_snapshot(str(adb), voiceprint_db=str(vdb))
    vp = props["voiceprints"]
    assert vp["people"] == [{"name": "Maya", "role": "known",
                             "enrolled": vp["people"][0]["enrolled"]}]
    assert "centroids only" in vp["retention"]
    assert "never raw audio" in vp["retention"]


def test_console_snapshot_without_voiceprint_db(tmp_path):
    import replay
    adb = tmp_path / "audit.db"
    adb.touch()
    props = replay.console_snapshot(str(adb))
    assert props["voiceprints"]["people"] == []
    assert "forget my voice" in props["voiceprints"]["retention"]
