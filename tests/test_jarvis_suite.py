"""Tests for the advanced JARVIS suite: autonomy ritual, wellbeing, E.D.I.T.H.
query fan-out, deep sweep prompts, ambient vision state."""
import sys
import os
import time
import asyncio
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from autonomy import AutonomyLevels
from wellbeing import WellbeingMonitor
from edith import build_queries, gather_sources, build_prompt
from deep_sweep import build_sweep_queries, build_verdict_prompt
from ambient_vision import AmbientVision


def make_autonomy():
    return AutonomyLevels(tempfile.mkdtemp())


# ── autonomy ritual (F0: real gates — the full-auto directive is gone) ──
def test_default_is_standard_and_gating():
    a = make_autonomy()
    assert a.level == 1  # Standard default, never armed by construction
    assert a.allows_background_tool("run_web_agent") is False  # external needs armed L3
    assert a.allows_background_tool("search_web") is True      # read-only flows
    assert a.allows_autonomous_destructive() is False


def test_arm_ritual_requires_confirm():
    a = make_autonomy()   # default is L1; ritual is the only way to L3
    msg = a.request_arm()
    assert "confirm" in msg.lower()
    assert a.level != 3                      # not armed by request alone
    msg2 = a.confirm_arm()
    assert a.level == 3 and a.armed is True
    assert "stand down" in msg2.lower()


def test_confirm_without_arm_fails():
    a = make_autonomy()
    assert "nothing to confirm" in a.confirm_arm().lower()


def test_stand_down_disarms_instantly():
    a = make_autonomy()
    a.request_arm(); a.confirm_arm()
    assert a.level == 3
    msg = a.stand_down()
    assert a.level == 1 and a.armed is False
    assert "disengaged" in msg.lower()


def test_arm_request_expires():
    a = make_autonomy()
    a.request_arm()
    a._arm_pending_ts = time.time() - 999    # stale
    assert "expired" in a.confirm_arm().lower()


def test_cad_never_gated():
    a = make_autonomy()
    a.set_level(0)
    assert a.allows_background_tool("generate_cad") is True   # standing order


def test_autonomy_state_persists():
    d = tempfile.mkdtemp()
    a = AutonomyLevels(d)
    a.request_arm(); a.confirm_arm()
    b = AutonomyLevels(d)
    assert b.level == 3


def test_level_zero_gates_background():
    # L0 Training Wheels: nothing runs ungated — every action needs a yes.
    a = make_autonomy()
    a.set_level(0)
    assert a.allows_background_tool("run_web_agent") is False
    assert a.allows_background_tool("search_web") is False
    assert a.allows_background_tool("generate_cad") is True   # CAD outside the ladder


# ── wellbeing ─────────────────────────────────────────────────────────────────
def test_wellbeing_snapshot_and_break_nudge():
    w = WellbeingMonitor(tempfile.mkdtemp())
    w.work_stretch_s = 3 * 3600
    nudge = w.break_nudge()
    assert nudge and "two and a half hours" in nudge[1]
    assert w.break_nudge() is None           # fires once per stretch


def test_wellbeing_morning_sleep_note():
    w = WellbeingMonitor(tempfile.mkdtemp())
    w.late_night_active_min = 60
    note = w.morning_sleep_note()
    assert note and "midnight" in note[1]
    assert w.morning_sleep_note() is None    # once only


def test_wellbeing_snapshot_text():
    w = WellbeingMonitor(tempfile.mkdtemp())
    w.active_s_today = 5400
    snap = w.snapshot()
    assert "1.5h today" in snap


# ── E.D.I.T.H. ────────────────────────────────────────────────────────────────
def test_edith_query_fanout_person():
    qs = build_queries("Tony Stark", "person")
    assert len(qs) == 5
    assert any("biography" in q for q in qs)
    assert any("controversy" in q for q in qs)


def test_edith_query_fanout_domain():
    qs = build_queries("starkindustries.com", "domain")
    assert any("whois" in q for q in qs)


def test_edith_gather_sanitizes_and_formats():
    async def run():
        async def fake_search(q, k):
            return f"results for {q}"
        def sanitize(t):
            return t, False
        out = await gather_sources(fake_search, "Pepper Potts", "person", sanitize)
        assert "results for" in out and "###" in out
    asyncio.run(run())


def test_edith_prompt_contains_structure():
    p = build_prompt("X", "person", "RAW")
    assert "===BRIEF===" in p and "RAW FINDINGS" in p and "never repeat" in p.lower()


def test_edith_extract_spoken_robust():
    from edith import extract_spoken
    # new delimiter format
    t1 = "Sir, here is the brief aloud. ===BRIEF===\n- Identity: x"
    assert extract_spoken(t1) == "Sir, here is the brief aloud."
    # legacy SPOKEN: header
    t2 = "SPOKEN:\nSpoken words here.\nBRIEF:\n- a"
    assert extract_spoken(t2) == "Spoken words here."
    # model echoing instructions -> stripped
    t3 = "SPOKEN (exactly 4-5 sentences):\nReal content survives."
    assert "Real content survives." in extract_spoken(t3)
    assert "exactly 4-5" not in extract_spoken(t3)
    # empty
    assert extract_spoken("") == ""


# ── deep sweep ────────────────────────────────────────────────────────────────
def test_sweep_queries_angles():
    qs = build_sweep_queries("the earth is flat")
    assert len(qs) >= 5
    assert any("rebuttal" in q or "criticism" in q for q in qs)
    assert any("debunked" in q for q in qs)


def test_verdict_prompt_structure():
    p = build_verdict_prompt("claim X", "### a\n1\n### b\n2")
    assert "VERDICT" in p and "CONTRADICTIONS" in p and "2" in p


# ── ambient vision ────────────────────────────────────────────────────────────
def test_vision_absorbs_and_dedupes():
    v = AmbientVision(get_frame_fn=lambda: None, vision_fn=None, speak_fn=None)
    assert v._absorb("Sir is holding a red bracket") is False   # first look: learn silently
    assert v._absorb("Sir is holding a red bracket") is False   # duplicate
    assert v._absorb("A second person just walked in") is True  # notable change
    assert len(v.scene_facts) == 2


def test_vision_context_block():
    v = AmbientVision(get_frame_fn=lambda: None, vision_fn=None, speak_fn=None)
    v._absorb("whiteboard shows equations")
    block = v.context_block()
    assert "CURRENT SCENE" in block and "equations" in block


def test_vision_clear_frames_ignored():
    v = AmbientVision(get_frame_fn=lambda: None, vision_fn=None, speak_fn=None)
    assert v._absorb("") is False
    assert v.context_block() == ""
