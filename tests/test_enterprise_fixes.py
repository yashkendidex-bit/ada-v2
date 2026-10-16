"""Enterprise regression tests for the Jarvis-hardening pass.

Covers, without any network access:
  - TTS circuit breaker (Jane stays the only voice; outages fail fast)
  - Desktop autopilot fail-closed destructive + action validation + drag
  - Proactive urgency: overdue escalation, urgent bypass, morning briefing
  - Planner: per-step failure capture + run_goal dry run
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import tts_agent
from desktop_autopilot import DesktopAutopilot
from proactive import ProactiveEngine


# ── TTS breaker ───────────────────────────────────────────────────────────────
def _reset_tts():
    tts_agent._tts_consec_failures = 0
    tts_agent._tts_disabled_until = 0.0


def test_tts_health_reports_jane_only():
    _reset_tts()
    h = tts_agent.tts_health()
    assert h["voice"] == "jane"
    assert h["engine"] == "voxtral"
    assert h["breaker_open"] is False


def test_tts_breaker_opens_and_recovers():
    _reset_tts()
    for i in range(tts_agent.TTS_MAX_CONSECUTIVE_FAILURES):
        tts_agent._record_tts_failure(f"boom {i}")
    assert tts_agent._tts_breaker_open() is True
    assert tts_agent._tts_muted() is True  # fail fast while open
    out = asyncio.run(tts_agent.synthesize("hello sir"))
    assert out == (b"", "mp3")  # immediate, no network
    tts_agent._record_tts_success()
    assert tts_agent._tts_breaker_open() is False


# ── desktop autopilot ─────────────────────────────────────────────────────────
def test_desktop_validates_actions():
    assert DesktopAutopilot._validate_action({"action": "click", "x": 100, "y": 200}) is None
    assert DesktopAutopilot._validate_action({"action": "teleport", "x": 1, "y": 1}) is not None
    assert DesktopAutopilot._validate_action({"action": "click", "x": -5, "y": 10}) is not None
    assert DesktopAutopilot._validate_action({"action": "type", "text": "  "}) is not None
    assert DesktopAutopilot._validate_action({"action": "drag", "x": 1, "y": 1, "text": "nope"}) is not None
    assert DesktopAutopilot._validate_action({"action": "drag", "x": 1, "y": 1, "text": "300,400"}) is None


def test_desktop_destructive_fail_closed_without_confirmation_channel():
    async def run():
        ap = DesktopAutopilot(vision_fn=None, ask_user_fn=None)
        ap._screenshot_b64 = lambda: ("x", 1.0)

        async def fake_decide(g, h, img):
            return '{"action": "click", "x": 10, "y": 10, "note": "send the email", "destructive": true}'
        ap._decide = fake_decide
        executed = []

        async def fake_exec(act, scale):
            executed.append(act)
            return "clicked"
        ap._execute = fake_exec
        result = await ap.run_task("send the email")
        assert "no confirmation channel" in result
        assert executed == []  # NOTHING executed
    asyncio.run(run())


def test_desktop_stuck_loop_reports_inconclusive():
    async def run():
        ap = DesktopAutopilot(vision_fn=None, ask_user_fn=None)
        ap._screenshot_b64 = lambda: ("x", 1.0)

        async def fake_decide(g, h, img):
            return '{"action": "click", "x": 10, "y": 10, "note": "same button"}'
        ap._decide = fake_decide
        n = {"c": 0}

        async def fake_exec(act, scale):
            n["c"] += 1
            return "clicked"
        ap._execute = fake_exec
        result = await ap.run_task("do the thing")
        assert result.startswith("INCONCLUSIVE")
        assert n["c"] < 15  # stopped early, did not burn the budget
    asyncio.run(run())


# ── proactive urgency ─────────────────────────────────────────────────────────
class FakeScheduler:
    def __init__(self, items):
        self._items = items

    def list(self):
        return self._items


class FakeKernel:
    def __init__(self, sched=None):
        self.scheduler = sched or FakeScheduler([])
        self.dossier = None


def _engine(kernel, **kw):
    async def speak(line):
        return True
    kw.setdefault("interval_s", 60)
    return ProactiveEngine(kernel, speak_fn=speak, **kw)


def test_proactive_overdue_escalates_once():
    eng = _engine(FakeKernel(FakeScheduler([
        {"id": "r9", "descr": "Standup", "next_run": time.time() - 300},
    ])))
    first = eng._reminder_overdue()
    assert first and first[0] == "overdue_r9"
    eng._last_fired[first[0]] = time.time()
    assert eng._reminder_overdue() is None  # never nags twice


def test_proactive_urgent_bypasses_daily_cap_once(tmp_path):
    """Feature 8: the urgent bypass now runs through the learned policy —
    it skips the daily cap/quiet hours/cooldowns exactly once per key
    (never the voice-turn rail), then the normal policy resumes."""
    async def run():
        spoken = []

        async def speak(line):
            spoken.append(line)
            return True
        import proactive as _p
        eng = _p.ProactiveEngine(
            FakeKernel(FakeScheduler([
                {"id": "rx", "descr": "Medication", "next_run": time.time() + 60},
            ])), speak_fn=speak, interval_s=60, max_per_day=0,
            db_path=str(tmp_path / "p.db"), quiet_hours=(0, 0),
            voice_turn_active_fn=lambda: False)
        try:
            key, line = eng._reminder_warnings()
            assert key == "reminder_rx"
            # daily cap is 0 — a non-urgent line would be suppressed...
            d = eng._decide(key, "reminder", urgent=False)
            assert d["action"] == "suppressed" and d["reason"] == "daily-cap"
            # ...but the urgent bypass whispers through once.
            spoke = await eng._maybe_speak(key, line, "reminder")
            assert spoke is True
            assert spoken and "Medication" in spoken[0]
            # second urgent for the same reminder: bypass spent -> cooldown.
            d2 = eng._decide(key, "reminder", urgent=True)
            assert d2["action"] == "suppressed"
            assert d2["reason"] == "cooldown"
            # hard rail still wins: never interrupt an active voice turn.
            eng._voice_turn_active_fn = lambda: True
            d3 = eng._decide("reminder_rx2", "reminder", urgent=True)
            assert d3["action"] == "suppressed"
            assert d3["reason"] == "voice-turn-active"
        finally:
            eng.policy.store.close()
    asyncio.run(run())


def test_proactive_morning_briefing_once_per_day():
    import proactive as _p

    eng = _engine(FakeKernel(FakeScheduler([
        {"id": "r1", "descr": "Demo", "next_run": time.time() + 3600},
    ])))

    class _FakeNow:
        hour = 8

        def strftime(self, fmt):
            return "2099-01-01"

    class _FakeDT:
        @staticmethod
        def now():
            return _FakeNow()

    real_dt = _p.datetime
    _p.datetime = _FakeDT
    try:
        first = eng._morning_briefing()
        assert first and first[0] == "briefing"
        assert "schedule" in first[1]
        eng._last_fired["briefing"] = 0  # cooldown irrelevant: day-flag owns it
        assert eng._morning_briefing() is None  # same day -> silent
    finally:
        _p.datetime = real_dt


# ── optimization: thinking budgets + ambient staggering ───────────────────
def test_reasoning_budget_classifier():
    import ada as _ada

    assert _ada.AudioLoop._reasoning_for("hey ada") == "low"
    assert _ada.AudioLoop._reasoning_for("yes") == "low"
    assert _ada.AudioLoop._reasoning_for(
        "deep dive into why titanium brackets crack under vibration") is None
    long_chat = ("can you explain in a fair amount of detail how the whole memory "
                 "consolidation pipeline works end to end, including decay curves, "
                 "interference passes and the fidelity ladder")
    assert len(long_chat) > 150
    assert _ada.AudioLoop._reasoning_for(long_chat) is None
    assert _ada.AudioLoop._reasoning_for("") is None


def test_reasoning_budget_env_override(monkeypatch):
    import ada as _ada

    monkeypatch.setenv("ADA_REASONING_EFFORT", "high")
    assert _ada.AudioLoop._reasoning_for("hey ada") == "high"
    monkeypatch.setenv("ADA_REASONING_EFFORT", "bogus")
    assert _ada.AudioLoop._reasoning_for("hey ada") == "low"


def test_ambient_pause_resume_and_interval(monkeypatch):
    from ambient_vision import AmbientVision, LOOK_INTERVAL_S

    assert LOOK_INTERVAL_S >= 60  # the 30s quota furnace is gone
    monkeypatch.setenv("ADA_AMBIENT_INTERVAL_S", "120")
    import importlib
    import ambient_vision as _av
    importlib.reload(_av)
    assert _av.LOOK_INTERVAL_S == 120
    monkeypatch.undo()
    importlib.reload(_av)

    av = AmbientVision(get_frame_fn=lambda: None,
                       vision_fn=lambda m: "", speak_fn=lambda l: True)
    assert av._paused is False
    av.pause()
    assert av._paused is True
    av.resume()
    assert av._paused is False


def test_proactive_interval_env(monkeypatch):
    import proactive as _pro

    monkeypatch.setenv("ADA_PROACTIVE_INTERVAL_S", "300")

    class FakeSched:
        def list(self):
            return []

    class FakeKernel:
        scheduler = FakeSched()
        dossier = None

    async def speak(line):
        return True
    eng = _pro.ProactiveEngine(FakeKernel(), speak_fn=speak)
    assert eng.interval_s == 300


def test_server_survives_missing_authenticator(monkeypatch):
    """MediaPipe absent/broken must never kill the backend (lazy face auth)."""
    import sys
    import importlib
    monkeypatch.setitem(sys.modules, "authenticator", None)
    for mod in ("mediapipe", "mediapipe.tasks"):
        monkeypatch.setitem(sys.modules, mod, None)
    sys.modules.pop("server", None)
    import server as _srv
    importlib.reload(_srv)
    assert _srv.FaceAuthenticator is None
    assert _srv.authenticator is None


# ── web agent fail-closed ───────────────────────────────────────────────────
def test_web_agent_never_auto_acknowledges_safety():
    """The model/backend is untouched; only the acknowledgement policy changed:
    require_confirmation is refused, never auto-approved. page=None proves no
    browser action executes on the refusal path."""
    from web_agent import WebAgent

    class FakeCall:
        id = "c1"
        name = "click_at"
        args = {"x": 100, "y": 100,
                "safety_decision": {"decision": "require_confirmation",
                                    "explanation": "payment step"}}

    async def run():
        agent = WebAgent()
        agent.page = None  # would explode if any action executed
        results = await agent.execute_function_calls([FakeCall()])
        assert len(results) == 1
        _, _, data = results[0]
        assert "refused" in data.get("error", "")
        assert "safety_acknowledgement" not in data
    asyncio.run(run())


# ── planner evidence ────────────────────────────────────────────────────────
def test_planner_execute_captures_step_failures():
    from agent_kernel import Planner

    async def run():
        async def plan_fn(prompt):
            if prompt.startswith("The goal was:"):
                return ("PARTIAL — beta FAILED: RuntimeError: boom. "
                        "What remains: retry beta with a different approach.")
            return "1. alpha\n2. beta\n"
        p = Planner(plan_fn)

        async def execute_fn(step):
            if step == "beta":
                raise RuntimeError("boom")
            return "ok"

        results = await p.execute(["alpha", "beta"], execute_fn)
        assert results[0]["result"] == "ok"
        assert results[1]["result"] is None
        assert "boom" in results[1]["error"]
        verdict = await p.verify("goal", results)
        assert "beta" in verdict and "FAILED" in verdict
    asyncio.run(run())


def test_planner_run_goal_dry_run():
    from agent_kernel import Planner

    async def run():
        async def plan_fn(prompt):
            if prompt.startswith("The goal was:"):
                return "DONE — all steps hold."
            return "1. alpha\n2. beta\n"
        p = Planner(plan_fn)
        verdict, results = await p.run_goal("test goal", execute_fn=None)
        assert "DONE" in verdict
        assert results == []
    asyncio.run(run())
