"""Tests for the JARVIS autonomy layer: unified memory, proactive engine,
overnight dossier state machine."""
import sys
import os
import time
import asyncio
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from agent_kernel import MemoryBank
from proactive import ProactiveEngine
from overnight_dossier import OvernightDossier


def make_memory():
    return MemoryBank(os.path.join(tempfile.mkdtemp(), "test_ada.db"))


# ── unified memory ────────────────────────────────────────────────────────────
def test_recent_exchanges_unified_chronological():
    mem = make_memory()
    mem.remember_context("what is steel", "an alloy of iron and carbon")
    time.sleep(0.01)
    mem.remember_context("and titanium", "a strong light metal")
    out = mem.recent_exchanges(k=5)
    assert len(out) == 2
    assert "steel" in out[0] and "titanium" in out[1]   # oldest first


def test_recent_exchanges_empty():
    mem = make_memory()
    assert mem.recent_exchanges() == []


# ── proactive engine gating ───────────────────────────────────────────────────
class FakeScheduler:
    def __init__(self, items):
        self._items = items
    def list(self):
        return self._items


class FakeKernel:
    def __init__(self, sched=None, dossier=None):
        self.scheduler = sched or FakeScheduler([])
        self.dossier = dossier


def make_engine(kernel, speak_fn, quiet_hours=(23, 8), max_per_hour=3):
    return ProactiveEngine(kernel, speak_fn=speak_fn, interval_s=60,
                           quiet_hours=quiet_hours, max_per_hour=max_per_hour)


def test_proactive_fires_on_upcoming_reminder():
    spoken = []
    async def speak(line):
        spoken.append(line)
        return True
    eng = make_engine(FakeKernel(FakeScheduler([
        {"id": "r1", "descr": "Call with supplier", "next_run": time.time() + 600},
    ])), speak)
    key_line = eng._reminder_warnings()
    assert key_line and "supplier" in key_line[1]


def test_proactive_ignores_distant_reminders():
    eng = make_engine(FakeKernel(FakeScheduler([
        {"id": "r2", "descr": "Far away task", "next_run": time.time() + 7200},
    ])), speak_fn=None)
    assert eng._reminder_warnings() is None


def test_proactive_cooldown_and_rate_limit():
    async def run():
        spoken = []
        async def speak(line):
            spoken.append(line)
            return True
        eng = make_engine(FakeKernel(), speak, max_per_hour=1)
        assert eng._gate("k") is True
        eng._last_fired["k"] = time.time()
        assert eng._gate("k") is False            # per-key cooldown
        assert eng._gate("other") is True
        eng._hour_stamps = [time.time()] * 3      # hour ceiling hit
        eng._hour_stamps = [time.time(), time.time(), time.time()]
        assert eng._gate("third") is False
    asyncio.run(run())


def test_proactive_quiet_hours():
    eng = make_engine(FakeKernel(), speak_fn=None, quiet_hours=(0, 23))  # always quiet
    assert eng._in_quiet_hours() is True
    eng2 = make_engine(FakeKernel(), speak_fn=None, quiet_hours=(23, 8))
    assert isinstance(eng2._in_quiet_hours(), bool)


# ── overnight dossier state machine ──────────────────────────────────────────
def make_dossier():
    d = OvernightDossier(kernel=type("K", (), {"memory": make_memory()})(),
                         chat_fn=None, search_fn=None,
                         state_dir=tempfile.mkdtemp())
    return d


def test_dossier_schedule_and_status():
    d = make_dossier()
    msg = d.schedule("build123d vs CadQuery")
    assert "scheduled" in msg.lower()
    assert d.status() == f"Dossier 'build123d vs CadQuery' — status: pending " or "pending" in d.status()


def test_dossier_ready_announces_once():
    d = make_dossier()
    d.schedule("topic A")
    d._state.update({"status": "ready", "ready_ts": time.time(), "announced": False})
    topic1, _ = d.ready_unannounced()
    topic2, _ = d.ready_unannounced()
    assert topic1 == "topic A"
    assert topic2 is None            # announced exactly once


def test_dossier_state_persists():
    import json
    d = make_dossier()
    d.schedule("persistent topic")
    d2 = OvernightDossier(kernel=d.kernel, chat_fn=None, search_fn=None,
                          state_dir=d.state_path.parent)
    assert d2._state.get("topic") == "persistent topic"


def test_dossier_research_writes_notebook():
    async def run():
        mem = make_memory()
        async def fake_chat(prompt):
            return "SHORT VERSION: build123d wins. " * 3
        d = OvernightDossier(kernel=type("K", (), {"memory": mem})(),
                             chat_fn=fake_chat, search_fn=None,
                             state_dir=tempfile.mkdtemp())
        d.schedule("test topic")
        await d._run_pending()
        assert d._state["status"] == "ready"
        nb = mem.notebook_read()
        assert any("test topic" in s for s in nb)
    asyncio.run(run())
