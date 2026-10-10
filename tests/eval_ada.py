"""Ada eval set — the regression law (29 assertions, all offline).

Every change to the brain, memory, voice, or narration must keep these green.
All OFFLINE (no provider calls): they assert the decision layer, the data
layer, and the prompt contracts — the parts that must never silently regress.
Live latency/cost measurement belongs in logs/turns.jsonl, read after real use.

Run: .venv/bin/python -m pytest tests/eval_ada.py -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from memory import Memory  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    yield Memory(tmp_path / "eval.db")


@pytest.fixture()
def loop():
    from ada import AudioLoop  # noqa: E402
    al = AudioLoop.__new__(AudioLoop)
    al._latest_image_payload = None
    return al


# ── self-routing: FULL detector (6) ───────────────────────────────────────────
class TestSelfRouting:
    def test_bare_full_escalates(self, loop):
        assert loop._is_full_reply("FULL") is True

    def test_full_with_reason_escalates(self, loop):
        assert loop._is_full_reply("FULL: needs the browser") is True

    def test_tagged_full_escalates(self, loop):
        assert loop._is_full_reply("[emotion:confident] FULL") is True

    def test_full_moon_is_answer(self, loop):
        assert loop._is_full_reply("Full moon tonight, Sir.") is False

    def test_greeting_is_answer(self, loop):
        assert loop._is_full_reply("Yes, Sir? [emotion:warm]") is False

    def test_empty_is_answer(self, loop):
        assert loop._is_full_reply("") is False


# ── temporal memory (4) ──────────────────────────────────────────────────────
class TestTemporalMemory:
    def test_supersede_links(self, mem):
        mem.remember_fact("t_drink", "tea", importance=3)
        mem.remember_fact("t_drink", "coffee", importance=3)
        rows = mem.store.query(
            "SELECT superseded_by FROM core_facts WHERE key='t_drink' AND t_invalid IS NOT NULL")
        assert rows and rows[0]["superseded_by"] is not None

    def test_live_block_hides_old(self, mem):
        mem.remember_fact("t_drink", "tea", importance=3)
        mem.remember_fact("t_drink", "coffee", importance=3)
        assert "tea" not in mem.core_block() and "coffee" in mem.core_block()

    def test_changed_block_phrases_history(self, mem):
        mem.remember_fact("t_drink", "tea", importance=3)
        mem.remember_fact("t_drink", "coffee", importance=3)
        block = mem.changed_block()
        assert "tea" in block and "coffee" in block and "USED to be true" in block

    def test_forget_retires_not_deletes(self, mem):
        mem.remember_fact("t_phase", "beta", importance=3)
        mem.forget("t_phase")
        assert "beta" not in mem.core_block()
        assert "beta" in mem.changed_block() and "retired" in mem.changed_block()


# ── identity blocks (3) ──────────────────────────────────────────────────────
class TestIdentity:
    def test_blocks_render(self, mem):
        mem.remember_fact("self_taste", "brutalism", importance=9, pinned=True)
        mem.remember_fact("sir_goal", "jarvis", importance=9, pinned=True)
        assert "brutalism" in mem.self_block() and "jarvis" in mem.sir_block()

    def test_blocks_empty_by_default(self, mem):
        assert mem.self_block() == "" and mem.sir_block() == ""

    def test_correction_supersedes_line(self, mem):
        mem.remember_fact("sir_drink", "tea", importance=9, pinned=True)
        mem.remember_fact("sir_drink", "coffee", importance=9, pinned=True)
        assert "coffee" in mem.sir_block() and "tea" not in mem.sir_block()


# ── compaction (2) ───────────────────────────────────────────────────────────
class TestCompaction:
    def _contents(self, n):
        from google.genai import types  # noqa: E402
        return [types.Content(role="user" if i % 2 == 0 else "model",
                              parts=[types.Part(text=f"turn {i}")]) for i in range(n)]

    def test_short_passthrough(self, loop):
        assert len(loop.compact_contents(self._contents(10))) == 10

    def test_long_compacts_with_note(self, loop):
        out = loop.compact_contents(self._contents(60))
        assert len(out) == 1 + loop.COMPACT_KEEP
        assert out[0].parts[0].text.startswith("[CONVERSATION SO FAR")


# ── vision freshness (3) ─────────────────────────────────────────────────────
class TestVisionFreshness:
    def test_no_frame(self, loop):
        assert loop._fresh_frame() is False

    def test_stale_frame(self, loop):
        loop._latest_image_payload = {"mime_type": "image/jpeg", "data": "abc",
                                      "ts": time.time() - 300}
        assert loop._fresh_frame() is False

    def test_fresh_frame(self, loop):
        loop._latest_image_payload = {"mime_type": "image/jpeg", "data": "abc",
                                      "ts": time.time()}
        assert loop._fresh_frame() is True


# ── offline lane (3) ─────────────────────────────────────────────────────────
class TestOffline:
    def test_greeting_offline(self, loop):
        from agent_kernel import AgentKernel  # noqa: E402
        import tempfile
        loop.kernel = AgentKernel(root_dir=tempfile.mkdtemp())
        assert "Sir" in loop._offline_reply("hey ada")

    def test_clock_offline(self, loop):
        from agent_kernel import AgentKernel  # noqa: E402
        import tempfile
        loop.kernel = AgentKernel(root_dir=tempfile.mkdtemp())
        assert "offline" in loop._offline_reply("what time is it").lower()

    def test_unknown_offline_degrades_spoken(self, loop):
        from agent_kernel import AgentKernel  # noqa: E402
        import tempfile
        loop.kernel = AgentKernel(root_dir=tempfile.mkdtemp())
        assert "internet" in loop._offline_reply("explain quantum foam").lower()


# ── narration contracts (4) ──────────────────────────────────────────────────
class TestNarration:
    def test_web_beats_match_engine_lines(self):
        from ada import milestone_beat_for  # noqa: E402
        assert milestone_beat_for("web agent", "Web Agent Initialized")
        assert milestone_beat_for("web agent", "Executed: click ok")
        assert milestone_beat_for("web agent", "Task Finished")
        assert milestone_beat_for("web agent", "some random log") == ""

    def test_watchdog_and_milestone_exist(self, loop):
        assert hasattr(loop, "_board_watchdog") and hasattr(loop, "_milestone_for")

    def test_ack_lines_cover_background_jobs(self):
        from ada import JOB_ACK_LINES  # noqa: E402
        for key in ("CAD build", "web agent", "desktop autopilot"):
            assert key in JOB_ACK_LINES and "Sir" in JOB_ACK_LINES[key]

    def test_tool_schemas_convert_cleanly(self):
        from ada import AudioLoop as AL  # noqa: E402
        tools = AL._openai_tools()
        assert len(tools) > 40
        assert all(t.get("type") == "function" and t["function"].get("name") for t in tools)

    def test_lean_subset_covers_everyday(self):
        from ada import AudioLoop as AL  # noqa: E402
        AL._tools_cache.clear()
        lean = AL._openai_tools(lean=True)
        names = {t["function"]["name"] for t in lean}
        assert len(lean) < len(AL._openai_tools())
        for must in ("remember", "recall", "search_web", "run_web_agent",
                     "generate_cad", "create_reminder"):
            assert must in names

    def test_salvage_parses_pseudo_xml(self):
        from ada import AudioLoop as AL  # noqa: E402
        clean, salv = AL._strip_backend_leak(
            "On it. <tool_call> <function=run_web_agent> "
            "<parameter=prompt>find watches</parameter> </function> </tool_call>")
        assert salv == [("run_web_agent", {"prompt": "find watches"})]
        assert "<tool_call" not in clean

    def test_salvage_ignores_unknown_tools(self):
        from ada import AudioLoop as AL  # noqa: E402
        clean, salv = AL._strip_backend_leak(
            "Hi <function=nuke_world> <parameter=x>y</parameter> </function> bye")
        assert salv == [] and "<function" not in clean and "Hi" in clean

    def test_open_block_holds_not_voices(self):
        from ada import AudioLoop as AL  # noqa: E402
        assert AL._has_open_leak("On it <tool_call> <function=run_web") is True
        assert AL._has_open_leak("done <tool_call>x</tool_call> ok") is False
