"""Tests for Feature 6 — learning gated by evaluation.

Covers the eval ladder commit path, the shadow/canary/active serving modes,
voice approval for high-blast-radius skills, rollback, and eval-derived
dream importance. All tmp dirs; the registry DB is runtime state, never
committed.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import skill_eval
from skill_eval import SkillEvalGate, SkillRegistry


CLEAN_CODE = (
    "def celsius_to_f(celsius: float) -> float:\n"
    "    \"\"\"Convert Celsius to Fahrenheit (pure function).\"\"\"\n"
    "    return celsius * 9.0 / 5.0 + 32.0\n"
)

DIRTY_CODE = (
    "import os\n"
    "def wipe():\n"
    "    os.remove('/tmp/x')\n"
    "    return 1\n"
)


@pytest.fixture()
def forge(tmp_path):
    from mcp.forge import MCPForge
    custom = tmp_path / "custom_tools"
    mcpd = tmp_path / "mcp_tools"
    custom.mkdir()
    mcpd.mkdir()
    return MCPForge(custom, mcpd)


@pytest.fixture()
def gate(tmp_path, forge):
    g = SkillEvalGate(tmp_path / "state", tmp_path / "mcp_tools")
    g.attach_forge(forge)
    yield g
    g.registry.close()


def _module(forge, tmp_path, name, code):
    made = forge.synthesise(name, code)
    assert "created" in made, made
    return forge.mcp_dir / f"{name}.py"


def _green_subset(monkeypatch):
    monkeypatch.setattr(skill_eval, "run_eval_subset",
                        lambda repo_root, timeout=180: (True, "8 passed"))


# ── commit path ───────────────────────────────────────────────────────────────

def test_candidate_blocked_when_l0_fails(gate, forge, tmp_path):
    # A module that slipped past synthesis (hand-edited hot-load): the L0
    # AST check must still block the candidate.
    mod = tmp_path / "mcp_tools" / "wipe.py"
    mod.write_text(DIRTY_CODE)
    verdict = gate.evaluate_candidate("wipe", "delete temp files", DIRTY_CODE,
                                      {}, mod, repo_root=tmp_path)
    assert verdict["status"] == "blocked"
    assert verdict["stage"] == "L0"
    assert "ast_gate" in verdict["failed"]
    assert gate.registry.get("wipe", verdict["version"]) is None


def test_candidate_blocked_when_eval_subset_fails(gate, forge, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(skill_eval, "run_eval_subset",
                        lambda repo_root, timeout=180: (False, "2 failed"))
    mod = _module(forge, tmp_path, "celsius_to_f", CLEAN_CODE)
    verdict = gate.evaluate_candidate("celsius_to_f", "temp conversion",
                                      CLEAN_CODE, {"celsius": 100.0},
                                      mod, repo_root=tmp_path)
    assert verdict["status"] == "blocked"
    assert verdict["stage"] == "L2"
    assert gate.registry.get("celsius_to_f", verdict["version"]) is None


def test_all_green_registers_candidate_and_auto_promotes_to_shadow(
        gate, forge, tmp_path, monkeypatch):
    _green_subset(monkeypatch)
    mod = _module(forge, tmp_path, "celsius_to_f", CLEAN_CODE)
    verdict = gate.evaluate_candidate("celsius_to_f", "temp conversion",
                                      CLEAN_CODE, {"celsius": 100.0},
                                      mod, repo_root=tmp_path)
    assert verdict["status"] == "shadow", verdict
    assert verdict["blast_radius"] == "low"
    row = gate.registry.get("celsius_to_f", verdict["version"])
    assert row["status"] == "shadow"
    assert json.loads(row["eval_scores_json"])["passed"] is True
    # versioned dir archived
    assert (forge.mcp_dir / "celsius_to_f"
            / f"v{verdict['version']}" / "tool.py").exists()


# ── shadow serving ────────────────────────────────────────────────────────────

def _seed_two_versions(gate, forge, tmp_path):
    """v1 active (returns a+b), v2 shadow (returns a+b+100)."""
    v1 = ("def adder(a: int, b: int) -> int:\n"
          "    \"\"\"Add two numbers.\"\"\"\n"
          "    return a + b\n")
    v2 = ("def adder(a: int, b: int) -> int:\n"
          "    \"\"\"Add two numbers (candidate).\"\"\"\n"
          "    return a + b + 100\n")
    _module(forge, tmp_path, "adder", v1)
    gate.registry.register("adder", 1, status="active",
                           scores={"passed": True},
                           code_ref=str(skill_eval.archive_version(
                               forge.mcp_dir, "adder", 1,
                               forge.mcp_dir / "adder.py")),
                           note="seed")
    gate.registry.set_active("adder", 1, reason="seed")
    skill_eval.archive_version(
        forge.mcp_dir, "adder", 2,
        _write_tmp_module(tmp_path, "adder_v2", v2))
    gate.registry.register("adder", 2, status="shadow",
                           scores={"passed": True},
                           code_ref=str(skill_eval.version_dir(
                               forge.mcp_dir, "adder", 2) / "tool.py"),
                           note="seed")


def _write_tmp_module(tmp_path, name, code):
    p = tmp_path / f"{name}.py"
    p.write_text(code)
    return p


def test_shadow_logs_but_does_not_serve(gate, forge, tmp_path, monkeypatch):
    _seed_two_versions(gate, forge, tmp_path)
    monkeypatch.setenv("ADA_SKILL_SHADOW", "1")
    try:
        result = gate.serve("adder", {"a": 1, "b": 2})
    finally:
        monkeypatch.delenv("ADA_SKILL_SHADOW", raising=False)
    assert result == 3  # active v1 served, NOT the shadow's 103
    rows = gate.registry._db.execute(
        "SELECT kind, ok, err FROM served_runs WHERE name='adder'"
        " AND kind='shadow'").fetchall()
    assert len(rows) == 1
    assert rows[0]["ok"] == 1
    assert "diverged" in rows[0]["err"]  # shadow disagreed, logged not shown


# ── canary ────────────────────────────────────────────────────────────────────

def _seed_canary(gate, canary_ok_runs, canary_bad_runs,
                 active_ok_runs, active_bad_runs):
    for _ in range(canary_ok_runs):
        gate.registry.log_run("adder", 2, "canary", True)
    for _ in range(canary_bad_runs):
        gate.registry.log_run("adder", 2, "canary", False, "boom")
    for _ in range(active_ok_runs):
        gate.registry.log_run("adder", 1, "active", True)
    for _ in range(active_bad_runs):
        gate.registry.log_run("adder", 1, "active", False, "flaky")


def test_canary_rollback_on_error_regression(gate, forge, tmp_path):
    _seed_two_versions(gate, forge, tmp_path)
    gate.registry.set_status("adder", 2, "canary", reason="test")
    _seed_canary(gate, canary_ok_runs=2, canary_bad_runs=8,
                 active_ok_runs=10, active_bad_runs=0)
    out = gate.maybe_promote_canary("adder", 2, min_samples=10)
    assert out["action"] == "rolled_back", out
    assert gate.registry.get("adder", 2)["status"] == "shadow"
    assert gate.registry.active_version("adder") == "1"


def test_canary_clean_promotes_to_active(gate, forge, tmp_path):
    _seed_two_versions(gate, forge, tmp_path)
    gate.registry.set_status("adder", 2, "canary", reason="test")
    _seed_canary(gate, canary_ok_runs=10, canary_bad_runs=0,
                 active_ok_runs=10, active_bad_runs=0)
    out = gate.maybe_promote_canary("adder", 2, min_samples=10)
    assert out["ok"] and out["status"] == "active", out
    assert gate.registry.active_version("adder") == "2"
    assert gate.registry.get("adder", 1)["status"] == "retired"


def test_canary_demoted_on_sustained_useless_votes(gate, forge, tmp_path):
    from observe import ObserveTiers
    _seed_two_versions(gate, forge, tmp_path)
    gate.registry.set_status("adder", 2, "canary", reason="test")
    obs = ObserveTiers(tmp_path / "ostate")
    gate.attach_observe(obs)
    for _ in range(3):
        obs.skill_vote("adder", False)
    assert obs.skill_signal("adder") == "demote"
    _seed_canary(gate, canary_ok_runs=10, canary_bad_runs=0,
                 active_ok_runs=10, active_bad_runs=0)
    out = gate.maybe_promote_canary("adder", 2, min_samples=10)
    assert out["action"] == "demoted_to_shadow", out
    assert gate.registry.get("adder", 2)["status"] == "shadow"


# ── voice approval for high blast radius ──────────────────────────────────────

EMAIL_CODE = (
    "def draft_reminder_email(to: str) -> str:\n"
    "    \"\"\"Draft a reminder email body (pure text shaping).\"\"\"\n"
    "    return 'Reminder for ' + to\n"
)


def test_high_blast_radius_requires_voice_approval(gate, forge, tmp_path,
                                                  monkeypatch):
    _green_subset(monkeypatch)
    mod = _module(forge, tmp_path, "draft_reminder_email", EMAIL_CODE)
    verdict = gate.evaluate_candidate(
        "draft_reminder_email", "send reminder emails to Sir", EMAIL_CODE,
        {"to": "Sir"}, mod, repo_root=tmp_path)
    assert verdict["blast_radius"] == "high"
    assert verdict["status"] == "shadow"
    assert "voice_proposal" in verdict  # enqueued, surfaced by voice

    # auto-promote past canary is REFUSED for high blast radius
    gate.registry.set_status("draft_reminder_email", verdict["version"],
                             "canary", reason="test")
    refused = gate.promote("draft_reminder_email", verdict["version"], "active")
    assert refused["ok"] is False
    assert refused["error"] == "voice_approval_required"
    # NOT promoted: still canary, never active
    assert gate.registry.get(
        "draft_reminder_email", verdict["version"])["status"] == "canary"

    # the voice queue holds the pending item for the brain to ask out loud
    pending = gate.voice.next_voice_proposal()
    assert pending and pending["name"] == "draft_reminder_email"


def test_voice_approval_activates_skill(gate, forge, tmp_path, monkeypatch):
    from skill_loop import SkillLoop
    _green_subset(monkeypatch)
    mod = _module(forge, tmp_path, "draft_reminder_email", EMAIL_CODE)
    verdict = gate.evaluate_candidate(
        "draft_reminder_email", "send reminder emails to Sir", EMAIL_CODE,
        {"to": "Sir"}, mod, repo_root=tmp_path)
    ver = verdict["version"]
    gate.registry.set_status("draft_reminder_email", ver, "canary",
                             reason="test")

    kernel = SimpleNamespace(forge=forge)
    loop = SkillLoop(kernel, chat_fn=None, state_dir=tmp_path / "state2")
    # point this loop at the same gate (same state dir) the skill was
    # evaluated under
    loop._eval_gate = gate

    spoken = loop.resolve_voice_proposal("draft_reminder_email", approved=True)
    assert "live" in spoken.lower()
    assert gate.registry.active_version("draft_reminder_email") == str(ver)
    assert gate.registry.get("draft_reminder_email", ver)["status"] == "active"

    # and a rejection retires instead
    mod2 = _module(forge, tmp_path, "draft_reminder_email2",
                   EMAIL_CODE.replace("draft_reminder_email",
                                      "draft_reminder_email2"))
    v2 = gate.evaluate_candidate(
        "draft_reminder_email2", "send reminder emails to Sir",
        EMAIL_CODE.replace("draft_reminder_email", "draft_reminder_email2"),
        {"to": "Sir"}, mod2, repo_root=tmp_path)
    spoken_no = loop.resolve_voice_proposal("draft_reminder_email2",
                                            approved=False)
    assert "retire" in spoken_no.lower()
    assert gate.registry.get("draft_reminder_email2",
                             v2["version"])["status"] == "retired"


# ── rollback ──────────────────────────────────────────────────────────────────

def test_rollback_flips_active_pointer(gate, forge, tmp_path):
    _seed_two_versions(gate, forge, tmp_path)
    gate.registry.set_status("adder", 2, "canary", reason="test")
    gate._go_active("adder", 2, reason="test promote")
    assert gate.registry.active_version("adder") == "2"
    out = gate.rollback("adder", reason="v2 misbehaves")
    assert out["ok"] and out["to"] == "1" and out["from"] == "2"
    assert gate.registry.active_version("adder") == "1"
    assert gate.registry.get("adder", 2)["status"] == "retired"


# ── dream importance from eval signal ─────────────────────────────────────────

def test_dream_importance_scales_with_eval_signal(gate, tmp_path):
    from dream import DreamRunner
    gate.registry.register("celsius_to_f", 1, status="active",
                           scores={"passed": True, "l2_subset": "8 passed"},
                           note="seed")
    gate.registry.set_active("celsius_to_f", 1, reason="seed")
    kernel = SimpleNamespace(skills=SimpleNamespace(eval_gate=gate))
    dreamer = DreamRunner(kernel, chat_fn=None,
                          state_dir=tmp_path / "dreamstate")

    imp, pinned = dreamer._lesson_confidence(
        "sir celsius_to_f the celsius_to_f skill converts temps reliably")
    assert (imp, pinned) == (9, True)

    imp2, pinned2 = dreamer._lesson_confidence(
        "sir drink Sir prefers tea in the morning")
    assert (imp2, pinned2) == (5, False)  # unvalidated: low, unpinned


def test_dream_confidence_without_gate_falls_back_low(tmp_path):
    from dream import DreamRunner
    kernel = SimpleNamespace(skills=SimpleNamespace(eval_gate=None))
    dreamer = DreamRunner(kernel, chat_fn=None,
                          state_dir=tmp_path / "dreamstate2")
    assert dreamer._lesson_confidence("anything at all") == (5, False)


# ── end-to-end: the skill loop runs exactly the marked subset ─────────────────

def test_rep_now_runs_marked_subset_end_to_end(forge, tmp_path, monkeypatch):
    """Full rep with the REAL eval subset (not mocked): proves the loop runs
    exactly the @pytest.mark.skill_eval slice."""
    from skill_loop import SkillLoop
    state = tmp_path / "loopstate"
    kernel = SimpleNamespace(forge=forge)
    proposal = (
        "NAME: celsius_to_f\n"
        "PURPOSE: convert Celsius to Fahrenheit\n"
        "TEST_CALL: {\"celsius\": 100.0}\n"
        "CODE:\n```python\n" + CLEAN_CODE + "```\n"
    )

    async def chat_fn(prompt):
        return proposal

    loop = SkillLoop(kernel, chat_fn=chat_fn, state_dir=state)
    # pin the eval subset to the real repo root (state_dir is tmp here)
    monkeypatch.setattr(loop, "_repo_root",
                        lambda: Path(__file__).parent.parent)
    result = asyncio.run(loop.rep_now(force=True))
    assert "shadow" in result, result
    row = loop.eval_gate.registry.get("celsius_to_f", 1)
    assert row is not None and row["status"] == "shadow"
    scores = json.loads(row["eval_scores_json"])
    assert scores["passed"] is True
    assert "l2_subset" in scores  # the marked slice ran
    loop.eval_gate.registry.close()
