"""F0 — real autonomy gates (decorative ladder fix, D2-D4).

Covers: default level + constructor-never-arms, the arm ritual (challenge /
90s confirm / 8h expiry / stand_down), level x risk-class gating for
background tools, risk-class mapping (unknown tools fail closed), the
voice-confirmation gate in ada.py::_check_confirmation (mocked voice ask),
the filesystem jail, and a D4 regression: a write outside the jail is
refused even when confirmation is bypassed/approved.
"""
import asyncio
import json
import sys
import time
import types
from pathlib import Path

# `playwright` is stubbed in tests/conftest.py (missing optional dep in this
# env) so the REAL ada.py — which carries the fs jail and the gate — can be
# imported and exercised.
BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

import autonomy
from autonomy import (AutonomyLevels, risk_class, is_ladder_exempt,
                      ARM_EXPIRY_S, CONFIRM_TIMEOUT_S,
                      RISK_READONLY, RISK_REVERSIBLE,
                      RISK_DESTRUCTIVE, RISK_EXTERNAL)
import ada


# ── helpers ──────────────────────────────────────────────────────────────────

def fresh_autonomy(tmp_path, **kw):
    return AutonomyLevels(str(tmp_path / "state"), **kw)


def arm_fully(aut):
    """Run the complete spoken ritual: challenge + confirm inside the window."""
    assert aut.request_arm().startswith("Understood")
    msg = aut.confirm_arm()
    assert "ARMED" in msg
    assert aut.level == 3 and aut.armed
    return aut


class FakeAudit:
    def __init__(self):
        self.rows = []
    def log(self, actor, action, target="", decision="allow", detail="", **kw):
        self.rows.append({"actor": actor, "action": action, "target": target,
                          "decision": decision, "detail": detail, **kw})


def make_loop(tmp_path, voice_answer=True):
    """A real ada.AudioLoop gate with everything heavy stubbed out."""
    loop = ada.AudioLoop.__new__(ada.AudioLoop)
    audit = FakeAudit()
    aut = AutonomyLevels(str(tmp_path / "aut"), audit=audit.log)
    kernel = types.SimpleNamespace(
        autonomy=aut,
        security=types.SimpleNamespace(audit=audit),
    )
    loop.kernel = kernel
    asked = []

    async def fake_ask(question):
        asked.append(question)
        return voice_answer

    loop._ask_user_confirmation = fake_ask
    return loop, aut, asked, audit


class FakePM:
    def __init__(self, root):
        self._root = Path(root)
    def get_current_project_path(self):
        return self._root


def make_fs(tmp_path):
    """Bind the real fs handlers to a stub self (project jail only)."""
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    notified = []
    stub = types.SimpleNamespace(
        project_manager=FakePM(proj),
        kernel=types.SimpleNamespace(
            security=types.SimpleNamespace(
                sanitize=lambda text: (text, False))),
    )
    stub._is_sensitive_path = ada.AudioLoop._is_sensitive_path
    stub._resolve_fs_path = ada.AudioLoop._resolve_fs_path.__get__(stub)

    async def fake_notify(result):
        notified.append(result)
    stub._notify_model = fake_notify
    stub.handle_write_file = ada.AudioLoop.handle_write_file.__get__(stub)
    stub.handle_read_file = ada.AudioLoop.handle_read_file.__get__(stub)
    stub.handle_read_directory = ada.AudioLoop.handle_read_directory.__get__(stub)
    return stub, proj, notified


def run(coro):
    return asyncio.run(coro)


# ── the ladder itself ─────────────────────────────────────────────────────────

def test_default_level_is_standard_and_never_armed(tmp_path):
    aut = fresh_autonomy(tmp_path)
    assert aut.level == 1
    assert aut.armed is False
    assert aut.allows_autonomous_destructive() is False
    # background contexts have no voice channel: nothing destructive runs
    assert aut.allows_background_tool("send_email") is False
    assert aut.allows_background_tool("desktop_autopilot") is False


def test_destructive_refused_at_levels_0_to_2_no_exceptions(tmp_path):
    aut = fresh_autonomy(tmp_path)
    for lvl in (0, 1, 2):
        aut.set_level(lvl)
        assert aut.allows_autonomous_destructive() is False, f"L{lvl}"
        assert aut.allows_background_tool("send_email") is False, f"L{lvl}"
        assert aut.allows_background_tool("desktop_autopilot") is False, f"L{lvl}"
        assert aut.allows_background_tool("create_custom_tool") is False, f"L{lvl}"
        assert aut.allows_background_tool("run_web_agent") is False, f"L{lvl}"


def test_confirm_without_challenge_does_nothing(tmp_path):
    aut = fresh_autonomy(tmp_path)
    msg = aut.confirm_arm()
    assert "Nothing to confirm" in msg
    assert aut.level == 1
    assert aut.allows_autonomous_destructive() is False


def test_arm_ritual_arms_and_allows(tmp_path):
    aut = arm_fully(fresh_autonomy(tmp_path))
    assert aut.allows_autonomous_destructive() is True
    assert aut.allows_background_tool("send_email") is True


def test_arm_confirm_after_timeout_expires(tmp_path):
    aut = fresh_autonomy(tmp_path)
    aut.request_arm()
    aut._arm_pending_ts -= (CONFIRM_TIMEOUT_S + 10)
    msg = aut.confirm_arm()
    assert "expired" in msg
    assert aut.level == 1
    assert aut.allows_autonomous_destructive() is False


def test_arm_expiry_revokes_back_to_standard(tmp_path):
    aut = arm_fully(fresh_autonomy(tmp_path))
    aut._armed_ts -= (ARM_EXPIRY_S + 10)   # the 8h window elapsed
    assert aut.armed is False
    assert aut.allows_autonomous_destructive() is False
    assert aut.level == 1
    assert aut.allows_background_tool("send_email") is False


def test_stand_down_revokes_immediately(tmp_path):
    aut = arm_fully(fresh_autonomy(tmp_path))
    msg = aut.stand_down()
    assert "Standing down" in msg
    assert aut.level == 1
    assert aut._armed_ts is None
    assert aut.armed is False
    assert aut.allows_autonomous_destructive() is False
    assert aut.allows_background_tool("send_email") is False


def test_legacy_full_auto_state_file_cannot_resurrect_armed_l3(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # what the old full-auto constructor used to leave behind: L3 + armed_ts
    (state_dir / "autonomy_state.json").write_text(
        json.dumps({"level": 3, "armed_ts": time.time()}))
    aut = AutonomyLevels(str(state_dir))
    assert aut.level == 1
    assert aut.allows_autonomous_destructive() is False


def test_new_era_armed_state_survives_restart_inside_window(tmp_path):
    aut = arm_fully(fresh_autonomy(tmp_path))   # persists era=ladder-v2
    aut2 = AutonomyLevels(str(tmp_path / "state"))
    assert aut2.level == 3
    assert aut2.allows_autonomous_destructive() is True


def test_expired_persisted_arm_drops_to_standard(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "autonomy_state.json").write_text(
        json.dumps({"era": "ladder-v2", "level": 3,
                    "armed_ts": time.time() - ARM_EXPIRY_S - 5}))
    aut = AutonomyLevels(str(state_dir))
    assert aut.level == 1
    assert aut.allows_autonomous_destructive() is False


def test_ritual_transitions_are_audit_logged(tmp_path):
    audit = FakeAudit()
    aut = AutonomyLevels(str(tmp_path / "state"), audit=audit.log)
    aut.request_arm()
    aut.confirm_arm()
    aut.stand_down()
    actions = [r["action"] for r in audit.rows]
    assert "arm-request" in actions
    assert "arm-confirm" in actions
    assert "stand-down" in actions


# ── risk classes ──────────────────────────────────────────────────────────────

def test_risk_class_mapping():
    assert risk_class("send_email") == RISK_EXTERNAL
    assert risk_class("smart_home") == RISK_EXTERNAL
    assert risk_class("launch_app") == RISK_EXTERNAL
    assert risk_class("run_web_agent") == RISK_EXTERNAL
    assert risk_class("calendar_event") == RISK_EXTERNAL
    assert risk_class("print_it") == RISK_EXTERNAL
    assert risk_class("fabricate") == RISK_EXTERNAL
    assert risk_class("forget") == RISK_DESTRUCTIVE
    assert risk_class("forget_complete") == RISK_DESTRUCTIVE
    assert risk_class("create_custom_tool") == RISK_DESTRUCTIVE
    assert risk_class("run_custom_tool") == RISK_DESTRUCTIVE
    assert risk_class("desktop_autopilot") == RISK_DESTRUCTIVE
    assert risk_class("write_file") == RISK_REVERSIBLE
    assert risk_class("create_reminder") == RISK_REVERSIBLE
    assert risk_class("remember") == RISK_REVERSIBLE
    assert risk_class("search_web") == RISK_READONLY
    assert risk_class("read_file") == RISK_READONLY
    assert risk_class("read_directory") == RISK_READONLY
    assert risk_class("audit_log") == RISK_READONLY


def test_unknown_tools_fail_closed():
    # anything the ladder does not recognise is treated as destructive:
    # voice-confirmed below armed L3, never silent
    assert risk_class("mystery_tool_xyz") == RISK_DESTRUCTIVE
    assert risk_class("") == RISK_DESTRUCTIVE


def test_ladder_exemptions():
    assert is_ladder_exempt("edit_part") is True
    assert is_ladder_exempt("save_version") is True
    assert is_ladder_exempt("autonomy_control") is True
    assert is_ladder_exempt("send_email") is False


def test_background_gated_by_level_x_risk(tmp_path):
    aut = fresh_autonomy(tmp_path)
    # L1: read-only only
    aut.set_level(1)
    assert aut.allows_background_tool("search_web") is True
    assert aut.allows_background_tool("read_file") is True
    assert aut.allows_background_tool("write_file") is False
    assert aut.allows_background_tool("create_reminder") is False
    assert aut.allows_background_tool("mystery_tool_xyz") is False
    # L2: non-destructive flows
    aut.set_level(2)
    assert aut.allows_background_tool("write_file") is True
    assert aut.allows_background_tool("create_reminder") is True
    assert aut.allows_background_tool("send_email") is False
    assert aut.allows_background_tool("forget") is False
    # L0: nothing ungated
    aut.set_level(0)
    assert aut.allows_background_tool("search_web") is False
    assert aut.allows_background_tool("write_file") is False
    # CAD domain is outside the ladder even at L0
    assert aut.allows_background_tool("edit_part") is True
    # armed L3: everything
    arm_fully(aut)
    assert aut.allows_background_tool("send_email") is True
    assert aut.allows_background_tool("forget") is True


# ── the live dispatch gate (ada.py::_check_confirmation) ──────────────────────

def test_confirmation_fires_for_side_effect_tools(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=True)
    aut.set_level(1)
    assert run(loop._check_confirmation("send_email",
                                        {"to": "a@b.c", "subject": "hi"})) is True
    assert asked, "voice ask never fired for send_email"
    assert "a@b.c" in asked[0], "no spoken read-back of the recipient"
    asked.clear()
    assert run(loop._check_confirmation("write_file",
                                        {"path": "notes/x.txt"})) is True
    assert asked and "notes/x.txt" in asked[0]
    asked.clear()
    assert run(loop._check_confirmation("desktop_autopilot",
                                        {"goal": "open settings"})) is True
    assert asked, "voice ask never fired for desktop_autopilot"
    asked.clear()
    assert run(loop._check_confirmation("launch_app", {})) is True
    assert asked, "voice ask never fired for launch_app"


def test_voice_denial_blocks_the_tool(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=False)
    aut.set_level(1)
    assert run(loop._check_confirmation("send_email", {"to": "a@b.c"})) is False
    denies = [r for r in audit.rows
              if r["action"] == "tool-gate" and r["decision"] == "denied"]
    assert denies, "denial was not audit-logged"


def test_readonly_tools_stay_frictionless_at_l1(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=True)
    aut.set_level(1)
    assert run(loop._check_confirmation("search_web", {"query": "x"})) is True
    assert run(loop._check_confirmation("read_file", {"path": "x"})) is True
    assert asked == [], "read-only tools must not ask at L1"


def test_l0_asks_even_for_readonly(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=True)
    aut.set_level(0)
    assert run(loop._check_confirmation("search_web", {"query": "x"})) is True
    assert asked, "training wheels must ask for everything"


def test_l2_reversible_flows_without_asking_but_destructive_asks(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=True)
    aut.set_level(2)
    assert run(loop._check_confirmation("write_file", {"path": "x"})) is True
    assert asked == [], "reversible tools must flow at L2"
    assert run(loop._check_confirmation("send_email", {"to": "a@b.c"})) is True
    assert asked, "external tools must still ask at L2"


def test_armed_instant_kill_skips_nagging_but_discloses(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=False)
    arm_fully(aut)
    assert run(loop._check_confirmation("send_email", {"to": "a@b.c"})) is True
    assert asked == [], "armed Instant Kill must not nag"
    disclosed = [r for r in audit.rows
                 if r["action"] == "tool-gate" and "armed Instant Kill" in r["detail"]]
    assert disclosed, "armed auto-approve was not disclosed in the audit log"


def test_cad_tools_outside_the_ladder(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=False)
    aut.set_level(0)
    assert run(loop._check_confirmation("edit_part", {"target": "p1"})) is True
    assert asked == [], "CAD tools must never be gated"


def test_every_gate_decision_is_audit_logged(tmp_path):
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=True)
    aut.set_level(1)
    run(loop._check_confirmation("search_web", {}))
    run(loop._check_confirmation("send_email", {}))
    gated = [r for r in audit.rows if r["action"] == "tool-gate"]
    assert len(gated) == 2
    assert all(r["decision"] == "approved" for r in gated)


# ── filesystem jail (D3) + the D4 chained-exploit regression ──────────────────

def test_jail_refuses_parent_traversal_write(tmp_path):
    stub, proj, notified = make_fs(tmp_path)
    run(stub.handle_write_file("../escape.txt", "pwned"))
    assert notified and "refused" in notified[0].lower()
    assert not (tmp_path / "escape.txt").exists()


def test_jail_refuses_absolute_write_outside_project(tmp_path):
    stub, proj, notified = make_fs(tmp_path)
    outside = tmp_path / "outside.txt"
    run(stub.handle_write_file(str(outside), "pwned"))
    assert notified and "refused" in notified[0].lower()
    assert not outside.exists()


def test_jail_refuses_sensitive_read(tmp_path):
    stub, proj, notified = make_fs(tmp_path)
    key = tmp_path / ".ssh" / "id_rsa"
    key.parent.mkdir(exist_ok=True)
    key.write_text("PRIVATE KEY")
    run(stub.handle_read_file(str(key)))
    assert notified and "refused" in notified[0].lower()


def test_jail_allows_write_inside_project(tmp_path):
    stub, proj, notified = make_fs(tmp_path)
    run(stub.handle_write_file("notes/todo.txt", "buy milk"))
    target = proj / "notes" / "todo.txt"
    assert target.exists()
    assert target.read_text() == "buy milk"


def test_d4_write_outside_jail_refused_even_if_confirmation_bypassed(tmp_path):
    """D4 regression: the jail is a second, independent line of defence.
    Even if the voice confirmation were bypassed (attacker gets a 'yes'),
    the filesystem handler itself refuses a write outside the project."""
    loop, aut, asked, audit = make_loop(tmp_path, voice_answer=True)
    aut.set_level(1)
    # 1. confirmation "succeeds" (bypassed / socially engineered yes)
    assert run(loop._check_confirmation("write_file",
                                        {"path": "../../evil.sh"})) is True
    assert asked, "confirmation must fire for write_file (D2)"
    # 2. the handler STILL refuses the jail escape (D3) — the chain is dead
    stub, proj, notified = make_fs(tmp_path)
    run(stub.handle_write_file("../../evil.sh", "malicious"))
    assert notified and "refused" in notified[0].lower()
    assert not (tmp_path / "evil.sh").exists()
