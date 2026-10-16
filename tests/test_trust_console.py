"""Feature 9 — decision replay + trust console.

Covers: full-fidelity audit records (args untruncated, detail 500->4000,
migration preserves legacy rows), replay.inspect() context reconstruction,
replay.redecide() policy-change detection, the trust_console widget kind
(backend validation + sanitization, frontend allow-list parity), one-click
undo of a reversible tool record, the restraint panel from seeded
policy.decision events, per-capability autonomy toggles, and the weekly
trust summary in ledger.py. Tmp dirs only — never the real runtime DBs.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import agent_kernel as ak
import autonomy
import hud_spec
import ledger
import replay
import widgets as w
from autonomy import AutonomyLevels


# ── fixtures ──────────────────────────────────────────────────────────────

def make_audit_db(tmp_path, legacy=False):
    """An audit DB. legacy=True builds the pre-F9 6-column schema + one row,
    to prove the migration preserves it."""
    db = tmp_path / "ada.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE audit (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "ts REAL, actor TEXT, action TEXT, target TEXT, decision TEXT, detail TEXT)")
    if legacy:
        conn.execute("INSERT INTO audit(ts,actor,action,target,decision,detail)"
                     " VALUES(1.0,'ada','tool-gate','write_file','approved','old row')")
        conn.commit()
    conn.close()
    return db


def make_timeline_db(tmp_path):
    tl = tmp_path / "ada_events.db"
    conn = sqlite3.connect(str(tl))
    conn.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, ts_utc REAL NOT NULL,"
                 "source TEXT NOT NULL, type TEXT NOT NULL, payload_json TEXT NOT NULL,"
                 "schema_v INTEGER NOT NULL, session_id TEXT NOT NULL)")
    conn.commit()
    conn.close()
    return tl


def add_event(tl, seq, ts, typ, payload):
    conn = sqlite3.connect(str(tl))
    conn.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?)",
                 (seq, ts, "agent", typ, json.dumps(payload), 1, "s1"))
    conn.commit()
    conn.close()


def full_gate_record(audit, ts, tool="write_file", decision="approved",
                     gate=None, tl_hash=None, wo_hash=None, args=None):
    gate = gate or {"level": 2, "risk": "state-changing-reversible",
                    "armed": False, "exempt": False,
                    "capability": "files", "capability_mode": "act",
                    "needs_voice": False, "verdict": "auto-approved",
                    "reason": "frictionless at L2 (risk=state-changing-reversible)"}
    return audit.log("ada", "tool-gate", tool, decision, gate["reason"],
                     session_id="sess1", step="tool-gate",
                     tool_args=args or {"path": "/tmp/a.txt", "content": "hello"},
                     model_lane="instant", tool_version="1.0.0",
                     prompt_hash="ph" * 32, trace_id="trace1", gate=gate,
                     world_hash=wo_hash, timeline_hash=tl_hash)


# ── 1. full-fidelity audit ────────────────────────────────────────────────

def test_full_fidelity_record_args_not_truncated(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    big_args = {"path": "/tmp/x.txt", "content": "y" * 5000, "nested": {"k": "v" * 2000}}
    rid = audit.log("ada", "tool-gate", "write_file", "approved", "ok",
                    tool_args=big_args, trace_id="t1")
    rec = audit.get(rid)
    stored = json.loads(rec["tool_args_json"])
    assert stored == big_args                      # full args, no truncation
    assert len(rec["tool_args_json"]) > 5000


def test_detail_bound_raised_to_4000(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    rid = audit.log("ada", "tool-gate", "write_file", "approved", "d" * 6000)
    assert len(audit.get(rid)["detail"]) == 4000


def test_migration_preserves_legacy_rows(tmp_path):
    db = make_audit_db(tmp_path, legacy=True)
    audit = ak.AuditLog(db)                        # migration runs here
    rows = audit.recent(5)
    assert rows[0]["target"] == "write_file"        # old row survived
    assert rows[0]["detail"] == "old row"
    cols = {r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(audit)")}
    for col in ("tool_args_json", "model_lane", "prompt_hash", "output_summary",
                "alternatives_json", "trace_id", "gate_json",
                "world_hash", "timeline_hash", "session_id", "step",
                "tool_version", "undone"):
        assert col in cols
    # and new writes still work on the migrated table
    rid = audit.log("ada", "tool-gate", "read_file", "approved", "new",
                    tool_args={"p": 1})
    assert audit.get(rid)["tool_args_json"] == '{"p": 1}'


def test_set_output_attaches_result(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    rid = audit.log("ada", "tool-gate", "search_web", "approved", "ok")
    assert audit.get(rid)["output_summary"] is None
    audit.set_output(rid, "3 results found")
    assert audit.get(rid)["output_summary"] == "3 results found"


# ── 2. inspect() ──────────────────────────────────────────────────────────

def test_inspect_reconstructs_context_from_fabricated_record(tmp_path):
    db = make_audit_db(tmp_path)
    tl = make_timeline_db(tmp_path)
    audit = ak.AuditLog(db)
    now = time.time()
    add_event(tl, 1, now - 60, "policy.decision",
              {"action": "suppressed", "reason": "focus time"})
    add_event(tl, 2, now - 30, "audit.decision",
              {"tool": "write_file", "decision": "approved"})
    evs = replay.timeline_slice(str(tl), now)
    tl_hash = replay.timeline_slice_hash(evs)
    rid = audit.log("ada", "tool-gate", "write_file", "approved",
                    "frictionless at L2 (risk=state-changing-reversible)",
                    session_id="sess1", step="tool-gate",
                    tool_args={"path": "/tmp/a.txt"},
                    trace_id="trace9",
                    gate={"level": 2, "risk": "state-changing-reversible",
                          "verdict": "auto-approved", "needs_voice": False,
                          "capability": "files", "capability_mode": "act",
                          "reason": "frictionless at L2"},
                    timeline_hash=tl_hash, world_hash=None)
    # push the record's ts into the event window
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE audit SET ts=? WHERE id=?", (now, rid))
    conn.commit()
    conn.close()

    b = replay.inspect(rid, str(db), timeline_db=str(tl))
    assert b["ok"] and b["tool"] == "write_file"
    assert b["inputs"] == {"path": "/tmp/a.txt"}
    assert b["timeline_events"] == 2
    assert b["hash_check"]["timeline"] == "match"   # content-addressed
    assert b["hash_check"]["world"] == "unavailable"
    assert "write_file" in b["narrative"]
    assert "frictionless at L2" in b["narrative"]


def test_inspect_reports_hash_drift(tmp_path):
    db = make_audit_db(tmp_path)
    tl = make_timeline_db(tmp_path)
    audit = ak.AuditLog(db)
    now = time.time()
    add_event(tl, 1, now - 60, "audit.decision", {"tool": "x"})
    rid = full_gate_record(audit, now, tl_hash="deadbeef" * 8)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE audit SET ts=? WHERE id=?", (now, rid))
    conn.commit()
    conn.close()
    b = replay.inspect(rid, str(db), timeline_db=str(tl))
    assert b["hash_check"]["timeline"] == "drift"


def test_inspect_unknown_id(tmp_path):
    db = make_audit_db(tmp_path)
    b = replay.inspect(99999, str(db))
    assert not b["ok"]


# ── 3. redecide() ─────────────────────────────────────────────────────────

def test_redecide_detects_policy_change(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    now = time.time()
    rid = full_gate_record(audit, now)
    aut = AutonomyLevels(str(tmp_path / "st"))
    aut.set_capability_mode("files", "act")        # as recorded: L2, frictionless

    same = replay.redecide(rid, str(db), autonomy=aut)
    assert same["ok"] and not same["changed"]
    assert same["now"]["verdict"] == "auto-approved"
    assert "re-run" in same["mode"]                # honestly labelled

    aut.set_capability_mode("files", "ask-first")  # flip the gate
    changed = replay.redecide(rid, str(db), autonomy=aut)
    assert changed["ok"] and changed["changed"]
    assert changed["now"]["verdict"] == "voice-ask"
    assert "auto-approved" in changed["why"] and "voice-ask" in changed["why"]


def test_redecide_detects_risk_class_change(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    now = time.time()
    rid = full_gate_record(audit, now, tool="calendar_event", gate={
        "level": 2, "risk": "state-changing-reversible", "verdict": "auto-approved",
        "needs_voice": False, "capability": "calendar", "capability_mode": "ladder",
        "reason": "old mapping"})
    aut = AutonomyLevels(str(tmp_path / "st"))
    aut.set_level(2)
    # calendar_event is external-side-effect today: would be voice-asked.
    r = replay.redecide(rid, str(db), autonomy=aut)
    assert r["ok"] and r["changed"]
    assert r["now"]["verdict"] == "voice-ask"


def test_redecide_legacy_record_without_gate(tmp_path):
    db = make_audit_db(tmp_path, legacy=True)
    audit = ak.AuditLog(db)
    rid = audit.recent(5)[0]["id"]
    r = replay.redecide(rid, str(db))
    assert not r["ok"] and "predates" in r["error"]


# ── 4. widget kind + actions ──────────────────────────────────────────────

def test_trust_console_kind_validates_and_sanitizes():
    props = {"decisions": [{"id": 1, "tool": "write_file",
                            "onclick": "alert(1)", "__proto__": {"x": 1}}],
             "toggles": [{"capability": "files", "mode": "act"}]}
    msg, why = w.build_open("trust_console", props)
    assert why == "" and msg["kind"] == "trust_console"
    d0 = msg["props"]["decisions"][0]
    assert "onclick" not in d0 and "__proto__" not in d0
    assert d0["tool"] == "write_file"


def test_trust_console_requires_decisions():
    msg, why = w.build_open("trust_console", {"toggles": []})
    assert msg is None and "decisions" in why


def test_trust_console_unknown_kind_still_refused():
    assert w.build_open("trust_o_console", {})[0] is None


def test_new_actions_allowlisted():
    for a in ("replay_decision", "undo_action", "set_autonomy_toggle"):
        assert a in hud_spec.ACTIONS
    # the original eight are untouched
    assert len(hud_spec.ACTIONS) == 11


def test_console_snapshot_props_shape(tmp_path):
    db = make_audit_db(tmp_path)
    tl = make_timeline_db(tmp_path)
    audit = ak.AuditLog(db)
    now = time.time()
    full_gate_record(audit, now)
    add_event(tl, 1, now - 10, "policy.decision",
              {"action": "suppressed", "reason": "focus time"})
    aut = AutonomyLevels(str(tmp_path / "st"))
    props = replay.console_snapshot(str(db), timeline_db=str(tl), autonomy=aut)
    assert props["decisions"] and props["decisions"][0]["tool"] == "write_file"
    assert any(t["capability"] == "files" for t in props["toggles"])
    assert props["restraint"]["suppressed"] == 1
    assert props["summary"]["decisions"] == 1
    # and the props pass the backend widget validator untouched
    msg, why = w.build_open("trust_console", props)
    assert why == "" and msg["kind"] == "trust_console"


# ── 5. undo ───────────────────────────────────────────────────────────────

class _FakeScheduler:
    def __init__(self):
        self.cancelled = []
    def cancel(self, tid):
        self.cancelled.append(tid)
        return f"Cancelled task {tid}"


class _FakeMemory:
    def __init__(self):
        self.forgotten = []
    def forget_fact(self, key):
        self.forgotten.append(key)
        return f"forgot {key}"


class _FakeKernel:
    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.memory = _FakeMemory()


def test_undo_reversible_reminder(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    rid = audit.log("ada", "tool-gate", "create_reminder", "approved",
                    "frictionless", tool_args={"description": "stand up", "delay_seconds": 60},
                    output_summary="Reminder set (abc123def456): stand up in 60s.")
    kernel = _FakeKernel()
    res = replay.undo_record(rid, str(db), replay.default_undo_handlers(kernel))
    assert res["ok"] and kernel.scheduler.cancelled == ["abc123def456"]
    rec = audit.get(rid)
    assert rec["undone"] == 1
    # the undo itself is audit-logged
    undo_rows = [r for r in audit.decisions(10) if r["action"] == "undo"]
    assert undo_rows and undo_rows[0]["target"] == "create_reminder"


def test_undo_reversible_remember(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    rid = audit.log("ada", "tool-gate", "remember", "approved", "frictionless",
                    tool_args={"key": "coffee", "value": "oat milk"})
    kernel = _FakeKernel()
    res = replay.undo_record(rid, str(db), replay.default_undo_handlers(kernel))
    assert res["ok"] and kernel.memory.forgotten == ["coffee"]
    assert "not restored" in res["note"]           # honest about the limit


def test_undo_refuses_irreversible_tool(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    rid = audit.log("ada", "tool-gate", "send_email", "approved", "asked+approved",
                    tool_args={"to": "x@y.z"})
    res = replay.undo_record(rid, str(db), replay.default_undo_handlers(_FakeKernel()))
    assert not res["ok"] and "no reversible inverse" in res["reason"]
    assert audit.get(rid)["undone"] == 0           # untouched


def test_undo_is_idempotent(tmp_path):
    db = make_audit_db(tmp_path)
    audit = ak.AuditLog(db)
    rid = audit.log("ada", "tool-gate", "remember", "approved", "ok",
                    tool_args={"key": "k", "value": "v"})
    kernel = _FakeKernel()
    handlers = replay.default_undo_handlers(kernel)
    assert replay.undo_record(rid, str(db), handlers)["ok"]
    second = replay.undo_record(rid, str(db), handlers)
    assert not second["ok"] and "already undone" in second["reason"]
    assert kernel.memory.forgotten == ["k"]        # only once


# ── 6. restraint panel ────────────────────────────────────────────────────

def test_restraint_counts_suppressions(tmp_path):
    tl = make_timeline_db(tmp_path)
    now = time.time()
    add_event(tl, 1, now - 100, "policy.decision",
              {"action": "suppressed", "reason": "focus time"})
    add_event(tl, 2, now - 90, "policy.decision",
              {"action": "suppressed", "reason": "quiet hours"})
    add_event(tl, 3, now - 80, "policy.decision",
              {"action": "suppressed", "reason": "focus time"})
    add_event(tl, 4, now - 70, "policy.decision", {"action": "sent"})
    add_event(tl, 5, now - 99999, "policy.decision",
              {"action": "suppressed", "reason": "stale"})  # outside window
    r = replay.restraint_summary(str(tl), hours=1)
    assert r["suppressed"] == 3 and r["sent"] == 1
    assert r["top_suppress_reasons"][0] == {"reason": "focus time", "count": 2}


def test_restraint_unavailable_without_db(tmp_path):
    r = replay.restraint_summary(str(tmp_path / "missing.db"), hours=1)
    assert r["unavailable"] and r["suppressed"] == 0


# ── 7. capability toggles ─────────────────────────────────────────────────

def test_capability_toggle_roundtrip_and_persist(tmp_path):
    aut = AutonomyLevels(str(tmp_path / "st"))
    assert aut.capability_modes()["email"] == "ladder"
    assert aut.effective_level("send_email") == aut.level
    msg = aut.set_capability_mode("email", "ask-first")
    assert "ask first" in msg.lower()
    assert aut.effective_level("send_email") == 1
    aut.set_capability_mode("files", "act")
    assert aut.effective_level("write_file") == 2
    # destructive tools never go free on a toggle
    assert autonomy.evaluate_gate("send_email", level=aut.effective_level("send_email"))["needs_voice"]
    # persists
    aut2 = AutonomyLevels(str(tmp_path / "st"))
    assert aut2.capability_mode("email") == "ask-first"
    assert aut2.capability_mode("files") == "act"


def test_capability_toggle_rejects_unknown(tmp_path):
    aut = AutonomyLevels(str(tmp_path / "st"))
    assert "Unknown capability" in aut.set_capability_mode("teleport", "act")
    assert "ask-first, act, or ladder" in aut.set_capability_mode("email", "yolo")


def test_evaluate_gate_is_pure_and_shared():
    g = autonomy.evaluate_gate("write_file", level=1, armed=False)
    assert g["verdict"] == "voice-ask" and g["needs_voice"]
    g2 = autonomy.evaluate_gate("write_file", level=2, armed=False)
    assert g2["verdict"] == "auto-approved"
    g3 = autonomy.evaluate_gate("send_email", level=3, armed=True)
    assert g3["verdict"] == "auto-approved" and "Instant Kill" in g3["reason"]


# ── 8. weekly trust summary ───────────────────────────────────────────────

def test_weekly_trust_summary(tmp_path):
    root = tmp_path / "root"
    kdir = root / "ada_kernel"
    kdir.mkdir(parents=True)
    db = kdir / "ada.db"
    tl = kdir / "ada_events.db"
    audit = ak.AuditLog(db)
    now = time.time()
    conn = sqlite3.connect(str(tl))
    conn.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, ts_utc REAL NOT NULL,"
                 "source TEXT NOT NULL, type TEXT NOT NULL, payload_json TEXT NOT NULL,"
                 "schema_v INTEGER NOT NULL, session_id TEXT NOT NULL)")
    conn.commit()
    conn.close()
    add_event(tl, 1, now - 100, "policy.decision",
              {"action": "suppressed", "reason": "focus time"})
    full_gate_record(audit, now)
    rid2 = audit.log("ada", "tool-gate", "send_email", "denied", "voice ask denied",
                     gate={"level": 1, "risk": "external-side-effect",
                           "verdict": "voice-ask", "needs_voice": True,
                           "reason": "voice ask at L1"})
    # fix timestamps into the window
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE audit SET ts=?", (now,))
    conn.commit()
    conn.close()
    out = ledger.weekly_trust(str(root), days=7)
    assert "2 decisions" in out
    assert "confirmations asked" in out
    assert "denied by you" in out
    assert "1 suggestions held back" in out
    assert "audit log" in out


def test_weekly_trust_no_db(tmp_path):
    assert "No trust ledger" in ledger.weekly_trust(str(tmp_path / "empty"))
