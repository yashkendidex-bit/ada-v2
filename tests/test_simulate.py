"""Tests for Feature 5 — simulation & counterfactual engine.

All tmp dirs; no network (the provider router is mocked — rollout takes a
chat_fn, and the ada.py hook test monkeypatches simulate.rollout).
"""
import asyncio
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import simulate


def _sim():
    # Fresh import binding for clarity in every test.
    import importlib
    importlib.reload(simulate)
    return simulate


SIM_OUTPUT = """DIRECT EFFECTS: {direct}
SIDE EFFECTS: {side}
FAILURE MODES: {fail}
USER REACTION: {react}
REVERSIBILITY: {rev}
IMPACT: {imp}
PREFERENCE FIT: {fit}
PREDICTED OUTCOME: {outcome}
PREDICTION: {pred}"""


def make_mock_chat(rev=9, imp=2, fit=8, outcome="smooth", pred="all good"):
    async def chat(messages):
        sys_txt = ""
        for m in messages:
            if m.get("role") == "system":
                sys_txt += m.get("content", "")
        if "skeptic" in sys_txt.lower():
            return "Objection 1: side effects underestimated. Objection 2: user may not want this now."
        return SIM_OUTPUT.format(
            direct="does the thing", side="none expected",
            fail="disk full (1%)", react="pleased",
            rev=rev, imp=imp, fit=fit, outcome=outcome, pred=pred)
    return chat


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) \
        if not asyncio.get_event_loop().is_running() else None


def arun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── 1. dry_run: file domain ──────────────────────────────────────────────

def test_dry_run_file_write_produces_diff_without_touching_real_files(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("line one\nline two\n", encoding="utf-8")
    safe, report = simulate.dry_run(
        "write_file", {"path": str(target), "content": "line one\nline THREE\n"})
    assert safe is True
    assert "+line THREE" in report and "-line two" in report
    assert target.read_text(encoding="utf-8") == "line one\nline two\n", \
        "real file must be untouched by the sandbox"


def test_dry_run_file_create_reports_new_file(tmp_path):
    target = tmp_path / "brand_new.txt"
    safe, report = simulate.dry_run(
        "write_file", {"path": str(target), "content": "hello\n"})
    assert safe is True
    assert "CREATE" in report
    assert not target.exists(), "sandbox must not create the real file"


# ── 2. dry_run: money domain ─────────────────────────────────────────────

def test_dry_run_budget_check_flags_over_budget():
    safe, report = simulate.dry_run(
        "buy", {"price": "50000", "item": "laptop"},
        {"budget": 1000.0})
    assert safe is False
    assert "OVER BUDGET" in report


def test_dry_run_money_within_budget_is_safe():
    safe, report = simulate.dry_run(
        "buy", {"price": "200", "item": "book"},
        {"budget": 1000.0, "spending_history": [150.0, 180.0, 220.0]})
    assert safe is True
    assert "within budget" in report


def test_dry_run_money_flags_unusual_spike_without_budget():
    safe, report = simulate.dry_run(
        "buy", {"amount": 10000},
        {"spending_history": [50.0, 60.0, 55.0]})
    assert safe is False
    assert "unusual" in report


# ── 3. dry_run: schedule domain ──────────────────────────────────────────

def test_dry_run_calendar_conflict_detected():
    events = [{"title": "Dentist", "start": "2026-10-17T10:00:00"},
              {"title": "Gym", "start": "2026-10-17T18:00:00"}]
    safe, report = simulate.dry_run(
        "schedule_event",
        {"title": "Call", "datetime": "2026-10-17T10:00:00"},
        {"calendar": events})
    assert safe is False
    assert "CONFLICT" in report and "Dentist" in report


def test_dry_run_calendar_no_conflict():
    events = [{"title": "Dentist", "start": "2026-10-17T10:00:00"}]
    safe, report = simulate.dry_run(
        "schedule_event",
        {"title": "Call", "datetime": "2026-10-17T12:00:00"},
        {"calendar": events})
    assert safe is True
    assert "no conflicts" in report


# ── 4. dry_run: code domain ──────────────────────────────────────────────

def test_dry_run_code_compiles():
    safe, report = simulate.dry_run("run_code", {"code": "x = 1 + 2\nprint(x)"})
    assert safe is True
    assert "compiles clean" in report


def test_dry_run_code_syntax_error_is_unsafe():
    safe, report = simulate.dry_run("run_code", {"code": "def broken(:\n  pass"})
    assert safe is False
    assert "syntax error" in report


def test_dry_run_code_bad_gcode_flagged():
    safe, report = simulate.dry_run(
        "generate_gcode", {"gcode": "G1 X10 Y20\nBOGUSCOMMAND 123\n"})
    assert safe is False
    assert "unrecognized" in report


# ── 5. snapshot: content-addressed ───────────────────────────────────────

class FakeWorld:
    def get_entity_context(self, name, max_edges=12):
        return f"{name}: owns printer, prefers PLA"
    def check(self, action):
        return "routine note: focus block 9-11"


def test_snapshot_hash_is_stable_and_content_addressed(tmp_path):
    ctx = {"tool": "write_file",
           "description": "write_file: notes.txt",
           "entities": ["printer"],
           "world": FakeWorld(),
           "files": [],
           "calendar": []}
    s1 = simulate.snapshot(ctx)
    s2 = simulate.snapshot(ctx)
    assert s1["snapshot_hash"] == s2["snapshot_hash"]
    assert len(s1["snapshot_hash"]) == 64
    assert s1["entities"]["printer"] == "printer: owns printer, prefers PLA"
    assert "focus block" in s1["routine_note"]
    s3 = simulate.snapshot({**ctx, "description": "different action"})
    assert s3["snapshot_hash"] != s1["snapshot_hash"]


def test_snapshot_file_manifest_hashes_real_files(tmp_path):
    f = tmp_path / "plan.md"
    f.write_text("abc", encoding="utf-8")
    s = simulate.snapshot({"files": [str(f), "/nonexistent/x"]})
    man = {m["path"]: m for m in s["file_manifest"]}
    assert man[str(f)]["exists"] is True and "sha256" in man[str(f)]
    assert man["/nonexistent/x"]["exists"] is False


# ── 6. rollout: reversible beats irreversible (mocked router) ────────────

def test_rollout_ranks_reversible_above_irreversible():
    async def chat(messages):
        sys_txt = "".join(m.get("content", "") for m in messages
                          if m.get("role") == "system")
        if "skeptic" in sys_txt.lower():
            return "Objection 1: irreversible side effects. Objection 2: none."
        user = "".join(m.get("content", "") for m in messages
                       if m.get("role") == "user")
        if "irreversible-delete" in user:
            return SIM_OUTPUT.format(
                direct="deletes everything", side="backups gone too",
                fail="data loss (40%)", react="annoyed",
                rev=2, imp=9, fit=6, outcome="major-issue",
                pred="likely data loss")
        return SIM_OUTPUT.format(
            direct="moves to trash", side="none",
            fail="accidental move (2%)", react="pleased",
            rev=9, imp=2, fit=8, outcome="smooth",
            pred="moves to trash, easy to undo")

    cands = [
        {"id": "a", "description": "irreversible-delete: wipe the folder permanently"},
        {"id": "b", "description": "soft-move: move the folder to trash (undoable)"},
    ]
    out = arun(simulate.rollout(cands, {"entities": {}}, chat_fn=chat))
    assert out["winner"]["candidate"]["id"] == "b"
    assert out["ranked"][0]["simulation"]["score"] > \
        out["ranked"][1]["simulation"]["score"]
    assert out["skeptic_critique"].startswith("Objection 1")
    assert out["calls"] == 3  # 2 sim calls + 1 skeptic critique


def test_rollout_caps_at_three_candidates():
    chat = make_mock_chat()
    cands = [{"id": str(i), "description": f"cand {i}"} for i in range(5)]
    out = arun(simulate.rollout(cands, {}, chat_fn=chat))
    assert len(out["ranked"]) == 3
    assert out["calls"] == 4  # 3 sim + 1 skeptic


def test_rollout_survives_chat_failure():
    async def bad_chat(messages):
        raise RuntimeError("provider down")
    out = arun(simulate.rollout(
        [{"id": "a", "description": "do the thing"}], {}, chat_fn=bad_chat))
    assert out["winner"] is not None
    assert "no simulation available" in out["winner"]["simulation"]["prediction"]


# ── 7. blast radius gating + kill switch ─────────────────────────────────

def test_is_high_blast_radius():
    # F6's classifier (mandate) + autonomy-ladder destructive/external.
    assert simulate.is_high_blast_radius("send_email", {"to": "a@b.c"}) is True
    assert simulate.is_high_blast_radius("delete_file", {"path": "x"}) is True
    assert simulate.is_high_blast_radius("desktop_autopilot", {"goal": "x"}) is True
    # Reversible per the ladder / "low" per F6: skipped, zero added latency.
    assert simulate.is_high_blast_radius("write_file", {"path": "x"}) is False
    assert simulate.is_high_blast_radius("search_web", {"query": "x"}) is False
    assert simulate.is_high_blast_radius("read_file", {"path": "x"}) is False


def test_dry_run_delete_reports_without_removing(tmp_path):
    target = tmp_path / "doomed.txt"
    target.write_text("do not lose me\nsecond line\n", encoding="utf-8")
    safe, report = simulate.dry_run("delete_file", {"path": str(target)})
    assert safe is False
    assert "would DELETE" in report and "do not lose me" in report
    assert target.exists(), "sandbox must never delete the real file"


def test_kill_switch_disables_simulation(monkeypatch):
    monkeypatch.setenv("ADA_SIMULATE", "0")
    assert simulate.sim_enabled() is False
    monkeypatch.setenv("ADA_SIMULATE", "1")
    assert simulate.sim_enabled() is True
    monkeypatch.setenv("ADA_SIMULATE", "false")
    assert simulate.sim_enabled() is False


def test_kill_switch_skips_maybe_simulate(monkeypatch, tmp_path):
    monkeypatch.setenv("ADA_SIMULATE", "0")
    import ada
    loop = ada.AudioLoop.__new__(ada.AudioLoop)
    loop.kernel = SimpleNamespace(autonomy=None, security=None, world=None)
    out = arun(loop._maybe_simulate("write_file", {"path": "x", "content": "y"}))
    assert out is None, "kill switch must skip everything"


# ── 8. counterfactual log + calibration math ─────────────────────────────

def test_counterfactual_log_and_calibration_report(tmp_path, monkeypatch):
    import perception
    tl = perception.init_timeline(tmp_path / "timeline.db")
    try:
        for i, (pred, act) in enumerate([
                ("smooth", "smooth"),
                ("smooth", "minor-issue"),
                ("minor-issue", "minor-issue")]):
            h = f"hash-{i}"
            simulate.log_counterfactual(
                h, f"action {i}",
                {"predicted_outcome": pred, "prediction": f"pred {i}"},
                {"outcome": act})
        rep = simulate.calibration_report(n=50)
        assert rep["events"] == 3
        assert rep["matched"] == 2
        assert rep["accuracy"] == pytest.approx(2 / 3, abs=0.001)
        assert rep["last_10_accuracy"] == pytest.approx(2 / 3, abs=0.001)
        # Join check: outcomes carry the same snapshot hash as predictions.
        rows = tl.query(source="simulation", type="simulation.outcome", limit=10)
        hashes = {json.loads(r["payload_json"])["snapshot_hash"] for r in rows}
        assert hashes == {"hash-0", "hash-1", "hash-2"}
    finally:
        perception._INSTANCE = None


def test_calibration_report_empty_timeline(tmp_path):
    import perception
    tl = perception.init_timeline(tmp_path / "tl2.db")
    try:
        rep = simulate.calibration_report()
        assert rep["events"] == 0 and rep["accuracy"] is None
    finally:
        perception._INSTANCE = None


# ── 9. hook order: _maybe_simulate BEFORE _check_confirmation ────────────

def _make_loop(tmp_path, voice_answer=True):
    import ada
    from autonomy import AutonomyLevels
    loop = ada.AudioLoop.__new__(ada.AudioLoop)
    audit_rows = []

    class _Audit:
        def log(self, *a, **k):
            audit_rows.append((a, k))
    aut = AutonomyLevels(str(tmp_path / "aut"))
    loop.kernel = SimpleNamespace(
        autonomy=aut,
        security=SimpleNamespace(audit=_Audit()),
        world=None)
    asked = []

    async def fake_ask(question):
        asked.append(question)
        return voice_answer
    loop._ask_user_confirmation = fake_ask
    return loop, asked


def test_hook_fires_before_confirmation_and_note_appended(tmp_path, monkeypatch):
    monkeypatch.setenv("ADA_SIMULATE", "1")
    order = []

    async def fake_rollout(candidates, snapshot, **kw):
        order.append("rollout")
        return {"ranked": [], "winner": None,
                "skeptic_critique": "watch the arg types",
                "calls": 1}

    monkeypatch.setattr(simulate, "rollout", fake_rollout)
    loop, asked = _make_loop(tmp_path)

    sim = arun(loop._maybe_simulate(
        "send_email", {"to": "a@b.c", "subject": "hi"}))
    order.append("simulate-returned")
    assert sim is not None and sim.get("note"), "hook should fire for send_email"

    orig_ask = loop._ask_user_confirmation

    async def order_ask(question):
        order.append("confirm")
        return await orig_ask(question)
    loop._ask_user_confirmation = order_ask

    ok = arun(loop._check_confirmation(
        "send_email", {"to": "a@b.c", "subject": "hi"},
        sim_note=sim["note"]))
    assert ok is True
    assert order == ["rollout", "simulate-returned", "confirm"], order
    assert "simulated" in asked[0].lower(), \
        "the 'I considered alternatives' summary must be spoken in the question"
    assert "caution" in asked[0].lower()


def test_hook_skips_low_blast_radius_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("ADA_SIMULATE", "1")
    called = []
    monkeypatch.setattr(
        simulate, "rollout",
        lambda *a, **k: called.append(1) or {"ranked": []})
    loop, _ = _make_loop(tmp_path)
    out = arun(loop._maybe_simulate("search_web", {"query": "weather"}))
    assert out is None
    out = arun(loop._maybe_simulate("write_file", {"path": "x", "content": "y"}))
    assert out is None, "reversible write_file skips simulation by design"
    assert called == [], "low-blast-radius tools must add zero latency"


def test_dry_run_unknown_domain_is_safe_by_default():
    safe, report = simulate.dry_run("frobnicator", {"foo": "bar"})
    assert safe is True
    assert "no domain sandbox" in report
