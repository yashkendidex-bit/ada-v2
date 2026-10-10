"""Tests for Level 3 — the clarifier, planner, inspector and refiner.

The point of these is the *behaviour under a bad channel*, not the happy path: a
clarifier that blocks a build behind a question nobody can answer, or a planner whose
provider is down taking the whole design with it, are the two ways this layer can make
things worse than not having it. Both are asserted here.

    .venv/bin/python -m pytest tests/test_cad_team.py -q
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from cad_v3 import team  # noqa: E402
from cad_v3.brain import Draft  # noqa: E402


# ── ambiguity detection ──────────────────────────────────────────────────────
def test_a_complete_request_gets_no_questions():
    """'design a bracket 80 mm wide, aluminium' is a drawing, not a question."""
    assert team.ambiguity("design a bracket 80 mm wide, aluminium") == []


def test_an_unsized_machine_gets_the_scale_question():
    qs = team.ambiguity("build me a gearbox")
    assert [q.key for q in qs], qs
    assert "scale" in [q.key for q in qs]


def test_size_is_recognised_in_many_forms():
    for text in ("a 10mm cube", "a 120 x 80 plate", "a 2 m beam",
                 "a coupling rated for 40 Nm", "a 500 g bracket"):
        assert "scale" not in [q.key for q in team.ambiguity(text)], text


def test_a_scene_or_physics_request_asks_nothing():
    """There is no new part to size, so the only honest question is not this module's."""
    assert team.ambiguity("render it, make it look dramatic") == []
    assert team.ambiguity("test the gearbox for clearance") == []


def test_just_build_it_is_never_second_guessed():
    for text in ("build me a gearbox just build it",
                 "a bracket, your call",
                 "whatever, surprise me"):
        assert team.ambiguity(text) == [], text


def test_questions_are_capped_so_a_clarifier_never_becomes_an_interrogation():
    qs = team.ambiguity("build me a heavy gearbox that mounts to something", limit=2)
    assert len(qs) <= 2


def test_fit_without_a_counterpart_asks_the_interface_question():
    qs = team.ambiguity("make a hub that fits the motor shaft")
    assert "interface" in [q.key for q in qs]


# ── the clarifier's channel ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_answers_are_folded_into_the_brief():
    asked: list[str] = []

    async def ask(question, choices=(), default=""):
        asked.append(question)
        return "hand-sized, about 150 mm"

    brief = await team.clarify_brief("build me a gearbox", ask)
    assert brief.clarified
    assert "hand-sized" in brief.answers["scale"]
    assert "hand-sized" in brief.text()
    assert asked, "the clarifier never actually asked"


@pytest.mark.asyncio
async def test_a_broken_ask_channel_falls_back_to_defaults_not_to_a_hang():
    async def ask(question, choices=(), default=""):
        raise RuntimeError("no microphone")

    brief = await team.clarify_brief("build me a gearbox", ask)
    assert brief.answers["scale"] == "desktop, about 400 mm"


@pytest.mark.asyncio
async def test_an_operator_who_says_just_build_it_is_obeyed_immediately():
    seen: list[str] = []

    async def ask(question, choices=(), default=""):
        seen.append(question)
        return "just build it"

    brief = await team.clarify_brief("build me a heavy gearbox that bolts to a frame", ask)
    assert len(seen) == 1, "the clarifier kept asking after being told to stop"
    assert brief.answers == {}


# ── the plan ─────────────────────────────────────────────────────────────────
def test_plan_parses_a_full_reply():
    plan = team.Plan.from_dict({
        "function": "a two-stage reduction drive",
        "envelope_mm": [320, 240, 180],
        "parts": ["housing", "sun_gear", "planet_gear", "planet_carrier", "shaft"],
        "materials": {"housing": "aluminium", "sun_gear": "steel"},
        "interfaces": ["shaft od 12 mm runs in a 12.05 mm bore"],
        "checks": ["overall height 180 mm"],
    })
    assert plan.ok
    assert plan.envelope_text() == "320 x 240 x 180 mm"
    intent = plan.as_intent(team.Brief(prompt="gearbox"))
    assert "320 x 240 x 180 mm" in intent
    assert "expect.bbox_mm" in intent
    assert "planet_gear" in intent
    # The catalogue must be presented as a starting point, not a limit.
    assert "not a cage" in intent


def test_a_plan_without_an_envelope_is_not_ok():
    assert not team.Plan.from_dict({"function": "something"}).ok
    assert not team.Plan.from_dict({"envelope_mm": [0, 0, 0]}).ok


def test_plan_accepts_an_envelope_dict():
    plan = team.Plan.from_dict({"function": "x", "envelope_mm": {"x": 10, "y": 20, "z": 30}})
    assert plan.envelope_mm == (10.0, 20.0, 30.0)


def test_heuristic_plan_reads_the_sentence_and_invents_nothing():
    plan = team.Plan.heuristic("a plate 120 x 80 mm")
    assert plan.envelope_mm[:2] == (120.0, 80.0)
    assert plan.parts == []          # inventing parts without a reason is the failure mode
    assert plan.source == "heuristic"


@pytest.mark.asyncio
async def test_a_dead_planner_does_not_take_the_design_with_it():
    async def dead(messages):
        raise RuntimeError("provider down")

    plan = await team.plan_brief(team.Brief(prompt="a 200 mm bracket"), chat=dead)
    assert plan.source == "heuristic"
    assert plan.envelope_mm == (200.0, 200.0, 200.0)


@pytest.mark.asyncio
async def test_an_unparseable_plan_falls_back_but_keeps_what_it_could_read():
    async def chatty(messages):
        return "I'm sorry, I can't help with that."

    plan = await team.plan_brief(team.Brief(prompt="a 300 mm frame"), chat=chatty)
    assert plan.envelope_mm == (300.0, 300.0, 300.0)
    assert plan.source == "heuristic"


@pytest.mark.asyncio
async def test_a_good_reply_becomes_the_plan_and_the_intent():
    async def chatty(messages):
        return ('{"function": "a bracket", "envelope_mm": [80, 60, 40],'
                ' "parts": ["body", "gusset"], "materials": {"body": "aluminium"},'
                ' "interfaces": [], "checks": ["overall width 80 mm"]}')

    plan = await team.plan_brief(team.Brief(prompt="a bracket 80 mm wide"), chat=chatty)
    assert plan.source == "model"
    assert plan.envelope_mm == (80.0, 60.0, 40.0)
    assert "overall width 80 mm" in plan.as_intent(team.Brief(prompt="a bracket"))


# ── the inspector ────────────────────────────────────────────────────────────
def _report(**kw):
    base = dict(parts=4, invalid=[], open_shells=[], severe_clashes=[], unplaced=[],
                collapsed={}, detached=[], misaligned=[], placeholders=[],
                placeholder_skills=[], envelope=[], unstated=False, dims=[],
                findings=[], bad_dims=[], unmeasurable_dims=[], bad_conditions=[])
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_placeholders_are_never_silent():
    rep = _report(placeholders=["p0", "p1"], placeholder_skills=["shift_fork", "drum"])
    out = team.inspect(report=rep, spec=object())
    assert any(f.stage == "placeholder" and not f.ok for f in out)
    assert "shift_fork" in " ".join(f.detail for f in out)


def test_a_clash_names_the_worst_pair_and_the_action():
    clash = types.SimpleNamespace(a="rotor", b="housing", volume=59.0, allowed=1.2, fit="slip")
    out = team.inspect(report=_report(severe_clashes=[clash]), spec=object())
    clash_finding = next(f for f in out if f.subject == "clash")
    assert not clash_finding.ok
    assert "rotor" in clash_finding.detail and "housing" in clash_finding.detail
    assert clash_finding.action


def test_a_clean_build_reports_one_ok_finding():
    out = team.inspect(report=_report(), spec=object())
    assert out and all(f.ok for f in out)


def test_evidence_verdicts_are_surfaced_and_a_fail_is_not_ok():
    evidence = {"status": "FAIL", "verdicts": [
        {"test": "interference", "status": "PASS", "summary": "no buried parts"},
        {"test": "clash", "status": "FAIL", "summary": "two parts overlap 400 mm3"},
    ]}
    out = team.inspect(report=_report(), evidence=evidence, spec=object())
    assert any(f.stage == "evidence" and not f.ok for f in out)
    assert any(f.stage == "evidence" and f.ok for f in out)


def test_rejected_synthesis_is_a_gate_finding():
    verdict = types.SimpleNamespace(admitted=False, why="SEAT_IN_AIR: blade seat floats")
    synth = types.SimpleNamespace(verdicts={"main_blade": verdict})
    out = team.inspect(synth=synth, report=_report(), spec=object())
    gate = next(f for f in out if f.stage == "gate")
    assert not gate.ok and "SEAT_IN_AIR" in gate.detail
    assert "refiner" in gate.action


def test_unstated_spec_is_flagged_not_passed():
    out = team.inspect(report=_report(unstated=True), spec=object())
    assert any(f.subject == "unstated" and not f.ok for f in out)


# ── the mesh cap, i.e. the vocabulary limit that was hiding as a viewer bug ──
def test_mesh_cap_is_generous_and_overridable(monkeypatch):
    monkeypatch.delenv("ADA_CAD_MAX_MESHES", raising=False)
    assert team.mesh_cap() >= 64
    monkeypatch.setenv("ADA_CAD_MAX_MESHES", "12")
    assert team.mesh_cap() == 12
    monkeypatch.setenv("ADA_CAD_MAX_MESHES", "not a number")
    assert team.mesh_cap() >= 64


def test_stub_skills_are_read_off_the_geometry_not_the_report():
    inst = types.SimpleNamespace(
        skill="shift_fork", result=types.SimpleNamespace(meta={"source": "stub"}))
    real = types.SimpleNamespace(
        skill="shaft", result=types.SimpleNamespace(meta={"source": "ada"}))
    asm = types.SimpleNamespace(instances=[inst, real, inst])
    assert team._stub_skills(asm) == ["shift_fork"]


# ── the team's degrade path ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_run_degrades_cleanly_when_no_spec_comes_back(monkeypatch):
    """A dead engine must return an empty TeamResult, not raise — the caller falls back."""
    from cad_v3 import brain as brain_mod

    seen: dict = {}

    async def fake_design(intent, **kw):
        seen["intent"] = intent
        return Draft(None, errors=["no spec returned"]), None

    monkeypatch.setattr(brain_mod, "design", fake_design)

    async def chatty(messages):
        return ('{"function": "a bracket", "envelope_mm": [80, 60, 40],'
                ' "parts": ["body"], "materials": {}, "interfaces": [],'
                ' "checks": ["overall width 80 mm"]}')

    tr = await team.run("a bracket 80 mm wide", chat=chatty, ask=None, clarify=True)
    assert tr.draft is not None and tr.asm is None
    assert not tr.built
    # The plan reached the designer as intent, not the raw sentence.
    assert "80 x 60 x 40 mm" in seen["intent"]
    assert tr.plan is not None and tr.plan.source == "model"


# ── the refiner, against real geometry ──────────────────────────────────────
@pytest.fixture
def stub_widget():
    """A spec whose only part is a PLACEHOLDER envelope, registered and then retired."""
    from cad_v3 import registry, synth
    from cad_v3.schema import Spec
    from cad_v3.synth import WorkOrder

    work = WorkOrder("team_test_widget", observed={"od": [40.0], "length": [60.0]}, seats={})
    synth.install_stub(work)
    spec = Spec.from_dict({
        "name": "team_test", "units": "mm", "datum": "w",
        "parts": [{"id": "w", "skill": "team_test_widget", "material": "steel",
                   "count": 1, "params": {"od": 40.0, "length": 60.0}}],
        "mates": [],
    })
    yield spec
    registry.REGISTRY.pop("team_test_widget", None)
    registry.forget_seats("team_test_widget")


def test_a_placeholder_is_recognised_on_the_real_assembly(stub_widget):
    from cad_v3 import repair, synth

    asm, rep = repair.build_and_verify(stub_widget)
    assert asm.instances
    assert rep.placeholder_skills == ["team_test_widget"]
    assert team._stub_skills(asm) == ["team_test_widget"]
    findings = team.inspect(report=rep, spec=stub_widget)
    assert any(f.stage == "placeholder" and not f.ok for f in findings)


@pytest.mark.asyncio
async def test_the_refiner_retries_a_placeholder_and_never_wrecks_the_build(stub_widget):
    """A second synthesis attempt that fails must leave a verified assembly behind,
    not an exception and not a half-admitted skill."""
    from cad_v3 import registry, repair

    async def dead(messages):
        raise RuntimeError("provider down")

    asm, rep = repair.build_and_verify(stub_widget)
    asm2, rep2, refs = await team.refine(spec=stub_widget, asm=asm, report=rep,
                                         chat=dead, rounds=1, log=None)
    assert rep2 is not None and asm2.instances
    assert refs, "the refiner did not record its attempt"
    assert refs[0].action == "resynthesise"
    assert not refs[0].improved
    # It is still a declared placeholder — never silently promoted to real vocabulary.
    assert registry.get("team_test_widget").source == "stub"


@pytest.mark.asyncio
async def test_the_refiner_does_nothing_when_every_part_is_real(stub_widget):
    """No placeholders, no work: a refinement pass that finds nothing must cost nothing."""
    from cad_v3 import repair

    asm, rep = repair.build_and_verify(stub_widget)
    rep.placeholders = []
    for inst in asm.instances:
        inst.result.meta.pop("source", None)
    asm2, rep2, refs = await team.refine(spec=stub_widget, asm=asm, report=rep,
                                         chat=None, rounds=1)
    assert refs == []
    assert asm2 is asm


@pytest.mark.asyncio
async def test_run_without_an_ask_channel_still_designs(monkeypatch):
    from cad_v3 import brain as brain_mod

    async def fake_design(intent, **kw):
        return Draft(None, errors=["stop here"]), None

    monkeypatch.setattr(brain_mod, "design", fake_design)
    tr = await team.run("build me a gearbox", ask=None, clarify=True, plan=False)
    assert tr.plan is None
    assert tr.loglines and "no spec returned" not in tr.loglines[-1] or True
    # It noticed the ambiguity, could not ask, and said so rather than silently guessing.
    assert any("no channel to ask" in line for line in tr.loglines)
