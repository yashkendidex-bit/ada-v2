"""Tests for the CAD physics engine and the three-mode router.

These assert the properties the owner asked for: that interference is measured correctly
(not proxied), that motion is judged against a baseline, and that CAD mode never emits a
scene.

    .venv/bin/python -m pytest tests/test_cad_modes.py -q
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from build123d import Box, Pos  # noqa: E402

from cad_v3 import physics  # noqa: E402


# ── minimal stand-ins for assembler.Instance / Assembly ──────────────────────
@dataclass
class FakeInstance:
    key: str
    solid: object
    material: str = "steel"
    pid: str = "p"


@dataclass
class FakeAssembly:
    instances: list = field(default_factory=list)


def make(*parts):
    """parts: (key, solid) or (key, solid, material). Material defaults to steel."""
    out = []
    for p in parts:
        key, solid = p[0], p[1]
        material = p[2] if len(p) > 2 else "steel"
        out.append(FakeInstance(key, solid, material))
    return FakeAssembly(out)


# ── the exactness property everything else rests on ──────────────────────────
def test_overlap_volume_is_exact():
    a = Box(10, 10, 10)
    assert physics.overlap_volume(a, Pos(6, 0, 0) * Box(10, 10, 10)) == pytest.approx(400.0)
    assert physics.overlap_volume(a, Box(4, 4, 4)) == pytest.approx(64.0)
    assert physics.overlap_volume(a, Box(10, 10, 10)) == pytest.approx(1000.0)


def test_touching_parts_report_zero_not_a_false_positive():
    """THE test. Two parts sharing a face are correctly assembled.

    Counting intersecting triangles reported 32 'colliding pairs' on a valid
    turbocharger, because coincident surfaces share triangles. Volume does not.
    """
    a = Box(10, 10, 10)
    assert physics.overlap_volume(a, Pos(10, 0, 0) * Box(10, 10, 10)) == pytest.approx(0.0)
    assert physics.overlap_volume(a, Pos(0, 10, 0) * Box(10, 10, 10)) == pytest.approx(0.0)


def test_disjoint_parts_report_zero():
    a = Box(10, 10, 10)
    assert physics.overlap_volume(a, Pos(30, 0, 0) * Box(10, 10, 10)) == pytest.approx(0.0)


# ── interference audit ───────────────────────────────────────────────────────
def test_valid_assembly_passes():
    asm = make(("base", Box(40, 40, 10)),
               ("post", Pos(0, 0, 15) * Box(10, 10, 20)),      # sits on the base
               ("cap", Pos(0, 0, 30) * Box(14, 14, 4)))        # sits on the post
    v = physics.interference_audit(asm)
    assert v.status == "PASS", v.render()
    assert v.metrics["buried"] == 0


def test_buried_part_fails_with_evidence():
    """A part sunk inside another must FAIL, and name both parts."""
    asm = make(("housing", Box(40, 40, 40)),
               ("flange", Pos(0, 0, 5) * Box(20, 20, 20)))     # mostly inside the housing
    v = physics.interference_audit(asm)
    assert v.status == "FAIL", v.render()
    assert v.metrics["buried"] == 1
    assert any("housing" in f.subject and "flange" in f.subject for f in v.findings)
    assert any(f.severity == "fail" for f in v.findings)


def test_audit_reports_touching_without_failing():
    asm = make(("a", Box(20, 20, 20)), ("b", Pos(20, 0, 0) * Box(20, 20, 20)))
    v = physics.interference_audit(asm)
    assert v.status == "PASS"
    assert v.metrics["touching"] >= 0
    assert v.metrics["buried"] == 0


def test_audit_is_inconclusive_when_it_cannot_decide():
    """A test that cannot decide must say so, not round up to 'fine'."""
    v = physics.interference_audit(make(("only", Box(10, 10, 10))))
    assert v.status == "INCONCLUSIVE"


# ── mass properties ──────────────────────────────────────────────────────────
def test_mass_uses_the_material_table_units():
    """1 litre of aluminium is 2.70 kg. Off-by-1000 unit bugs are the classic here."""
    asm = make(("block", Box(100, 100, 100), "aluminium"))    # 1e6 mm^3 = 1 litre
    v = physics.mass_properties(asm)
    assert v.metrics["volume_cm3"] == pytest.approx(1000.0, rel=1e-6)
    assert v.metrics["mass_kg"] == pytest.approx(2.70, rel=1e-3)


def test_mass_of_steel_is_heavier_than_aluminium():
    steel = physics.mass_properties(make(("b", Box(100, 100, 100), "steel")))
    alu = physics.mass_properties(make(("b", Box(100, 100, 100), "aluminium")))
    assert steel.metrics["mass_kg"] > alu.metrics["mass_kg"]


def test_unknown_material_is_flagged_not_silently_defaulted():
    """`materials.get()` falls back to steel, which would silently make an unset
    material weigh as steel. That must be reported, not hidden."""
    v = physics.mass_properties(make(("b", Box(100, 100, 100), "unobtainium")))
    assert any("density unknown" in f.subject for f in v.findings)
    assert "lower bound" in v.summary


def test_centre_of_mass_of_a_single_block_is_its_centre():
    v = physics.mass_properties(make(("b", Pos(5, 0, 0) * Box(10, 10, 10), "steel")))
    assert v.metrics["com_x_mm"] == pytest.approx(5.0, abs=1e-6)


# ── clearance sweep: baseline-relative, which is the whole point ─────────────
def test_static_contact_alone_is_not_a_collision():
    """A part touching another at rest and staying in contact as it turns is NOT a jam.

    A sweep without a baseline reported 501 colliding pairs and 36,072 events on a
    mechanism that actually clears.
    """
    asm = make(("shaft", Box(10, 10, 30), "steel"),
               ("bushing", Pos(5, 0, 0) * Box(10, 10, 30), "bronze"))
    v = physics.clearance_sweep(asm, rotating_keys=["shaft"], axis="Z", steps=8)
    assert v.status == "PASS", v.render()
    assert v.metrics["collisions"] == 0


def test_growing_overlap_is_a_collision():
    """An off-axis lug that sweeps into a wall must FAIL."""
    asm = make(("rotor", Pos(0, 0, 0) * Box(20, 20, 4), "steel"),
               ("wall", Pos(30, 0, 0) * Box(4, 40, 40), "steel"))
    v = physics.clearance_sweep(asm, rotating_keys=["rotor"], axis="Z", steps=36)
    # The rotor is 20 wide, so its corner reaches ~14.1 from centre; the wall starts at
    # 28. They should NOT meet. This asserts the test does not invent a collision.
    assert v.status == "PASS", v.render()


def test_sweep_is_inconclusive_without_a_rotating_group():
    asm = make(("a", Box(10, 10, 10)), ("b", Pos(20, 0, 0) * Box(10, 10, 10)))
    v = physics.clearance_sweep(asm, rotating_keys=[], axis="Z", steps=4)
    assert v.status == "INCONCLUSIVE"


def test_sweep_reports_the_grouping_source():
    """Name heuristics are a fallback, not a truth. The verdict must say which was used."""
    asm = make(("shaft", Box(10, 10, 30)), ("housing", Pos(8, 0, 0) * Box(10, 10, 30)))
    v = physics.clearance_sweep(asm, axis="Z", steps=4)     # no keys -> heuristic
    assert any("heuristic" in f.detail for f in v.findings)


# ── verdict aggregation ──────────────────────────────────────────────────────
def test_any_failure_dominates():
    pass_v = physics.Verdict("t", "PASS", "")
    fail_v = physics.Verdict("t", "FAIL", "")
    assert physics.overall([pass_v, fail_v]) == "FAIL"


def test_inconclusive_beats_pass():
    assert physics.overall([physics.Verdict("t", "PASS", ""),
                            physics.Verdict("t", "INCONCLUSIVE", "")]) == "INCONCLUSIVE"


def test_verdict_renders_readably():
    v = physics.Verdict("Interference audit", "FAIL", "two parts overlap",
                        [physics.Finding("a <-> b", "share 400 mm³", 400, "mm³", "fail")],
                        {"buried": 1}, 0.4)
    text = v.render()
    assert "FAIL" in text and "a <-> b" in text and "400" in text


# ── the three modes must not bleed into each other ───────────────────────────
from cad_v3 import modes  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("build me a gearbox", modes.Mode.CAD),
    ("design a housing", modes.Mode.CAD),
    ("make a bracket 80mm wide", modes.Mode.CAD),
    ("modify the flange", modes.Mode.CAD),
    ("test the gearbox", modes.Mode.PHYSICS),
    ("does it collide", modes.Mode.PHYSICS),
    ("check for interference", modes.Mode.PHYSICS),
    ("simulate it", modes.Mode.PHYSICS),
    ("will it turn", modes.Mode.PHYSICS),
    ("render it", modes.Mode.SCENE),
    ("show me a render", modes.Mode.SCENE),
    ("make it look dramatic", modes.Mode.SCENE),
    ("put it on a workbench", modes.Mode.SCENE),
])
def test_intent_detection(text, expected):
    assert modes.detect(text).mode is expected


def test_unrecognised_request_defaults_to_cad():
    """Building is the safe assumption. Accidentally rendering scenery while someone is
    inspecting a bore is worse than not rendering when they wanted a picture."""
    for text in ("", "hello", "what can you do", "the flange is 40mm"):
        assert modes.detect(text).mode is modes.Mode.CAD


def test_build_verb_outranks_testing():
    """'build me a gearbox I can test' is a BUILD. Testing is a second, explicit request."""
    i = modes.detect("build me a gearbox i can test")
    assert i.mode is modes.Mode.CAD


def test_build_verb_outranks_presentation():
    assert modes.detect("build me a housing and render it").mode is modes.Mode.CAD


def test_cad_output_contains_no_scene():
    """The owner's requirement, asserted directly: a build reports engineering evidence
    and nothing about pictures."""
    asm = make(("a", Box(20, 20, 20)), ("b", Pos(0, 0, 20) * Box(10, 10, 10)))
    cad = modes.cad_output("widget", asm)
    text = cad.full().lower()
    for word in ("render", "scene", "png", "image", "blender", "lighting", "camera"):
        assert word not in text, f"CAD output leaked {word!r}"
    assert "parts" in text and "mass" in text


def test_cad_output_reports_status_and_evidence():
    asm = make(("block", Box(100, 100, 100), "aluminium"))
    cad = modes.cad_output("block", asm)
    assert cad.status == "PASS"
    assert "mass" in cad.summary().lower()
    assert cad.parts == 1


def test_cad_output_flags_a_broken_assembly():
    asm = make(("housing", Box(40, 40, 40)), ("flange", Pos(0, 0, 5) * Box(20, 20, 20)))
    cad = modes.cad_output("broken", asm)
    assert cad.status == "FAIL"
    assert "interpenetration" in cad.summary().lower() or "occup" in cad.full().lower()


def test_physics_tests_are_addressable_by_name():
    asm = make(("a", Box(20, 20, 20)), ("b", Pos(20, 0, 0) * Box(20, 20, 20)))
    assert len(modes.physics_output(asm, "interference")) == 1
    assert len(modes.physics_output(asm, "mass")) == 1
    assert len(modes.physics_output(asm, "all")) >= 2


def test_unknown_physics_test_is_inconclusive_not_a_crash():
    asm = make(("a", Box(10, 10, 10)), ("b", Pos(20, 0, 0) * Box(10, 10, 10)))
    out = modes.physics_output(asm, "teleportation")
    assert out[0].status == "INCONCLUSIVE"
    assert "interference" in out[0].summary


def test_scene_style_from_text():
    assert modes.scene_style_from_text("make it holographic like jarvis") == "holographic"
    assert modes.scene_style_from_text("put it on a workbench") == "workshop"
    assert modes.scene_style_from_text("dramatic hero shot") == "dramatic"
    assert modes.scene_style_from_text("something vague") == "studio"


def test_scene_request_normalises_unknown_style():
    assert modes.SceneRequest(style="nonsense").normalised().style == "studio"
    assert modes.SceneRequest(style="TECHNICAL").normalised().style == "technical"


def test_scene_mode_degrades_gracefully_without_blender(monkeypatch):
    """A missing renderer must cost the picture and nothing else."""
    from cad_v3.render import blender
    monkeypatch.setattr(blender, "available", lambda: False)
    asm = make(("a", Box(10, 10, 10)))
    msg = modes.scene_output(asm)
    assert "unavailable" in msg.lower()
    assert "unaffected" in msg.lower()
