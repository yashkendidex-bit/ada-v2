"""The checks that catch what neither a schema nor a model can see, pinned.

Every case here is a real failure from a real run, and every fixture is either the
shipped turbocharger spec with one field moved or ADA's own rotor-head spec recorded
verbatim (`fixtures/rotor_head.json`, authored by the design loop on 2026-08-25). No
assembly is designed here; the geometry is only the instrument.

What is being defended:

  * `residual_mm` recomposes the assembler's placement algebra, so it must read exactly
    zero for every mate the traversal solved. That makes the loop-closure check a live
    test of `_reorient`/`_mate_transform` as well as of the spec — and it is the reason
    a 467 mm design error was finally named instead of chased as a clash 300 mm away.
  * a pattern with more parts than its seat has stations stacks the surplus copies
    inside earlier ones, which no render and no clash test will ever show;
  * a part flung into free space is `placed`, valid, and clash-free;
  * X-ray has to find internals by measurement now that the list of housing skill
    names is gone;
  * `seat_hints` decides a stub's seat layout by arithmetic on the spec, replacing a
    list of fourteen nouns that worked only for the machines whose parts were on it;
  * severity is how far past its fit the metal goes, not how many places it does so —
    ranking by count threw away the one correct fix the repair loop ever found.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .. import assembler, bootstrap, repair, synth, verify
from ..schema import Spec

HERE = Path(__file__).resolve().parent
TURBO = HERE.parent / "specs" / "turbocharger.json"
ROTOR = HERE / "fixtures" / "rotor_head.json"

# The rotor stack: mates 0..10 hang twelve turned parts off the datum shaft, all
# count=1, no volutes — a whole real assembly for about four seconds of kernel. Add
# 11..13 and the bearing housing wraps them, which is what X-ray has to discover.
STACK = list(range(0, 11))
HOUSED = list(range(0, 14))

_WORKER = None
_CACHE: dict[str, object] = {}


def _raw() -> dict:
    bootstrap()
    return json.loads(TURBO.read_text())


def _sub(keep, extra=(), tweak=None) -> Spec:
    """The turbocharger cut down to the mates in `keep`, plus whatever `tweak` breaks."""
    raw = _raw()
    mates = [raw["mates"][i] for i in keep] + [copy.deepcopy(m) for m in extra]
    ids = {raw["datum"]}
    for m in mates:
        ids.add(m["a"].split(".")[0])
        ids.add(m["b"].split(".")[0])
    raw["mates"] = mates
    raw["parts"] = [p for p in raw["parts"] if p["id"] in ids]
    if tweak:
        tweak(raw)
    return Spec.from_dict(raw)


def _run(spec: Spec, *, clash: bool = False):
    global _WORKER
    if _WORKER is None:
        _WORKER = assembler.Worker()
    asm = assembler.build(spec, worker=_WORKER, strict=False)
    return asm, verify.verify(asm, clash=clash)


def _built(keep, extra=(), tweak=None, *, key: str, clash: bool = False):
    """Cached, because several tests want the same fifteen solids."""
    if key not in _CACHE:
        _CACHE[key] = _run(_sub(keep, extra, tweak), clash=clash)
    return _CACHE[key]


def _child_seat(spec: Spec, idx: int, pid: str) -> str:
    mate = spec.mates[idx]
    ref = mate.a if mate.part_a == pid else mate.b
    return mate.split(ref)[1]


# ── the placement algebra, measured against itself ──────────────────────────

def test_every_solved_mate_closes_to_zero():
    """The invariant the whole engine rests on: `host . l_a == child . l_b`.

    `residual_mm` composes that product from the spec and the connectors, not from the
    poses the kernel wrote, so agreement to the micron is a real check on `_reorient`,
    `_mate_transform`, the flip rule and the station indexing at once. If anyone ever
    changes one of those without changing the other, this is what says so.
    """
    asm, rep = _built(HOUSED, key="housed")
    assert not asm.errors, [e.code for e in asm.errors]
    assert not rep.unplaced and not rep.invalid, rep.text()
    solved = [i for i in asm.instances if i.via_mate >= 0]
    assert len(solved) == len(asm.instances) - 1, "only the datum places itself"
    for inst in solved:
        host = asm.by_key(inst.via_host)
        mm = verify.residual_mm(asm, inst.via_mate, host.pid, inst.via_seat,
                                inst.pid, _child_seat(asm.spec, inst.via_mate, inst.pid))
        assert mm is not None and mm < 1e-6, f"{inst.key} via mates[{inst.via_mate}]: {mm}"


def test_a_loop_closure_that_holds_reads_zero():
    """Over-constraint is normal, so restating a solved mate must not raise an alarm."""
    honest = copy.deepcopy(_raw()["mates"][4])          # collar_r.front_face -> thrust_plate
    asm, rep = _built(STACK, extra=[honest], key="honest")
    assert len(asm.loop_mates) == 1, asm.loop_mates
    assert rep.misaligned == [], rep.text()
    loop = asm.loop_mates[0]
    assert verify.residual_mm(asm, loop.idx, loop.host_id, loop.host_seat,
                              loop.moved_id, loop.moved_seat) < 1e-6


def test_a_loop_closure_that_lies_is_measured_not_ignored():
    """The rotor-head defect in miniature: a coincidence nothing made true.

    The assembler cannot re-solve a mate onto an already-placed part, and used only to
    count it in a log line. Moving the restated offset 40 mm asserts a coincidence that
    is 40 mm from happening, and the report has to say so — otherwise the only symptom
    is a clash somewhere else, and every sizing rule in `repair` gets spent on it.
    """
    liar = copy.deepcopy(_raw()["mates"][4])
    liar["offset_mm"] = float(liar.get("offset_mm", 0.0)) + 40.0
    asm, rep = _built(STACK, extra=[liar], key="liar")
    assert len(asm.loop_mates) == 1
    assert len(rep.misaligned) == 1, rep.text()
    idx, pair, mm = rep.misaligned[0]
    assert idx == len(asm.spec.mates) - 1 and "collar_r" in pair
    assert abs(mm - 40.0) < 1e-6, mm
    assert not rep.passed, "a stated coincidence that does not hold is a failure"


# ── the layout facts only the boxes can tell you ────────────────────────────

def test_a_pattern_larger_than_its_seat_collapses_and_is_caught():
    """Six stations, eight studs: `Connector.location` wraps, so two pairs coincide.

    Nothing upstream objects — the schema is happy, the seat exists, the solids are
    valid, and the duplicates hide inside each other in every render. The clash test
    skips same-part pairs by construction, so this check is the only thing standing
    between a wrapped pattern and a shipped assembly.
    """
    ring = STACK + [11, 12, 18]                      # ... -> turbine_flange.bolt_circle

    def eight(raw):
        for p in raw["parts"]:
            if p["id"] == "stud_t":
                p["count"] = 8

    asm, rep = _run(_sub(ring, tweak=eight))
    assert not asm.errors, [e.code for e in asm.errors]
    assert rep.collapsed == {"stud_t": (8, 6)}, rep.text()
    assert not rep.passed

    honest, ok = _run(_sub(ring))
    assert not ok.collapsed, ok.text()               # and six studs are not flagged
    assert len(honest.of_part("stud_t")) == 6


def test_a_part_flung_into_free_space_is_caught():
    """`UNPLACED` covers the part with no pose. This covers the wrong pose.

    A 400 mm offset on the last mate leaves the nut valid, watertight, clash-free and
    standing in mid-air beside the machine — which is exactly what a bad `offset_mm`
    looks like, and what a positive box gap proves rather than guesses.
    """
    def fling(raw):
        mate = next(m for m in raw["mates"] if m["b"].startswith("nut_n."))
        mate["offset_mm"] = 400.0

    asm, rep = _run(_sub(STACK, tweak=fling))
    assert not asm.errors and not rep.unplaced and not rep.invalid
    assert rep.detached == ["nut_n"], rep.text()
    assert not rep.passed


def test_xray_finds_internals_by_measurement():
    """No skill-name list survives, so enclosure has to be arithmetic on the boxes.

    The bearing housing is a hollow shell: a boolean containment test reports it as
    holding nothing, which is why coverage is sampled against the union of the *boxes*
    of every larger neighbour. What must come out is the running gear, and what must
    never come out is the casing that wraps it or the wheels hanging outside it.
    """
    asm, _rep = _built(HOUSED, key="housed")
    inner = assembler.inner_keys(asm)
    for gear in ("jb1", "jb2", "collar_r", "collar_f", "thrust_plate", "seal_t"):
        assert gear in inner, sorted(inner)
    for shell in ("bearing_housing", "turbine_flange", "back_plate"):
        assert shell not in inner, sorted(inner)
    assert "turbine_wheel" not in inner and "impeller" not in inner, sorted(inner)


# ── a stub's seats, laid out by arithmetic instead of by a list of nouns ─────

def _rotor() -> Spec:
    bootstrap()
    return Spec.from_dict(json.loads(ROTOR.read_text()))


def test_seat_hints_reads_the_layout_off_the_spec():
    """ADA's own rotor head, verbatim, is the case the noun list was hiding.

    `hub.bolt_circle` carries four grips, and each grip carries a 1.5 m blade: the limb
    reaches 1625 mm on a 170 mm ring, so those stations have to face out of the axis or
    the blades stand up the mast and no offset will ever fix it. The two genuine bolt
    circles on the same spec carry a 52 mm link and a 59 mm actuator and must stay
    axial — bolts stand up out of a flange, exactly as they should. And `mast_shaft.seats`
    serves three *different* parts, which is a sequence along the shaft, not a ring.
    """
    spec = _rotor()
    assert synth.seat_hints(spec, "hub").get("bolt_circle") == "radial"
    assert synth.seat_hints(spec, "mast_shaft").get("seats") == "stack"
    assert synth.seat_hints(spec, "rot_ring").get("bolt_circle") == "ring"
    assert synth.seat_hints(spec, "mast_housing").get("bolt_circle") == "ring"
    assert synth.seat_hints(spec, "blade_grip") == {}, "single-use seats need no layout"
    assert synth.seats_of(spec, "hub") == {"bore": 1, "bolt_circle": 4}


def test_reach_is_the_whole_limb_not_the_one_part():
    """A 60 mm grip carrying a 1500 mm blade is a 1560 mm limb.

    Measured down the BFS placement tree rather than the mate list, because the mates
    form a graph with loops in it and "what hangs off this" is only defined once the
    traversal has chosen a parent for everything.
    """
    reach = synth._subtree_reach(_rotor())
    assert reach["blade_grip"] > 1500.0, reach
    assert reach["hub"] > reach["blade_grip"]
    assert reach["pitch_link"] < 100.0, reach


def test_no_engine_module_keeps_a_list_of_part_nouns():
    """The promise that broke twice: `RADIAL_SEAT_WORDS`, `XRAY_FEATURE_SKILLS`.

    Both were lists of names that worked for the one machine whose parts were on them.
    Prose about them is fine — this walks the modules' actual values, so a comment
    explaining the removal cannot trip it and a re-introduction under any name will.
    """
    nouns = {"blade", "spoke", "vane", "trunnion", "grip", "lug", "paddle", "bucket",
             "fin", "boom", "spigot", "housing", "casing", "shroud", "volute"}
    for mod in (synth, assembler, verify, repair):
        for name, value in vars(mod).items():
            if not name.isupper() or not isinstance(value, (set, frozenset, list, tuple)):
                continue
            words = {v.lower() for v in value if isinstance(v, str)}
            assert not (words & nouns), f"{mod.__name__}.{name} names parts: {words & nouns}"
        assert not hasattr(mod, "RADIAL_SEAT_WORDS")
        assert not hasattr(mod, "XRAY_FEATURE_SKILLS")


# ── how the repair loop ranks two imperfect specs ───────────────────────────

def _attempt(clashes, *, broken: int = 0, rnd: int = 1) -> repair.Attempt:
    """An attempt with nothing built: `score` only ever reads the report."""
    if "bare" not in _CACHE:
        _CACHE["bare"] = Spec.from_dict(_raw())
    spec = _CACHE["bare"]
    rep = verify.Report(parts=1, clashes=[
        verify.Clash("a", "b", vol, allowed, "press") for vol, allowed in clashes])
    rep.detached = [f"p{i}" for i in range(broken)]
    return repair.Attempt(rnd, spec, assembler.Assembly(spec=spec), rep)


def test_severity_is_volume_past_the_fit_not_the_number_of_clashes():
    """The rotor head found the one correct fix and threw it away on this ordering.

    Laying four blades out into the disc plane traded a 59 cm3 interpenetration for four
    press fits a fraction over their allowance. Count-first read that as 2 clashes
    becoming 5 and kept the spec where the blades pass through the mast. Two parts
    occupying the same metal is not better than five joints being tight.
    """
    passing_through = _attempt([(59000.0, 200.0), (400.0, 200.0)])
    laid_out = _attempt([(380.0, 200.0)] * 5)
    assert laid_out.score < passing_through.score, (laid_out.score, passing_through.score)
    assert not laid_out.clean and not passing_through.clean
    assert _attempt([(150.0, 200.0)] * 5).clean, "inside its fit is not a clash"


def test_a_part_in_the_wrong_place_outranks_any_amount_of_overlap():
    """A collapsed pattern or a detached part is a pose error, not a sizing error."""
    misplaced = _attempt([], broken=1)
    overlapping = _attempt([(90000.0, 100.0)] * 3)
    assert overlapping.score < misplaced.score, (overlapping.score, misplaced.score)


def test_progress_retires_a_knob_that_only_shaves_rounding_error():
    """Three rounds were spent growing one bore by 0.15 mm at a time.

    Each nudge removed about two parts in a thousand of a 51 cm3 overlap whose cause was
    a ring mounted off-axis — measurably "improving" while the actual defect went
    unexamined. So the test is relative: the tiers that count parts must move, or the
    unexplained volume must fall by a real fraction of itself.
    """
    prev = _attempt([(51000.0, 100.0)])
    assert not repair._progress(_attempt([(50900.0, 100.0)]), prev), "0.2% is not progress"
    assert repair._progress(_attempt([(45000.0, 100.0)]), prev), "12% is"
    assert repair._progress(_attempt([(51000.0, 100.0)]), _attempt([], broken=1)) is True
    # Same volume, one fewer place it happens: still worth keeping.
    assert repair._progress(_attempt([(25000.0, 100.0), (26000.0, 100.0)]),
                            _attempt([(25000.0, 100.0), (26000.0, 100.0), (200.0, 100.0)]))


# ── one whole part on one station of a ring ─────────────────────────────────

def _rotor_built():
    if "rotor" not in _CACHE:
        spec = _rotor()
        asm = assembler.build(spec, strict=False)
        _CACHE["rotor"] = (spec, asm, verify.verify(asm, clash=False))
    return _CACHE["rotor"]


def test_a_single_part_on_one_station_of_a_ring_is_named_as_the_cause():
    """The 467 mm defect in ADA's own rotor head, and why it is arithmetic.

    The fixed swashplate ring is one part mated to `actuator.front_face` — one actuator of
    three, standing on a bolt circle 120 mm off the axis. The machine means all three carry
    it and it is centred; the spec cannot say "all three", so the ring is built 120 mm out
    and half a metre down, and the mate that separately declares its bore to run on the
    mast is left 467 mm from true. Four pitch links on a four-station ring are the same
    shape of statement and are *correct*, so counts are what tell them apart.
    """
    spec, asm, rep = _rotor_built()
    assert len(rep.misaligned) == 2, rep.text()
    idx = repair._pattern_artefact(spec, asm, "fixed_ring")
    assert idx >= 0 and spec.mates[idx].part_a == "actuator", spec.mates[idx].a
    assert repair._pattern_artefact(spec, asm, "pitch_link") == -1, "4 links, 4 stations"
    assert repair._pattern_artefact(spec, asm, "mast_housing") == -1, "on the axis already"


def test_the_ring_is_not_repaired_by_stacking_four_links_on_one_point():
    """A part with count N on a seat of N stations is off the axis on purpose.

    Re-hosting it onto an on-axis face would put all N copies in one place — which the
    pattern check would then report as a collapse, trading a stated coincidence for a
    hidden one. So the re-host lever is offered for single parts only, and the dimensional
    disagreement is handed to the brain as one precise question instead.
    """
    spec, asm, _rep = _rotor_built()
    assert repair._loop_patches(spec, asm, "pitch_link") == []
    assert repair._loop_patches(spec, asm, "blade_grip") == []
    loops = [d for d in repair.diagnose(spec, asm, _rep) if d.code == "LOOP_MISALIGNED"]
    assert len(loops) == 2, loops
    asked = [d for d in loops if not d.patches]
    assert len(asked) == 1 and "pitch_link" in asked[0].question


def test_dropping_the_artefact_mate_makes_the_coincidence_true():
    """The repair the loop now finds unaided: 110315 mm3 of overlap down to 730.

    Deleting the mate that hung the ring on one actuator leaves the violated closure as
    the only path to it, so the traversal places the ring *by* that mate and the stated
    coincidence becomes true by construction rather than by arithmetic — and the 51 cm3
    interpenetration against the mast, 300 mm away, goes with it.
    """
    spec, asm, _rep = _rotor_built()
    patches = repair._loop_patches(spec, asm, "fixed_ring")
    assert patches and patches[0].op == "del_mate", patches
    fixed, err = repair.apply_patch(spec, patches[0])
    assert not err, err
    after = verify.verify(assembler.build(fixed, strict=False), clash=True)
    assert not any("fixed_ring" in pair for _idx, pair, _mm in after.misaligned), after.text()
    assert not any({c.a, c.b} == {"mast_shaft", "fixed_ring"} for c in after.severe_clashes)
    assert sum(max(c.volume - c.allowed, 0.0) for c in after.severe_clashes) < 5000.0





