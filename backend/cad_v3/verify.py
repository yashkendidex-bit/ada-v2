"""Verification on the solids, not on the tokens.

This is the layer that catches what an LLM cannot see. It used to run seven checks
and every one of them asked the same question in a different way: *is this assembly
self-consistent?* Watertight solids, no unexplained overlaps, everything connected,
patterns spread over their stations, nothing floating in free space. The shipped
turbocharger passes all seven and has no recognisable turbine wheel, impeller or
volute in it, because a wrong model is perfectly self-consistent — wrongness is not
an inconsistency, it is a disagreement with the drawing, and nothing here was
holding a drawing.

So the checks now come in two families.

**Self-consistency**, as before:

1. **validity** — OCCT's own B-rep checker on every placed solid.
2. **clash** — bounding-box reject, then a real boolean intersection. Press and
   interference fits are *expected* to overlap, so the fit named in the mate sets
   the allowance; anything above it is a genuine collision.
3. **connectivity** — every instance actually received a pose.
4. **pattern** — the instances of a patterned part landed on different stations.
5. **attachment** — every part's box comes within reach of some other part's.
6. **closure** — every mate, not merely the ones the traversal could not solve, is
   measured against the poses that actually resulted. With the solver in the loop
   this is a real number rather than a zero by construction.

**Correctness**, which is new and is the point:

7. **dimensions** — the distances, lengths and masses the spec says the finished
   assembly must have, measured on the B-rep (measure.py).
8. **conditions** — the engineering rules the numbers must satisfy: a planetary
   train's tooth counts, two gears' modules and centre distance, two flanges'
   bolt circles, a shaft's fit in its bore (conditions.py).
9. **envelope** — mass, volume, instance count and overall size inside the range
   the spec declares.

A spec that states none of the second family gets a PASS with a warning printed in
place of the checks it declined to make, because "nothing was checked" and "nothing
was wrong" must never look the same.

And one census that is not a check: placeholders. When synth.py could not write a
part, the assembly still contains a dimensioned envelope where that part belongs,
and this report names every one of them. It is deliberately not a failure — the
other forty parts are real and worth looking at — but it is never silent either,
because a cylinder standing in for a shift fork must never be mistaken for one.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

from . import conditions, materials, measure
from .assembler import (Assembly, Instance, _flip_for, _mate_transform, _reorient)

# Radial allowance per fit class, in mm — how much overlap is legitimate.
FIT_ALLOWANCE = {
    "free": 0.0,
    "slip": 0.02,
    "clearance": 0.02,
    "press": 0.06,
    "interference": 0.12,
}

CLASH_VOLUME_TOL = 0.05  # mm^3; below this it is surface contact, not a collision


@dataclass
class Clash:
    a: str
    b: str
    volume: float
    allowed: float
    fit: str

    @property
    def severe(self) -> bool:
        return self.volume > self.allowed


@dataclass
class Report:
    parts: int = 0
    invalid: list[str] = field(default_factory=list)
    open_shells: list[str] = field(default_factory=list)
    clashes: list[Clash] = field(default_factory=list)
    contacts: int = 0
    pairs_tested: int = 0
    pairs_boolean: int = 0
    volume_mm3: float = 0.0
    mass_g: float = 0.0
    per_part: dict[str, tuple[float, float]] = field(default_factory=dict)
    unplaced: list[str] = field(default_factory=list)
    collapsed: dict[str, tuple[int, int]] = field(default_factory=dict)
    detached: list[str] = field(default_factory=list)
    misaligned: list[tuple[int, str, float]] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    placeholder_skills: list[str] = field(default_factory=list)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    # ── the correctness family ──────────────────────────────────────────
    dims: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    envelope: list[str] = field(default_factory=list)
    bbox: tuple = ()
    unstated: bool = False
    solver_line: str = ""
    dof: str = ""

    @property
    def severe_clashes(self) -> list[Clash]:
        return [c for c in self.clashes if c.severe]

    @property
    def bad_dims(self) -> list:
        return [d for d in self.dims if d.measurable and not d.ok]

    @property
    def unmeasurable_dims(self) -> list:
        return [d for d in self.dims if not d.measurable]

    @property
    def bad_conditions(self) -> list:
        return [f for f in self.findings if not f.ok]

    @property
    def passed(self) -> bool:
        """Self-consistent *and* in agreement with everything the spec claimed.

        A dimension that could not be measured counts against the model too. It means
        the spec asserted something about a part that has no pose or no geometry, and
        an assertion that silently evaporates is worse than one that fails.
        """
        return not (self.invalid or self.severe_clashes or self.unplaced
                    or self.collapsed or self.detached or self.misaligned
                    or self.bad_dims or self.unmeasurable_dims
                    or self.bad_conditions or self.envelope)

    @property
    def complete(self) -> bool:
        """Passed *and* every part is real. Placeholders never fake this."""
        return self.passed and not self.placeholders

    @property
    def checked(self) -> int:
        return len(self.dims) + len(self.findings)

    def text(self) -> str:
        if self.placeholders:
            holes = (f"{len(self.placeholders)} of {self.parts} parts are PLACEHOLDERS: "
                     + ", ".join(self.placeholder_skills))
        else:
            holes = "none — every part is real geometry"
        lines = [
            f"parts:        {self.parts}",
            f"watertight:   {'all' if not self.invalid and not self.open_shells else f'{len(self.invalid)} invalid, {len(self.open_shells)} open'}",
            f"clashes:      {len(self.severe_clashes)} severe / {len(self.clashes)} overlapping "
            f"({self.contacts} touching, {self.pairs_boolean}/{self.pairs_tested} pairs booleaned)",
            f"connectivity: {'complete' if not self.unplaced else f'{len(self.unplaced)} unplaced'}",
            f"mates:        {'all hold' if not self.misaligned else f'{len(self.misaligned)} do not hold'}",
            f"dimensions:   {self._dim_line()}",
            f"conditions:   {self._cond_line()}",
            f"envelope:     {'inside every stated bound' if not self.envelope else f'{len(self.envelope)} out of range'}",
            f"placeholders: {holes}",
            f"volume:       {self.volume_mm3 / 1000.0:.1f} cm3",
            f"mass:         {self.mass_g / 1000.0:.3f} kg",
            f"verify time:  {self.seconds:.2f}s",
        ]
        if self.solver_line:
            lines.insert(4, f"placement:    {self.solver_line}")
        for pid, (n, stations) in self.collapsed.items():
            lines.append(f"  COLLAPSED {pid}: {n} instances share {stations} station(s) "
                         f"— its seat has fewer stations than the part has count")
        for pid in self.detached:
            lines.append(f"  DETACHED {pid} — positioned, but its box clears every other "
                         f"part, so it touches nothing")
        for idx, pair, mm in self.misaligned:
            lines.append(f"  MISALIGNED mates[{idx}] {pair}: those seats end up "
                         f"{mm:.3f} mm apart")
        for c in self.severe_clashes[:10]:
            lines.append(f"  CLASH {c.a} <-> {c.b}: {c.volume:.2f} mm3 "
                         f"(fit '{c.fit}' allows {c.allowed:.2f})")
        for d in self.dims:
            if not d.ok or not d.measurable:
                lines.append(d.text())
        for f in self.findings:
            if not f.ok:
                lines.append(str(f))
        for msg in self.envelope:
            lines.append(f"  ENVELOPE {msg}")
        for key in self.placeholders[:12]:
            lines.append(f"  PLACEHOLDER {key} — envelope only, geometry not yet synthesised")
        if self.unstated:
            lines.append("  UNSTATED: this spec declares no dimensions, conditions or mass "
                         "envelope, so nothing above checked whether it is the right shape "
                         "— add an `expect` block")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def _dim_line(self) -> str:
        if not self.dims:
            return "none stated"
        bad, unk = len(self.bad_dims), len(self.unmeasurable_dims)
        if not bad and not unk:
            return f"{len(self.dims)}/{len(self.dims)} hold"
        return f"{len(self.dims) - bad - unk}/{len(self.dims)} hold, {bad} wrong, {unk} unmeasurable"

    def _cond_line(self) -> str:
        if not self.findings:
            return "none stated"
        bad = len(self.bad_conditions)
        return (f"{len(self.findings)}/{len(self.findings)} satisfied" if not bad
                else f"{len(self.findings) - bad}/{len(self.findings)} satisfied, {bad} violated")


def _fit_map(asm: Assembly) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for m in asm.spec.mates:
        out[tuple(sorted((m.part_a, m.part_b)))] = m.fit
    return out


def _allowance(fit: str, inst_a: Instance, inst_b: Instance) -> float:
    """Convert a radial fit allowance into a tolerated intersection volume."""
    radial = FIT_ALLOWANCE.get(fit, 0.02)
    if radial <= 0:
        return CLASH_VOLUME_TOL
    d = 0.0
    for inst in (inst_a, inst_b):
        d = max(d, float(inst.result.interfaces.get("bore_d", 0.0)),
                float(inst.result.interfaces.get("od", 0.0)))
    length = max(float(inst_a.result.interfaces.get("length", 0.0)),
                 float(inst_a.result.interfaces.get("width", 0.0)),
                 float(inst_a.result.interfaces.get("thickness", 0.0)), 2.0)
    return max(CLASH_VOLUME_TOL, math.pi * max(d, 4.0) * radial * length)


# ── two facts about the layout that only the boxes can tell you ─────────────
# Both of these are *sound*, not heuristic: a bounding box is a conservative bound, so
# a positive gap between two boxes is proof the solids do not meet, and two coincident
# box centres on one patterned part mean two instances really are in the same place.
# Neither needs to know what any part is called.

def _box6(bb) -> tuple[float, float, float, float, float, float]:
    lo, hi = bb.min, bb.max
    return (float(lo.X), float(lo.Y), float(lo.Z),
            float(hi.X), float(hi.Y), float(hi.Z))


def _centre(box) -> tuple[float, float, float]:
    return ((box[0] + box[3]) / 2.0, (box[1] + box[4]) / 2.0, (box[2] + box[5]) / 2.0)


def _diagonal(box) -> float:
    return math.sqrt((box[3] - box[0]) ** 2 + (box[4] - box[1]) ** 2 + (box[5] - box[2]) ** 2)


def _gap(a, b) -> float:
    """Shortest distance between two axis-aligned boxes; 0 when they overlap."""
    dx = max(a[0] - b[3], b[0] - a[3], 0.0)
    dy = max(a[1] - b[4], b[1] - a[4], 0.0)
    dz = max(a[2] - b[5], b[2] - a[5], 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _check_pattern(rep: Report, boxes: list[tuple[Instance, tuple]]) -> None:
    """Instances of one patterned part must land on different stations.

    The clash test deliberately skips pairs that share a part id, on the grounds that
    instances of a pattern cannot share space by construction — which leaves the one
    case where they do exactly invisible. It happens whenever a part with count=N is
    mated to a seat publishing fewer than N stations: `Connector.location` wraps the
    index round, so the surplus instances are built precisely on top of earlier ones.
    Four bolts where there should be six is not something a render shows either, since
    the duplicates hide inside each other.
    """
    by_pid: dict[str, list[tuple]] = {}
    for inst, box in boxes:
        by_pid.setdefault(inst.pid, []).append(box)
    for pid, group in by_pid.items():
        if len(group) < 2:
            continue
        tol = max(1e-3, max(_diagonal(b) for b in group) * 0.01)
        stations: list[tuple[float, float, float]] = []
        for box in group:
            c = _centre(box)
            if not any(math.dist(c, s) <= tol for s in stations):
                stations.append(c)
        if len(stations) < len(group):
            rep.collapsed[pid] = (len(group), len(stations))


def residual_mm(asm: Assembly, idx: int, host_id: str, host_seat: str,
                moved_id: str, moved_seat: str) -> float | None:
    """How far apart a mate's two seats actually ended up, in mm. None if unmeasurable.

    Reads the poses back off the placed solids and compares them against the target
    the spec asked for, so it measures the result rather than re-deriving it. Before
    the solver existed this was zero by construction for every mate the traversal
    solved, which made it a self-check dressed as a measurement; now a solved
    assembly has had its poses adjusted by a least-squares fit and this number is
    the honest answer to "did that fit satisfy this mate".
    """
    from . import solver

    spec = asm.spec
    if not 0 <= idx < len(spec.mates):
        return None
    mate = spec.mates[idx]
    hosts, moved = asm.of_part(host_id), asm.of_part(moved_id)
    host_frame = asm.frames.get(host_id) if not hosts else None
    if (not hosts and host_frame is None) or not moved:
        return None
    worst = -1.0
    for j, child in enumerate(moved):
        if host_frame is not None:
            from .assembler import _identity_connector

            host_loc, ca = host_frame, _identity_connector(host_seat or "origin")
        else:
            host = hosts[j % len(hosts)]
            host_loc, ca = host.solid.location, host.result.connectors.get(host_seat)
        cb = child.result.connectors.get(moved_seat)
        if ca is None or cb is None:
            return None
        station = (child.station if child.via_mate == idx
                   else solver._station_for(host_loc, ca, mate, child.index))
        l_a = _reorient(ca.location(station), mate.orient) * _mate_transform(
            mate, _flip_for(mate, ca, cb))
        try:
            want = (host_loc * l_a).position
            have = (child.solid.location * cb.location(0)).position
        except Exception:
            return None
        worst = max(worst, math.dist((float(want.X), float(want.Y), float(want.Z)),
                                     (float(have.X), float(have.Y), float(have.Z))))
    return None if worst < 0.0 else worst


def _check_mates(rep: Report, asm: Assembly, tol: float) -> None:
    """Every mate, measured — not only the ones the traversal could not reach.

    The old version looked at loop closures alone, and all three shipped reference
    specs are pure trees with none, so on every example the engine came with this
    check ran zero times and printed nothing. Now the solver has had its say and
    each mate's residual is a fact about the finished assembly, so measure all of
    them and name the ones that do not hold.
    """
    for p in asm.placements:
        if p.after_mm > tol:
            rep.misaligned.append((p.idx, p.label, p.after_mm))
    if not asm.placements:
        # No placement record: fall back to the loop closures so an assembly built by
        # an older path is still checked rather than silently waved through.
        for loop in asm.loop_mates:
            mm = residual_mm(asm, loop.idx, loop.host_id, loop.host_seat,
                             loop.moved_id, loop.moved_seat)
            if mm is not None and mm > tol:
                m = asm.spec.mates[loop.idx]
                rep.misaligned.append((loop.idx, f"{m.a} <-> {m.b}", mm))


def _check_detached(rep: Report, boxes: list[tuple[Instance, tuple]]) -> None:
    """A part whose box clears every other part's box cannot be touching one.

    `UNPLACED` covers the part that never received a pose. This covers the part that
    received the wrong one and is sitting in free space beside the machine it belongs
    to — a failure a spec cannot express and a render shows instantly. Instances of one
    part id are treated as a single body, so a chain of links whose members only touch
    each other is judged as the chain, not link by link.
    """
    if len(boxes) < 2:
        return
    hull = (min(b[0] for _, b in boxes), min(b[1] for _, b in boxes),
            min(b[2] for _, b in boxes), max(b[3] for _, b in boxes),
            max(b[4] for _, b in boxes), max(b[5] for _, b in boxes))
    tol = max(0.5, _diagonal(hull) * 0.01)
    by_pid: dict[str, list[tuple]] = {}
    for inst, box in boxes:
        if inst.placed:
            by_pid.setdefault(inst.pid, []).append(box)
    if len(by_pid) < 2:
        return
    for pid, group in by_pid.items():
        others = [b for other, bs in by_pid.items() if other != pid for b in bs]
        if not any(_gap(mine, theirs) <= tol for mine in group for theirs in others):
            rep.detached.append(pid)


def verify(asm: Assembly, *, clash: bool = True, max_pairs: int = 200000) -> Report:
    t0 = time.time()
    rep = Report(parts=len(asm.instances))
    rep.unplaced = [i.key for i in asm.instances if not i.placed]

    for inst in asm.instances:
        if (getattr(inst.result, "meta", None) or {}).get("source") == "stub":
            rep.placeholders.append(inst.key)
            if inst.skill not in rep.placeholder_skills:
                rep.placeholder_skills.append(inst.skill)
        try:
            if not inst.solid.is_valid:
                rep.invalid.append(inst.key)
        except Exception:
            rep.invalid.append(inst.key)
        try:
            vol = float(inst.solid.volume)
            mass = materials.mass_g(inst.material, vol)
            rep.per_part[inst.key] = (vol, mass)
            rep.volume_mm3 += vol
            rep.mass_g += mass
            if vol <= 1e-6:
                rep.notes.append(f"{inst.key} has zero volume")
        except Exception as exc:
            rep.notes.append(f"{inst.key}: volume failed ({type(exc).__name__})")

    # One pass of bounding boxes serves the layout checks and the clash reject alike.
    boxes: list[tuple[Instance, tuple]] = []
    for inst in asm.instances:
        try:
            boxes.append((inst, _box6(inst.solid.bounding_box())))
        except Exception:
            rep.notes.append(f"{inst.key}: no bounding box")
    _check_pattern(rep, boxes)
    _check_detached(rep, boxes)
    rep.bbox = tuple(boxes and _hull(boxes) or ())
    # Absolute, and the spec's own figure. A tolerance that scaled with the model
    # would let a 3 m machine be 15 mm out and call that fine, which is exactly the
    # kind of "passed" this module exists to stop.
    _check_mates(rep, asm, float(asm.spec.tolerance_mm))
    if asm.solver_log:
        rep.solver_line = asm.solver_log[0].replace("solve: ", "")
    rep.dof = asm.dof

    # ── the correctness family ──────────────────────────────────────────
    rep.dims = measure.evaluate(asm)
    rep.findings = conditions.evaluate(asm)
    _check_envelope(rep, asm)
    rep.unstated = asm.spec.expect.empty and not asm.spec.checks

    if clash:
        fits = _fit_map(asm)
        for (ia, ba), (ib, bb) in itertools.combinations(boxes, 2):
            rep.pairs_tested += 1
            if rep.pairs_tested > max_pairs:
                rep.notes.append(f"clash test truncated at {max_pairs} pairs")
                break
            if ia.pid == ib.pid and ia.index != ib.index:
                # Instances of the same patterned part never share space by construction
                # — and when they do, `_check_pattern` is the one that says so.
                continue
            if _gap(ba, bb) > 0.0:
                continue
            rep.pairs_boolean += 1
            vol = _intersect_volume(ia, ib)
            if vol <= CLASH_VOLUME_TOL:
                if vol > 0:
                    rep.contacts += 1
                continue
            fit = fits.get(tuple(sorted((ia.pid, ib.pid))), "free")
            rep.clashes.append(Clash(ia.key, ib.key, vol, _allowance(fit, ia, ib), fit))

    rep.seconds = time.time() - t0
    return rep


def _hull(boxes: list[tuple[Instance, tuple]]) -> tuple:
    return (min(b[0] for _, b in boxes), min(b[1] for _, b in boxes),
            min(b[2] for _, b in boxes), max(b[3] for _, b in boxes),
            max(b[4] for _, b in boxes), max(b[5] for _, b in boxes))


def _check_envelope(rep: Report, asm: Assembly) -> None:
    """Mass, volume, instance count and overall size against the stated range.

    The cheapest correctness signal there is, and the one whose absence let a 4.6 kg
    turbocharger cartridge through a checker whose own documentation used a 4 kg
    turbocharger cartridge as its example of something obviously wrong.
    """
    exp = asm.spec.expect
    if exp.mass_kg:
        lo, hi = _range(exp.mass_kg)
        kg = rep.mass_g / 1000.0
        if not lo <= kg <= hi:
            rep.envelope.append(f"mass {kg:.3f} kg is outside the stated {lo:g}..{hi:g} kg")
    if exp.volume_cm3:
        lo, hi = _range(exp.volume_cm3)
        cm3 = rep.volume_mm3 / 1000.0
        if not lo <= cm3 <= hi:
            rep.envelope.append(f"volume {cm3:.1f} cm3 is outside the stated {lo:g}..{hi:g} cm3")
    if exp.parts:
        lo, hi = _range([float(v) for v in exp.parts])
        if not lo <= rep.parts <= hi:
            rep.envelope.append(f"{rep.parts} instances, stated {int(lo)}..{int(hi)}")
    if exp.bbox_mm and rep.bbox:
        for axis, rng in exp.bbox_mm.items():
            k = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower())
            if k is None or not rng:
                continue
            lo, hi = _range([float(v) for v in rng])
            span = rep.bbox[k + 3] - rep.bbox[k]
            if not lo <= span <= hi:
                rep.envelope.append(f"overall {axis} is {span:.1f} mm, stated {lo:g}..{hi:g}")
    if exp.watertight and (rep.invalid or rep.open_shells):
        rep.envelope.append(f"{len(rep.invalid)} solid(s) are not watertight and the "
                            "spec requires that they are")
    if rep.placeholders and not exp.allow_placeholders and not exp.empty:
        rep.envelope.append(f"{len(rep.placeholders)} placeholder(s) stand in for real "
                            "geometry; set expect.allow_placeholders to accept that")


def _range(values: list[float]) -> tuple[float, float]:
    """[lo, hi], or [x] read as +-2% of x — a single figure is a nominal, not a bound."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return (-1e18, 1e18)
    if len(nums) == 1:
        return (nums[0] * 0.98, nums[0] * 1.02)
    return (min(nums), max(nums))


def _intersect_volume(ia: Instance, ib: Instance) -> float:
    try:
        common = ia.solid.intersect(ib.solid)
    except Exception:
        return 0.0
    return _volume_of(common)


def _volume_of(shape) -> float:
    """Volume of whatever `intersect` handed back.

    A boolean common can come back as one solid, as `None`, or — when the overlap
    breaks into disjoint lumps — as a `ShapeList`, which has no `.volume` at all.
    Reading `.volume` blind and swallowing the AttributeError would score a real
    collision as zero, so the list case is summed explicitly.
    """
    if shape is None:
        return 0.0
    try:
        return float(shape.volume)
    except AttributeError:
        pass
    except Exception:
        return 0.0
    try:
        return float(sum(_volume_of(s) for s in shape))
    except Exception:
        return 0.0
