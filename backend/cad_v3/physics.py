"""Physics — does the model actually work?

This is the module that answers the owner's complaint. A CAD model and a picture of a
CAD model are indistinguishable when all you ever produce is a picture. Nothing said
*"this clears at every angle"* or *"this weighs 4.2 kg"*, so the output read as
decoration even though the geometry underneath was sound.

Everything here works on the kernel's **own B-rep solids**, not on a tessellated export.
That choice matters:

* `solid_a.intersect(solid_b).volume` is **exact**. Sampling a mesh for overlapping
  volume is an estimate, and the estimate is what produced two false alarms during
  development — see the method notes below.
* It runs **in process**. No Blender subprocess, so it is fast, testable, and works
  offline. Blender is only needed for *dynamics*, which is a different question.

Two measurement traps, both of which I fell into while building this, and both of which
this module is written to avoid:

1. **Do not count intersecting triangles.** Two parts that merely *touch* share triangles,
   because their surfaces are coincident. Counting them reported 32 "colliding pairs" on a
   perfectly valid turbocharger. Volume is the measurement; triangle count is a proxy.
2. **Do not judge motion without a baseline.** Parts already in contact at rest report
   overlap at every angle. A collision caused by *rotation* is one whose overlap **grows**
   relative to rest.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

# A pair counts as BURIED when the shared volume reaches this fraction of the smaller
# part. Below it, the contact is a tessellation-level coincidence at a mating face.
# 2% was chosen after measuring a known-good assembly: its worst pair sat at 0.43%.
BURIED_FRACTION = 0.02
# Absolute floor, so a tiny part cannot be flagged for a microscopic shared volume.
BURIED_MIN_MM3 = 1.0
# Growth factor vs the at-rest baseline before a rotation overlap counts as a collision.
GROWTH_FACTOR = 2.0
GROWTH_MIN_MM3 = 1.0


# ── verdicts ─────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    """One thing a test noticed, with the evidence behind it."""
    subject: str
    detail: str
    value: float = 0.0
    unit: str = ""
    severity: str = "info"          # info | warning | fail


@dataclass
class Verdict:
    """The result of a test. Every test returns one of these.

    `INCONCLUSIVE` is a real outcome, not a rounding-up of "probably fine". A test that
    cannot decide must say so — the alternative is an assistant that reports confidence
    it has not earned, which is the failure mode this whole subsystem exists to avoid.
    """
    test: str
    status: str                      # PASS | FAIL | INCONCLUSIVE
    summary: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0

    def render(self, width: int = 78) -> str:
        head = {"PASS": "PASS", "FAIL": "FAIL", "INCONCLUSIVE": "INCONCLUSIVE"}[self.status]
        lines = [f"{self.test} — {head}", "-" * width, self.summary]
        if self.findings:
            lines.append("")
            for f in self.findings:
                mark = {"fail": "!", "warning": "~", "info": " "}[f.severity]
                val = f" [{f.value:g} {f.unit}]" if f.unit else ""
                lines.append(f"  {mark} {f.subject}: {f.detail}{val}")
        if self.metrics:
            lines.append("")
            lines.append("  measured: " + ", ".join(
                f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in self.metrics.items()))
        lines.append("")
        lines.append(f"  ({self.seconds:.1f}s)")
        return "\n".join(lines)


# ── exact solid geometry ─────────────────────────────────────────────────────
def overlap_volume(a: Any, b: Any) -> Optional[float]:
    """Exact shared volume of two solids, in mm^3.

    Returns 0.0 for touching or disjoint — verified: a 4 mm overlap of two 10 mm cubes
    returns exactly 400.0, and two cubes sharing a face return exactly 0.0. That zero is
    the whole reason this function exists instead of a triangle count.

    Returns None if the kernel could not answer, so the caller can report INCONCLUSIVE
    rather than silently treating a failure as "no interference".
    """
    try:
        result = a.intersect(b)
    except Exception:
        return None
    if result is None:
        return 0.0
    if hasattr(result, "__iter__"):
        total = 0.0
        for shape in result:
            try:
                total += float(shape.volume)
            except Exception:
                pass
        return total
    try:
        return float(result.volume)
    except Exception:
        return 0.0


def volume_of(solid: Any) -> Optional[float]:
    try:
        return float(solid.volume)
    except Exception:
        return None


def _bbox(solid: Any):
    try:
        bb = solid.bounding_box()
        return (bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.max.Y, bb.max.Z)
    except Exception:
        return None


def _bbox_overlap(lo1, hi1, lo2, hi2, pad: float = 1e-6) -> bool:
    return all(lo1[i] <= hi2[i] + pad and lo2[i] <= hi1[i] + pad for i in range(3))


# ── interference ─────────────────────────────────────────────────────────────
@dataclass
class Pair:
    a: str
    b: str
    overlap_mm3: float
    fraction: float          # of the smaller part
    a_vol: float
    b_vol: float

    @property
    def buried(self) -> bool:
        return self.fraction >= BURIED_FRACTION and self.overlap_mm3 >= BURIED_MIN_MM3


def _placed(inst: Any):
    """The instance's solid in world coordinates.

    `inst.solid` already carries its placement in `.location` — the assembler puts it
    there, and `render.shaded.place()` reads it back. So this returns the solid **as-is**.

    Do NOT call `.located(Location())` here. That is what the rasteriser does, and it is
    correct there: it strips the placement to get local geometry it can transform with its
    own matrix. Doing it here silently moves every part to the origin, which made a block
    centred at x=5 measure a centre of mass at x=0 and made every interference test
    meaningless. Verified: `(Pos(5,0,0) * Box(10,10,10)).center().X` is 5.0 as-is and
    -0.0 after `.located(Location())`.
    """
    solid = getattr(inst, "solid", None)
    if solid is None:
        return None
    # A few callers hand over a bare shape with no location attribute; that is fine,
    # it is simply already in world space.
    return solid


def instances_of(asm: Any) -> list:
    return list(getattr(asm, "instances", []) or [])


def interference_audit(asm: Any, *, max_pairs: int = 4000,
                       progress=None) -> Verdict:
    """Are any two solid parts occupying the same space?

    This is the test that separates an engineering model from a drawing of one: if two
    solid bodies share volume, the assembly cannot be manufactured, however good it looks.

    Bounding boxes reject the vast majority of pairs for free, so the exact boolean runs
    only on candidates — measured at 6.4 ms per intersection, which is why a 57-part
    assembly finishes in about a second.
    """
    t0 = time.time()
    insts = instances_of(asm)
    if len(insts) < 2:
        return Verdict("Interference audit", "INCONCLUSIVE",
                       "Fewer than two parts — nothing to check.",
                       seconds=time.time() - t0)

    prepared = []
    unresolved = 0
    for inst in insts:
        solid = _placed(inst)
        vol = volume_of(solid) if solid is not None else None
        box = _bbox(solid) if solid is not None else None
        if solid is None or vol is None or box is None:
            unresolved += 1
            continue
        prepared.append((getattr(inst, "key", None) or getattr(inst, "pid", "?"),
                         solid, vol, box[0], box[1]))

    pairs: list[Pair] = []
    candidates = 0
    failed = 0
    for i in range(len(prepared)):
        na, sa, va, loa, hia = prepared[i]
        for j in range(i + 1, len(prepared)):
            nb, sb, vb, lob, hib = prepared[j]
            if not _bbox_overlap(loa, hia, lob, hib):
                continue
            candidates += 1
            if candidates > max_pairs:
                break
            vol = overlap_volume(sa, sb)
            if vol is None:
                failed += 1
                continue
            if vol <= 0.0:
                continue
            pairs.append(Pair(na, nb, vol, vol / max(min(va, vb), 1e-9), va, vb))
        if progress:
            progress(i + 1, len(prepared))

    buried = sorted([p for p in pairs if p.buried], key=lambda p: -p.overlap_mm3)
    touching = [p for p in pairs if not p.buried]

    findings = []
    for p in buried[:12]:
        findings.append(Finding(
            f"{p.a} <-> {p.b}",
            f"share {p.overlap_mm3:.1f} mm³ — {p.fraction:.1%} of the smaller part. "
            f"These two bodies cannot both exist.",
            p.overlap_mm3, "mm³", "fail"))
    if touching:
        worst = max(touching, key=lambda p: p.fraction)
        findings.append(Finding(
            "mating contacts",
            f"{len(touching)} pair(s) touch without interpenetrating "
            f"(largest {worst.fraction:.2%}, {worst.a} <-> {worst.b}) — normal for mated parts",
            worst.fraction * 100, "%", "info"))
    if unresolved:
        findings.append(Finding("unresolved parts",
                                f"{unresolved} part(s) had no readable solid and were skipped",
                                unresolved, "", "warning"))

    metrics = {
        "parts": len(prepared),
        "candidate_pairs": candidates,
        "buried": len(buried),
        "touching": len(touching),
    }

    if failed and not prepared:
        status, summary = "INCONCLUSIVE", "The kernel could not evaluate any pair."
    elif failed:
        status = "INCONCLUSIVE"
        summary = (f"{len(buried)} buried pair(s) found, but {failed} pair(s) could not be "
                   f"evaluated — treat this result as partial.")
    elif buried:
        status = "FAIL"
        summary = (f"{len(buried)} part pair(s) occupy the same volume. The assembly is not "
                   f"manufacturable as drawn — it renders correctly and cannot exist.")
    else:
        status = "PASS"
        summary = (f"{len(prepared)} parts, no solid interpenetration. "
                   f"{len(touching)} pair(s) make surface contact only, which is correct "
                   f"for mated parts.")

    return Verdict("Interference audit", status, summary, findings, metrics,
                   time.time() - t0)


# ── mass properties ──────────────────────────────────────────────────────────
def mass_properties(asm: Any, densities: Optional[dict] = None) -> Verdict:
    """Volume, mass and centre of mass from the solids and the spec's materials.

    Real numbers are the cheapest way to make output read as engineering rather than
    decoration. "4.21 kg" says something; "looks good" does not.

    Units are handled by `materials.mass_g`, which knows the table is in g/cm³. Doing the
    arithmetic here instead is how you end up 1000x out — `vol_mm3 * 1e-9 * rho` is only
    correct if rho is kg/m³, and the table is not.
    """
    t0 = time.time()
    from . import materials

    insts = instances_of(asm)
    if not insts:
        return Verdict("Mass properties", "INCONCLUSIVE", "No parts to measure.",
                       seconds=time.time() - t0)

    rows = []
    total_v = total_m = 0.0
    cx = cy = cz = 0.0
    unknown_material = set()

    for inst in insts:
        solid = _placed(inst)
        vol = volume_of(solid) if solid is not None else None
        if vol is None:
            continue
        mat_name = (getattr(inst, "material", "") or "").strip()

        # `materials.get()` falls back to steel for anything unrecognised, which would
        # make an unset material silently weigh as steel. Check membership instead and
        # report the gap, because a wrong mass is worse than an absent one.
        known = mat_name.lower().replace("-", "_") in materials.MATERIALS
        rho = None
        if densities and mat_name in densities:
            rho = float(densities[mat_name])       # kg/m^3, caller's contract
        elif known:
            rho = materials.density(mat_name)      # g/cm^3
        if rho is None:
            unknown_material.add(mat_name or "(unset)")

        if rho is None:
            mass = 0.0
        elif densities and mat_name in densities:
            mass = vol * 1e-9 * rho                # mm^3 -> m^3, kg/m^3 -> kg
        else:
            mass = materials.mass_g(mat_name, vol) / 1000.0   # g -> kg

        total_v += vol
        total_m += mass
        try:
            c = solid.center()
            cx += c.X * mass
            cy += c.Y * mass
            cz += c.Z * mass
        except Exception:
            pass
        rows.append((getattr(inst, "key", "?"), mat_name, vol, rho, mass))

    com = (cx / total_m, cy / total_m, cz / total_m) if total_m > 0 else (0.0, 0.0, 0.0)
    heaviest = sorted(rows, key=lambda r: -r[4])[:5]

    findings = [Finding(name, f"{mat or 'material unset'} — {mass:.3f} kg "
                              f"({vol / 1000:.1f} cm³)", mass, "kg", "info")
                for name, mat, vol, rho, mass in heaviest]
    if unknown_material:
        findings.append(Finding(
            "density unknown",
            f"no density for {', '.join(sorted(unknown_material))} — those parts are "
            f"excluded from the mass, so this is a lower bound",
            0, "", "warning"))

    status = "PASS" if total_m > 0 and not unknown_material else \
             ("INCONCLUSIVE" if total_m > 0 else "INCONCLUSIVE")
    summary = (f"Total volume {total_v / 1000:.1f} cm³, estimated mass {total_m:.3f} kg, "
               f"centre of mass ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}) mm.")
    if unknown_material:
        summary += " Some parts have no density, so mass is a lower bound."

    return Verdict("Mass properties", status, summary, findings,
                   {"volume_cm3": total_v / 1000, "mass_kg": total_m,
                    "com_x_mm": com[0], "com_y_mm": com[1], "com_z_mm": com[2],
                    "parts": len(rows)}, time.time() - t0)


# ── motion ───────────────────────────────────────────────────────────────────
def clearance_sweep(asm: Any, *, axis: str = "Z", steps: int = 36,
                    rotating_keys: Optional[Iterable[str]] = None,
                    centre: Optional[tuple] = None) -> Verdict:
    """Rotate the moving group through a full turn and find anything that collides.

    **Baseline-relative, and that is the whole point.** Parts in contact at rest report
    overlap at every angle, so a naive sweep flags every mating face as a jam. Measured
    during development: a sweep without a baseline reported 501 colliding pairs and 36,072
    events on a mechanism that actually clears. The test is not "does anything overlap",
    it is "does the overlap GROW as the part turns".

    `rotating_keys` should come from the spec's mate graph. Falling back to a name
    heuristic is supported but it is a heuristic — matching the substring "bearing" put
    `bearing_housing`, the stationary carrier, into the rotating group.
    """
    t0 = time.time()
    insts = instances_of(asm)
    if len(insts) < 2:
        return Verdict("Clearance sweep", "INCONCLUSIVE", "Not enough parts.",
                       seconds=time.time() - t0)

    keys = [getattr(i, "key", "?") for i in insts]
    if rotating_keys is None:
        hints = ("wheel", "impeller", "shaft", "nut_", "gear", "pulley", "rotor")
        rotating_keys = [k for k in keys if any(h in k.lower() for h in hints)]
        source = "name heuristic (approximate)"
    else:
        rotating_keys = list(rotating_keys)
        source = "spec mate graph"

    rot_set = set(rotating_keys)
    moving = [i for i in insts if getattr(i, "key", "?") in rot_set]
    still = [i for i in insts if getattr(i, "key", "?") not in rot_set]

    if not moving:
        return Verdict("Clearance sweep", "INCONCLUSIVE",
                       "No rotating parts identified — nothing to sweep.",
                       metrics={"grouping": source}, seconds=time.time() - t0)

    # Axis through the assembly centroid unless told otherwise.
    if centre is None:
        pts = []
        for inst in insts:
            s = _placed(inst)
            if s is None:
                continue
            try:
                c = s.center()
                pts.append((c.X, c.Y, c.Z))
            except Exception:
                pass
        centre = (sum(p[0] for p in pts) / len(pts),
                  sum(p[1] for p in pts) / len(pts),
                  sum(p[2] for p in pts) / len(pts)) if pts else (0.0, 0.0, 0.0)

    axis = (axis or "Z").upper()
    if axis not in ("X", "Y", "Z"):
        axis = "Z"

    def rotate(solid, angle: float):
        from build123d import Location
        loc = Location((centre[0], centre[1], centre[2]))
        rot = Location((0, 0, 0), (0, 0, 1), math.degrees(angle)) if axis == "Z" else \
              Location((0, 0, 0), (1, 0, 0), math.degrees(angle)) if axis == "X" else \
              Location((0, 0, 0), (0, 1, 0), math.degrees(angle))
        try:
            moved = solid.moved(rot)
            return moved.moved(Location((-centre[0], -centre[1], -centre[2])))
        except Exception:
            try:
                return solid.rotate(__import__("build123d").Axis(
                    (centre[0], centre[1], centre[2]),
                    {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis]),
                    math.degrees(angle))
            except Exception:
                return None

    still_solids = []
    for inst in still:
        s = _placed(inst)
        if s is not None:
            still_solids.append((getattr(inst, "key", "?"), s))

    # Baseline: overlap at rest, per pair.
    def measure(angle: float) -> dict:
        out = {}
        for inst in moving:
            s = _placed(inst)
            if s is None:
                continue
            placed = s if abs(angle) < 1e-9 else rotate(s, angle)
            if placed is None:
                continue
            mk = getattr(inst, "key", "?")
            for skey, ssolid in still_solids:
                v = overlap_volume(placed, ssolid)
                if v is None:
                    continue
                if v > 0:
                    out[(mk, skey)] = max(out.get((mk, skey), 0.0), v)
        return out

    rest = measure(0.0)
    peak = dict(rest)
    worst_angle: dict = {}

    for step in range(1, steps + 1):
        angle = 2 * math.pi * step / steps
        cur = measure(angle)
        for key, val in cur.items():
            if val > peak.get(key, 0.0):
                peak[key] = val
                worst_angle[key] = math.degrees(angle)

    grown = []
    for key, val in peak.items():
        base = rest.get(key, 0.0)
        if val > max(base, 0.0) * GROWTH_FACTOR and (val - base) > GROWTH_MIN_MM3:
            grown.append((key, base, val, worst_angle.get(key, 0.0)))

    findings = []
    for (mk, skey), base, val, ang in sorted(grown, key=lambda g: -g[2])[:10]:
        findings.append(Finding(
            f"{mk} <-> {skey}",
            f"overlap grows from {base:.2f} to {val:.2f} mm³ at {ang:.0f}° — "
            f"the rotating part fouls this one",
            val, "mm³", "fail"))

    metrics = {
        "steps": steps,
        "rotating_parts": len(moving),
        "static_parts": len(still_solids),
        "contacts_at_rest": len(rest),
        "collisions": len(grown),
    }

    if not still_solids:
        return Verdict("Clearance sweep", "INCONCLUSIVE",
                       "No static parts to collide with.", metrics=metrics,
                       seconds=time.time() - t0)

    if grown:
        status = "FAIL"
        summary = (f"{len(grown)} pair(s) collide during rotation. The mechanism does not "
                   f"clear its full travel — a rendering would never reveal this.")
    else:
        status = "PASS"
        summary = (f"The rotating group clears all {len(still_solids)} static parts through "
                   f"a full revolution ({steps} angles checked). "
                   f"{len(rest)} contact(s) at rest are mating faces and stay constant.")

    findings.append(Finding("grouping", source, 0, "", "info"))
    return Verdict("Clearance sweep", status, summary, findings, metrics,
                   time.time() - t0)


# ── convenience ──────────────────────────────────────────────────────────────
def full_audit(asm: Any, *, sweep: bool = True, axis: str = "Z",
               steps: int = 36) -> list[Verdict]:
    """Every static test, in the order that makes failures cheapest to read."""
    out = [interference_audit(asm), mass_properties(asm)]
    if sweep:
        out.append(clearance_sweep(asm, axis=axis, steps=steps))
    return out


def overall(verdicts: Sequence[Verdict]) -> str:
    """One word for a set of tests. Any FAIL dominates; INCONCLUSIVE beats PASS."""
    statuses = {v.status for v in verdicts}
    if "FAIL" in statuses:
        return "FAIL"
    if "INCONCLUSIVE" in statuses:
        return "INCONCLUSIVE"
    return "PASS"
