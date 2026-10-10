"""Layer 1's admission gate — the referee that lets the vocabulary grow itself.

A closed vocabulary is what makes coordinate hallucination impossible, and it is also
what makes the engine finite: a machine class nobody hand-wrote a skill for cannot be
built at all. The way out is not to open the vocabulary, it is to put a *kernel* between
the model and the vocabulary. The brain may propose a new skill; only geometry admits it.

This module is that geometry. It never asks a model anything. It takes a registered
skill and interrogates the solid it produces, because a single part is the one thing in
this architecture that is mechanically gradeable:

1. **solid**       — it builds, OCCT calls it valid, and it encloses volume.
2. **range**       — it survives the parameter box it declares, not just its defaults.
3. **determinism** — same parameters, same volume and bounding box, so the sha1 cache
                     in `worker` is not lying to itself.
4. **seats**       — every published Connector is *true*: a mating face sits on a real
                     face pointing out of the material, a bore matches the diameter the
                     skill advertises in `interfaces`, a polar ring is actually regular.
5. **axis**        — the +Z invariant the whole assembler rests on.

Check 4 is the one that matters. Everything downstream trusts connectors absolutely:
`validate` allows a mate to name any connector the skill declares, and the assembler
derives a pose from it with no further questions. A seat that lies — right name, wrong
place, or a face normal pointing into the material instead of out of it — produces an
assembly that passes every existing check and is wrong. So the gate measures the seat
against the solid rather than against the skill's own claims about itself.

Findings are graded. A fatal one blocks admission; the rest are advice a repair pass can
act on (usually "your declared min/max is wider than your geometry actually supports").
The text is written to be handed straight back to the model, same as Layer 4's patches.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from . import registry
from .frames import KINDS, Connector, SkillResult, bind

LIN_TOL = 0.05      # mm — how far a seat may sit from the surface it claims
ANG_TOL = 1e-3      # unit-vector slack for "parallel to Z"
DET_TOL = 1e-9      # relative tolerance for the determinism replay
MIN_VOLUME = 1e-6   # mm^3 — below this it is not a part
SWEEP_PASS = 0.75   # fraction of declared-range corners that must build
SWEEP_BUDGET = 25.0  # s — wall clock the range sweep may spend on one skill
HEAVY_BUILD = 0.5   # s — a default build slower than this gets the reduced sweep

# Codes that block admission. Everything else is advice. Kept as one collection so the
# gate's severity is a single edit rather than a fatal= flag scattered over 200 lines.
FATAL = frozenset({
    "BUILD_FAILED", "NO_SOLID", "NOT_A_SKILL_RESULT", "INVALID_BREP", "ZERO_VOLUME",
    "DEFAULTS_FAILED", "RANGE_COLLAPSE", "NONDETERMINISTIC",
    "CONNECTOR_MISMATCH", "BAD_KIND", "BIND_FAILED", "SEAT_IN_AIR",
    "SEAT_FACING_INWARD", "PATTERN_IRREGULAR", "DEGENERATE", "OFF_AXIS",
})


@dataclass
class Finding:
    code: str
    detail: str

    @property
    def fatal(self) -> bool:
        return self.code in FATAL

    def __str__(self) -> str:
        return f"{'FAIL' if self.fatal else 'warn'} {self.code}: {self.detail}"


@dataclass
class Check:
    name: str
    findings: list[Finding] = field(default_factory=list)
    seconds: float = 0.0
    detail: str = ""

    @property
    def passed(self) -> bool:
        return not any(f.fatal for f in self.findings)

    def add(self, code: str, detail: str) -> None:
        self.findings.append(Finding(code, detail))


@dataclass
class GateReport:
    skill: str
    checks: list[Check] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def findings(self) -> list[Finding]:
        return [f for c in self.checks for f in c.findings]

    @property
    def passed(self) -> bool:
        return not any(f.fatal for f in self.findings)

    def text(self) -> str:
        head = f"{self.skill}: {'ADMIT' if self.passed else 'REJECT'}  ({self.seconds:.2f}s)"
        lines = [head]
        for c in self.checks:
            mark = "ok  " if c.passed else "FAIL"
            lines.append(f"  {mark} {c.name:13s} {c.detail}")
            lines += [f"         {f}" for f in c.findings]
        return "\n".join(lines)

    def feedback(self) -> str:
        """The rejection, phrased for whoever has to fix the skill — model or human."""
        if self.passed and not self.findings:
            return f"{self.skill}: admitted, no findings."
        bad = [f for f in self.findings if f.fatal]
        soft = [f for f in self.findings if not f.fatal]
        out = [f"skill '{self.skill}' was "
               + ("REJECTED by the geometry gate." if bad else "admitted with warnings.")]
        out += [f"  - {f.code}: {f.detail}" for f in bad + soft]
        return "\n".join(out)


# ── building one candidate ──────────────────────────────────────────────
def _build(sk: registry.Skill, params: dict[str, Any]) -> tuple[SkillResult | None, str]:
    """Run a skill through the same coercion path a spec would take."""
    kwargs, errs = registry.coerce_params(sk, params)
    if errs:
        return None, "; ".join(errs)
    try:
        res = sk.fn(**kwargs)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(res, SkillResult):
        return None, f"returned {type(res).__name__}, not a SkillResult"
    if res.solid is None:
        return None, "returned a SkillResult with no solid"
    return res, ""


def _defaults(sk: registry.Skill) -> dict[str, Any]:
    return {n: d.default for n, d in sk.params.items() if d.default is not None}


def _measure(solid) -> tuple[float, tuple[float, ...]]:
    bb = solid.bounding_box()
    return float(solid.volume), (float(bb.min.X), float(bb.min.Y), float(bb.min.Z),
                                 float(bb.max.X), float(bb.max.Y), float(bb.max.Z))


def _axial(vec) -> int:
    """+1 / -1 if this direction is parallel to global Z, else 0."""
    x, y, z = float(vec.X), float(vec.Y), float(vec.Z)
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return 0
    if abs(abs(z) / n - 1.0) > ANG_TOL:
        return 0
    return 1 if z > 0 else -1


def _local_z(loc) -> Any:
    """The connector's own +Z, in the part's frame."""
    from build123d import Vector

    t = loc.wrapped.Transformation()
    return Vector(t.Value(1, 3), t.Value(2, 3), t.Value(3, 3))


@dataclass
class Surfaces:
    """The part's faces, reduced to the two families a connector can claim.

    Classified once per candidate: reading `geom_type` and `radius` off a few hundred
    faces is cheap, doing it inside a per-connector loop is not.
    """
    planes: list[tuple[float, int, float]] = field(default_factory=list)   # z, sign, area
    cylinders: list[tuple[float, float]] = field(default_factory=list)     # radius, area

    def cylinder_at(self, radius: float, tol: float) -> bool:
        return any(abs(r - radius) <= tol for r, _ in self.cylinders)


def _surfaces(solid) -> Surfaces:
    from build123d import GeomType

    out = Surfaces()
    for f in solid.faces():
        try:
            gt = f.geom_type
            if gt is GeomType.PLANE:
                sign = _axial(f.normal_at())
                if sign:
                    out.planes.append((float(f.center().Z), sign, float(f.area)))
            elif gt is GeomType.CYLINDER:
                ax = f.axis_of_rotation
                if ax is not None and _axial(ax.direction):
                    out.cylinders.append((float(f.radius), float(f.area)))
        except Exception:
            continue
    return out


# ── check 1: it is a solid at all ───────────────────────────────────────
def _check_solid(res: SkillResult) -> Check:
    chk = Check("solid")
    t0 = time.time()
    solid = res.solid
    try:
        vol = float(solid.volume)
    except Exception as exc:
        chk.add("ZERO_VOLUME", f"volume unreadable ({type(exc).__name__})")
        chk.seconds = time.time() - t0
        return chk
    if vol <= MIN_VOLUME:
        chk.add("ZERO_VOLUME", f"encloses {vol:.3g} mm3 — nothing was unioned")
    try:
        if not solid.is_valid:
            chk.add("INVALID_BREP", "OCCT's own checker rejects the shape")
    except Exception as exc:
        chk.add("INVALID_BREP", f"validity check raised {type(exc).__name__}")
    lumps = 1
    try:
        lumps = len(solid.solids())
    except Exception:
        pass
    if lumps > 1:
        chk.add("MULTI_SOLID", f"{lumps} disjoint solids — legal, but a part that comes "
                               "apart is usually an un-unioned feature")
    chk.detail = f"{vol / 1000.0:.2f} cm3, {lumps} solid(s)"
    chk.seconds = time.time() - t0
    return chk


# ── check 5: the invariant the assembler rests on ───────────────────────
def _check_axis(res: SkillResult) -> Check:
    """+Z is the functional axis — so the Z axis has to run through the part.

    A part modelled off to one side looks perfect on its own and is wrong the moment it
    is mated, because every seat is expressed relative to an axis the material does not
    share. That failure is invisible to a validator reading the spec and obvious here.
    """
    chk = Check("axis")
    t0 = time.time()
    try:
        bb = res.solid.bounding_box()
    except Exception as exc:
        chk.add("DEGENERATE", f"no bounding box ({type(exc).__name__})")
        chk.seconds = time.time() - t0
        return chk
    dx, dy, dz = float(bb.size.X), float(bb.size.Y), float(bb.size.Z)
    if min(dx, dy, dz) <= LIN_TOL:
        chk.add("DEGENERATE", f"extent {dx:.3g} x {dy:.3g} x {dz:.3g} mm — flat or empty "
                              "in one direction")
    off = []
    for name, lo, hi in (("X", float(bb.min.X), float(bb.max.X)),
                         ("Y", float(bb.min.Y), float(bb.max.Y))):
        if lo > LIN_TOL or hi < -LIN_TOL:
            off.append(f"{name} spans [{lo:.2f}, {hi:.2f}]")
    if off:
        chk.add("OFF_AXIS", "the Z axis misses the material: " + ", ".join(off)
                + " — model the part about its own axis and let mates move it")
    chk.detail = f"{dx:.1f} x {dy:.1f} x {dz:.1f} mm about Z"
    chk.seconds = time.time() - t0
    return chk


# ── check 3: the cache is only honest if the skill is ────────────────────
def _check_determinism(sk: registry.Skill, base: tuple[float, tuple]) -> Check:
    chk = Check("determinism")
    t0 = time.time()
    res, err = _build(sk, _defaults(sk))
    if res is None:
        chk.add("NONDETERMINISTIC", f"replay of the default build failed: {err}")
        chk.seconds = time.time() - t0
        return chk
    vol, box = _measure(res.solid)
    ref_vol, ref_box = base
    drift = abs(vol - ref_vol) / max(abs(ref_vol), 1e-9)
    moved = max(abs(a - b) for a, b in zip(box, ref_box))
    if drift > DET_TOL:
        chk.add("NONDETERMINISTIC", f"volume moved {drift:.3g} between identical builds")
    if moved > DET_TOL:
        chk.add("NONDETERMINISTIC", f"bounding box moved {moved:.3g} mm between "
                                    "identical builds")
    chk.detail = f"volume drift {drift:.1e}, box drift {moved:.1e} mm"
    chk.seconds = time.time() - t0
    return chk


# ── check 2: the declared parameter box, not just the defaults ───────────
def _corners(sk: registry.Skill, heavy: bool = False) -> list[tuple[str, dict[str, Any]]]:
    """One variation per axis of the parameter box, plus the two extreme corners.

    Deliberately not the cross product: a skill with eight parameters would be 6561
    builds and the interesting failures are all one-dimensional — a wall thickness that
    goes negative at its own declared minimum, a count of zero, a choice nobody tested.

    Ordered cheap-first, because the cost of a corner is wildly uneven: a volute at its
    declared maximum is a minute of booleans while every `choices` branch is milliseconds.
    `heavy` drops the expensive half for a skill whose *default* build is already slow —
    the small end and the untested branches are where degenerate geometry actually hides,
    and the report says plainly which corners were skipped rather than implying coverage.
    """
    base = _defaults(sk)
    branches: list[tuple[str, dict[str, Any]]] = []
    mins: list[tuple[str, dict[str, Any]]] = []
    maxes: list[tuple[str, dict[str, Any]]] = []
    lo_all, hi_all = dict(base), dict(base)
    for name, d in sk.params.items():
        if d.choices:
            branches += [(f"{name}={c!r}", {**base, name: c})
                         for c in d.choices if c != base.get(name)]
            continue
        if d.type is bool:
            branches.append((f"{name}={not base.get(name)}",
                             {**base, name: not base.get(name)}))
            continue
        if d.type not in (int, float):
            continue
        for tag, val, bucket, corner in (("min", d.min, mins, lo_all),
                                         ("max", d.max, maxes, hi_all)):
            if val is None:
                continue
            v = int(round(val)) if d.type is int else float(val)
            bucket.append((f"{name}={tag}({v:g})", {**base, name: v}))
            corner[name] = v
    cases = branches + mins
    if heavy:
        return cases
    cases += maxes
    if lo_all != base:
        cases.append(("all-min", lo_all))
    if hi_all != base:
        cases.append(("all-max", hi_all))
    return cases


def _check_range(sk: registry.Skill, *, heavy: bool = False,
                 budget: float = SWEEP_BUDGET) -> Check:
    chk = Check("range")
    t0 = time.time()
    cases = _corners(sk, heavy=heavy)
    bad: list[str] = []
    done = 0
    for tag, params in cases:
        if time.time() - t0 > budget:
            chk.add("RANGE_BUDGET", f"stopped after {done}/{len(cases)} corners and "
                                    f"{budget:.0f}s — the rest are unexercised, not passing")
            break
        done += 1
        res, err = _build(sk, params)
        if res is None:
            bad.append(f"{tag} -> {err}")
            continue
        try:
            if float(res.solid.volume) <= MIN_VOLUME:
                bad.append(f"{tag} -> built an empty solid")
        except Exception as exc:
            bad.append(f"{tag} -> volume unreadable ({type(exc).__name__})")
    good = done - len(bad)
    rate = good / done if done else 1.0
    for msg in bad[:12]:
        chk.add("RANGE", msg)
    if len(bad) > 12:
        chk.add("RANGE", f"...and {len(bad) - 12} more")
    if done and rate < SWEEP_PASS:
        chk.add("RANGE_COLLAPSE", f"only {good}/{done} declared-range corners build — the "
                                  f"min/max on these parameters describe a part this skill "
                                  f"cannot actually make")
    skipped = len(cases) - done
    chk.detail = (f"{good}/{done} corners build"
                  + (f", {skipped} unexercised" if skipped else "")
                  + (" (heavy: extremes skipped)" if heavy else ""))
    chk.seconds = time.time() - t0
    return chk


# ── check 4: connector truth ────────────────────────────────────────────
def _seat_geometry(chk: Check, name: str, conn: Connector, surf: Surfaces, bb) -> None:
    """Measure one seat against the material it claims to sit on."""
    pad = LIN_TOL
    for i, loc in enumerate(conn.locations):
        pos = loc.position
        tag = name if conn.count == 1 else f"{name}[{i}]"
        if not (float(bb.min.X) - pad <= float(pos.X) <= float(bb.max.X) + pad
                and float(bb.min.Y) - pad <= float(pos.Y) <= float(bb.max.Y) + pad
                and float(bb.min.Z) - pad <= float(pos.Z) <= float(bb.max.Z) + pad):
            chk.add("SEAT_IN_AIR",
                    f"'{tag}' sits at ({pos.X:.2f}, {pos.Y:.2f}, {pos.Z:.2f}) but the part "
                    f"only occupies x[{bb.min.X:.1f},{bb.max.X:.1f}] "
                    f"y[{bb.min.Y:.1f},{bb.max.Y:.1f}] z[{bb.min.Z:.1f},{bb.max.Z:.1f}] "
                    f"— a mate to this seat would leave a gap of that size")

    if not conn.flip:
        return  # not a surface: a bore, a shaft station, a reference height
    loc = conn.location(0)
    sign = _axial(_local_z(loc))
    if not sign:
        return  # a radial or tilted seat; there is no axial face to check it against
    z = float(loc.position.Z)
    # OCCT orients a valid solid's face normals outward, so the kernel already knows
    # which way this height faces. Comparing against that beats any centre-of-mass
    # heuristic, which an L-bracket or a bolt shank passing through the seat defeats.
    here = {s for pz, s, _ in surf.planes if abs(pz - z) <= pad}
    if sign in here:
        return
    if -sign in here:
        chk.add("SEAT_FACING_INWARD",
                f"'{name}' declares an outward face at z={z:.2f} pointing "
                f"{'+Z' if sign > 0 else '-Z'}, but the solid's own face at that height "
                f"points {'-Z' if sign > 0 else '+Z'} — the seat normal aims into the "
                f"material, so anything mated here interpenetrates instead of stacking")
        return
    extremum = float(bb.max.Z) if sign > 0 else float(bb.min.Z)
    if abs(z - extremum) <= pad:
        chk.add("SEAT_TANGENT", f"'{name}' is at the part's {'top' if sign > 0 else 'bottom'} "
                                f"but there is no planar face there — a tangent seat "
                                f"(fine for a torus, wrong for a shoulder)")
    else:
        chk.add("SEAT_NOT_ON_FACE",
                f"'{name}' claims a planar face at z={z:.2f} facing "
                f"{'+Z' if sign > 0 else '-Z'}; the solid has none within {pad} mm "
                f"(axial faces at {sorted({round(p, 2) for p, s, _ in surf.planes if s == sign})})")


def _pattern(chk: Check, name: str, conn: Connector) -> None:
    """A patterned seat is a ring or a stack; either way it has to be regular.

    The assembler hands instance *i* of a patterned part to station *i* of the seat, so
    an irregular ring silently scatters a bolt circle instead of failing.
    """
    if conn.count < 2:
        return
    pts = [c.position for c in conn.locations]
    radii = [math.hypot(float(p.X), float(p.Y)) for p in pts]
    zs = [float(p.Z) for p in pts]
    if max(radii) <= LIN_TOL:                      # a stack along the axis
        if len(set(round(v, 4) for v in zs)) != len(zs):
            chk.add("PATTERN_IRREGULAR", f"'{name}' stacks {conn.count} seats but some "
                                         f"share the same z: {[round(v, 2) for v in zs]}")
        return
    if max(radii) - min(radii) > LIN_TOL:
        chk.add("PATTERN_IRREGULAR", f"'{name}' is a ring of {conn.count} seats at "
                                     f"radii {min(radii):.2f}..{max(radii):.2f} mm — a bolt "
                                     f"circle has one pitch radius")
    if max(zs) - min(zs) > LIN_TOL:
        chk.add("PATTERN_IRREGULAR", f"'{name}' is a ring of {conn.count} seats spread over "
                                     f"z {min(zs):.2f}..{max(zs):.2f} mm")
    ang = sorted(math.degrees(math.atan2(float(p.Y), float(p.X))) % 360.0 for p in pts)
    step = 360.0 / conn.count
    gaps = [(b - a) for a, b in zip(ang, ang[1:])] + [360.0 - ang[-1] + ang[0]]
    if max(abs(g - step) for g in gaps) > 0.05:
        chk.add("PATTERN_IRREGULAR", f"'{name}' spaces {conn.count} seats at "
                                     f"{min(gaps):.2f}..{max(gaps):.2f} deg instead of "
                                     f"{step:.2f} deg")


def _interfaces(chk: Check, res: SkillResult, surf: Surfaces, bb) -> None:
    """`interfaces` is what tolerance and clash checking believe; make the solid agree."""
    bore = float(res.interfaces.get("bore_d", 0.0) or 0.0)
    od = float(res.interfaces.get("od", 0.0) or 0.0)
    if bore > 0 and not surf.cylinder_at(bore / 2.0, max(LIN_TOL, bore * 0.01)):
        chk.add("INTERFACE_UNSUPPORTED",
                f"interfaces declare bore_d={bore:.2f} but no Z-coaxial cylindrical face "
                f"has radius {bore / 2:.2f} mm "
                f"(found {sorted({round(r * 2, 2) for r, _ in surf.cylinders})[:8]})")
    if od > 0:
        span = max(float(bb.max.X) - float(bb.min.X), float(bb.max.Y) - float(bb.min.Y))
        if not surf.cylinder_at(od / 2.0, max(LIN_TOL, od * 0.01)) and abs(span - od) > od * 0.02:
            chk.add("INTERFACE_UNSUPPORTED",
                    f"interfaces declare od={od:.2f} but the part measures {span:.2f} mm "
                    f"across and has no cylindrical face at that diameter")


def _conditional(sk: registry.Skill, missing: set[str]) -> dict[str, str]:
    """Which parameter switches each un-published seat on, if any does.

    Several skills publish a bolt circle only when `bolt_count > 0`, a keyway only when
    `keyway` is true. That is not a lie, but it is a trap: `validate` allows a mate to
    any *declared* connector, so a spec can name `flange.bolt_circle` while leaving the
    count at zero and only find out at assembly time. Naming the switch turns a fatal
    rejection into an accurate warning.
    """
    found: dict[str, str] = {}
    for tag, params in _corners(sk):
        if not missing - set(found):
            break
        res, _ = _build(sk, params)
        if res is None:
            continue
        for name in (missing & set(res.connectors)) - set(found):
            found[name] = tag
    return found


def _check_seats(sk: registry.Skill, res: SkillResult) -> Check:
    chk = Check("seats")
    t0 = time.time()
    declared, published = set(sk.connectors), set(res.connectors)
    extra = sorted(published - declared)
    if extra:
        chk.add("CONNECTOR_MISMATCH", f"published but not declared: {extra} — no spec can "
                                      f"reach them, so add them to the @skill decorator")
    missing = declared - published
    if missing:
        switch = _conditional(sk, set(missing))
        for name in sorted(missing):
            if name in switch:
                chk.add("CONNECTOR_CONDITIONAL",
                        f"'{name}' only exists when {switch[name]}; a spec that mates to "
                        f"it at the default parameters will not find it")
            else:
                chk.add("CONNECTOR_MISMATCH",
                        f"'{name}' is declared but no parameter in the whole declared range "
                        f"publishes it — a spec may legally mate to a seat that cannot exist")
    if not published:
        chk.add("CONNECTOR_MISMATCH", "publishes no connectors, so nothing can be mated "
                                      "to it and it can only ever be the datum")
        chk.seconds = time.time() - t0
        return chk

    try:
        bb = res.solid.bounding_box()
        surf = _surfaces(res.solid)
    except Exception as exc:
        chk.add("SEAT_IN_AIR", f"cannot measure the solid to check seats "
                               f"({type(exc).__name__}: {exc})")
        chk.seconds = time.time() - t0
        return chk

    for name, conn in res.connectors.items():
        if conn.kind not in KINDS:
            chk.add("BAD_KIND", f"'{name}' has kind '{conn.kind}'; one of {list(KINDS)}")
        else:
            try:
                bind(res.solid, conn, 0, label=f"_gate_{name}")
            except Exception as exc:
                chk.add("BIND_FAILED", f"'{name}' will not become a joint "
                                       f"({type(exc).__name__}: {exc})")
        _seat_geometry(chk, name, conn, surf, bb)
        _pattern(chk, name, conn)
    _interfaces(chk, res, surf, bb)

    rings = sum(1 for c in res.connectors.values() if c.count > 1)
    chk.detail = (f"{len(published)} seats ({rings} patterned), "
                  f"{len(surf.planes)} axial faces, {len(surf.cylinders)} coaxial bores")
    chk.seconds = time.time() - t0
    return chk


# ── the gate itself ─────────────────────────────────────────────────────
def run(name: str, *, sweep: bool = True) -> GateReport:
    """Interrogate one registered skill. No model, no network, no spec."""
    t0 = time.time()
    rep = GateReport(name)
    sk = registry.get(name)
    if sk is None:
        chk = Check("solid")
        chk.add("NOT_A_SKILL_RESULT", f"'{name}' is not registered")
        rep.checks.append(chk)
        rep.seconds = time.time() - t0
        return rep

    res, err = _build(sk, _defaults(sk))
    if res is None:
        chk = Check("solid")
        chk.add("DEFAULTS_FAILED", f"will not build at its own defaults: {err}")
        rep.checks.append(chk)
        rep.seconds = time.time() - t0
        return rep
    first = time.time() - t0

    rep.checks.append(_check_solid(res))
    if sweep:
        rep.checks.append(_check_range(sk, heavy=first > HEAVY_BUILD))
    try:
        base = _measure(res.solid)
    except Exception:
        base = (0.0, (0.0,) * 6)
    rep.checks.append(_check_determinism(sk, base))
    rep.checks.append(_check_seats(sk, res))
    rep.checks.append(_check_axis(res))
    rep.seconds = time.time() - t0
    return rep


def run_all(names: list[str] | None = None, *, sweep: bool = True) -> list[GateReport]:
    return [run(n, sweep=sweep) for n in (names or registry.names())]


def table(reports: list[GateReport]) -> str:
    """One line per skill, then the detail for anything with something to say."""
    order = ["solid", "range", "determinism", "seats", "axis"]
    head = f"{'skill':22s} {'':6s} " + " ".join(f"{c[:5]:>5s}" for c in order) + "   findings"
    lines = [head, "-" * len(head)]
    for rep in reports:
        by = {c.name: c for c in rep.checks}
        cells = [f"{'-':>5s}" if by.get(c) is None
                 else f"{'ok' if by[c].passed else 'FAIL':>5s}" for c in order]
        bad = sum(1 for f in rep.findings if f.fatal)
        warn = len(rep.findings) - bad
        note = f"{bad} fatal, {warn} warn" if rep.findings else "clean"
        lines.append(f"{rep.skill:22s} {'ADMIT' if rep.passed else 'REJECT':6s} "
                     + " ".join(cells) + f"   {note}")
    lines.append("-" * len(head))
    lines.append(f"{sum(1 for r in reports if r.passed)}/{len(reports)} admitted   "
                 f"{sum(r.seconds for r in reports):.1f}s")
    detail = [r.text() for r in reports if r.findings]
    return "\n".join(lines + ([""] + detail if detail else []))
