"""Engineering conditions — the facts geometry cannot tell you.

A boolean intersection test can prove two solids do not overlap. It cannot tell you
that three planet gears will not assemble because the sun and ring tooth counts do
not divide by three, or that two gears which mesh perfectly in the render have
different modules and would destroy each other in a week. Those are conditions on
the *numbers*, and until now nothing in the engine looked at them: the shipped
planetary gearbox satisfies its own assembly condition entirely by coincidence,
because someone typed tooth counts that happened to work.

Each rule reads the parts' resolved parameters and, where it needs to, measures the
built assembly. All of them are cheap; none of them needs a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import registry, xform
from .schema import CheckSpec, MateSpec, Spec


@dataclass
class Finding:
    ok: bool
    kind: str
    code: str
    message: str

    def __str__(self) -> str:
        mark = "ok" if self.ok else "FAIL"
        return f"  {mark} {self.kind}: {self.message}"


def evaluate(asm, checks: list[CheckSpec] | None = None) -> list[Finding]:
    """Run every declared condition. Unknown kinds are reported, never skipped."""
    spec: Spec = asm.spec
    out: list[Finding] = []
    for chk in (spec.checks if checks is None else checks):
        fn = _RULES.get(chk.kind)
        if fn is None:
            out.append(Finding(False, chk.kind, "UNKNOWN_CHECK",
                               f"no rule implements '{chk.kind}'"))
            continue
        try:
            out += fn(asm, chk)
        except Exception as exc:  # a broken check must not hide the others
            out.append(Finding(False, chk.kind, "CHECK_FAILED",
                               f"{type(exc).__name__}: {exc}"))
    return out


# ── helpers ─────────────────────────────────────────────────────────────────

def _params(asm, ref: str) -> dict:
    pid = MateSpec.split(str(ref))[0]
    part = asm.spec.part(pid)
    return asm.spec.params_of(part) if part is not None else {}


def _defaults(asm, ref: str) -> dict:
    """Resolved params plus the skill's own defaults, so an omitted value is known."""
    pid = MateSpec.split(str(ref))[0]
    part = asm.spec.part(pid)
    if part is None:
        return {}
    sk = registry.get(part.skill)
    base = {n: d.default for n, d in sk.params.items()} if sk else {}
    base.update({k: v for k, v in asm.spec.params_of(part).items() if v is not None})
    return base


def _count(asm, ref: str) -> int:
    pid = MateSpec.split(str(ref))[0]
    part = asm.spec.part(pid)
    return max(1, part.count) if part is not None else 0


def _axis_line(asm, ref: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(point on axis, unit direction) for a part, datum or seat, in world coordinates.

    A patterned seat needs care: `frame_of` hands back station 0, so taking its
    origin as the axis puts a bolt circle's "axis" through one of its holes, and two
    flanges whose station 0 happens to face opposite ways then read as 156 mm out of
    line when they are in fact perfectly concentric. The centre of the stations is
    the only defensible answer.
    """
    pid, conn = MateSpec.split(str(ref))
    insts = asm.of_part(pid)
    if conn and insts:
        c = insts[0].result.connectors.get(conn)
        if c is not None and c.count > 1:
            pts = []
            for i in range(c.count):
                p = (insts[0].solid.location * c.location(i)).position
                pts.append([float(p.X), float(p.Y), float(p.Z)])
            centre = np.array([sum(q[k] for q in pts) / len(pts) for k in range(3)])
            R, _ = xform.loc_to_Rt(insts[0].solid.location)
            axis = xform.canonical_axis(R[:, 2])
            if axis is not None:
                return (centre, axis)
    loc = asm.frame_of(ref)
    if loc is None:
        return None
    R, t = xform.loc_to_Rt(loc)
    return (t, R[:, 2])


def _axis_gap(a: tuple, b: tuple) -> float:
    """Shortest distance between two axis lines — 0 when they are collinear."""
    (p, u), (q, v) = a, b
    w = q - p
    n = np.cross(u, v)
    if float(np.linalg.norm(n)) < 1e-9:          # parallel: distance point-to-line
        return float(np.linalg.norm(w - u * float(w @ u)))
    return abs(float(w @ n)) / float(np.linalg.norm(n))


def _ring_radius(asm, ref: str) -> tuple[float, int] | None:
    """(radius, station count) of a patterned connector named `part.connector`."""
    pid, conn = MateSpec.split(str(ref))
    insts = asm.of_part(pid)
    if not insts or not conn:
        return None
    c = insts[0].result.connectors.get(conn)
    if c is None:
        return None
    radii = [math.hypot(float(loc.position.X), float(loc.position.Y))
             for loc in c.locations]
    return (sum(radii) / len(radii), c.count)


def _interface(asm, ref: str, *keys: str) -> float:
    pid = MateSpec.split(str(ref))[0]
    insts = asm.of_part(pid)
    if not insts:
        return 0.0
    face = insts[0].result.interfaces or {}
    for key in keys:
        if float(face.get(key, 0.0) or 0.0) > 0.0:
            return float(face[key])
    return 0.0


# ── the rules ───────────────────────────────────────────────────────────────

def _planetary(asm, chk: CheckSpec) -> list[Finding]:
    """The three conditions a planetary train has to satisfy to exist at all.

    Tooth-count compatibility is the one that bites: with a sun of 24 and a ring of
    60 you may have three planets or four, but never five, because the planets have
    to land on teeth. A model that draws five anyway looks entirely convincing.
    """
    sun, planet, ring = chk.get("sun"), chk.get("planet"), chk.get("ring")
    n = int(chk.get("n") or _count(asm, planet))
    ps, pp, pr = _defaults(asm, sun), _defaults(asm, planet), _defaults(asm, ring)
    ts, tp, tr = (int(ps.get("teeth", 0)), int(pp.get("teeth", 0)), int(pr.get("teeth", 0)))
    out: list[Finding] = []

    mods = {"sun": float(ps.get("module", 0.0)), "planet": float(pp.get("module", 0.0)),
            "ring": float(pr.get("module", 0.0))}
    spread = max(mods.values()) - min(mods.values()) if mods else 0.0
    out.append(Finding(spread <= 1e-6, "planetary", "MODULE_MISMATCH",
                       f"module {mods} — every gear in one train shares a module"
                       if spread > 1e-6 else
                       f"all three gears are module {mods['sun']:g}"))

    want_ring = ts + 2 * tp
    out.append(Finding(tr == want_ring, "planetary", "RING_TEETH",
                       f"ring has {tr} teeth; sun {ts} + 2 x planet {tp} = {want_ring}"))

    if n > 0:
        share = (ts + tr) / n
        ok = abs(share - round(share)) < 1e-9
        out.append(Finding(ok, "planetary", "ASSEMBLY_CONDITION",
                           f"(sun {ts} + ring {tr}) / {n} planets = {share:g} — "
                           f"{'a whole number, so the planets land on teeth' if ok else 'not a whole number, so the planets cannot all mesh'}"))

    carrier = chk.get("carrier")
    if carrier and ts and tp:
        pcd = float(mods["sun"]) * (ts + tp) / 2.0
        ring_info = _ring_radius(asm, carrier)
        if ring_info is not None:
            have, stations = ring_info
            out.append(Finding(abs(have - pcd) <= max(0.05, 0.002 * pcd), "planetary",
                               "CARRIER_PCD",
                               f"carrier pin circle radius {have:.3f} mm vs required "
                               f"{pcd:.3f} mm (module x (sun + planet) / 2)"))
            out.append(Finding(stations == n, "planetary", "CARRIER_STATIONS",
                               f"carrier publishes {stations} pin station(s) for {n} planet(s)"))
    return out


def _gear_mesh(asm, chk: CheckSpec) -> list[Finding]:
    """Two gears mesh only if they share a module and sit at the right centres."""
    a, b = chk.get("a"), chk.get("b")
    pa, pb = _defaults(asm, a), _defaults(asm, b)
    ma, mb = float(pa.get("module", 0.0)), float(pb.get("module", 0.0))
    ta, tb = int(pa.get("teeth", 0)), int(pb.get("teeth", 0))
    internal = bool(pa.get("internal")) or bool(pb.get("internal"))
    out = [Finding(abs(ma - mb) <= 1e-6, "gear_mesh", "MODULE_MISMATCH",
                   f"module {ma:g} vs {mb:g}")]
    if ta and tb and ma:
        want = ma * (abs(tb - ta) if internal else (ta + tb)) / 2.0
        la, lb = _axis_line(asm, a), _axis_line(asm, b)
        if la and lb:
            have = _axis_distance(la, lb)
            tol = float(chk.get("tol") or max(0.05, 0.004 * max(want, 1.0)))
            out.append(Finding(abs(have - want) <= tol, "gear_mesh", "CENTRE_DISTANCE",
                               f"axes {have:.3f} mm apart; a {'internal' if internal else 'external'} "
                               f"pair of {ta}/{tb} teeth at module {ma:g} needs {want:.3f} mm"))
    return out


def _axis_distance(a: tuple, b: tuple) -> float:
    """Perpendicular distance between two parallel-ish axes, in the plane across them."""
    (p, u), (q, _v) = a, b
    w = q - p
    along = float(w @ u)
    return float(np.linalg.norm(w - u * along))


def _bolt_circle(asm, chk: CheckSpec) -> list[Finding]:
    """Two flanges bolt together only if their circles agree in size and count."""
    a, b = chk.get("a"), chk.get("b")
    ra, rb = _ring_radius(asm, a), _ring_radius(asm, b)
    if ra is None or rb is None:
        return [Finding(False, "bolt_circle", "NO_RING",
                        f"'{a if ra is None else b}' publishes no patterned seat")]
    tol = float(chk.get("tol") or 0.2)
    out = [Finding(abs(ra[0] - rb[0]) <= tol, "bolt_circle", "PCD_MISMATCH",
                   f"{a} radius {ra[0]:.2f} mm vs {b} radius {rb[0]:.2f} mm"),
           Finding(ra[1] == rb[1], "bolt_circle", "COUNT_MISMATCH",
                   f"{a} has {ra[1]} holes, {b} has {rb[1]}")]
    la, lb = _axis_line(asm, a), _axis_line(asm, b)
    if la and lb:
        gap = _axis_gap(la, lb)
        out.append(Finding(gap <= max(tol, 0.2), "bolt_circle", "NOT_COAXIAL",
                           f"the two circles' axes are {gap:.3f} mm apart"))
    return out


def _fit(asm, chk: CheckSpec) -> list[Finding]:
    """A shaft in a bore: the numbers have to say what the fit class says."""
    shaft, bore = chk.get("shaft"), chk.get("bore")
    od = _interface(asm, shaft, "od", "d", "shaft_d")
    hole = _interface(asm, bore, "bore_d", "id")
    if od <= 0 or hole <= 0:
        return [Finding(False, "fit", "NOT_MEASURABLE",
                        f"'{shaft}' or '{bore}' publishes no diameter interface")]
    slack = hole - od
    want = str(chk.get("class") or "clearance").lower()
    ok = {"clearance": slack > 0.0, "slip": slack >= -0.005,
          "press": -0.15 <= slack <= 0.005, "interference": slack < 0.0,
          "free": slack > 0.0}.get(want, slack >= 0.0)
    return [Finding(ok, "fit", "FIT_CLASS",
                    f"{shaft} {od:.3f} mm in {bore} {hole:.3f} mm leaves {slack:+.3f} mm "
                    f"— asked for '{want}'")]


def _coaxial(asm, chk: CheckSpec) -> list[Finding]:
    a, b = chk.get("a"), chk.get("b")
    la, lb = _axis_line(asm, a), _axis_line(asm, b)
    if la is None or lb is None:
        return [Finding(False, "coaxial", "NOT_MEASURABLE", f"'{a}' or '{b}' has no pose")]
    tol = float(chk.get("tol") or 0.1)
    gap = _axis_gap(la, lb)
    cos = abs(float(la[1] @ lb[1]))
    return [Finding(gap <= tol, "coaxial", "OFFSET",
                    f"{a} and {b} axes are {gap:.4f} mm apart (tol {tol})"),
            Finding(cos >= 0.9999, "coaxial", "TILT",
                    f"{a} and {b} axes differ by {math.degrees(math.acos(min(1.0, cos))):.3f} deg")]


def _stack_length(asm, chk: CheckSpec) -> list[Finding]:
    """The overall length of a run of parts, measured rather than assumed."""
    parts = chk.get("parts") or []
    axis = str(chk.get("axis") or "z").lower()
    k = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
    lo, hi = 1e18, -1e18
    for ref in parts:
        for inst in asm.of_part(MateSpec.split(str(ref))[0]):
            try:
                bb = inst.solid.bounding_box()
            except Exception:
                continue
            lo = min(lo, float((bb.min.X, bb.min.Y, bb.min.Z)[k]))
            hi = max(hi, float((bb.max.X, bb.max.Y, bb.max.Z)[k]))
    if hi < lo:
        return [Finding(False, "stack_length", "NOT_MEASURABLE",
                        f"none of {parts} produced a bounding box")]
    have = hi - lo
    want = float(chk.get("value") or 0.0)
    tol = float(chk.get("tol") or 0.5)
    return [Finding(abs(have - want) <= tol, "stack_length", "LENGTH",
                    f"{len(parts)} part(s) span {have:.3f} mm along {axis}; "
                    f"expected {want:.3f} +-{tol}")]


def _count_rule(asm, chk: CheckSpec) -> list[Finding]:
    part = chk.get("part")
    want = int(chk.get("value") or 0)
    have = len(asm.of_part(MateSpec.split(str(part))[0]))
    return [Finding(have == want, "count", "INSTANCES",
                    f"'{part}' produced {have} instance(s), expected {want}")]


def _clearance(asm, chk: CheckSpec) -> list[Finding]:
    """A minimum gap, checked on bounding boxes — conservative, so never a false pass."""
    a, b = chk.get("a"), chk.get("b")
    want = float(chk.get("min") or chk.get("value") or 0.0)
    boxes = {}
    for ref in (a, b):
        pid = MateSpec.split(str(ref))[0]
        got = []
        for inst in asm.of_part(pid):
            try:
                bb = inst.solid.bounding_box()
            except Exception:
                continue
            got.append((float(bb.min.X), float(bb.min.Y), float(bb.min.Z),
                        float(bb.max.X), float(bb.max.Y), float(bb.max.Z)))
        boxes[ref] = got
    if not boxes.get(a) or not boxes.get(b):
        return [Finding(False, "clearance", "NOT_MEASURABLE", f"'{a}' or '{b}' has no geometry")]
    gap = min(_box_gap(x, y) for x in boxes[a] for y in boxes[b])
    return [Finding(gap >= want, "clearance", "GAP",
                    f"closest boxes of {a} and {b} are {gap:.3f} mm apart, "
                    f"need at least {want:.3f}")]


def _box_gap(p, q) -> float:
    dx = max(p[0] - q[3], q[0] - p[3], 0.0)
    dy = max(p[1] - q[4], q[1] - p[4], 0.0)
    dz = max(p[2] - q[5], q[2] - p[5], 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


_RULES = {
    "planetary": _planetary,
    "gear_mesh": _gear_mesh,
    "bolt_circle": _bolt_circle,
    "fit": _fit,
    "coaxial": _coaxial,
    "stack_length": _stack_length,
    "count": _count_rule,
    "clearance": _clearance,
}

