"""The 2D design plan — the stage that has to happen before any solid is built.

The engine's whole difficulty with placement came from doing the hard spatial
thinking in the most expensive possible medium. A model would author a 26-part
turbocharger, fifteen seconds of OCCT would run, and only then would anyone
discover that the volute was pointing the wrong way — at which point the only
feedback available was a picture of a lump.

A layout plan moves that thinking earlier and makes it reviewable. It is a
dimensioned 2D sketch: named points, the distances between them, the envelope each
part occupies, the bolt circles, and every number as a variable with equations
between them. It costs no geometry at all, it renders to an SVG a human can read in
five seconds, and it is *solved* — a real constraint solve, which means the plan can
tell you it is under-constrained before it can mislead you.

Once a human has approved it, the solved points become the datums the 3D assembly
hangs off (see `DatumSpec.point` and `assembler._layout_shift`). So the drawing that
was reviewed is literally what positions the model, and verification can compare the
finished B-rep against it instead of against nothing.

The plane the sketch lives in is named rather than assumed: `top` is XY, `front` is
XZ, `right` is YZ. A machine bed is drawn in plan and a turbocharger in elevation,
and forcing either into the other's plane is how a plan stops being worth reviewing.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import expr

VIEWS = ("top", "front", "right")

# Which world axes the sketch's (u, v) map onto, per view.
AXES: dict[str, tuple[int, int]] = {"top": (0, 1), "front": (0, 2), "right": (1, 2)}

CONSTRAINTS = ("fix", "dx", "dy", "distance", "horizontal", "vertical", "coincident",
               "angle", "on_x", "on_y", "collinear", "symmetric")

# Residual counts per constraint, needed for the degrees-of-freedom arithmetic.
WIDTH: dict[str, int] = {"fix": 2, "coincident": 2, "symmetric": 2}


@dataclass
class Point:
    id: str
    u: float = 0.0
    v: float = 0.0
    fix: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict, scope: dict[str, float]) -> "Point":
        u, _ = expr.number(d.get("x", d.get("u", 0.0)), scope)
        v, _ = expr.number(d.get("y", d.get("v", 0.0)), scope)
        return cls(id=str(d.get("id", "")), u=u, v=v,
                   fix=bool(d.get("fix", False)), note=str(d.get("note", "") or ""))


@dataclass
class Constraint:
    kind: str
    a: str = ""
    b: str = ""
    c: str = ""
    about: str = ""
    value: float = 0.0
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict, scope: dict[str, float]) -> "Constraint":
        value, _ = expr.number(d.get("value", 0.0), scope)
        return cls(kind=str(d.get("kind", "") or "").lower(),
                   a=str(d.get("a", "") or ""), b=str(d.get("b", "") or ""),
                   c=str(d.get("c", "") or ""), about=str(d.get("about", "") or ""),
                   value=value, note=str(d.get("note", "") or ""))

    @property
    def width(self) -> int:
        return WIDTH.get(self.kind, 1)

    @property
    def refs(self) -> list[str]:
        return [r for r in (self.a, self.b, self.c, self.about) if r]


@dataclass
class Solution:
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    residual: float = 0.0
    unknowns: int = 0
    equations: int = 0
    rank: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def free_dof(self) -> int:
        return max(0, self.unknowns - self.rank)

    @property
    def redundant(self) -> int:
        return max(0, self.equations - self.rank)

    @property
    def ok(self) -> bool:
        return not self.errors and self.residual <= 1e-6

    @property
    def state(self) -> str:
        if self.free_dof and self.redundant:
            return "under-constrained in places and over-constrained in others"
        if self.free_dof:
            return f"under-constrained by {self.free_dof} degree(s) of freedom"
        if self.redundant:
            return f"over-constrained by {self.redundant} redundant dimension(s)"
        return "fully constrained"

    def text(self) -> str:
        head = (f"layout: {len(self.points)} point(s), {self.equations} equation(s), "
                f"{self.unknowns} unknown(s) — {self.state}, "
                f"worst residual {self.residual:.4f} mm")
        return "\n".join([head] + [f"  {e}" for e in self.errors])


@dataclass
class Plan:
    """A parsed layout: everything the solver and the renderer need, resolved."""

    name: str = "plan"
    view: str = "front"
    scope: dict[str, float] = field(default_factory=dict)
    points: list[Point] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    boxes: list[dict] = field(default_factory=list)
    circles: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    dims: list[dict] = field(default_factory=list)
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        d = dict(d or {})
        scope, _errs = expr.resolve(d.get("vars") or {})
        view = str(d.get("view", "front") or "front").lower()
        return cls(
            name=str(d.get("name", "plan")),
            view=view if view in VIEWS else "front",
            scope=scope,
            points=[Point.from_dict(x, scope) for x in (d.get("points") or [])],
            constraints=[Constraint.from_dict(x, scope) for x in (d.get("constraints") or [])],
            boxes=[dict(x) for x in (d.get("boxes") or [])],
            circles=[dict(x) for x in (d.get("circles") or [])],
            lines=[dict(x) for x in (d.get("lines") or [])],
            dims=[dict(x) for x in (d.get("dims") or [])],
            note=str(d.get("note", "") or ""),
        )

    def point(self, pid: str) -> Point | None:
        for p in self.points:
            if p.id == pid:
                return p
        return None

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.points]


# ── validation ──────────────────────────────────────────────────────────────

def check(raw: dict) -> list[str]:
    """Everything wrong with a plan on paper, before it is solved."""
    plan = Plan.from_dict(raw)
    errs: list[str] = []
    _, verrs = expr.resolve((raw or {}).get("vars") or {})
    errs += [f"vars: {e}" for e in verrs]
    if str((raw or {}).get("view", "front")).lower() not in VIEWS:
        errs.append(f"view '{(raw or {}).get('view')}' is not one of {list(VIEWS)}")
    seen: set[str] = set()
    for i, pt in enumerate(plan.points):
        if not pt.id:
            errs.append(f"points[{i}]: every point needs an id")
        elif pt.id in seen:
            errs.append(f"points[{i}]: '{pt.id}' is declared twice")
        seen.add(pt.id)
    for i, c in enumerate(plan.constraints):
        if c.kind not in CONSTRAINTS:
            errs.append(f"constraints[{i}]: '{c.kind}' is not one of {list(CONSTRAINTS)}")
            continue
        need = {"fix": 1, "on_x": 1, "on_y": 1, "collinear": 3, "symmetric": 3}.get(c.kind, 2)
        refs = c.refs
        if len(refs) < need:
            errs.append(f"constraints[{i}] {c.kind}: needs {need} point(s), got {len(refs)}")
        for ref in refs:
            if ref not in seen:
                errs.append(f"constraints[{i}] {c.kind}: '{ref}' is not a declared point")
    for group, key in ((plan.boxes, "boxes"), (plan.circles, "circles")):
        for i, item in enumerate(group):
            at = str(item.get("at", "") or "")
            if at and at not in seen:
                errs.append(f"{key}[{i}]: '{at}' is not a declared point")
    for i, ln in enumerate(plan.lines + plan.dims):
        for ref in (str(ln.get("a", "")), str(ln.get("b", ""))):
            if ref and ref not in seen:
                errs.append(f"lines/dims[{i}]: '{ref}' is not a declared point")
    return errs


# ── the 2D constraint solve ─────────────────────────────────────────────────

def solve(raw: dict) -> Solution:
    """Solve the sketch and say honestly how well determined it is.

    The degrees-of-freedom report is the part worth having. A plan that is under-
    constrained has a shape the author did not pin down, and finding that out here
    costs nothing; finding it out in 3D looks like a part mysteriously in the wrong
    place, forty seconds later, with no indication that the plan never said where it
    should be.
    """
    plan = Plan.from_dict(raw)
    sol = Solution()
    sol.errors = check(raw)
    if not plan.points:
        sol.errors.append("a plan needs at least one point")
        return sol

    index = {p.id: n for n, p in enumerate(plan.points)}
    x0 = np.array([v for p in plan.points for v in (p.u, p.v)], dtype=float)
    sol.unknowns = int(x0.size)
    terms = [c for c in plan.constraints if c.kind in CONSTRAINTS
             and all(r in index for r in c.refs)]
    sol.equations = sum(c.width for c in terms)

    # Scale for the angular and collinearity residuals, so a 0.001 rad tilt on a
    # 900 mm bed is not swamped by a 0.5 mm distance error.
    span = max(1.0, float(np.abs(x0).max()) if x0.size else 1.0)

    def residual(x: np.ndarray) -> np.ndarray:
        out: list[float] = []
        for c in terms:
            out += _residual(c, x, index, plan, span)
        return np.array(out, dtype=float) if out else np.zeros(1)

    if not terms:
        sol.points = {p.id: (p.u, p.v) for p in plan.points}
        sol.rank = 0
        sol.errors.append("no constraints: every point keeps the position it was "
                          "written with, which is a sketch rather than a plan")
        return sol

    try:
        from scipy.optimize import least_squares

        res = least_squares(residual, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        x = res.x
        J = np.asarray(res.jac, dtype=float)
    except ImportError:
        x, J = _newton(residual, x0)

    r = residual(x)
    sol.residual = float(np.abs(r).max())
    sol.points = {p.id: (float(x[2 * index[p.id]]), float(x[2 * index[p.id] + 1]))
                  for p in plan.points}
    try:
        sol.rank = int(np.linalg.matrix_rank(J, tol=1e-9 * max(1.0, float(np.abs(J).max()))))
    except Exception:
        sol.rank = 0
    if sol.residual > 1e-6:
        sol.errors.append(f"the dimensions contradict each other: the best fit still "
                          f"misses by {sol.residual:.4f} mm")
    return sol


def _residual(c: Constraint, x: np.ndarray, index: dict[str, int],
              plan: Plan, span: float) -> list[float]:
    def pt(pid: str) -> tuple[float, float]:
        n = index[pid]
        return (float(x[2 * n]), float(x[2 * n + 1]))

    if c.kind == "fix":
        want = plan.point(c.a)
        u, v = pt(c.a)
        return [u - want.u, v - want.v]
    if c.kind == "on_x":
        return [pt(c.a)[0] - c.value]
    if c.kind == "on_y":
        return [pt(c.a)[1] - c.value]
    a, b = pt(c.a), pt(c.b)
    if c.kind == "dx":
        return [(b[0] - a[0]) - c.value]
    if c.kind == "dy":
        return [(b[1] - a[1]) - c.value]
    if c.kind == "distance":
        return [math.hypot(b[0] - a[0], b[1] - a[1]) - c.value]
    if c.kind == "horizontal":
        return [b[1] - a[1]]
    if c.kind == "vertical":
        return [b[0] - a[0]]
    if c.kind == "coincident":
        return [b[0] - a[0], b[1] - a[1]]
    if c.kind == "angle":
        have = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        return [((have - c.value + 180.0) % 360.0 - 180.0) * span / 57.29578]
    if c.kind == "collinear":
        d = pt(c.c)
        cross = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
        return [cross / span]
    if c.kind == "symmetric":
        mid = pt(c.about)
        return [(a[0] + b[0]) / 2.0 - mid[0], (a[1] + b[1]) / 2.0 - mid[1]]
    return []


def _newton(residual, x0, rounds: int = 40):
    """Levenberg-flavoured fallback so a plan solves without scipy installed."""
    x = np.asarray(x0, dtype=float).copy()
    J = np.zeros((residual(x).size, x.size))
    for _ in range(rounds):
        r = residual(x)
        for c in range(x.size):
            probe = x.copy()
            probe[c] += 1e-6
            J[:, c] = (residual(probe) - r) / 1e-6
        if float(np.abs(r).max()) < 1e-10:
            break
        try:
            step, *_ = np.linalg.lstsq(J, -r, rcond=1e-12)
        except np.linalg.LinAlgError:
            break
        x = x + step
    return x, J


# ── freezing, and the join to the 3D model ──────────────────────────────────

def freeze(raw: dict) -> tuple[dict, Solution]:
    """Solve the plan and write the answer back into it.

    A frozen plan is the contract between the review and the build: the solved
    coordinates and a hash of the inputs that produced them. If someone edits a
    dimension afterwards the hash stops matching and `stale` says so, because a
    model built from a plan nobody re-approved is a model nobody approved.
    """
    sol = solve(raw)
    out = dict(raw or {})
    out["solved"] = {k: [round(v[0], 6), round(v[1], 6)] for k, v in sol.points.items()}
    out["frozen"] = True
    out["hash"] = digest(raw)
    out["state"] = sol.state
    return out, sol


def digest(raw: dict) -> str:
    """A hash of everything that decides the solved positions, and nothing else."""
    keep = {k: (raw or {}).get(k) for k in ("view", "vars", "points", "constraints")}
    blob = json.dumps(keep, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def stale(raw: dict) -> bool:
    """True when the plan has been edited since it was last frozen and solved."""
    if not raw or not raw.get("frozen"):
        return bool(raw)
    return str(raw.get("hash", "")) != digest(raw)


_CACHE: dict[str, dict[str, tuple[float, float]]] = {}


def solved_points(raw: dict) -> dict[str, tuple[float, float]]:
    """The plan's solved 2D positions, solving on demand and caching by hash."""
    if not raw:
        return {}
    if raw.get("solved") and not stale(raw):
        return {k: (float(v[0]), float(v[1])) for k, v in (raw.get("solved") or {}).items()}
    key = digest(raw)
    if key not in _CACHE:
        _CACHE[key] = solve(raw).points
    return _CACHE[key]


def solved_point(raw: dict, pid: str) -> tuple[float, float, float] | None:
    """One solved point, lifted from the sketch plane into world coordinates.

    This is the function the assembler calls for every datum that names a layout
    point, and it is the whole mechanism by which an approved drawing positions a
    3D assembly.
    """
    pts = solved_points(raw)
    if pid not in pts:
        return None
    u, v = pts[pid]
    return lift(str((raw or {}).get("view", "front") or "front").lower(), u, v)


def lift(view: str, u: float, v: float) -> tuple[float, float, float]:
    """(u, v) in the named sketch plane -> (x, y, z) in the assembly."""
    ax = AXES.get(view, AXES["front"])
    out = [0.0, 0.0, 0.0]
    out[ax[0]] = float(u)
    out[ax[1]] = float(v)
    return (out[0], out[1], out[2])


def flatten(view: str, xyz) -> tuple[float, float]:
    """The inverse of `lift`: a world point projected into the sketch plane."""
    ax = AXES.get(view, AXES["front"])
    return (float(xyz[ax[0]]), float(xyz[ax[1]]))


# ── the drawing a human actually reviews ────────────────────────────────────
# Deliberately hand-written SVG rather than a plotting library: the output has to
# read like a drawing, which means real dimension lines with arrowheads and witness
# lines, a title block, and nothing else. It also has to be diffable, because the
# review step is a conversation and two versions of a plan should be comparable.

STYLE = """
  .outline { fill: none; stroke: #1a1a1a; stroke-width: 0.9; }
  .env     { fill: #4a7ebb14; stroke: #4a7ebb; stroke-width: 0.7; }
  .centre  { fill: none; stroke: #b03030; stroke-width: 0.5;
             stroke-dasharray: 9 3 1.5 3; }
  .hole    { fill: none; stroke: #1a1a1a; stroke-width: 0.6; }
  .pcd     { fill: none; stroke: #b03030; stroke-width: 0.45;
             stroke-dasharray: 7 2.5 1.2 2.5; }
  .dim     { fill: none; stroke: #205020; stroke-width: 0.45; }
  .wit     { fill: none; stroke: #205020; stroke-width: 0.35;
             stroke-dasharray: 2 2; }
  .pt      { fill: none; stroke: #303030; stroke-width: 0.5; }
  text     { font-family: "DejaVu Sans", "Helvetica Neue", Arial, sans-serif;
             fill: #1a1a1a; }
  .lbl     { font-size: 3.6px; }
  .dimtext { font-size: 3.4px; fill: #205020; }
  .title   { font-size: 4.6px; font-weight: 600; }
  .meta    { font-size: 3.2px; fill: #555; }
"""


def svg(raw: dict, *, width_px: int = 1100) -> str:
    """The plan as a dimensioned drawing. Pure string building, no dependencies."""
    plan = Plan.from_dict(raw)
    pts = solved_points(raw) or {p.id: (p.u, p.v) for p in plan.points}
    sol = solve(raw)
    body: list[str] = []
    xs: list[float] = []
    ys: list[float] = []

    def note(u: float, v: float) -> None:
        xs.append(u)
        ys.append(v)

    for u, v in pts.values():
        note(u, v)

    # envelopes first, so dimensions and points draw over them
    for box in plan.boxes:
        rect = _box_rect(plan, pts, box)
        if rect is None:
            continue
        u0, v0, w, h = rect
        note(u0, v0)
        note(u0 + w, v0 + h)
        body.append(f'<rect class="env" x="{u0:.4f}" y="{-(v0 + h):.4f}" '
                    f'width="{w:.4f}" height="{h:.4f}" rx="0.6"/>')
        label = str(box.get("label") or box.get("id") or "")
        if label:
            body.append(f'<text class="lbl" x="{u0 + w / 2:.4f}" y="{-(v0 + h / 2):.4f}" '
                        f'text-anchor="middle">{_esc(label)}</text>')

    for ln in plan.lines:
        a, b = pts.get(str(ln.get("a"))), pts.get(str(ln.get("b")))
        if a is None or b is None:
            continue
        klass = "centre" if str(ln.get("kind", "centre")) == "centre" else "outline"
        body.append(f'<line class="{klass}" x1="{a[0]:.4f}" y1="{-a[1]:.4f}" '
                    f'x2="{b[0]:.4f}" y2="{-b[1]:.4f}"/>')

    for circ in plan.circles:
        at = pts.get(str(circ.get("at", "")))
        if at is None:
            continue
        d, _ = expr.number(circ.get("d", circ.get("diameter", 0.0)), plan.scope)
        r = float(d) / 2.0
        note(at[0] - r, at[1] - r)
        note(at[0] + r, at[1] + r)
        body.append(f'<circle class="pcd" cx="{at[0]:.4f}" cy="{-at[1]:.4f}" r="{r:.4f}"/>')
        body += _cross(at[0], at[1], max(2.0, r * 0.25))
        count = int(circ.get("count", 0) or 0)
        hole, _ = expr.number(circ.get("hole_d", 0.0), plan.scope)
        for i in range(count):
            th = math.radians(360.0 * i / count + float(circ.get("start_angle", 0.0) or 0.0))
            hx, hy = at[0] + r * math.cos(th), at[1] + r * math.sin(th)
            body.append(f'<circle class="hole" cx="{hx:.4f}" cy="{-hy:.4f}" '
                        f'r="{max(0.8, float(hole) / 2.0):.4f}"/>')
        label = str(circ.get("label") or "")
        if label or count:
            text = label or f"{count} x on {d:g} PCD"
            body.append(f'<text class="lbl" x="{at[0]:.4f}" y="{-(at[1] + r + 4.0):.4f}" '
                        f'text-anchor="middle">{_esc(text)}</text>')

    for pt in plan.points:
        at = pts.get(pt.id)
        if at is None:
            continue
        body += _cross(at[0], at[1], 2.2)
        body.append(f'<text class="lbl" x="{at[0] + 2.8:.4f}" y="{-(at[1] + 2.4):.4f}">'
                    f'{_esc(pt.id)}</text>')

    for dim in plan.dims:
        drawn, box = _dimension(plan, pts, dim)
        body += drawn
        for u, v in box:
            note(u, v)

    return _document(plan, sol, body, xs, ys, width_px)


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _cross(u: float, v: float, r: float) -> list[str]:
    return [f'<line class="pt" x1="{u - r:.4f}" y1="{-v:.4f}" x2="{u + r:.4f}" y2="{-v:.4f}"/>',
            f'<line class="pt" x1="{u:.4f}" y1="{-(v - r):.4f}" x2="{u:.4f}" y2="{-(v + r):.4f}"/>']


def _box_rect(plan: Plan, pts: dict, box: dict) -> tuple[float, float, float, float] | None:
    """(u0, v0, w, h) for an envelope, honouring its anchor convention."""
    at = pts.get(str(box.get("at", "")))
    if at is None:
        return None
    w, _ = expr.number(box.get("w", box.get("width", 0.0)), plan.scope)
    h, _ = expr.number(box.get("h", box.get("height", 0.0)), plan.scope)
    w, h = abs(float(w)), abs(float(h))
    anchor = str(box.get("anchor", "centre") or "centre").lower()
    if anchor in ("centre", "center"):
        return (at[0] - w / 2.0, at[1] - h / 2.0, w, h)
    if anchor in ("bottom", "bottom_centre", "bottom_center"):
        return (at[0] - w / 2.0, at[1], w, h)
    if anchor in ("top", "top_centre", "top_center"):
        return (at[0] - w / 2.0, at[1] - h, w, h)
    if anchor in ("left", "left_centre"):
        return (at[0], at[1] - h / 2.0, w, h)
    if anchor in ("right", "right_centre"):
        return (at[0] - w, at[1] - h / 2.0, w, h)
    return (at[0], at[1], w, h)


def _dimension(plan: Plan, pts: dict, dim: dict) -> tuple[list[str], list[tuple[float, float]]]:
    """One dimension line with witness lines, arrows and its value."""
    a, b = pts.get(str(dim.get("a", ""))), pts.get(str(dim.get("b", "")))
    if a is None or b is None:
        return ([], [])
    measure = str(dim.get("measure", "dx") or "dx").lower()
    off, _ = expr.number(dim.get("offset", 12.0), plan.scope)
    off = float(off)
    out: list[str] = []
    touched: list[tuple[float, float]] = []

    if measure in ("dx", "x", "horizontal"):
        v = min(a[1], b[1]) - off if off >= 0 else max(a[1], b[1]) - off
        out += [_line("wit", a[0], a[1], a[0], v), _line("wit", b[0], b[1], b[0], v)]
        out += _arrowed(a[0], v, b[0], v)
        value = abs(b[0] - a[0])
        out.append(_text((a[0] + b[0]) / 2.0, v + 1.4, f"{value:.4g}"))
        touched += [(a[0], v), (b[0], v)]
    elif measure in ("dy", "y", "vertical"):
        u = max(a[0], b[0]) + off
        out += [_line("wit", a[0], a[1], u, a[1]), _line("wit", b[0], b[1], u, b[1])]
        out += _arrowed(u, a[1], u, b[1])
        value = abs(b[1] - a[1])
        out.append(_text(u + 1.6, (a[1] + b[1]) / 2.0, f"{value:.4g}", anchor="start"))
        touched += [(u, a[1]), (u, b[1])]
    else:
        out += _arrowed(a[0], a[1], b[0], b[1])
        value = math.hypot(b[0] - a[0], b[1] - a[1])
        out.append(_text((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0 + 1.6, f"{value:.4g}"))
        touched += [a, b]
    label = str(dim.get("label") or "")
    if label:
        out.append(_text((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0 - 3.0, label))
    return (out, touched)


def _line(klass: str, u1: float, v1: float, u2: float, v2: float) -> str:
    return (f'<line class="{klass}" x1="{u1:.4f}" y1="{-v1:.4f}" '
            f'x2="{u2:.4f}" y2="{-v2:.4f}"/>')


def _arrowed(u1: float, v1: float, u2: float, v2: float, head: float = 2.2) -> list[str]:
    ang = math.atan2(v2 - v1, u2 - u1)
    out = [_line("dim", u1, v1, u2, v2)]
    for (u, v, direction) in ((u1, v1, ang), (u2, v2, ang + math.pi)):
        for sign in (+1, -1):
            th = direction + sign * math.radians(16.0)
            out.append(_line("dim", u, v, u + head * math.cos(th), v + head * math.sin(th)))
    return out


def _text(u: float, v: float, body: str, *, anchor: str = "middle") -> str:
    return (f'<text class="dimtext" x="{u:.4f}" y="{-v:.4f}" '
            f'text-anchor="{anchor}">{_esc(body)}</text>')


def _document(plan: Plan, sol: Solution, body: list[str], xs: list[float],
              ys: list[float], width_px: int) -> str:
    if not xs:
        xs, ys = [0.0, 100.0], [0.0, 60.0]
    pad = max(24.0, 0.10 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad * 1.9
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    scale = width_px / w
    bar = _scale_bar(x0 + pad * 0.35, y0 + pad * 0.35, w)
    head = (f'<text class="title" x="{x0 + pad * 0.3:.3f}" y="{-(y1 - pad * 0.55):.3f}">'
            f'{_esc(plan.name)}</text>'
            f'<text class="meta" x="{x0 + pad * 0.3:.3f}" y="{-(y1 - pad * 0.95):.3f}">'
            f'{_esc(plan.view)} view, mm — {_esc(sol.state)}'
            f'{", CONTRADICTORY" if sol.residual > 1e-6 else ""}</text>')
    frame = (f'<rect class="outline" x="{x0 + 1.5:.3f}" y="{-(y1 - 1.5):.3f}" '
             f'width="{w - 3.0:.3f}" height="{h - 3.0:.3f}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" '
        f'height="{int(h * scale)}" viewBox="{x0:.3f} {-y1:.3f} {w:.3f} {h:.3f}">'
        f'<style>{STYLE}</style><rect x="{x0:.3f}" y="{-y1:.3f}" width="{w:.3f}" '
        f'height="{h:.3f}" fill="#fdfdfb"/>{frame}{head}'
        + "".join(body) + "".join(bar) + "</svg>"
    )


def _scale_bar(u: float, v: float, span: float) -> list[str]:
    """A labelled bar, so the drawing survives being screenshotted at any zoom."""
    step = 10.0 ** math.floor(math.log10(max(span, 1.0) / 5.0))
    for mult in (1.0, 2.0, 5.0):
        if step * mult >= span / 8.0:
            step *= mult
            break
    out = [_line("outline", u, v, u + step, v),
           _line("outline", u, v, u, v + 1.4),
           _line("outline", u + step, v, u + step, v + 1.4),
           _text(u + step / 2.0, v + 2.2, f"{step:g} mm")]
    return out


# ── deriving a plan from something that already exists ──────────────────────

def draft_from(asm, *, view: str = "front", name: str = "") -> dict:
    """A first-cut plan that reproduces an assembly, ready for a human to edit.

    Useful in both directions. Forwards, it is how an existing spec joins the
    2D-first workflow without being rewritten. Backwards, it is the cheapest
    possible review artefact for a model that already built: one page showing where
    every part actually ended up, in dimensions rather than in pixels.
    """
    view = view if view in VIEWS else "front"
    ax = AXES[view]
    datum = asm.spec.datum
    points: list[dict] = []
    boxes: list[dict] = []
    circles: list[dict] = []
    constraints: list[dict] = [{"kind": "fix", "a": datum}] if datum else []
    dims: list[dict] = []

    origin = None
    for part in asm.spec.parts:
        box = _part_box(asm, part.id)
        if box is None:
            continue
        centre = [(box[k] + box[k + 3]) / 2.0 for k in range(3)]
        u, v = centre[ax[0]], centre[ax[1]]
        points.append({"id": part.id, "x": round(u, 4), "y": round(v, 4),
                       "note": part.label or part.skill})
        boxes.append({"id": part.id, "part": part.id, "at": part.id,
                      "w": round(box[ax[0] + 3] - box[ax[0]], 4),
                      "h": round(box[ax[1] + 3] - box[ax[1]], 4),
                      "anchor": "centre",
                      "label": f"{part.id}" + (f" x{part.count}" if part.count > 1 else "")})
        if part.id == datum:
            origin = (u, v)
    if origin is not None:
        for pt in points:
            if pt["id"] == datum:
                continue
            constraints.append({"kind": "dx", "a": datum, "b": pt["id"],
                                "value": round(pt["x"] - origin[0], 4)})
            constraints.append({"kind": "dy", "a": datum, "b": pt["id"],
                                "value": round(pt["y"] - origin[1], 4)})
    circles += _ring_circles(asm, view)
    have = {p["id"] for p in points}
    for circ in circles:
        pid = str(circ["at"])
        if pid in have:
            continue
        points.append({"id": pid, "x": circ.pop("_u"), "y": circ.pop("_v"),
                       "note": "bolt circle centre"})
        have.add(pid)
        if origin is not None and datum:
            constraints.append({"kind": "dx", "a": datum, "b": pid,
                                "value": round(points[-1]["x"] - origin[0], 4)})
            constraints.append({"kind": "dy", "a": datum, "b": pid,
                                "value": round(points[-1]["y"] - origin[1], 4)})
    for circ in circles:
        circ.pop("_u", None)
        circ.pop("_v", None)
    # Only the largest few spans are worth annotating; a drawing with one dimension
    # per part is unreadable, and unreadable is the same as unreviewed.
    for pt in sorted(points, key=lambda p: -abs(p["x"] - (origin or (0, 0))[0]))[:4]:
        if pt["id"] == datum or origin is None:
            continue
        dims.append({"a": datum, "b": pt["id"], "measure": "dx", "offset": 18.0})
    return {
        "name": name or f"{asm.spec.name} — derived plan",
        "view": view,
        "note": "derived from a built assembly; edit the dimensions and re-freeze",
        "points": points,
        "constraints": constraints,
        "boxes": boxes,
        "circles": circles,
        "dims": dims,
    }


def _part_box(asm, pid: str) -> tuple[float, ...] | None:
    lo = [1e18] * 3
    hi = [-1e18] * 3
    found = False
    for inst in asm.of_part(pid):
        try:
            bb = inst.solid.bounding_box()
        except Exception:
            continue
        found = True
        for k, (a, b) in enumerate(((bb.min.X, bb.max.X), (bb.min.Y, bb.max.Y),
                                    (bb.min.Z, bb.max.Z))):
            lo[k] = min(lo[k], float(a))
            hi[k] = max(hi[k], float(b))
    return tuple(lo + hi) if found else None


def _ring_circles(asm, view: str) -> list[dict]:
    """Bolt circles that face the reader in this view, so the plan shows them round."""
    from . import xform

    ax = AXES[view]
    normal = [k for k in (0, 1, 2) if k not in ax][0]
    out: list[dict] = []
    seen: set[str] = set()
    for inst in asm.instances:
        if inst.pid in seen:
            continue
        for name, conn in (inst.result.connectors or {}).items():
            if conn.count < 2 or not conn.layout.startswith("ring"):
                continue
            world = [inst.solid.location * conn.location(i) for i in range(conn.count)]
            pos = [[float(w.position.X), float(w.position.Y), float(w.position.Z)]
                   for w in world]
            centre = [sum(p[k] for p in pos) / len(pos) for k in range(3)]
            axis = xform.canonical_axis(xform.loc_to_Rt(inst.solid.location)[0][:, 2])
            if axis is None or abs(float(axis[normal])) < 0.9:
                continue
            radius = max(math.dist([p[ax[0]], p[ax[1]]], [centre[ax[0]], centre[ax[1]]])
                         for p in pos)
            pid = f"{inst.pid}~{name}"
            out.append({"id": pid, "at": pid, "d": round(radius * 2.0, 3),
                        "count": conn.count, "hole_d": 3.0,
                        "label": f"{inst.pid}.{name}: {conn.count} x on "
                                 f"{radius * 2.0:.1f} PCD",
                        "_u": round(centre[ax[0]], 4), "_v": round(centre[ax[1]], 4)})
            seen.add(inst.pid)
            break
    return out


def compare(asm, raw: dict, *, tol: float = 1.0) -> list[str]:
    """Where the built assembly disagrees with the plan that was approved.

    An envelope in the plan names a part, a size and a place. That is a claim about
    the finished model, so check it: a part that came out 40 mm from where the
    reviewed drawing put it is a failure even if every mate in the spec holds, and
    it is the failure the 2D-first workflow exists to catch.
    """
    if not raw:
        return []
    plan = Plan.from_dict(raw)
    pts = solved_points(raw)
    view = plan.view
    ax = AXES.get(view, AXES["front"])
    out: list[str] = []
    for box in plan.boxes:
        pid = str(box.get("part") or box.get("id") or "")
        if not pid or asm.spec.part(pid) is None:
            continue
        at = pts.get(str(box.get("at", "")))
        rect = _box_rect(plan, pts, box)
        got = _part_box(asm, pid)
        if at is None or rect is None or got is None:
            continue
        cu = (got[ax[0]] + got[ax[0] + 3]) / 2.0
        cv = (got[ax[1]] + got[ax[1] + 3]) / 2.0
        want_u = rect[0] + rect[2] / 2.0
        want_v = rect[1] + rect[3] / 2.0
        off = math.hypot(cu - want_u, cv - want_v)
        if off > tol:
            out.append(f"'{pid}' sits {off:.2f} mm from where the plan puts it "
                       f"({cu:.1f}, {cv:.1f}) vs ({want_u:.1f}, {want_v:.1f}) in {view} view")
        for k, (label, want) in enumerate((("width", rect[2]), ("height", rect[3]))):
            have = got[ax[k] + 3] - got[ax[k]]
            if want > 0 and abs(have - want) > max(tol, 0.02 * want):
                out.append(f"'{pid}' {label} is {have:.2f} mm; the plan drew "
                           f"{want:.2f} mm")
    return out






