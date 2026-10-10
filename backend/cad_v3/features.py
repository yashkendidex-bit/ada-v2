"""Sketch and feature modelling — how the vocabulary stops being closed.

Every skill in the original vocabulary is a hand-written Python function, and all
but a handful are bodies of revolution about +Z. That is a fine way to build a
turbocharger and no way at all to build the cast bearing housing with lugs and
ribs, the sheet bracket, the hose runs or the PCB in a real machine: those shapes
are not revolutions, and there was no way to describe one without an engineer
writing a new function first.

Real CAD does not work that way. It draws a 2D profile and turns it into a solid,
then adds and removes more solids, then rounds the edges. That sequence is a small
language, and this module is an interpreter for it — so a new shape is a *plan* in
the spec rather than new code in the repository, and `synth.py` writing a skill
becomes the exception rather than the only route.

    {"op": "profile", "name": "web", "plane": "XZ",
     "shapes": [{"kind": "polygon", "points": [[0, 0], [60, 0], [60, 8], [12, 34], [0, 34]]}]}
    {"op": "revolve", "profile": "web", "arc": 360}
    {"op": "hole", "d": 12, "depth": 40, "at": [0, 0, -1]}
    {"op": "fillet", "radius": 1.5, "select": {"kind": "top"}}

Two rules keep this from becoming a hole in the anti-hallucination design. First,
a plan describes a *part in its own frame* and nothing else — there is no way to
say where the part goes, which is still decided entirely by mates and datums.
Second, every op is validated and every failure is reported by name, so a plan
that cannot build produces an error a repair round can read rather than an
exception nobody sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from build123d import (Align, Axis, Circle, Cone, Cylinder, Ellipse, GeomType, Kind,
                       Location, Plane, Polygon, Pos, Rectangle, RegularPolygon,
                       Rot, Sphere, Spline, Torus, Wedge, chamfer, extrude, fillet,
                       loft, mirror, offset, revolve, sweep)
from build123d import Curve, Polyline, SlotOverall

PLANES = {
    "XY": Plane.XY, "YX": Plane.YX, "XZ": Plane.XZ,
    "ZX": Plane.ZX, "YZ": Plane.YZ, "ZY": Plane.ZY,
}
AXES = {"X": Axis.X, "Y": Axis.Y, "Z": Axis.Z}
MODES = ("add", "cut", "intersect")
SHAPE_KINDS = ("rect", "circle", "ellipse", "polygon", "slot", "regular", "spline")
OPS = ("box", "cylinder", "cone", "sphere", "torus", "wedge", "profile", "extrude",
       "revolve", "sweep", "scroll", "loft", "tube", "hole", "shell", "fillet",
       "chamfer", "pattern_circular", "pattern_linear", "mirror", "scale_last")


class FeatureError(ValueError):
    """A plan that cannot build, named precisely enough to repair."""


@dataclass
class Result:
    solid: Any = None
    warnings: list[str] = field(default_factory=list)
    ops: int = 0


def _f(d: dict, key: str, default: float = 0.0) -> float:
    val = d.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        raise FeatureError(f"'{key}' = {val!r} is not a number") from None


def _i(d: dict, key: str, default: int = 0) -> int:
    return int(round(_f(d, key, default)))


def _vec(d: dict, key: str, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    raw = d.get(key) or default
    try:
        out = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise FeatureError(f"'{key}' = {raw!r} must be three numbers") from None
    while len(out) < 3:
        out.append(0.0)
    return (out[0], out[1], out[2])


def _align(name: str) -> tuple:
    name = str(name or "centre").lower()
    if name in ("bottom", "z_min", "base"):
        return (Align.CENTER, Align.CENTER, Align.MIN)
    if name in ("top", "z_max"):
        return (Align.CENTER, Align.CENTER, Align.MAX)
    if name == "min":
        return (Align.MIN, Align.MIN, Align.MIN)
    return (Align.CENTER, Align.CENTER, Align.CENTER)


def _place(shape, op: dict):
    """Apply the op's own `at`/`rot`, in that fixed order: rotate then translate."""
    at, rot = _vec(op, "at"), _vec(op, "rot")
    if any(rot):
        shape = Rot(rot[0], rot[1], rot[2]) * shape
    if any(at):
        shape = Pos(at[0], at[1], at[2]) * shape
    return shape


def _delta(a, b) -> tuple[float, float, float]:
    """b - a, used to aim a swept section along the path it starts on."""
    return (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]),
            float(b[2]) - float(a[2]))


# ── 2D profiles ─────────────────────────────────────────────────────────────

def _shape2d(item: dict):
    """One entry of a profile's `shapes` list, in the profile's own plane."""
    kind = str(item.get("kind", "rect") or "rect").lower()
    if kind == "rect":
        w, h = _f(item, "w", _f(item, "width", 10.0)), _f(item, "h", _f(item, "height", 10.0))
        sk = Rectangle(max(w, 1e-4), max(h, 1e-4), align=_xy_align(item))
    elif kind == "circle":
        d = _f(item, "d", _f(item, "diameter", 10.0))
        sk = Circle(max(d, 1e-4) / 2.0)
    elif kind == "ellipse":
        sk = Ellipse(max(_f(item, "a", 10.0), 1e-4), max(_f(item, "b", 6.0), 1e-4))
    elif kind == "regular":
        d, n = _f(item, "d", 10.0), max(3, _i(item, "n", 6))
        sk = RegularPolygon(max(d, 1e-4) / 2.0, n)
    elif kind == "slot":
        sk = SlotOverall(max(_f(item, "length", 20.0), 1e-3),
                         max(_f(item, "h", _f(item, "height", 8.0)), 1e-3))
    elif kind == "polygon":
        pts = _points2d(item)
        if len(pts) < 3:
            raise FeatureError("a polygon needs at least three points")
        sk = Polygon(*pts, align=None)
    elif kind == "spline":
        pts = _points2d(item)
        if len(pts) < 3:
            raise FeatureError("a spline profile needs at least three points")
        from build123d import make_face

        sk = make_face(Curve() + Spline(*[(p[0], p[1], 0.0) for p in pts],
                                        tangents=None) +
                       Polyline((pts[-1][0], pts[-1][1], 0.0), (pts[0][0], pts[0][1], 0.0)))
    else:
        raise FeatureError(f"'{kind}' is not a profile shape; use one of {list(SHAPE_KINDS)}")
    spin = _f(item, "rot", 0.0)
    if spin:
        sk = Rot(0.0, 0.0, spin) * sk
    at = item.get("at") or (0.0, 0.0)
    try:
        u, v = float(at[0]), float(at[1])
    except (TypeError, ValueError, IndexError):
        raise FeatureError(f"'at' = {at!r} must be two numbers in the profile plane") from None
    if u or v:
        sk = Pos(u, v, 0.0) * sk
    return sk


def _xy_align(item: dict) -> tuple:
    name = str(item.get("align", "centre") or "centre").lower()
    if name == "min":
        return (Align.MIN, Align.MIN)
    if name in ("bottom", "v_min"):
        return (Align.CENTER, Align.MIN)
    if name in ("top", "v_max"):
        return (Align.CENTER, Align.MAX)
    if name == "left":
        return (Align.MIN, Align.CENTER)
    if name == "right":
        return (Align.MAX, Align.CENTER)
    return (Align.CENTER, Align.CENTER)


def _points2d(item: dict) -> list[tuple[float, float]]:
    raw = item.get("points") or []
    out: list[tuple[float, float]] = []
    for p in raw:
        try:
            out.append((float(p[0]), float(p[1])))
        except (TypeError, ValueError, IndexError):
            raise FeatureError(f"point {p!r} must be two numbers") from None
    return out


def _points3d(raw: Sequence) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for p in raw or []:
        try:
            q = [float(v) for v in p]
        except (TypeError, ValueError):
            raise FeatureError(f"path point {p!r} must be numbers") from None
        while len(q) < 3:
            q.append(0.0)
        out.append((q[0], q[1], q[2]))
    return out


def _plane_of(op: dict) -> Plane:
    name = str(op.get("plane", "XY") or "XY").upper()
    if name not in PLANES:
        raise FeatureError(f"'{name}' is not a plane; use one of {sorted(PLANES)}")
    plane = PLANES[name]
    z = _f(op, "offset", 0.0)
    if z:
        plane = plane.offset(z)
    return plane


# ── the interpreter ─────────────────────────────────────────────────────────

class Shape:
    """Runs a plan. One instance per part; no state survives between parts."""

    def __init__(self) -> None:
        self.part = None
        self.profiles: dict[str, Any] = {}
        self.order: list[str] = []
        self.last = None          # the tool the previous op produced, for patterning
        self.last_mode = "add"
        self.warnings: list[str] = []

    # ── plumbing ────────────────────────────────────────────────────────
    def run(self, plan: Sequence[dict]) -> Result:
        if not plan:
            raise FeatureError("a shape plan needs at least one operation")
        count = 0
        for i, raw in enumerate(plan):
            if not isinstance(raw, dict):
                raise FeatureError(f"plan[{i}] is {type(raw).__name__}, not an operation")
            op = str(raw.get("op", "") or "").lower()
            fn = getattr(self, f"_op_{op}", None)
            if fn is None:
                raise FeatureError(f"plan[{i}]: '{op}' is not an operation; "
                                   f"use one of {list(OPS)}")
            try:
                fn(raw)
            except FeatureError as exc:
                raise FeatureError(f"plan[{i}] ({op}): {exc}") from None
            except Exception as exc:
                raise FeatureError(f"plan[{i}] ({op}): {type(exc).__name__}: {exc}") from None
            count += 1
        if self.part is None:
            raise FeatureError("the plan produced no solid — the first op that makes "
                               "material must be an add")
        return Result(solid=self.part, warnings=self.warnings, ops=count)

    def _apply(self, tool, op: dict) -> None:
        mode = str(op.get("mode", "add") or "add").lower()
        if mode not in MODES:
            raise FeatureError(f"mode '{mode}' is not one of {list(MODES)}")
        self.last, self.last_mode = tool, mode
        self._combine(tool, mode)

    def _combine(self, tool, mode: str) -> None:
        if tool is None:
            return
        if self.part is None:
            if mode != "add":
                raise FeatureError(f"'{mode}' needs existing material to work on")
            self.part = tool
            return
        if mode == "add":
            self.part = self.part + tool
        elif mode == "cut":
            self.part = self.part - tool
        else:
            self.part = self.part & tool

    def _profile(self, op: dict, key: str = "profile"):
        name = str(op.get(key, "") or "")
        if name:
            if name not in self.profiles:
                raise FeatureError(f"no profile named '{name}'; "
                                   f"defined: {self.order or 'none'}")
            return self.profiles[name]
        if not self.order:
            raise FeatureError("no profile has been drawn yet")
        return self.profiles[self.order[-1]]

    # ── primitives ──────────────────────────────────────────────────────
    def _op_box(self, op: dict) -> None:
        from build123d import Box

        w = max(_f(op, "w", _f(op, "width", 10.0)), 1e-4)
        d = max(_f(op, "d", _f(op, "depth", 10.0)), 1e-4)
        h = max(_f(op, "h", _f(op, "height", 10.0)), 1e-4)
        self._apply(_place(Box(w, d, h, align=_align(op.get("align"))), op), op)

    def _op_cylinder(self, op: dict) -> None:
        r = max(_f(op, "d", 10.0), 1e-4) / 2.0
        h = max(_f(op, "h", _f(op, "height", 10.0)), 1e-4)
        arc = _f(op, "arc", 360.0)
        cyl = Cylinder(r, h, arc_size=max(1e-3, min(360.0, arc)),
                       align=_align(op.get("align")))
        self._apply(_place(cyl, op), op)

    def _op_cone(self, op: dict) -> None:
        r1 = max(_f(op, "d1", 20.0), 0.0) / 2.0
        r2 = max(_f(op, "d2", 0.0), 0.0) / 2.0
        h = max(_f(op, "h", 20.0), 1e-4)
        self._apply(_place(Cone(r1, r2, h, align=_align(op.get("align"))), op), op)

    def _op_sphere(self, op: dict) -> None:
        r = max(_f(op, "d", 20.0), 1e-4) / 2.0
        self._apply(_place(Sphere(r, align=_align(op.get("align"))), op), op)

    def _op_torus(self, op: dict) -> None:
        major = max(_f(op, "d", 40.0), 1e-3) / 2.0
        minor = max(_f(op, "tube_d", 6.0), 1e-4) / 2.0
        self._apply(_place(Torus(major, minor, align=_align(op.get("align"))), op), op)

    def _op_wedge(self, op: dict) -> None:
        w = max(_f(op, "w", 20.0), 1e-3)
        d = max(_f(op, "d", 20.0), 1e-3)
        h = max(_f(op, "h", 10.0), 1e-3)
        # `xmin`/`zmin` are the *top* face's extents in build123d's argument order,
        # so a wedge is described as "this size at the base, that size at the top".
        top_w = max(_f(op, "top_w", w * 0.4), 0.0)
        top_d = max(_f(op, "top_d", d * 0.4), 0.0)
        self._apply(_place(Wedge(w, d, h, (w - top_w) / 2.0, (d - top_d) / 2.0,
                                 (w + top_w) / 2.0, (d + top_d) / 2.0,
                                 align=_align(op.get("align"))), op), op)

    # ── sketch, then feature ────────────────────────────────────────────
    def _op_profile(self, op: dict) -> None:
        """Draw a 2D profile on a named plane and keep it for a later feature."""
        plane = _plane_of(op)
        shapes = op.get("shapes") or []
        if not shapes:
            raise FeatureError("a profile needs at least one entry in `shapes`")
        sketch = None
        for item in shapes:
            if not isinstance(item, dict):
                raise FeatureError(f"shape {item!r} is not an object")
            piece = _shape2d(item)
            mode = str(item.get("mode", "add") or "add").lower()
            if sketch is None:
                if mode != "add":
                    raise FeatureError("the first shape in a profile must be an add")
                sketch = piece
            elif mode == "cut":
                sketch = sketch - piece
            elif mode == "intersect":
                sketch = sketch & piece
            else:
                sketch = sketch + piece
        name = str(op.get("name", "") or f"profile{len(self.order)}")
        self.profiles[name] = plane * sketch
        if name not in self.order:
            self.order.append(name)

    def _op_extrude(self, op: dict) -> None:
        amount = _f(op, "amount", _f(op, "h", 10.0))
        if abs(amount) < 1e-6:
            raise FeatureError("extrude needs a non-zero amount")
        taper = _f(op, "taper", 0.0)
        both = bool(op.get("both", False))
        tool = extrude(self._profile(op), amount=amount, both=both,
                       taper=max(-80.0, min(80.0, taper)))
        self._apply(_place(tool, op), op)

    def _op_revolve(self, op: dict) -> None:
        axis = AXES.get(str(op.get("axis", "Z") or "Z").upper())
        if axis is None:
            raise FeatureError(f"axis '{op.get('axis')}' is not X, Y or Z")
        arc = max(1e-3, min(360.0, _f(op, "arc", 360.0)))
        self._apply(_place(revolve(self._profile(op), axis, revolution_arc=arc), op), op)

    def _op_sweep(self, op: dict) -> None:
        """A profile driven along a real 3D path — what a volute actually is."""
        path = self._path(op)
        section = self._profile(op)
        tool = sweep(section, path, is_frenet=bool(op.get("frenet", True)))
        self._apply(_place(tool, op), op)

    def _op_loft(self, op: dict) -> None:
        names = [str(n) for n in (op.get("profiles") or [])]
        if len(names) < 2:
            raise FeatureError("loft needs at least two profiles by name")
        sections = []
        for n in names:
            if n not in self.profiles:
                raise FeatureError(f"no profile named '{n}'; defined: {self.order}")
            sections.append(self.profiles[n])
        self._apply(_place(loft(sections, ruled=bool(op.get("ruled", False))), op), op)

    def _op_tube(self, op: dict) -> None:
        """A round section along a path, optionally hollow — hoses, pipes, cable runs."""
        path = self._path(op)
        d = max(_f(op, "d", 8.0), 1e-3)
        wall = max(_f(op, "wall", 0.0), 0.0)
        pts = self._path_points(op)
        start, nxt = pts[0], pts[1]
        direction = tuple(nxt[k] - start[k] for k in range(3))
        plane = Plane(origin=start, z_dir=direction)
        tool = sweep(plane * Circle(d / 2.0), path, is_frenet=True)
        if 0.0 < wall < d / 2.0:
            tool = tool - sweep(plane * Circle(d / 2.0 - wall), path, is_frenet=True)
        self._apply(_place(tool, op), op)

    def _path_points(self, op: dict) -> list[tuple[float, float, float]]:
        pts = _points3d(op.get("path") or [])
        if len(pts) < 2:
            raise FeatureError("a path needs at least two points")
        return pts

    def _op_scroll(self, op: dict) -> None:
        """A swept section that grows along its path — what a volute really is.

        The existing volute in the vocabulary approximates its gas passage with a
        chain of lofted circles subtracted from a revolved shell, which is why it
        renders as a lump. A multi-section sweep gives the same description a smooth
        surface and an area schedule that follows the two diameters stated here.
        """
        pts = self._path_points(op)
        path = self._path(op)
        d0 = max(_f(op, "d_start", _f(op, "d", 20.0)), 1e-3)
        d1 = max(_f(op, "d_end", d0 * 1.6), 1e-3)
        wall = max(_f(op, "wall", 0.0), 0.0)
        tool = self._multisweep(pts, path, d0, d1)
        if 0.0 < wall < min(d0, d1) / 2.0:
            tool = tool - self._multisweep(pts, path, d0 - 2 * wall, d1 - 2 * wall)
        self._apply(_place(tool, op), op)

    @staticmethod
    def _multisweep(pts, path, d0: float, d1: float):
        head = Plane(origin=pts[0], z_dir=_delta(pts[0], pts[1])) * Circle(d0 / 2.0)
        tail = Plane(origin=pts[-1], z_dir=_delta(pts[-2], pts[-1])) * Circle(d1 / 2.0)
        return sweep([head, tail], path, multisection=True)

    def _path(self, op: dict):
        pts = self._path_points(op)
        straight = str(op.get("path_kind", "spline") or "spline").lower() == "line"
        if straight or len(pts) == 2:
            return Curve() + Polyline(*pts)
        return Curve() + Spline(*pts)

    # ── material removal and edge work ──────────────────────────────────
    def _op_hole(self, op: dict) -> None:
        """A bore, optionally counterbored — the most common cut there is."""
        if self.part is None:
            raise FeatureError("a hole needs material to go through")
        d = max(_f(op, "d", 6.0), 1e-3)
        depth = _f(op, "depth", 0.0)
        if depth <= 0.0:
            bb = self.part.bounding_box()
            depth = float(bb.max.Z - bb.min.Z) * 3.0 + 10.0
        axis = str(op.get("dir", "Z") or "Z").upper()
        rot = {"Z": (0.0, 0.0, 0.0), "X": (0.0, 90.0, 0.0), "Y": (-90.0, 0.0, 0.0)}.get(axis)
        if rot is None:
            raise FeatureError(f"dir '{axis}' is not X, Y or Z")
        tool = Cylinder(d / 2.0, depth, align=(Align.CENTER, Align.CENTER, Align.MIN))
        cb_d, cb_h = _f(op, "counterbore_d", 0.0), _f(op, "counterbore_h", 0.0)
        if cb_d > d and cb_h > 0.0:
            tool = tool + Cylinder(cb_d / 2.0, cb_h,
                                   align=(Align.CENTER, Align.CENTER, Align.MIN))
        at = _vec(op, "at")
        tool = Pos(*at) * Rot(*rot) * tool
        self.last, self.last_mode = tool, "cut"
        self._combine(tool, "cut")

    def _op_shell(self, op: dict) -> None:
        """Hollow the solid to a wall thickness, opening whichever faces are named."""
        if self.part is None:
            raise FeatureError("shell needs a solid")
        t = _f(op, "thickness", _f(op, "wall", 2.0))
        if t <= 0.0:
            raise FeatureError("shell thickness must be positive")
        opening = str(op.get("openings", "top") or "top").lower()
        faces = None
        if opening != "none":
            picked = self._faces(opening)
            faces = picked if picked else None
        self.part = offset(self.part, amount=-abs(t), openings=faces, kind=Kind.INTERSECTION)
        self.last, self.last_mode = None, "add"

    def _faces(self, which: str):
        faces = self.part.faces()
        if which in ("top", "z_max"):
            return faces.sort_by(Axis.Z)[-1:]
        if which in ("bottom", "z_min"):
            return faces.sort_by(Axis.Z)[:1]
        if which in ("both", "top_and_bottom"):
            ordered = faces.sort_by(Axis.Z)
            return ordered[:1] + ordered[-1:]
        if which in ("front", "y_min"):
            return faces.sort_by(Axis.Y)[:1]
        if which in ("back", "y_max"):
            return faces.sort_by(Axis.Y)[-1:]
        return faces.sort_by(Axis.Z)[-1:]

    def _op_fillet(self, op: dict) -> None:
        self._edge_work(op, "fillet")

    def _op_chamfer(self, op: dict) -> None:
        self._edge_work(op, "chamfer")

    def _edge_work(self, op: dict, which: str) -> None:
        """Round or cut the selected edges, and survive being asked for the impossible.

        An over-large radius is a common and entirely recoverable mistake: OCCT
        refuses the whole operation, and losing a 40-part assembly because one
        fillet was 0.2 mm too big is a worse outcome than an unfilleted edge and a
        warning that says so.
        """
        if self.part is None:
            raise FeatureError(f"{which} needs a solid")
        size = _f(op, "radius", _f(op, "length", 1.0))
        if size <= 0.0:
            raise FeatureError(f"{which} size must be positive")
        edges = self._select_edges(op.get("select") or {})
        if not edges:
            self.warnings.append(f"{which}: the selector matched no edges")
            return
        try:
            self.part = (fillet(edges, size) if which == "fillet"
                         else chamfer(edges, size))
        except Exception as exc:
            self.warnings.append(f"{which} {size:g} mm on {len(edges)} edge(s) refused "
                                 f"by the kernel ({type(exc).__name__}); left sharp")
        self.last, self.last_mode = None, "add"

    def _select_edges(self, sel: dict):
        kind = str(sel.get("kind", "all") or "all").lower()
        edges = self.part.edges()
        if kind in ("top", "z_max"):
            edges = edges.group_by(Axis.Z)[-1]
        elif kind in ("bottom", "z_min"):
            edges = edges.group_by(Axis.Z)[0]
        elif kind == "vertical":
            edges = edges.filter_by(Axis.Z)
        elif kind == "circle":
            edges = edges.filter_by(GeomType.CIRCLE)
        elif kind == "line":
            edges = edges.filter_by(GeomType.LINE)
        elif kind != "all":
            raise FeatureError(f"select kind '{kind}' is not recognised")
        lo, hi = _f(sel, "min_length", 0.0), _f(sel, "max_length", 0.0)
        picked = [e for e in edges
                  if e.length >= lo and (hi <= 0.0 or e.length <= hi)]
        limit = _i(sel, "max", 0)
        if limit > 0:
            picked = sorted(picked, key=lambda e: -e.length)[:limit]
        return picked

    # ── repetition ──────────────────────────────────────────────────────
    # Every pattern acts on the tool the previous op produced, not on the whole
    # part. That is what makes `hole` then `pattern_circular` mean "six of those
    # holes" — the alternative, patterning the finished solid, would duplicate the
    # part itself, which is never what a bolt circle means.

    def _op_pattern_circular(self, op: dict) -> None:
        if self.last is None:
            raise FeatureError("pattern_circular has nothing to repeat; put it "
                               "immediately after the feature it should copy")
        n = max(2, _i(op, "count", 4))
        axis = str(op.get("axis", "Z") or "Z").upper()
        span = _f(op, "arc", 360.0)
        start = _f(op, "start_angle", 0.0)
        step = span / n if abs(span - 360.0) < 1e-9 else span / max(1, n - 1)
        for i in range(1, n):
            deg = start + step * i
            rot = {"Z": (0.0, 0.0, deg), "X": (deg, 0.0, 0.0), "Y": (0.0, deg, 0.0)}.get(axis)
            if rot is None:
                raise FeatureError(f"axis '{axis}' is not X, Y or Z")
            self._combine(Rot(*rot) * self.last, self.last_mode)
        if start:
            self._combine(Rot(*{"Z": (0.0, 0.0, start), "X": (start, 0.0, 0.0),
                                "Y": (0.0, start, 0.0)}[axis]) * self.last, self.last_mode)

    def _op_pattern_linear(self, op: dict) -> None:
        if self.last is None:
            raise FeatureError("pattern_linear has nothing to repeat")
        n = max(2, _i(op, "count", 2))
        step = _vec(op, "step", (10.0, 0.0, 0.0))
        if not any(step):
            raise FeatureError("pattern_linear needs a non-zero step")
        for i in range(1, n):
            self._combine(Pos(step[0] * i, step[1] * i, step[2] * i) * self.last,
                          self.last_mode)

    def _op_mirror(self, op: dict) -> None:
        plane = _plane_of({"plane": op.get("plane", "YZ"), "offset": op.get("offset", 0.0)})
        target = str(op.get("of", "last") or "last").lower()
        if target == "part":
            if self.part is None:
                raise FeatureError("mirror needs a solid")
            self.part = self.part + mirror(self.part, plane)
            self.last = None
            return
        if self.last is None:
            raise FeatureError("mirror has nothing to repeat")
        self._combine(mirror(self.last, plane), self.last_mode)

    def _op_scale_last(self, op: dict) -> None:
        from build123d import scale

        if self.last is None:
            raise FeatureError("scale_last has nothing to scale")
        factor = _f(op, "by", 1.0)
        if factor <= 0.0:
            raise FeatureError("scale factor must be positive")
        self._combine(_place(scale(self.last, factor), op), self.last_mode)


def build(plan: Sequence[dict]) -> Result:
    """Run a plan and return the solid, or raise `FeatureError` naming the op."""
    return Shape().run(plan)


# ── declared seats ──────────────────────────────────────────────────────────
# A generated shape still has to publish connector frames, or nothing can mate to
# it. Declaring them alongside the plan keeps the two in one place, and resolving
# `"z": "top"` against the finished bounding box means a seat cannot drift out of
# agreement with the geometry when a dimension changes.

SEAT_TYPES = ("face", "at", "polar", "stack")


def seats_to_connectors(seats: Sequence[dict], solid) -> dict:
    """Turn the declarative `seats` list into real `Connector`s on a built solid."""
    from .frames import Connector

    try:
        bb = solid.bounding_box()
        span = {"top": float(bb.max.Z), "bottom": float(bb.min.Z),
                "mid": (float(bb.min.Z) + float(bb.max.Z)) / 2.0,
                "x_max": float(bb.max.X), "x_min": float(bb.min.X),
                "y_max": float(bb.max.Y), "y_min": float(bb.min.Y)}
    except Exception:
        span = {"top": 0.0, "bottom": 0.0, "mid": 0.0}

    def z_of(item: dict, key: str = "z") -> float:
        raw = item.get(key, 0.0)
        if isinstance(raw, str):
            token = raw.strip().lower()
            if token in span:
                return span[token]
            try:
                return float(token)
            except ValueError:
                raise FeatureError(f"seat z '{raw}' is neither a number nor one of "
                                   f"{sorted(span)}") from None
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise FeatureError(f"seat z {raw!r} is not a number") from None

    out: dict[str, Connector] = {}
    for i, item in enumerate(seats or []):
        if not isinstance(item, dict):
            raise FeatureError(f"seats[{i}] is not an object")
        name = str(item.get("name", "") or "")
        if not name:
            raise FeatureError(f"seats[{i}] needs a name")
        kind = str(item.get("type", "face") or "face").lower()
        joint = str(item.get("kind", "rigid") or "rigid").lower()
        doc = str(item.get("doc", "") or "")
        if kind == "face":
            out[name] = Connector.face(name, z_of(item),
                                       outward=bool(item.get("outward", True)),
                                       doc=doc, flip=bool(item.get("flip", True)))
        elif kind == "at":
            out[name] = Connector.at(name, _f(item, "x"), _f(item, "y"), z_of(item),
                                     rx=_f(item, "rx"), ry=_f(item, "ry"),
                                     rz=_f(item, "rz"), kind=joint, doc=doc,
                                     flip=bool(item.get("flip", False)))
        elif kind == "polar":
            radius = _f(item, "radius", _f(item, "d", 0.0) / 2.0)
            out[name] = Connector.polar(name, radius, max(1, _i(item, "count", 1)),
                                        z_of(item), start_angle=_f(item, "start_angle"),
                                        tilt_deg=_f(item, "tilt", _f(item, "tilt_deg")),
                                        kind=joint, doc=doc,
                                        flip=bool(item.get("flip", True)))
        elif kind == "stack":
            values = item.get("z_values") or item.get("values") or []
            try:
                zs = [float(v) for v in values]
            except (TypeError, ValueError):
                raise FeatureError(f"seats[{i}] z_values must be numbers") from None
            if not zs:
                raise FeatureError(f"seats[{i}] stack needs z_values")
            out[name] = Connector.stack(name, zs, kind=joint, doc=doc)
        else:
            raise FeatureError(f"seats[{i}] type '{kind}' is not one of {list(SEAT_TYPES)}")
    return out


def auto_interfaces(solid, declared: dict | None = None) -> dict[str, float]:
    """Sizes read off the built solid, so fit checking works on generated shapes."""
    out: dict[str, float] = {}
    try:
        bb = solid.bounding_box()
        dx, dy, dz = (float(bb.max.X - bb.min.X), float(bb.max.Y - bb.min.Y),
                      float(bb.max.Z - bb.min.Z))
        out.update(od=max(dx, dy), length=dz, width=min(dx, dy), thickness=dz)
    except Exception:
        pass
    for key, val in (declared or {}).items():
        try:
            out[str(key)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def helix_path(r_start: float, r_end: float, wrap_deg: float, z_start: float = 0.0,
               z_end: float = 0.0, n: int = 12, power: float = 1.0) -> list[list[float]]:
    """A growing-radius spiral as plain points — the centreline of a real volute.

    Exposed here so a plan can ask for one by parameters instead of listing twenty
    hand-computed points, which is how the existing volute ended up as a chain of
    lofted circles standing in for a sweep.
    """
    out: list[list[float]] = []
    n = max(3, int(n))
    for i in range(n):
        u = i / (n - 1)
        r = r_start + (r_end - r_start) * u
        th = math.radians(wrap_deg * (u ** max(0.05, power)))
        z = z_start + (z_end - z_start) * (u ** 0.85)
        out.append([r * math.cos(th), r * math.sin(th), z])
    return out






