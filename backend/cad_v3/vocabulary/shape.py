"""Open-ended shapes: the sketch-and-feature skill, and the two it makes easy.

Every other module here publishes a *specific* part — a flange, a spur gear, a
labyrinth seal — with named parameters and a fixed topology. That is the right
shape for a catalogue and the wrong shape for a machine that also contains a cast
housing with three mounting lugs, a formed bracket, and eleven hoses.

`shape` closes that gap without adding one function per shape. It takes a plan (see
features.py) and a list of seats, so a new part is data in the spec rather than new
code in the repository — and because the plan cannot express a position, the
coordinate-free guarantee is untouched: where the part goes is still decided by
mates and datums alone.

`hose` and `volute` are the two cases common enough to be worth naming. A hose is a
swept tube along a spline, which is what every flexible line in a real engine bay
is. A volute is a swept scroll with a growing radius, which is what a turbocharger
housing's gas passage actually is — as opposed to a chain of lofted circles
subtracted from a revolved shell, which is what stands in for one today.
"""

from __future__ import annotations

from .. import features
from ..frames import Connector, SkillResult
from ..registry import p, skill

# What `shape` builds when nobody says otherwise. It exists so the geometry gate and
# the catalogue probe can both exercise the skill: a skill that cannot be built at
# its defaults publishes no seats to the brain and cannot be gated.
DEFAULT_PLAN = [
    {"op": "box", "w": 40.0, "d": 40.0, "h": 20.0, "align": "bottom"},
    {"op": "fillet", "radius": 2.0, "select": {"kind": "vertical"}},
]
DEFAULT_SEATS = [
    {"name": "top_face", "type": "face", "z": "top"},
    {"name": "bottom_face", "type": "face", "z": "bottom", "outward": False},
]


@skill(
    "shape",
    category="general",
    connectors=("top_face", "bottom_face"),
    params=[
        p("plan", list, None, doc="ordered feature operations; see the op list below"),
        p("seats", list, None, doc="connector frames this part publishes"),
        p("interfaces", list, None, doc="[[name, value], ...] sizes for fit checking"),
        p("label", str, "", doc="what this shape is, for the report and the drawing"),
    ],
)
def shape(**kw) -> SkillResult:
    """Any solid describable as sketches and features — the open end of the vocabulary.

    ops: box, cylinder, cone, sphere, torus, wedge, profile, extrude, revolve,
    sweep, loft, tube, hole, shell, fillet, chamfer, pattern_circular,
    pattern_linear, mirror, scale_last.
    profile shapes: rect, circle, ellipse, polygon, slot, regular, spline.
    seat types: face, at, polar, stack. A seat z may be "top", "bottom" or "mid".
    """
    plan = kw.get("plan") or DEFAULT_PLAN
    result = features.build(plan)
    seats = kw.get("seats") or DEFAULT_SEATS
    connectors = features.seats_to_connectors(seats, result.solid)
    declared = _pairs(kw.get("interfaces"))
    out = SkillResult(
        solid=result.solid,
        connectors=connectors,
        interfaces=features.auto_interfaces(result.solid, declared),
        meta={"source": "shape", "ops": result.ops, "label": kw.get("label", ""),
              "warnings": result.warnings},
    )
    return out


def _pairs(raw) -> dict[str, float]:
    """`[["bore_d", 12], ...]` — a list of pairs, because params carry no dicts."""
    out: dict[str, float] = {}
    for item in raw or []:
        try:
            out[str(item[0])] = float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
    return out


@skill(
    "hose",
    category="general",
    connectors=("end_a", "end_b", "mid"),
    params=[
        p("path", list, None, doc="[[x, y, z], ...] centreline, >= 2 points, part frame"),
        p("d", float, 12.0, min=0.5, max=400, doc="outside diameter"),
        p("wall", float, 2.0, min=0, max=60, doc="0 makes it solid, e.g. a cable"),
        p("straight", bool, False, doc="polyline through the points instead of a spline"),
    ],
)
def hose(**kw) -> SkillResult:
    """A flexible line swept along its own centreline — hoses, pipes, cable runs."""
    path = [list(map(float, q)) for q in (kw.get("path") or
                                         [[0.0, 0.0, 0.0], [0.0, 0.0, 60.0]])]
    if len(path) < 2:
        raise ValueError("a hose needs at least two path points")
    plan = [{"op": "tube", "path": path, "d": kw["d"], "wall": kw["wall"],
             "path_kind": "line" if kw["straight"] else "spline"}]
    result = features.build(plan)
    connectors = {
        # A line's ends face *out* of the material, which is along the tangent at the
        # last point and against it at the first. Declaring both as +Z — which is what
        # a plain `Connector.at` gives — aims one of them into the hose, so anything
        # mated there interpenetrates instead of butting up.
        "end_a": _aimed("end_a", path[0], _sub(path[0], path[1]),
                        "the end the line starts at"),
        "end_b": _aimed("end_b", path[-1], _sub(path[-1], path[-2]),
                        "the end the line finishes at"),
        "mid": _aimed("mid", path[len(path) // 2],
                      _sub(path[min(len(path) - 1, len(path) // 2 + 1)],
                           path[len(path) // 2]), "a clamp point", flip=False),
    }
    return SkillResult(result.solid, connectors,
                       interfaces={"od": float(kw["d"]),
                                   "bore_d": max(0.0, float(kw["d"]) - 2 * float(kw["wall"]))},
                       meta={"source": "shape", "warnings": result.warnings})


def _sub(a, b) -> tuple[float, float, float]:
    """a - b, i.e. the direction pointing away from b and out through a."""
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]),
            float(a[2]) - float(b[2]))


def _aimed(name: str, origin, z_dir, doc: str, *, flip: bool = True) -> Connector:
    """A seat whose +Z is an arbitrary direction, not an Euler triple.

    Every seat in the vocabulary that sits on a swept path needs this: the outward
    normal is whatever the path's tangent happens to be, and rounding it to a
    rotation about X, Y and Z is how a hose end ends up facing into its own bore.
    """
    from build123d import Plane

    if not any(abs(float(v)) > 1e-9 for v in z_dir):
        z_dir = (0.0, 0.0, 1.0)
    loc = Plane(origin=tuple(float(v) for v in origin),
                z_dir=tuple(float(v) for v in z_dir)).location
    return Connector(name, [loc], "rigid", (0.0, 0.0, 1.0), doc, flip)


@skill(
    "volute",
    category="general",
    connectors=("inlet", "outlet", "mount_flange", "bore", "axis"),
    params=[
        p("throat_d", float, 26.0, min=2, max=600, doc="passage diameter at the tongue"),
        p("outlet_d", float, 44.0, min=2, max=800, doc="passage diameter at the flange"),
        p("r_inner", float, 34.0, min=2, max=1200, doc="scroll radius at the tongue"),
        p("r_outer", float, 62.0, min=3, max=2000, doc="scroll radius at the outlet"),
        p("wrap_deg", float, 330.0, min=90, max=360),
        p("z_rise", float, 6.0, min=-200, max=200, doc="axial drift over the wrap"),
        p("wall", float, 4.0, min=0.8, max=40),
        p("segments", int, 16, min=6, max=60, doc="points on the swept centreline"),
        p("flange_od", float, 0.0, min=0, max=2000,
          doc="0 = auto; the mounting face that meets the bearing housing"),
        p("flange_t", float, 6.0, min=0, max=80),
        p("bore_d", float, 30.0, min=0, max=1000, doc="rotor clearance through the flange"),
    ],
)
def volute(**kw) -> SkillResult:
    """A scroll passage swept along a growing-radius spiral — a real volute.

    The passage is a genuine multi-section sweep, so its area grows smoothly from
    throat to outlet the way the two diameters describe. The alternative the
    vocabulary shipped with — a chain of circles lofted together and subtracted from
    a revolved shell — has neither a smooth surface nor a meaningful area schedule,
    and renders as a lump.
    """
    path = features.helix_path(kw["r_inner"], kw["r_outer"], kw["wrap_deg"],
                               0.0, kw["z_rise"], n=int(kw["segments"]))
    wall = float(kw["wall"])
    flange_t = float(kw["flange_t"])
    flange_od = float(kw["flange_od"]) or (float(kw["r_inner"]) * 2.0
                                           + float(kw["throat_d"]) + 2 * wall)
    bore_d = min(float(kw["bore_d"]), flange_od - 4.0)
    plan: list[dict] = [{"op": "scroll", "path": path,
                         "d_start": float(kw["throat_d"]) + 2 * wall,
                         "d_end": float(kw["outlet_d"]) + 2 * wall, "wall": wall}]
    if flange_t > 0.0:
        # A mounting face has to exist as geometry before a seat may claim one. The
        # flange sits below the scroll so its outer face is a real planar -Z face.
        plan.append({"op": "cylinder", "d": flange_od, "h": flange_t,
                     "align": "bottom", "at": [0.0, 0.0, -flange_t]})
    if bore_d > 0.0:
        plan.append({"op": "hole", "d": bore_d, "depth": flange_t + 4.0,
                     "at": [0.0, 0.0, -flange_t - 2.0]})
    result = features.build(plan)
    connectors = {
        "inlet": _aimed("inlet", path[0], _sub(path[0], path[1]),
                        "the tongue end of the scroll"),
        "outlet": _aimed("outlet", path[-1], _sub(path[-1], path[-2]),
                         "the discharge, where the outlet flange bolts on"),
        "mount_flange": Connector.face("mount_flange", -flange_t, outward=False,
                                       doc="the face that meets the bearing housing"),
        "bore": Connector.at("bore", 0.0, 0.0, -flange_t,
                             doc="the rotor passage through the mounting flange"),
        "axis": Connector.at("axis", 0.0, 0.0, 0.0, doc="the rotor centreline"),
    }
    return SkillResult(result.solid, connectors,
                       interfaces={"od": flange_od, "bore_d": max(0.0, bore_d),
                                   "length": abs(float(kw["z_rise"])) + flange_t},
                       meta={"source": "shape", "warnings": result.warnings})
