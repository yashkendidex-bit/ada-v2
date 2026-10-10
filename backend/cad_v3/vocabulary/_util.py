"""Shared geometry helpers for the vocabulary. Pure build123d, no LLM.

`revolve_profile_solid` and `blade_sweep` are the two workhorses — between them
they cover housings, hubs, shafts, volute sections, impellers and turbine
wheels. Both patterns were validated by the capability spike before any skill
was written on top of them.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from build123d import (
    Align,
    Axis,
    BuildLine,
    BuildPart,
    BuildSketch,
    Cylinder,
    Ellipse,
    Location,
    Mode,
    Plane,
    Polyline,
    Spline,
    Vector,
    make_face,
    revolve,
    sweep,
)

# ISO metric fastener data: nominal d, coarse pitch, socket head, hex AF, washer.
METRIC: dict[str, dict[str, float]] = {
    "M2": dict(d=2.0, pitch=0.4, head_d=3.8, head_h=2.0, hex_af=1.5, nut_af=4.0, nut_h=1.6, wash_od=5.0, wash_t=0.3, clear=2.4),
    "M3": dict(d=3.0, pitch=0.5, head_d=5.5, head_h=3.0, hex_af=2.5, nut_af=5.5, nut_h=2.4, wash_od=7.0, wash_t=0.5, clear=3.4),
    "M4": dict(d=4.0, pitch=0.7, head_d=7.0, head_h=4.0, hex_af=3.0, nut_af=7.0, nut_h=3.2, wash_od=9.0, wash_t=0.8, clear=4.5),
    "M5": dict(d=5.0, pitch=0.8, head_d=8.5, head_h=5.0, hex_af=4.0, nut_af=8.0, nut_h=4.7, wash_od=10.0, wash_t=1.0, clear=5.5),
    "M6": dict(d=6.0, pitch=1.0, head_d=10.0, head_h=6.0, hex_af=5.0, nut_af=10.0, nut_h=5.2, wash_od=12.0, wash_t=1.6, clear=6.6),
    "M8": dict(d=8.0, pitch=1.25, head_d=13.0, head_h=8.0, hex_af=6.0, nut_af=13.0, nut_h=6.8, wash_od=16.0, wash_t=1.6, clear=9.0),
    "M10": dict(d=10.0, pitch=1.5, head_d=16.0, head_h=10.0, hex_af=8.0, nut_af=17.0, nut_h=8.4, wash_od=20.0, wash_t=2.0, clear=11.0),
    "M12": dict(d=12.0, pitch=1.75, head_d=18.0, head_h=12.0, hex_af=10.0, nut_af=19.0, nut_h=10.8, wash_od=24.0, wash_t=2.5, clear=13.5),
}

SIZES = tuple(METRIC)


def metric(size: str) -> dict[str, float]:
    return METRIC.get(str(size).upper().split("-")[0].split("X")[0], METRIC["M6"])


def bd(module: str) -> Any | None:
    """Import a bd_warehouse module if installed, else None.

    Every bd-backed skill has a native fallback, so the vocabulary is fully
    functional without bd_warehouse and simply gets more standards-accurate
    fasteners/bearings when it is present.
    """
    try:
        return __import__(f"bd_warehouse.{module}", fromlist=["*"])
    except Exception:
        return None


def dedupe(points: Sequence[tuple[float, float]], eps: float = 1e-7) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for pt in points:
        if not out or abs(pt[0] - out[-1][0]) > eps or abs(pt[1] - out[-1][1]) > eps:
            out.append((float(pt[0]), float(pt[1])))
    if len(out) > 2 and abs(out[0][0] - out[-1][0]) < eps and abs(out[0][1] - out[-1][1]) < eps:
        out.pop()
    return out


def revolve_profile_solid(steps: Sequence[Sequence[float]], bore_d: float = 0.0, arc: float = 360.0):
    """Revolve a [[diameter, z], ...] outline about Z. The keystone operation.

    `steps` is the outer contour bottom-to-top. The profile is closed down the
    bore (or the axis when bore_d is 0) automatically, so a caller describes
    only the shape they care about.
    """
    pts = [(max(float(d) / 2.0, 0.0), float(z)) for d, z in steps]
    if len(pts) < 2:
        raise ValueError("revolve_profile_solid needs >= 2 [diameter, z] steps")

    z_lo, z_hi = pts[0][1], pts[-1][1]
    r_in = max(float(bore_d) / 2.0, 0.0)
    outline = list(pts)
    if r_in > 0.02:
        outline += [(r_in, z_hi), (r_in, z_lo)]
    else:
        if pts[-1][0] > 1e-6:
            outline.append((0.0, z_hi))
        if pts[0][0] > 1e-6:
            outline.append((0.0, z_lo))

    outline = dedupe(outline)
    if len(outline) < 3:
        raise ValueError("degenerate revolve profile")

    with BuildPart() as bp:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                Polyline(*outline, close=True)
            make_face()
        revolve(axis=Axis.Z, revolution_arc=arc)
    return bp.part


def blade_sweep(path_pts: Sequence[Sequence[float]], chord: float, thick: float):
    """Sweep an aerofoil-ish section along a 3D spline. Genuine twist, no fakery."""
    pts = [tuple(float(v) for v in q) for q in path_pts]
    if len(pts) < 3:
        raise ValueError("blade_sweep needs >= 3 path points")
    tangent = Vector(*pts[1]) - Vector(*pts[0])
    with BuildPart() as bp:
        with BuildLine():
            Spline(*pts)
        with BuildSketch(Plane(origin=pts[0], z_dir=tangent)):
            Ellipse(max(chord, 0.4) / 2.0, max(thick, 0.2) / 2.0)
        sweep(is_frenet=True)
    return bp.part


def spiral_path(
    r_start: float,
    r_end: float,
    wrap_deg: float,
    z_start: float = 0.0,
    z_end: float = 0.0,
    n: int = 9,
    power: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Points on a growing-radius helical spiral — impeller and volute centrelines."""
    out: list[tuple[float, float, float]] = []
    for i in range(max(3, n)):
        u = i / (max(3, n) - 1)
        r = r_start + (r_end - r_start) * u
        th = math.radians(wrap_deg * (u ** power))
        z = z_start + (z_end - z_start) * (u ** 0.85)
        out.append((r * math.cos(th), r * math.sin(th), z))
    return out


def axial_bore(radius: float, height: float, z_center: float = 0.0):
    """A generous subtractive bore — always overshoots so the cut is clean."""
    return Cylinder(
        radius=max(radius, 0.05),
        height=max(height, 0.1),
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
        mode=Mode.SUBTRACT,
    )
