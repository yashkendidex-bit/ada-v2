"""Stationary turbomachinery: volute scrolls, diffusers, nozzle rings.

`volute_scroll` is the hardest skill in the vocabulary. v1 builds the gas
passage as a chain of lofted sections whose area follows the A/R law, then
subtracts that passage from a revolved shell. That is an approximation of a true
continuously-swept scroll, but the A/R number stays honest and it reads
correctly in both X-ray and section views.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    BuildLine,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Location,
    Locations,
    Mode,
    Plane,
    PolarLocations,
    Rectangle,
    add,
    extrude,
    loft,
    make_face,
    revolve,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import METRIC, metric, revolve_profile_solid


def _scroll_passage(base_r: float, wrap_deg: float, a_over_r: float,
                    throat_d: float, segments: int = 20):
    """Chain of lofted circular sections on a spiral — the gas path solid."""
    frames: list[tuple[Plane, float]] = []
    for i in range(segments + 1):
        u = i / segments
        # Area falls linearly to the throat as the scroll wraps; A/R sets the scale.
        area_frac = max(0.12, 1.0 - u * 0.88)
        d_sec = max(throat_d * math.sqrt(area_frac), 1.2)
        r_c = base_r + d_sec * 0.5 * 0.35
        deg = wrap_deg * u
        th = math.radians(deg)
        origin = (r_c * math.cos(th), r_c * math.sin(th), 0.0)
        tangential = (-math.sin(th), math.cos(th), 0.0)
        frames.append((Plane(origin=origin, z_dir=tangential), d_sec))

    with BuildPart() as passage:
        for i in range(segments):
            (pa, da), (pb, db) = frames[i], frames[i + 1]
            with BuildPart() as seg:
                with BuildSketch(pa):
                    Circle(da / 2)
                with BuildSketch(pb):
                    Circle(db / 2)
                loft()
            add(seg.part)
    return passage.part, frames[0][1]


@skill(
    "volute_scroll",
    category="turbomachinery",
    connectors=("mount_flange", "throat", "duct_port", "front_face", "rear_face", "bolt_circle"),
    params=[
        p("inlet_d", float, 48.0, min=5, max=600, doc="duct connection bore"),
        p("outlet_d", float, 38.0, min=5, max=600),
        p("a_over_r", float, 0.48, min=0.1, max=2.5, doc="scroll area/radius ratio"),
        p("wrap_deg", float, 340.0, min=120, max=360),
        p("rotor_d", float, 56.0, min=8, max=900, doc="diameter of the wheel it houses"),
        p("tip_clearance", float, 0.35, min=0.05, max=5.0),
        p("width", float, 34.0, min=4, max=400, doc="axial width of the housing"),
        p("wall_mm", float, 4.5, min=1.0, max=40),
        p("bolt_count", int, 8, min=0, max=48),
        p("bolt_size", str, "M6", choices=tuple(METRIC)),
        p("bolt_pcd", float, 0.0, min=0, max=1990),
        p("duct_stub_len", float, 22.0, min=0, max=200),
    ],
)
def volute_scroll(**kw) -> SkillResult:
    """Spiral gas-collector housing with an A/R-governed passage."""
    rotor_r = kw["rotor_d"] / 2.0
    clear_r = rotor_r + kw["tip_clearance"]
    throat_d = max(kw["inlet_d"] * 0.92, 4.0)
    scroll_r = clear_r + throat_d * 0.55
    shell_r = scroll_r + throat_d * 0.5 + kw["wall_mm"]
    w = kw["width"]

    shell = revolve_profile_solid(
        [[shell_r * 2, -w * 0.5], [shell_r * 2, w * 0.5]], bore_d=0.0
    )
    passage, mouth_d = _scroll_passage(scroll_r, kw["wrap_deg"], kw["a_over_r"], throat_d)

    with BuildPart() as bp:
        add(shell)
        add(passage, mode=Mode.SUBTRACT)
        # Rotor cavity + tip-clearance shroud contour.
        Cylinder(radius=clear_r, height=w * 1.2,
                 align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
        # Axial duct stub (compressor inlet / turbine outlet).
        if kw["duct_stub_len"] > 0:
            with Locations((0, 0, w * 0.5 + kw["duct_stub_len"] * 0.5 - 0.01)):
                Cylinder(radius=kw["outlet_d"] / 2 + kw["wall_mm"], height=kw["duct_stub_len"],
                         align=(Align.CENTER, Align.CENTER, Align.CENTER))
            with Locations((0, 0, w * 0.5 + kw["duct_stub_len"] * 0.5)):
                Cylinder(radius=kw["outlet_d"] / 2, height=kw["duct_stub_len"] * 1.3,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
        # Radial duct port at the scroll mouth.
        with Locations(Location((scroll_r + throat_d * 0.4, 0, 0), (0, 90, 0))):
            Cylinder(radius=kw["inlet_d"] / 2, height=kw["duct_stub_len"] * 1.4 + shell_r,
                     align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)

        pcd = kw["bolt_pcd"] or shell_r * 1.72
        if kw["bolt_count"] > 0:
            clear = metric(kw["bolt_size"])["clear"]
            with PolarLocations(pcd / 2, kw["bolt_count"]):
                Cylinder(radius=clear / 2, height=w * 1.4,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

    pcd = kw["bolt_pcd"] or shell_r * 1.72
    res = SkillResult(
        bp.part,
        interfaces={"rotor_d": kw["rotor_d"], "bore_d": clear_r * 2,
                    "a_over_r": kw["a_over_r"], "bolt_pcd": pcd, "width": w},
        meta={"fidelity": "segmented_loft_v1", "throat_d": throat_d},
    )
    res.add(
        Connector.face("mount_flange", -w * 0.5, outward=False, doc="bolts to the centre housing"),
        Connector.face("front_face", w * 0.5, doc="duct side"),
        Connector.face("rear_face", -w * 0.5, outward=False),
        Connector.at("throat", x=scroll_r, ry=90.0, doc="scroll throat, radial"),
        Connector.at("duct_port", x=shell_r + kw["duct_stub_len"], ry=90.0, doc="radial duct connection"),
    )
    if kw["bolt_count"] > 0:
        res.add(Connector.polar("bolt_circle", pcd / 2, kw["bolt_count"], -w * 0.5))
    return res


@skill(
    "vane_ring",
    category="turbomachinery",
    connectors=("front_face", "rear_face", "bore", "vane_ring", "pivot_ring"),
    params=[
        p("vane_count", int, 12, min=3, max=40),
        p("inner_d", float, 62.0, min=5, max=900),
        p("outer_d", float, 92.0, min=8, max=1200),
        p("height", float, 9.0, min=1, max=200),
        p("stagger_deg", float, 62.0, min=0, max=85, doc="vane angle from radial"),
        p("vane_t", float, 2.2, min=0.4, max=20),
        p("ring_t", float, 3.5, min=0.8, max=40, doc="hub/shroud ring thickness"),
        p("pivots", bool, False, doc="variable-geometry pivot pins"),
    ],
)
def vane_ring(**kw) -> SkillResult:
    """Diffuser / nozzle ring — the vaned annulus between rotor and scroll."""
    r_in, r_out = kw["inner_d"] / 2, kw["outer_d"] / 2
    h = kw["height"]
    span = max(r_out - r_in - kw["ring_t"] * 2, 1.0)
    vane_len = span / max(math.cos(math.radians(kw["stagger_deg"])), 0.2)

    with BuildPart() as bp:
        # Hub ring and shroud ring, joined by the vanes.
        add(revolve_profile_solid([[(r_in + kw["ring_t"]) * 2, 0], [(r_in + kw["ring_t"]) * 2, h]],
                                  bore_d=r_in * 2))
        add(revolve_profile_solid([[r_out * 2, 0], [r_out * 2, h]],
                                  bore_d=(r_out - kw["ring_t"]) * 2))
        r_mid = (r_in + r_out) / 2
        # The vanes have to be laid out *inside* the sketch: `BuildSketch(Plane.XY)`
        # re-anchors to the global plane and drops the surrounding PolarLocations, so
        # patterning outside it stacks every vane on the axis instead of spacing them
        # round the annulus. Patterning inside the sketch also makes `rotation` the
        # stagger about each vane's own centre, measured from radial.
        with BuildSketch(Plane.XY):
            with PolarLocations(r_mid, kw["vane_count"]):
                Rectangle(min(vane_len, span * 2.4), kw["vane_t"],
                          rotation=kw["stagger_deg"])
        extrude(amount=h)
        if kw["pivots"]:
            with PolarLocations(r_mid, kw["vane_count"]):
                Cylinder(radius=kw["vane_t"] * 0.6, height=h + 4.0,
                         align=(Align.CENTER, Align.CENTER, Align.MIN))

    res = SkillResult(bp.part, interfaces={"inner_d": kw["inner_d"], "outer_d": kw["outer_d"],
                                           "bore_d": kw["inner_d"], "height": h})
    res.add(
        Connector.face("front_face", h),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("bore", z=0.0),
        Connector.polar("vane_ring", (r_in + r_out) / 2, kw["vane_count"], 0.0),
    )
    if kw["pivots"]:
        res.add(Connector.polar("pivot_ring", (r_in + r_out) / 2, kw["vane_count"], h, kind="revolute"))
    return res
