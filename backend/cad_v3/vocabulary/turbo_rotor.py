"""Rotating turbomachinery — the parts bd_warehouse does not have.

`impeller` and `turbine_wheel` are the reason cad_v2 could never build the
reference images: both are swept 3D splines with real backsweep and twist, and
neither is expressible as boxes-and-cylinders at any level of prompt cleverness.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    BuildPart,
    Cylinder,
    Mode,
    PolarLocations,
    add,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import blade_sweep, revolve_profile_solid, spiral_path


@skill(
    "impeller",
    category="turbomachinery",
    connectors=("bore", "nose", "hub_face", "inlet_eye", "tip_shroud"),
    params=[
        p("blade_count", int, 7, min=3, max=32, doc="full blades"),
        p("splitter_count", int, 0, min=0, max=32, doc="shorter intermediate blades"),
        p("hub_d", float, 14.0, min=3, max=400),
        p("tip_d", float, 56.0, min=8, max=900, doc="exducer / outer diameter"),
        p("height", float, 17.0, min=2, max=400),
        p("backsweep_deg", float, 40.0, min=0, max=80, doc="blade wrap at the tip"),
        p("blade_t", float, 1.5, min=0.3, max=20),
        p("chord", float, 0.0, min=0, max=200, doc="0 = auto from height"),
        p("bore_d", float, 9.0, min=0, max=200),
        p("nose_d", float, 0.0, min=0, max=200, doc="0 = auto nut boss"),
    ],
)
def impeller(**kw) -> SkillResult:
    """Backswept centrifugal compressor wheel with real 3D blade twist."""
    hub_r = kw["hub_d"] / 2.0
    tip_r = kw["tip_d"] / 2.0
    h = kw["height"]
    chord = kw["chord"] or max(h * 0.75, 3.0)
    nose_d = kw["nose_d"] or max(kw["bore_d"] * 1.6, kw["hub_d"] * 0.55)

    # Ogive hub: real compressor wheels are not cylinders.
    hub_steps = [
        [kw["hub_d"] * 1.28, 0.0],
        [kw["hub_d"] * 1.28, h * 0.10],
        [kw["hub_d"] * 1.08, h * 0.46],
        [kw["hub_d"] * 0.74, h * 0.78],
        [nose_d, h * 0.94],
        [nose_d, h],
    ]
    hub = revolve_profile_solid(hub_steps, bore_d=0.0)

    def make_blade(span_start: float, span_end: float, chord_scale: float):
        pts = spiral_path(
            r_start=hub_r * 1.18 + (tip_r - hub_r * 1.18) * span_start,
            r_end=hub_r * 1.18 + (tip_r - hub_r * 1.18) * span_end,
            wrap_deg=kw["backsweep_deg"] * (span_end - span_start),
            z_start=h * (0.10 + 0.72 * span_start),
            z_end=h * (0.10 + 0.72 * span_end),
            n=8,
            power=1.35,
        )
        return blade_sweep(pts, chord * chord_scale, kw["blade_t"])

    with BuildPart() as bp:
        add(hub)
        with PolarLocations(0, kw["blade_count"]):
            add(make_blade(0.0, 1.0, 1.0))
        if kw["splitter_count"] > 0:
            offset = 180.0 / max(1, kw["splitter_count"])
            with PolarLocations(0, kw["splitter_count"], start_angle=offset):
                add(make_blade(0.42, 1.0, 0.82))
        if kw["bore_d"] > 0:
            Cylinder(radius=kw["bore_d"] / 2, height=h * 3,
                     align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={"bore_d": kw["bore_d"], "tip_d": kw["tip_d"], "height": h})
    res.add(
        Connector.at("bore", z=0.0, doc="shaft seat, blades run +Z"),
        Connector.face("nose", h, doc="retaining-nut face"),
        Connector.face("hub_face", 0.0, outward=False, doc="back face against the thrust collar"),
        Connector.face("inlet_eye", h, doc="air inlet plane, normal out of the eye"),
        Connector.at("tip_shroud", x=tip_r, z=h * 0.55, ry=90.0, doc="tip clearance reference"),
    )
    return res


@skill(
    "turbine_wheel",
    category="turbomachinery",
    connectors=("bore", "hub_face", "nose", "tip_ring", "exducer_face"),
    params=[
        p("blade_count", int, 11, min=5, max=40),
        p("tip_d", float, 53.0, min=8, max=900, doc="inducer diameter at the scroll"),
        p("hub_d", float, 16.0, min=3, max=400),
        p("height", float, 22.0, min=3, max=400),
        p("exducer_angle", float, 32.0, min=0, max=75, doc="blade turn at the outlet"),
        p("blade_t", float, 1.9, min=0.4, max=20),
        p("bore_d", float, 0.0, min=0, max=200, doc="0 = friction-welded to the shaft"),
    ],
)
def turbine_wheel(**kw) -> SkillResult:
    """Radial-inflow turbine wheel: gas enters at the rim, exits axially."""
    tip_r = kw["tip_d"] / 2.0
    hub_r = kw["hub_d"] / 2.0
    h = kw["height"]

    hub_steps = [
        [kw["hub_d"], 0.0],
        [kw["tip_d"] * 0.42, h * 0.22],
        [kw["tip_d"] * 0.5, h * 0.55],
        [kw["tip_d"] * 0.34, h * 0.86],
        [kw["hub_d"] * 0.7, h],
    ]
    hub = revolve_profile_solid(hub_steps, bore_d=0.0)

    # Flow turns from radial (at the rim, z high) to axial (at the hub, z low).
    pts = spiral_path(
        r_start=tip_r * 0.98,
        r_end=hub_r * 1.5,
        wrap_deg=kw["exducer_angle"],
        z_start=h * 0.80,
        z_end=h * 0.06,
        n=8,
        power=1.15,
    )
    blade = blade_sweep(pts, max(h * 0.42, 3.0), kw["blade_t"])

    with BuildPart() as bp:
        add(hub)
        with PolarLocations(0, kw["blade_count"]):
            add(blade)
        if kw["bore_d"] > 0:
            Cylinder(radius=kw["bore_d"] / 2, height=h * 3,
                     align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={"tip_d": kw["tip_d"], "bore_d": kw["bore_d"], "height": h})
    res.add(
        Connector.face("hub_face", h, doc="shaft/weld face, +Z"),
        Connector.at("bore", z=h, doc="shaft seat at the hub back face; wheel hangs -Z"),
        Connector.face("nose", 0.0, outward=False, doc="exducer nose"),
        Connector.at("tip_ring", x=tip_r, z=h * 0.8, ry=90.0, doc="inducer rim at the scroll throat"),
        Connector.face("exducer_face", h * 0.06, outward=False, doc="axial gas outlet plane"),
    )
    return res
