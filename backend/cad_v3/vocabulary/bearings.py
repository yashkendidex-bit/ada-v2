"""Bearings. Native geometry by default, bd_warehouse when it is installed.

The native deep-groove bearing is a real toroidal-groove race pair with discrete
balls, not a plain ring — it reads correctly in the X-ray render and gives the
clash checker something meaningful to test.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Plane,
    PolarLocations,
    Rectangle,
    Sphere,
    Torus,
    add,
    revolve,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import bd, revolve_profile_solid


@skill(
    "ball_bearing",
    category="bearings",
    connectors=("bore", "front_face", "rear_face", "od_mid", "ball_ring"),
    params=[
        p("bore_d", float, 12.0, min=1, max=500),
        p("od", float, 28.0, min=3, max=900),
        p("width", float, 8.0, min=0.5, max=200),
        p("ball_count", int, 8, min=3, max=60),
        p("ball_d", float, 0.0, min=0, max=120, doc="0 = auto 0.32 x (od - bore)"),
        p("contact_angle", float, 0.0, min=0, max=40, doc=">0 makes it angular-contact"),
        p("shielded", bool, False, doc="add a shield disc on both faces"),
    ],
)
def ball_bearing(**kw) -> SkillResult:
    """Deep-groove or angular-contact ball bearing with discrete balls."""
    bore_d, od, w = kw["bore_d"], kw["od"], kw["width"]
    ball_d = kw["ball_d"] or max((od - bore_d) * 0.32, 0.8)
    pcd = (bore_d + od) / 2.0
    z_mid = w / 2.0

    inner_od = pcd - ball_d * 0.42
    outer_id = pcd + ball_d * 0.42

    with BuildPart() as bp:
        # Inner race with its toroidal groove.
        add(revolve_profile_solid([[inner_od, 0], [inner_od, w]], bore_d=bore_d))
        with Locations((0, 0, z_mid)):
            Torus(major_radius=pcd / 2, minor_radius=ball_d / 2 * 1.02, mode=Mode.SUBTRACT)
        # Outer race with its groove.
        add(revolve_profile_solid([[od, 0], [od, w]], bore_d=outer_id))
        with Locations((0, 0, z_mid)):
            Torus(major_radius=pcd / 2, minor_radius=ball_d / 2 * 1.02, mode=Mode.SUBTRACT)
        # Balls.
        with Locations((0, 0, z_mid)):
            with PolarLocations(pcd / 2, kw["ball_count"]):
                Sphere(radius=ball_d / 2)
        if kw["shielded"]:
            # The profile z values are already absolute, so these must not be wrapped in
            # a Locations() as well -- doing so put the front shield a whole 'w - 0.4'
            # clear of the bearing, floating in space. The shield overlaps the outer race
            # land by 0.2 so it fuses to it, and clears the inner race the way a real
            # non-contact shield does.
            for z in (0.0, w - 0.4):
                add(revolve_profile_solid([[outer_id + 0.4, z], [outer_id + 0.4, z + 0.4]],
                                          bore_d=inner_od + 0.4))

    res = SkillResult(
        bp.part,
        interfaces={"bore_d": bore_d, "od": od, "width": w, "ball_d": ball_d},
        meta={"source": "ada_native", "bd_available": bool(bd("bearing"))},
    )
    res.add(
        Connector.at("bore", z=0.0, doc="press onto the shaft here"),
        Connector.face("front_face", w),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("od_mid", x=od / 2, z=z_mid, ry=90.0, doc="outer race seat in the housing"),
        Connector.polar("ball_ring", pcd / 2, kw["ball_count"], z_mid, kind="ball"),
    )
    return res


@skill(
    "journal_bearing",
    category="bearings",
    connectors=("bore", "front_face", "rear_face", "od_mid", "oil_port"),
    params=[
        p("bore_d", float, 12.0, min=1, max=500),
        p("od", float, 20.0, min=2, max=800),
        p("length", float, 16.0, min=1, max=400),
        p("oil_hole_count", int, 2, min=0, max=12),
        p("oil_hole_d", float, 2.4, min=0.3, max=30),
        p("floating", bool, True, doc="full-floating bush (turbo style)"),
        p("flange_od", float, 0.0, min=0, max=800, doc=">0 makes it a flanged bush"),
    ],
)
def journal_bearing(**kw) -> SkillResult:
    """Plain / full-floating journal bush with oil feed holes."""
    L, od = kw["length"], kw["od"]
    steps = [[od, 0.0], [od, L]]
    if kw["flange_od"] > od:
        steps = [[kw["flange_od"], 0.0], [kw["flange_od"], L * 0.16], [od, L * 0.16], [od, L]]

    with BuildPart() as bp:
        add(revolve_profile_solid(steps, bore_d=kw["bore_d"]))
        if kw["oil_hole_count"] > 0:
            with Locations((0, 0, L / 2)):
                with PolarLocations(0, kw["oil_hole_count"]):
                    Cylinder(radius=kw["oil_hole_d"] / 2, height=od * 1.4,
                             rotation=(0, 90, 0), mode=Mode.SUBTRACT)
            # Circumferential oil-distribution groove.
            with BuildSketch(Plane.XZ):
                with Locations((od / 2 - 0.5, L / 2)):
                    Rectangle(1.2, kw["oil_hole_d"] * 1.4)
            revolve(axis=Axis.Z, mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={"bore_d": kw["bore_d"], "od": od, "length": L})
    res.add(
        Connector.at("bore", z=0.0),
        Connector.face("front_face", L),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("od_mid", x=od / 2, z=L / 2, ry=90.0),
    )
    if kw["oil_hole_count"] > 0:
        res.add(Connector.polar("oil_port", od / 2, kw["oil_hole_count"], L / 2, tilt_deg=90.0))
    return res
