"""Structural vocabulary: flanges, plates, brackets, heat shields, ribs.

`flange` is the skill that makes reference image 1 reachable — bolt circle, raised
face, gasket groove, and a published `bolt_circle` connector with one seat per
hole, so the spec says "12 x M6" once and the assembler places 12 screws.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Cylinder,
    GeomType,
    Locations,
    Mode,
    Plane,
    PolarLocations,
    Rectangle,
    SortBy,
    add,
    chamfer,
    extrude,
    fillet,
    revolve,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import METRIC, metric, revolve_profile_solid


@skill(
    "flange",
    category="structure",
    connectors=("front_face", "rear_face", "bore", "bolt_circle", "od_mid", "raised_face"),
    params=[
        p("od", float, 76.0, min=4, max=2000),
        p("bore_d", float, 44.0, min=0, max=1990),
        p("thickness", float, 9.0, min=0.5, max=200),
        p("bolt_count", int, 12, min=0, max=64),
        p("bolt_size", str, "M6", choices=tuple(METRIC)),
        p("bolt_pcd", float, 0.0, min=0, max=1990, doc="0 = auto midway between bore and od"),
        p("gasket_groove", bool, True),
        p("raised_face_h", float, 0.0, min=0, max=40, doc=">0 adds an ASME-style raised face"),
        p("hub_d", float, 0.0, min=0, max=1990, doc=">0 adds a weld-neck hub"),
        p("hub_len", float, 14.0, min=0, max=400),
        p("fillet_mm", float, 1.2, min=0, max=30),
        p("scallop", bool, False, doc="scalloped OD between bolt holes (weight saving)"),
    ],
)
def flange(**kw) -> SkillResult:
    """Bolted flange with gasket groove and an indexed bolt circle."""
    od, bore, t = kw["od"], kw["bore_d"], kw["thickness"]
    pcd = kw["bolt_pcd"] or (bore + od) / 2.0 * 1.0
    if pcd <= bore + 2 or pcd >= od - 2:
        pcd = bore + (od - bore) * 0.62

    steps: list[list[float]] = [[od, 0.0], [od, t]]
    if kw["raised_face_h"] > 0:
        steps += [[max(pcd - metric(kw["bolt_size"])["clear"] * 1.6, bore + 3), t],
                  [max(pcd - metric(kw["bolt_size"])["clear"] * 1.6, bore + 3), t + kw["raised_face_h"]]]
    if kw["hub_d"] > 0 and kw["hub_len"] > 0:
        steps = [[kw["hub_d"], -kw["hub_len"]], [kw["hub_d"], -kw["hub_len"] * 0.3],
                 [od, 0.0]] + steps[1:]

    z_top = max(s[1] for s in steps)
    z_bot = min(s[1] for s in steps)

    with BuildPart() as bp:
        add(revolve_profile_solid(steps, bore_d=bore))
        if kw["gasket_groove"] and bore > 4:
            with BuildSketch(Plane.XZ):
                with Locations(((bore / 2 + pcd / 2) / 2, z_top - 0.9)):
                    Rectangle(2.6, 1.8)
            revolve(axis=Axis.Z, mode=Mode.SUBTRACT)
        if kw["bolt_count"] > 0:
            clear = metric(kw["bolt_size"])["clear"]
            with PolarLocations(pcd / 2, kw["bolt_count"]):
                Cylinder(radius=clear / 2, height=(z_top - z_bot) * 3,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
            if kw["scallop"]:
                with PolarLocations(od / 2 + clear * 0.35, kw["bolt_count"],
                                    start_angle=180.0 / kw["bolt_count"]):
                    Cylinder(radius=clear * 1.15, height=(z_top - z_bot) * 3,
                             align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
        if kw["fillet_mm"] > 0:
            try:
                circles = bp.edges().filter_by(GeomType.CIRCLE).group_by(SortBy.RADIUS)[-1]
                fillet(circles, min(kw["fillet_mm"], t * 0.35))
            except Exception:
                pass

    res = SkillResult(bp.part, interfaces={"od": od, "bore_d": bore, "thickness": t, "bolt_pcd": pcd})
    res.add(
        Connector.face("front_face", z_top),
        Connector.face("rear_face", z_bot, outward=False),
        Connector.at("bore", z=z_bot),
        Connector.at("od_mid", x=od / 2, z=(z_top + z_bot) / 2, ry=90.0),
        Connector.face("raised_face", z_top, doc="gasket seating plane"),
    )
    if kw["bolt_count"] > 0:
        res.add(Connector.polar("bolt_circle", pcd / 2, kw["bolt_count"], z_top,
                                doc="one seat per hole; mate a screw here with count = bolt_count"))
    return res


@skill(
    "bracket",
    category="structure",
    connectors=("base_face", "wall_face", "hole_a", "hole_b", "top"),
    params=[
        p("length", float, 60.0, min=3, max=1200),
        p("width", float, 34.0, min=3, max=1200),
        p("thickness", float, 5.0, min=0.5, max=120),
        p("upstand", float, 30.0, min=0, max=800, doc="height of the vertical leg, 0 = flat plate"),
        p("hole_d", float, 6.6, min=0, max=120),
        p("hole_count", int, 2, min=0, max=24),
        p("rib_count", int, 1, min=0, max=12, doc="gusset ribs between the legs"),
        p("fillet_mm", float, 3.0, min=0, max=40),
    ],
)
def bracket(**kw) -> SkillResult:
    """L-bracket or mounting plate with gussets."""
    L, W, t, up = kw["length"], kw["width"], kw["thickness"], kw["upstand"]
    with BuildPart() as bp:
        Box(L, W, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
        if up > 0:
            with Locations((-L / 2 + t / 2, 0, 0)):
                Box(t, W, up, align=(Align.CENTER, Align.CENTER, Align.MIN))
            for i in range(kw["rib_count"]):
                y = -W / 2 + W * (i + 1) / (kw["rib_count"] + 1)
                with Locations((-L / 2 + t, y, t)):
                    Box(min(L * 0.4, up), max(t * 0.7, 1.2), up * 0.7,
                        align=(Align.MIN, Align.CENTER, Align.MIN))
        if kw["hole_count"] > 0 and kw["hole_d"] > 0:
            for i in range(kw["hole_count"]):
                x = -L / 2 + L * (i + 1) / (kw["hole_count"] + 1) + L * 0.12
                with Locations((x, 0, 0)):
                    Cylinder(radius=kw["hole_d"] / 2, height=t * 3,
                             align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
        if kw["fillet_mm"] > 0:
            try:
                fillet(bp.edges().filter_by(Axis.Y).sort_by(Axis.X)[:2], min(kw["fillet_mm"], t * 0.9))
            except Exception:
                pass

    res = SkillResult(bp.part, interfaces={"length": L, "width": W, "thickness": t})
    hole_x = -L / 2 + L / (kw["hole_count"] + 1) + L * 0.12 if kw["hole_count"] else 0.0
    res.add(
        Connector.face("base_face", 0.0, outward=False),
        Connector.face("top", t),
        Connector.at("wall_face", x=-L / 2, z=up / 2 if up else t / 2, ry=-90.0, flip=True),
        Connector.at("hole_a", x=hole_x, z=t),
        Connector.at("hole_b", x=-hole_x, z=t),
    )
    return res


@skill(
    "heat_shield",
    category="structure",
    connectors=("front_face", "rear_face", "bore", "bolt_circle", "od_mid"),
    params=[
        p("od", float, 92.0, min=4, max=2000),
        p("bore_d", float, 30.0, min=0, max=1990),
        p("thickness", float, 1.2, min=0.2, max=30),
        p("dish", float, 6.0, min=0, max=200, doc="axial dish depth (conical shield)"),
        p("bolt_count", int, 0, min=0, max=48),
        p("bolt_size", str, "M5", choices=tuple(METRIC)),
        p("bolt_pcd", float, 0.0, min=0, max=1990),
        p("bead_count", int, 0, min=0, max=12, doc="stiffening beads"),
    ],
)
def heat_shield(**kw) -> SkillResult:
    """Thin dished heat shield — the part that sits between turbine and centre housing."""
    od, bore, t, dish = kw["od"], kw["bore_d"], kw["thickness"], kw["dish"]
    # The dish tapers from `od` down to a lip just clear of the bore. Interpolating
    # between those two diameters (rather than a fixed fraction of od) keeps the
    # profile monotonic on narrow annuli, where od*0.7 can fall below the bore and
    # hand OCCT a self-crossing wire.
    lip = max(bore + 2.0, 2.0)
    steps = [[od, 0.0], [od, t]] if dish <= 0 else [
        [od, 0.0], [od, t], [lip + (od - lip) * 0.55, dish * 0.55 + t], [lip, dish + t],
    ]
    with BuildPart() as bp:
        add(revolve_profile_solid(steps, bore_d=bore))
        if kw["bead_count"] > 0:
            for i in range(kw["bead_count"]):
                r = bore / 2 + (od / 2 - bore / 2) * (i + 1) / (kw["bead_count"] + 1)
                with BuildSketch(Plane.XZ):
                    with Locations((r, t + 0.1)):
                        Rectangle(t * 1.4, t * 0.9)
                revolve(axis=Axis.Z)
        pcd = kw["bolt_pcd"] or od * 0.82
        if kw["bolt_count"] > 0:
            clear = metric(kw["bolt_size"])["clear"]
            with PolarLocations(pcd / 2, kw["bolt_count"]):
                Cylinder(radius=clear / 2, height=(dish + t) * 4,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

    pcd = kw["bolt_pcd"] or od * 0.82
    res = SkillResult(bp.part, interfaces={"od": od, "bore_d": bore, "thickness": t, "bolt_pcd": pcd})
    res.add(
        Connector.face("front_face", t + max(dish, 0.0)),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=od / 2, z=t / 2, ry=90.0),
    )
    if kw["bolt_count"] > 0:
        res.add(Connector.polar("bolt_circle", pcd / 2, kw["bolt_count"], t))
    return res
