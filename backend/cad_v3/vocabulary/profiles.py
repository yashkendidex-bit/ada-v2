"""The keystone skill: bodies of revolution from a named step profile.

This one skill replaces roughly 60% of what cad_v2 did with boxes. Housings,
centre housings, hubs, shrouds, spacers, covers, bearing carriers — anything
whose defining feature is a turned or cast axisymmetric contour.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    Box,
    BuildLine,
    BuildPart,
    BuildSketch,
    Cylinder,
    GeomType,
    Location,
    Locations,
    Mode,
    Plane,
    PolarLocations,
    Polyline,
    Rectangle,
    SortBy,
    add,
    fillet,
    make_face,
    revolve,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import METRIC, metric, revolve_profile_solid

PROFILE_PRESETS = ("explicit", "stepped_bore", "ogive", "cone", "barrel", "spool")


def _preset_steps(profile: str, od: float, length: float) -> list[list[float]]:
    r = od
    if profile == "ogive":
        return [[r * 0.9, 0], [r, length * 0.25], [r * 0.82, length * 0.62], [r * 0.35, length * 0.9], [0.001, length]]
    if profile == "cone":
        return [[r, 0], [r * 0.45, length]]
    if profile == "barrel":
        return [[r * 0.78, 0], [r, length * 0.5], [r * 0.78, length]]
    if profile == "spool":
        return [[r, 0], [r, length * 0.15], [r * 0.62, length * 0.15], [r * 0.62, length * 0.85], [r, length * 0.85], [r, length]]
    return [[r, 0], [r, length]]


@skill(
    "revolved_profile",
    category="structure",
    connectors=("front_face", "rear_face", "bore_front", "bore_rear", "od_mid", "bolt_circle", "port_ring"),
    params=[
        p("profile", str, "explicit", choices=PROFILE_PRESETS, doc="named contour, or 'explicit' to use steps"),
        p("steps", list, None, doc="[[diameter_mm, z_mm], ...] outer contour, bottom to top"),
        p("od", float, 60.0, min=1, max=2000, doc="outer diameter when using a preset profile"),
        p("length", float, 40.0, min=0.5, max=2000),
        p("bore_d", float, 0.0, min=0, max=1990, doc="through bore diameter, 0 = solid"),
        p("wall_mm", float, 0.0, min=0, max=200, doc=">0 hollows the body to this wall thickness"),
        p("fillet_mm", float, 0.0, min=0, max=50),
        p("fin_count", int, 0, min=0, max=96, doc="radial cooling fins"),
        p("fin_h", float, 4.0, min=0.4, max=80),
        p("fin_t", float, 1.6, min=0.3, max=20),
        p("port_count", int, 0, min=0, max=16, doc="radial ports / oil feeds through the wall"),
        p("port_d", float, 6.0, min=0.5, max=120),
        p("port_z_frac", float, 0.5, min=0.02, max=0.98),
        p("bolt_count", int, 0, min=0, max=48),
        p("bolt_size", str, "M6", choices=tuple(METRIC)),
        p("bolt_pcd", float, 0.0, min=0, max=1990, doc="0 = auto (0.86 x od)"),
        p("bolt_face", str, "front", choices=("front", "rear")),
        p("water_jacket", bool, False, doc="annular cooling cavity inside the wall"),
        p("oil_gallery", bool, False, doc="annular oil groove on the bore + radial feed"),
    ],
)
def revolved_profile(**kw) -> SkillResult:
    """Axisymmetric turned or cast body from a step profile."""
    profile = kw["profile"]
    od, length = kw["od"], kw["length"]
    steps = kw["steps"] or _preset_steps(profile if profile != "explicit" else "cone", od, length)
    steps = [[float(s[0]), float(s[1])] for s in steps]

    z_lo = min(s[1] for s in steps)
    z_hi = max(s[1] for s in steps)
    d_max = max(s[0] for s in steps)
    bore_d = kw["bore_d"]

    body = revolve_profile_solid(steps, bore_d=bore_d)

    with BuildPart() as bp:
        add(body)

        if kw["wall_mm"] > 0 and bore_d <= 0:
            inner = [[max(s[0] - 2 * kw["wall_mm"], 0.4), s[1]] for s in steps]
            inner[0][1] = z_lo + kw["wall_mm"]
            inner[-1][1] = z_hi - kw["wall_mm"]
            if inner[-1][1] > inner[0][1]:
                add(revolve_profile_solid(inner), mode=Mode.SUBTRACT)

        if kw["water_jacket"] and kw["wall_mm"] <= 0 and bore_d > 0:
            j_out = d_max - 6.0
            j_in = max(bore_d + 6.0, j_out - 10.0)
            if j_out > j_in + 2:
                jacket = [[j_out, z_lo + 4], [j_out, z_hi - 4]]
                add(revolve_profile_solid(jacket, bore_d=j_in), mode=Mode.SUBTRACT)

        if kw["oil_gallery"] and bore_d > 2:
            with BuildSketch(Plane.XZ) as g:
                with Locations((bore_d / 2 + 1.2, (z_lo + z_hi) / 2)):
                    Rectangle(2.4, 3.0)
            revolve(axis=Axis.Z, mode=Mode.SUBTRACT)
            with Locations((0, 0, (z_lo + z_hi) / 2)):
                Cylinder(radius=1.6, height=d_max * 1.2, rotation=(0, 90, 0), mode=Mode.SUBTRACT)

        if kw["fin_count"] > 0:
            fin_z = (z_lo + z_hi) / 2
            with PolarLocations(d_max / 2 + kw["fin_h"] / 2 - 0.2, kw["fin_count"]):
                Box(kw["fin_h"], kw["fin_t"], (z_hi - z_lo) * 0.78,
                    align=(Align.CENTER, Align.CENTER, Align.CENTER))

        if kw["port_count"] > 0:
            pz = z_lo + (z_hi - z_lo) * kw["port_z_frac"]
            with Locations((0, 0, pz)):
                with PolarLocations(0, kw["port_count"]):
                    Cylinder(radius=kw["port_d"] / 2, height=d_max * 1.3,
                             rotation=(0, 90, 0), mode=Mode.SUBTRACT)

        pcd = kw["bolt_pcd"] or d_max * 0.86
        if kw["bolt_count"] > 0:
            clear = metric(kw["bolt_size"])["clear"]
            # Centre the drill on the profile's own mid-height. A CENTER-aligned
            # cylinder sits at z=0, which for a profile that starts at z_lo=0 only
            # reaches 70% of the way up and leaves the top of the flange solid — a
            # bolt then reads as an interference with the plate it passes through.
            with Locations((0.0, 0.0, (z_lo + z_hi) / 2)):
                with PolarLocations(pcd / 2, kw["bolt_count"]):
                    Cylinder(radius=clear / 2, height=(z_hi - z_lo) * 1.4,
                             align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

        if kw["fillet_mm"] > 0:
            try:
                circles = bp.edges().filter_by(GeomType.CIRCLE).group_by(SortBy.RADIUS)[-1]
                fillet(circles, min(kw["fillet_mm"], d_max * 0.1))
            except Exception:
                pass

    part = bp.part
    res = SkillResult(part, interfaces={"od": d_max, "bore_d": bore_d, "length": z_hi - z_lo})
    res.add(
        Connector.face("front_face", z_hi, doc="upper mating face, +Z out"),
        Connector.face("rear_face", z_lo, outward=False, doc="lower mating face, -Z out"),
        Connector.at("bore_front", z=z_hi, doc="bore mouth at the top"),
        Connector.at("bore_rear", z=z_lo, doc="bore mouth at the bottom"),
        Connector.at("od_mid", x=d_max / 2, z=(z_lo + z_hi) / 2, ry=90, doc="outer surface, mid height"),
    )
    if kw["bolt_count"] > 0:
        res.add(Connector.polar("bolt_circle", pcd / 2,
                                kw["bolt_count"], z_hi if kw["bolt_face"] == "front" else z_lo,
                                doc="one seat per bolt hole"))
        res.interfaces["bolt_pcd"] = pcd
    if kw["port_count"] > 0:
        pz = z_lo + (z_hi - z_lo) * kw["port_z_frac"]
        res.add(Connector.polar("port_ring", d_max / 2, kw["port_count"], pz, tilt_deg=90.0))
    return res
