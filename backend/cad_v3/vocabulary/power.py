"""Rotor-group hardware: shafts, thrust collars, seals, slingers.

The shaft is the datum of most rotating assemblies, so it publishes a *stack* of
indexed seats. Every other rotor part is positioned by mating to one of those
seats — which is why a turbocharger spec never needs to state an axial position.
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
    Locations,
    Mode,
    Plane,
    PolarLocations,
    Rectangle,
    add,
    chamfer,
    revolve,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import METRIC, metric, revolve_profile_solid


@skill(
    "shaft",
    category="power",
    connectors=("end_a", "end_b", "seats", "mid", "key_seat", "thread_end"),
    params=[
        p("steps", list, None, doc="[[diameter_mm, z_mm], ...]; overrides d/length"),
        p("d", float, 12.0, min=1, max=400, doc="plain shaft diameter"),
        p("length", float, 90.0, min=2, max=3000),
        p("seats", list, None, doc="[z_mm, ...] axial seat positions for mating parts"),
        p("bore_d", float, 0.0, min=0, max=390, doc="hollow shaft bore"),
        p("keyway", bool, False),
        p("key_w", float, 4.0, min=0.5, max=40),
        p("key_h", float, 2.5, min=0.3, max=25),
        p("key_z", float, 0.0, min=0, max=3000, doc="keyway centre; 0 = auto 0.25 x length"),
        p("oil_groove_count", int, 0, min=0, max=12, doc="annular oil grooves"),
        p("chamfer_mm", float, 0.6, min=0, max=8),
    ],
)
def shaft(**kw) -> SkillResult:
    """Stepped rotor shaft with indexed axial seats."""
    length = kw["length"]
    steps = kw["steps"] or [[kw["d"], 0.0], [kw["d"], length]]
    steps = [[float(s[0]), float(s[1])] for s in steps]
    z_lo, z_hi = steps[0][1], steps[-1][1]
    length = z_hi - z_lo
    d_max = max(s[0] for s in steps)

    with BuildPart() as bp:
        add(revolve_profile_solid(steps, bore_d=kw["bore_d"]))

        if kw["keyway"]:
            kz = kw["key_z"] or z_lo + length * 0.25
            with Locations((0, 0, kz)):
                Box(d_max, kw["key_w"], kw["key_h"] * 2,
                    align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)

        n_groove = kw["oil_groove_count"]
        for i in range(n_groove):
            gz = z_lo + length * (i + 1) / (n_groove + 1)
            with BuildSketch(Plane.XZ):
                with Locations((d_max / 2 - 0.35, gz)):
                    Rectangle(1.4, 1.6)
            revolve(axis=Axis.Z, mode=Mode.SUBTRACT)

        if kw["chamfer_mm"] > 0:
            try:
                ends = bp.faces().sort_by(Axis.Z)
                chamfer(ends[0].edges() + ends[-1].edges(), min(kw["chamfer_mm"], d_max * 0.12))
            except Exception:
                pass

    seat_z = [float(z) for z in (kw["seats"] or [])] or [z_lo + length * f for f in (0.25, 0.5, 0.75)]
    res = SkillResult(bp.part, interfaces={"od": d_max, "length": length, "bore_d": kw["bore_d"]})
    res.add(
        Connector.face("end_a", z_lo, outward=False, doc="lower end, -Z out"),
        Connector.face("end_b", z_hi, doc="upper end, +Z out"),
        Connector.stack("seats", seat_z, doc="indexed axial seats for rotor parts"),
        Connector.at("mid", z=(z_lo + z_hi) / 2),
        Connector.at("thread_end", z=z_hi, doc="nut lands here"),
    )
    if kw["keyway"]:
        res.add(Connector.at("key_seat", z=kw["key_z"] or z_lo + length * 0.25,
                             doc="keyway centre"))
    return res


@skill(
    "thrust_collar",
    category="power",
    connectors=("front_face", "rear_face", "bore", "od_mid"),
    params=[
        p("bore_d", float, 12.0, min=1, max=400),
        p("od", float, 30.0, min=2, max=800),
        p("thickness", float, 5.0, min=0.4, max=100),
        p("pad_count", int, 8, min=0, max=24, doc="tapered-land thrust pads"),
        p("pad_depth", float, 0.4, min=0.05, max=5.0),
    ],
)
def thrust_collar(**kw) -> SkillResult:
    """Thrust collar / tapered-land bearing washer."""
    t = kw["thickness"]
    with BuildPart() as bp:
        add(revolve_profile_solid([[kw["od"], 0], [kw["od"], t]], bore_d=kw["bore_d"]))
        if kw["pad_count"] > 0:
            r_mid = (kw["bore_d"] / 2 + kw["od"] / 2) / 2
            pad_w = max((kw["od"] - kw["bore_d"]) * 0.34, 0.8)
            with Locations((0, 0, t - kw["pad_depth"] / 2)):
                with PolarLocations(r_mid, kw["pad_count"]):
                    Box(pad_w, pad_w * 0.55, kw["pad_depth"] * 2,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={"bore_d": kw["bore_d"], "od": kw["od"], "thickness": t})
    res.add(
        Connector.face("front_face", t),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=kw["od"] / 2, z=t / 2, ry=90.0),
    )
    return res


@skill(
    "labyrinth_seal",
    category="power",
    connectors=("front_face", "rear_face", "bore", "od_mid"),
    params=[
        p("bore_d", float, 12.0, min=1, max=400),
        p("od", float, 26.0, min=2, max=800),
        p("length", float, 12.0, min=1, max=300),
        p("fin_count", int, 4, min=1, max=24, doc="knife-edge sealing fins"),
        p("fin_h", float, 1.6, min=0.2, max=20),
        p("carbon_ring", bool, False, doc="add a piston-ring groove instead of fins"),
    ],
)
def labyrinth_seal(**kw) -> SkillResult:
    """Knife-edge labyrinth or piston-ring shaft seal."""
    L, n = kw["length"], kw["fin_count"]
    r_body = kw["od"] / 2 - kw["fin_h"]
    steps: list[list[float]] = [[r_body * 2, 0.0]]
    if kw["carbon_ring"]:
        steps += [[r_body * 2, L]]
    else:
        for i in range(n):
            z0 = L * (i + 0.18) / n
            z1 = L * (i + 0.5) / n
            z2 = L * (i + 0.82) / n
            steps += [[r_body * 2, z0], [kw["od"], z1], [r_body * 2, z2]]
        steps += [[r_body * 2, L]]

    with BuildPart() as bp:
        add(revolve_profile_solid(steps, bore_d=kw["bore_d"]))
        if kw["carbon_ring"]:
            with BuildSketch(Plane.XZ):
                with Locations((r_body - 0.9, L / 2)):
                    Rectangle(1.8, 2.2)
            revolve(axis=Axis.Z, mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={"bore_d": kw["bore_d"], "od": kw["od"], "length": L})
    res.add(
        Connector.face("front_face", L),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=kw["od"] / 2, z=L / 2, ry=90.0),
    )
    return res
