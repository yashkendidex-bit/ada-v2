"""Synthesised skill `pressure_plate` — admitted by the geometry gate on 2026-08-24.

Written by ADA, not by hand. It passed: builds at defaults, builds across its
declared parameter range, twice identically, seats on material, axis about Z,
plus a build at the parameters of the spec that asked for it.

Edit freely — from here on this is ordinary vocabulary.
"""

from __future__ import annotations

import math  # noqa: F401

from build123d import *  # noqa: F401,F403

from ...frames import Connector, SkillResult  # noqa: F401
from ...registry import p, skill  # noqa: F401
from .._util import (  # noqa: F401
    METRIC,
    axial_bore,
    blade_sweep,
    metric,
    revolve_profile_solid,
    spiral_path,
)


@skill(
    "pressure_plate",
    category="power",
    connectors=("front_face", "rear_face", "bore", "od_mid", "lug_ring"),
    params=[
        p("od", float, 210.0, min=50, max=600),
        p("bore_d", float, 140.0, min=10, max=560),
        p("thickness", float, 20.0, min=4, max=80),
        p("lug_count", int, 3, min=0, max=12),
        p("friction_face_w", float, 30.0, min=4, max=200),
        p("lug_h", float, 6.0, min=1, max=30, doc="drive-strap boss height"),
    ],
)
def pressure_plate(**kw) -> SkillResult:
    """Clutch pressure plate: cast ring, machined friction band, drive-strap bosses."""
    od, t = kw["od"], kw["thickness"]
    bore = min(kw["bore_d"], od - 16.0)
    band = min(kw["friction_face_w"], (od - bore) / 2.0)
    lug_h = min(kw["lug_h"], t * 0.6)
    n_lug = kw["lug_count"]
    r_lug = od / 2.0 - max(od * 0.03, 4.0)
    lug_w = max(od * 0.05, 6.0)
    with BuildPart() as bp:
        add(revolve_profile_solid([[od, 0.0], [od, t]], bore_d=bore))
        if band < (od - bore) / 2.0 - 1.0:
            # relieve the rear face so only the outer band touches the disc
            add(revolve_profile_solid([[od - 2.0 * band, 0.0], [od - 2.0 * band, 1.2]],
                                      bore_d=bore, arc=360.0), mode=Mode.SUBTRACT)
        if n_lug > 0:
            with Locations((0.0, 0.0, t + lug_h / 2.0)):
                with PolarLocations(r_lug, n_lug):
                    Box(lug_w, lug_w, lug_h,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER))
    res = SkillResult(bp.part, interfaces={"od": od, "bore_d": bore, "length": t,
                                           "thickness": t})
    res.add(
        Connector.face("front_face", t),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=od / 2.0, z=t / 2.0, ry=90.0),
        Connector.polar("lug_ring", r_lug, max(n_lug, 1), t + lug_h),
    )
    return res
