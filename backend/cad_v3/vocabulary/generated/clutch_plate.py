"""Synthesised skill `clutch_plate` — admitted by the geometry gate on 2026-08-24.

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
    "clutch_plate",
    category="power",
    connectors=("front_face", "rear_face", "bore", "od_mid", "hub_face"),
    params=[
        p("od", float, 200.0, min=40, max=600, doc="facing outside diameter"),
        p("bore_d", float, 25.0, min=5, max=400, doc="splined hub bore"),
        p("thickness", float, 8.0, min=2, max=40),
        p("facing_thickness", float, 3.5, min=0.5, max=12),
        p("spline_count", int, 24, min=6, max=60),
        p("hub_od", float, 55.0, min=20, max=300),
    ],
)
def clutch_plate(**kw) -> SkillResult:
    """Dry friction disc: splined hub, sprung web, an annular facing each side."""
    od, t = kw["od"], kw["thickness"]
    bore = min(kw["bore_d"], od * 0.4)
    hub_od = min(max(kw["hub_od"], bore + 8.0), od * 0.55)
    web_d = max(od * 0.62, hub_od + 4.0)
    web_t = max(min(t - 2.0 * kw["facing_thickness"], t * 0.5), t * 0.2)
    with BuildPart() as bp:
        add(revolve_profile_solid([[od, 0.0], [od, t]], bore_d=web_d))
        add(revolve_profile_solid([[web_d, (t - web_t) / 2.0],
                                   [web_d, (t + web_t) / 2.0]], bore_d=hub_od))
        add(revolve_profile_solid([[hub_od, 0.0], [hub_od, t]], bore_d=bore))
        with Locations((0.0, 0.0, t / 2.0)):
            with PolarLocations(bore / 2.0, kw["spline_count"]):
                Box(bore * 0.12 + 0.4, bore * 0.12 + 0.4, t + 2.0,
                    align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
    res = SkillResult(bp.part, interfaces={"od": od, "bore_d": bore, "length": t,
                                           "thickness": t, "hub_od": hub_od})
    res.add(
        Connector.face("front_face", t),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=od / 2.0, z=t / 2.0, ry=90.0),
        Connector.face("hub_face", t),
    )
    return res
