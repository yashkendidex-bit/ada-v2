"""Synthesised skill `diaphragm_spring` — admitted by the geometry gate on 2026-08-24.

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
    "diaphragm_spring",
    category="power",
    connectors=("front_face", "rear_face", "bore", "od_mid"),
    params=[
        p("od", float, 210.0, min=40, max=600),
        p("bore_d", float, 60.0, min=8, max=560),
        p("thickness", float, 3.0, min=0.5, max=15),
        p("finger_count", int, 18, min=0, max=48),
        p("cone_height", float, 9.0, min=0.0, max=60,
          doc="free-state lift; here it fixes how far the release fingers reach in"),
    ],
)
def diaphragm_spring(**kw) -> SkillResult:
    """Belleville diaphragm spring: one annular ring split into release fingers.

    The lift a diaphragm gives is set by finger length, so `cone_height` drives the
    slot depth rather than being carried as decoration.
    """
    od, t = kw["od"], kw["thickness"]
    bore = min(kw["bore_d"], od - 8.0)
    n = kw["finger_count"]
    lift = min(max(kw["cone_height"], 1.0), od * 0.2)
    r_i = bore / 2.0
    slot_len = min(1.5 * lift, (od / 2.0 - r_i) * 0.75)
    r_mid = r_i + slot_len / 2.0
    slot_w = max(2.0 * math.pi * r_mid / max(n, 1) * 0.35, 0.6)
    with BuildPart() as bp:
        add(revolve_profile_solid([[od, 0.0], [od, t]], bore_d=bore))
        if n > 0:
            with Locations((0.0, 0.0, t / 2.0)):
                with PolarLocations(r_mid, n):
                    Box(slot_len, slot_w, t + 2.0,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
    res = SkillResult(bp.part, interfaces={"od": od, "bore_d": bore, "length": t,
                                           "thickness": t})
    res.add(
        Connector.face("front_face", t),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=od / 2.0, z=t / 2.0, ry=90.0),
    )
    return res
