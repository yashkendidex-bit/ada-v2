"""Synthesised skill `emitter_ring` — admitted by the geometry gate on 2026-08-25.

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
    "emitter_ring",
    category="optics",
    connectors=("front_face", "rear_face"),
    params=[
        p("od", float, 44.0, min=12, max=140),
        p("bore_d", float, 16.0, min=2, max=100),
        p("height", float, 6.0, min=2, max=20),
        p("segment_count", int, 8, min=2, max=24),
        p("gap_deg", float, 6.0, min=0.5, max=30),
        p("web_frac", float, 0.25, min=0.1, max=0.5),
    ],
)
def emitter_ring(**kw) -> SkillResult:
    """Segmented emission ring: arc segments split by radial gaps on a rear carrier web."""
    od, bore, h = kw["od"], kw["bore_d"], kw["height"]
    n = max(2, int(kw["segment_count"]))
    web_t = max(0.8, h * kw["web_frac"])

    seg_deg = 360.0 / n
    gap = min(max(kw["gap_deg"], 0.2), seg_deg * 0.55)
    rm = (bore / 2.0 + od / 2.0) / 2.0
    half_w = rm * math.tan(math.radians(gap) / 2.0) + 0.05
    r_in = max(0.0, bore / 2.0 - 1.0)
    r_out = od / 2.0 + 1.0

    with BuildPart() as bp:
        add(revolve_profile_solid([[od, 0.0], [od, h]], bore_d=bore))
        # radial gap cutters: prismatic boxes swung to each segment boundary,
        # starting at the top of the carrier web so segments stay attached
        for i in range(n):
            ang = i * seg_deg
            with Locations(Location((0, 0, web_t), (0, 0, ang))):
                with Locations(((r_in + r_out) / 2.0, 0, 0)):
                    Box(r_out - r_in, 2.0 * half_w, h - web_t + 0.5,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                        mode=Mode.SUBTRACT)

    res = SkillResult(bp.part,
                      interfaces={"od": od, "bore_d": bore, "height": h,
                                  "segment_count": n, "gap_deg": kw["gap_deg"],
                                  "web_thickness": web_t},
                      meta={"source": "synth"})
    res.add(
        Connector.face("front_face", h),
        Connector.face("rear_face", 0.0, outward=False),
    )
    return res
