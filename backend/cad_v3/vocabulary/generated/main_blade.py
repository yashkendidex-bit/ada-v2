"""Synthesised skill `main_blade` — admitted by the geometry gate on 2026-08-25.

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
    "main_blade",
    category="rotor",
    connectors=("rear_face",),
    params=[
        p("root_chord", float, 150.0, min=10, max=600),
        p("root_thickness", float, 14.0, min=2, max=60),
        p("span", float, 1500.0, min=50, max=5000),
        p("thickness", float, 26.0, min=2, max=100),
        p("tip_chord", float, 105.0, min=10, max=500),
        p("twist_deg", float, -9.0, min=-45, max=45),
    ],
)
def main_blade(**kw) -> SkillResult:
    """Tapered, washed-out main rotor blade lofted root to tip along +Z."""
    span, tw = kw["span"], kw["twist_deg"]
    mid_ch = (kw["root_chord"] + kw["tip_chord"]) / 2
    tip_th = kw["thickness"] * 0.85

    with BuildPart() as bp:
        with BuildSketch(Plane.XY):
            Rectangle(kw["root_chord"], kw["root_thickness"], rotation=0.0)
        with BuildSketch(Plane.XY.offset(span * 0.45)):
            Rectangle(mid_ch, kw["thickness"], rotation=tw * 0.45)
        with BuildSketch(Plane.XY.offset(span)):
            Rectangle(kw["tip_chord"], tip_th, rotation=tw)
        loft()

    res = SkillResult(bp.part, interfaces={
        "root_chord": kw["root_chord"], "tip_chord": kw["tip_chord"],
        "span": span, "thickness": kw["thickness"],
        "root_thickness": kw["root_thickness"], "twist_deg": tw})
    res.add(Connector.face("rear_face", 0.0, outward=False))
    return res
