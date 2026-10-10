"""Synthesised skill `sphere` — admitted by the geometry gate on 2026-09-27.

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
    "sphere",
    category="primitives",
    connectors=("center", "north_pole", "south_pole", "equator"),
    params=[
        p("radius", float, 25.0, min=0.5, max=1000),
    ],
)
def sphere(**kw) -> SkillResult:
    """Datum sphere of the given radius, resting with its base on z=0."""
    r = kw["radius"]
    with BuildPart() as bp:
        Sphere(radius=r, align=(Align.CENTER, Align.CENTER, Align.MIN))

    res = SkillResult(
        bp.part,
        interfaces={"radius": r, "od": 2 * r, "diameter": 2 * r},
        meta={"source": "synth"},
    )
    res.add(
        Connector.at("center", z=r),
        Connector.at("north_pole", z=2 * r),
        Connector.at("south_pole", z=0.0),
        Connector.at("equator", x=r, z=r, ry=90.0),
    )
    return res
