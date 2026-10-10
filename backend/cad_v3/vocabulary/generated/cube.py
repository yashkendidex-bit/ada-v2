"""Synthesised skill `cube` — admitted by the geometry gate on 2026-09-27.

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
    "cube",
    category="calibration",
    connectors=("bottom", "top", "side"),
    params=[
        p("size", float, 10.0, min=1, max=100, doc="edge length of the calibration cube"),
    ],
)
def cube(**kw) -> SkillResult:
    """Calibration cube, base on z=0, axis +Z."""
    s = kw["size"]
    with BuildPart() as bp:
        Box(s, s, s, align=(Align.CENTER, Align.CENTER, Align.MIN))
    res = SkillResult(
        bp.part,
        interfaces={"size": s, "od": s, "width": s, "length": s, "height": s},
        meta={"source": "synth"},
    )
    res.add(
        Connector.face("bottom", 0.0, outward=False),
        Connector.face("top", s),
        Connector.polar("side", s / 2, 4, s / 2, tilt_deg=90.0),
    )
    return res
