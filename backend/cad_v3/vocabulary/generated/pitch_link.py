"""Synthesised skill `pitch_link` — admitted by the geometry gate on 2026-08-25.

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
    "pitch_link",
    category="rotor",
    connectors=("front_face", "rear_face"),
    params=[
        p("eye_id", float, 10.5, min=2, max=60),
        p("eye_od", float, 24.0, min=5, max=120),
        p("length", float, 52.0, min=10, max=300),
        p("rod_d", float, 14.0, min=2, max=80),
    ],
)
def pitch_link(**kw) -> SkillResult:
    """Straight pitch link rod with a bored eye boss at each end."""
    L, od = kw["length"], kw["eye_od"]
    bh = min(od * 0.6, L * 0.35)
    rod_r = max(min(kw["rod_d"], od - 4.0), 2.0) / 2
    id_r = max(min(kw["eye_id"], od - 4.0), 1.0) / 2

    with BuildPart() as bp:
        Cylinder(radius=rod_r, height=L, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, 0, bh / 2), (0, 0, L - bh / 2)):
            Cylinder(radius=od / 2, height=bh, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, 0, bh / 2), (0, 0, L - bh / 2)):
            Cylinder(radius=id_r, height=od * 3, rotation=(90, 0, 0), mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={
        "eye_id": id_r * 2, "eye_od": od, "length": L, "rod_d": rod_r * 2})
    res.add(
        Connector.face("front_face", L),
        Connector.face("rear_face", 0.0, outward=False),
    )
    return res
