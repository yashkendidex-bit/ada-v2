"""Synthesised skill `field_coil` — admitted by the geometry gate on 2026-08-25.

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
    "field_coil",
    category="coils",
    connectors=("bore", "front_face", "rear_face"),
    params=[
        p("inner_d", float, 40.4, min=10, max=130),
        p("outer_d", float, 56.0, min=16, max=170),
        p("height", float, 9.0, min=3, max=30),
        p("turns", int, 12, min=1, max=36),
        p("wire_d", float, 1.2, min=0.3, max=4.0),
    ],
)
def field_coil(**kw) -> SkillResult:
    """Layer-wound drive coil: bobbin sleeve with embedded wire turns."""
    id_, od, h = kw["inner_d"], kw["outer_d"], kw["height"]
    wr = kw["wire_d"] / 2.0
    sleeve_t = max(1.0, id_ * 0.04)
    sor = id_ / 2.0 + sleeve_t
    n_a = max(1, int(round(math.sqrt(kw["turns"]))))
    n_r = max(1, int(math.ceil(kw["turns"] / n_a)))
    pitch = kw["wire_d"] * 0.85
    span = (n_a - 1) * pitch + kw["wire_d"]
    z0 = (h - span) / 2.0 + wr
    with BuildPart() as bp:
        add(revolve_profile_solid([[sor, 0.0], [sor, h]], bore_d=id_))
        for idx in range(kw["turns"]):
            kcol = idx // n_a
            jrow = idx % n_a
            major = sor + wr * 0.5 + kcol * pitch
            zc = z0 + jrow * pitch
            with Locations((0, 0, zc)):
                Torus(major_radius=major, minor_radius=wr)

    res = SkillResult(bp.part,
                      interfaces={"inner_d": id_, "outer_d": od, "height": h,
                                  "turns": kw["turns"], "wire_d": kw["wire_d"]},
                      meta={"source": "synth"})
    res.add(
        Connector.at("bore", z=0.0),
        Connector.face("front_face", h),
        Connector.face("rear_face", 0.0, outward=False),
    )
    return res
