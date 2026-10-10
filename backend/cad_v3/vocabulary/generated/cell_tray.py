"""Synthesised skill `cell_tray` — admitted by the geometry gate on 2026-08-25.

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
    "cell_tray",
    category="power",
    connectors=("rim_face", "seat"),
    params=[
        p("od", float, 70.0, min=20, max=220),
        p("depth", float, 26.0, min=8, max=80),
        p("wall_mm", float, 3.0, min=1, max=12),
        p("boss_d", float, 28.0, min=6, max=90),
        p("boss_h", float, 5.0, min=1, max=16),
        p("bolt_count", int, 3, min=2, max=12),
        p("bolt_pcd", float, 60.0, min=15, max=190),
        p("bolt_size", str, "M3"),
    ],
)
def cell_tray(**kw) -> SkillResult:
    """Open circular power-cell tray with centre seating boss and bolt-down floor."""
    od, depth, wall = kw["od"], kw["depth"], kw["wall_mm"]
    floor_t = wall

    # bolt size -> nominal hole diameter (parse "M3"/"M4" style strings safely)
    s = str(kw["bolt_size"]).strip().upper()
    digits = "".join(c for c in s if c.isdigit() or c == ".")
    hole_d = float(digits) if digits else 3.0

    boss_d = min(kw["boss_d"], od - 2.0 * wall - 2.0)
    pcd = min(kw["bolt_pcd"], od - 2.0 * wall - hole_d - 2.0)
    pcd = max(pcd, boss_d / 2.0 + hole_d / 2.0 + 2.0)

    with BuildPart() as bp:
        Cylinder(radius=od / 2, height=depth,
                 align=(Align.CENTER, Align.CENTER, Align.MIN))
        # tray cavity: open top, floored at floor_t
        with Locations((0, 0, floor_t)):
            Cylinder(radius=(od - 2.0 * wall) / 2, height=depth,
                     align=(Align.CENTER, Align.CENTER, Align.MIN),
                     mode=Mode.SUBTRACT)
        # centre boss the cell bottoms out on
        with Locations((0, 0, floor_t)):
            Cylinder(radius=boss_d / 2, height=kw["boss_h"],
                     align=(Align.CENTER, Align.CENTER, Align.MIN))
        # bolt holes through the floor only
        with Locations((0, 0, -1.0)):
            with PolarLocations(pcd / 2, kw["bolt_count"]):
                Cylinder(radius=hole_d / 2, height=floor_t + 2.0,
                         mode=Mode.SUBTRACT)

    res = SkillResult(bp.part,
                      interfaces={"od": od, "depth": depth, "wall_mm": wall,
                                  "boss_d": boss_d, "boss_h": kw["boss_h"],
                                  "bolt_pcd": pcd, "bolt_count": kw["bolt_count"],
                                  "bolt_size": str(kw["bolt_size"])},
                      meta={"source": "synth"})
    res.add(
        Connector.face("rim_face", depth),
        Connector.at("seat", z=floor_t + kw["boss_h"]),
    )
    return res
