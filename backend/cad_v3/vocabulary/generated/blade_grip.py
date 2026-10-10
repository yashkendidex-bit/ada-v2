"""Synthesised skill `blade_grip` — admitted by the geometry gate on 2026-08-25.

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
    "blade_grip",
    category="rotor",
    connectors=("blade_seat", "horn_seat", "rear_face"),
    params=[
        p("grip_d", float, 64.0, min=10, max=250),
        p("horn_drop", float, 34.0, min=5, max=150),
        p("horn_reach", float, 110.0, min=20, max=400),
        p("horn_thickness", float, 22.0, min=4, max=100),
        p("horn_width", float, 26.0, min=4, max=120),
        p("length", float, 125.0, min=20, max=500),
        p("trunnion_d", float, 30.0, min=5, max=120),
        p("trunnion_len", float, 36.0, min=5, max=150),
        p("horn_bolt_d", float, 12.0, min=2, max=60),
    ],
)
def blade_grip(**kw) -> SkillResult:
    """Rotor blade grip barrel with radial trunnion pins and a drooping pitch horn."""
    gd, L = kw["grip_d"], kw["length"]
    r = gd / 2
    t = min(kw["horn_thickness"], L * 0.35)
    zc = L * 0.72
    zt, zb = zc + t / 2, zc - t / 2
    w = min(kw["horn_width"], kw["horn_reach"] * 0.5)
    reach = max(kw["horn_reach"], r + w + 10.0)
    drop = max(min(kw["horn_drop"], zb - max(L * 0.05, 2.0)), 2.0)
    ztb = zb - drop
    x0 = -(r - min(10.0, r * 0.3))
    td = min(kw["trunnion_d"], L * 0.44)
    tl = kw["trunnion_len"]

    with BuildPart() as bp:
        Cylinder(radius=r, height=L, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, 0, L * 0.25)):
            Cylinder(radius=td / 2, height=gd + 2 * tl, rotation=(0, 90, 0))
        with Locations(((x0 + reach - w) / 2, 0, (zb + zt) / 2)):
            Box(reach - w - x0, w, t)
        with Locations((reach - w / 2, 0, (ztb + zt) / 2)):
            Box(w, w, t + drop)
        bolt_r = min(kw["horn_bolt_d"], w * 0.5, (t + drop) * 0.5) / 2
        with Locations((reach - w / 2, 0, (ztb + zt) / 2)):
            Cylinder(radius=bolt_r, height=w * 3, rotation=(90, 0, 0), mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={
        "grip_d": gd, "length": L, "horn_reach": reach, "horn_drop": drop,
        "horn_thickness": t, "horn_width": w, "trunnion_d": td,
        "trunnion_len": tl, "horn_bolt_d": bolt_r * 2})
    res.add(
        Connector.face("blade_seat", L),
        Connector.at("horn_seat", x=reach - w / 2, z=(ztb + zt) / 2, ry=90.0),
        Connector.face("rear_face", 0.0, outward=False),
    )
    return res
