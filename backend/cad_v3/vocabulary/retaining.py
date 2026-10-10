"""Retaining and sealing hardware — snap rings, keys, pins, O-rings.

Small parts, but they are exactly what makes an exploded technical drawing look
like an engineering document instead of a render.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Plane,
    Torus,
    add,
    fillet,
    revolve,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import revolve_profile_solid


@skill(
    "snap_ring",
    category="retaining",
    connectors=("front_face", "rear_face", "bore", "od_mid"),
    params=[
        p("bore_d", float, 28.0, min=2, max=800, doc="housing bore it retains into"),
        p("thickness", float, 1.2, min=0.2, max=12),
        p("radial_w", float, 2.0, min=0.2, max=25),
        p("internal", bool, True, doc="internal (housing) vs external (shaft) circlip"),
        p("gap_deg", float, 40.0, min=0, max=120, doc="open-end gap"),
    ],
)
def snap_ring(**kw) -> SkillResult:
    """Circlip / retaining ring with a real open-end gap."""
    t, w = kw["thickness"], kw["radial_w"]
    if kw["internal"]:
        od, id_ = kw["bore_d"], kw["bore_d"] - 2 * w
    else:
        od, id_ = kw["bore_d"] + 2 * w, kw["bore_d"]
    arc = max(360.0 - kw["gap_deg"], 30.0)
    solid = revolve_profile_solid([[od, 0.0], [od, t]], bore_d=id_, arc=arc)

    res = SkillResult(solid, interfaces={"bore_d": id_, "od": od, "thickness": t})
    res.add(
        Connector.face("rear_face", 0.0, outward=False),
        Connector.face("front_face", t),
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=od / 2, z=t / 2, ry=90.0),
    )
    return res


@skill(
    "o_ring",
    category="retaining",
    connectors=("bore", "od_mid", "front_face"),
    params=[
        p("bore_d", float, 24.0, min=1, max=900, doc="inside diameter"),
        p("cord_d", float, 2.0, min=0.3, max=20, doc="cross-section diameter"),
    ],
)
def o_ring(**kw) -> SkillResult:
    """Elastomeric O-ring — a genuine torus, sized to an ISO 3601 cord."""
    r_mid = (kw["bore_d"] + kw["cord_d"]) / 2.0
    with BuildPart() as bp:
        Torus(major_radius=r_mid, minor_radius=kw["cord_d"] / 2)
    res = SkillResult(bp.part, interfaces={"bore_d": kw["bore_d"], "od": (r_mid * 2) + kw["cord_d"],
                                           "thickness": kw["cord_d"]})
    res.add(
        Connector.at("bore", z=0.0),
        Connector.at("od_mid", x=r_mid + kw["cord_d"] / 2, ry=90.0),
        Connector.face("front_face", kw["cord_d"] / 2),
    )
    return res


@skill(
    "shaft_key",
    category="retaining",
    connectors=("seat", "top", "mid"),
    params=[
        p("width", float, 4.0, min=0.5, max=60),
        p("height", float, 4.0, min=0.5, max=60),
        p("length", float, 14.0, min=1, max=400),
        p("style", str, "parallel", choices=("parallel", "woodruff", "gib")),
    ],
)
def shaft_key(**kw) -> SkillResult:
    """DIN 6885 parallel key, or a Woodruff half-moon."""
    w, h, L = kw["width"], kw["height"], kw["length"]
    with BuildPart() as bp:
        if kw["style"] == "woodruff":
            add(revolve_profile_solid([[L, 0.0], [L, w]], bore_d=0.0, arc=140.0))
        else:
            Box(L, w, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
            try:
                fillet(bp.edges().filter_by(Axis.Z), min(w, h) * 0.18)
            except Exception:
                pass

    res = SkillResult(bp.part, interfaces={"width": w, "height": h, "length": L})
    res.add(
        Connector.face("seat", 0.0, outward=False, doc="sits in the shaft keyway"),
        Connector.face("top", h),
        Connector.at("mid", z=h / 2),
    )
    return res


@skill(
    "dowel_pin",
    category="retaining",
    connectors=("end_a", "end_b", "mid"),
    params=[
        p("d", float, 4.0, min=0.4, max=60),
        p("length", float, 16.0, min=1, max=400),
        p("style", str, "solid", choices=("solid", "roll", "split")),
    ],
)
def dowel_pin(**kw) -> SkillResult:
    """ISO 2338 dowel / roll pin — the locating feature bolts cannot provide."""
    d, L = kw["d"], kw["length"]
    bore = d * 0.55 if kw["style"] in ("roll", "split") else 0.0
    arc = 340.0 if kw["style"] == "split" else 360.0
    steps = [[d * 0.92, 0.0], [d, d * 0.14], [d, L - d * 0.14], [d * 0.92, L]]
    solid = revolve_profile_solid(steps, bore_d=bore, arc=arc)
    res = SkillResult(solid, interfaces={"od": d, "length": L})
    res.add(
        Connector.face("end_a", 0.0, outward=False),
        Connector.face("end_b", L),
        Connector.at("mid", z=L / 2),
    )
    return res
