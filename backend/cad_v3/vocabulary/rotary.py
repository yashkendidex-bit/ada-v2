"""Gears with real involute flanks. Ported maths, deterministic geometry.

The involute is computed, not approximated by a trapezoid: theta = tan(a) - a at
each radius, which is what makes the flank actually conjugate. Helical gears are
built as a loft through progressively rotated tooth sections, since build123d has
no twist-extrude.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    BuildLine,
    BuildPart,
    BuildSketch,
    Cylinder,
    Location,
    Locations,
    Mode,
    Plane,
    PolarLocations,
    Polyline,
    add,
    extrude,
    loft,
    make_face,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import revolve_profile_solid


def _inv(alpha: float) -> float:
    return math.tan(alpha) - alpha


def _tooth_points(module: float, teeth: int, pa_deg: float, *, internal: bool = False,
                  steps: int = 7) -> list[tuple[float, float]]:
    """One tooth outline in the XY plane, centred on the +X axis."""
    pa = math.radians(pa_deg)
    r_p = module * teeth / 2.0
    r_b = r_p * math.cos(pa)
    r_tip = r_p + module * (1.25 if internal else 1.0)
    r_root = r_p - module * (1.0 if internal else 1.25)
    half = math.pi / (2.0 * teeth) + _inv(math.acos(min(r_b / r_p, 1.0)))

    def ang(r: float) -> float:
        if r <= r_b:
            return half
        return half - _inv(math.acos(min(r_b / r, 1.0)))

    lo = max(r_root, r_b * 0.999)
    radii = [lo + (r_tip - lo) * (i / (steps - 1)) for i in range(steps)]
    right = [(r * math.cos(-ang(r)), r * math.sin(-ang(r))) for r in radii]
    left = [(r * math.cos(ang(r)), r * math.sin(ang(r))) for r in reversed(radii)]
    root_r = max(r_root, 0.2)
    pts = [(root_r * math.cos(-half * 1.02), root_r * math.sin(-half * 1.02))]
    pts += right + left
    pts += [(root_r * math.cos(half * 1.02), root_r * math.sin(half * 1.02))]
    return pts


def _gear_body(module: float, teeth: int, pa: float, width: float, helix_deg: float,
               internal: bool, rim_d: float):
    r_p = module * teeth / 2.0
    r_root = r_p - module * 1.25
    with BuildPart() as bp:
        if internal:
            add(revolve_profile_solid([[rim_d, 0], [rim_d, width]], bore_d=(r_p + module) * 2))
        else:
            add(revolve_profile_solid([[max(r_root, 0.3) * 2, 0], [max(r_root, 0.3) * 2, width]]))

        pts = _tooth_points(module, teeth, pa, internal=internal)
        if abs(helix_deg) < 0.05:
            with BuildPart() as one:
                with BuildSketch(Plane.XY):
                    with BuildLine():
                        Polyline(*pts, close=True)
                    make_face()
                extrude(amount=width)
            tooth = one.part
        else:
            twist = math.degrees(width * math.tan(math.radians(helix_deg)) / r_p)
            n = 5
            with BuildPart() as one:
                for i in range(n):
                    u = i / (n - 1)
                    with BuildSketch(Plane.XY.offset(width * u)) as sk:
                        with Locations(Location((0, 0), twist * u)):
                            with BuildLine():
                                Polyline(*pts, close=True)
                            make_face()
                loft()
            tooth = one.part

        with PolarLocations(0, teeth):
            add(tooth, mode=Mode.SUBTRACT if internal else Mode.ADD)
    return bp.part


@skill(
    "spur_gear",
    category="rotary",
    connectors=("bore", "front_face", "rear_face", "pitch_ring", "hub_face"),
    params=[
        p("module", float, 2.0, min=0.2, max=40),
        p("teeth", int, 24, min=6, max=300),
        p("width", float, 10.0, min=0.4, max=400),
        p("pressure_angle", float, 20.0, min=12, max=30),
        p("helix_deg", float, 0.0, min=-45, max=45, doc="non-zero makes it a helical gear"),
        p("bore_d", float, 12.0, min=0, max=990),
        p("hub_d", float, 0.0, min=0, max=990, doc=">0 adds a projecting hub"),
        p("hub_len", float, 8.0, min=0, max=200),
        p("spoke_count", int, 0, min=0, max=16, doc="lightening spokes/web cutouts"),
        p("internal", bool, False, doc="ring gear with internal teeth"),
        p("rim_d", float, 0.0, min=0, max=1990, doc="outer rim diameter for internal gears"),
    ],
)
def spur_gear(**kw) -> SkillResult:
    """Involute spur / helical / internal gear."""
    module, teeth = kw["module"], kw["teeth"]
    r_p = module * teeth / 2.0
    w = kw["width"]
    rim_d = kw["rim_d"] or (r_p + module * 4) * 2

    with BuildPart() as bp:
        add(_gear_body(module, teeth, kw["pressure_angle"], w, kw["helix_deg"],
                       kw["internal"], rim_d))
        if kw["hub_d"] > 0 and kw["hub_len"] > 0:
            add(revolve_profile_solid([[kw["hub_d"], w], [kw["hub_d"], w + kw["hub_len"]]]))
        if kw["spoke_count"] > 0 and not kw["internal"] and kw["bore_d"] > 0:
            r_mid = (kw["bore_d"] / 2 + r_p) / 2
            hole_r = max((r_p - kw["bore_d"] / 2) * 0.24, 0.6)
            with PolarLocations(r_mid, kw["spoke_count"]):
                Cylinder(radius=hole_r, height=w * 3,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
        if kw["bore_d"] > 0:
            Cylinder(radius=kw["bore_d"] / 2, height=(w + kw["hub_len"]) * 3,
                     align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

    res = SkillResult(
        bp.part,
        interfaces={"bore_d": kw["bore_d"], "pitch_d": r_p * 2, "width": w,
                    "module": module, "teeth": teeth},
        meta={"pitch_d": r_p * 2, "tip_d": (r_p + module) * 2, "internal": kw["internal"]},
    )
    res.add(
        Connector.at("bore", z=0.0, kind="revolute", doc="shaft axis"),
        Connector.face("rear_face", 0.0, outward=False),
        Connector.face("front_face", w),
        Connector.face("hub_face", w + kw["hub_len"] if kw["hub_d"] > 0 else w),
        Connector.polar("pitch_ring", r_p, max(teeth, 1), w / 2, tilt_deg=90.0,
                        doc="pitch circle — mate to another gear's pitch_ring to mesh"),
    )
    return res
