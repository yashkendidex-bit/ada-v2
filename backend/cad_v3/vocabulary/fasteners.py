"""Fasteners. ISO-dimensioned, with a real helical thread available on demand.

`thread_detail` is the load-bearing LOD control. A 12-turn swept ISO thread cost
3.9 s in the capability spike; an 80-bolt assembly cannot afford 80 of those, so
`cosmetic` (turned annular grooves, correct pitch and depth) is the default and
`real` is opt-in for hero parts and for the STEP handed to a machinist.
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
    Helix,
    Location,
    Locations,
    Mode,
    Plane,
    Polyline,
    RegularPolygon,
    add,
    chamfer,
    extrude,
    make_face,
    revolve,
    sweep,
)

from ..frames import Connector, SkillResult
from ..registry import p, skill
from ._util import METRIC, metric, revolve_profile_solid

DETAIL = ("none", "cosmetic", "real")


def _thread_solid(major_d: float, pitch: float, length: float, z0: float = 0.0):
    """ISO 68-1 60-degree thread as a swept helix. h = 0.6134 x pitch."""
    h = 0.6134 * pitch
    r_maj = major_d / 2.0
    r_path = r_maj - h * 0.625
    with BuildPart() as bp:
        with BuildLine():
            Helix(pitch=pitch, height=length, radius=r_path)
        with BuildSketch(Plane(origin=(r_path, 0, 0), z_dir=(0, 1, 0))):
            with BuildLine():
                Polyline(
                    (-h * 0.625, -pitch / 2),
                    (h * 0.375, -pitch * 0.125),
                    (h * 0.375, pitch * 0.125),
                    (-h * 0.625, pitch / 2),
                    close=True,
                )
            make_face()
        sweep(is_frenet=True)
    part = bp.part
    return part.moved(Location((0, 0, z0))) if z0 else part


def _cosmetic_thread(part_ctx_major_d: float, pitch: float, length: float, z0: float):
    """Turned annular grooves at the correct pitch — cheap, and reads as a thread."""
    h = 0.6134 * pitch
    turns = max(int(length / pitch), 1)
    with BuildPart() as bp:
        for i in range(turns):
            z = z0 + pitch * (i + 0.5)
            with BuildSketch(Plane.XZ):
                with Locations((part_ctx_major_d / 2 - h * 0.32, z)):
                    from build123d import Rectangle

                    Rectangle(h * 0.7, pitch * 0.55)
            revolve(axis=Axis.Z)
    return bp.part


def _threaded_shank(major_d: float, pitch: float, length: float, detail: str, z0: float = 0.0):
    core = revolve_profile_solid([[major_d - 2 * 0.6134 * pitch * 0.625, z0],
                                  [major_d - 2 * 0.6134 * pitch * 0.625, z0 + length]])
    if detail == "none":
        return revolve_profile_solid([[major_d, z0], [major_d, z0 + length]])
    with BuildPart() as bp:
        add(core)
        if detail == "real":
            add(_thread_solid(major_d, pitch, length, z0))
        else:
            add(revolve_profile_solid([[major_d, z0], [major_d, z0 + length]]))
            add(_cosmetic_thread(major_d, pitch, length, z0), mode=Mode.SUBTRACT)
    return bp.part


@skill(
    "screw",
    category="fasteners",
    connectors=("head_face", "seat", "tip", "shank_mid"),
    params=[
        p("size", str, "M6", choices=tuple(METRIC)),
        p("length", float, 20.0, min=2, max=400, doc="shank length under the head"),
        p("head", str, "socket", choices=("socket", "hex", "button", "flat", "none")),
        p("thread_detail", str, "cosmetic", choices=DETAIL),
        p("thread_len", float, 0.0, min=0, max=400, doc="0 = fully threaded"),
    ],
)
def screw(**kw) -> SkillResult:
    """ISO metric cap screw. Z=0 is the bearing face under the head."""
    m = metric(kw["size"])
    d, pitch = m["d"], m["pitch"]
    L = kw["length"]
    t_len = min(kw["thread_len"] or L, L)
    plain = L - t_len
    head_h = m["head_h"] if kw["head"] != "flat" else m["head_h"] * 0.6

    with BuildPart() as bp:
        # Shank runs -Z from the bearing face so the head sits at +Z.
        if plain > 0:
            add(revolve_profile_solid([[d, -plain], [d, 0.0]]))
        add(_threaded_shank(d, pitch, t_len, kw["thread_detail"], z0=-L))
        if kw["head"] == "hex":
            with BuildSketch(Plane.XY.offset(0.0)):
                RegularPolygon(m["nut_af"] / 2 / math.cos(math.pi / 6), 6)
            extrude(amount=head_h)
        elif kw["head"] == "flat":
            add(revolve_profile_solid([[d, 0.0], [m["head_d"], head_h], [m["head_d"], head_h]]))
        elif kw["head"] != "none":
            add(revolve_profile_solid([[m["head_d"], 0.0], [m["head_d"], head_h]]))
            if kw["head"] == "button":
                with Locations((0, 0, head_h)):
                    Cylinder(radius=m["head_d"] / 2, height=m["head_d"] * 0.5,
                             align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
            with BuildSketch(Plane.XY.offset(head_h - m["hex_af"] * 0.8)):
                RegularPolygon(m["hex_af"] / 2 / math.cos(math.pi / 6), 6)
            extrude(amount=m["hex_af"], mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={"size": d, "length": L, "clear_d": m["clear"]},
                      meta={"iso": kw["size"], "pitch": pitch})
    res.add(
        Connector.face("seat", 0.0, outward=False, doc="bearing face under the head, mates to a bolt hole"),
        Connector.face("head_face", head_h if kw["head"] != "none" else 0.0, doc="driver end"),
        Connector.face("tip", -L, outward=False, doc="thread tip"),
        Connector.at("shank_mid", z=-L / 2),
    )
    return res


@skill(
    "nut",
    category="fasteners",
    connectors=("front_face", "rear_face", "bore"),
    params=[
        p("size", str, "M6", choices=tuple(METRIC)),
        p("style", str, "hex", choices=("hex", "flange", "nyloc", "round")),
        p("thread_detail", str, "cosmetic", choices=DETAIL),
    ],
)
def nut(**kw) -> SkillResult:
    """ISO hex / flange / nyloc nut."""
    m = metric(kw["size"])
    af, h = m["nut_af"], m["nut_h"]
    if kw["style"] == "nyloc":
        h *= 1.25
    across_corners = af / math.cos(math.pi / 6)

    with BuildPart() as bp:
        if kw["style"] == "round":
            add(revolve_profile_solid([[af, 0], [af, h]], bore_d=m["d"]))
        else:
            with BuildSketch(Plane.XY):
                RegularPolygon(across_corners / 2, 6)
            extrude(amount=h)
            if kw["style"] == "flange":
                add(revolve_profile_solid([[across_corners * 1.18, 0], [across_corners * 1.18, h * 0.22]]))
            Cylinder(radius=m["d"] / 2, height=h * 3,
                     align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
        if kw["thread_detail"] != "none":
            add(_cosmetic_thread(m["d"] + 0.6134 * m["pitch"] * 1.2, m["pitch"], h, 0.0),
                mode=Mode.SUBTRACT)

    res = SkillResult(bp.part, interfaces={"size": m["d"], "bore_d": m["d"], "thickness": h},
                      meta={"iso": kw["size"]})
    res.add(
        Connector.face("rear_face", 0.0, outward=False, doc="bearing face, mates to the joint"),
        Connector.face("front_face", h),
        Connector.at("bore", z=0.0),
    )
    return res


@skill(
    "washer",
    category="fasteners",
    connectors=("front_face", "rear_face", "bore"),
    params=[
        p("size", str, "M6", choices=tuple(METRIC)),
        p("style", str, "plain", choices=("plain", "spring", "shim", "conical")),
        p("thickness", float, 0.0, min=0, max=20, doc="0 = ISO default"),
        p("od", float, 0.0, min=0, max=200, doc="0 = ISO default"),
    ],
)
def washer(**kw) -> SkillResult:
    """Plain / spring / shim washer."""
    m = metric(kw["size"])
    t = kw["thickness"] or m["wash_t"]
    od = kw["od"] or m["wash_od"]
    if kw["style"] == "conical":
        steps = [[od, 0.0], [od * 0.98, t * 0.4], [m["clear"] * 1.6, t]]
    else:
        steps = [[od, 0.0], [od, t]]

    solid = revolve_profile_solid(steps, bore_d=m["clear"])
    res = SkillResult(solid, interfaces={"bore_d": m["clear"], "od": od, "thickness": t},
                      meta={"iso": kw["size"]})
    res.add(
        Connector.face("rear_face", 0.0, outward=False),
        Connector.face("front_face", t),
        Connector.at("bore", z=0.0),
    )
    return res


@skill(
    "stud",
    category="fasteners",
    connectors=("end_a", "end_b", "mid"),
    params=[
        p("size", str, "M8", choices=tuple(METRIC)),
        p("length", float, 45.0, min=4, max=600),
        p("thread_len", float, 14.0, min=1, max=600, doc="threaded length at each end"),
        p("thread_detail", str, "cosmetic", choices=DETAIL),
    ],
)
def stud(**kw) -> SkillResult:
    """Double-ended stud — the correct fastener for a turbine housing."""
    m = metric(kw["size"])
    L, tl = kw["length"], min(kw["thread_len"], kw["length"] / 2 - 0.5)
    with BuildPart() as bp:
        add(revolve_profile_solid([[m["d"] * 0.96, 0.0], [m["d"] * 0.96, L]]))
        add(_threaded_shank(m["d"], m["pitch"], tl, kw["thread_detail"], z0=0.0))
        add(_threaded_shank(m["d"], m["pitch"], tl, kw["thread_detail"], z0=L - tl))

    res = SkillResult(bp.part, interfaces={"size": m["d"], "length": L}, meta={"iso": kw["size"]})
    res.add(
        Connector.face("end_a", 0.0, outward=False),
        Connector.face("end_b", L),
        Connector.at("mid", z=L / 2),
    )
    return res
