"""Connector frames — the only mechanism by which a part gets a position.

A Connector is a named, oriented seat on a part. Skills publish them; the
assembler consumes them. Because a Connector carries orientation as well as
translation, `connect_to` produces face-to-face contact with no sign flips and
no per-skill conventions to remember.

Invariant enforced across the whole vocabulary: **+Z is the functional axis.**
Rotational parts spin about Z, stacked parts stack along Z, mating faces point
outward along their own local +Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from build123d import (
    Axis,
    BallJoint,
    CylindricalJoint,
    LinearJoint,
    Location,
    RevoluteJoint,
    RigidJoint,
    Vector,
)

Z_AXIS = (0.0, 0.0, 1.0)

KINDS = ("rigid", "revolute", "cylindrical", "linear", "ball")


@dataclass
class Connector:
    """A named seat on a part. `locations` has >1 entry for patterned seats.

    `flip` records whether this seat is an *outward-facing surface*. Two outward
    faces that meet must have opposing normals, so the assembler adds a 180 degree
    turn about X when both sides of a mate are flippable. Axial seats (a bore, a
    position on a shaft) are not surfaces and must never flip, or a part threaded
    onto a shaft would end up pointing backwards.
    """

    name: str
    locations: list[Location]
    kind: str = "rigid"
    axis: tuple = Z_AXIS
    doc: str = ""
    flip: bool = False

    @property
    def count(self) -> int:
        return len(self.locations)

    def location(self, index: int = 0) -> Location:
        return self.locations[index % max(1, self.count)]

    # ── constructors ────────────────────────────────────────────────────
    @classmethod
    def at(
        cls,
        name: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        *,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        kind: str = "rigid",
        doc: str = "",
        flip: bool = False,
    ) -> "Connector":
        return cls(name, [Location((x, y, z), (rx, ry, rz))], kind, Z_AXIS, doc, flip)

    @classmethod
    def face(cls, name: str, z: float, *, outward: bool = True, doc: str = "",
             flip: bool = True) -> "Connector":
        """A planar mating face at height z, local +Z pointing out of the part."""
        ry = 0.0 if outward else 180.0
        return cls(name, [Location((0.0, 0.0, z), (0.0, ry, 0.0))], "rigid", Z_AXIS, doc, flip)

    @classmethod
    def polar(
        cls,
        name: str,
        radius: float,
        count: int,
        z: float = 0.0,
        *,
        start_angle: float = 0.0,
        tilt_deg: float = 0.0,
        kind: str = "rigid",
        doc: str = "",
        flip: bool = True,
    ) -> "Connector":
        """A patterned ring of `count` seats — bolt circles, vanes, ball pockets.

        The ring angle and the tilt must be composed in that order. Passing both
        through one Euler triple as `(0, tilt, deg)` makes `deg` a spin about the
        *already tilted* local Z, so a radial seat (`tilt_deg=90`) would keep the
        same axis at every station and only the position would walk the circle —
        every oil hole in the ring would point the same way. Rotating about global
        Z first, then tilting about the new local Y, gives a genuinely radial ring.
        """
        locs: list[Location] = []
        n = max(1, int(count))
        for i in range(n):
            deg = start_angle + 360.0 * i / n
            th = math.radians(deg)
            spin = Location((0.0, 0.0, 0.0), (0.0, 0.0, deg))
            tilt = Location((0.0, 0.0, 0.0), (0.0, tilt_deg, 0.0))
            locs.append(
                Location((radius * math.cos(th), radius * math.sin(th), z)) * spin * tilt
            )
        return cls(name, locs, kind, Z_AXIS, doc, flip)

    @classmethod
    def stack(cls, name: str, z_values: list[float], *, kind: str = "rigid", doc: str = "") -> "Connector":
        """A patterned set of seats along the Z axis — bearing seats, shaft steps."""
        return cls(name, [Location((0.0, 0.0, float(z))) for z in z_values], kind, Z_AXIS, doc, False)

    # ── what direction does this seat face? ─────────────────────────────
    # A seat carries orientation, but until now only the skill author knew which
    # way it pointed. That is the hole every "the part came out sideways" bug fell
    # through: the brain picks a seat by name, the kernel honours its frame, and
    # nothing in between ever states that `bolt_circle` faces along the axis while
    # `vane_ring` faces out of it. Classifying it here — from the frame itself, so
    # it cannot drift from the geometry — lets the catalogue say so out loud.

    def station_direction(self, index: int = 0) -> str:
        """Where station `index` points, in the part's own frame.

        `axial+` / `axial-`  along the part's Z, the stacking direction
        `radial`             out of the axis; `radial_in` back towards it
        `tangential`         across the axis, neither in nor out
        `skew`               a compound angle, e.g. a coned seat
        """
        loc = self.location(index)
        p = loc.position
        x, y, z = float(p.X), float(p.Y), float(p.Z)
        t = loc.wrapped.Transformation()
        vx, vy, vz = t.Value(1, 3), t.Value(2, 3), t.Value(3, 3)
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        if n < 1e-12:
            return "skew"
        vx, vy, vz = vx / n, vy / n, vz / n
        if abs(abs(vz) - 1.0) < 0.02:
            return "axial+" if vz > 0 else "axial-"
        r = math.hypot(x, y)
        if r > 1e-6 and abs(vz) < 0.05:
            dot = (x * vx + y * vy) / r
            if dot > 0.98:
                return "radial"
            if dot < -0.98:
                return "radial_in"
            return "tangential"
        if abs(vz) < 0.05:
            return "tangential"
        return "skew"

    @property
    def direction(self) -> str:
        """The seat's facing, or `mixed` if its stations disagree."""
        seen = {self.station_direction(i) for i in range(self.count)}
        return seen.pop() if len(seen) == 1 else "mixed"

    @property
    def layout(self) -> str:
        """`single`, `ring of N`, or `stack of N` — read off the station positions."""
        if self.count < 2:
            return "single"
        radii = [math.hypot(float(c.position.X), float(c.position.Y))
                 for c in self.locations]
        return f"stack of {self.count}" if max(radii) <= 1e-6 else f"ring of {self.count}"

    @property
    def tag(self) -> str:
        """`bolt_circle(ring of 12, axial+)` — the compact form for the catalogue.

        Deliberately terse: this goes into every authoring request, so the facing has
        to be cheap enough that publishing it for all 31 skills costs less than one
        wasted build.
        """
        bits = [b for b in (self.layout if self.count > 1 else "", self.direction) if b]
        return f"{self.name}({', '.join(bits)})"

    def describe(self) -> str:
        """One line for the catalogue the brain reads."""
        tail = f"  # {self.doc}" if self.doc else ""
        return f"{self.tag}{tail}"


@dataclass
class SkillResult:
    """Everything a skill returns. `interfaces` feeds tolerance/fit checking."""

    solid: Any
    connectors: dict[str, Connector] = field(default_factory=dict)
    interfaces: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def add(self, *connectors: Connector) -> "SkillResult":
        for c in connectors:
            self.connectors[c.name] = c
        return self


def bind(solid: Any, connector: Connector, index: int = 0, label: str | None = None) -> Any:
    """Attach one build123d Joint to `solid` for `connector[index]`.

    Returns the Joint. Idempotent per label — re-binding an existing label is a
    no-op so the assembler can mate the same seat from two directions safely.
    """
    name = label or connector.name
    existing = getattr(solid, "joints", None) or {}
    if name in existing:
        return existing[name]

    loc = connector.location(index)
    axis = Axis(loc.position, Vector(*connector.axis))
    kind = connector.kind

    if kind == "revolute":
        return RevoluteJoint(name, solid, axis=axis)
    if kind == "cylindrical":
        return CylindricalJoint(name, solid, axis=axis)
    if kind == "linear":
        return LinearJoint(name, solid, axis=axis)
    if kind == "ball":
        return BallJoint(name, solid, joint_location=loc)
    return RigidJoint(name, solid, loc)
