"""Dimension assertions — measuring the built model against what was claimed.

This is the check the engine did not have, and its absence is the whole reason a
turbocharger with no recognisable turbine wheel reported PASS. Every other check
asks whether the assembly is self-consistent: watertight solids, no unexplained
overlaps, everything connected. A wrong model satisfies all of those, because
wrongness is not an inconsistency — it is a disagreement with the drawing, and
nothing was holding a drawing.

So a spec now states its own key dimensions and this module measures them on the
finished B-rep. `210 mm from the bed datum to the cross-slide, +-0.2` is either
true of the solids or it is not, and no amount of plausible geometry can fake it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import materials, xform
from .schema import DimSpec, MateSpec

ASSEMBLY = ("", "assembly", "ASSEMBLY", "*")


@dataclass
class DimResult:
    dim: DimSpec
    have: float = 0.0
    ok: bool = False
    measurable: bool = True
    note: str = ""

    @property
    def error(self) -> float:
        return self.have - float(self.dim.value)

    def text(self) -> str:
        if not self.measurable:
            return f"  ?    {self.dim.label} — {self.note}"
        mark = "ok  " if self.ok else "FAIL"
        tail = f"   {self.dim.note}" if self.dim.note else ""
        return (f"  {mark} {self.dim.label} = {self.have:.3f} "
                f"(want {self.dim.value:.3f} +-{self.dim.tol:g}, "
                f"off by {self.error:+.3f}){tail}")


def evaluate(asm, dims: list[DimSpec] | None = None) -> list[DimResult]:
    todo = asm.spec.expect.dims if dims is None else dims
    return [_one(asm, d) for d in todo]


def _one(asm, dim: DimSpec) -> DimResult:
    try:
        have, note = _measure(asm, dim)
    except Exception as exc:
        return DimResult(dim, measurable=False, note=f"{type(exc).__name__}: {exc}")
    if have is None:
        return DimResult(dim, measurable=False, note=note or "not measurable")
    return DimResult(dim, have=float(have), ok=abs(float(have) - float(dim.value)) <= float(dim.tol))


def _measure(asm, dim: DimSpec) -> tuple[float | None, str]:
    kind = dim.measure
    if kind in ("mass_kg", "volume_cm3"):
        return _mass_or_volume(asm, dim)
    if kind in ("length_x", "length_y", "length_z"):
        box = _box_of(asm, dim.of)
        if box is None:
            return (None, f"'{dim.of}' has no geometry")
        k = "xyz".index(kind[-1])
        return (box[k + 3] - box[k], "")
    if kind == "diameter":
        return _diameter(asm, dim.of)
    if kind == "gap":
        pa, pb = _box_of(asm, dim.of), _box_of(asm, dim.to)
        if pa is None or pb is None:
            return (None, f"'{dim.of}' or '{dim.to}' has no geometry")
        return (_box_gap(pa, pb), "")

    fa = asm.frame_of(dim.of)
    if fa is None:
        return (None, f"'{dim.of}' has no pose")
    Ra, ta = xform.loc_to_Rt(fa)
    if kind in ("x", "y", "z") and not dim.to:
        return (float(ta["xyz".index(kind)]), "")
    fb = asm.frame_of(dim.to)
    if fb is None:
        return (None, f"'{dim.to}' has no pose")
    Rb, tb = xform.loc_to_Rt(fb)
    d = tb - ta
    if kind in ("x", "y", "z"):
        # Signed, because a drawing's 87.1 mm is to one side and the sign is the
        # difference between a working machine and a mirror image of one.
        return (float(d["xyz".index(kind)]), "")
    if kind == "distance":
        return (float(np.linalg.norm(d)), "")
    if kind == "distance_xy":
        return (float(math.hypot(d[0], d[1])), "")
    if kind == "axial":
        return (float(d @ Ra[:, 2]), "")
    if kind == "radial":
        return (float(np.linalg.norm(d - Ra[:, 2] * float(d @ Ra[:, 2]))), "")
    if kind == "angle":
        cos = max(-1.0, min(1.0, float(Ra[:, 2] @ Rb[:, 2])))
        return (math.degrees(math.acos(cos)), "")
    return (None, f"'{kind}' is not implemented")


def _mass_or_volume(asm, dim: DimSpec) -> tuple[float | None, str]:
    ref = str(dim.of)
    insts = asm.instances if ref in ASSEMBLY else asm.of_part(MateSpec.split(ref)[0])
    if not insts:
        return (None, f"'{ref}' names no instance")
    vol = 0.0
    mass = 0.0
    for inst in insts:
        try:
            v = float(inst.solid.volume)
        except Exception:
            continue
        vol += v
        mass += materials.mass_g(inst.material, v)
    if dim.measure == "mass_kg":
        return (mass / 1000.0, "")
    return (vol / 1000.0, "")


def _diameter(asm, ref: str) -> tuple[float | None, str]:
    """An outside diameter: the published interface if there is one, else the box."""
    pid = MateSpec.split(str(ref))[0]
    insts = asm.of_part(pid)
    if insts:
        face = insts[0].result.interfaces or {}
        for key in ("od", "d", "outer_d", "rim_d"):
            if float(face.get(key, 0.0) or 0.0) > 0.0:
                return (float(face[key]), "")
    box = _box_of(asm, ref)
    if box is None:
        return (None, f"'{ref}' has no geometry")
    return (max(box[3] - box[0], box[4] - box[1]), "measured from the bounding box")


def _box_of(asm, ref: str) -> tuple[float, ...] | None:
    """The union bounding box of every instance of the part `ref` names."""
    pid = MateSpec.split(str(ref))[0]
    insts = asm.instances if str(ref) in ASSEMBLY else asm.of_part(pid)
    lo = [1e18] * 3
    hi = [-1e18] * 3
    found = False
    for inst in insts:
        try:
            bb = inst.solid.bounding_box()
        except Exception:
            continue
        found = True
        for k, (a, b) in enumerate(((bb.min.X, bb.max.X), (bb.min.Y, bb.max.Y),
                                    (bb.min.Z, bb.max.Z))):
            lo[k] = min(lo[k], float(a))
            hi[k] = max(hi[k], float(b))
    return tuple(lo + hi) if found else None


def _box_gap(p, q) -> float:
    dx = max(p[0] - q[3], q[0] - p[3], 0.0)
    dy = max(p[1] - q[4], q[1] - p[4], 0.0)
    dz = max(p[2] - q[5], q[2] - p[5], 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def bbox(asm) -> tuple[float, ...] | None:
    """The whole assembly's bounding box, for the envelope check."""
    return _box_of(asm, "")
