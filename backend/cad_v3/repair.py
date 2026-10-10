"""Layer 4 — the closed loop. A typed failure becomes a spec patch, not a code edit.

The premise of the engine is that the brain never writes geometry, so it must
never *repair* geometry either. Everything here narrows a kernel-level or
verifier-level failure down to the one spec line that caused it and expresses the
fix as a `SpecPatch`: change a param, move a mate, add a mate. Most classes of
failure need no model call at all — a range violation has exactly one correct
fix, and a misspelled connector has exactly one close match. Only failures that
need design *intent* are handed up to the brain, and even then the answer comes
back as patches, so the model still never touches code or coordinates.

Zero-to-CAD's headline result (~40% first-pass, ~90% after repair) is why this
layer exists: the first spec is a hypothesis and the kernel is the experiment.
"""

from __future__ import annotations

import copy
import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import assembler, registry, verify
from .schema import BANNED_PARAM_KEYS, FITS, MATE_KINDS, ORIENTS, MateSpec, Spec

PATCH_OPS = (
    "set_param", "del_param", "set_mate", "add_mate", "del_mate",
    "set_count", "set_skill", "set_material", "set_datum",
)

# Optional decorations, in the order they are worth sacrificing. When a skill
# throws, one of these is usually what handed OCCT a self-crossing wire, and none
# of them changes what the part *is* — so dropping one is a legal repair.
DECORATIONS = (
    "gasket_groove", "oil_gallery", "water_jacket", "oil_groove_count",
    "bead_count", "pad_count", "fin_count", "dish", "pivots", "splitter_count",
    "port_count", "chamfer_mm", "fillet_mm", "duct_stub_len",
)

# Params that buy radial room for whatever a part encloses, best first.
GROW_KNOBS = ("tip_clearance", "inner_d", "rotor_d", "bore_d")
# Params that make a part smaller, when nothing around it can be grown.
SHRINK_KNOBS = ("tip_d", "od", "outer_d", "d", "vane_t", "blade_t")

# A mate whose fit is one of these is *meant* to overlap; a severe clash there is
# a dimensioning error, not a placement error, so the offset is the wrong knob.
TIGHT_FITS = ("press", "interference")

# Rounds are cheap (one worker cache spans them all) but not free, and a spec that
# is not converging by the fourth pass needs the brain, not another nudge.
MAX_ROUNDS = 4


@dataclass
class SpecPatch:
    """One minimal, typed edit to a spec — the brain's entire write surface.

    Deliberately flat so a model can emit it as one JSON object per line. There is
    no op that can introduce a coordinate: `set_param` refuses banned keys, and
    position is still reachable only through mates.
    """

    op: str
    part: str = ""          # target part id, for part ops
    a: str = ""             # "part_id.connector", for mate ops
    b: str = ""
    key: str = ""           # param name, or which mate field to set
    value: Any = None
    reason: str = ""

    @property
    def fingerprint(self) -> tuple:
        """Two patches with the same fingerprint touch the same spec cell."""
        return (self.op, self.part, self.a, self.b, self.key)

    def __str__(self) -> str:
        target = self.part or (f"{self.a}>{self.b}" if self.a or self.b else "?")
        head = f"{self.op} {target}" + (f".{self.key}" if self.key else "")
        if self.value is not None:
            head += f" = {self.value!r}"
        return head + (f"   # {self.reason}" if self.reason else "")

    def to_dict(self) -> dict:
        d = {"op": self.op}
        for k in ("part", "a", "b", "key", "reason"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.value is not None:
            d["value"] = self.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SpecPatch":
        op = str(d.get("op", "")).strip()
        value = d.get("value")
        if op == "add_mate" and not isinstance(value, dict):
            # Allow the natural spelling: mate fields inline instead of nested.
            value = {k: d[k] for k in ("kind", "offset_mm", "angle_deg", "fit", "orient")
                     if k in d}
        return cls(
            op=op,
            part=str(d.get("part", "")),
            a=str(d.get("a", "")),
            b=str(d.get("b", "")),
            key=str(d.get("key", "")),
            value=value,
            reason=str(d.get("reason", "")),
        )


def _find_mate(spec: Spec, a: str, b: str) -> MateSpec | None:
    """Locate a mate by connector refs, falling back to the part pair.

    Matching on refs first keeps a patch stable when mates are reordered; the
    part-pair fallback means the brain can say "the mate between X and Y" without
    having to reproduce both connector names exactly.
    """
    for m in spec.mates:
        if m.a == a and m.b == b:
            return m
    pa, pb = MateSpec.split(a)[0], MateSpec.split(b)[0]
    if not pa or not pb:
        return None
    for m in spec.mates:
        if {m.part_a, m.part_b} == {pa, pb}:
            return m
    return None


def _clamp(pd: registry.ParamDef, value: Any) -> Any:
    if pd.type in (int, float):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return pd.default
        if pd.min is not None:
            v = max(v, float(pd.min))
        if pd.max is not None:
            v = min(v, float(pd.max))
        return int(round(v)) if pd.type is int else round(v, 4)
    if pd.choices and value not in pd.choices:
        near = difflib.get_close_matches(str(value), [str(c) for c in pd.choices], 1, 0.4)
        return near[0] if near else pd.default
    return value


def apply_patch(spec: Spec, patch: SpecPatch) -> tuple[Spec, str]:
    """Apply one patch to a copy of `spec`. Returns (new_spec, error) — '' is OK.

    Every rejection path returns the spec unchanged, so a bad patch from the brain
    costs one line of feedback and never corrupts a working spec.
    """
    if patch.op not in PATCH_OPS:
        return spec, f"unknown op '{patch.op}' (expected one of {list(PATCH_OPS)})"
    out = copy.deepcopy(spec)

    if patch.op in ("set_param", "del_param", "set_count", "set_skill", "set_material"):
        target = out.part(patch.part)
        if target is None:
            return spec, f"no part '{patch.part}'"
        sk = registry.get(target.skill)
        if patch.op == "del_param":
            target.params.pop(patch.key, None)
        elif patch.op == "set_param":
            if patch.key.lower() in BANNED_PARAM_KEYS:
                return spec, f"'{patch.key}' is a coordinate; position comes from mates only"
            if sk is None or patch.key not in sk.params:
                legal = sorted(sk.params) if sk else []
                return spec, f"skill '{target.skill}' has no param '{patch.key}' (has {legal})"
            target.params[patch.key] = _clamp(sk.params[patch.key], patch.value)
        elif patch.op == "set_count":
            try:
                target.count = max(1, int(patch.value))
            except (TypeError, ValueError):
                return spec, f"count must be an integer, got {patch.value!r}"
        elif patch.op == "set_skill":
            if registry.get(patch.value) is None:
                return spec, f"'{patch.value}' is not a registered skill"
            target.skill = str(patch.value)
        else:
            target.material = str(patch.value)
        return out, ""

    if patch.op == "set_datum":
        if patch.value not in out.ids:
            return spec, f"'{patch.value}' is not a part id"
        out.datum = str(patch.value)
        return out, ""

    if patch.op == "add_mate":
        for ref in (patch.a, patch.b):
            pid, conn = MateSpec.split(ref)
            if pid not in out.ids or not conn:
                return spec, f"'{ref}' must be part_id.connector with a known part"
        fields = patch.value if isinstance(patch.value, dict) else {}
        out.mates.append(MateSpec.from_dict({"a": patch.a, "b": patch.b, **fields}))
        return out, ""

    m = _find_mate(out, patch.a, patch.b)
    if m is None:
        return spec, f"no mate joins '{patch.a}' and '{patch.b}'"
    if patch.op == "del_mate":
        out.mates.remove(m)
        return out, ""

    key = patch.key or "offset_mm"
    if key in ("offset_mm", "angle_deg"):
        try:
            setattr(m, key, float(patch.value))
        except (TypeError, ValueError):
            return spec, f"{key} must be a number, got {patch.value!r}"
    elif key == "fit":
        if patch.value not in FITS:
            return spec, f"fit '{patch.value}' invalid (one of {list(FITS)})"
        m.fit = str(patch.value)
    elif key == "kind":
        if patch.value not in MATE_KINDS:
            return spec, f"kind '{patch.value}' invalid (one of {list(MATE_KINDS)})"
        m.kind = str(patch.value)
    elif key == "orient":
        if patch.value not in ORIENTS:
            return spec, f"orient '{patch.value}' invalid (one of {list(ORIENTS)})"
        m.orient = str(patch.value)
    elif key in ("a", "b"):
        pid, conn = MateSpec.split(str(patch.value))
        if pid not in out.ids or not conn:
            return spec, f"'{patch.value}' must be part_id.connector with a known part"
        setattr(m, key, str(patch.value))
    else:
        return spec, f"a mate has no field '{key}'"
    return out, ""


def apply_all(spec: Spec, patches: list[SpecPatch]) -> tuple[Spec, list[SpecPatch], list[str]]:
    """Apply patches in order, skipping duplicates. Returns (spec, applied, errors)."""
    seen: set[tuple] = set()
    applied: list[SpecPatch] = []
    errors: list[str] = []
    for patch in patches:
        if patch.fingerprint in seen:
            continue
        seen.add(patch.fingerprint)
        spec, err = apply_patch(spec, patch)
        if err:
            errors.append(f"{patch}  ->  REJECTED: {err}")
        else:
            applied.append(patch)
    return spec, applied, errors


@dataclass
class Diagnosis:
    """One failure, its cause in spec terms, and the candidate fixes, best first."""

    code: str
    where: str
    detail: str
    patches: list[SpecPatch] = field(default_factory=list)
    question: str = ""

    @property
    def auto(self) -> bool:
        return bool(self.patches)

    def __str__(self) -> str:
        head = f"[{self.code}] {self.where}: {self.detail}"
        if self.patches:
            return head + f"\n    fix: {self.patches[0]}"
        return head + (f"\n    ask: {self.question}" if self.question else "")


def _near(word: str, options) -> str:
    hit = difflib.get_close_matches(str(word), [str(o) for o in options], 1, 0.35)
    return hit[0] if hit else ""


def _param_name(message: str) -> str:
    m = re.search(r"param '([A-Za-z_][A-Za-z0-9_]*)'", message)
    return m.group(1) if m else ""


def _unknown_params(message: str) -> list[str]:
    m = re.search(r"unknown params \[(.*?)\]", message)
    if not m:
        return []
    return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]


def _radius(bb) -> float:
    """Distance from the assembly axis to the furthest corner of a bounding box.

    Half the box width only equals a radius for a part centred on the axis, which a
    bolt at a 42 mm PCD is not. Absolute extents get both cases right.
    """
    return max(abs(bb.min.X), abs(bb.max.X), abs(bb.min.Y), abs(bb.max.Y))


def _span(inst) -> tuple[float, float, float]:
    """(radius from the axis, axial height, box volume) of an instance."""
    try:
        bb = inst.solid.bounding_box()
        return (_radius(bb), max(bb.size.Z, 0.1),
                max(bb.size.X * bb.size.Y * bb.size.Z, 1e-6))
    except Exception:
        return (1.0, 1.0, 1.0)


def _overlap_reach(ia, ib) -> float:
    """How far out the shared solid actually reaches.

    This is the one measurement that pins a rotor/shroud foul exactly: the common
    volume starts at the shroud's current clearance radius and ends at whatever the
    blade tip really does, curvature and all. Reading it off the boolean beats every
    estimate from the parts' own bounding boxes, which is why it is worth the second
    intersect after the verifier already did one.
    """
    try:
        common = ia.solid.intersect(ib.solid)
    except Exception:
        return 0.0
    if common is None:
        return 0.0
    lumps = common if isinstance(common, (list, tuple)) else [common]
    try:
        lumps = list(lumps)
    except TypeError:
        lumps = [common]
    out = 0.0
    for lump in lumps:
        try:
            out = max(out, _radius(lump.bounding_box()))
        except Exception:
            continue
    return out



# ── is a ring seat aiming its children along the axis instead of out of it? ──
# The failure this answers looked, from every angle the loop could see, like a sizing
# error: a limb hanging off a hub fouled a part 300 mm away, so the clash rules grew a
# bore and shrank a diameter and never touched the cause, which was that the seat the
# limb sat on faced up the shaft. `orient` on the mate is the coordinate-free way to
# say so, and this is the evidence that justifies spending a round on it.
#
# The evidence is measured, never named. A limb that belongs in the disc reaches OUT;
# one that has been stood up reaches ALONG. So take the ring's own axis, project the
# clashing part's box onto it, and compare its axial extent with both the ring radius
# and its own standoff from the ring station. A bolt on a bolt circle — the case that
# must never be flagged — is shorter than the circle it stands on and fails the first
# test outright, whichever way it points.

def _axis_frame(inst) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(origin, unit +Z) of an instance's own frame, in world coordinates."""
    loc = inst.solid.location
    p = loc.position
    t = loc.wrapped.Transformation()
    k = (t.Value(1, 3), t.Value(2, 3), t.Value(3, 3))
    n = math.sqrt(sum(c * c for c in k)) or 1.0
    return ((float(p.X), float(p.Y), float(p.Z)), (k[0] / n, k[1] / n, k[2] / n))


def _extent_about(inst, origin, k) -> tuple[float, float]:
    """(extent along `k`, furthest distance from the axis) of an instance's box."""
    try:
        bb = inst.solid.bounding_box()
        lo, hi = bb.min, bb.max
    except Exception:
        return (0.0, 0.0)
    lows, highs = (float(lo.X), float(lo.Y), float(lo.Z)), (float(hi.X), float(hi.Y), float(hi.Z))
    a_lo, a_hi, r_max = math.inf, -math.inf, 0.0
    for i in range(8):
        pt = tuple(highs[j] if (i >> j) & 1 else lows[j] for j in range(3))
        d = (pt[0] - origin[0], pt[1] - origin[1], pt[2] - origin[2])
        a = d[0] * k[0] + d[1] * k[1] + d[2] * k[2]
        a_lo, a_hi = min(a_lo, a), max(a_hi, a)
        r_max = max(r_max, math.sqrt(max(d[0] ** 2 + d[1] ** 2 + d[2] ** 2 - a * a, 0.0)))
    return (a_hi - a_lo, r_max)


def _orient_patch(spec: Spec, asm, inst) -> SpecPatch | None:
    """A mate above `inst` whose ring seat is standing it up rather than laying it out.

    Walks the placement chain rather than looking at one mate, because the seat that
    aimed a part is often two or three links above the part that ended up in the wrong
    place: the blade fouls the swashplate, but the mate to fix is the grip's.
    """
    seen: set[str] = set()
    cur = inst
    while cur is not None and cur.key not in seen:
        seen.add(cur.key)
        idx = getattr(cur, "via_mate", -1)
        host = asm.by_key(getattr(cur, "via_host", "") or "")
        if idx < 0 or idx >= len(spec.mates) or host is None:
            return None
        mate = spec.mates[idx]
        conn = host.result.connectors.get(getattr(cur, "via_seat", ""))
        if (conn is not None and mate.orient == "seat" and conn.count > 1
                and conn.layout.startswith("ring") and conn.direction.startswith("axial")):
            station = conn.location(0).position
            ring_r = math.hypot(float(station.X), float(station.Y))
            origin, k = _axis_frame(host)
            along, across = _extent_about(inst, origin, k)
            if along > max(ring_r, 1.0) and along > 1.5 * max(across - ring_r, 0.1):
                return SpecPatch(
                    "set_mate", a=mate.a, b=mate.b, key="orient", value="radial",
                    reason=f"'{cur.pid}' sits on a {conn.layout} seat facing "
                           f"{conn.direction}, and the chain it carries reaches "
                           f"{along:.0f} mm along that axis on a {ring_r:.0f} mm ring "
                           f"— lay it out instead of standing it up")
        cur = host
    return None


def _radial_delta(volume: float, inner) -> float:
    """Radial clearance that would undo an overlap of `volume`.

    The overlap of two coaxial parts is an annular band, so V ~ 2*pi*r*h*delta.
    Inverting that gives a first guess rather than a fudge factor, and the loop
    re-measures anyway — but starting from the physics means one round usually does
    it instead of five. Rounded up to 0.05 mm so a patch is never a no-op.
    """
    r, h, _ = _span(inner)
    delta = volume / max(2.0 * math.pi * max(r, 0.5) * h, 1e-6)
    return min(max(math.ceil(delta * 20.0) / 20.0, 0.05), 3.0)


def _axial_delta(volume: float, inner) -> float:
    """Axial push that would undo an overlap of `volume` across a mate."""
    r, _, _ = _span(inner)
    delta = volume / max(math.pi * max(r, 0.5) ** 2, 1e-6)
    return min(max(math.ceil(delta * 20.0) / 20.0, 0.05), 8.0)


def _knob(spec: Spec, pid: str, names) -> tuple[str, float] | None:
    """First param from `names` that the part's skill actually declares."""
    part = spec.part(pid)
    sk = registry.get(part.skill) if part else None
    if sk is None:
        return None
    for name in names:
        if name in sk.params:
            pd = sk.params[name]
            if pd.type not in (int, float):
                continue
            current = part.params.get(name, pd.default)
            try:
                return name, float(current)
            except (TypeError, ValueError):
                continue
    return None


def _path_part(spec: Spec, path: str):
    """Resolve a SpecError path to a PartSpec. Validators index, the assembler names."""
    m = re.search(r"parts\[(.+?)\]", path)
    if not m:
        return None
    token = m.group(1)
    if token.isdigit() and int(token) < len(spec.parts):
        return spec.parts[int(token)]
    return spec.part(token)


def _path_mate(spec: Spec, path: str) -> MateSpec | None:
    m = re.search(r"mates\[(\d+)\]", path)
    if not m:
        return None
    i = int(m.group(1))
    return spec.mates[i] if i < len(spec.mates) else None


def _seat_hosts(spec: Spec, want: int) -> list[tuple[str, str]]:
    """Parts publishing a patterned connector whose station count is `want`.

    Rather than hard-coding which param drives which ring, this looks for any int
    param on the candidate skill that already equals the pattern count. It is a
    heuristic, but a cheap and checkable one: the suggestion is re-validated by the
    next build, and a wrong guess costs one round.
    """
    hits: list[tuple[str, str]] = []
    for other in spec.parts:
        sk = registry.get(other.skill)
        if sk is None:
            continue
        matches = False
        for pname, pd in sk.params.items():
            if pd.type is not int:
                continue
            try:
                if int(other.params.get(pname, pd.default) or 0) == want:
                    matches = True
                    break
            except (TypeError, ValueError):
                continue
        if not matches:
            continue
        for conn in sk.connectors:
            if "circle" in conn or "ring" in conn or conn == "seats":
                hits.append((other.id, conn))
    return hits


def _diagnose_spec(spec: Spec, errors) -> list[Diagnosis]:
    """Turn validator and assembler errors into patches. No geometry needed."""
    out: list[Diagnosis] = []
    for err in errors:
        part = _path_part(spec, err.path)
        mate = _path_mate(spec, err.path)
        side = err.path.rsplit(".", 1)[-1] if err.path.endswith((".a", ".b")) else ""
        patches: list[SpecPatch] = []
        question = ""

        if err.code == "UNKNOWN_SKILL" and part is not None:
            near = _near(part.skill, registry.names())
            if near:
                patches.append(SpecPatch("set_skill", part=part.id, value=near,
                                         reason=f"'{part.skill}' is not in the vocabulary"))
            else:
                question = f"no skill resembles '{part.skill}'; pick from the catalogue"

        elif err.code == "BAD_PARAM" and part is not None:
            sk = registry.get(part.skill)
            for gone in _unknown_params(err.message):
                near = _near(gone, sorted(sk.params) if sk else [])
                if near and near not in part.params:
                    patches.append(SpecPatch("set_param", part=part.id, key=near,
                                             value=part.params.get(gone),
                                             reason=f"'{gone}' renamed to '{near}'"))
                patches.append(SpecPatch("del_param", part=part.id, key=gone,
                                         reason=f"'{gone}' is not a param of {part.skill}"))
            name = _param_name(err.message)
            if name and sk and name in sk.params:
                # A range or type violation has exactly one correct repair: the nearest
                # legal value. `apply_patch` would clamp anyway, but the patch is also
                # printed and shown to the brain, so it carries the clamped number
                # rather than the rejected one.
                fixed = _clamp(sk.params[name], part.params.get(name))
                patches.append(SpecPatch("set_param", part=part.id, key=name,
                                         value=fixed,
                                         reason="nearest value the signature allows"))

        elif err.code == "COORDINATE_IN_PARAMS" and part is not None:
            for key in sorted(set(k for k in part.params if k.lower() in BANNED_PARAM_KEYS)):
                patches.append(SpecPatch("del_param", part=part.id, key=key,
                                         reason="position comes from mates only"))

        elif err.code == "UNKNOWN_CONNECTOR" and mate is not None and side:
            pid, conn = MateSpec.split(getattr(mate, side))
            target = spec.part(pid)
            sk = registry.get(target.skill) if target else None
            near = _near(conn, sk.connectors if sk else [])
            if near:
                patches.append(SpecPatch("set_mate", a=mate.a, b=mate.b, key=side,
                                         value=f"{pid}.{near}",
                                         reason=f"'{conn}' -> '{near}'"))

        elif err.code in ("UNKNOWN_PART", "UNKNOWN_DATUM"):
            token = err.message.split("'")[1] if "'" in err.message else ""
            near = _near(token, spec.ids)
            if near and err.code == "UNKNOWN_DATUM":
                patches.append(SpecPatch("set_datum", value=near, reason=f"'{token}' -> '{near}'"))
            elif near and mate is not None and side:
                _, conn = MateSpec.split(getattr(mate, side))
                patches.append(SpecPatch("set_mate", a=mate.a, b=mate.b, key=side,
                                         value=f"{near}.{conn}", reason=f"'{token}' -> '{near}'"))

        elif err.code == "MISSING_DATUM" and spec.parts:
            patches.append(SpecPatch("set_datum", value=spec.parts[0].id,
                                     reason="the first part anchors the assembly"))

        elif err.code == "BAD_FIT" and mate is not None:
            patches.append(SpecPatch("set_mate", a=mate.a, b=mate.b, key="fit",
                                     value=_near(mate.fit, FITS) or "slip", reason="legal fit"))

        elif err.code == "BAD_KIND" and mate is not None:
            patches.append(SpecPatch("set_mate", a=mate.a, b=mate.b, key="kind",
                                     value=_near(mate.kind, MATE_KINDS) or "rigid",
                                     reason="legal mate kind"))

        elif err.code == "BAD_ORIENT" and mate is not None:
            patches.append(SpecPatch("set_mate", a=mate.a, b=mate.b, key="orient",
                                     value=_near(mate.orient, ORIENTS) or "seat",
                                     reason="legal orientation"))

        elif err.code == "BAD_COUNT" and part is not None:
            patches.append(SpecPatch("set_count", part=part.id, value=1, reason="count >= 1"))

        elif err.code == "SELF_MATE" and mate is not None:
            patches.append(SpecPatch("del_mate", a=mate.a, b=mate.b,
                                     reason="a part cannot position itself"))

        elif err.code == "PATTERN_WITHOUT_SEAT" and part is not None:
            hosts = _seat_hosts(spec, part.count)
            if hosts:
                pid, conn = hosts[0]
                sk = registry.get(part.skill)
                seat = "end_a" if sk and "end_a" in sk.connectors else (
                    "rear_face" if sk and "rear_face" in sk.connectors else
                    (sk.connectors[0] if sk else "rear_face"))
                patches.append(SpecPatch("add_mate", a=f"{pid}.{conn}", b=f"{part.id}.{seat}",
                                         value={"fit": "slip"},
                                         reason=f"{part.count} seats on {pid}.{conn}"))
            else:
                question = (f"'{part.id}' has count={part.count} but no part publishes a "
                            f"{part.count}-station ring to seat it on")

        elif err.code == "FLOATING_PARTS":
            question = (f"{err.message} — each needs a mate onto the chain that reaches "
                        f"datum '{spec.datum}'; choose the face it really sits against")

        elif err.code == "SKILL_FAILED" and part is not None:
            sk = registry.get(part.skill)
            for key in DECORATIONS:
                if key in part.params and (sk is None or key in sk.params):
                    patches.append(SpecPatch("del_param", part=part.id, key=key,
                                             reason="kernel rejected the profile; drop the decoration"))
                    break
            if not patches:
                question = (f"'{part.id}' ({part.skill}) will not build: {err.message}. "
                            f"Re-proportion its params so the profile stays monotonic.")

        elif err.code == "MATE_FAILED" and mate is not None:
            question = f"mate {mate.a} -> {mate.b} could not be solved: {err.message}"

        elif err.code in ("DUPLICATE_ID", "MISSING_ID", "EMPTY", "MISSING_CONNECTOR", "UNPLACED"):
            question = f"{err.message} ({err.hint})" if err.hint else err.message

        out.append(Diagnosis(err.code, err.path, err.message, patches, question))
    return out


FASTENERS = ("bolt", "screw", "nut", "washer", "stud", "dowel_pin", "rivet", "circlip")


def _is_fastener(spec: Spec, pid: str) -> bool:
    part = spec.part(pid)
    if part is None:
        return False
    sk = registry.get(part.skill)
    return part.skill in FASTENERS or (sk is not None and sk.category == "fastener")


def _mate_between(spec: Spec, pa: str, pb: str) -> MateSpec | None:
    for m in spec.mates:
        if {m.part_a, m.part_b} == {pa, pb}:
            return m
    return None


def _radial_target(spec: Spec, outer_pid: str, knob: str, inner, reach: float = 0.0,
                   margin: float = 0.3) -> float:
    """The value a clearance knob needs, measured rather than inferred.

    Inverting the overlap volume badly underestimates a rotor/shroud foul: eleven
    nozzle vanes touching a wheel share only a few percent of the circumference, so
    V/(2*pi*r*h) reports hundredths of a millimetre for an interference of one
    whole millimetre, and the loop then creeps at the floor step for ever.

    So the requirement is read off the solids instead. `reach` is how far the shared
    volume itself extends from the axis, which is exactly where the enclosed part
    stops being clear; the enclosed part's own bounding radius is the fallback when
    the boolean could not be taken. Bounding box, not vertices, because a curved
    blade bulges past every vertex it has (the impeller reaches 31.8 mm while its
    furthest vertex is at 29.58).
    """
    need = max(reach, _span(inner)[0]) + margin
    part = spec.part(outer_pid)
    if knob.endswith("clearance") and part is not None:
        try:
            rotor_r = float(part.params.get("rotor_d", 0.0) or 0.0) / 2.0
        except (TypeError, ValueError):
            rotor_r = 0.0
        return round(max(need - rotor_r, margin), 3)
    return round(need * 2.0, 3)


def _diagnose_clash(spec: Spec, asm, clash) -> Diagnosis:
    """One severe clash -> the spec cell that owns the space it happened in."""
    ia, ib = asm.by_key(clash.a), asm.by_key(clash.b)
    if ia is None or ib is None:
        return Diagnosis("CLASH", f"{clash.a} <-> {clash.b}",
                         f"{clash.volume:.2f} mm3 overlap", [],
                         "the clashing instances are no longer in the assembly")
    pa, pb = ia.pid, ib.pid
    where = f"{clash.a} <-> {clash.b}"
    detail = (f"{clash.volume:.2f} mm3 of shared solid; fit '{clash.fit}' "
              f"allows {clash.allowed:.2f} mm3")
    mate = _mate_between(spec, pa, pb)
    excess = clash.volume - clash.allowed if clash.fit in TIGHT_FITS else clash.volume
    excess = max(excess, clash.volume * 0.2)

    # Before reaching for a diameter: is either side of this clash there because a
    # patterned seat aimed it along the axis? No amount of resizing cures that, and it
    # is the one cause whose fix is a mate field rather than a param.
    for side in (ia, ib):
        aim = _orient_patch(spec, asm, side)
        if aim is not None:
            return Diagnosis("CLASH_ORIENT", where, detail, [aim])

    # A fastener inside a plate is never a clearance problem — the hole is either
    # the wrong size, on the wrong circle, or not drilled through. Guessing at a
    # bore diameter here would move the wrong part, so this case is named instead.
    if _is_fastener(spec, pa) != _is_fastener(spec, pb):
        fast, plate = (pa, pb) if _is_fastener(spec, pa) else (pb, pa)
        fp, pp = spec.part(fast), spec.part(plate)
        patches: list[SpecPatch] = []
        size = fp.params.get("size") if fp else None
        psk = registry.get(pp.skill) if pp else None
        if size and psk and "bolt_size" in psk.params and pp.params.get("bolt_size") != size:
            patches.append(SpecPatch("set_param", part=plate, key="bolt_size", value=size,
                                     reason=f"hole must match the {size} fastener"))
        return Diagnosis("CLASH_FASTENER", where, detail, patches,
                         "" if patches else
                         f"'{fast}' passes through '{plate}': check that {plate}'s bolt_pcd, "
                         f"bolt_count and bolt_size match the circle it is seated on")

    if mate is not None and mate.fit not in TIGHT_FITS:
        # Mated and meant to clear: the mate's own offset is the honest knob, and
        # its sign is already "along the host seat's outward normal", so growing the
        # magnitude always separates the pair whichever way the stack runs.
        inner = ia if _span(ia)[0] <= _span(ib)[0] else ib
        delta = _axial_delta(excess, inner)
        base = mate.offset_mm
        value = round(base + math.copysign(delta, base if base else 1.0), 3)
        return Diagnosis("CLASH_STACK", where, detail,
                         [SpecPatch("set_mate", a=mate.a, b=mate.b, key="offset_mm",
                                    value=value,
                                    reason=f"open the joint by {delta:.2f} mm")])

    # Coaxial interference: whichever part reaches further out is the container, so
    # it is the one with room to give. Radius, not bounding-box volume — a squat
    # wide ring can enclose a tall narrow wheel.
    outer, inner = (ia, ib) if _span(ia)[0] >= _span(ib)[0] else (ib, ia)
    reach = _overlap_reach(ia, ib)
    patches = []
    knob = _knob(spec, outer.pid, GROW_KNOBS)
    if knob:
        name, current = knob
        # A step floor of 0.15 mm keeps the round from being a no-op when the measured
        # requirement lands exactly on the current value, which is what a just-touching
        # rotor does: the target says "already clear" while the kernel says otherwise.
        want = max(_radial_target(spec, outer.pid, name, inner, reach),
                   round(current + 0.15, 3))
        if want > current:
            patches.append(SpecPatch("set_param", part=outer.pid, key=name, value=want,
                                     reason=f"clear '{inner.pid}' out to "
                                            f"r={max(reach, _span(inner)[0]):.2f}"))
    shrink = _knob(spec, inner.pid, SHRINK_KNOBS)
    if shrink:
        name, current = shrink
        delta = _radial_delta(excess, inner)
        patches.append(SpecPatch("set_param", part=inner.pid, key=name,
                                 value=round(current - delta * 2.0, 3),
                                 reason=f"or take {delta:.2f} mm off '{inner.pid}'"))
    return Diagnosis("CLASH_RADIAL", where, detail, patches,
                     "" if patches else
                     f"'{outer.pid}' and '{inner.pid}' overlap but neither publishes a "
                     f"clearance param; re-proportion one of them")


def _world_r(inst, conn, station: int = 0) -> tuple[float, float] | None:
    """(radius from the machine's axis, height along it) for a seat frame, in world.

    The datum is never moved, so the assembly's axis is the world Z: a seat 120 mm out is
    120 mm out whether it got there from a bolt circle on its host or from the host itself
    standing on one. Measuring in world rather than on the part's own frame is what makes
    the two shapes of the same mistake one test instead of two.
    """
    try:
        p = (inst.solid.location * conn.location(station)).position
    except Exception:
        return None
    return math.hypot(float(p.X), float(p.Y)), float(p.Z)


def _onaxis_alternatives(host, seat_name: str) -> list[str]:
    """Seats on the same host that sit on the axis, best first; [] if this is not it.

    Nothing is named and nothing is guessed: a station's radius is arithmetic on its own
    frame. Where more than one seat qualifies the order is the one that moves the chain
    least — same facing first, then the seat nearest along the axis to where it already
    sits — and the loop pays for them one round at a time, so a second candidate costs a
    round rather than a wrong answer. Earlier this returned a name only when exactly one
    candidate existed, which on a real rotor head meant three faces and therefore silence.
    """
    conns = getattr(host.result, "connectors", {}) or {}
    cur = conns.get(seat_name)
    if cur is None:
        return []
    here = _world_r(host, cur)
    if here is None or here[0] < 1e-6:
        return []                         # already on the axis; not this failure
    picks: list[tuple[int, float, str]] = []
    for name, c in conns.items():
        if name == seat_name or c.count > 1:
            continue
        there = _world_r(host, c)
        if there is None or there[0] >= 1e-6:
            continue
        picks.append((0 if c.direction == cur.direction else 1,
                      abs(there[1] - here[1]), name))
    return [name for _dir, _dz, name in sorted(picks)][:3]


def _pattern_artefact(spec: Spec, asm, pid: str) -> int:
    """The mate that placed one whole part on one station of a ring, or -1.

    A part with count=1 mated to a seat of N stations — or to one instance of an N-count
    part — is positioned on station 0, a radius off the axis. What the machine means is
    that all N carry it and it is centred; the spec has no way to say "all N", so that
    pose is an artefact of the pattern rather than a statement about the part, and every
    other coincidence declared about it is then violated by exactly that radius. Counts
    and radii only: a bolt on a bolt circle is count=N and is never touched, and a single
    part hung off a single off-axis boss is a real design, not this.
    """
    part = spec.part(pid)
    if part is None or max(1, part.count) != 1:
        return -1
    inst = next(iter(asm.of_part(pid)), None)
    idx = getattr(inst, "via_mate", -1) if inst else -1
    host = asm.by_key(getattr(inst, "via_host", "") or "") if inst else None
    if host is None or not 0 <= idx < len(spec.mates):
        return -1
    seat = (getattr(host.result, "connectors", {}) or {}).get(getattr(inst, "via_seat", ""))
    hpart = spec.part(host.pid)
    if seat is None or not (seat.count > 1 or (hpart is not None and hpart.count > 1)):
        return -1
    here = _world_r(host, seat)
    return idx if here is not None and here[0] > 1e-6 else -1



def _still_connected(spec: Spec, drop: int) -> bool:
    """Would every part still have a mate path to the datum without `mates[drop]`?

    Asked before proposing a deletion, so the repair never trades a violated coincidence
    for a floating part. Cheap: it is the same reachability the validator runs.
    """
    from . import graph

    edges = [m.edge for i, m in enumerate(spec.mates) if i != drop]
    try:
        return not graph.unreachable(spec.datum, spec.ids, edges)
    except Exception:
        return False


def _loop_patches(spec: Spec, asm, pid: str) -> list[SpecPatch]:
    """How to make a stated coincidence true, best first.

    A violated closure means two paths reach one part and disagree, so the repair is
    always to stop trusting the wrong path. Two ways, in order of how much they assume:

    1. If the path that placed the part is a pattern artefact — one whole part positioned
       on station 0 of a ring — then *delete* that mate. The closure stops being redundant
       and becomes the mate that places the part, which makes the coincidence true by
       construction rather than by arithmetic, and the part lands centred where the other
       path always said it was. Only offered when the graph survives it.
    2. Otherwise walk up the chain to the first mate that hangs it off an off-axis station
       and re-point that one field at the host's own on-axis face. It is one field of one
       mate — no pose, no coordinate — and everything below follows rigidly.
    """
    out: list[SpecPatch] = []
    idx = _pattern_artefact(spec, asm, pid)
    if idx >= 0 and _still_connected(spec, idx):
        m = spec.mates[idx]
        out.append(SpecPatch(
            "del_mate", a=m.a, b=m.b,
            reason=f"'{pid}' is one part sitting on one station of a ring, which puts it a "
                   f"radius off the axis; drop this and the mate that is violated places "
                   f"it instead, centred, which is what the ring was carrying"))
    seen: set[str] = set()
    cur = next(iter(asm.of_part(pid)), None)
    while cur is not None and cur.key not in seen:
        seen.add(cur.key)
        idx = getattr(cur, "via_mate", -1)
        host = asm.by_key(getattr(cur, "via_host", "") or "")
        if idx < 0 or idx >= len(spec.mates) or host is None:
            break
        seat = getattr(cur, "via_seat", "")
        mate = spec.mates[idx]
        ref = f"{host.pid}.{seat}"
        key = "a" if mate.a == ref else ("b" if mate.b == ref else "")
        part = spec.part(cur.pid)
        # N parts on an N-station ring are off the axis on purpose, and re-hosting them
        # would stack all N on one point — which `_check_pattern` would then report as a
        # collapse. Only a single part can be in the wrong place this way. Keep walking
        # though: the ring itself may be what is mis-hosted, and that is one link up.
        single = part is not None and max(1, part.count) == 1
        for alt in (_onaxis_alternatives(host, seat) if key and single else []):
            out.append(SpecPatch(
                "set_mate", a=mate.a, b=mate.b, key=key, value=f"{host.pid}.{alt}",
                reason=f"'{cur.pid}' hangs off '{ref}', a station off the axis, and the "
                       f"chain below it inherits that; '{host.pid}.{alt}' is on the axis"))
        if len(out) > 1:
            break
        cur = host
    return out



def diagnose(spec: Spec, asm, rep=None) -> list[Diagnosis]:
    """Every failure in one build, typed and paired with its candidate patches."""
    out = _diagnose_spec(spec, asm.errors)
    if rep is None:
        return out

    for key in rep.invalid:
        inst = asm.by_key(key)
        part = spec.part(inst.pid) if inst else None
        patches = []
        if part is not None:
            sk = registry.get(part.skill)
            for name in DECORATIONS:
                if name in part.params and (sk is None or name in sk.params):
                    patches.append(SpecPatch("del_param", part=part.id, key=name,
                                             reason="OCCT rejects the solid; drop the decoration"))
                    break
        out.append(Diagnosis("INVALID_SOLID", key, "failed the B-rep validity check",
                             patches,
                             "" if patches else f"'{key}' builds an invalid solid at these params"))

    for note in rep.notes:
        if "zero volume" in note:
            out.append(Diagnosis("ZERO_VOLUME", note.split()[0],
                                 "the skill returned an empty solid", [],
                                 "its params cancel out — check that no diameter or "
                                 "length collapses to zero"))

    for pid, (n, stations) in (getattr(rep, "collapsed", None) or {}).items():
        # Two ways to reconcile a count with a seat, and the honest first choice is to
        # give the seat the stations the spec asked for rather than to quietly build
        # fewer parts. Which param drives the ring is not hard-coded: it is the int
        # param on the host that currently equals the station count.
        patches = []
        insts = asm.of_part(pid)
        host = asm.by_key(getattr(insts[0], "via_host", "") if insts else "")
        hostpart = spec.part(host.pid) if host else None
        hsk = registry.get(hostpart.skill) if hostpart else None
        if hsk and hostpart:
            for pname, pd in hsk.params.items():
                if pd.type is not int:
                    continue
                try:
                    if int(hostpart.params.get(pname, pd.default) or 0) != stations:
                        continue
                except (TypeError, ValueError):
                    continue
                patches.append(SpecPatch("set_param", part=hostpart.id, key=pname, value=n,
                                         reason=f"the seat carrying '{pid}' offers {stations} "
                                                f"station(s) for {n} instances"))
                break
        patches.append(SpecPatch("set_count", part=pid, value=stations,
                                 reason=f"or ask for only the {stations} station(s) that exist"))
        out.append(Diagnosis("PATTERN_COLLAPSED", pid,
                             f"{n} instances landed on {stations} station(s)", patches))

    for pid in (getattr(rep, "detached", None) or []):
        insts = asm.of_part(pid)
        idx = getattr(insts[0], "via_mate", -1) if insts else -1
        patches = []
        if 0 <= idx < len(spec.mates) and abs(spec.mates[idx].offset_mm) > 1e-9:
            m = spec.mates[idx]
            patches.append(SpecPatch("set_mate", a=m.a, b=m.b, key="offset_mm", value=0.0,
                                     reason=f"an offset of {m.offset_mm:g} mm carried '{pid}' "
                                            f"clear of everything; sit it on the face"))
        out.append(Diagnosis("DETACHED", pid,
                             "positioned, but its box clears every other part in the assembly",
                             patches,
                             "" if patches else
                             f"'{pid}' touches nothing — mate it to the part it physically "
                             f"sits against, or size it so it reaches"))

    for idx, pair, mm in (getattr(rep, "misaligned", None) or []):
        # The spec stated a coincidence the traversal could not solve, and measurement
        # says it does not hold. Nothing about a diameter can cure that, so this is
        # diagnosed before the clash it causes — which on a rotor head was 300 mm away
        # against the shaft, and had three rounds of bore-growing spent on it.
        loop = next((lc for lc in getattr(asm, "loop_mates", []) if lc.idx == idx), None)
        patches = []
        for pid in ((loop.moved_id, loop.host_id) if loop else ()):
            patches += _loop_patches(spec, asm, pid)
            if patches:
                break
        out.append(Diagnosis(
            "LOOP_MISALIGNED", f"mates[{idx}]",
            f"{pair} are declared coincident but the chain leaves them {mm:.1f} mm apart",
            patches,
            "" if patches else
            f"two paths reach {pair} and they disagree — re-host whichever mate placed "
            f"it onto the part it really touches, or drop the mate that is not "
            f"physically there"))

    out += [_diagnose_clash(spec, asm, c) for c in rep.severe_clashes]
    return out


# ── the loop ────────────────────────────────────────────────────────────────

@dataclass
class Attempt:
    round: int
    spec: Spec
    asm: Any
    report: Any
    diagnoses: list[Diagnosis] = field(default_factory=list)
    applied: list[SpecPatch] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    source: str = "deterministic"

    @property
    def score(self) -> tuple[int, int, float, int]:
        """Lower is better. Spec errors, then broken solids, then how far past the
        fits the metal goes, and only then how many places it does so.

        A collapsed pattern and a detached part count with the broken solids, not with
        the clashes: both mean a part is not where the design says it is, which is worth
        more than any amount of tolerable overlap.

        Ranking by *unexplained* volume — each clash measured against the allowance its
        own fit grants — before counting clashes is not a cosmetic ordering. Count-first
        says two parts passing straight through each other are better than five press
        fits five per cent over allowance, which is backwards, and it is not a
        hypothetical: on a rotor head the loop found the one correct fix (lay four
        blades out into the disc plane instead of standing them up the mast), traded a
        59 cm3 interpenetration for four 180 mm3 press overruns, saw 2 clashes become 5,
        and threw the fix away. Severity is how far past the fit you are, not that you
        are past it.
        """
        rep = self.report
        return (
            len(self.asm.errors),
            (len(rep.invalid) + len(rep.unplaced) + len(rep.collapsed)
             + len(rep.detached) + len(rep.misaligned)) if rep else 99,
            round(sum(max(c.volume - c.allowed, 0.0)
                      for c in rep.severe_clashes), 3) if rep else 1e9,
            len(rep.severe_clashes) if rep else 99,
        )

    @property
    def clean(self) -> bool:
        errors, broken, _excess, clashes = self.score
        return not (errors or broken or clashes)


@dataclass
class RepairResult:
    spec: Spec
    asm: Any
    report: Any
    attempts: list[Attempt] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    @property
    def fixed(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].clean

    @property
    def rounds(self) -> int:
        return len(self.attempts)

    def text(self) -> str:
        lines = []
        for att in self.attempts:
            e, broken, excess, clashes = att.score
            lines.append(f"round {att.round}: {e} spec errors, {broken} broken, "
                         f"{clashes} clashes ({excess:.2f} mm3 past the fits)  "
                         f"[{att.source}]")
            for patch in att.applied:
                lines.append(f"    patch  {patch}")
            for bad in att.rejected:
                lines.append(f"    {bad}")
        lines.append("repaired" if self.fixed else "not fully repaired")
        for q in self.open_questions:
            lines.append(f"  ask: {q}")
        return "\n".join(lines)


def build_and_verify(spec: Spec, *, worker=None, clash: bool = True):
    """One full pass: geometry, placement, then verification on the solids."""
    asm = assembler.build(spec, worker=worker, strict=False)
    rep = verify.verify(asm, clash=clash) if asm.instances else None
    return asm, rep


def _progress(now: Attempt, prev: Attempt, *, gain: float = 0.05) -> bool:
    """Did `prev`'s patches buy enough to justify paying for the same cells again?

    Any-improvement is the wrong test, because a knob that is not the cause still
    shaves a rounding error off the overlap every time it moves. On a rotor head one
    bore was grown 0.15 mm in three consecutive rounds, taking a 51 cm3
    interpenetration down by two parts in a thousand — the loop was measurably
    "improving" while the actual cause, a ring hung off an off-axis bolt circle, went
    unexamined. So the test is relative and scale-free: the tiers that count parts must
    improve, or the unexplained volume must fall by a real fraction of itself.
    """
    ne, nb, nx, nc = now.score
    pe, pb, px, pc = prev.score
    if (ne, nb) != (pe, pb):
        return (ne, nb) < (pe, pb)
    if nx <= px * (1.0 - gain):
        return True
    return nx <= px and nc < pc


def repair(spec: Spec, *, rounds: int = MAX_ROUNDS, clash: bool = True,
           ask: Callable[[Spec, list[Diagnosis]], list[SpecPatch]] | None = None,
           log: Callable[[str], None] | None = None) -> RepairResult:
    """Build, verify, patch, repeat — keeping the best spec seen.

    One `Worker` spans every round, so a patch that touches one part's params
    invalidates one cache entry and re-runs one skill; the other twenty solids are
    reused. That is what makes a repair round cost a second instead of a rebuild.

    `ask` is the escape hatch to the brain: it is only consulted for the rounds
    where deterministic rules produced nothing, and it must answer in patches.
    """
    worker = assembler.Worker()
    attempts: list[Attempt] = []
    current = copy.deepcopy(spec)
    best: Attempt | None = None
    # Cells that have already been paid for and bought nothing. The deterministic rules
    # size a knob from the overlap they can see, which is the right move when the knob
    # is the cause and useless when it is not: on a rotor head the same bore was grown
    # 0.15 mm in three consecutive rounds, each time removing a hundredth of a 51 cm3
    # overlap that came from the ring being mounted off-axis in the first place. Three
    # of four rounds spent proving the same thing. So a patch whose round did not
    # improve the score retires its cell, and the next-best fix — or the model — gets
    # the round instead.
    spent: set[tuple] = set()

    for n in range(1, max(1, rounds) + 1):
        asm, rep = build_and_verify(current, worker=worker, clash=clash)
        att = Attempt(n, current, asm, rep, diagnose(current, asm, rep))
        if attempts and not _progress(att, attempts[-1]):
            spent.update(p.fingerprint for p in attempts[-1].applied)
        attempts.append(att)
        if log:
            log(f"round {n}: score={att.score}")
        if best is None or att.score < best.score:
            best = att
        if att.clean or n == rounds:
            break

        candidates: list[SpecPatch] = []
        for d in att.diagnoses:
            if not d.auto:
                continue
            # Diagnoses carry their fixes best first, so a retired cell falls through to
            # the alternative the same diagnosis already proposed rather than to nothing.
            fresh = next((p for p in d.patches if p.fingerprint not in spent), None)
            if fresh is not None:
                candidates.append(fresh)
        source = "deterministic"
        if not candidates and ask is not None:
            candidates = ask(current, att.diagnoses) or []
            source = "brain"
        att.source = source
        if not candidates:
            break
        nxt, applied, rejected = apply_all(current, candidates)
        att.applied, att.rejected = applied, rejected
        if not applied:
            break
        current = nxt

    winner = best or attempts[-1]
    questions = [d.question for a in attempts for d in a.diagnoses if d.question]
    return RepairResult(winner.spec, winner.asm, winner.report, attempts,
                        sorted(set(questions)))


PATCH_GRAMMAR = """Answer with a JSON array of patches and nothing else. Legal ops:
{"op":"set_param","part":"<id>","key":"<param>","value":<number|string|bool>,"reason":"..."}
{"op":"del_param","part":"<id>","key":"<param>"}
{"op":"set_count","part":"<id>","value":<int>}
{"op":"set_skill","part":"<id>","value":"<skill>"}
{"op":"set_material","part":"<id>","value":"<material>"}
{"op":"set_mate","a":"<id.connector>","b":"<id.connector>","key":"offset_mm|angle_deg|fit|kind|orient|a|b","value":<...>}
{"op":"add_mate","a":"<id.connector>","b":"<id.connector>","fit":"slip","offset_mm":0.0}
{"op":"del_mate","a":"<id.connector>","b":"<id.connector>"}
{"op":"set_datum","value":"<id>"}
No other keys. No coordinates: x/y/z/rx/ry/rz/position/transform are rejected.
`orient` is "seat" (honour the seat frame), "radial" (turn the child to face out of
the axis) or "axial" (along it) — the fix when a part is on the right seat pointing
the wrong way, which no diameter can cure.
Prefer the smallest edit that removes the cause. One patch per failure."""


def repair_prompt(spec: Spec, diagnoses: list[Diagnosis], *, limit: int = 12) -> str:
    """The brain-facing repair request: only the failures and only the spec they touch.

    Sending the whole spec back on every round is what makes closed-loop repair
    expensive, so this quotes just the parts and mates named by the diagnoses.
    """
    open_ = [d for d in diagnoses if not d.auto][:limit]
    ids: set[str] = set()
    for d in open_:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f"{d.where} {d.detail} {d.question}"):
            if token in spec.ids:
                ids.add(token)
    parts = [p for p in spec.parts if p.id in ids]
    mates = [m for m in spec.mates if m.part_a in ids or m.part_b in ids]
    import json as _json

    lines = [f"Assembly '{spec.name}' failed verification. Repair it by patching the spec.", ""]
    lines += ["FAILURES:"] + [str(d) for d in open_] + [""]
    if parts:
        lines += ["THE PARTS INVOLVED:"]
        lines += [_json.dumps({"id": p.id, "skill": p.skill, "count": p.count,
                               "params": p.params}) for p in parts]
        lines.append("")
    if mates:
        lines += ["THE MATES INVOLVED:"]
        lines += [_json.dumps({"a": m.a, "b": m.b, "fit": m.fit,
                               "offset_mm": m.offset_mm, "orient": m.orient}) for m in mates]
        lines.append("")
    if parts:
        lines += ["SIGNATURES:"] + [registry.signature_text(p.skill) for p in parts] + [""]
    lines.append(PATCH_GRAMMAR)
    return "\n".join(lines)


def parse_patches(text: str) -> list[SpecPatch]:
    """Pull a patch array out of a model reply, fenced or bare."""
    import json as _json

    body = text.strip()
    if "```" in body:
        chunks = [c for c in body.split("```") if "{" in c]
        body = max(chunks, key=len) if chunks else body
        body = body.split("\n", 1)[1] if body.lstrip().startswith("json") else body
    start, end = body.find("["), body.rfind("]")
    if start >= 0 and end > start:
        try:
            data = _json.loads(body[start:end + 1])
            return [SpecPatch.from_dict(d) for d in data if isinstance(d, dict)]
        except ValueError:
            pass
    out: list[SpecPatch] = []
    for line in body.splitlines():           # one object per line, JSONL style
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(SpecPatch.from_dict(_json.loads(line)))
            except ValueError:
                continue
    return out
