"""The SPEC contract — the entire interface between the brain and the kernel.

Deliberately dependency-free (no pydantic): the validators here produce error
messages shaped for the repair loop, which is more valuable than generic ones.

The load-bearing design choice used to be stated as "there is no coordinate field
anywhere in this schema", and it cost the engine every assembly whose parts are
*not* stacked end to end. A mate could slide a part along the seat it landed on
and spin it about that seat's normal, and nothing else — so a carriage sitting
210 mm along the bed, 87.1 mm across it and 25 mm up could not be described at
all. That is not anti-hallucination; that is a missing degree of freedom.

What actually prevents a hallucinated pose is not the absence of numbers, it is
the absence of a *global frame* to put them in. So the rule is now sharper and
strictly more expressive: **no field in this schema is ever measured from world
origin.** Every offset is relative to something the spec names — a seat on a
part, or a declared datum. `dx/dy/dz/rx/ry/rz` on a mate are read in the host
seat's own frame, and a datum is only ever positioned relative to another datum
or to a part. A model still cannot place a part in space; it can only place a
part *with respect to* something else, which is what an engineer does too.

Three further things arrived with that change:

* `vars` — named dimensions with equations between them (see expr.py), so a
  tooth count and a pin circle can no longer disagree silently.
* `datums` — named reference frames. A machine with a bed, a spindle line and
  three stations along it is described the way its drawing describes it.
* `expect` / `checks` — what the finished assembly must measure. Without these
  the verifier can only ask whether the model is *self-consistent*, which is how
  a turbocharger with no recognisable turbine passed every check it had.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import expr

FITS = ("slip", "clearance", "press", "interference", "free")
MATE_KINDS = ("rigid", "revolute", "cylindrical", "linear", "ball")

# How the child is turned once it has been brought to the seat. `seat` honours the
# seat frame exactly, which is what every mate did before this existed and is right
# whenever the seat already points where the child should go. The other two are the
# coordinate-free way to say "that seat faces the wrong way for this part": they
# rotate the child relative to the seat it is already on, so no pose is authored and
# the assembler still derives everything. This is the lever a repair round needs —
# a part that came out along the axis when it should have reached out of it is not a
# sizing error and cannot be fixed by changing a diameter.
ORIENTS = ("seat", "radial", "axial")

# Two solids in face contact must have opposing normals, so the assembler adds a
# 180 degree turn when both sides of a mate are real surfaces. `auto` is that rule.
# The overrides exist because the rule is a heuristic about what a seat *means*,
# and a repair round that has measured the result should be able to overrule it.
FLIP_MODES = ("auto", "yes", "no")

# Which station of a patterned seat receives instance i. `world` sorts the host's
# stations by their angle in world space, so two co-axial bolt circles line up no
# matter how the parts carrying them were mated; `local` keeps the seat's own
# declaration order, which is what an asymmetric vane ring wants.
STATION_ORDERS = ("world", "local")

DATUM_KINDS = ("frame", "plane", "axis", "point")
WORLD = "WORLD"
FRAME_PREFIX = "@"

# What a dimension assertion can measure between two anchors.
MEASURES = ("x", "y", "z", "distance", "distance_xy", "axial", "radial",
            "angle", "length_x", "length_y", "length_z", "gap",
            "mass_kg", "volume_cm3", "diameter")

CHECK_KINDS = ("planetary", "gear_mesh", "bolt_circle", "fit", "coaxial",
               "stack_length", "count", "clearance")

# Keys that would smuggle a coordinate into a params block. A param is a
# *dimension of a part*, never a position of one, so these stay banned even now
# that mates carry offsets: the offsets live on the mate, in a named frame.
BANNED_PARAM_KEYS = {
    "x", "y", "z", "rx", "ry", "rz", "sx", "sy", "sz",
    "position", "transform", "origin", "location", "translate",
    "rotate", "rotation", "offset_x", "offset_y", "offset_z", "pose",
}


@dataclass
class SpecError:
    path: str
    code: str
    message: str
    hint: str = ""

    def __str__(self) -> str:
        tail = f"  -> {self.hint}" if self.hint else ""
        return f"[{self.code}] {self.path}: {self.message}{tail}"


# ── expression-carrying numeric fields ──────────────────────────────────────
# Any numeric field may arrive as a formula. The formula text is kept in `expr`
# and the field itself holds the resolved number, so a spec round-trips through
# to_dict with its equations intact rather than being flattened to constants the
# first time it is read.

def _num(d: dict, key: str, default: float, exprs: dict[str, str]) -> float:
    val = d.get(key, default)
    if val is None:
        return float(default)
    if expr.is_expression(val):
        exprs[key] = str(val)
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        exprs[key] = str(val)
        return float(default)


class HasExpressions:
    """Mixin: resolve this object's formulas against a variable scope."""

    expr: dict[str, str]

    def bake(self, scope: dict[str, float]) -> list[str]:
        errs: list[str] = []
        for key, text in self.expr.items():
            value, err = expr.number(text, scope, where=key)
            if err:
                errs.append(err)
                continue
            current = getattr(self, key, 0.0)
            setattr(self, key, int(round(value)) if isinstance(current, int)
                    and not isinstance(current, bool) else value)
        return errs

    def unbake(self, d: dict) -> dict:
        for key, text in self.expr.items():
            d[key] = text
        d.pop("expr", None)
        return d


@dataclass
class PartSpec:
    id: str
    skill: str
    params: dict[str, Any] = field(default_factory=dict)
    material: str = "steel"
    count: int = 1
    label: str = ""
    # Which seat this part is measured from when a dimension assertion or a layout
    # point names it. Defaults to the part origin, which is right for a body of
    # revolution and wrong for a bracket whose datum is one of its faces.
    anchor: str = ""
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "PartSpec":
        return cls(
            id=str(d.get("id", "")),
            skill=str(d.get("skill", "")),
            params=dict(d.get("params", {}) or {}),
            material=str(d.get("material", "steel")),
            count=int(d.get("count", 1) or 1),
            label=str(d.get("label", "")),
            anchor=str(d.get("anchor", "") or ""),
            note=str(d.get("note", "") or ""),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"id": self.id, "skill": self.skill, "params": self.params}
        if self.material != "steel":
            out["material"] = self.material
        if self.count != 1:
            out["count"] = self.count
        for key in ("label", "anchor", "note"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out

    def baked(self, scope: dict[str, float]) -> dict[str, Any]:
        """Params with every formula replaced by its number, for the vocabulary."""
        return expr.bake(self.params, scope)


@dataclass
class MateSpec(HasExpressions):
    """One statement of the form "this seat of A meets that seat of B, so".

    `offset_mm` and `angle_deg` are kept as the shorthand they always were —
    offset along the seat normal, spin about it — and are simply added to `dz`
    and `rz`. Every one of the six numbers is read in the *host seat's* frame,
    which is what keeps a spec free of world coordinates while still letting it
    say "and 87.1 mm across".
    """

    a: str  # "part_id.connector", "part_id", or "@datum_id"
    b: str
    kind: str = "rigid"
    offset_mm: float = 0.0
    angle_deg: float = 0.0
    fit: str = "slip"
    orient: str = "seat"
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    flip: str = "auto"
    station_order: str = "world"
    stations: list[int] = field(default_factory=list)
    # Whether the global solver is allowed to move this mate's residual. A mate
    # marked False is a statement about intent that the solver treats as a
    # measurement instead of a constraint — useful for a nominal dimension that
    # the real geometry cannot satisfy exactly.
    solve: bool = True
    note: str = ""
    expr: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MateSpec":
        e: dict[str, str] = {}
        return cls(
            a=str(d.get("a", "")),
            b=str(d.get("b", "")),
            kind=str(d.get("kind", "rigid")),
            offset_mm=_num(d, "offset_mm", 0.0, e),
            angle_deg=_num(d, "angle_deg", 0.0, e),
            fit=str(d.get("fit", "slip")),
            orient=str(d.get("orient", "seat") or "seat"),
            dx=_num(d, "dx", 0.0, e),
            dy=_num(d, "dy", 0.0, e),
            dz=_num(d, "dz", 0.0, e),
            rx=_num(d, "rx", 0.0, e),
            ry=_num(d, "ry", 0.0, e),
            rz=_num(d, "rz", 0.0, e),
            flip=str(d.get("flip", "auto") or "auto").lower(),
            station_order=str(d.get("station_order", "world") or "world").lower(),
            stations=[int(v) for v in (d.get("stations") or [])],
            solve=bool(d.get("solve", True)),
            note=str(d.get("note", "") or ""),
            expr=e,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"a": self.a, "b": self.b, "kind": self.kind}
        for key, default in (("offset_mm", 0.0), ("angle_deg", 0.0),
                             ("dx", 0.0), ("dy", 0.0), ("dz", 0.0),
                             ("rx", 0.0), ("ry", 0.0), ("rz", 0.0)):
            if getattr(self, key) != default or key in self.expr:
                out[key] = getattr(self, key)
        if self.fit != "slip":
            out["fit"] = self.fit
        if self.orient != "seat":
            out["orient"] = self.orient
        if self.flip != "auto":
            out["flip"] = self.flip
        if self.station_order != "world":
            out["station_order"] = self.station_order
        if self.stations:
            out["stations"] = list(self.stations)
        if not self.solve:
            out["solve"] = False
        if self.note:
            out["note"] = self.note
        return self.unbake(out)

    # ── reference parsing ───────────────────────────────────────────────
    @staticmethod
    def split(ref: str) -> tuple[str, str]:
        part, _, conn = str(ref).partition(".")
        return part, conn

    @staticmethod
    def is_frame(ref: str) -> bool:
        """True when this side names a declared datum rather than a part."""
        return str(ref).startswith(FRAME_PREFIX)

    @staticmethod
    def frame_id(ref: str) -> str:
        return str(ref)[len(FRAME_PREFIX):].partition(".")[0]

    @property
    def part_a(self) -> str:
        return self.split(self.a)[0]

    @property
    def part_b(self) -> str:
        return self.split(self.b)[0]

    @property
    def edge(self) -> tuple[str, str]:
        return (self.part_a, self.part_b)

    @property
    def total_dz(self) -> float:
        return float(self.dz) + float(self.offset_mm)

    @property
    def total_rz(self) -> float:
        return float(self.rz) + float(self.angle_deg)

    @property
    def free_axes(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        """Which of (tx, ty, tz, rx, ry, rz) this joint kind leaves unconstrained.

        Read in the host seat's frame, whose +Z is the joint axis by the
        vocabulary's own invariant. The solver drops a residual component for
        every free axis, so a revolute mate holds position exactly and says
        nothing at all about the angle it was authored with.
        """
        if self.kind == "revolute":
            return (False, False, False, False, False, True)
        if self.kind == "cylindrical":
            return (False, False, True, False, False, True)
        if self.kind == "linear":
            return (False, False, True, False, False, False)
        if self.kind == "ball":
            return (False, False, False, True, True, True)
        return (False,) * 6


@dataclass
class DatumSpec(HasExpressions):
    """A named reference frame — the thing a drawing measures from.

    `on` is what this datum hangs off: another datum (`@bed`), a seat on a part
    (`bed_casting.top_face`), a part origin (`bed_casting`), or nothing, which
    means the assembly origin. The six offsets are read in that parent frame.
    A datum has no geometry, no mass and no volume; it exists so that ten parts
    can be positioned from one place instead of down a chain of ten neighbours,
    which is the difference between an error that stays 0.1 mm and one that
    accumulates to 1 mm by the far end of the machine.

    `point` names a solved point in the 2D layout plan (see layout.py). When it
    is set, the layout supplies this datum's in-plane position and the offsets
    below are applied on top — so the human-approved 2D drawing is literally
    what positions the 3D model.
    """

    id: str
    on: str = ""
    kind: str = "frame"
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    point: str = ""
    note: str = ""
    expr: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DatumSpec":
        e: dict[str, str] = {}
        return cls(
            id=str(d.get("id", "")),
            on=str(d.get("on", "") or ""),
            kind=str(d.get("kind", "frame") or "frame"),
            dx=_num(d, "dx", 0.0, e),
            dy=_num(d, "dy", 0.0, e),
            dz=_num(d, "dz", 0.0, e),
            rx=_num(d, "rx", 0.0, e),
            ry=_num(d, "ry", 0.0, e),
            rz=_num(d, "rz", 0.0, e),
            point=str(d.get("point", "") or ""),
            note=str(d.get("note", "") or ""),
            expr=e,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"id": self.id, "on": self.on}
        if self.kind != "frame":
            out["kind"] = self.kind
        for key in ("dx", "dy", "dz", "rx", "ry", "rz"):
            if getattr(self, key) or key in self.expr:
                out[key] = getattr(self, key)
        for key in ("point", "note"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return self.unbake(out)

    @property
    def parent(self) -> str:
        """The node this datum is measured from — `WORLD` when it is rooted."""
        if not self.on or self.on.upper() == WORLD:
            return WORLD
        if MateSpec.is_frame(self.on):
            return MateSpec.frame_id(self.on)
        return MateSpec.split(self.on)[0]

    @property
    def parent_seat(self) -> str:
        if not self.on or MateSpec.is_frame(self.on):
            return ""
        return MateSpec.split(self.on)[1]


@dataclass
class DimSpec(HasExpressions):
    """A dimension the finished assembly must actually measure.

    This is the check the engine had no way to express, and its absence is why a
    turbocharger with no turbine passed: every other check asks whether the model
    is self-consistent, and a wrong model is perfectly self-consistent. A
    dimension assertion asks whether it is *right*.

    `of` and `to` are anchors — `part_id`, `part_id.connector`, or `@datum`. With
    only `of`, the measure is a property of that part (its own length along an
    axis, its mass, its volume). With both, it is the distance between them.
    """

    of: str
    to: str = ""
    measure: str = "distance"
    value: float = 0.0
    tol: float = 0.5
    note: str = ""
    expr: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DimSpec":
        e: dict[str, str] = {}
        return cls(
            of=str(d.get("of", "") or ""),
            to=str(d.get("to", "") or ""),
            measure=str(d.get("measure", "distance") or "distance").lower(),
            value=_num(d, "value", 0.0, e),
            tol=_num(d, "tol", 0.5, e),
            note=str(d.get("note", "") or ""),
            expr=e,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"of": self.of, "measure": self.measure,
                               "value": self.value, "tol": self.tol}
        if self.to:
            out["to"] = self.to
        if self.note:
            out["note"] = self.note
        return self.unbake(out)

    @property
    def label(self) -> str:
        span = f"{self.of} -> {self.to}" if self.to else self.of
        return f"{self.measure}({span})"


@dataclass
class CheckSpec:
    """An engineering condition, named rather than measured.

    `kind` selects a rule in conditions.py and the rest of the dict is its
    arguments. These are the facts a geometry checker cannot infer: that three
    planets must divide evenly into the sun and ring tooth counts, that two
    meshing gears must share a module, that a bolt circle only bolts to a bolt
    circle of the same diameter and count.
    """

    kind: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckSpec":
        d = dict(d or {})
        kind = str(d.pop("kind", "") or "").lower()
        return cls(kind=kind, args=d)

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self.args}

    def get(self, key: str, default: Any = None) -> Any:
        return self.args.get(key, default)


@dataclass
class ExpectSpec:
    """The envelope the finished assembly has to land inside.

    Every field is optional and an absent field is not checked, so an old spec
    keeps building. But a spec that states nothing here is telling the verifier
    it has no opinion about what it built, and `verify` now says so out loud
    rather than printing PASS.
    """

    mass_kg: list[float] = field(default_factory=list)      # [lo, hi]
    volume_cm3: list[float] = field(default_factory=list)    # [lo, hi]
    bbox_mm: dict[str, list[float]] = field(default_factory=dict)  # {"x": [lo, hi], ...}
    parts: list[int] = field(default_factory=list)           # [lo, hi] instances
    dims: list[DimSpec] = field(default_factory=list)
    features: list[str] = field(default_factory=list)        # for the vision judge
    watertight: bool = True
    allow_placeholders: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "ExpectSpec":
        d = dict(d or {})
        return cls(
            mass_kg=[float(v) for v in (d.get("mass_kg") or [])],
            volume_cm3=[float(v) for v in (d.get("volume_cm3") or [])],
            bbox_mm={str(k): [float(x) for x in v]
                     for k, v in (d.get("bbox_mm") or {}).items()},
            parts=[int(v) for v in (d.get("parts") or [])],
            dims=[DimSpec.from_dict(x) for x in (d.get("dims") or [])],
            features=[str(v) for v in (d.get("features") or [])],
            watertight=bool(d.get("watertight", True)),
            allow_placeholders=bool(d.get("allow_placeholders", False)),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        if self.mass_kg:
            out["mass_kg"] = self.mass_kg
        if self.volume_cm3:
            out["volume_cm3"] = self.volume_cm3
        if self.bbox_mm:
            out["bbox_mm"] = self.bbox_mm
        if self.parts:
            out["parts"] = self.parts
        if self.dims:
            out["dims"] = [x.to_dict() for x in self.dims]
        if self.features:
            out["features"] = self.features
        if not self.watertight:
            out["watertight"] = False
        if self.allow_placeholders:
            out["allow_placeholders"] = True
        return out

    @property
    def empty(self) -> bool:
        return not (self.mass_kg or self.volume_cm3 or self.bbox_mm
                    or self.parts or self.dims or self.features)

    def bake(self, scope: dict[str, float]) -> list[str]:
        errs: list[str] = []
        for i, dim in enumerate(self.dims):
            errs += [f"expect.dims[{i}].{e}" for e in dim.bake(scope)]
        return errs


@dataclass
class Spec:
    name: str
    datum: str
    parts: list[PartSpec] = field(default_factory=list)
    mates: list[MateSpec] = field(default_factory=list)
    datums: list[DatumSpec] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)
    expect: ExpectSpec = field(default_factory=ExpectSpec)
    checks: list[CheckSpec] = field(default_factory=list)
    # The frozen 2D design plan this assembly was built from, inline. Kept in the
    # spec rather than beside it so a spec is one reviewable artefact.
    layout: dict[str, Any] = field(default_factory=dict)
    intent: str = ""
    units: str = "mm"
    # How far a mate's two seats may end up apart before it is a failure. Absolute,
    # in mm, because a tolerance that scales with the model would let a 3 m machine
    # be 15 mm out and call it fine.
    tolerance_mm: float = 0.05
    # Resolved variable values. Not serialised — derived from `vars` by `resolve`.
    scope: dict[str, float] = field(default_factory=dict, repr=False)
    var_errors: list[str] = field(default_factory=list, repr=False)
    baked: bool = field(default=False, repr=False)

    # ── serialisation ───────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        spec = cls(
            name=str(d.get("name", "assembly")),
            datum=str(d.get("datum", "")),
            parts=[PartSpec.from_dict(x) for x in (d.get("parts") or [])],
            mates=[MateSpec.from_dict(x) for x in (d.get("mates") or [])],
            datums=[DatumSpec.from_dict(x) for x in (d.get("datums") or [])],
            vars=dict(d.get("vars") or {}),
            expect=ExpectSpec.from_dict(d.get("expect") or {}),
            checks=[CheckSpec.from_dict(x) for x in (d.get("checks") or [])],
            layout=dict(d.get("layout") or {}),
            intent=str(d.get("intent", "")),
            units=str(d.get("units", "mm")),
            tolerance_mm=float(d.get("tolerance_mm", 0.05) or 0.05),
        )
        spec.resolve()
        return spec

    @classmethod
    def from_json(cls, text: str) -> "Spec":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"name": self.name, "datum": self.datum}
        if self.intent:
            out["intent"] = self.intent
        if self.units != "mm":
            out["units"] = self.units
        if self.tolerance_mm != 0.05:
            out["tolerance_mm"] = self.tolerance_mm
        if self.vars:
            out["vars"] = self.vars
        if self.datums:
            out["datums"] = [x.to_dict() for x in self.datums]
        out["parts"] = [x.to_dict() for x in self.parts]
        out["mates"] = [x.to_dict() for x in self.mates]
        if self.checks:
            out["checks"] = [x.to_dict() for x in self.checks]
        expect = self.expect.to_dict()
        if expect:
            out["expect"] = expect
        if self.layout:
            out["layout"] = self.layout
        return out

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # ── equations ───────────────────────────────────────────────────────
    def resolve(self) -> list[str]:
        """Evaluate `vars`, then push the numbers into every numeric field.

        Idempotent and safe to call again after a repair round has edited a
        formula. Errors are kept on the spec rather than raised, because a bad
        equation is a spec error like any other and belongs in the same report.
        """
        scope, errors = expr.resolve(self.vars)
        # The layout's own variables are visible to the assembly, so a station
        # dimension can be stated once and used by both.
        for key, val in (self.layout.get("vars") or {}).items():
            if key not in scope and not expr.is_expression(val):
                try:
                    scope[key] = float(val)
                except (TypeError, ValueError):
                    pass
        self.scope = scope
        for i, mate in enumerate(self.mates):
            errors += [f"mates[{i}].{e}" for e in mate.bake(scope)]
        for i, dat in enumerate(self.datums):
            errors += [f"datums[{i}].{e}" for e in dat.bake(scope)]
        errors += self.expect.bake(scope)
        self.var_errors = errors
        self.baked = True
        return errors

    def params_of(self, part: PartSpec) -> dict[str, Any]:
        return part.baked(self.scope)

    # ── lookups ─────────────────────────────────────────────────────────
    def part(self, pid: str) -> PartSpec | None:
        for p in self.parts:
            if p.id == pid:
                return p
        return None

    def frame(self, fid: str) -> DatumSpec | None:
        fid = MateSpec.frame_id(fid) if MateSpec.is_frame(fid) else fid
        for d in self.datums:
            if d.id == fid:
                return d
        return None

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.parts]

    @property
    def frame_ids(self) -> list[str]:
        return [d.id for d in self.datums]

    @property
    def nodes(self) -> list[str]:
        """Everything the mate graph can reach: parts, datums and the origin."""
        return [WORLD] + self.ids + self.frame_ids

    @property
    def total_instances(self) -> int:
        return sum(max(1, p.count) for p in self.parts)

    def node_of(self, ref: str) -> str:
        """The graph node a mate side refers to."""
        return MateSpec.frame_id(ref) if MateSpec.is_frame(ref) else MateSpec.split(ref)[0]

    def graph_edges(self) -> list[tuple[str, str]]:
        """Mate edges followed by the implicit datum edges, in that index order.

        The assembler relies on the ordering: an index below `len(mates)` is a
        real mate, and anything above it is a datum hanging off its parent. One
        traversal then places parts and reference frames in dependency order,
        which is what lets a datum be measured from a part and a part from a
        datum in the same assembly.
        """
        out = [(self.node_of(m.a), self.node_of(m.b)) for m in self.mates]
        out += [(d.parent, d.id) for d in self.datums]
        if self.datum:
            out.append((WORLD, self.datum))
        return out

    def link_kind(self, idx: int) -> tuple[str, int]:
        """`('mate', i)`, `('datum', i)` or `('root', -1)` for a graph edge index."""
        if idx < len(self.mates):
            return ("mate", idx)
        idx -= len(self.mates)
        if idx < len(self.datums):
            return ("datum", idx)
        return ("root", -1)

    def anchor_ref(self, pid: str) -> str:
        """The reference a dimension uses when it names a bare part id."""
        part = self.part(pid)
        if part is not None and part.anchor:
            return f"{pid}.{part.anchor}"
        return pid
