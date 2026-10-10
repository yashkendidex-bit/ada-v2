"""Layer 1 grows itself: ADA writes the part she does not have.

The wall this removes: a spec that named `shift_fork` used to die at validate.py
rule 2 with UNKNOWN_SKILL, so a machine outside the curated vocabulary was
unbuildable no matter how good the design was. Here an unknown skill becomes a
*work order* instead of an error, and every work order in a spec is fulfilled by
one batched model request:

    work_orders(spec)   one request for every missing skill at once
      -> screen the code, exec it, let the @skill decorator register it
      -> gate.run() referees the geometry, then it is built at the spec's OWN
         params and asked for the connectors the spec's mates actually reference
      -> admitted:  written to vocabulary/generated/, known forever after
      -> rejected:  a dimensioned PLACEHOLDER envelope that verify reports

Three deliberate asymmetries.

Admitted skills are persisted; placeholders never are. A persisted stub would
look like a known skill on the next run, and the real part would never be
synthesised — the vocabulary would quietly fill up with cylinders.

A rejected candidate is removed from the registry again. Half-admitting it would
let the assembler build geometry the gate has already called wrong.

A placeholder does not fail the build, it is reported. A job that stops dead
teaches you nothing; a job that finishes with three parts stamped PLACEHOLDER
tells you exactly what is still missing and lets you look at everything else.

Safety: this module executes model-written Python in-process. That is inherent
to "invent a new part", not something a design choice can avoid, so candidates
are screened for filesystem/network/eval tokens, exec'd against a namespace that
holds build123d and the frame primitives and nothing else, and only written to
disk once the gate admits them. The screen stops accidents and casual mischief.
It is not a sandbox, and it is not sold as one.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import gate, registry
from .schema import Spec

GENERATED = Path(__file__).resolve().parent / "vocabulary" / "generated"

Chat = Callable[[list[dict]], Awaitable[str]]


# ── the work order ──────────────────────────────────────────────────────
@dataclass
class WorkOrder:
    """One missing skill, plus everything the spec already implied about it.

    The spec is a better brief than any prose description: it names the params
    the design wants, the connectors its mates reach for, and — through the
    counts of the parts on the other side of those mates — how many stations a
    patterned seat has to have.
    """

    skill: str
    observed: dict[str, list[Any]] = field(default_factory=dict)
    seats: dict[str, int] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)

    @property
    def params(self) -> dict[str, Any]:
        """The first parameter set the spec used — what the skill must build at."""
        return {k: v[0] for k, v in self.observed.items()}

    def note(self, part) -> None:
        for k, v in (part.params or {}).items():
            self.observed.setdefault(k, []).append(v)
        if part.label:
            self.labels.append(part.label)
        self.counts.append(max(1, int(part.count)))

    def text(self) -> str:
        head = f"- skill `{self.skill}`"
        if self.labels:
            head += f"  ({'; '.join(sorted(set(self.labels)))})"
        if max(self.counts or [1]) > 1:
            head += f"  x{max(self.counts)} instances"
        lines = [head]
        if self.observed:
            lines.append("    params the design passes: "
                         + ", ".join(f"{k}={v[0]!r}" for k, v in sorted(self.observed.items())))
        for name, n in sorted(self.seats.items()):
            line = f"    must publish connector '{name}'"
            if n > 1:
                line += f" as a PATTERNED seat with exactly {n} stations"
                # The layout is not a preference: it is read off the spec's own mates and
                # the size of what those mates hang there, so it is as much a requirement
                # as the station count.
                shape = self.hints.get(name, "")
                if shape == "radial":
                    line += ", each one FACING OUT of the axis (Connector.polar with " \
                            "tilt_deg=90) because what it carries reaches further than " \
                            "the ring itself"
                elif shape == "stack":
                    line += " spaced ALONG the axis (Connector.stack), one per part that " \
                            "sits on it"
                elif shape == "ring":
                    line += " round the axis, facing along it (Connector.polar)"
            lines.append(line)
        return "\n".join(lines)


def seats_of(spec: Spec, pid: str) -> dict[str, int]:
    """Every connector the spec's mates ask of one part, and how many stations.

    A single part hosting N children needs N seats; a part that is itself one of
    N children only ever uses station 0. Both facts are already in the spec, so
    a skill that does not exist yet still has its seat count fully determined.
    """
    count_of = {p.id: max(1, int(p.count)) for p in spec.parts}
    mine = count_of.get(pid, 1)
    seats: dict[str, int] = {}
    for m in spec.mates:
        for ref, other in ((m.a, m.b), (m.b, m.a)):
            rid, conn = m.split(ref)
            if rid != pid or not conn:
                continue
            need = count_of.get(m.split(other)[0], 1) if mine == 1 else 1
            seats[conn] = max(seats.get(conn, 1), int(need))
    return seats


def work_orders(spec: Spec) -> list[WorkOrder]:
    """Every part whose skill is not registered, grouped by skill name."""
    orders: dict[str, WorkOrder] = {}
    for part in spec.parts:
        if registry.get(part.skill) is not None:
            continue
        wo = orders.setdefault(part.skill, WorkOrder(part.skill))
        wo.note(part)
        for conn, need in seats_of(spec, part.id).items():
            wo.seats[conn] = max(wo.seats.get(conn, 1), need)
        # A seat that any one use needs to face outward faces outward for all of them:
        # two parts sharing a skill share its geometry, so the stronger requirement wins.
        for conn, shape in seat_hints(spec, part.id).items():
            if shape == "radial" or conn not in wo.hints:
                wo.hints[conn] = shape
    return [orders[k] for k in sorted(orders)]


# ── the brief the model gets ────────────────────────────────────────────
SKILL_CONTRACT = """You are writing new words for ADA's CAD vocabulary: real build123d
part generators, one per missing part, in a single reply.

SHAPE — every skill looks exactly like this and nothing else:

    @skill("name", category="...", connectors=("a", "b"),
           params=[p("bore_d", float, 12.0, min=1, max=500)])
    def name(**kw) -> SkillResult:
        \"\"\"One line saying what the part is.\"\"\"
        with BuildPart() as bp:
            ...
        res = SkillResult(bp.part, interfaces={"od": od}, meta={"source": "synth"})
        res.add(Connector.face("a", h), Connector.at("b", z=0.0))
        return res

HARD RULES

1. NO IMPORTS. Everything is already in scope: every build123d name
   (BuildPart, BuildSketch, Plane, Axis, Mode, Align, Location, Locations,
   PolarLocations, GridLocations, Cylinder, Box, Sphere, Torus, Cone, Circle,
   Rectangle, Polyline, Line, RegularPolygon, extrude, revolve, loft, sweep,
   fillet, chamfer, add, make_face, mirror, offset, split, section), plus
   `math`, `Connector`, `SkillResult`, `skill`, `p`, and the ADA helpers
   `revolve_profile_solid(steps, bore_d=0.0, arc=360.0)` (steps are
   [diameter, z] pairs, absolute z), `axial_bore`, `blade_sweep`,
   `spiral_path`, `metric(size)` and `METRIC`.
2. GEOMETRY LIVES ON +Z. The part is centred on x=y=0, its base sits at z=0,
   its functional axis IS +Z. Never place a part where you think it belongs in
   the machine — the assembler does that from the mates. Millimetres always.
3. ONE SOLID. Everything must fuse into a single connected solid. Blades must
   actually intersect their hub, ribs must overlap their wall. Floating shapes
   are rejected.
4. DECLARE EVERY PARAM the work order lists, under that exact name, with a
   type and min/max wide enough to contain the value shown (and 3x beyond).
   Give every param a default that builds. Extra params are fine.
5. PUBLISH EVERY CONNECTOR the work order lists, under those exact names.
   A connector is a promise about where a neighbour will be bolted:
     Connector.face(name, z, outward=True)   a planar mating face at height z
     Connector.at(name, x=.., z=.., ry=..)   a single seat / bore axis
     Connector.polar(name, radius, count, z, tilt_deg=0)   a ring of seats
     Connector.stack(name, [z0, z1, z2])     seats along the axis
   `tilt_deg` on a polar ring decides which way its stations FACE, and it is the
   difference between a rotor head and a bundle of sticks. tilt_deg=0 is AXIAL:
   every station's +Z points along the part's own +Z, which is what a bolt circle
   on a flange wants. tilt_deg=90 is RADIAL: every station's +Z points straight
   out from the axis, each one rotated to its own angle round the ring. Any seat
   that carries a part reaching OUTWARD — a blade station, a trunnion, an arm or
   spoke root, a vane, a radial port — must use tilt_deg=90. A child mated to a
   station is built along that station's +Z, so publishing a blade ring at
   tilt_deg=0 stands all the blades up parallel to the shaft instead of laying
   them out in the rotor disc, and nothing downstream can rotate them back.
   If you say a face is at z=w there must be real material with a planar face
   at z=w. A patterned seat must have EXACTLY the number of stations asked
   for. This is measured, and a lie here is the most common rejection.
6. DETERMINISTIC. Same params in, same solid out. No randomness, no clock, no
   files, no network, no printing.
7. Fill `interfaces` with the real fitting numbers you used (od, bore_d,
   width, length, pcd) — the clash checker reads them.
8. Prefer boolean-simple geometry that survives at every size in the range:
   revolves, extrudes, polar patterns. Avoid fillets/chamfers larger than a
   quarter of the local wall.

OUTPUT FORMAT — for each missing part, a header line then one fenced block:

### SKILL <exact_skill_name>
```python
@skill("exact_skill_name", ...)
def exact_skill_name(**kw) -> SkillResult:
    ...
```

No prose, no explanation, no markdown headings other than the SKILL lines."""


def _example_source(name: str = "journal_bearing") -> str:
    """Quote a real admitted skill. The best spec of the contract is the code."""
    import inspect

    sk = registry.get(name)
    if sk is None:
        return ""
    try:
        return inspect.getsource(sk.fn).rstrip()
    except Exception:
        return ""


def synth_prompt(orders: list[WorkOrder], *, example: bool = True) -> str:
    blocks = ["MISSING PARTS — write one skill for each, all in this one reply:\n"
              + "\n".join(o.text() for o in orders)]
    if example:
        src = _example_source()
        if src:
            blocks.append("WORKED EXAMPLE — a real skill already in the vocabulary. "
                          "Copy this shape exactly:\n```python\n" + src + "\n```")
    blocks.append(f"Write {len(orders)} skill block(s) now.")
    return "\n\n".join(blocks)


# ── reading the reply ───────────────────────────────────────────────────
_HEADER = re.compile(r"^[\s#*>-]*#{1,4}\s*SKILL[:\s]+`?([A-Za-z_]\w*)`?\s*$", re.M)
_DECOR = re.compile(r"@skill\(\s*[\"']([A-Za-z_]\w*)[\"']")


def _unfence(body: str) -> str:
    """Take the code out of the largest fenced block, or the text as-is."""
    if "```" not in body:
        return body.strip("\n")
    inner = body.split("```")[1::2]
    if not inner:
        return body.strip("\n")
    best = max(inner, key=len)
    first, _, rest = best.partition("\n")
    if first.strip().lower() in ("python", "py", "python3"):
        best = rest
    return best.strip("\n")


def parse_skills(raw: str) -> dict[str, str]:
    """`### SKILL name` sections -> code. A reply with no headers is one blob.

    The blob case is not a fallback for tidiness — a model that returns one
    module containing three @skill functions has answered correctly, and the
    registry diff after exec tells us which names arrived either way.
    """
    marks = list(_HEADER.finditer(raw or ""))
    if not marks:
        code = _unfence(raw or "")
        return {"": code} if "@skill" in code else {}
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        code = _unfence(raw[m.end():end])
        if code:
            out[m.group(1)] = code
    return out


# ── the screen ──────────────────────────────────────────────────────────
_FORBIDDEN: tuple[tuple[str, str], ...] = (
    (r"\b(?:import|from)\s+(?:os|sys|subprocess|socket|shutil|pathlib|urllib|http|"
     r"requests|pickle|ctypes|importlib|builtins|multiprocessing|threading|tempfile)\b",
     "system import"),
    (r"\bfrom\s+\.", "relative import (nothing needs importing)"),
    (r"\b__import__\b", "dynamic import"),
    (r"\bopen\s*\(", "file access"),
    (r"\beval\s*\(|\bexec\s*\(|\bcompile\s*\(", "dynamic code"),
    (r"\bglobals\s*\(|\blocals\s*\(|\bvars\s*\(", "scope introspection"),
    (r"\b(?:setattr|delattr)\s*\(", "attribute mutation"),
    (r"\bREGISTRY\b|\bregistry\s*\.", "registry tampering"),
    (r"__(?:subclasses|bases|mro|class|globals|dict)__", "dunder escape"),
    (r"\bwhile\s+True\b", "unbounded loop"),
    (r"\b(?:input|breakpoint|help|exit|quit)\s*\(", "interactive call"),
)


def screen(code: str) -> list[str]:
    """Reject candidates that reach outside geometry. Not a sandbox — a filter."""
    return [f"{why}: {m.group(0).strip()!r}"
            for pat, why in _FORBIDDEN for m in [re.search(pat, code)] if m]


# ── exec against a geometry-only namespace ──────────────────────────────
_SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "callable", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "getattr", "hasattr", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "pow", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError",
    "KeyError", "RuntimeError", "TypeError", "ValueError", "ZeroDivisionError",
    "True", "False", "None", "__build_class__", "__name__",
)


def _namespace(module_name: str) -> dict[str, Any]:
    import builtins
    import math

    import build123d as b3d

    from .frames import Connector, SkillResult
    from .vocabulary import _util

    ns: dict[str, Any] = {k: getattr(b3d, k) for k in dir(b3d) if not k.startswith("_")}
    ns.update(
        __name__=module_name,
        __doc__=None,
        __builtins__={n: getattr(builtins, n) for n in _SAFE_BUILTINS if hasattr(builtins, n)},
        math=math,
        Connector=Connector,
        SkillResult=SkillResult,
        skill=registry.skill,
        p=registry.p,
        METRIC=_util.METRIC,
        metric=_util.metric,
        revolve_profile_solid=_util.revolve_profile_solid,
        axial_bore=_util.axial_bore,
        blade_sweep=_util.blade_sweep,
        spiral_path=_util.spiral_path,
        dedupe=_util.dedupe,
    )
    return ns


def install(code: str, *, expect: str | None = None) -> tuple[list[str], str]:
    """Screen, exec, and report which skill names the code registered.

    The curated vocabulary is protected by identity, not by name lists: any
    pre-existing Skill object that the exec replaced is put back and reported,
    so a candidate cannot quietly redefine `shaft` and be believed.
    """
    bad = screen(code)
    if bad:
        return [], "screen rejected " + "; ".join(bad)
    if expect and registry.get(expect) is not None:
        return [], f"'{expect}' is already registered — refusing to overwrite it"

    before = dict(registry.REGISTRY)
    ns = _namespace(f"cad_v3.vocabulary.generated.{expect or 'batch'}")
    try:
        exec(compile(code, f"<synth:{expect or 'batch'}>", "exec"), ns)
    except Exception as exc:
        for name in set(registry.REGISTRY) - set(before):
            registry.REGISTRY.pop(name, None)
        return [], f"{type(exc).__name__}: {exc}"

    clobbered = [n for n, sk in before.items() if registry.REGISTRY.get(n) is not sk]
    for name in clobbered:
        registry.REGISTRY[name] = before[name]
    fresh = sorted(set(registry.REGISTRY) - set(before))
    for name in fresh:
        registry.REGISTRY[name].source = "synth"
    if clobbered:
        return fresh, "tried to redefine existing skill(s): " + ", ".join(sorted(clobbered))
    if expect and expect not in fresh:
        got = ", ".join(fresh) or "nothing"
        return fresh, f"registered {got}, not the requested '{expect}'"
    return fresh, ""


def uninstall(name: str) -> None:
    """A rejected candidate must not linger in the registry."""
    sk = registry.REGISTRY.get(name)
    if sk is not None and sk.source in ("synth", "stub"):
        registry.REGISTRY.pop(name, None)


# ── the referee ─────────────────────────────────────────────────────────
@dataclass
class Verdict:
    skill: str
    admitted: bool = False
    problems: list[str] = field(default_factory=list)
    report: Any = None
    seconds: float = 0.0

    @property
    def why(self) -> str:
        return " | ".join(self.problems)


def admit(order: WorkOrder, *, sweep: bool = False) -> Verdict:
    """Three tests, in increasing order of relevance to this particular spec.

    gate.run is the general one: does it build at its defaults, across its
    declared range, twice identically, with seats on material and about Z.
    Then the two tests only this spec can pose — does it build at the params
    the design actually passes, and does it publish the seats the design's
    mates actually reach for, with enough stations for the pattern.
    """
    t0 = time.time()
    v = Verdict(order.skill)
    sk = registry.get(order.skill)
    if sk is None:
        v.problems.append("not registered after exec")
        return v

    v.report = gate.run(order.skill, sweep=sweep)
    if not v.report.passed:
        v.problems.append(v.report.feedback().replace("\n", " "))

    kwargs, errs = registry.coerce_params(sk, order.params)
    if errs:
        v.problems.append("the spec's own params are rejected: " + "; ".join(errs))
    else:
        try:
            res = sk.fn(**kwargs)
        except Exception as exc:
            v.problems.append(f"failed at the spec's params: {type(exc).__name__}: {exc}")
        else:
            # A candidate that forgot the wrapper hands back a raw solid. The gate
            # says so too, but this test must not trip over it on the way past.
            conns = getattr(res, "connectors", None)
            if conns is None:
                v.problems.append(
                    f"returned {type(res).__name__}, not a SkillResult — end the "
                    "function with `return SkillResult(part, ...)` and publish the seats")
                conns = {}
            for name, need in sorted(order.seats.items()):
                conn = conns.get(name)
                if conn is None:
                    v.problems.append(f"does not publish connector '{name}' "
                                      f"(has: {', '.join(sorted(conns)) or 'none'})")
                elif conn.count < need:
                    v.problems.append(f"connector '{name}' has {conn.count} station(s), "
                                      f"the spec patterns {need} parts onto it")
            missing = [n for n in order.seats if n not in sk.connectors]
            if missing:
                v.problems.append("declared connectors omit " + ", ".join(sorted(missing)))

    v.admitted = not v.problems
    v.seconds = time.time() - t0
    return v


# ── the placeholder ─────────────────────────────────────────────────────
_D_KEYS = ("od", "diameter", "d", "outer_d", "rim_d", "tip_d", "flange_od", "major_d", "pcd")
_L_KEYS = ("length", "width", "height", "thickness", "depth", "len", "l", "span")
_BORE_KEYS = ("bore_d", "bore", "id", "inner_d", "shaft_d", "minor_d")


def _numbers(kw: dict) -> dict[str, float]:
    return {k.lower(): float(v) for k, v in kw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _biggest(num: dict[str, float], keys: tuple[str, ...]) -> float:
    """The largest value under any param whose name *reads* like one of `keys`.

    Exact matching alone loses `free_length`, `hub_od` and `plate_thickness` —
    which is how a 40 mm spring ends up as a 7 mm envelope. Compound names are
    matched on their head or tail so the number lands where it belongs.
    """
    hit = 0.0
    for name, v in num.items():
        if name in keys or any(name.endswith("_" + k) or name.startswith(k + "_")
                               for k in keys):
            hit = max(hit, v)
    return hit


def _envelope(kw: dict) -> tuple[float, float, float]:
    """Guess od / length / bore from the param names the design used.

    A placeholder that is the right *size* is worth far more than a unit cube:
    it holds the space, keeps the neighbours apart, and shows up in the clash
    report at roughly the volume the real part will occupy.
    """
    num = _numbers(kw)
    bore = _biggest(num, _BORE_KEYS)
    od = _biggest(num, _D_KEYS)
    if od <= 0.0 and "module" in num and "teeth" in num:
        od = num["module"] * num["teeth"]
    if od <= 0.0:
        size = kw.get("size")
        if isinstance(size, str):
            try:
                from .vocabulary._util import metric
                od = metric(size)["d"] * 2.0
            except Exception:
                od = 0.0
    if od <= 0.0:
        od = max([v for v in num.values() if 0.0 < v < 5000.0] or [24.0])
    length = _biggest(num, _L_KEYS) or od * 0.6
    od = max(od, bore + 4.0, 2.0)
    return od, max(length, 1.0), max(0.0, min(bore, od - 2.0))


def _param_defs(observed: dict[str, list]) -> list:
    """Types from what the spec passed, no ranges — a placeholder refuses nothing."""
    defs = []
    for name, vals in sorted(observed.items()):
        v = vals[0]
        if isinstance(v, bool):
            defs.append(registry.p(name, bool, v))
        elif isinstance(v, (list, tuple)):
            defs.append(registry.p(name, list, list(v)))
        elif isinstance(v, str):
            defs.append(registry.p(name, str, v))
        elif isinstance(v, int):
            defs.append(registry.p(name, int, int(v)))
        else:
            defs.append(registry.p(name, float, float(v)))
    return defs


def _stub_names(seats: dict[str, int]) -> tuple[str, ...]:
    extra = [n for n in ("bore", "front_face", "rear_face", "od_mid", "mid") if n not in seats]
    return tuple(list(seats) + extra)


# ── how a placeholder's patterned seats have to be laid out ─────────────────
# A stub is a promise about where its neighbours go, so getting a seat's facing wrong is
# worse than having no geometry at all: the blades of a stubbed rotor hub end up parallel
# to the mast, and every repair round afterwards fights a clash no offset can cure.
#
# This used to be decided by a list of fourteen nouns — blade, arm, spoke, vane, grip,
# lug, paddle, bucket, fin… — and that is the same mistake as a list of housing names. It
# worked for the machines whose parts happened to be on the list. Both facts it was
# guessing at are written down in the spec already, and both are arithmetic:
#
#   * A seat serving ONE part with count N is a pattern — N copies of one thing, which
#     goes round. A seat serving N *different* parts is a sequence of stations, which
#     goes along: three bearings up one shaft are not a bolt circle.
#   * A ring faces out of the axis when what it carries reaches further than the ring it
#     sits on and is much longer than it is thick. Reach means the whole sub-assembly,
#     not the one part: a 60 mm grip carrying a 1500 mm blade is a 1560 mm limb, and it
#     is the 1560 that cannot stand up the mast. An M10 stud on a 240 mm bolt circle
#     reaches 30 mm and stands up out of the flange, exactly as it should.


def _subtree_reach(spec: Spec) -> dict[str, float]:
    """Each part's own length plus the longest chain of lengths hanging below it.

    Below means below in the placement tree the assembler will actually build, which is
    why this walks `graph.mate_order` rather than the mate list: the spec's mates form a
    graph with loops in it, and "what hangs off this part" is only well defined once the
    traversal has chosen a parent for everything.
    """
    from . import graph

    length = {p.id: _envelope(p.params or {})[1] for p in spec.parts}
    try:
        order = graph.mate_order(spec.datum, [m.edge for m in spec.mates])
    except Exception:
        return length
    tree: list[tuple[str, str]] = []
    seen: set[str] = {spec.datum}
    for parent, child, _ in order:
        if child in seen:
            continue                      # a loop closure states no parentage
        seen.add(child)
        tree.append((parent, child))
    reach = dict(length)
    for parent, child in reversed(tree):  # BFS is parents first, so reverse is bottom-up
        reach[parent] = max(reach.get(parent, 0.0),
                           length.get(parent, 0.0) + reach.get(child, 0.0))
    return reach


def seat_hints(spec: Spec, pid: str) -> dict[str, str]:
    """Per connector asked of `pid`: 'stack', 'radial', 'ring', or absent for a single.

    Measured off the spec's own mates and params — no part name is read, so a machine
    nobody has ever built gets the same answer as a rotor head.
    """
    count_of = {p.id: max(1, int(p.count)) for p in spec.parts}
    part_of = {p.id: p for p in spec.parts}
    if count_of.get(pid, 1) != 1:
        return {}                         # one of N children; it only ever uses station 0
    users: dict[str, list[str]] = {}
    for m in spec.mates:
        for ref, other in ((m.a, m.b), (m.b, m.a)):
            rid, conn = m.split(ref)
            if rid == pid and conn:
                users.setdefault(conn, []).append(m.split(other)[0])
    host = part_of.get(pid)
    ring_r = max(_envelope(host.params or {})[0] * 0.5, 1.0) if host else 1.0
    reach = _subtree_reach(spec)
    out: dict[str, str] = {}
    for conn, kids in users.items():
        uniq = [k for i, k in enumerate(kids) if k not in kids[:i]]
        if not uniq:
            continue
        if len(uniq) > 1 and all(count_of.get(k, 1) == 1 for k in uniq):
            out[conn] = "stack"
            continue
        if count_of.get(uniq[0], 1) < 2:
            continue                      # a single child needs no layout at all
        child = part_of.get(uniq[0])
        thickness = _envelope(child.params or {})[0] if child else 0.0
        limb = reach.get(uniq[0], 0.0)
        out[conn] = ("radial" if limb > ring_r and limb > 1.5 * thickness else "ring")
    return out


def _stub_connectors(seats: dict[str, int], names, od: float, length: float,
                     hints: dict[str, str] | None = None) -> list:
    from .frames import Connector

    hints = hints or {}
    out = []
    for n in names:
        low, need = n.lower(), int(seats.get(n, 1))
        doc = "PLACEHOLDER seat"
        if need > 1:
            shape = hints.get(n, "ring")
            if shape == "stack":
                out.append(Connector.stack(n, [length * (i + 0.5) / need for i in range(need)],
                                          doc=doc))
            elif shape == "radial":
                out.append(Connector.polar(n, max(od * 0.5, 1.0), need, length / 2.0,
                                          tilt_deg=90.0, doc=doc + " (radial)"))
            else:
                out.append(Connector.polar(n, max(od * 0.35, 1.0), need, length / 2.0, doc=doc))
        elif "face" in low or low in ("front", "rear", "top", "bottom", "end_a", "end_b"):
            rear = any(t in low for t in ("rear", "bottom", "back", "end_a"))
            out.append(Connector.face(n, 0.0 if rear else length, outward=not rear, doc=doc))
        elif "bore" in low or "axis" in low or low in ("mid", "centre", "center"):
            z = length / 2.0 if low in ("mid", "centre", "center") else 0.0
            out.append(Connector.at(n, z=z, kind="revolute" if "bore" in low else "rigid", doc=doc))
        elif "od" in low or "rim" in low or "outer" in low:
            out.append(Connector.at(n, x=od / 2.0, z=length / 2.0, ry=90.0, doc=doc))
        else:
            out.append(Connector.face(n, length, doc=doc))
    return out


def stub_result(skill: str, params: dict, seats: dict[str, int], names=None,
                hints: dict[str, str] | None = None):
    """A dimensioned envelope for `skill`, registered nowhere.

    Also the assembler's last resort: a *known* skill that throws at a spec's
    params gets one of these too, so one bad part costs you that part and not
    the other forty.
    """
    from .frames import SkillResult
    from .vocabulary._util import revolve_profile_solid

    names = tuple(names or _stub_names(seats))
    od, length, bore = _envelope(params)
    res = SkillResult(
        revolve_profile_solid([[od, 0.0], [od, length]], bore_d=bore),
        interfaces={"od": od, "length": length, "bore_d": bore},
        meta={"source": "stub", "placeholder": True, "requested": skill},
    )
    res.add(*_stub_connectors(seats, names, od, length, hints))
    return res


def install_stub(order: WorkOrder) -> str:
    """Register a dimensioned envelope for a skill that could not be synthesised.

    Never written to disk. On disk it would masquerade as vocabulary and the
    real part would never be attempted again.
    """
    names = _stub_names(order.seats)

    def _fn(**kw):
        return stub_result(order.skill, kw, order.seats, names, order.hints)

    _fn.__name__ = order.skill
    _fn.__doc__ = (f"PLACEHOLDER for '{order.skill}': a dimensioned envelope holding the "
                   "space, not the real part.")
    registry.skill(order.skill, category="placeholder", connectors=names,
                   params=_param_defs(order.observed), source="stub")(_fn)
    return order.skill


# ── persistence: the vocabulary keeps what it learns ────────────────────
PREAMBLE = '''"""Synthesised skill `{name}` — admitted by the geometry gate {when}.

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


'''


def persist(name: str, code: str) -> str:
    GENERATED.mkdir(parents=True, exist_ok=True)
    init = GENERATED / "__init__.py"
    if not init.exists():
        init.write_text(_LOADER, encoding="utf-8")
    path = GENERATED / f"{name}.py"
    path.write_text(PREAMBLE.format(name=name, when=time.strftime("on %Y-%m-%d"))
                    + code.rstrip() + "\n", encoding="utf-8")
    return str(path)


# ── the loop ────────────────────────────────────────────────────────────
@dataclass
class SynthResult:
    admitted: list[str] = field(default_factory=list)
    stubbed: list[str] = field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    written: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.stubbed

    def text(self) -> str:
        n = len(self.admitted) + len(self.stubbed)
        lines = [f"  synthesis: {n} missing skill(s), {self.calls} model call(s), "
                 f"{self.seconds:.1f}s"]
        for name in self.admitted:
            lines.append(f"    ADMIT  {name:24s} -> {Path(self.written.get(name, '')).name}")
        for name in self.stubbed:
            why = self.verdicts[name].why if name in self.verdicts else "no candidate returned"
            lines.append(f"    STUB   {name:24s} PLACEHOLDER: {why[:180]}")
        return "\n".join(lines + [f"    note: {t}" for t in self.notes])


async def fulfil(orders: list[WorkOrder], *, chat: Chat | None = None, tries: int = 2,
                 sweep: bool = False, persist_ok: bool = True,
                 log=None) -> SynthResult:
    """One request for every missing skill; a placeholder for whatever fails.

    `tries` is a request budget, not a per-skill budget: the second attempt
    carries only the skills that were rejected, with the gate's own words as
    the correction. Ten missing parts still cost one request when they work.
    """
    t0 = time.time()
    res = SynthResult()
    if not orders:
        return res
    if chat is None:
        from . import brain
        chat = brain._default_chat

    def say(msg: str) -> None:
        if log:
            log(msg)

    pending = {o.skill: o for o in orders}
    messages = [{"role": "system", "content": SKILL_CONTRACT},
                {"role": "user", "content": synth_prompt(orders)}]

    for attempt in range(1, max(1, tries) + 1):
        say(f"synth: requesting {len(pending)} skill(s) — {', '.join(sorted(pending))}")
        try:
            raw = await chat(messages)
        except Exception as exc:
            res.notes.append(f"model call failed: {type(exc).__name__}: {exc}")
            break
        res.calls += 1
        blocks = parse_skills(raw or "")
        if not blocks:
            res.notes.append("reply contained no skill code")
            break

        code_of: dict[str, str] = {}
        for key, code in blocks.items():
            if key == "":
                names, err = install(code, expect=None)
                if err and not names:
                    res.notes.append(f"batch block: {err}")
                for n in names:
                    code_of[n] = code
            elif key in pending:
                names, err = install(code, expect=key)
                if err:
                    res.notes.append(f"{key}: {err}")
                for n in names:
                    code_of[n] = code
            else:
                res.notes.append(f"ignored unrequested skill '{key}'")
                uninstall(key)

        for name in sorted(set(code_of) - set(pending)):
            res.notes.append(f"ignored unrequested skill '{name}'")
            uninstall(name)

        rejected: list[Verdict] = []
        for name in sorted(pending):
            if name not in code_of:
                continue
            try:
                v = admit(pending[name], sweep=sweep)
            except Exception as exc:
                # The referee itself must never be the thing that fails: a candidate
                # pathological enough to break `admit` is simply not admitted, and the
                # other skills in the batch still get their verdict.
                v = Verdict(name, problems=[f"the gate could not judge it: "
                                            f"{type(exc).__name__}: {exc}"])
            res.verdicts[name] = v
            if v.admitted:
                res.admitted.append(name)
                if persist_ok:
                    res.written[name] = persist(name, code_of[name])
                say(f"synth: ADMIT {name} ({v.seconds:.1f}s)")
                pending.pop(name)
            else:
                uninstall(name)
                rejected.append(v)
                say(f"synth: reject {name} — {v.why[:160]}")

        if not pending:
            break
        if attempt >= max(1, tries):
            break
        asks = "\n".join(f"### {v.skill}\n{v.why}" for v in rejected) or \
            "no code arrived for: " + ", ".join(sorted(pending))
        messages = messages[:2] + [
            {"role": "assistant", "content": (raw or "")[:6000]},
            {"role": "user", "content":
                "The geometry gate rejected these. Fix ONLY these and resend them in the "
                f"same format:\n\n{asks}\n\nRemember: seats must sit on real material, the "
                "part is built about +Z with its base at z=0, and every connector the work "
                "order named must exist with the right number of stations."},
        ]

    for name in sorted(pending):
        install_stub(pending[name])
        res.stubbed.append(name)
        say(f"synth: STUB {name} (placeholder envelope)")
    res.seconds = time.time() - t0
    return res


def fulfil_sync(orders: list[WorkOrder], **kw) -> SynthResult:
    import asyncio

    return asyncio.run(fulfil(orders, **kw))


def stub_all(orders: list[WorkOrder]) -> SynthResult:
    """Placeholders for everything, no model call. The offline path."""
    t0 = time.time()
    res = SynthResult()
    for o in orders:
        install_stub(o)
        res.stubbed.append(o.skill)
    res.notes.append("no model call: every missing skill is a placeholder envelope")
    res.seconds = time.time() - t0
    return res


# ── the auto-loader written into vocabulary/generated/ ──────────────────
_LOADER = '''"""Skills ADA synthesised for herself. Imported by bootstrap(), never by hand.

One file per skill, each admitted by the geometry gate before it was written
here. A file that fails to import is skipped and named in FAILED rather than
being allowed to take the whole vocabulary down with it.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from ...registry import REGISTRY

FAILED: list[tuple[str, str]] = []


def load() -> list[str]:
    """Import every generated module; return the skill names that appeared."""
    before = set(REGISTRY)
    here = Path(__file__).resolve().parent
    for mod in sorted(m.name for m in pkgutil.iter_modules([str(here)])):
        try:
            importlib.import_module(f"{__name__}.{mod}")
        except Exception as exc:
            FAILED.append((mod, f"{type(exc).__name__}: {exc}"))
    fresh = sorted(set(REGISTRY) - before)
    for name in fresh:
        REGISTRY[name].source = "synth"
    return fresh
'''


def load_generated() -> list[str]:
    """Called by bootstrap(). Silent when nothing has been synthesised yet."""
    if not (GENERATED / "__init__.py").exists():
        return []
    try:
        from .vocabulary import generated
    except Exception:
        return []
    try:
        return generated.load()
    except Exception:
        return []
