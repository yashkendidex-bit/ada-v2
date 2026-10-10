"""The assembler: turns a coordinate-free Spec into a placed B-rep assembly.

Placement happens in two passes, and the second one is the reason a part can now
land where the drawing says it does.

**Pass one — traversal.** Breadth-first from the assembly origin over a graph
whose nodes are the parts *and* the declared datums, so a parent (or a reference
frame) is always resolved before anything measured from it. Each mate contributes
`world_child = world_host . L_a . M . L_b^-1`, where `M` carries the six relative
numbers a mate may state — three offsets and three rotations, all read in the host
seat's own frame — plus a 180 degree flip when both sides are real surfaces,
because two solids in contact must have opposing normals.

**Pass two — solve.** A tree traversal can only satisfy one mate per part, so
every additional statement about a part used to be recorded and abandoned. Now the
whole system is handed to a least-squares solver (solver.py) seeded from the
traversal: loops close, over-constraint is distributed instead of ignored, and the
residual that comes back is a measurement rather than a zero by construction.

One further correctness rule lives here. A patterned seat's stations are declared
in the part's local frame, so a part that was itself mated face-to-face carries a
bolt circle whose station 0 has moved in world space — which used to mis-pair every
fastener on the far side of a flip by a fixed angle. Stations are therefore ordered
by their *world* angle about the seat's axis before instances are handed out, which
makes the pairing independent of how the host got where it is.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from build123d import Compound, Location, RigidJoint

from . import graph, materials, registry, validate, xform
from .frames import Connector, SkillResult
from .schema import WORLD, MateSpec, Spec, SpecError
from .worker import Worker


@dataclass
class Instance:
    pid: str
    index: int
    key: str
    skill: str
    material: str
    label: str
    solid: Any
    result: SkillResult
    placed: bool = False
    via: str = ""
    # Which spec line put it here, and on whose seat. `via` is prose for the log; these
    # three are the machine-readable version, because a repair that wants to re-aim a
    # part has to name the exact mate — and the mate that aimed it wrongly is often two
    # or three links up the chain from the part that ended up in the wrong place.
    via_mate: int = -1
    via_host: str = ""
    via_seat: str = ""
    # Which station of a patterned host seat this instance took. Recorded rather than
    # recomputed because the station order depends on where the host ended up, and by
    # the time anything measures the result the host may have moved again.
    station: int = 0


@dataclass
class LoopClosure:
    """A mate the traversal met after both its parts were already placed.

    Real assemblies are deliberately over-constrained — a swashplate is bolted to its
    actuator *and* running on the mast — so a second mate onto an already-placed part is
    not an error. The traversal cannot satisfy it, because a tree gives each part exactly
    one parent; the solver can, and does. What is recorded here is which mates the tree
    left unsatisfied, so the solve has a work list and the report can say what it cost.
    """

    idx: int
    host_id: str
    host_seat: str
    moved_id: str
    moved_seat: str


@dataclass
class Placement:
    """Where one mate ended up, once everything else had its say.

    `before` is what the traversal alone produced and `after` what survives the
    solve. Both are kept because the difference is the most informative number in
    the report: a mate whose residual the solver had to spread across the assembly
    is a mate the design over-specifies, and that is a fact about the design.
    """

    idx: int
    label: str
    before_mm: float = 0.0
    after_mm: float = 0.0
    before_deg: float = 0.0
    after_deg: float = 0.0
    solved: bool = False
    kind: str = "rigid"


@dataclass
class Assembly:
    spec: Spec
    instances: list[Instance] = field(default_factory=list)
    compound: Any = None
    errors: list[SpecError] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    seconds: float = 0.0
    worker: Worker | None = None
    loop_mates: list[LoopClosure] = field(default_factory=list)
    # Declared reference frames, in world coordinates. `WORLD` is always present.
    frames: dict[str, Location] = field(default_factory=dict)
    # Which instance each datum rides on, and its pose in that instance's frame.
    # A datum measured from a part has to follow that part when the solver moves it;
    # an empty owner means the datum is rooted at the assembly origin and never moves.
    frame_owner: dict[str, str] = field(default_factory=dict)
    frame_local: dict[str, Location] = field(default_factory=dict)
    placements: list[Placement] = field(default_factory=list)
    solved: bool = False
    solver_log: list[str] = field(default_factory=list)
    dof: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors and self.compound is not None

    def by_key(self, key: str) -> Instance | None:
        for inst in self.instances:
            if inst.key == key:
                return inst
        return None

    def of_part(self, pid: str) -> list[Instance]:
        return [i for i in self.instances if i.pid == pid]

    def placement(self, idx: int) -> Placement | None:
        for p in self.placements:
            if p.idx == idx:
                return p
        return None

    def frame_of(self, ref: str) -> Location | None:
        """The world frame a mate side or a dimension anchor names.

        Accepts `@datum`, `part_id`, and `part_id.connector`, which is every form a
        spec can write, so callers that measure things (verify, conditions, layout
        comparison) never have to re-implement the lookup.
        """
        ref = str(ref)
        if MateSpec.is_frame(ref):
            return self.frames.get(MateSpec.frame_id(ref))
        pid, conn = MateSpec.split(ref)
        insts = self.of_part(pid)
        if not insts:
            return self.frames.get(pid)
        inst = insts[0]
        if not conn:
            return inst.solid.location
        c = inst.result.connectors.get(conn)
        if c is None:
            return inst.solid.location
        return inst.solid.location * c.location(0)


# ── which parts are inside which, measured rather than named ─────────────────
# X-ray mode exists to show internals: the casing goes translucent and whatever it
# encloses stays hot and opaque. Deciding that from a list of skill names worked for
# exactly one machine — the seven names on the list were a turbocharger's — and did
# nothing at all for a gearbox, a pump or a rotor head. Enclosure is a geometric fact
# about a particular assembly, so measure it there.
#
# Boxes rather than solids, because a housing is a shell and its box is exactly the
# volume it wraps; a boolean containment test reports a hollow casing as containing
# nothing. Coverage is sampled on a grid rather than integrated, because the union of
# N boxes has no closed form and a part that runs through three housings is enclosed
# by their union, not by any one of them.

_GRID = 5


def _bounds(inst: Instance) -> tuple[float, ...] | None:
    """(x0, y0, z0, x1, y1, z1, volume) in plain floats, or None if unmeasurable."""
    try:
        bb = inst.solid.bounding_box()
        lo, hi = bb.min, bb.max
    except Exception:
        return None
    x0, y0, z0 = float(lo.X), float(lo.Y), float(lo.Z)
    x1, y1, z1 = float(hi.X), float(hi.Y), float(hi.Z)
    vol = max((x1 - x0) * (y1 - y0) * (z1 - z0), 1e-9)
    return (x0, y0, z0, x1, y1, z1, vol)


def inner_keys(asm: Assembly, *, coverage: float = 0.55, ratio: float = 1.6) -> set[str]:
    """Instance keys that larger parts substantially enclose.

    `coverage` is how much of an instance's own box must fall inside the union of the
    candidates; `ratio` is how much bigger a candidate must be before it counts as an
    enclosure at all, so two interlocking parts of similar size do not each declare
    the other a housing.
    """
    boxes = [(i.key, b) for i in asm.instances if (b := _bounds(i)) is not None]
    out: set[str] = set()
    for key, me in boxes:
        x0, y0, z0, x1, y1, z1, vol = me
        bigger = [b for k, b in boxes
                  if k != key and b[6] >= ratio * vol
                  and b[0] < x1 and b[3] > x0 and b[1] < y1
                  and b[4] > y0 and b[2] < z1 and b[5] > z0]
        if not bigger:
            continue
        need = coverage * _GRID ** 3
        hits = 0
        for i in range(_GRID):
            x = x0 + (x1 - x0) * (i + 0.5) / _GRID
            for j in range(_GRID):
                y = y0 + (y1 - y0) * (j + 0.5) / _GRID
                for k in range(_GRID):
                    z = z0 + (z1 - z0) * (k + 0.5) / _GRID
                    for b in bigger:
                        if (b[0] <= x <= b[3] and b[1] <= y <= b[4]
                                and b[2] <= z <= b[5]):
                            hits += 1
                            break
        if hits >= need:
            out.add(key)
    return out


def _reorient(seat: Location, orient: str) -> Location:
    """Re-aim a station frame without authoring a pose.

    `seat` is whatever the skill published. When a part has to reach out of the axis
    and the only seat on offer points along it, the spec needs some way to say so —
    and the one thing it may never say is a coordinate. So the statement is made
    relative to the seat itself: keep the station where it is, keep its position on
    the ring, and rebuild only its facing.

    The construction is the same composition `Connector.polar` uses — rotate about
    the part's Z to the station's own angle, then tilt about the new local Y — so a
    reoriented seat is indistinguishable from one the skill author declared radial,
    and `offset_mm` keeps meaning "along this seat's outward normal" (which is now
    the radius). A station on the axis has no radial direction, so it is left alone.
    """
    if orient not in ("radial", "axial"):
        return seat
    p = seat.position
    x, y, z = float(p.X), float(p.Y), float(p.Z)
    if math.hypot(x, y) < 1e-6:
        return seat
    deg = math.degrees(math.atan2(y, x))
    spin = Location((0.0, 0.0, 0.0), (0.0, 0.0, deg))
    tilt = Location((0.0, 0.0, 0.0), (0.0, 90.0 if orient == "radial" else 0.0, 0.0))
    return Location((x, y, z)) * spin * tilt


def _mate_transform(mate, flip: bool) -> Location:
    """The relative freedom a mate has, composed explicitly and never guessed.

    Six numbers, all read in the host seat's frame: `dx/dy/dz` across and along
    it, `rx/ry/rz` about its axes. `offset_mm` and `angle_deg` are the original
    two-number shorthand and are simply added to `dz` and `rz`, so every spec ever
    written still means exactly what it meant.

    Order is translate, then Rz, then Ry, then Rx, then the contact flip. It has to
    be fixed and it has to be documented, because `dx=10, rz=90` and `rz=90, dx=10`
    put a part in two different places and a spec that cannot predict which is not
    a spec.
    """
    m = Location((float(mate.dx), float(mate.dy), float(mate.total_dz)))
    if mate.total_rz:
        m = m * Location((0, 0, 0), (0.0, 0.0, float(mate.total_rz)))
    if mate.ry:
        m = m * Location((0, 0, 0), (0.0, float(mate.ry), 0.0))
    if mate.rx:
        m = m * Location((0, 0, 0), (float(mate.rx), 0.0, 0.0))
    if flip:
        m = m * Location((0, 0, 0), (180.0, 0.0, 0.0))
    return m


def _flip_for(mate, ca: Connector | None, cb: Connector | None) -> bool:
    """Whether this mate turns the child over.

    `auto` keeps the old rule — two outward-facing surfaces must oppose — and the
    two overrides exist because that rule is an inference about what a seat *means*.
    A repair round that has measured a part pointing into its housing should be able
    to say so directly instead of hunting for a seat whose `flip` flag is wrong.
    """
    if mate.flip == "yes":
        return True
    if mate.flip == "no":
        return False
    return bool((ca is not None and ca.flip) or (cb is not None and cb.flip))


# ── which station of a ring receives which instance ─────────────────────────
# A patterned seat declares its stations in the part's own frame. The moment that
# part is itself mated face-to-face, the 180 degree contact flip moves station 0
# somewhere else in world space — so six studs on a flange and six washers on the
# housing bolted to it, both authored as "station i takes instance i", came out a
# fixed angle apart. Physically those holes line up; which of them is called hole 0
# is an accident of authoring order and must not decide anything.
#
# So the stations of a ring are sorted by world angle about the seat's own axis,
# measured from a fixed world reference, before instances are handed out. Two
# co-axial rings then agree whatever chain of flips brought them together, and a
# ring in an unflipped part keeps exactly the order it always had.

def _station_order(host_loc: Location, ca: Connector, mate) -> list[int]:
    """Station indices in the order instances should occupy them."""
    n = ca.count
    if n < 2:
        return [0]
    if mate.stations:
        return [int(s) % n for s in mate.stations]
    if mate.station_order == "local" or not ca.layout.startswith("ring"):
        return list(range(n))
    try:
        Rh, th = xform.loc_to_Rt(host_loc)
        pts = []
        for i in range(n):
            p = (host_loc * ca.location(i)).position
            pts.append([float(p.X), float(p.Y), float(p.Z)])
        axis = Rh @ [0.0, 0.0, 1.0]
        centre = [sum(p[k] for p in pts) / n for k in range(3)]
        keyed = [(round(xform.world_angle(p, centre, axis), 6), i)
                 for i, p in enumerate(pts)]
        keyed.sort()
        return [i for _, i in keyed]
    except Exception:
        return list(range(n))


def _seat_of(inst: "Instance", ca: Connector, mate, child_index: int) -> Location:
    """The host frame this mate hands the child, before `M` is applied."""
    order = _station_order(inst.solid.location, ca, mate)
    station = order[child_index % len(order)] if ca.count > 1 else 0
    return _reorient(ca.location(station), mate.orient)


def _identity_connector(name: str = "origin") -> Connector:
    """A datum's seat: the frame itself, and never a surface, so it never flips."""
    return Connector(name, [Location()], "rigid", (0.0, 0.0, 1.0),
                     "declared reference frame", False)


def _connector(inst: Instance, name: str) -> Connector | None:
    return inst.result.connectors.get(name)


def _envelope_for(spec: Spec, part):
    """A placeholder for a part whose own skill just threw.

    The skill is registered and was gated, so this is not a vocabulary gap — it
    is a parameter combination its geometry cannot survive. Substituting an
    envelope keeps the failure local: the error is still reported, verify still
    names the part as a PLACEHOLDER, and the other forty parts still assemble,
    export and render instead of being lost to one bad revolve.
    """
    try:
        from . import synth

        return synth.stub_result(part.skill, part.params, synth.seats_of(spec, part.id),
                                 hints=synth.seat_hints(spec, part.id))
    except Exception:
        return None


def build(spec: Spec, *, worker: Worker | None = None, strict: bool = True,
          solve: bool = True) -> Assembly:
    """Validate, build every part, place every part, then solve the whole system."""
    t0 = time.time()
    asm = Assembly(spec=spec, worker=worker or Worker())
    if not spec.baked:
        spec.resolve()
    asm.errors = validate.validate(spec)
    if asm.errors and strict:
        asm.seconds = time.time() - t0
        return asm

    # ── 1. geometry ─────────────────────────────────────────────────────
    holes: list[str] = []
    for part in spec.parts:
        params = spec.params_of(part)
        try:
            res = asm.worker.build(part.skill, params)
        except Exception as exc:
            asm.errors.append(SpecError(f"parts[{part.id}]", "SKILL_FAILED",
                                        f"{type(exc).__name__}: {exc}",
                                        hint=registry.signature_text(part.skill)))
            res = _envelope_for(spec, part)
            if res is None:
                holes.append(part.id)
                continue
        n = max(1, part.count)
        for i in range(n):
            solid = asm.worker.instance(res)
            key = part.id if n == 1 else f"{part.id}#{i}"
            solid.label = key
            asm.instances.append(Instance(
                pid=part.id, index=i, key=key, skill=part.skill,
                material=part.material, label=part.label or part.id,
                solid=solid, result=res,
            ))
    if holes and strict:
        asm.seconds = time.time() - t0
        return asm

    # ── 2. traversal ────────────────────────────────────────────────────
    _traverse(asm, spec)

    unplaced = [i.key for i in asm.instances if not i.placed]
    if unplaced:
        asm.errors.append(SpecError("mates", "UNPLACED",
                                    f"{len(unplaced)} instance(s) never received a pose: {unplaced[:8]}",
                                    hint="add a mate that reaches them from the datum"))

    # ── 3. solve ────────────────────────────────────────────────────────
    # The traversal satisfied one mate per part. Everything else the spec stated is
    # still only stated, so hand the whole system to the solver with the traversal
    # as its seed. With no loops there is nothing to distribute and the solve is a
    # no-op that costs one residual evaluation — which is worth paying, because it
    # is also the only independent check that the traversal did what it claims.
    from . import solver

    try:
        asm.placements = solver.measure(asm)
        if solve and not unplaced:
            solver.solve(asm)
    except Exception as exc:  # a solver failure must never lose the geometry
        asm.solver_log.append(f"solve skipped: {type(exc).__name__}: {exc}")

    # ── 4. one compound, coloured by material ───────────────────────────
    kids = []
    for inst in asm.instances:
        try:
            inst.solid.label = inst.key
            inst.solid.color = _color(inst)
        except Exception:
            pass
        kids.append(inst.solid)
    if kids:
        asm.compound = Compound(children=kids, label=spec.name)

    asm.log.append(asm.worker.summary())
    asm.log.append(f"{len(asm.instances)} instances, {len(spec.mates)} mates, "
                   f"{len(spec.datums)} datums, {len(asm.loop_mates)} loop closures")
    asm.log += asm.solver_log
    asm.seconds = time.time() - t0
    _LAST_BUILT["asm"] = asm
    _LAST_BUILT["spec"] = spec
    return asm


# The most recent build, kept so a physics or scene request does not have to rebuild.
# Rebuilding a 57-part model from its spec costs ~17 seconds, and the caller that wants
# to test or render it has just watched it be built. `modes.last_built()` reads this.
_LAST_BUILT: dict = {"asm": None, "spec": None}


def last_built():
    """(spec, assembly) for the most recent build in this process, or (None, None).

    Deliberately in-memory and deliberately not persisted: a stale assembly is worse
    than a rebuild, because it would report on geometry that no longer exists. If the
    process restarted, callers fall back to loading the spec from disk.
    """
    return _LAST_BUILT.get("spec"), _LAST_BUILT.get("asm")


def _traverse(asm: Assembly, spec: Spec) -> None:
    """Breadth-first placement over parts and datums together.

    The graph is rooted at `WORLD` rather than at the datum part, so a declared
    reference frame measured from the assembly origin is a first-class node. That
    is what lets a spec say "the bed's top face is the origin; the tailstock sits
    288.6 mm along it" instead of hanging the tailstock off whatever part happens
    to be next to it.
    """
    asm.frames[WORLD] = Location()
    edges = spec.graph_edges()
    for parent_id, child_id, idx in graph.mate_order(WORLD, edges):
        kind, i = spec.link_kind(idx)
        if kind == "root":
            for inst in asm.of_part(spec.datum):
                if not inst.placed:
                    inst.placed = True
                    inst.via = "datum"
            continue
        if kind == "datum":
            _place_datum(asm, spec, i, parent_id, child_id)
            continue
        _place_mate(asm, spec, i, parent_id)


def _place_datum(asm: Assembly, spec: Spec, i: int, parent_id: str, child_id: str) -> None:
    """Resolve one declared frame against whatever it is measured from."""
    dat = spec.datums[i]
    if child_id != dat.id:
        # Reached from the datum's own side: the frame is defined by its parent, not
        # the other way round, so there is nothing to solve here. A part that should
        # be positioned *by* a datum is written as a mate onto `@datum`.
        return
    base = asm.frames.get(parent_id)
    if base is None and dat.parent_seat:
        base = asm.frame_of(dat.on)
    if base is None:
        insts = asm.of_part(parent_id)
        base = insts[0].solid.location if insts else None
    if base is None:
        asm.errors.append(SpecError(f"datums[{i}]", "DATUM_UNRESOLVED",
                                    f"'{dat.id}' is measured from '{dat.on or WORLD}', "
                                    "which has no pose",
                                    hint="datums resolve in declaration order from the origin"))
        return
    if dat.parent_seat:
        seat = asm.frame_of(dat.on)
        if seat is not None:
            base = seat
    offs = Location((float(dat.dx), float(dat.dy), float(dat.dz)),
                    (float(dat.rx), float(dat.ry), float(dat.rz)))
    asm.frames[dat.id] = base * _layout_shift(spec, dat) * offs
    _own_frame(asm, spec, dat, parent_id)


def _own_frame(asm: Assembly, spec: Spec, dat, parent_id: str) -> None:
    """Record which instance this datum rides on, so it follows when that part moves."""
    owner = ""
    if parent_id != WORLD:
        if parent_id in asm.frames and spec.frame(parent_id) is not None:
            owner = asm.frame_owner.get(parent_id, "")
        else:
            insts = asm.of_part(parent_id)
            owner = insts[0].key if insts else ""
    asm.frame_owner[dat.id] = owner
    if owner:
        inst = asm.by_key(owner)
        base = inst.solid.location if inst is not None else Location()
        asm.frame_local[dat.id] = base.inverse() * asm.frames[dat.id]
    else:
        asm.frame_local[dat.id] = asm.frames[dat.id]


def _layout_shift(spec: Spec, dat) -> Location:
    """A datum's in-plane position as solved by the 2D design plan, if it has one.

    This is the join between the two halves of the workflow. The human approves a
    dimensioned 2D layout; every datum that names one of its points inherits that
    point's solved coordinates here, so the 3D model is positioned by the drawing
    that was reviewed rather than by a second, independent guess.
    """
    if not dat.point or not spec.layout:
        return Location()
    from . import layout as layout_mod

    pt = layout_mod.solved_point(spec.layout, dat.point)
    if pt is None:
        return Location()
    return Location((float(pt[0]), float(pt[1]), float(pt[2])))


def _place_mate(asm: Assembly, spec: Spec, idx: int, parent_id: str) -> None:
    """Satisfy one mate, with the already-resolved side as the host."""
    mate = spec.mates[idx]
    node_a, node_b = spec.node_of(mate.a), spec.node_of(mate.b)
    if node_a == parent_id:
        host_ref, moved_ref = mate.a, mate.b
    else:
        host_ref, moved_ref = mate.b, mate.a
    host_id, host_conn_name = MateSpec.split(host_ref)
    moved_id, moved_conn_name = MateSpec.split(moved_ref)

    # A datum on the moved side is a statement, not a placement: the frame already
    # has a definition of its own. Record it so the solver and the report can treat
    # it as the measurement it is.
    if MateSpec.is_frame(moved_ref):
        asm.loop_mates.append(LoopClosure(idx, host_id, host_conn_name,
                                          MateSpec.frame_id(moved_ref), ""))
        return

    host_frame = asm.frames.get(MateSpec.frame_id(host_ref)) if MateSpec.is_frame(host_ref) else None
    if MateSpec.is_frame(host_ref) and host_frame is None:
        asm.errors.append(SpecError(f"mates[{idx}]", "FRAME_MISSING",
                                    f"datum '{host_ref}' has no pose when this mate is reached",
                                    hint="declare the datum before whatever measures from it"))
        return

    hosts = asm.of_part(host_id) if host_frame is None else []
    moved = asm.of_part(moved_id)
    if (not hosts and host_frame is None) or not moved:
        return
    if all(m.placed for m in moved):
        asm.loop_mates.append(LoopClosure(idx, host_id, host_conn_name,
                                          moved_id, moved_conn_name))
        return

    for j, child in enumerate(moved):
        if child.placed:
            continue
        if host_frame is not None:
            host_loc, ca = host_frame, _identity_connector(host_conn_name or "origin")
        else:
            host = hosts[j % len(hosts)]
            host_loc, ca = host.solid.location, _connector(host, host_conn_name)
        cb = _connector(child, moved_conn_name)
        if ca is None or cb is None:
            asm.errors.append(SpecError(
                f"mates[{idx}]", "CONNECTOR_MISSING",
                f"'{host_conn_name}' or '{moved_conn_name}' not published at build time",
                hint=(f"host publishes {sorted(hosts[j % len(hosts)].result.connectors)}; "
                      if hosts else "") + f"child publishes {sorted(child.result.connectors)}",
            ))
            continue

        order = _station_order(host_loc, ca, mate)
        station = order[child.index % len(order)] if ca.count > 1 else 0
        seat = _reorient(ca.location(station), mate.orient)
        flip = _flip_for(mate, ca, cb)
        l_a = seat * _mate_transform(mate, flip)
        l_b = cb.location(0)

        try:
            target = host_loc * l_a * l_b.inverse()
            child.solid.location = target
        except Exception as exc:
            asm.errors.append(SpecError(f"mates[{idx}]", "MATE_FAILED",
                                        f"{type(exc).__name__}: {exc}"))
            continue
        child.placed = True
        host_key = host_ref if host_frame is not None else hosts[j % len(hosts)].key
        child.via = f"{host_key}.{host_conn_name} -> {moved_conn_name}".replace("..", ".")
        child.via_mate = idx
        child.via_host = host_key
        child.via_seat = host_conn_name
        child.station = station


def _color(inst: Instance):
    from build123d import Color

    r, g, b, a = materials.rgba(inst.material)
    return Color(r, g, b, a)
