"""The global placement solver — what makes a mate a constraint instead of a hint.

A breadth-first traversal can satisfy exactly one mate per part, because a tree
gives each node one parent. Every further statement a spec makes about that part
was therefore recorded and abandoned: the swashplate bolted to its actuator *and*
running on the mast got its pose from whichever of those two mates the traversal
reached first, and the other one was reported as a "loop closure" and left to be
wrong by however much it was wrong. Since all three shipped reference specs happen
to be pure trees, the number that measured this was structurally zero on every
example the engine shipped with.

This module solves the whole system at once instead. Every instance carries six
unknowns, every mate contributes up to six residual equations in the host seat's
own frame, and the traversal result is the seed — which is a very good seed, so
convergence is two or three iterations rather than a search.

Three deliberate choices:

* **Body-frame increments.** An instance's unknowns are a small translation and
  rotation applied on the *right* of its seeded pose, so a rotation variable turns
  a part about its own origin. Left multiplication would turn it about the world
  origin, giving a part 400 mm out a 400 mm lever on its own rotation variables and
  a Jacobian conditioned accordingly.

* **Frozen station pairing.** Which station of a bolt circle an instance occupies
  is decided once, by the traversal, from world angles. Re-deriving it inside the
  residual would let a station swap mid-solve, which turns a smooth least-squares
  problem into a combinatorial one.

* **Accept only an improvement.** A genuinely contradictory spec — two mates that
  disagree by 467 mm — has no solution, and spreading that error evenly across
  forty parts would replace one visible mistake with forty invisible ones. So the
  solve is kept only if it converges or at least halves the worst residual;
  otherwise the deterministic traversal poses stand and the report says by how much
  the design does not close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from build123d import Location

from . import xform
from .schema import WORLD, MateSpec

# Radians are turned into millimetres for the residual vector by this lever, so a
# tilt and an offset are traded off against each other on comparable terms. Derived
# from the assembly's own size: 1 degree matters more on a 3 m rotor than on a
# 60 mm cartridge, and no fixed constant is right for both.
DEFAULT_LEVER_MM = 25.0
MIN_LEVER_MM = 5.0
MAX_LEVER_MM = 250.0


@dataclass
class Pair:
    """One (host frame, child instance) coupling that a mate asks to coincide."""

    idx: int
    host_kind: str          # "part" | "frame"
    host_key: str           # instance key, or datum id
    l_a: Location           # seat * M, in the host's frame
    child_key: str
    l_b: Location           # the child's own seat, in the child's frame
    free: tuple[bool, bool, bool, bool, bool, bool] = (False,) * 6
    solve: bool = True
    label: str = ""
    station: int = 0


@dataclass
class Solution:
    residual_before_mm: float = 0.0
    residual_after_mm: float = 0.0
    angle_before_deg: float = 0.0
    angle_after_deg: float = 0.0
    bodies: int = 0
    equations: int = 0
    rank: int = 0
    iterations: int = 0
    accepted: bool = False
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def free_dof(self) -> int:
        return max(0, 6 * self.bodies - self.rank)

    @property
    def redundant(self) -> int:
        return max(0, self.equations - self.rank)

    def text(self) -> str:
        state = "closed" if self.accepted else "left to the traversal"
        return (f"solve: {self.bodies} bodies, {self.equations} equations, rank {self.rank} "
                f"({self.free_dof} free DOF, {self.redundant} redundant) — "
                f"worst mate {self.residual_before_mm:.3f} -> {self.residual_after_mm:.3f} mm, "
                f"{state} in {self.iterations} iteration(s), {self.seconds:.2f}s")


# ── enumerating what the spec actually asked for ────────────────────────────

def pairs_of(asm) -> list[Pair]:
    """Every coupling in the spec, including the ones the traversal could not use.

    Built once per assembly and cached, because the station pairing inside it is a
    decision rather than a derivation: it was taken by the traversal from world
    angles, and re-taking it against poses the solver is in the middle of changing
    would let a bolt hop between holes.
    """
    cached = getattr(asm, "_pairs", None)
    if cached is not None:
        return cached
    from . import assembler as A

    spec = asm.spec
    out: list[Pair] = []
    for idx, mate in enumerate(spec.mates):
        node_a, node_b = spec.node_of(mate.a), spec.node_of(mate.b)
        # Host is whichever side the traversal treated as already placed. Instances
        # record the mate that placed them, so ask them rather than guessing.
        moved_pid = _moved_side(asm, idx, node_a, node_b)
        if moved_pid is None:
            continue
        host_ref, moved_ref = (mate.b, mate.a) if moved_pid == node_a else (mate.a, mate.b)
        host_id, host_conn = MateSpec.split(host_ref)
        moved_id, moved_conn = MateSpec.split(moved_ref)
        host_is_frame = MateSpec.is_frame(host_ref)
        if MateSpec.is_frame(moved_ref):
            continue  # a datum is defined, never placed by a mate
        if host_is_frame:
            host_id = MateSpec.frame_id(host_ref)
            if host_id not in asm.frames:
                continue
        hosts = [] if host_is_frame else asm.of_part(host_id)
        moved = asm.of_part(moved_id)
        if not moved or (not hosts and not host_is_frame):
            continue
        for j, child in enumerate(moved):
            if host_is_frame:
                host_loc, ca = asm.frames[host_id], A._identity_connector(host_conn or "origin")
                host_key = host_id
            else:
                host = hosts[j % len(hosts)]
                host_loc, ca, host_key = host.solid.location, host.result.connectors.get(host_conn), host.key
            cb = child.result.connectors.get(moved_conn)
            if ca is None or cb is None:
                continue
            station = (child.station if child.via_mate == idx
                       else _station_for(host_loc, ca, mate, child.index))
            seat = A._reorient(ca.location(station), mate.orient)
            flip = A._flip_for(mate, ca, cb)
            out.append(Pair(
                idx=idx, host_kind="frame" if host_is_frame else "part", host_key=host_key,
                l_a=seat * A._mate_transform(mate, flip), child_key=child.key,
                l_b=cb.location(0), free=mate.free_axes, solve=bool(mate.solve),
                label=f"{mate.a} <-> {mate.b}", station=station,
            ))
    asm._pairs = out
    return out


def _moved_side(asm, idx: int, node_a: str, node_b: str) -> str | None:
    """Which of the mate's two sides is the one that gets moved."""
    for inst in asm.instances:
        if inst.via_mate == idx:
            return inst.pid
    # Never used by the traversal: both ends were already placed, so the mate is a
    # closure. Orient it the way the spec wrote it and let the solver decide.
    if node_b != WORLD and asm.of_part(node_b):
        return node_b
    if node_a != WORLD and asm.of_part(node_a):
        return node_a
    return None


def _station_for(host_loc, ca, mate, child_index: int) -> int:
    from . import assembler as A

    order = A._station_order(host_loc, ca, mate)
    return order[child_index % len(order)] if ca.count > 1 else 0


# ── measuring ───────────────────────────────────────────────────────────────

def _host_pose(asm, pair: Pair, poses: dict[str, tuple]) -> tuple | None:
    if pair.host_kind == "part":
        return poses.get(pair.host_key)
    owner = asm.frame_owner.get(pair.host_key, "")
    local = asm.frame_local.get(pair.host_key)
    if local is None:
        loc = asm.frames.get(pair.host_key)
        return None if loc is None else xform.loc_to_Rt(loc)
    Rl, tl = xform.loc_to_Rt(local)
    if not owner:
        return (Rl, tl)
    base = poses.get(owner)
    if base is None:
        return None
    return xform.compose(base[0], base[1], Rl, tl)


def _pair_error(asm, pair: Pair, poses: dict[str, tuple]) -> tuple[float, float] | None:
    """(position error mm, angle error degrees) for one coupling at the given poses."""
    host = _host_pose(asm, pair, poses)
    child = poses.get(pair.child_key)
    if host is None or child is None:
        return None
    Rla, tla = xform.loc_to_Rt(pair.l_a)
    Rlb, tlb = xform.loc_to_Rt(pair.l_b)
    Rw, tw = xform.compose(host[0], host[1], Rla, tla)
    Rh, th = xform.compose(child[0], child[1], Rlb, tlb)
    dt = Rw.T @ (th - tw)
    dr = xform.R_to_rotvec(Rw.T @ Rh)
    keep = [not f for f in pair.free]
    pos = math.sqrt(sum(float(dt[i]) ** 2 for i in range(3) if keep[i]))
    ang = math.sqrt(sum(float(dr[i]) ** 2 for i in range(3) if keep[3 + i]))
    return (pos, math.degrees(ang))


def measure(asm) -> list:
    """One `Placement` per mate, holding how far it is from what it claimed.

    This replaces the old `residual_mm`, which recomposed the traversal's own algebra
    and was therefore zero by construction for every mate the traversal solved — a
    self-check dressed up as a measurement. Here the poses are read back off the
    solids, so a mate reads zero only if the geometry really does satisfy it.
    """
    from .assembler import Placement

    poses = {i.key: xform.loc_to_Rt(i.solid.location) for i in asm.instances}
    worst: dict[int, tuple[float, float]] = {}
    for pair in pairs_of(asm):
        err = _pair_error(asm, pair, poses)
        if err is None:
            continue
        prev = worst.get(pair.idx, (0.0, 0.0))
        worst[pair.idx] = (max(prev[0], err[0]), max(prev[1], err[1]))
    out: list[Placement] = []
    solved_idx = {i.via_mate for i in asm.instances if i.via_mate >= 0}
    for idx, mate in enumerate(asm.spec.mates):
        mm, deg = worst.get(idx, (0.0, 0.0))
        prior = asm.placement(idx)
        out.append(Placement(
            idx=idx, label=f"{mate.a} <-> {mate.b}",
            before_mm=prior.before_mm if prior else mm,
            before_deg=prior.before_deg if prior else deg,
            after_mm=mm, after_deg=deg,
            solved=idx in solved_idx, kind=mate.kind,
        ))
    return out


# ── solving ─────────────────────────────────────────────────────────────────

def _lever(asm) -> float:
    """How many millimetres one radian of misalignment is worth in the residual."""
    try:
        lo = [1e18, 1e18, 1e18]
        hi = [-1e18, -1e18, -1e18]
        for inst in asm.instances:
            bb = inst.solid.bounding_box()
            for k, (a, b) in enumerate(((bb.min.X, bb.max.X), (bb.min.Y, bb.max.Y),
                                        (bb.min.Z, bb.max.Z))):
                lo[k] = min(lo[k], float(a))
                hi[k] = max(hi[k], float(b))
        diag = math.dist(lo, hi)
    except Exception:
        return DEFAULT_LEVER_MM
    if not math.isfinite(diag) or diag <= 0:
        return DEFAULT_LEVER_MM
    return max(MIN_LEVER_MM, min(MAX_LEVER_MM, diag / 8.0))


def solve(asm, *, tol: float | None = None, max_nfev: int = 400) -> Solution:
    """Refine every pose until every mate holds, or report that they cannot.

    Seeded from the traversal, so this is a correction rather than a search: with no
    loops it confirms the traversal in one residual evaluation and returns, and with
    loops it distributes the closure error the way the joints allow.
    """
    import time

    t0 = time.time()
    sol = Solution()
    spec = asm.spec
    tol = float(spec.tolerance_mm if tol is None else tol)
    pairs = [p for p in pairs_of(asm) if p.solve]
    if not pairs:
        asm.placements = measure(asm)
        asm.solved = True
        return sol

    poses = {i.key: xform.loc_to_Rt(i.solid.location) for i in asm.instances}
    before = _worst(asm, pairs, poses)
    sol.residual_before_mm, sol.angle_before_deg = before
    if before[0] <= tol and before[1] <= 1e-4:
        asm.placements = measure(asm)
        asm.solved = True
        sol.accepted = True
        sol.seconds = time.time() - t0
        sol.bodies = 0
        asm.solver_log.append(
            f"solve: every mate already holds to {before[0]:.2e} mm — "
            f"the traversal is exact and there is nothing to distribute")
        return sol

    free_keys = _free_bodies(asm, pairs)
    if not free_keys:
        asm.placements = measure(asm)
        sol.notes.append("nothing free to move: every body is the datum or unmated")
        sol.seconds = time.time() - t0
        return sol

    lever = _lever(asm)
    prob = _Problem(asm, pairs, free_keys, poses, lever)
    x0 = np.zeros(6 * len(free_keys))
    sol.bodies = len(free_keys)
    sol.equations = prob.rows

    try:
        from scipy.optimize import least_squares

        res = least_squares(prob.residual, x0, jac_sparsity=prob.sparsity(),
                            method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=max_nfev)
        x, sol.iterations = res.x, int(getattr(res, "nfev", 0))
        sol.rank = _rank(res.jac)
    except ImportError:
        x, sol.iterations = _gauss_newton(prob, x0)
        sol.rank = _rank(prob.jacobian(x))

    after_poses = prob.poses(x)
    after = _worst(asm, pairs, after_poses)
    # Polish, never smear. Two mates that disagree by 467 mm have no solution, and a
    # least-squares fit will happily split the difference — replacing one visible
    # mistake with forty invisible ones and telling the repair loop the residual
    # halved. So a solve is only kept if it lands inside a small multiple of the
    # tolerance; anything worse is a contradiction in the design and is reported as
    # one, with the traversal's deterministic poses left standing.
    polish_mm = max(10.0 * tol, tol)
    keep = after[0] < before[0] and after[0] <= polish_mm
    sol.residual_after_mm, sol.angle_after_deg = after if keep else before
    sol.accepted = bool(keep and after[0] <= tol)

    if keep:
        _write_back(asm, after_poses)
    else:
        sol.notes.append(
            f"reverted: the best fit still leaves {after[0]:.2f} mm, so the spec "
            "over-constrains itself and the traversal's deterministic poses stand")

    asm.placements = measure(asm)
    asm.solved = sol.accepted
    asm.dof = (f"{sol.free_dof} free DOF, {sol.redundant} redundant equation(s) "
               f"of {sol.equations}")
    asm.solver_log.append(sol.text())
    for note in sol.notes:
        asm.solver_log.append(f"  {note}")
    sol.seconds = time.time() - t0
    return sol


def _worst(asm, pairs: list[Pair], poses: dict[str, tuple]) -> tuple[float, float]:
    mm = deg = 0.0
    for pair in pairs:
        err = _pair_error(asm, pair, poses)
        if err is None:
            continue
        mm, deg = max(mm, err[0]), max(deg, err[1])
    return (mm, deg)


def _free_bodies(asm, pairs: list[Pair]) -> list[str]:
    """Instances the solver may move: everything mated, except the datum part.

    The datum is pinned because a system of purely relative constraints is invariant
    under a global rigid motion, and six null directions in the Jacobian would make
    the rank report meaningless even though the fit would still converge.
    """
    fixed = {i.key for i in asm.instances if i.pid == asm.spec.datum}
    touched: set[str] = set()
    for pair in pairs:
        touched.add(pair.child_key)
        if pair.host_kind == "part":
            touched.add(pair.host_key)
        else:
            owner = asm.frame_owner.get(pair.host_key, "")
            if owner:
                touched.add(owner)
    return [i.key for i in asm.instances if i.key in touched and i.key not in fixed]


def _write_back(asm, poses: dict[str, tuple]) -> None:
    for inst in asm.instances:
        pose = poses.get(inst.key)
        if pose is None:
            continue
        inst.solid.location = xform.Rt_to_loc(pose[0], pose[1])
    # Datums measured from a part have to follow it.
    for fid, owner in asm.frame_owner.items():
        local = asm.frame_local.get(fid)
        if local is None:
            continue
        if not owner:
            asm.frames[fid] = local
            continue
        inst = asm.by_key(owner)
        if inst is not None:
            asm.frames[fid] = inst.solid.location * local


def _rank(jac) -> int:
    try:
        J = jac.toarray() if hasattr(jac, "toarray") else np.asarray(jac)
        if J.size == 0:
            return 0
        return int(np.linalg.matrix_rank(J, tol=1e-8 * max(1.0, float(np.abs(J).max()))))
    except Exception:
        return 0


class _Problem:
    """The residual system, precompiled so the optimiser's inner loop is arithmetic."""

    def __init__(self, asm, pairs: list[Pair], free_keys: list[str],
                 seed: dict[str, tuple], lever: float):
        self.asm = asm
        self.lever = float(lever)
        self.seed = seed
        self.slot = {k: n for n, k in enumerate(free_keys)}
        self.cols = 6 * len(free_keys)
        self.terms: list[tuple] = []
        rows = 0
        for pair in pairs:
            child = seed.get(pair.child_key)
            if child is None:
                continue
            host = self._host_static(pair)
            if host is None:
                continue
            Rla, tla = xform.loc_to_Rt(pair.l_a)
            Rlb, tlb = xform.loc_to_Rt(pair.l_b)
            keep = [i for i, f in enumerate(pair.free) if not f]
            if not keep:
                continue
            self.terms.append((pair, host, Rla, tla, Rlb, tlb, keep, rows))
            rows += len(keep)
        self.rows = rows

    # ── how a host's frame is reached ───────────────────────────────────
    def _host_static(self, pair: Pair):
        """('part', key) or ('frame', owner_key_or_None, Rl, tl)."""
        if pair.host_kind == "part":
            return ("part", pair.host_key, None, None)
        local = self.asm.frame_local.get(pair.host_key)
        if local is None:
            loc = self.asm.frames.get(pair.host_key)
            if loc is None:
                return None
            R, t = xform.loc_to_Rt(loc)
            return ("fixed", None, R, t)
        Rl, tl = xform.loc_to_Rt(local)
        owner = self.asm.frame_owner.get(pair.host_key, "")
        if not owner:
            return ("fixed", None, Rl, tl)
        return ("frame", owner, Rl, tl)

    # ── unknowns -> poses ───────────────────────────────────────────────
    def poses(self, x) -> dict[str, tuple]:
        x = np.asarray(x, dtype=float)
        out = dict(self.seed)
        for key, n in self.slot.items():
            Rs, ts = self.seed[key]
            d = x[6 * n:6 * n + 3]
            w = x[6 * n + 3:6 * n + 6]
            R = Rs @ xform.rotvec_to_R(w)
            out[key] = (R, Rs @ d + ts)
        return out

    def _frame(self, static, poses):
        kind, key, Rl, tl = static
        if kind == "part":
            return poses.get(key)
        if kind == "fixed":
            return (Rl, tl)
        base = poses.get(key)
        if base is None:
            return None
        return xform.compose(base[0], base[1], Rl, tl)

    # ── the residual ────────────────────────────────────────────────────
    def residual(self, x) -> np.ndarray:
        poses = self.poses(x)
        out = np.zeros(self.rows)
        for pair, static, Rla, tla, Rlb, tlb, keep, row in self.terms:
            host = self._frame(static, poses)
            child = poses.get(pair.child_key)
            if host is None or child is None:
                continue
            Rw, tw = xform.compose(host[0], host[1], Rla, tla)
            Rh, th = xform.compose(child[0], child[1], Rlb, tlb)
            dt = Rw.T @ (th - tw)
            dr = xform.R_to_rotvec(Rw.T @ Rh) * self.lever
            for k, axis in enumerate(keep):
                out[row + k] = dt[axis] if axis < 3 else dr[axis - 3]
        return out

    def sparsity(self):
        """Which unknowns each equation can possibly touch.

        Without this the optimiser probes all 6N unknowns for every column of the
        Jacobian; with it, scipy colours the graph and a 57-body assembly costs a
        dozen residual evaluations per iteration instead of three hundred.
        """
        from scipy.sparse import lil_matrix

        S = lil_matrix((max(1, self.rows), max(1, self.cols)), dtype=float)
        for pair, static, _Rla, _tla, _Rlb, _tlb, keep, row in self.terms:
            cols: list[int] = []
            for key in (pair.child_key, static[1] if static[0] != "fixed" else None):
                if key is None or key not in self.slot:
                    continue
                n = self.slot[key]
                cols += list(range(6 * n, 6 * n + 6))
            for k in range(len(keep)):
                for c in cols:
                    S[row + k, c] = 1.0
        return S

    def jacobian(self, x) -> np.ndarray:
        """Forward differences — only used when scipy is unavailable."""
        x = np.asarray(x, dtype=float)
        base = self.residual(x)
        J = np.zeros((self.rows, self.cols))
        for c in range(self.cols):
            step = 1e-6 if c % 6 < 3 else 1e-7
            probe = x.copy()
            probe[c] += step
            J[:, c] = (self.residual(probe) - base) / step
        return J


def _gauss_newton(prob: _Problem, x0, rounds: int = 12) -> tuple[np.ndarray, int]:
    """A dependency-free fallback so placement never needs scipy to be installed."""
    x = np.asarray(x0, dtype=float).copy()
    for it in range(rounds):
        r = prob.residual(x)
        if float(np.abs(r).max()) < 1e-9:
            return x, it + 1
        J = prob.jacobian(x)
        try:
            step, *_ = np.linalg.lstsq(J, -r, rcond=1e-10)
        except np.linalg.LinAlgError:
            return x, it + 1
        x = x + step
    return x, rounds



