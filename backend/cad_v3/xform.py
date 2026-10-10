"""SE(3) helpers — the bridge between build123d Locations and plain matrices.

The assembler speaks build123d's `Location`, which is an OCCT `gp_Trsf` behind a
property wrapper. A least-squares solver speaks numpy. Every conversion in the
engine goes through this module rather than being open-coded, because the two
directions have to be exact inverses of each other: a solver that reads a pose,
improves it by 0.01 mm and writes back a pose that differs by 0.4 mm from what it
read would report convergence while making the assembly worse.

Rotations are carried as 3-vectors (axis * angle, radians) inside the solver.
That parametrisation has no gimbal lock and no redundant degree of freedom, which
matters because the residual Jacobian is estimated numerically: an Euler triple
would hand the optimiser a singular direction near ry = +-90 degrees, exactly the
pose every radial seat in the vocabulary sits at.
"""

from __future__ import annotations

import math

import numpy as np
from build123d import Location
from OCP.gp import gp_Trsf

IDENT_R = np.eye(3)
ZERO_T = np.zeros(3)


# ── Location <-> (R, t) ─────────────────────────────────────────────────────

def loc_to_Rt(loc: Location) -> tuple[np.ndarray, np.ndarray]:
    """A Location as a 3x3 rotation and a 3-vector translation, in mm."""
    t = loc.wrapped.Transformation()
    R = np.array([[t.Value(i, j) for j in range(1, 4)] for i in range(1, 4)], dtype=float)
    p = loc.position
    return R, np.array([float(p.X), float(p.Y), float(p.Z)], dtype=float)


def Rt_to_loc(R: np.ndarray, t: np.ndarray) -> Location:
    """The inverse of `loc_to_Rt`, exact to the last bit for a proper rotation.

    Goes through `gp_Trsf.SetValues` rather than through an Euler triple because
    `SetValues` stores the matrix the caller supplies; reconstructing from angles
    would re-derive it and lose the guarantee that reading a pose and writing it
    straight back is the identity.
    """
    tr = gp_Trsf()
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    tr.SetValues(float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(t[0]),
                 float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(t[1]),
                 float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(t[2]))
    return Location(tr)


# ── rotation vectors ────────────────────────────────────────────────────────

def rotvec_to_R(v) -> np.ndarray:
    """Rodrigues. `v` is axis * angle in radians."""
    v = np.asarray(v, dtype=float).reshape(3)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        # Second-order term kept so the derivative at zero is right; a bare
        # identity here would flatten the Jacobian on the very first iteration,
        # when the seed is already exact and every rotation variable is zero.
        K = _skew(v)
        return IDENT_R + K + 0.5 * (K @ K)
    k = v / th
    K = _skew(k)
    return IDENT_R + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    """The inverse of `rotvec_to_R`, branch-safe near 0 and near pi."""
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    cos_th = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    th = math.acos(cos_th)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if abs(math.pi - th) < 1e-6:
        # sin(th) -> 0, so the antisymmetric part carries no information. Take the
        # axis from the dominant column of (R + I), which stays well conditioned.
        A = R + IDENT_R
        col = int(np.argmax(np.linalg.norm(A, axis=0)))
        axis = A[:, col]
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.zeros(3)
        return axis / n * th
    s = math.sin(th)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * (th / (2.0 * s))


def _skew(v) -> np.ndarray:
    x, y, z = (float(q) for q in v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# ── frame algebra used by the solver and the residual reports ────────────────

def compose(Ra, ta, Rb, tb) -> tuple[np.ndarray, np.ndarray]:
    """Frame a applied to frame b — the matrix form of `Location * Location`."""
    return Ra @ Rb, Ra @ tb + ta


def invert(R, t) -> tuple[np.ndarray, np.ndarray]:
    Rt = np.asarray(R, dtype=float).T
    return Rt, -Rt @ np.asarray(t, dtype=float)


def pose_delta(R_want, t_want, R_have, t_have) -> tuple[np.ndarray, np.ndarray]:
    """(position error mm, rotation error rad) between two frames that should agree.

    The rotation error is expressed in the *wanted* frame, so a residual vector
    built from it has one component per axis of the seat: a revolute mate can drop
    the component about its own axis and keep the other two, which is exactly what
    "free to spin, rigid in every other sense" means.
    """
    R_want = np.asarray(R_want, dtype=float)
    dR = R_want.T @ np.asarray(R_have, dtype=float)
    return np.asarray(t_have, dtype=float) - np.asarray(t_want, dtype=float), R_to_rotvec(dR)


def euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Intrinsic X-Y-Z degrees, for human-readable logs only. Never round-tripped."""
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(abs(sy) - 1.0) < 1e-8:
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    else:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def axis_of(loc: Location) -> np.ndarray:
    """The frame's own +Z in world coordinates — the functional axis invariant."""
    R, _ = loc_to_Rt(loc)
    return R[:, 2]


def world_angle(point, centre, axis, reference=None) -> float:
    """Where `point` sits around `axis`, in degrees 0..360, measured in world space.

    This is the function that makes a patterned seat's station order independent of
    how the part carrying it was placed. Two co-axial bolt circles physically line
    up; whether the parts holding them were mated face-to-face (which flips one of
    them) is a fact about the mate chain and must not change which hole is hole 0.

    The axis sign is canonicalised first. Without that, a housing bolted onto a
    flange measures its own ring about -Z while the flange measures its about +Z,
    the two angle sets come out mirrored, and the fix that was supposed to make the
    holes line up lines them up backwards instead.
    """
    axis = canonical_axis(axis)
    if axis is None:
        return 0.0
    ref = _perp_reference(axis) if reference is None else np.asarray(reference, dtype=float)
    ref = ref - axis * float(ref @ axis)
    rn = float(np.linalg.norm(ref))
    if rn < 1e-12:
        return 0.0
    ref = ref / rn
    other = np.cross(axis, ref)
    v = np.asarray(point, dtype=float) - np.asarray(centre, dtype=float)
    v = v - axis * float(v @ axis)
    if float(np.linalg.norm(v)) < 1e-9:
        return 0.0
    deg = math.degrees(math.atan2(float(v @ other), float(v @ ref)))
    # Rounding before the wrap matters: a station at -1e-16 degrees would otherwise
    # come back as 360 and sort last instead of first, which silently rotates a
    # whole bolt circle by one station.
    deg = round(deg, 6) % 360.0
    return 0.0 if deg > 360.0 - 1e-6 else deg


def canonical_axis(axis) -> np.ndarray | None:
    """`axis` with a sign convention, so +Z and -Z are the same line.

    A direction and its opposite describe one axis. Every consumer here cares about
    the line, not the arrow: which way a bolt circle's normal happens to point is a
    property of the part that published it, and it must not decide which hole two
    parts consider hole 0.
    """
    v = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    v = v / n
    dominant = int(np.argmax(np.abs(v)))
    return -v if v[dominant] < 0 else v


def _perp_reference(axis: np.ndarray) -> np.ndarray:
    """A deterministic direction across `axis`: world +X, or +Y when +X is parallel."""
    if abs(float(axis @ np.array([1.0, 0.0, 0.0]))) < 0.95:
        return np.array([1.0, 0.0, 0.0])
    return np.array([0.0, 1.0, 0.0])
