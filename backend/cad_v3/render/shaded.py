"""Shaded offscreen renderer: what the model actually looks like, as a PNG.

The drawing module answers "what would a draughtsman ink?"; this one answers "what
does it look like?". There is no VTK or OpenGL in this environment, so the whole
pipeline is here: tessellate once, transform per variant with numpy, rasterise into a
z-buffer, then shade flat per triangle with two lights and a specular term.

Two details do most of the work visually. Triangles are lit two-sided (`abs` of the
Lambert term), because a tessellation that flips a normal on one face should not open a
black hole in a machined surface. And the outline comes from discontinuities in the
depth and part-id buffers rather than from geometry, which gives every part a crisp
edge for free — the same reason a CAD viewport looks sharper than a raw render.

Normals are averaged per vertex and interpolated across each triangle. OCCT hands back
one vertex set per face, so an averaged normal smooths the inside of a blade or a bore
without rounding over the sharp edge where two faces meet — a flat facet per triangle
turned a compressor wheel seen small into visual gravel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .. import materials
from ..assembler import Assembly, inner_keys
from .drawing import VIEWS

BACKDROP = ((0.972, 0.976, 0.984), (0.882, 0.898, 0.921))   # top, bottom
SS = 2                      # supersample factor, downsampled with Lanczos
XRAY_ALPHA = 0.34           # how much of a housing survives in front of a rotor


@dataclass
class Part:
    """One instance's mesh in its own frame, ready to be placed by any variant."""
    key: str
    skill: str
    verts: np.ndarray            # (n, 3) float32, millimetres
    norms: np.ndarray            # (n, 3) float32, unit, smoothed within a face
    tris: np.ndarray             # (m, 3) int32
    rgb: tuple[float, float, float]
    feature: bool


@dataclass
class ShadeResult:
    files: dict[str, int] = field(default_factory=dict)
    triangles: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"  {Path(p).name:34s} {n / 1024.0:9.1f} KB" for p, n in self.files.items()]
        lines.append(f"  {self.triangles} triangles, {self.seconds:.2f}s")
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


# ── geometry, tessellated once and reused by every variant ──────────────
def _smooth(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals.

    The cross product is not normalised before it is accumulated, so a big triangle
    votes proportionally harder than a sliver — that is what keeps a coarse
    tessellation from being dragged off by its own degenerate corners.
    """
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face = np.cross(b - a, c - a)
    acc = np.zeros_like(verts)
    for col in range(3):
        np.add.at(acc, tris[:, col], face)
    length = np.linalg.norm(acc, axis=1)
    live = length > 1e-12
    acc[live] /= length[live][:, None]
    acc[~live] = np.array([0.0, 0.0, 1.0], dtype=verts.dtype)
    return acc


def collect(asm: Assembly, *, linear: float = 0.08, angular: float = 0.25,
            xray: bool = False) -> list[Part]:
    """Mesh every instance in its own frame.

    Local rather than placed, so an exploded frame or a second camera costs one numpy
    matmul instead of another trip through the mesher.
    """
    from build123d import Location

    out: list[Part] = []
    inner = inner_keys(asm) if xray else set()
    for inst in asm.instances:
        feature = inst.key in inner
        try:
            shape = inst.solid.located(Location())
            verts, tris = shape.tessellate(linear, angular)
        except Exception:
            continue
        if not tris:
            continue
        r, g, b, _ = materials.rgba(inst.material, xray=xray, feature=feature)
        vs = np.array([(v.X, v.Y, v.Z) for v in verts], dtype=np.float32)
        ts = np.array(tris, dtype=np.int32)
        out.append(Part(inst.key, inst.skill, vs, _smooth(vs, ts), ts,
                        (r, g, b), feature))
    return out


def _matrix(loc) -> tuple[np.ndarray, np.ndarray]:
    t = loc.wrapped.Transformation()
    rot = np.array([[t.Value(1, 1), t.Value(1, 2), t.Value(1, 3)],
                    [t.Value(2, 1), t.Value(2, 2), t.Value(2, 3)],
                    [t.Value(3, 1), t.Value(3, 2), t.Value(3, 3)]], dtype=np.float32)
    return rot, np.array([t.Value(1, 4), t.Value(2, 4), t.Value(3, 4)], dtype=np.float32)


def place(asm: Assembly, parts: list[Part]) -> list[tuple[Part, np.ndarray, np.ndarray]]:
    """Apply each instance's current location — read fresh, so scopes are honoured.

    Normals get the rotation only. Every placement in this engine is a rigid
    `Location`, so there is no non-uniform scale to invert-transpose around.
    """
    by_key = {i.key: i for i in asm.instances}
    out = []
    for part in parts:
        inst = by_key.get(part.key)
        if inst is None:
            continue
        rot, off = _matrix(inst.solid.location)
        out.append((part, part.verts @ rot.T + off, part.norms @ rot.T))
    return out


# ── camera and rasteriser ───────────────────────────────────────────────
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _basis(view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, toward-eye). Orthographic, so this is the whole camera."""
    direction, up0 = VIEWS.get(view, VIEWS["front"])
    eye = _unit(direction)
    right = np.cross(_unit(up0), eye)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), eye)
    right = _unit(right)
    return right, _unit(np.cross(eye, right)), eye


def _raster(frames, view: str, width: int, height: int, *, only=None):
    """Z-buffer rasterise. Returns (rgb float32 HxWx3, depth, part-id, hit mask).

    One pass per part, one inner iteration per triangle, and the colour interpolated
    from the three vertex normals — so a bore reads as round rather than as a fan of
    facets, at the cost of three multiply-adds per covered pixel.
    """
    right, up, eye = _basis(view)
    pts = [np.stack([w @ right, w @ up, w @ eye], axis=1) for _, w, _ in frames]
    if not pts:
        return None
    allp = np.concatenate(pts, axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-3))
    pad = 0.06 * span
    scale = min((width - 1) / (hi[0] - lo[0] + 2 * pad),
                (height - 1) / (hi[1] - lo[1] + 2 * pad))
    cx = (width - 1) / 2.0 - scale * (lo[0] + hi[0]) / 2.0
    cy = (height - 1) / 2.0 + scale * (lo[1] + hi[1]) / 2.0

    depth = np.full((height, width), -np.inf, dtype=np.float32)
    color = np.zeros((height, width, 3), dtype=np.float32)
    ident = np.zeros((height, width), dtype=np.int32)

    key = _unit(right * -0.45 + up * 0.62 + eye * 0.75)
    fill = _unit(right * 0.7 + up * -0.25 + eye * 0.45)
    half = _unit(key + eye)

    for pid, ((part, _world, norm), proj) in enumerate(zip(frames, pts), start=1):
        if only is not None and part.feature is not only:
            continue
        tris = part.tris
        # Vertex colours for the whole part in one shot; the triangle loop only ever
        # indexes into this, so the lighting cost is per vertex, not per pixel.
        lam = np.abs(norm @ key) * 0.78 + np.abs(norm @ fill) * 0.30
        spec = np.clip(np.abs(norm @ half), 0.0, 1.0) ** 24 * 0.42
        vcol = np.clip(np.clip(lam + 0.12, 0.0, 1.35)[:, None]
                       * np.array(part.rgb, np.float32) + spec[:, None], 0.0, 1.0)

        sx = proj[:, 0] * scale + cx
        sy = -proj[:, 1] * scale + cy
        sz = proj[:, 2]
        px, py = sx[tris], sy[tris]
        x0 = np.clip(np.floor(px.min(axis=1)).astype(np.int32), 0, width - 1)
        x1 = np.clip(np.ceil(px.max(axis=1)).astype(np.int32), 0, width - 1)
        y0 = np.clip(np.floor(py.min(axis=1)).astype(np.int32), 0, height - 1)
        y1 = np.clip(np.ceil(py.max(axis=1)).astype(np.int32), 0, height - 1)
        area = ((px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0])
                - (px[:, 2] - px[:, 0]) * (py[:, 1] - py[:, 0]))
        pz = sz[tris]

        for t in np.nonzero(np.abs(area) > 1e-9)[0]:
            xs, xe, ys, ye = x0[t], x1[t] + 1, y0[t], y1[t] + 1
            if xs >= xe or ys >= ye:
                continue
            gx = np.arange(xs, xe, dtype=np.float32)[None, :]
            gy = np.arange(ys, ye, dtype=np.float32)[:, None]
            ax, ay = px[t, 0], py[t, 0]
            bx, by = px[t, 1], py[t, 1]
            cxx, cyy = px[t, 2], py[t, 2]
            inv = 1.0 / area[t]
            w1 = ((gx - ax) * (cyy - ay) - (gy - ay) * (cxx - ax)) * inv
            w2 = ((gy - ay) * (bx - ax) - (gx - ax) * (by - ay)) * inv
            w0 = 1.0 - w1 - w2
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            zz = w0 * pz[t, 0] + w1 * pz[t, 1] + w2 * pz[t, 2]
            win = inside & (zz > depth[ys:ye, xs:xe])
            if not win.any():
                continue
            ca, cb, cc = vcol[tris[t, 0]], vcol[tris[t, 1]], vcol[tris[t, 2]]
            shade = (w0[:, :, None] * ca + w1[:, :, None] * cb + w2[:, :, None] * cc)
            depth[ys:ye, xs:xe] = np.where(win, zz, depth[ys:ye, xs:xe])
            ident[ys:ye, xs:xe] = np.where(win, pid, ident[ys:ye, xs:xe])
            color[ys:ye, xs:xe][win] = np.clip(shade[win], 0.0, 1.0)
    return color, depth, ident, depth > -np.inf


# ── compositing ─────────────────────────────────────────────────────────
def _backdrop(height: int, width: int) -> np.ndarray:
    top = np.array(BACKDROP[0], dtype=np.float32)
    bottom = np.array(BACKDROP[1], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    return top[None, None, :] * (1.0 - ramp) + bottom[None, None, :] * ramp


def _outline(depth: np.ndarray, ident: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Edges where the part id changes or the depth *bends* — not merely slopes.

    A first-difference test on depth inks every steeply inclined surface, which shreds
    a turbine blade seen nearly edge-on: the depth across such a face changes faster
    per pixel than any silhouette tolerance small enough to be useful. The second
    difference is ~0 along a flat or gently curved run and spikes only where the
    surface actually steps away, so the slanted face stays clean and the occlusion
    behind it still gets its line.
    """
    live = depth[hit]
    span = float(live.max() - live.min()) if live.size else 1.0
    tol = max(span * 0.010, 1e-4)
    flat = np.where(hit, depth, 0.0)
    edge = np.zeros(depth.shape, dtype=bool)
    for axis in (0, 1):
        prev, nxt = np.roll(flat, 1, axis=axis), np.roll(flat, -1, axis=axis)
        hp, hn = np.roll(hit, 1, axis=axis), np.roll(hit, -1, axis=axis)
        # An id change is a silhouette by definition, including against the backdrop,
        # whose id is 0 — that is what gives every part its outline for free.
        edge |= (ident != np.roll(ident, 1, axis=axis)) & (hit | hp)
        edge |= hit & hp & hn & (np.abs(2.0 * flat - prev - nxt) > tol)
    return edge


def _ink(img: np.ndarray, edge: np.ndarray, strength: float = 1.0) -> np.ndarray:
    return np.where(edge[:, :, None], img * (1.0 - 0.72 * strength), img)


def _save(img: np.ndarray, path: str | Path, width: int, height: int) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pic = Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    if pic.size != (width, height):
        pic = pic.resize((width, height), Image.LANCZOS)
    pic.save(out, format="PNG", optimize=True)
    return out.stat().st_size


# ── the two picture modes ───────────────────────────────────────────────
def render_png(asm: Assembly, path: str | Path, *, view: str = "iso",
               xray: bool = False, width: int = 1500, height: int = 1050,
               parts: list[Part] | None = None, linear: float = 0.08,
               angular: float = 0.25) -> tuple[int, int]:
    """One shaded picture. Returns (bytes, triangles)."""
    mesh = parts if parts is not None else collect(asm, linear=linear, angular=angular,
                                                   xray=xray)
    frames = place(asm, mesh)
    if not frames:
        raise ValueError("nothing to render")
    tris = sum(int(p.tris.shape[0]) for p, _, _ in frames)
    w, h = width * SS, height * SS
    bg = _backdrop(h, w)

    if not xray:
        color, depth, ident, hit = _raster(frames, view, w, h)
        img = np.where(hit[:, :, None], color, bg)
        img = _ink(img, _outline(depth, ident, hit))
        return _save(img, path, width, height), tris

    # X-ray: the rotating group is rendered opaque and the housings are blended over
    # it, so a part is only ghosted where it actually stands in front of something.
    inner = _raster(frames, view, w, h, only=True)
    outer = _raster(frames, view, w, h, only=False)
    ci, di, ii, hi = inner
    co, do, io, ho = outer
    img = np.where(hi[:, :, None], np.clip(ci * 1.12, 0.0, 1.0), bg)
    img = _ink(img, _outline(di, ii, hi))
    ghost = ho & (~hi | (do > di))
    blended = XRAY_ALPHA * co + (1.0 - XRAY_ALPHA) * img
    img = np.where(ghost[:, :, None], blended, img)
    img = _ink(img, _outline(do, io, ho) & ghost, strength=0.45)
    return _save(img, path, width, height), tris


def export_shaded(asm: Assembly, outdir: str | Path, *, name: str | None = None,
                  views=("iso",), modes=("solid",), explode: float = 0.6,
                  width: int = 1500, height: int = 1050,
                  linear: float = 0.08, angular: float = 0.25) -> ShadeResult:
    """Shaded PNGs for the requested views. Modes: solid, xray, exploded."""
    from .exporters import explode_scope

    t0 = time.time()
    res = ShadeResult()
    if not asm.instances:
        res.notes.append("nothing to render")
        return res
    stem = name or asm.spec.name.replace(" ", "_").lower()
    cache: dict[bool, list[Part]] = {}
    for mode in modes:
        xray = mode == "xray"
        if xray not in cache:
            cache[xray] = collect(asm, linear=linear, angular=angular, xray=xray)
        for view in views:
            out = Path(outdir) / f"{stem}_{view}_{mode}.png"
            try:
                if mode == "exploded":
                    with explode_scope(asm, explode):
                        size, tris = render_png(asm, out, view=view, parts=cache[xray],
                                                width=width, height=height)
                else:
                    size, tris = render_png(asm, out, view=view, xray=xray,
                                            parts=cache[xray], width=width, height=height)
            except Exception as exc:
                res.notes.append(f"{view}/{mode} failed: {type(exc).__name__}: {exc}")
                continue
            res.files[str(out)] = size
            res.triangles = tris
    res.seconds = time.time() - t0
    return res





