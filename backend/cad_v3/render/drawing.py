"""Technical-drawing mode: hidden-line orthographic views straight from the B-rep.

OCCT's HLR algorithm gives real visible/hidden edge sets, so this is a genuine
engineering projection rather than a screenshot with a filter. The SVG is written
by hand instead of through `ExportSVG` because a drawing sheet needs things the
edge exporter has no concept of: part numbers, leader lines and a title block.

Model millimetres map 1:1 onto SVG user units, and the whole drawing lives inside
one `scale(1,-1)` group so Z runs up the page the way a draughtsman expects.
"""

from __future__ import annotations

import copy
from pathlib import Path
from xml.sax.saxutils import escape

from build123d import Compound, GeomType, Keep, Plane

from ..assembler import Assembly

VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front":  ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "rear":   ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "right":  ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left":   ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top":    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "iso":    ((1.0, -1.0, 0.72), (0.0, 0.0, 1.0)),
}


def _instance_shapes(asm: Assembly, *, cutaway: bool = False,
                     plane: Plane = Plane.XZ, keep: Keep = Keep.BOTTOM,
                     keep_whole: set[str] | None = None):
    """Yield `(instance, shape)` pairs, sectioning each part individually.

    Sectioning per instance rather than on the merged compound is what lets a
    cutaway keep its part identity, so a drawing can be both sectioned *and*
    labelled — reference images 2 and 4 at the same time.
    """
    whole = keep_whole or set()
    for inst in asm.instances:
        if not cutaway or inst.skill in whole:
            yield inst, inst.solid
            continue
        try:
            piece = inst.solid.split(plane, keep=keep)
        except Exception:
            piece = None
        if piece is None:
            continue
        for s in (piece if isinstance(piece, (list, tuple)) else [piece]):
            if s is not None:
                yield inst, s


def cutaway_compound(asm: Assembly, *, plane: Plane = Plane.XZ,
                     keep: Keep = Keep.BOTTOM, keep_whole: set[str] | None = None) -> Compound:
    """Section every housing, leave the named skills solid — reference image 2."""
    kids = []
    for inst, shape in _instance_shapes(asm, cutaway=True, plane=plane, keep=keep,
                                        keep_whole=keep_whole):
        s = copy.deepcopy(shape) if shape is inst.solid else shape
        s.label = inst.key
        kids.append(s)
    return Compound(children=kids, label=f"{asm.spec.name}_section")


def _camera(shape, view: str, distance: float = 40.0):
    """Camera pose for one view.

    `project_to_viewport` builds an `HLRAlgo_Projector` with no focus, which is a
    *parallel* projector, so the distance only has to be big enough to put the eye
    clear of the solid — it sets the view direction and nothing else. That is what
    lets the sheet claim scale 1:1 honestly: model millimetres are page
    millimetres, with no perspective term to spread the outline.
    """
    direction, up = VIEWS.get(view, VIEWS["front"])
    bb = shape.bounding_box()
    look = bb.center()
    span = max(bb.size.X, bb.size.Y, bb.size.Z, 1.0)
    d = span * distance
    return (look.X + direction[0] * d, look.Y + direction[1] * d, look.Z + direction[2] * d), up, look
    direction, up = VIEWS.get(view, VIEWS["front"])
    bb = shape.bounding_box()
    look = bb.center()
    span = max(bb.size.X, bb.size.Y, bb.size.Z, 1.0)
    d = span * distance
    return (look.X + direction[0] * d, look.Y + direction[1] * d, look.Z + direction[2] * d), up, look


def _sample(edge, tol: float) -> list[tuple[float, float]]:
    try:
        if edge.geom_type == GeomType.LINE:
            n = 1
        else:
            n = max(4, min(96, int(edge.length / max(tol, 0.02))))
        return [(p.X, p.Y) for p in (edge.position_at(i / n) for i in range(n + 1))]
    except Exception:
        return []


def _polys(edges, tol: float) -> list[list[tuple[float, float]]]:
    out = []
    for e in edges:
        pts = _sample(e, tol)
        if len(pts) >= 2:
            out.append(pts)
    return out


def _path_d(poly: list[tuple[float, float]]) -> str:
    head = f"M{poly[0][0]:.3f},{poly[0][1]:.3f}"
    return head + "".join(f"L{x:.3f},{y:.3f}" for x, y in poly[1:])


def technical_svg(asm: Assembly, path: str | Path, *, view: str = "front",
                  hidden: bool = True, labels: bool = False, cutaway: bool = False,
                  tol: float = 0.35, title: str | None = None,
                  keep_whole: set[str] | None = None) -> Path:
    """Write one hidden-line view. `labels` adds exploded-view style callouts."""
    shape = cutaway_compound(asm, keep_whole=keep_whole) if cutaway else asm.compound
    origin, up, look = _camera(shape, view)

    vis: list[list[tuple[float, float]]] = []
    hid: list[list[tuple[float, float]]] = []
    anchors: list[tuple[str, float, float]] = []

    if labels:
        # One callout per instance, but a sectioned part can project as several
        # lumps, so centroids accumulate per key and the number is fixed by
        # assembly order rather than by projection order.
        numbers = {inst.key: n for n, inst in enumerate(asm.instances, 1)}
        acc: dict[str, tuple[str, float, float, int]] = {}
        for inst, piece in _instance_shapes(asm, cutaway=cutaway, keep_whole=keep_whole):
            try:
                v, h = piece.project_to_viewport(origin, up, look)
            except Exception:
                continue
            pv = _polys(v, tol)
            vis += pv
            if hidden:
                hid += _polys(h, tol)
            pts = [p for poly in pv for p in poly]
            if not pts:
                continue
            text, sx, sy, k = acc.get(inst.key,
                                      (f"{numbers[inst.key]}  {inst.label}", 0.0, 0.0, 0))
            acc[inst.key] = (text,
                             sx + sum(p[0] for p in pts),
                             sy + sum(p[1] for p in pts),
                             k + len(pts))
        anchors = [(t, sx / k, sy / k) for t, sx, sy, k in acc.values() if k]
    else:
        v, h = shape.project_to_viewport(origin, up, look)
        vis = _polys(v, tol)
        hid = _polys(h, tol) if hidden else []

    pts = [p for poly in vis + hid for p in poly] or [(0.0, 0.0)]
    x0 = min(p[0] for p in pts); x1 = max(p[0] for p in pts)
    y0 = min(p[1] for p in pts); y1 = max(p[1] for p in pts)
    pad = 16.0
    label_w = 92.0 if anchors else 0.0
    w = (x1 - x0) + 2 * pad + label_w
    h_page = (y1 - y0) + 2 * pad + 26.0
    tx, ty = pad - x0, pad + y1

    o: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}mm" height="{h_page:.1f}mm" '
        f'viewBox="0 0 {w:.1f} {h_page:.1f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<g transform="translate({tx:.2f},{ty:.2f}) scale(1,-1)" fill="none" '
        'stroke="#111111" stroke-linecap="round" stroke-linejoin="round">',
    ]
    if hid:
        o.append('<g stroke-width="0.16" stroke-dasharray="1.6,1.1" stroke="#7a7a7a">')
        o += [f'<path d="{_path_d(p)}"/>' for p in hid]
        o.append("</g>")
    o.append('<g stroke-width="0.42">')
    o += [f'<path d="{_path_d(p)}"/>' for p in vis]
    o += ["</g>", "</g>"]

    if anchors:
        lx = tx + (x1 - x0) + pad * 0.6
        step = max((y1 - y0) / max(1, len(anchors)), 4.2)
        o.append('<g font-family="Helvetica,Arial,sans-serif" font-size="3.1" fill="#111111">')
        for i, (text, cx, cy) in enumerate(sorted(anchors, key=lambda a: -a[2])):
            ly = pad + 4 + i * step
            o.append(f'<path d="M{tx + cx:.2f},{ty - cy:.2f}L{lx - 2:.2f},{ly - 1:.2f}" '
                     'stroke="#9a9a9a" stroke-width="0.14" fill="none"/>')
            o.append(f'<circle cx="{tx + cx:.2f}" cy="{ty - cy:.2f}" r="0.55" fill="#111111"/>')
            o.append(f'<text x="{lx:.2f}" y="{ly:.2f}">{escape(text)}</text>')
        o.append("</g>")

    name = title or asm.spec.name
    o.append(f'<g font-family="Helvetica,Arial,sans-serif" fill="#111111">'
             f'<path d="M{pad:.1f},{h_page - 18:.1f}H{w - pad:.1f}" stroke="#111111" '
             f'stroke-width="0.5" fill="none"/>'
             f'<text x="{pad:.1f}" y="{h_page - 12:.1f}" font-size="4.6">{escape(name)}</text>'
             f'<text x="{pad:.1f}" y="{h_page - 6:.1f}" font-size="2.9" fill="#555555">'
             f'{len(asm.instances)} parts &#183; view {escape(view)} &#183; scale 1:1 &#183; mm '
             f'&#183; third angle</text></g>')
    o.append("</svg>")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(o), encoding="utf-8")
    return p
