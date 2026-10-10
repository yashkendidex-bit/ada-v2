"""Instanced, level-of-detail glTF written directly from the tessellation.

`export_gltf` hands the whole compound to OCCT's writer, which meshes every solid it
is given — so the shipped turbocharger arrives as 57 meshes and 448k triangles even
though six of those studs are the same stud six times. This module meshes each
*distinct* part once and references it from as many nodes as the pattern has seats,
which is the same trick the assembler already relies on: instances of one part id
share a cache key, so they share geometry exactly.

The output deliberately matches what OCCT emits, because the viewer already loads
those files: millimetres are scaled to metres, and a root node carries the -90 deg
X rotation that takes CAD's Z-up into glTF's Y-up. A file from here and a file from
`export_gltf_file` drop into the same scene graph unchanged.
"""

from __future__ import annotations

import json
import math
import struct
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from build123d import Location

from .. import materials
from ..assembler import Assembly, inner_keys

# (linear deflection mm, angular deflection rad). "high" matches the default export;
# each step down roughly halves the triangle count of a curved part.
LEVELS: dict[str, tuple[float, float]] = {
    "high": (0.06, 0.25),
    "mid": (0.25, 0.5),
    "low": (0.9, 0.9),
}

MM_TO_M = 0.001
# glTF is Y-up. -90 deg about X as a quaternion, byte-identical to OCCT's own root.
ZUP_TO_YUP = [-0.7071067811865475, 0.0, 0.0, 0.7071067811865475]


@dataclass
class LodResult:
    files: dict[str, int] = field(default_factory=dict)
    triangles: dict[str, int] = field(default_factory=dict)
    nodes: int = 0
    meshes: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = []
        for path, size in self.files.items():
            tri = self.triangles.get(path, 0)
            lines.append(f"  {Path(path).name:34s} {size / 1024.0:9.1f} KB  "
                         f"{tri:7d} tri")
        lines.append(f"  {self.meshes} unique meshes -> {self.nodes} nodes, "
                     f"{self.seconds:.2f}s")
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


# ── geometry ────────────────────────────────────────────────────────────
def _local(inst):
    """The instance's geometry with its placement stripped off.

    Placement in this engine is pure `Location` composition — the assembler mates by
    setting a location, never by baking a transform into the B-rep — so resetting the
    location recovers the shape the skill built. That is what makes two seats of one
    bolt circle share a single mesh.
    """
    try:
        return inst.solid.located(Location())
    except Exception:
        return inst.solid


def _tessellate(shape, linear: float, angular: float):
    """(positions, normals, indices) as flat arrays, ready for a buffer view."""
    verts, tris = shape.tessellate(linear, angular)
    pos = array("f")
    for v in verts:
        pos.extend((v.X * MM_TO_M, v.Y * MM_TO_M, v.Z * MM_TO_M))
    idx = array("I")
    acc = [[0.0, 0.0, 0.0] for _ in verts]
    for a, b, c in tris:
        idx.extend((a, b, c))
        # Area-weighted face normal, accumulated per vertex. OCCT hands back one
        # vertex set per face, so this smooths inside a cylinder without rounding
        # over the sharp edge where two faces meet.
        ax, ay, az = pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2]
        ux, uy, uz = pos[b * 3] - ax, pos[b * 3 + 1] - ay, pos[b * 3 + 2] - az
        vx, vy, vz = pos[c * 3] - ax, pos[c * 3 + 1] - ay, pos[c * 3 + 2] - az
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        for i in (a, b, c):
            acc[i][0] += nx
            acc[i][1] += ny
            acc[i][2] += nz
    nrm = array("f")
    for nx, ny, nz in acc:
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-20:
            nrm.extend((0.0, 0.0, 1.0))
        else:
            nrm.extend((nx / length, ny / length, nz / length))
    return pos, nrm, idx


def _matrix(loc) -> list[float]:
    """A glTF column-major node matrix from an OCCT location, mm translation to m."""
    t = loc.wrapped.Transformation()
    return [t.Value(1, 1), t.Value(2, 1), t.Value(3, 1), 0.0,
            t.Value(1, 2), t.Value(2, 2), t.Value(3, 2), 0.0,
            t.Value(1, 3), t.Value(2, 3), t.Value(3, 3), 0.0,
            t.Value(1, 4) * MM_TO_M, t.Value(2, 4) * MM_TO_M,
            t.Value(3, 4) * MM_TO_M, 1.0]


def _geometry_key(asm: Assembly, inst) -> tuple:
    """What has to match for two instances to share one mesh.

    Skill plus params is the same identity the worker's cache uses, so this dedupes
    two differently-named parts that happen to be the same M8 nut as well as the six
    seats of one bolt circle. Material joins the key because a glTF primitive owns its
    material — a bronze copy of a steel part is a different mesh whatever its shape.
    """
    part = asm.spec.part(inst.pid)
    if part is None:
        return (inst.pid, inst.material)
    return (inst.skill, json.dumps(part.params, sort_keys=True, default=str),
            inst.material)


def _material_json(name: str, *, xray: bool, feature: bool) -> dict:
    r, g, b, a = materials.rgba(name, xray=xray, feature=feature)
    mat = {"name": name,
           "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, a],
                                    "metallicFactor": 0.85 if not xray else 0.2,
                                    "roughnessFactor": 0.35},
           "doubleSided": True}
    if a < 1.0:
        mat["alphaMode"] = "BLEND"
    return mat


def write_glb(asm: Assembly, path: str | Path, *, linear: float = 0.06,
              angular: float = 0.25, xray: bool = False,
              name: str | None = None) -> tuple[int, int, int, int]:
    """Write one instanced .glb. Returns (bytes, triangles, meshes, nodes)."""
    groups: dict[tuple, list] = {}
    for inst in asm.instances:
        groups.setdefault(_geometry_key(asm, inst), []).append(inst)
    inner = inner_keys(asm) if xray else set()

    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    meshes: list[dict] = []
    mats: list[dict] = []
    mat_index: dict[str, int] = {}
    nodes: list[dict] = [{}]                      # 0 is the Z-up -> Y-up root
    children: list[int] = []
    triangles = 0

    def _view(data: array, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(data.tobytes())
        views.append({"buffer": 0, "byteOffset": offset,
                      "byteLength": len(blob) - offset, "target": target})
        return len(views) - 1

    for key, insts in groups.items():
        head = insts[0]
        # One mesh serves the whole group, so one material must too: if any station of
        # a pattern is enclosed, the set is treated as internal.
        feature = any(i.key in inner for i in insts)
        try:
            pos, nrm, idx = _tessellate(_local(head), linear, angular)
        except Exception:
            continue
        if not idx:
            continue
        mjson = _material_json(head.material, xray=xray, feature=feature)
        # Keyed on the finished material, not on the part's material name: two skills
        # that resolve to the same colour share one glTF material instead of two
        # identical ones.
        mkey = json.dumps(mjson["pbrMetallicRoughness"], sort_keys=True)
        if mkey not in mat_index:
            mat_index[mkey] = len(mats)
            mats.append(mjson)

        v_idx = _view(idx, 34963)                 # ELEMENT_ARRAY_BUFFER
        v_pos = _view(pos, 34962)                 # ARRAY_BUFFER
        v_nrm = _view(nrm, 34962)
        count = len(pos) // 3
        lo = [min(pos[i::3]) for i in range(3)]
        hi = [max(pos[i::3]) for i in range(3)]
        a_idx = len(accessors)
        accessors.append({"bufferView": v_idx, "componentType": 5125,
                          "count": len(idx), "type": "SCALAR"})
        accessors.append({"bufferView": v_pos, "componentType": 5126,
                          "count": count, "type": "VEC3", "min": lo, "max": hi})
        accessors.append({"bufferView": v_nrm, "componentType": 5126,
                          "count": count, "type": "VEC3"})
        meshes.append({"name": head.pid,
                       "primitives": [{"attributes": {"POSITION": a_idx + 1,
                                                      "NORMAL": a_idx + 2},
                                       "indices": a_idx,
                                       "material": mat_index[mkey]}]})
        mesh_id = len(meshes) - 1
        triangles += len(idx) // 3 * len(insts)
        for inst in insts:
            children.append(len(nodes))
            nodes.append({"name": inst.key, "mesh": mesh_id,
                          "matrix": _matrix(inst.solid.location)})

    nodes[0] = {"name": name or asm.spec.name, "rotation": ZUP_TO_YUP,
                "children": children}
    doc = {
        "asset": {"version": "2.0",
                  "generator": f"ADA cad_v3 tessellate (instanced, linear={linear})"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes,
        "meshes": meshes, "materials": mats, "accessors": accessors,
        "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    total = 12 + 8 + len(raw) + 8 + len(blob)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(raw), 0x4E4F534A))
        fh.write(raw)
        fh.write(struct.pack("<II", len(blob), 0x004E4942))
        fh.write(bytes(blob))
    return out.stat().st_size, triangles, len(meshes), len(children)


def export_lod(asm: Assembly, outdir: str | Path, *, name: str | None = None,
               levels=("high", "mid", "low"), xray: bool = False) -> LodResult:
    """One .glb per level of detail, all sharing the same node graph.

    A viewer can load `_low` first and swap in `_high` once it is on screen; because
    every level is written from the same instances with the same names, the swap is a
    file substitution rather than a scene rebuild.
    """
    t0 = time.time()
    res = LodResult()
    stem = name or asm.spec.name.replace(" ", "_").lower()
    if not asm.instances:
        res.notes.append("nothing to tessellate")
        return res
    for level in levels:
        linear, angular = LEVELS.get(level, LEVELS["high"])
        out = Path(outdir) / f"{stem}_{level}.glb"
        try:
            size, tri, meshes, nodes = write_glb(asm, out, linear=linear,
                                                 angular=angular, xray=xray, name=stem)
        except Exception as exc:
            res.notes.append(f"{level} failed: {type(exc).__name__}: {exc}")
            continue
        res.files[str(out)] = size
        res.triangles[str(out)] = tri
        res.meshes, res.nodes = meshes, nodes
    res.seconds = time.time() - t0
    return res




