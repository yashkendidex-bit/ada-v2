"""Spike 3b: is ADA's turbocharger a real assembly, or art that looks like one?

Spike 3 found 32 part pairs with overlapping triangles. That number alone is not a
verdict, because two parts that merely *touch* also report triangle overlap — their
surfaces are coincident. The engineering question is not "do the triangles intersect",
it is **"do the solids occupy the same volume"**.

So this measures overlap VOLUME, by sampling points and testing containment in both
bodies. A nut resting on a washer: overlap ≈ 0, that is a correct assembly. A flange
buried inside a housing: overlap is a real fraction of the part, and that assembly
cannot be manufactured.

    /Applications/Blender.app/Contents/MacOS/Blender --background --python _spike_blender3b.py
"""

import os
import random
import time

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = "/Users/yashbisht/Downloads/ada_v2-main-2"
SRC_GLB = os.path.join(ROOT, "out", "turbocharger.glb")
OUT_TXT = os.path.join(ROOT, "out", "_spike_interference_report.txt")

SAMPLES = 3000          # points tested per candidate pair
TOUCHING_VOL = 0.02     # < 2% of the smaller part's bbox volume == touching, not buried

report = []


def say(line=""):
    print(f"[x] {line}")
    report.append(str(line))


t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=SRC_GLB)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
say(f"imported {len(meshes)} mesh objects, "
    f"{sum(len(o.data.polygons) for o in meshes)} faces")


def wbox(o):
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector((min(p[i] for p in pts) for i in range(3)))
    hi = Vector((max(p[i] for p in pts) for i in range(3)))
    return lo, hi


def boxes_overlap(alo, ahi, blo, bhi):
    return all(alo[i] <= bhi[i] and blo[i] <= ahi[i] for i in range(3))


def inside(bvh, pt, direction=Vector((1.0, 0.0, 0.0))):
    """Point-in-mesh by parity of ray crossings. Robust for closed solids."""
    hits = 0
    origin = pt.copy()
    for _ in range(64):
        loc, nor, idx, dist = bvh.ray_cast(origin, direction)
        if loc is None:
            break
        hits += 1
        origin = loc + direction * 1e-6
    return hits % 2 == 1


parts = []
for o in meshes:
    parts.append((o, BVHTree.FromObject(o, bpy.context.evaluated_depsgraph_get()), *wbox(o)))

say()
say("=" * 74)
say("SOLID OVERLAP ANALYSIS — do distinct bodies occupy the same volume?")
say("=" * 74)

random.seed(7)
results = []
for i in range(len(parts)):
    oi, bi, loi, hii = parts[i]
    for j in range(i + 1, len(parts)):
        oj, bj, loj, hij = parts[j]
        if not boxes_overlap(loi, hii, loj, hij):
            continue
        if not bi.overlap(bj):
            continue      # bounds overlap but no triangle contact -> fine

        # Intersection box, then sample inside it and test containment in BOTH bodies.
        lo = Vector((max(loi[k], loj[k]) for k in range(3)))
        hi = Vector((min(hii[k], hij[k]) for k in range(3)))
        span = hi - lo
        if min(span) <= 0:
            continue
        vol_box = span.x * span.y * span.z
        inside_both = 0
        for _ in range(SAMPLES):
            p = Vector((random.uniform(lo[k], hi[k]) for k in range(3)))
            if inside(bi, p) and inside(bj, p):
                inside_both += 1
        if inside_both == 0:
            continue
        frac = inside_both / SAMPLES
        overlap_vol = frac * vol_box
        small_vol = min(
            (hii - loi).x * (hii - loi).y * (hii - loi).z,
            (hij - loj).x * (hij - loj).y * (hij - loj).z) or 1.0
        ratio = overlap_vol / small_vol
        results.append((ratio, overlap_vol * 1e9, oi.name, oj.name, inside_both))

results.sort(reverse=True)

buried, touching = [], []
for ratio, vol_mm3, a, b, n in results:
    (buried if ratio >= TOUCHING_VOL else touching).append((ratio, vol_mm3, a, b, n))

say(f"candidate pairs (bounds + triangle contact) : {len(results)}")
say(f"  BURIED      (overlap >= {TOUCHING_VOL:.0%} of the smaller part) : {len(buried)}")
say(f"  TOUCHING    (overlap below that)                  : {len(touching)}")
say()
if buried:
    say("BURIED — these parts occupy the same space and cannot both exist:")
    say(f"    {'overlap':>9}  {'volume':>12}  {'pts':>5}  pair")
    for ratio, vol_mm3, a, b, n in buried[:20]:
        say(f"    {ratio:>8.1%}  {vol_mm3:>9.1f} mm3  {n:>5}  {a}  <->  {b}")
say()
if touching:
    say("TOUCHING — contact only, which is correct for mated parts:")
    for ratio, vol_mm3, a, b, n in touching[:8]:
        say(f"    {ratio:>8.2%}  {vol_mm3:>9.2f} mm3  {n:>5}  {a}  <->  {b}")

say()
say("=" * 74)
if buried:
    pct = len(buried) / max(len(results), 1)
    say(f"VERDICT: {len(buried)} of {len(results)} contacting pairs are genuinely buried "
        f"({pct:.0%}).")
    say("         This assembly is NOT manufacturable as drawn. It renders correctly")
    say("         and is physically impossible — which is the difference between a CAD")
    say("         model and a picture of one.")
else:
    say("VERDICT: no solid interpenetration. All contacts are surface contact only.")

say()
say(f"total {time.time() - t0:.1f}s")

with open(OUT_TXT, "w") as fh:
    fh.write("\n".join(report) + "\n")
print(f"[x] report -> {OUT_TXT}")
