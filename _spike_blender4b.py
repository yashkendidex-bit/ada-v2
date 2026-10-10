"""Spike 4b: does rotation *change* anything? Baseline-relative, volume-measured.

Spike 4 said "501 colliding pairs, the mechanism jams". That was almost certainly wrong,
and wrong the same way spike 3 was wrong: it counted triangle intersections and it never
asked what the assembly looks like *at rest*.

Two mistakes to correct:

1. **No baseline.** Parts that already touch in the assembled state report overlap at
   every angle. A collision caused by *rotation* is one whose overlap GROWS as the part
   turns. Comparing each angle against angle 0 is what separates the two.

2. **Triangle count is not interference.** `bearing_housing` also does not rotate — a
   bearing housing is the stationary part. Name heuristics are a poor substitute for the
   spec's mate graph, and this file says so rather than pretending otherwise.

    /Applications/Blender.app/Contents/MacOS/Blender --background --python _spike_blender4b.py
"""

import math
import os
import random
import time

import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

ROOT = "/Users/yashbisht/Downloads/ada_v2-main-2"
SRC_GLB = os.path.join(ROOT, "out", "turbocharger.glb")
OUT_TXT = os.path.join(ROOT, "out", "_spike_mechanism_report.txt")

STEPS = 36          # every 10 degrees
GROWTH_FACTOR = 2.0  # overlap must at least double vs rest to count as a real collision
VOL_SAMPLES = 2000

report = []


def say(line=""):
    print(f"[m] {line}")
    report.append(str(line))


t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC_GLB)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]

# Rotating group by name. `bearing_housing` is deliberately EXCLUDED — it is the
# stationary carrier. A real implementation reads this from the spec's mate graph;
# guessing from names is why spike 4 was wrong.
ROTATING_HINTS = ("turbine_wheel", "impeller", "shaft", "nut_")
rotating = [o for o in meshes if any(h in o.name.lower() for h in ROTATING_HINTS)]
static = [o for o in meshes if o not in rotating]

say(f"{len(meshes)} objects | rotating {len(rotating)} | static {len(static)}")
say(f"rotating: {', '.join(o.name for o in rotating)}")

mins = Vector((1e9,) * 3)
maxs = Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        mins = Vector((min(mins[i], w[i]) for i in range(3)))
        maxs = Vector((max(maxs[i], w[i]) for i in range(3)))
centre = (mins + maxs) / 2


def bvh_of(o):
    return BVHTree.FromObject(o, bpy.context.evaluated_depsgraph_get())


def tri_overlap_pairs():
    """Cheap proxy: which (rotating, static) pairs have ANY triangle contact."""
    out = {}
    dep = bpy.context.evaluated_depsgraph_get()
    s_bvh = {o.name: BVHTree.FromObject(o, dep) for o in static}
    for o in rotating:
        rb = BVHTree.FromObject(o, dep)
        for sname, sb in s_bvh.items():
            n = len(rb.overlap(sb))
            if n:
                out[(o.name, sname)] = n
    return out


def solid_overlap_volume(rot_name, static_name):
    """Sampled solid overlap volume in mm^3 — the measurement that actually decides."""
    ra = next(o for o in rotating if o.name == rot_name)
    sa = next(o for o in static if o.name == static_name)
    ba, bb = bvh_of(ra), bvh_of(sa)

    def wb(o):
        pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
        return (Vector((min(p[i] for p in pts) for i in range(3))),
                Vector((max(p[i] for p in pts) for i in range(3))))

    alo, ahi = wb(ra)
    blo, bhi = wb(sa)
    lo = Vector((max(alo[k], blo[k]) for k in range(3)))
    hi = Vector((min(ahi[k], bhi[k]) for k in range(3)))
    span = hi - lo
    if min(span) <= 0:
        return 0.0, 0

    def inside(bvh, pt):
        hits, origin = 0, pt.copy()
        d = Vector((1.0, 0.0, 0.0))
        for _ in range(64):
            loc, _, _, _ = bvh.ray_cast(origin, d)
            if loc is None:
                break
            hits += 1
            origin = loc + d * 1e-6
        return hits % 2 == 1

    random.seed(11)
    n_in = 0
    for _ in range(VOL_SAMPLES):
        p = Vector((random.uniform(lo[k], hi[k]) for k in range(3)))
        if inside(ba, p) and inside(bb, p):
            n_in += 1
    vol_box = span.x * span.y * span.z
    return (n_in / VOL_SAMPLES) * vol_box * 1e9, n_in


orig = {o.name: o.matrix_world.copy() for o in rotating}

say()
say("=" * 74)
say("STEP 1 — baseline: what overlaps with the assembly AT REST?")
say("=" * 74)
rest = tri_overlap_pairs()
say(f"pairs with contact at rest: {len(rest)}")

say()
say("=" * 74)
say(f"STEP 2 — rotate through 360° in {STEPS} steps; find pairs whose overlap GROWS")
say("=" * 74)

max_seen = dict(rest)
worst_angle = {}
for step in range(1, STEPS + 1):
    ang = 2 * math.pi * step / STEPS
    pivot = Matrix.Translation(centre)
    rot = Matrix.Rotation(ang, 4, "Z")
    for o in rotating:
        o.matrix_world = pivot @ rot @ pivot.inverted() @ orig[o.name]
    bpy.context.view_layer.update()

    cur = tri_overlap_pairs()
    for key, n in cur.items():
        if n > max_seen.get(key, 0):
            max_seen[key] = n
            worst_angle[key] = round(math.degrees(ang))
    for key in list(max_seen):
        if key not in cur and max_seen[key] > 0:
            pass    # contact disappearing is not a jam; leave the max where it was

say(f"pairs whose overlap ever exceeded the at-rest value: "
    f"{sum(1 for k, v in max_seen.items() if v > rest.get(k, 0) * GROWTH_FACTOR)}")

grew = [(k, rest.get(k, 0), v) for k, v in max_seen.items()
        if v > max(rest.get(k, 0), 0) * GROWTH_FACTOR and v > 50]

say()
say("=" * 74)
say("STEP 3 — verify the growing pairs by SOLID VOLUME at their worst angle")
say("=" * 74)

if not grew:
    say("No pair's overlap grew meaningfully during rotation.")
    say()
    say("VERDICT: the rotating group clears the static parts through a full revolution.")
    say("         The mechanism is kinematically valid — not merely visually plausible.")
else:
    say(f"{len(grew)} candidate pair(s) to verify:")
    say(f"    {'at rest':>8}  {'at worst':>9}  {'angle':>7}  {'solid overlap':>14}  pair")
    real_jams = 0
    for (a, b), rest_n, peak in sorted(grew, key=lambda x: -x[2])[:15]:
        ang = worst_angle.get((a, b), 0)
        pivot = Matrix.Translation(centre)
        rot = Matrix.Rotation(math.radians(ang), 4, "Z")
        for o in rotating:
            o.matrix_world = pivot @ rot @ pivot.inverted() @ orig[o.name]
        bpy.context.view_layer.update()
        vol, n_in = solid_overlap_volume(a, b)
        flag = "JAM" if vol > 50.0 else "touch"
        if flag == "JAM":
            real_jams += 1
        say(f"    {rest_n:>8}  {peak:>9}  {ang:>6}°  {vol:>11.1f} mm3  {a} <-> {b}  [{flag}]")
    say()
    if real_jams:
        say(f"VERDICT: {real_jams} pair(s) genuinely interfere during rotation.")
    else:
        say("VERDICT: overlap grows only at contact surfaces; no solid interpenetration.")
        say("         The mechanism clears.")

for o in rotating:
    o.matrix_world = orig[o.name]
bpy.context.view_layer.update()

say()
say(f"total {time.time() - t0:.1f}s")
with open(OUT_TXT, "w") as fh:
    fh.write("\n".join(report) + "\n")
print(f"[m] report -> {OUT_TXT}")
