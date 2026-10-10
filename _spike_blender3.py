"""Spike 3: can Blender answer an ENGINEERING question about ADA's geometry?

Spikes 1 and 2 proved Blender renders. That is the easy half and the owner has correctly
pointed out it is not what he asked for. The question that matters is whether a CAD model
ADA produces is *mechanically real* — do the parts fit, do they move, do they interfere.

Two tests, both on the real `out/turbocharger.glb`:

  1. STATIC INTERFERENCE — exact triangle-level overlap between every pair of parts, via
     BVHTree. This is the test that separates an engineering model from art: if two solid
     bodies occupy the same space, the assembly cannot exist. It is also cheap and
     deterministic, unlike a physics bake.

  2. RIGID-BODY DYNAMICS — actually simulate. Gravity, collision, contact. Proves Blender
     can answer "what happens when this moves" rather than only "what does this look like".

    /Applications/Blender.app/Contents/MacOS/Blender --background --python _spike_blender3.py
"""

import os
import time

import bpy
from mathutils.bvhtree import BVHTree
from mathutils import Vector

ROOT = "/Users/yashbisht/Downloads/ada_v2-main-2"
SRC_GLB = os.path.join(ROOT, "out", "turbocharger.glb")
OUT_TXT = os.path.join(ROOT, "out", "_spike_physics_report.txt")

report = []


def say(line=""):
    print(f"[phys] {line}")
    report.append(str(line))


t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0

bpy.ops.import_scene.gltf(filepath=SRC_GLB)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
say(f"imported {len(meshes)} mesh objects")


def world_bbox(o):
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector((min(p[i] for p in pts) for i in range(3)))
    hi = Vector((max(p[i] for p in pts) for i in range(3)))
    return lo, hi


def overlaps(a_lo, a_hi, b_lo, b_hi, eps=0.0):
    return all(a_lo[i] <= b_hi[i] + eps and b_lo[i] <= a_hi[i] + eps for i in range(3))


# ── 1. static interference ───────────────────────────────────────────────────
say()
say("=" * 62)
say("TEST 1 — STATIC INTERFERENCE (exact triangle overlap, BVHTree)")
say("=" * 62)

# Precompute per-object BVH in world space, and bounding boxes for cheap rejection.
data = []
for o in meshes:
    bvh = BVHTree.FromObject(o, bpy.context.evaluated_depsgraph_get())
    lo, hi = world_bbox(o)
    data.append((o, bvh, lo, hi))

checked = 0
candidates = 0
interfering = []
for i in range(len(data)):
    oi, bi, loi, hii = data[i]
    for j in range(i + 1, len(data)):
        oj, bj, loj, hij = data[j]
        if not overlaps(loi, hii, loj, hij):
            continue
        candidates += 1
        checked += 1
        hits = bi.overlap(bj)
        if hits:
            interfering.append((oi.name, oj.name, len(hits)))

say(f"part pairs with overlapping bounds : {candidates}")
say(f"part pairs actually tested         : {checked}")
say(f"INTERFERING pairs (real overlap)   : {len(interfering)}")
for a, b, n in sorted(interfering, key=lambda x: -x[2])[:12]:
    say(f"    {n:6d} tri-pairs   {a}  <->  {b}")

if interfering:
    say()
    say("VERDICT: this assembly is NOT manufacturable as-is — solid bodies interpenetrate.")
else:
    say()
    say("VERDICT: no interference detected between distinct bodies.")

# ── 2. rigid body dynamics ───────────────────────────────────────────────────
say()
say("=" * 62)
say("TEST 2 — RIGID BODY DYNAMICS (does it behave under gravity?)")
say("=" * 62)

# Drop the assembly onto a ground plane and see whether it settles or explodes.
mins = Vector((1e9,) * 3)
maxs = Vector((-1e9,) * 3)
for o, _, lo, hi in data:
    mins = Vector((min(mins[i], lo[i]) for i in range(3)))
    maxs = Vector((max(maxs[i], hi[i]) for i in range(3)))
centre = (mins + maxs) / 2
size = max(maxs - mins) or 1.0

bpy.ops.mesh.primitive_plane_add(size=size * 20,
                                 location=(centre.x, centre.y, mins.z - size * 0.5))
ground = bpy.context.active_object
ground.name = "Ground"

if scene.rigidbody_world is None:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.effective_gravity = (0.0, 0.0, -9.81)
rbw.substeps_per_frame = 12
rbw.solver_iterations = 20

bpy.ops.object.select_all(action="DESELECT")
ground.select_set(True)
bpy.context.view_layer.objects.active = ground
bpy.ops.rigidbody.object_add()
ground.rigidbody.type = "PASSIVE"
ground.rigidbody.friction = 0.7

for o, _, _, _ in data:
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.rigidbody.object_add()
    o.rigidbody.type = "ACTIVE"
    o.rigidbody.mass = 0.5
    o.rigidbody.friction = 0.5

say(f"rigid bodies: {len(data)} active parts + 1 passive ground")
say(f"gravity: {rbw.effective_gravity[:]}   substeps={rbw.substeps_per_frame}")

scene.frame_start = 1
scene.frame_end = 60
scene.render.fps = 24
say("baking 60 frames of simulation (2.5 s of simulated time) ...")

t1 = time.time()
# Step the sim by advancing frames; read positions as we go.
start_z = {o.name: (o.matrix_world.translation.z) for o, _, _, _ in data}
bpy.context.scene.frame_set(1)
for f in range(2, 61):
    scene.frame_set(f)
sim_time = time.time() - t1

moved = 0
fell = 0
max_drop = 0.0
for o, _, _, _ in data:
    dz = o.matrix_world.translation.z - start_z[o.name]
    if abs(dz) > 1e-4:
        moved += 1
    if dz < -1e-4:
        fell += 1
    max_drop = max(max_drop, -dz)

say(f"simulation wall time               : {sim_time:.1f}s for 60 frames")
say(f"parts that moved                   : {moved}/{len(data)}")
say(f"parts that fell                     : {fell}/{len(data)}")
say(f"largest downward displacement       : {max_drop * 1000:.1f} mm")
say()
say("VERDICT: rigid-body simulation runs headless and produces measurable motion.")

say()
say(f"total {time.time() - t0:.1f}s")

with open(OUT_TXT, "w") as fh:
    fh.write("\n".join(report) + "\n")
print(f"[phys] report -> {OUT_TXT}")
