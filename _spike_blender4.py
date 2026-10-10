"""Spike 4: the test the owner actually asked for — does the MECHANISM work?

Interference at rest (spike 3b) is necessary but not sufficient. A turbocharger's whole
point is that the rotating assembly spins inside a stationary housing with a running
clearance. A model can be perfectly valid at rest and still jam the moment it turns —
which is exactly what "looks like art, not like a real machine" means.

So: take the rotating group, spin it through a full revolution, and check for collision
with the static parts at every step. That is a real engineering question with a real
answer, and it is the capability the owner wants Blender for.

    /Applications/Blender.app/Contents/MacOS/Blender --background --python _spike_blender4.py
"""

import math
import os
import time

import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

ROOT = "/Users/yashbisht/Downloads/ada_v2-main-2"
SRC_GLB = os.path.join(ROOT, "out", "turbocharger.glb")
OUT_TXT = os.path.join(ROOT, "out", "_spike_mechanism_report.txt")

STEPS = 72          # 5-degree increments through one revolution

report = []


def say(line=""):
    print(f"[mech] {line}")
    report.append(str(line))


t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC_GLB)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
say(f"imported {len(meshes)} objects")

names = sorted(o.name for o in meshes)
say(f"part names: {', '.join(names[:18])}{' ...' if len(names) > 18 else ''}")

# ── classify: what rotates, what stays still ─────────────────────────────────
ROTATING_HINTS = ("wheel", "impeller", "shaft", "nut", "bearing")
rotating = [o for o in meshes
            if any(h in o.name.lower() for h in ROTATING_HINTS)]
static = [o for o in meshes if o not in rotating]

if not rotating:
    say("FAIL: could not identify a rotating group by name — a real run would read this")
    say("      from the spec's mate graph instead of guessing from names.")
    raise SystemExit(0)

say()
say(f"rotating group ({len(rotating)}): {', '.join(o.name for o in rotating[:10])}")
say(f"static group   ({len(static)}): {', '.join(o.name for o in static[:10])}")

# ── the axis: through the assembly centre, along Z (the shaft axis) ──────────
mins = Vector((1e9,) * 3)
maxs = Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        mins = Vector((min(mins[i], w[i]) for i in range(3)))
        maxs = Vector((max(maxs[i], w[i]) for i in range(3)))
centre = (mins + maxs) / 2
say(f"assembly centre {tuple(round(v, 4) for v in centre)}, "
    f"size {max(maxs - mins):.3f} m")

# ── sweep the rotating group and test for collision at each angle ────────────
say()
say("=" * 72)
say("MECHANISM TEST — sweep the rotating group through 360° and check clearance")
say("=" * 72)

dep = bpy.context.evaluated_depsgraph_get()
static_bvh = {}
for o in static:
    static_bvh[o.name] = BVHTree.FromObject(o, dep)

orig = {o.name: o.matrix_world.copy() for o in rotating}
worst = []
collided_at = {}

for step in range(STEPS):
    ang = 2 * math.pi * step / STEPS
    rot = Matrix.Rotation(ang, 4, "Z")
    pivot = Matrix.Translation(centre)
    for o in rotating:
        o.matrix_world = pivot @ rot @ pivot.inverted() @ orig[o.name]
    bpy.context.view_layer.update()

    dep = bpy.context.evaluated_depsgraph_get()
    for o in rotating:
        rb = BVHTree.FromObject(o, dep)
        for sname, sb in static_bvh.items():
            hits = rb.overlap(sb)
            if hits:
                collided_at.setdefault((o.name, sname), []).append(
                    (round(math.degrees(ang)), len(hits)))
                worst.append((len(hits), o.name, sname, round(math.degrees(ang))))

# restore
for o in rotating:
    o.matrix_world = orig[o.name]
bpy.context.view_layer.update()

say(f"angles tested                     : {STEPS} (every {360 // STEPS}°)")
say(f"colliding (part, part) pairs      : {len(collided_at)}")
say(f"total collision events            : {len(worst)}")
say()

if collided_at:
    say("COLLISIONS DURING ROTATION:")
    say(f"    {'deg range':>18}  {'worst tri-hits':>14}  pair")
    for (a, b), evs in sorted(collided_at.items(),
                              key=lambda kv: -max(e[1] for e in kv[1]))[:15]:
        degs = [e[0] for e in evs]
        worst_n = max(e[1] for e in evs)
        say(f"    {min(degs):>7}..{max(degs):<7}°  {worst_n:>14}  {a}  <->  {b}")
    say()
    say("VERDICT: the rotating group does NOT clear the housing through a full turn.")
    say("         The mechanism jams. A rendering would never reveal this.")
else:
    say("No collisions at any angle.")
    say()
    say("VERDICT: the rotating group clears the static parts through a full revolution.")
    say("         The mechanism is kinematically valid, not just visually plausible.")

say()
say(f"total {time.time() - t0:.1f}s")
with open(OUT_TXT, "w") as fh:
    fh.write("\n".join(report) + "\n")
print(f"[mech] report -> {OUT_TXT}")
