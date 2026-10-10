"""Capability spike: render one of ADA's existing GLB exports through Blender.

This answers the only question that matters for the plan — can Blender, driven
headlessly from a script, take geometry the cad_v3 kernel already produces and turn it
into a lit, materialled, shadowed image?

Run:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python _spike_blender.py
"""

import os
import sys
import time

import bpy
from mathutils import Vector

ROOT = "/Users/yashbisht/Downloads/ada_v2-main-2"
SRC_GLB = os.path.join(ROOT, "out", "turbocharger.glb")
OUT_PNG = os.path.join(ROOT, "out", "_spike_blender_render.png")

t0 = time.time()
print(f"[spike] blender {bpy.app.version_string}")

# ── clean slate ──────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ── import ADA's geometry ────────────────────────────────────────────────────
if not os.path.exists(SRC_GLB):
    print(f"[spike] FAIL: {SRC_GLB} not found")
    sys.exit(1)
bpy.ops.import_scene.gltf(filepath=SRC_GLB)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
tris = sum(len(o.data.polygons) for o in meshes)
print(f"[spike] imported {len(meshes)} mesh objects, {tris} faces")

# bounding box, so the camera and lights can be placed from the actual model
mins = Vector((1e9, 1e9, 1e9))
maxs = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for corner in o.bound_box:
        w = o.matrix_world @ Vector(corner)
        mins = Vector((min(mins[i], w[i]) for i in range(3)))
        maxs = Vector((max(maxs[i], w[i]) for i in range(3)))
centre = (mins + maxs) / 2
size = max((maxs - mins)) or 1.0
print(f"[spike] bounds size {size:.1f}, centre {tuple(round(c, 1) for c in centre)}")

# ── a real metal material, not a flat grey ───────────────────────────────────
mat = bpy.data.materials.new("ADA_Aluminium")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.72, 0.74, 0.78, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.95
    bsdf.inputs["Roughness"].default_value = 0.28
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = 0.35
for o in meshes:
    if not o.data.materials:
        o.data.materials.append(mat)

# ── studio: HDRI world + key/fill/rim, so there is something to reflect ──────
world = bpy.data.worlds.new("ADA_World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.05, 0.06, 0.08, 1.0)
    bg.inputs[1].default_value = 1.0

def add_area(name, loc, energy, size_):
    d = bpy.data.lights.new(name, type="AREA")
    d.energy = energy
    d.size = size_
    ob = bpy.data.objects.new(name, d)
    ob.location = loc
    scene.collection.objects.link(ob)
    # aim at the model
    direction = centre - Vector(loc)
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return ob

add_area("key",  (size * 1.4, -size * 1.6, size * 1.6), 900 * size * size, size * 0.9)
add_area("fill", (-size * 1.8, -size * 0.8, size * 0.6), 260 * size * size, size * 1.3)
add_area("rim",  (0, size * 1.9, size * 1.2), 500 * size * size, size * 0.8)

# ground so the model has something to sit on and cast onto
bpy.ops.mesh.primitive_plane_add(size=size * 12, location=(centre.x, centre.y, mins.z - size * 0.01))
ground = bpy.context.active_object
gmat = bpy.data.materials.new("ADA_Ground")
gmat.use_nodes = True
gb = gmat.node_tree.nodes.get("Principled BSDF")
if gb:
    gb.inputs["Base Color"].default_value = (0.045, 0.05, 0.06, 1.0)
    gb.inputs["Roughness"].default_value = 0.55
ground.data.materials.append(gmat)

# ── camera, three-quarter view ───────────────────────────────────────────────
cam_data = bpy.data.cameras.new("ADA_Cam")
cam_data.lens = 62
cam = bpy.data.objects.new("ADA_Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
dist = size * 2.5
cam.location = centre + Vector((dist * 0.75, -dist * 0.85, dist * 0.45))
cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()

# ── render settings: Cycles on CPU, modest samples ───────────────────────────
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.resolution_x = 1280
scene.render.resolution_y = 860
scene.render.film_transparent = False
scene.view_settings.view_transform = "AgX" if "AgX" in [
    v.identifier for v in type(scene.view_settings).bl_rna.properties["view_transform"].enum_items
] else "Filmic"
scene.render.filepath = OUT_PNG
scene.render.image_settings.file_format = "PNG"

print(f"[spike] rendering with {scene.render.engine}, {scene.cycles.samples} samples ...")
t1 = time.time()
bpy.ops.render.render(write_still=True)
print(f"[spike] render took {time.time() - t1:.1f}s")
print(f"[spike] wrote {OUT_PNG} ({os.path.getsize(OUT_PNG) // 1024} KB)")
print(f"[spike] total {time.time() - t0:.1f}s")
