"""Spike v2: prove the ceiling, not just that it runs.

v1 rendered, but blew out to white plastic because a metallic surface with nothing to
reflect looks like matte paper. Two fixes: a procedural sky environment so metal has
something to mirror, and light energies scaled to the model rather than guessed.

    /Applications/Blender.app/Contents/MacOS/Blender --background --python _spike_blender2.py
"""

import os
import time

import bpy
from mathutils import Vector

ROOT = "/Users/yashbisht/Downloads/ada_v2-main-2"
SRC_GLB = os.path.join(ROOT, "out", "turbocharger.glb")
OUT_PNG = os.path.join(ROOT, "out", "_spike_blender_v2.png")

t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

bpy.ops.import_scene.gltf(filepath=SRC_GLB)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]

mins = Vector((1e9,) * 3)
maxs = Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        mins = Vector((min(mins[i], w[i]) for i in range(3)))
        maxs = Vector((max(maxs[i], w[i]) for i in range(3)))
centre = (mins + maxs) / 2
size = max((maxs - mins)) or 1.0
print(f"[spike2] {len(meshes)} objects, size {size:.3f} m")

# ── environment: a procedural sky, so metal has a real world to reflect ──────
world = bpy.data.worlds.new("ADA_Sky")
scene.world = world
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
sky = nt.nodes.new("ShaderNodeTexSky")
# Blender 5.x renamed the sky models: NISHITA is now MULTIPLE_SCATTERING.
_sky_types = [i.identifier for i in
              type(sky).bl_rna.properties["sky_type"].enum_items]
sky.sky_type = next((s for s in ("MULTIPLE_SCATTERING", "NISHITA", "PREETHAM")
                     if s in _sky_types), _sky_types[0])
for prop, val in (("sun_elevation", 0.42), ("sun_rotation", 2.4), ("altitude", 120.0)):
    if hasattr(sky, prop):
        try:
            setattr(sky, prop, val)
        except Exception:
            pass
print(f"[spike2] sky_type={sky.sky_type}")
bg.inputs[1].default_value = 0.85
nt.links.new(sky.outputs[0], bg.inputs[0])
nt.links.new(bg.outputs[0], out.inputs[0])

# ── materials: read the GLB's own names, assign plausible metals ─────────────
def metal(name, base, rough, aniso=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    if b:
        b.inputs["Base Color"].default_value = (*base, 1.0)
        b.inputs["Metallic"].default_value = 1.0
        b.inputs["Roughness"].default_value = rough
        if "Anisotropic" in b.inputs:
            b.inputs["Anisotropic"].default_value = aniso
    return m

aluminium = metal("ADA_Alu", (0.62, 0.65, 0.70), 0.32, 0.35)
steel = metal("ADA_Steel", (0.45, 0.47, 0.50), 0.22, 0.15)
cast = metal("ADA_Cast", (0.33, 0.34, 0.36), 0.55)

for i, o in enumerate(sorted(meshes, key=lambda m: m.name)):
    # Crude but honest: alternate materials so the assembly reads as parts rather
    # than one lump. A real pass would drive this from the spec's material field.
    o.data.materials.clear()
    o.data.materials.append((aluminium, steel, cast)[i % 3])

# ── lights, energies scaled to the model (area light falls off with distance) ─
def add_area(name, loc, energy, size_):
    d = bpy.data.lights.new(name, type="AREA")
    d.energy, d.size = energy, size_
    ob = bpy.data.objects.new(name, d)
    ob.location = loc
    scene.collection.objects.link(ob)
    ob.rotation_euler = (centre - Vector(loc)).to_track_quat("-Z", "Y").to_euler()

k = size * size
add_area("key",  (size * 1.5, -size * 1.7, size * 1.5), 120 * k, size * 1.0)
add_area("fill", (-size * 1.9, -size * 0.9, size * 0.5), 40 * k, size * 1.4)
add_area("rim",  (0, size * 2.0, size * 1.1), 90 * k, size * 0.9)

# ── ground ───────────────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=size * 14,
                                 location=(centre.x, centre.y, mins.z - size * 0.005))
ground = bpy.context.active_object
gm = bpy.data.materials.new("ADA_Ground")
gm.use_nodes = True
gb = gm.node_tree.nodes.get("Principled BSDF")
if gb:
    gb.inputs["Base Color"].default_value = (0.035, 0.038, 0.045, 1.0)
    gb.inputs["Roughness"].default_value = 0.35
    gb.inputs["Metallic"].default_value = 0.0
ground.data.materials.append(gm)

# ── camera ───────────────────────────────────────────────────────────────────
cd = bpy.data.cameras.new("Cam")
cd.lens = 70
cam = bpy.data.objects.new("Cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam
dist = size * 2.4
cam.location = centre + Vector((dist * 0.72, -dist * 0.88, dist * 0.40))
cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()

# ── render: Cycles, denoised, ACES-ish grade ─────────────────────────────────
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6
scene.render.resolution_x = 1280
scene.render.resolution_y = 860
# `view_transform` is a DYNAMIC enum: introspecting it before the OCIO config is
# loaded reports a placeholder ('NONE'), so reading the choices and picking one fails
# even though the value is valid. Set it by attempting, which is the only reliable
# way with a dynamic enum.
for _vt in ("AgX", "Filmic", "Standard"):
    try:
        scene.view_settings.view_transform = _vt
        break
    except TypeError:
        continue
print(f"[spike2] view_transform={scene.view_settings.view_transform}")
scene.render.filepath = OUT_PNG
scene.render.image_settings.file_format = "PNG"

print("[spike2] rendering ...")
t1 = time.time()
bpy.ops.render.render(write_still=True)
print(f"[spike2] render {time.time() - t1:.1f}s   total {time.time() - t0:.1f}s")
print(f"[spike2] wrote {OUT_PNG} ({os.path.getsize(OUT_PNG) // 1024} KB)")
