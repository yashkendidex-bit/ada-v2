"""Ada Alfred-style plush — isolated lab build (does NOT touch src/ or main UI).
Deterministic entry script: clean scene -> triangle plush body -> fur -> glasses -> bowtie
-> studio renders + .blend + .glb (base mesh, hair is render-only, noted on export).

Run:
  blender --background --factory-startup --python _avatar_lab/ada_alfred_plush.py

Outputs (all inside _avatar_lab/):
  ada_plush.blend, ada_plush_base.glb, renders/hero_front.png, renders/three_quarter.png, renders/face_closeup.png
"""
import math
import os

import bpy
import mathutils

ROOT = "/Users/yashbisht/Downloads/ada_v2-main-2"
LAB = os.path.join(ROOT, "_avatar_lab")
RENDERS = os.path.join(LAB, "renders")
os.makedirs(RENDERS, exist_ok=True)

BLEND_OUT = os.path.join(LAB, "ada_plush.blend")
GLB_OUT = os.path.join(LAB, "ada_plush_base.glb")

# ---- art direction (from user Alfred reference) ----
BODY_YELLOW = (1.0, 0.62, 0.02, 1.0)      # rich plush yellow
BODY_YELLOW_TIP = (1.0, 0.74, 0.22, 1.0)  # fur tip highlight
FRAME_BLACK = (0.03, 0.03, 0.035, 1.0)    # glasses + bowtie satin black
EYE_WHITE = (1.0, 1.0, 1.0, 1.0)
PUPIL_BLACK = (0.01, 0.01, 0.012, 1.0)
CREAM_BG = (1.0, 0.972, 0.91, 1.0)

BODY_SUBDIV_RENDER = 2
FUR_COUNT = 9000
FUR_LENGTH = 0.014
FUR_CHILDREN = 12


def say(msg):
    print(f"[ada-plush] {msg}")


def mat_principled(name, base_color, roughness=0.6, subsurface=0.0, sub_color=None,
                   metallic=0.0, sheen=0.0, sheen_tint=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    # Blender 4.x/5.x renamed subsurface inputs; set what exists
    for key, val in (("Subsurface Weight", subsurface), ("Subsurface", subsurface)):
        if key in bsdf.inputs:
            try:
                bsdf.inputs[key].default_value = val
            except Exception:
                pass
    if sub_color is not None:
        for key in ("Subsurface Color", "Subsurface Tint"):
            if key in bsdf.inputs:
                try:
                    bsdf.inputs[key].default_value = (*sub_color[:3], 1.0)
                except Exception:
                    pass
    try:
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = sheen
        elif "Sheen" in bsdf.inputs:
            bsdf.inputs["Sheen"].default_value = sheen
    except Exception:
        pass
    return m


# ---- clean scene ----
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
try:
    scene.view_settings.exposure = -0.5  # tame blown-out yellow in Eevee
except Exception:
    pass
scene.eevee.taa_render_samples = 64
world = bpy.data.worlds.new("World")
world.use_nodes = True
scene.world = world
bg = world.node_tree.nodes.get("Background")
bg.inputs["Color"].default_value = CREAM_BG
bg.inputs["Strength"].default_value = 1.0

# ---- body: ico sphere -> rounded triangle (wide bottom, narrow top) ----
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0, location=(0, 0, 0.9))
body = bpy.context.active_object
body.name = "AdaBody"
mesh = body.data
for v in mesh.vertices:
    x, y, z = v.co.x, v.co.y, v.co.z  # z is up in object local (we keep default orientation)
    # taper: top (z->1) narrower, bottom (z->-1) wider -> triangle silhouette
    t = (z + 1.0) / 2.0  # 0 bottom .. 1 top
    taper = 1.0 - 0.48 * (t ** 1.6)
    v.co.x *= taper
    v.co.y *= (0.88 - 0.10 * t)  # slightly thinner front-back at top
    # flatten + widen sitting base
    if z < -0.55:
        v.co.z = -0.55 + (z + 0.55) * 0.35
        v.co.x *= 1.08
# plump overall
body.scale = (1.0, 1.0, 1.05)
bpy.ops.object.transform_apply(scale=True)
sub = body.modifiers.new("Subdiv", "SUBSURF")
sub.levels = BODY_SUBDIV_RENDER
sub.render_levels = BODY_SUBDIV_RENDER
bpy.ops.object.shade_smooth()

# fur density mask: keep fur off the glasses/bowtie zone so the face stays readable
vg = body.vertex_groups.new(name="FurMask")
eye_locs = [mathutils.Vector((-0.3, 0.8, 0.28)), mathutils.Vector((0.3, 0.8, 0.28))]
bow_loc = mathutils.Vector((0.0, 0.75, -0.35))
for v in mesh.vertices:
    co = mathutils.Vector((v.co.x, v.co.y, v.co.z))
    d_eye = min((co - e).length for e in eye_locs)
    d_bow = max(abs(co.x - bow_loc.x) / 0.38, abs(co.y - bow_loc.y) / 0.2,
                abs(co.z - bow_loc.z) / 0.28)
    w_eye = max(0.0, min(1.0, (d_eye - 0.30) / 0.25))
    w_bow = max(0.0, min(1.0, (d_bow - 1.0) / 0.6))
    vg.add([v.index], min(w_eye, w_bow), "REPLACE")

fur_mat = mat_principled("AdaFur", BODY_YELLOW, roughness=0.85, subsurface=0.35,
                         sub_color=BODY_YELLOW_TIP, sheen=1.0, sheen_tint=0.6)
# fur-tip variation via noise -> color ramp mixed into base
nodes = fur_mat.node_tree.nodes
links = fur_mat.node_tree.links
tex = nodes.new("ShaderNodeTexCoord")
noise = nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 18.0
noise.inputs["Detail"].default_value = 4.0
ramp = nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = BODY_YELLOW
ramp.color_ramp.elements[1].color = BODY_YELLOW_TIP
bsdf = nodes.get("Principled BSDF")
links.new(tex.outputs["Generated"], noise.inputs["Vector"])
links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
body.data.materials.append(fur_mat)

# ---- fur: short dense particle hair (render-only; excluded from GLB) ----
try:
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.particle_system_add()
    psys = body.particle_systems.active
    psys.name = "PlushFur"
    s = psys.settings
    s.type = "HAIR"
    s.count = FUR_COUNT
    s.hair_length = FUR_LENGTH
    s.hair_step = 4
    s.child_type = "SIMPLE"
    try:
        s.rendered_child_count = FUR_CHILDREN  # Blender 4.x/5.x API
    except Exception:
        try:
            s.child_nbr = FUR_CHILDREN  # pre-4.x fallback
        except Exception:
            pass
    s.child_length = 0.6
    s.child_radius = 0.02
    s.roughness_1 = 0.15
    s.clump_factor = 0.15
    # comb outward is default from sphere normals; slight random tilt reads as plush
    s.roughness_endpoint = 0.15
    psys.vertex_group_density = "FurMask"
    hm = body.modifiers.new("FurMat_Hold", "NODES")  # placeholder guard, removed below
    body.modifiers.remove(hm)
    say(f"fur: {FUR_COUNT} strands x {FUR_CHILDREN} children, len {FUR_LENGTH}")
except Exception as e:
    say(f"fur skipped ({e}); base material fur-fake still applies")

# ---- glasses: oversized round black frames, white eyeballs, playful pupils ----
plastic = mat_principled("FrameBlack", FRAME_BLACK, roughness=0.28, metallic=0.05)
eye_mat = mat_principled("EyeWhite", EYE_WHITE, roughness=0.25)
pupil_mat = mat_principled("Pupil", PUPIL_BLACK, roughness=0.15)
hl_mat = mat_principled("Highlight", EYE_WHITE, roughness=0.05)
hl_mat.node_tree.nodes.get("Principled BSDF").inputs["Emission Strength"] \
    if "Emission Strength" in hl_mat.node_tree.nodes.get("Principled BSDF").inputs else None

glasses = []
EYE_X = 0.30
EYE_Y = 0.80    # embedded at front surface (~0.73) so frames hug the face
EYE_Z = 1.18
FRAME_R = 0.175
TUBE_R = 0.028

for side, look in ((-1, mathutils.Vector((-0.05, 0.02, 0.03))),
                   (1, mathutils.Vector((0.05, 0.02, -0.035)))):
    x = side * EYE_X
    # white eyeball, slightly protruding + squashed
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.135,
                                         location=(x, EYE_Y, EYE_Z))
    eye = bpy.context.active_object
    eye.name = f"Eyeball_{'L' if side < 0 else 'R'}"
    eye.scale = (1.0, 0.72, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    eye.data.materials.append(eye_mat)
    bpy.ops.object.shade_smooth()
    # pupil + highlight, offset for playful strabismus (matches hero: L up-left, R down-right)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.052,
                                         location=(x + look.x, EYE_Y + 0.085, EYE_Z + look.z))
    pupil = bpy.context.active_object
    pupil.name = f"Pupil_{'L' if side < 0 else 'R'}"
    pupil.data.materials.append(pupil_mat)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.016,
                                         location=(x + look.x - 0.018, EYE_Y + 0.125, EYE_Z + look.z + 0.022))
    hl = bpy.context.active_object
    hl.name = f"Highlight_{'L' if side < 0 else 'R'}"
    hl.data.materials.append(hl_mat)
    # round frame
    bpy.ops.mesh.primitive_torus_add(major_radius=FRAME_R, minor_radius=TUBE_R,
                                     major_segments=48, minor_segments=16,
                                     location=(x, EYE_Y + 0.01, EYE_Z),
                                     rotation=(math.radians(90), 0, 0))
    frame = bpy.context.active_object
    frame.name = f"Frame_{'L' if side < 0 else 'R'}"
    frame.data.materials.append(plastic)
    bpy.ops.object.shade_smooth()
    glasses.extend([eye, pupil, hl, frame])

# bridge + temples
bpy.ops.mesh.primitive_torus_add(major_radius=0.09, minor_radius=0.022, major_segments=24,
                                 minor_segments=12, location=(0, EYE_Y + 0.01, EYE_Z + 0.02),
                                 rotation=(0, 0, math.radians(90)))
bridge = bpy.context.active_object
bridge.name = "Bridge"
bridge.data.materials.append(plastic)
for side in (-1, 1):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.35,
                                        location=(side * (EYE_X + FRAME_R + 0.06), EYE_Y - 0.05, EYE_Z + 0.02),
                                        rotation=(math.radians(78), 0, side * math.radians(12)))
    temple = bpy.context.active_object
    temple.name = f"Temple_{'L' if side < 0 else 'R'}"
    temple.data.materials.append(plastic)
    glasses.append(temple)
glasses.append(bridge)

# ---- bowtie: pinched center + two wings, satin black ----
satin = mat_principled("BowtieSatin", FRAME_BLACK, roughness=0.35, metallic=0.15, sheen=0.6)
BOW_Z = 0.55
BOW_Y = 0.78
bpy.ops.mesh.primitive_cube_add(size=0.16, location=(0, BOW_Y, BOW_Z))
knot = bpy.context.active_object
knot.name = "BowKnot"
knot.scale = (0.7, 0.6, 0.9)
bpy.ops.object.transform_apply(scale=True)
wings = [knot]
for side in (-1, 1):
    bpy.ops.mesh.primitive_cube_add(size=0.22, location=(side * 0.19, BOW_Y - 0.01, BOW_Z))
    wing = bpy.context.active_object
    wing.name = f"BowWing_{'L' if side < 0 else 'R'}"
    wing.scale = (1.25, 0.4, 0.7)
    wing.rotation_euler = (0, 0, side * math.radians(22))
    bpy.ops.object.transform_apply(scale=True)
    # pinch inner edge: pull verts near center inward for bow folds
    for v in wing.data.vertices:
        if side * v.co.x < 0:  # inner side
            v.co.x *= 0.45
    wings.append(wing)
for w in wings:
    w.data.materials.append(satin)
    # bevel-ish: one subdiv + smooth so it reads soft satin, not hard cube
    md = w.modifiers.new("Soft", "SUBSURF")
    md.levels = 2
    md.render_levels = 2
    bpy.context.view_layer.objects.active = w
    bpy.ops.object.shade_smooth()

# ---- studio: key / fill / rim + camera rig ----
bpy.ops.object.light_add(type="AREA", location=(3.2, 4.0, 5.0))
key = bpy.context.active_object
key.name = "Key"
key.data.energy = 300
key.data.size = 4.0
bpy.ops.object.light_add(type="AREA", location=(-4.0, 2.0, 3.0))
fill = bpy.context.active_object
fill.name = "Fill"
fill.data.energy = 120
fill.data.size = 5.0
bpy.ops.object.light_add(type="SUN", location=(0, -4.0, 4.0))
rim = bpy.context.active_object
rim.name = "Rim"
rim.data.energy = 0.8

bpy.ops.object.camera_add(location=(0.0, 3.4, 1.35))
cam = bpy.context.active_object
cam.name = "HeroCam"
cam.data.lens = 50
# track to body center
bpy.ops.object.empty_add(location=(0, 0, 0.95))
target = bpy.context.active_object
target.name = "LookTarget"
con = cam.constraints.new("TRACK_TO")
con.target = target
con.track_axis = "TRACK_NEGATIVE_Z"
con.up_axis = "UP_Y"
scene.camera = cam

rig = [body, *glasses, *wings]


def render_to(path, cam_loc=None, lens=None):
    if cam_loc is not None:
        cam.location = cam_loc
    if lens is not None:
        cam.data.lens = lens
    bpy.context.view_layer.update()
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    say(f"render -> {path}")


render_to(os.path.join(RENDERS, "hero_front.png"), cam_loc=(0.0, 3.4, 1.35), lens=50)
render_to(os.path.join(RENDERS, "three_quarter.png"), cam_loc=(2.0, 2.8, 1.5), lens=50)
render_to(os.path.join(RENDERS, "face_closeup.png"), cam_loc=(0.35, 1.7, 1.35), lens=70)

# ---- save blend + GLB (base mesh only: hide particles for export) ----
bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)
say(f"blend -> {BLEND_OUT}")
for o in rig:
    o.select_set(True)
bpy.context.view_layer.objects.active = body
# particles do not export to glTF; temporarily remove for a clean base-mesh export
kept_psys = [p.name for p in body.particle_systems]
for pname in kept_psys:
    body.particle_systems.active_index = 0
    bpy.ops.object.particle_system_remove()
try:
    bpy.ops.export_scene.gltf(filepath=GLB_OUT, export_format="GLB",
                              use_selection=True, export_materials="EXPORT")
    say(f"glb(base, no hair) -> {GLB_OUT}")
finally:
    say(f"note: fur was render-only ({kept_psys}); GLB carries base mesh + materials")
