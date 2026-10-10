"""Blender as the SCENE backend. Never used by CAD mode.

Blender is a rendering and physics engine, not a CAD kernel. It is called here for one
thing: turning geometry the kernel already produced into a lit, materialled image.

Two hard constraints, both discovered by running it:

1. **`bpy` is not importable into the venv.** Blender ships its own Python. The only route
   is `subprocess` with `--background --python <script>`. Everything below is therefore a
   script generator plus a runner.

2. **Blender 5.x renamed things, quietly.** Three separate failures cost real time:
   * Sky model `NISHITA` no longer exists — it is `MULTIPLE_SCATTERING`.
   * `view_transform` is a **dynamic enum** that reports a placeholder `'NONE'` before the
     OCIO config loads, so "read the choices, then pick one" throws `TypeError` even
     though the value is valid. It must be set by *attempting*.
   * Gravity moved to `scene.gravity`; `rigidbody_world.effective_gravity` is gone.
   The generated script handles all three defensively, because a version bump should
   degrade a render, not break it.

Absence is handled, not raised: a machine without Blender loses scene mode and keeps
everything else. That matters because CAD output does not depend on a renderer at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Where Blender lives. Checked in order; the app bundle path is first because on macOS
# `blender` is usually not on PATH even when it is installed.
BLENDER_PATHS = [
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
    "/usr/local/bin/blender",
    "/opt/homebrew/bin/blender",
    "/usr/bin/blender",
]
TIMEOUT_S = 900


def blender_path() -> Optional[str]:
    env = os.environ.get("ADA_BLENDER")
    if env and Path(env).exists():
        return env
    for p in BLENDER_PATHS:
        if Path(p).exists():
            return p
    found = shutil.which("blender")
    return found


def available() -> bool:
    return blender_path() is not None


def version() -> Optional[str]:
    path = blender_path()
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True,
                             timeout=60)
        return out.stdout.splitlines()[0].strip() if out.stdout else None
    except Exception:
        return None


@dataclass
class RenderResult:
    ok: bool
    path: Optional[str] = None
    seconds: float = 0.0
    engine: str = ""
    style: str = ""
    shot: str = ""
    parts: int = 0
    faces: int = 0
    message: str = ""
    log_tail: str = ""

    def summary(self) -> str:
        if not self.ok:
            return f"Render failed: {self.message}"
        return (f"Rendered {self.path} — {self.style}/{self.shot}, {self.parts} parts, "
                f"{self.faces:,} faces, {self.engine}, {self.seconds:.1f}s.")


# ── the generated scene script ───────────────────────────────────────────────
# Kept as one template string rather than a .py file so it travels with this module and
# cannot drift out of sync with the parameters passed to it.
_SCENE_SCRIPT = r'''
import json, math, os, sys, time
import bpy
from mathutils import Vector

CFG = json.loads(os.environ["ADA_SCENE_CFG"])
GLB = CFG["glb"]
OUT = CFG["out"]
STYLE = CFG["style"]
SHOT = CFG["shot"]
RES = tuple(CFG["resolution"])
SAMPLES = int(CFG["samples"])
TRANSPARENT = bool(CFG["transparent"])
MATERIALS = CFG.get("materials", {})
# Exposure controls, exposed so the look can be tuned without editing this template.
# Overexposure is the default failure of a three-point rig — each light is reasonable
# alone and the sum is not — so these need to be adjustable per style.
LIGHT_SCALE = float(CFG.get("light_scale", 1.0))
WORLD_STRENGTH = CFG.get("world_strength", None)

t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

bpy.ops.import_scene.gltf(filepath=GLB)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    print("ADA_ERROR: no meshes imported"); sys.exit(3)
faces = sum(len(o.data.polygons) for o in meshes)

mins = Vector((1e18, 1e18, 1e18)); maxs = Vector((-1e18, -1e18, -1e18))
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        mins = Vector((min(mins[i], w[i]) for i in range(3)))
        maxs = Vector((max(maxs[i], w[i]) for i in range(3)))
centre = (mins + maxs) / 2.0
size = max(maxs - mins) or 1.0
if size <= 0: size = 1.0

# ---- materials, from the spec ------------------------------------------------
def make_material(name, spec):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    if not b: return m
    col = spec.get("color", (0.6, 0.6, 0.62))
    b.inputs["Base Color"].default_value = (col[0], col[1], col[2], 1.0)
    if "Metallic" in b.inputs:
        b.inputs["Metallic"].default_value = float(spec.get("metal", 0.5))
    if "Roughness" in b.inputs:
        b.inputs["Roughness"].default_value = float(spec.get("rough", 0.4))
    if "Anisotropic" in b.inputs:
        b.inputs["Anisotropic"].default_value = float(spec.get("aniso", 0.0))
    return m

made = {}
for part, spec in MATERIALS.items():
    made[part] = make_material("ADA_" + part, spec)

fallback = make_material("ADA_default", {"color": (0.55, 0.57, 0.60), "metal": 0.9, "rough": 0.35})
assigned = 0
for o in meshes:
    # Match the exported object name against the spec's part key. GLTF keeps node names,
    # but Blender may suffix duplicates, so try exact then prefix.
    key = o.name if o.name in made else None
    if key is None:
        for k in made:
            if o.name.startswith(k) or k.startswith(o.name):
                key = k; break
    o.data.materials.clear()
    o.data.materials.append(made[key] if key else fallback)
    if key: assigned += 1

# ---- world -------------------------------------------------------------------
world = bpy.data.worlds.new("ADA_World")
scene.world = world
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()
wout = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")

sky_style = STYLE in ("studio", "outdoor", "workshop")
# Per-style environment strength. Tuned by rendering the real turbocharger at three
# exposure levels and comparing: the first attempt used 0.55 and came back blown out to
# white plastic. 0.06 is where metal reads as metal.
ENV_STRENGTH = {"studio": 0.06, "outdoor": 0.45, "workshop": 0.10,
                "dramatic": 0.02, "holographic": 0.03, "technical": 0.25}
if sky_style:
    sky = nt.nodes.new("ShaderNodeTexSky")
    # Blender 5.x renamed NISHITA -> MULTIPLE_SCATTERING. Set by attempting, never by
    # introspecting the enum, because the enum is version-dependent.
    for candidate in ("MULTIPLE_SCATTERING", "NISHITA", "PREETHAM", "HOSEK_WILKIE"):
        try:
            sky.sky_type = candidate; break
        except TypeError:
            continue
    for prop, val in (("sun_elevation", 0.42 if STYLE == "outdoor" else 0.75),
                      ("sun_rotation", 2.4), ("altitude", 120.0)):
        if hasattr(sky, prop):
            try: setattr(sky, prop, val)
            except Exception: pass
    nt.links.new(sky.outputs[0], bg.inputs[0])
    bg.inputs[1].default_value = ENV_STRENGTH.get(STYLE, 0.06)
else:
    base = {"dramatic": (0.02, 0.022, 0.03), "holographic": (0.01, 0.03, 0.05),
            "technical": (0.9, 0.9, 0.92)}.get(STYLE, (0.05, 0.055, 0.065))
    bg.inputs[0].default_value = (base[0], base[1], base[2], 1.0)
    bg.inputs[1].default_value = ENV_STRENGTH.get(STYLE, 0.06)
nt.links.new(bg.outputs[0], wout.inputs[0])
if WORLD_STRENGTH is not None:
    bg.inputs[1].default_value = float(WORLD_STRENGTH)

# ---- lights, energies scaled to the model ------------------------------------
# Area-light falloff is quadratic, so a fixed wattage that lights a 1 m part leaves a
# 0.05 m part black — and a wattage that lights a 0.19 m part correctly blows a 1 m part
# out. Scale by area, which is what "size * size" is doing.
#
# The base figures were reduced ~7x after the first real render came back blown out to
# white plastic. Overexposure is the default failure of a three-point rig: every light
# looks reasonable on its own and the sum does not. Tuned by rendering the actual
# turbocharger at three levels and comparing.
k = size * size
intensity = {"dramatic": 0.22, "holographic": 0.16, "technical": 0.24}.get(STYLE, 0.093)
# Per-shot exposure. Light energy is scaled to the MODEL, not to the camera, so a close
# camera sees surfaces that are far too hot — the macro shot came back as white paper.
# A tighter framing needs less light for the same apparent brightness.
SHOT_EXPOSURE = {"macro": 0.32, "hero": 0.80, "top": 1.15}
intensity *= SHOT_EXPOSURE.get(SHOT, 1.0)

def add_area(name, loc, energy, sz):
    d = bpy.data.lights.new(name, type="AREA")
    d.energy = energy * k * intensity * LIGHT_SCALE
    d.size = sz
    ob = bpy.data.objects.new(name, d)
    ob.location = loc
    scene.collection.objects.link(ob)
    ob.rotation_euler = (centre - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob

add_area("key",  (size*1.5, -size*1.7, size*1.5), 120.0, size*1.0)
add_area("fill", (-size*1.9, -size*0.9, size*0.5),  40.0, size*1.4)
add_area("rim",  (0.0, size*2.0, size*1.1),         90.0, size*0.9)

# ---- ground ------------------------------------------------------------------
if not TRANSPARENT:
    bpy.ops.mesh.primitive_plane_add(size=size*16,
        location=(centre.x, centre.y, mins.z - size*0.005))
    g = bpy.context.active_object
    gm = bpy.data.materials.new("ADA_Ground")
    gm.use_nodes = True
    gb = gm.node_tree.nodes.get("Principled BSDF")
    if gb:
        tone = {"technical": (0.95, 0.95, 0.96)}.get(STYLE, (0.035, 0.038, 0.045))
        gb.inputs["Base Color"].default_value = (tone[0], tone[1], tone[2], 1.0)
        gb.inputs["Roughness"].default_value = 0.5 if STYLE == "technical" else 0.32
    g.data.materials.append(gm)

# ---- camera ------------------------------------------------------------------
# `dist` must be the TRUE distance, so the view vector is normalised first. Treating it
# as a raw offset made the effective distance depend on which way the shot pointed, and
# the first real render came back cropped top and bottom.
#
# Framing: with a 36 mm sensor, a focal length F at distance D covers D*36/F of scene.
# To frame a model of `size` with ~60% margin, D ~= size * 1.6 * F / 36. For F=50 that
# is D ~= 2.2*size, and the multipliers below sit just outside it.
cam_data = bpy.data.cameras.new("ADA_Cam")
VIEWS = {
    "three_quarter": (0.72, -0.88, 0.40),
    "iso":           (0.85, -0.85, 0.55),
    "hero":          (0.55, -1.00, 0.22),
    "front":         (0.0, -1.35, 0.10),
    "back":          (0.0, 1.35, 0.10),
    "left":          (-1.35, 0.0, 0.10),
    "right":         (1.35, 0.0, 0.10),
    "top":           (0.0, -0.25, 1.30),
    "bottom":        (0.0, -0.25, -1.30),
    "macro":         (0.35, -0.45, 0.20),
}
d = Vector(VIEWS.get(SHOT, VIEWS["three_quarter"])).normalized()
if SHOT == "macro":
    dist = size * 1.9
    cam_data.lens = 90
elif SHOT == "hero":
    dist = size * 3.0
    cam_data.lens = 58
else:
    dist = size * 3.4
    cam_data.lens = 50
if SHOT == "technical":
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = size * 1.35
cam = bpy.data.objects.new("ADA_Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = centre + d * dist
cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()

# ---- render ------------------------------------------------------------------
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6 if STYLE != "technical" else 3
scene.render.resolution_x, scene.render.resolution_y = RES
scene.render.film_transparent = TRANSPARENT
scene.render.filepath = OUT
scene.render.image_settings.file_format = "PNG"

# view_transform is a DYNAMIC enum: before the OCIO config loads it reports a
# placeholder 'NONE', so introspecting the choices and picking one raises TypeError.
# Attempt the values in preference order instead.
for vt in ("AgX", "Filmic", "Standard"):
    try:
        scene.view_settings.view_transform = vt; break
    except TypeError:
        continue

print("ADA_META " + json.dumps({"faces": faces, "parts": len(meshes),
                               "assigned": assigned, "size": size,
                               "view_transform": scene.view_settings.view_transform}))
bpy.ops.render.render(write_still=True)
print("ADA_DONE " + json.dumps({"seconds": time.time() - t0, "out": OUT}))
'''


def _material_table(asm: Any) -> dict:
    """Per-part render materials from the spec, via the existing materials table.

    This is why scene mode is cheap to do well: `materials.py` already carries colour,
    metalness and roughness per material name, and every instance already knows which
    material it is. Passing that to Blender costs nothing and is the difference between
    "a grey part" and "cast iron housing, aluminium cover".
    """
    from .. import materials as mat_table

    out: dict = {}
    for inst in getattr(asm, "instances", []) or []:
        key = getattr(inst, "key", None)
        if not key:
            continue
        name = (getattr(inst, "material", "") or "").strip()
        spec = mat_table.get(name) if name else None
        if not spec or name.lower().replace("-", "_") not in mat_table.MATERIALS:
            spec = mat_table.DEFAULT
        out[str(key)] = {
            "color": list(spec.get("color", (0.6, 0.6, 0.62))),
            "metal": float(spec.get("metal", 0.5)),
            "rough": float(spec.get("rough", 0.4)),
            "aniso": 0.35 if spec.get("metal", 0) > 0.8 else 0.0,
        }
    return out


def render_assembly(asm: Any, *, style: str = "studio", shot: str = "three_quarter",
                    resolution: tuple = (1600, 1000), samples: int = 96,
                    transparent: bool = False,
                    out_path: Optional[str] = None,
                    light_scale: float = 1.0,
                    world_strength: Optional[float] = None) -> RenderResult:
    """Export the assembly, drive Blender headlessly, return the result.

    Never raises for a missing or failing Blender — a scene is the one thing a build can
    safely go without, and an exception here would take down a CAD request that does not
    need a renderer at all.
    """
    path = blender_path()
    if not path:
        return RenderResult(False, message="Blender not found", style=style, shot=shot)

    work = Path(tempfile.mkdtemp(prefix="ada_scene_"))
    glb = work / "model.glb"
    out = Path(out_path) if out_path else work / f"render_{style}_{shot}.png"

    try:
        from . import exporters
        exporters.export_gltf_file(asm, glb)
    except Exception as e:
        return RenderResult(False, message=f"GLB export failed: {type(e).__name__}: {e}",
                            style=style, shot=shot)

    if not glb.exists() or glb.stat().st_size == 0:
        return RenderResult(False, message="GLB export produced no file",
                            style=style, shot=shot)

    cfg = {
        "glb": str(glb), "out": str(out), "style": style, "shot": shot,
        "resolution": list(resolution), "samples": int(samples),
        "transparent": bool(transparent), "materials": _material_table(asm),
        "light_scale": float(light_scale),
    }
    if world_strength is not None:
        cfg["world_strength"] = float(world_strength)

    script = work / "scene.py"
    script.write_text(_SCENE_SCRIPT, encoding="utf-8")

    env = dict(os.environ)
    env["ADA_SCENE_CFG"] = json.dumps(cfg)
    env["PYTHONHASHSEED"] = "0"

    t0 = time.time()
    try:
        proc = subprocess.run([path, "--background", "--python", str(script)],
                              capture_output=True, text=True, env=env, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return RenderResult(False, message=f"Blender exceeded {TIMEOUT_S}s",
                            style=style, shot=shot)
    except Exception as e:
        return RenderResult(False, message=f"Blender failed to start: {e}",
                            style=style, shot=shot)

    elapsed = time.time() - t0
    log = (proc.stdout or "") + (proc.stderr or "")
    meta = _parse_meta(log, "ADA_META ")
    done = _parse_meta(log, "ADA_DONE ")

    if proc.returncode != 0 or not out.exists():
        return RenderResult(False, seconds=elapsed, style=style, shot=shot,
                            message=f"Blender exited {proc.returncode}",
                            log_tail=log[-1200:])

    return RenderResult(
        True, path=str(out), seconds=elapsed, engine="Cycles CPU",
        style=style, shot=shot,
        parts=int(meta.get("parts", 0)), faces=int(meta.get("faces", 0)),
        message=f"{meta.get('assigned', 0)} parts got spec materials; "
                f"view transform {meta.get('view_transform', '?')}",
    )


def _parse_meta(log: str, marker: str) -> dict:
    for line in log.splitlines():
        if marker in line:
            try:
                return json.loads(line.split(marker, 1)[1])
            except Exception:
                return {}
    return {}
