import os
import json
import asyncio
import glob
import shutil
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv(override=True)

MATERIAL_HINTS = {
    "titanium": ("titanium", ["titanium"]),
    "carbon": ("carbon", ["carbon", "carbon fiber", "fiber"]),
    "glass": ("glass", ["glass", "crystal", "transparent"]),
    "brass": ("brass", ["brass", "gold", "copper", "bronze"]),
    "aluminum": ("aluminum", ["aluminum", "aluminium", "aluminum alloy"]),
    "steel": ("steel", ["steel", "chrome", "stainless"]),
    "rubber": ("rubber", ["rubber", "silicone", "tire", "grip"]),
    "plastic": ("plastic", ["plastic", "acrylic", "nylon", "polymer", "3d printed"]),
    "wood": ("wood", ["wood", "walnut", "oak", "maple"]),
    "ceramic": ("ceramic", ["ceramic", "porcelain", "clay"]),
}


def pick_material(prompt):
    """Picks a material from the prompt's wording so the viewer can render it
    realistically. Falls back to ADA's signature cyan metal."""
    if not prompt:
        return "default"
    p = prompt.lower()
    for material, keywords in MATERIAL_HINTS.values():
        if any(k in p for k in keywords):
            return material
    return "default"


def pick_name(prompt):
    """Extracts a short display name from the prompt (up to 4 words)."""
    if not prompt:
        return "Part"
    words = prompt.split()[:4]
    name = " ".join(words).strip(" .,!?")
    return name or "Part"


def _build123d_hint(error_msg):
    """Maps common build123d runtime errors to targeted fixes so the retry
    model fixes the REAL problem instead of guessing."""
    err = error_msg or ""
    hints = []
    if "Nothing to subtract from" in err:
        hints.append("'Nothing to subtract from': the SUBTRACT shape must be drawn INSIDE the SAME "
                     "BuildSketch block as the base shape (e.g. `Circle(radius, mode=Mode.SUBTRACT)` "
                     "right after the base Rectangle), or inside a BuildPart after an extruded face exists. "
                     "Never open a fresh BuildSketch for the subtraction.")
    if "pending_edges" in err or "BuildLine" in err:
        hints.append("make_face() is INVALID inside BuildLine. Inside BuildSketch, draw closed "
                     "primitives directly (Rectangle, Circle, Polygon, Ellipse) - never call make_face().")
    if "Failed creating a fillet" in err or "BRep_API: command not done" in err:
        hints.append("Fillet/chamfer radius too large for the geometry. Use a SMALL radius (0.2-1.0mm) "
                     "or use max_fillet() to find the largest valid radius.")
    if "TopoDS::Face" in err or "Standard_TypeMismatch" in err:
        hints.append("A face could not be built from the wires - do NOT use make_face(); use closed "
                     "primitives (Rectangle/Circle/Polygon) inside BuildSketch instead.")
    if "NameError" in err:
        name = "unknown function"
        import re as _re
        m = _re.search(r"NameError: name '(\w+)' is not defined", err)
        if m:
            name = m.group(1)
        hints.append(f"'{name}' is NOT a build123d function. Only use documented build123d "
                     f"operations (Box, Cylinder, Rectangle, Circle, Polygon, extrude, fillet, chamfer, "
                     f"Hole, Locations, Pos, Mode.SUBTRACT). Use result_part.rotate_axes() or "
                     f"Locations/Pos for positioning instead.")
    if not hints:
        hints.append("Review the traceback: find the failing line, fix the build123d API usage, "
                     "and keep the rest of the script unchanged.")
    return "Build123d error hints:\n" + "\n".join(f"- {h}" for h in hints)


def _cad_staging_dir():
    """Shared staging dir so 'iterate' can find the previous script."""
    d = os.path.join(tempfile.gettempdir(), "ada_cad_work")
    os.makedirs(d, exist_ok=True)
    return d


def _collect_outputs(work_dir):
    """Returns (stls, steps) sorted: output*.stl and output*.step in work_dir."""
    stls = sorted(glob.glob(os.path.join(work_dir, "output*.stl")))
    steps = sorted(glob.glob(os.path.join(work_dir, "output*.step")))
    return stls, steps


def _inspect_stls(stls):
    """The INSPECTOR: measures every generated part (watertight? volume? size?)
    and returns (parts_info, issues_text). Issues drive one refinement round."""
    try:
        import trimesh
    except ImportError:
        return [{"name": os.path.basename(s)} for s in stls], ""
    parts_info = []
    issues = []
    for i, path in enumerate(stls):
        mesh = trimesh.load(path, force="mesh")
        size = (mesh.bounds[1] - mesh.bounds[0]).tolist()
        info = {
            "name": f"part_{i}",
            "size_mm": [round(v, 1) for v in size],
            "volume_mm3": round(float(mesh.volume), 1),
            "watertight": bool(mesh.is_watertight),
            "faces": len(mesh.faces),
        }
        parts_info.append(info)
        if not mesh.is_watertight:
            issues.append(f"part_{i} is NOT watertight (has holes - it is not a solid object).")
        if float(mesh.volume) <= 0.5:
            issues.append(f"part_{i} has near-zero volume ({info['volume_mm3']} mm3) - it is empty or a flat sheet.")
        for v in size:
            if v < 0.8:
                issues.append(f"part_{i} has a dimension of only {v:.1f}mm - it is too thin to be a real object.")
    return parts_info, "; ".join(issues)


def _build_cad_payload(prompt, stls, steps, work_dir):
    """Assembles the final payload: parts list (multi-part assemblies), a GLB
    export, persisted export files, material + name for the viewer."""
    import base64
    parts = []
    prompt_low = (prompt or "").lower()
    for i, path in enumerate(stls):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # Per-part realistic material assignment (Stark-lab style)
        pname = f"part_{i + 1}"
        pmat = pick_material(prompt)
        try:
            import trimesh as _tm2
            _m = _tm2.load(path, force="mesh")
            _sz = sorted((_m.bounds[1] - _m.bounds[0]).tolist())
            aspect = _sz[2] / max(_sz[0], 0.1)  # long/thin ratio
            if aspect > 3.0:
                pmat = "titanium" if "titan" in prompt_low else "steel"      # rods -> shafts
            elif _sz[2] < _sz[1] * 0.35:
                pmat = "aluminum"                                            # plates/housings
            elif "brass" in prompt_low:
                pmat = "brass"
        except Exception:
            pass
        parts.append({"name": pname, "data": b64, "material": pmat})
    if not parts:
        return None
    # Assembly metadata for real exploded view: per-part axis + size so viewer explodes along correct direction
    assembly = {"parts": []}
    try:
        import trimesh as _tm
        for i, path in enumerate(stls):
            m = _tm.load(path, force="mesh")
            sz = (m.bounds[1] - m.bounds[0]).tolist()
            sx, sy, sz_ = sz
            # Infer axis like frontend: thin = plate normal, long = rod axis, else Z
            dims_sorted = sorted([(sx, 0), (sy, 1), (sz_, 2)])
            small, mid, large = dims_sorted[0][0], dims_sorted[1][0], dims_sorted[2][0]
            if small < mid * 0.55:
                axis = [1 if dims_sorted[0][1]==0 else 0, 1 if dims_sorted[0][1]==1 else 0, 1 if dims_sorted[0][1]==2 else 0]
            elif large > mid * 1.9:
                axis = [1 if dims_sorted[2][1]==0 else 0, 1 if dims_sorted[2][1]==1 else 0, 1 if dims_sorted[2][1]==2 else 0]
            else:
                axis = [0,0,1]
            assembly["parts"].append({"name": f"part_{i+1}", "axis": axis, "size": [round(float(v),1) for v in sz]})
    except Exception as e:
        print(f"[CadAgent DEBUG] [ASSEMBLY] axis infer skipped: {str(e)[:80]}")
    # Bill of materials, derived from what the engine actually built. This used to
    # be an "enterprise enrichment" pass that re-parsed the prompt through
    # cad_v2/spec_compiler — a second, preset-driven guess at the same design. The
    # v3 assembly report is the single source of truth now, so the BOM is just the
    # real part list plus the material each part was assigned above.
    bom = [{"part": p["name"], "material": p["material"], "qty": 1} for p in parts]
    payload = {
        "format": "stl",
        "data": parts[0]["data"],
        "parts": parts,
        "assembly": assembly,
        "bom": bom,
        "material": pick_material(prompt),
        "name": pick_name(prompt),
        "exports": {"stls": stls, "steps": steps, "glb": None, "dir": None},
    }
    try:
        import trimesh
        meshes = []
        for path in stls:
            m = trimesh.load(path, force="mesh")
            m.apply_translation(-m.bounds.mean(0))
            meshes.append(m)
        if len(meshes) > 1:
            scene = trimesh.Scene(meshes)
            glb_bytes = scene.export(file_type="glb")
        else:
            glb_bytes = meshes[0].export(file_type="glb")
        if glb_bytes and len(glb_bytes) < 2_000_000:
            payload["exports"]["glb"] = base64.b64encode(glb_bytes).decode("utf-8")
    except Exception as e:
        print(f"[CadAgent DEBUG] [GLB] export skipped: {str(e)[:100]}")
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "projects", "cad_exports",
                              datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + pick_name(prompt).replace(" ", "_"))
    try:
        os.makedirs(export_dir, exist_ok=True)
        for path in stls + steps:
            shutil.copy2(path, export_dir)
        # The assembly engine drops the spec (and its verify report) beside the meshes.
        # Copy them out too: the staging dir is transient and every run overwrites it,
        # so without this the design that produced these STLs is unrecoverable.
        for sidecar in ("design_spec.json", "verify.txt", "evidence.json"):
            src = os.path.join(work_dir, sidecar)
            if os.path.exists(src):
                shutil.copy2(src, export_dir)
        # Layer 5's shaded views. They are the only artefact in here a person can look
        # at without loading a mesh viewer, so they must survive the staging dir too.
        for png in sorted(glob.glob(os.path.join(work_dir, "*.png"))):
            shutil.copy2(png, export_dir)
        payload["exports"]["dir"] = export_dir
    except Exception as e:
        print(f"[CadAgent DEBUG] [EXPORT] copy skipped: {str(e)[:100]}")
    return payload


class CadAgent:
    def __init__(self, on_thought=None, on_status=None, on_ask=None):
        # Code path is glm-family ONLY (providers.cad_chain(), NVIDIA NIM).
        # gemini-3.7-flash and the Nemotron->Mistral last-resort chain were
        # removed 2026-09-26 per standing order: one model family for CAD.
        self.on_thought = on_thought  # Callback for streaming thoughts 
        self.on_status = on_status  # Callback for retry status info
        # Level 3 clarifier: `async (question, choices, default) -> answer`.
        # None means "cannot ask" — the clarifier then still notices the ambiguity,
        # logs it, and lets the design proceed on its defaults rather than blocking.
        self.on_ask = on_ask
        
        self.system_instruction = """
You are ADA's senior CAD engineer. You write build123d Python scripts that produce REAL, recognizable, manufacturable 3D parts - not abstract blobs.

WORKFLOW - ALWAYS follow this exact order in your response:
1. DESIGN BRIEF: 3-5 short bullet lines describing the part (overall shape, key features, and REAL dimensions in mm). Research the real object in your head: a bottle opener is ~160-200mm long, ~30-40mm wide, ~3-5mm thick, with a curved head, a side notch that grabs the cap, and a lanyard hole. A phone stand is ~90x70mm. A gear is a disc with teeth. Match the real thing.
2. CODE: exactly one ```python block implementing the brief.

PROPORTION RULES:
- The part must be INSTANTLY RECOGNIZABLE as the requested object. If you are unsure of a dimension, pick a plausible real-world one and state it in the brief.
- Parts must be watertight, a single solid, centered near (0,0,0), sized in mm.
- Add the object's signature features (holes, notches, slots, fillets) - these are what make it look like the real thing.

build123d RULES (these are the ONLY safe idioms - deviating causes crashes):
- Start with `from build123d import *`.
- Use exactly ONE `with BuildPart() as p:` block.
- 2D shapes (Rectangle, Circle, Polygon, Ellipse) go ONLY inside `with BuildSketch() as s:`. NEVER call make_face() - it is invalid outside BuildSketch and crashes.
- Draw base shapes FIRST, then holes with `mode=Mode.SUBTRACT` INSIDE THE SAME BuildSketch block (e.g. `Circle(radius, mode=Mode.SUBTRACT)`). NEVER open a second BuildSketch for a subtraction.
- Use `Locations((x, y))` to place shapes within a sketch.
- Extrude with `extrude(amount=...)` right after each sketch.
- To cut a NOTCH from a side edge, place a small SUBTRACT rectangle so it overlaps the edge of the profile.
- For curved profiles use Polyline + RadiusArc/Line + close() inside BuildSketch, then extrude.
- Fillet only with small radii (0.5-2mm); chamfer smaller. If a fillet/chamfer fails, use a smaller value or skip it.
- End with exactly:
  result_part = p.part
  export_stl(result_part, 'output.stl')

MULTI-PART ASSEMBLIES:
- If the request has 2+ parts (e.g. "a gear and a shaft that fit together"), build EACH part as its own `with BuildPart() as pX:` block and export EACH with a distinct name:
  export_stl(p1.part, 'output_0.stl')
  export_stl(p2.part, 'output_1.stl')
  (output_2.stl, output_3.stl... as needed)
- Keep the parts dimensioned so they would FIT together (matching radii/diameters).
- You may use `Pos(...)` / `Locations(...)` inside a BuildPart to position features.

STEP EXPORT:
- For single-part designs, ALSO export a STEP file (the universal engineering format):
  export_step(result_part, 'output.step')
- This lets the user open the part in Fusion 360 / SolidWorks.

EXAMPLE - a flat bottle opener (proven to run):
```python
from build123d import *

with BuildPart() as p:
    with BuildSketch() as s:
        Rectangle(120, 28)                # handle
        with Locations((70, 0)):
            Circle(18)                    # rounded head at one end
        with Locations((70, -16)):
            Rectangle(16, 26, mode=Mode.SUBTRACT)   # side notch that grabs the cap
        with Locations((-45, 0)):
            Circle(5, mode=Mode.SUBTRACT)            # lanyard hole
    extrude(amount=4)
    fillet(p.edges(), radius=1)

result_part = p.part
export_stl(result_part, 'output.stl')
```
"""

    async def _stream_code(self, prompt: str) -> str:
        """Streams a code response for CAD generation.

        GLM ONLY: providers.cad_chain() (NVIDIA NIM, glm-5.3 family).
        Override anytime: ADA_CAD_MODEL='...' (see providers.cad_chain).
        No fallbacks to other families — a second model family here doubles
        the dead-id maintenance for a path that runs only when the assembly
        engine declines. Empty string means the caller falls back gracefully."""
        from providers import chat_with_fallback, cad_chain

        try:
            content = await chat_with_fallback(
                cad_chain(),
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=16384,
                timeout=300.0,
            )
            if content and content.strip():
                if self.on_thought:
                    self.on_thought(f"[CAD-CHAIN] CAD chain delivered {len(content)} chars of build123d code.")
                return content
            print("[CadAgent DEBUG] [CAD-CHAIN] returned empty content.")
        except Exception as e:
            print(f"[CadAgent DEBUG] [CAD-CHAIN] failed ({type(e).__name__}: {str(e)[:120]}).")

        print("[CadAgent DEBUG] [ERR] CAD code path failed.")
        return ""

    async def generate_prototype(self, prompt: str, output_dir: Optional[str] = None):
        """
        Generates 3D geometry by asking a model for a script, then running it LOCALLY.
        Supports multi-part assemblies (output_0.stl, output_1.stl, ...) and an
        INSPECTOR pass that measures the result and refines it when it is broken.
        Args:
            prompt: User's description of the model to generate.
            output_dir: Directory to save the script and STL. If None, uses temp dir.
        """
        print(f"[CadAgent DEBUG] [START] Generation started for: '{prompt}'")

        # ── PRIMARY: the cad_v3 assembly engine ──────────────────────────────
        # The model is asked for a SPEC, not for code. Any part ADA has no word for is
        # synthesised in one batched request and refereed by the geometry gate; the
        # kernel places every part by connector frame and `verify` measures the result.
        # Falls through to the legacy script-writing path below if that cannot produce
        # a buildable spec — a one-off decorative shape has no assembly to speak of.
        if os.getenv("ADA_CAD_ENGINE", "cad_v3").lower() != "legacy":
            try:
                from cad_v3_bridge import design_assembly

                v3_dir = os.path.abspath(output_dir) if output_dir else _cad_staging_dir()
                if self.on_status:
                    self.on_status({"status": "generating", "attempt": 1,
                                    "max_attempts": 1, "error": None})
                # Level 3 runs inside the bridge now: clarifier (via on_ask) -> planner
                # -> designer -> inspector -> refiner. It degrades on its own at every
                # step, so the only thing this call needs to get right is handing over
                # a channel to ask Sir a question.
                rep = await design_assembly(prompt, v3_dir, on_thought=self.on_thought,
                                            ask=self.on_ask)
                if rep and rep.get("stls"):
                    payload = _build_cad_payload(prompt, rep["stls"], rep["steps"], v3_dir)
                    if payload:
                        payload["engine"] = "cad_v3"
                        payload["design"] = {k: rep[k] for k in
                                             ("spec_name", "parts", "instances", "mates",
                                              "model_calls", "skills_written",
                                              "placeholders", "repaired", "rounds",
                                              "verify", "passed", "evidence") if k in rep}
                        payload["evidence"] = rep.get("evidence")
                        payload["name"] = rep.get("spec_name") or payload["name"]
                        print(f"[CadAgent DEBUG] [cad_v3] {rep['spec_name']}: "
                              f"{rep['parts']} parts, {rep['instances']} instances, "
                              f"{'PASS' if rep['passed'] else 'OPEN'}")
                        return payload
                print("[CadAgent DEBUG] [cad_v3] no assembly; using the script path.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[CadAgent DEBUG] [cad_v3] unavailable ({type(e).__name__}: "
                      f"{str(e)[:160]}); using the script path.")

        # TEMPLATE-FREE CUSTOM ROUTE: no fixed template — she invents every time.
        #
        # There used to be a hard-coded branch here that recognised the words
        # "ferrari", "gearbox" and "planetary gearbox" and swapped in a bespoke prompt
        # that read like a template and behaved like one: it told the model how to
        # build *that* machine (gears as cylinders, "never one single block"), and in
        # doing so it was a specific way of building that only worked for the parts it
        # happened to name. It has been removed. The vocabulary engine decides
        # structure from the prompt now, and a special case for one noun is exactly
        # the kind of limitation that made her feel like a part library instead of an
        # engineer.
        try:
            # Fresh staging dir per generation (iterate reuses it via _cad_staging_dir)
            # abspath: subprocess runs with cwd=work_dir, so a relative
            # output_dir would double-resolve and break script execution.
            if output_dir:
                work_dir = os.path.abspath(output_dir)
                os.makedirs(work_dir, exist_ok=True)
            else:
                work_dir = _cad_staging_dir()
            for old in glob.glob(os.path.join(work_dir, "output*.stl")) + glob.glob(os.path.join(work_dir, "output*.step")):
                os.remove(old)
            script_path = os.path.join(work_dir, "current_design.py")

            max_retries = 3
            current_prompt = f"""Design and build a {prompt}.

FIRST write a 3-5 line DESIGN BRIEF with REAL dimensions in mm and the key features.
THEN write exactly one ```python block that implements it with build123d.
The result must be INSTANTLY RECOGNIZABLE as {prompt} - include its signature features.
If the request is an assembly with multiple parts, build EACH part and export them as
'output_0.stl', 'output_1.stl', ... (one per part). A single object uses 'output.stl'."""

            for attempt in range(max_retries):
                print(f"[CadAgent DEBUG] Attempt {attempt + 1}/{max_retries}")

                if self.on_status:
                    status_info = {
                        "status": "generating" if attempt == 0 else "retrying",
                        "attempt": attempt + 1,
                        "max_attempts": max_retries,
                        "error": None
                    }
                    self.on_status(status_info)

                raw_content = await self._stream_code(current_prompt)

                if not raw_content:
                    print("[CadAgent DEBUG] [ERR] Empty response from model.")
                    return None

                import re
                code_match = re.search(r'```python(.*?)```', raw_content, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                else:
                    print("[CadAgent DEBUG] [WARN] No ```python block found. Trying heuristic...")
                    if "import build123d" in raw_content:
                        code = raw_content
                    else:
                        print("[CadAgent DEBUG] [ERR] Could not extract python code.")
                        return None

                with open(script_path, "w") as f:
                    f.write(code)

                print(f"[CadAgent DEBUG] [EXEC] Running local script: {script_path}")

                import subprocess
                import sys

                try:
                    proc = await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, script_path],
                        capture_output=True,
                        text=True,
                        cwd=work_dir,
                    )
                    stdout, stderr = proc.stdout, proc.stderr
                except Exception as e:
                     print(f"[CadAgent DEBUG] [ERR] Subprocess run failed: {e}")
                     proc = type('obj', (object,), {'returncode': 1})
                     stdout = ""
                     stderr = str(e)

                if proc.returncode != 0:
                    error_msg = stderr
                    error_lines = error_msg.strip().split('\n')
                    short_error = error_lines[-1][:100] if error_lines else "Unknown error"
                    print(f"[CadAgent DEBUG] [ERR] Script Execution Failed:\n{error_msg}")

                    if self.on_status:
                        self.on_status({
                            "status": "retrying",
                            "attempt": attempt + 1,
                            "max_attempts": max_retries,
                            "error": short_error
                        })

                    current_prompt = f"""
The Python script you generated failed to execute with the following error:
{error_msg}

Please fix the code to resolve this error. Return the full corrected script. 
Ensure you still export to 'output.stl' (or 'output_0.stl', 'output_1.stl'... for assemblies).

{_build123d_hint(error_msg)}
"""
                    continue

                print(f"[CadAgent DEBUG] [OK] Script executed successfully.")

                stls, steps = _collect_outputs(work_dir)
                if not stls:
                    print("[CadAgent DEBUG] [ERR] No output*.stl generated.")
                    current_prompt = "The script executed successfully but no output*.stl was written. Ensure you call `export_stl(result_part, 'output.stl')` at the end (or 'output_0.stl' etc for assemblies)."
                    continue

                # INSPECTOR: measure the geometry; refine once if it is broken
                parts_info, issues = _inspect_stls(stls)
                print(f"[CadAgent DEBUG] [INSPECT] {parts_info}")
                if issues and attempt < max_retries - 1:
                    if self.on_status:
                        self.on_status({
                            "status": "retrying",
                            "attempt": attempt + 1,
                            "max_attempts": max_retries,
                            "error": issues[:100]
                        })
                    current_prompt = f"""
Your design generated geometry, but the INSPECTOR measured it and found problems:
{issues}

Measurements of what you produced:
{parts_info}

The DESIGN BRIEF you wrote said it should be a recognizable {prompt} with real proportions.
Fix the script so the part is a proper solid with realistic dimensions. Return the full corrected script.
"""
                    continue

                payload = _build_cad_payload(prompt, stls, steps, work_dir)
                if payload:
                    return payload
                print("[CadAgent DEBUG] [ERR] Payload build failed.")
                return None

            print("[CadAgent DEBUG] [ERR] All attempts failed.")
            if self.on_status:
                self.on_status({
                    "status": "failed",
                    "attempt": max_retries,
                    "max_attempts": max_retries,
                    "error": "All generation attempts failed"
                })
            return None

        except Exception as e:
            print(f"CadAgent Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def iterate_prototype(self, prompt: str, output_dir: Optional[str] = None):
        """
        Iterates on the existing design by reading 'current_design.py' and applying changes.
        Args:
            prompt: User's description of the changes to make.
            output_dir: Directory containing existing script and where to save new STL.
        """
        print(f"[CadAgent DEBUG] [START] Iteration started for: '{prompt}'")

        work_dir = os.path.abspath(output_dir) if output_dir else _cad_staging_dir()
        os.makedirs(work_dir, exist_ok=True)
        script_path = os.path.join(work_dir, "current_design.py")

        existing_code = ""

        if os.path.exists(script_path):
            with open(script_path, "r") as f:
                existing_code = f.read()
        else:
             print("[CadAgent DEBUG] [WARN] No existing script found. Falling back to fresh generation.")
             return await self.generate_prototype(prompt)

        try:
            max_retries = 3
            current_prompt = f"""
You are iterating on an existing 3D model script.

Current Python Code:
```python
{existing_code}
```

User Request: {prompt}

Task: Rewrite the code to satisfy the user's request while maintaining the rest of the model structure.
Keep exporting to the same output file names ('output.stl' for a single part, or 'output_0.stl', 'output_1.stl'... for assemblies).
"""

            for attempt in range(max_retries):
                print(f"[CadAgent DEBUG] Iteration Attempt {attempt + 1}/{max_retries}")

                if self.on_status:
                    status_info = {
                        "status": "generating" if attempt == 0 else "retrying",
                        "attempt": attempt + 1,
                        "max_attempts": max_retries,
                        "error": None
                    }
                    self.on_status(status_info)

                for old in glob.glob(os.path.join(work_dir, "output*.stl")) + glob.glob(os.path.join(work_dir, "output*.step")):
                    os.remove(old)

                raw_content = await self._stream_code(current_prompt)

                if not raw_content:
                    print("[CadAgent DEBUG] [ERR] Empty response from model.")
                    return None

                import re
                code_match = re.search(r'```python(.*?)```', raw_content, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                else:
                    print("[CadAgent DEBUG] [WARN] No ```python block found. Trying heuristic...")
                    if "import build123d" in raw_content:
                        code = raw_content
                    else:
                        print("[CadAgent DEBUG] [ERR] Could not extract python code.")
                        return None

                with open(script_path, "w") as f:
                    f.write(code)

                print(f"[CadAgent DEBUG] [EXEC] Running local script: {script_path}")

                import subprocess
                import sys

                try:
                    proc = await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, script_path],
                        capture_output=True,
                        text=True,
                        cwd=work_dir,
                    )
                    stdout, stderr = proc.stdout, proc.stderr
                except Exception as e:
                    print(f"[CadAgent DEBUG] [ERR] Subprocess run failed: {e}")
                    proc = type('obj', (object,), {'returncode': 1})()
                    stdout = ""
                    stderr = str(e)

                if proc.returncode != 0:
                    error_msg = stderr
                    print(f"[CadAgent DEBUG] [ERR] Script Execution Failed:\n{error_msg}")

                    current_prompt = f"""
The updated Python script you generated failed to execute with the following error:
{error_msg}

Please fix the code to resolve this error. Return the full corrected script. 
Keep the same output file names.

{_build123d_hint(error_msg)}
"""
                    continue

                print(f"[CadAgent DEBUG] [OK] Script executed successfully.")

                stls, steps = _collect_outputs(work_dir)
                if not stls:
                    print("[CadAgent DEBUG] [ERR] No output*.stl generated.")
                    current_prompt = "The script executed successfully but no output*.stl was written. Keep the same output file names ('output.stl' or 'output_0.stl' etc)."
                    continue

                parts_info, issues = _inspect_stls(stls)
                print(f"[CadAgent DEBUG] [INSPECT] {parts_info}")
                if issues and attempt < max_retries - 1:
                    current_prompt = f"""
Your updated design generated geometry, but the INSPECTOR measured it and found problems:
{issues}

Measurements of what you produced:
{parts_info}

Fix the script so the part is a proper solid with realistic dimensions. Return the full corrected script.
"""
                    continue

                payload = _build_cad_payload(prompt, stls, steps, work_dir)
                if payload:
                    return payload
                print("[CadAgent DEBUG] [ERR] Payload build failed.")
                return None

            print("[CadAgent DEBUG] [ERR] All attempts failed.")
            if self.on_status:
                self.on_status({
                    "status": "failed",
                    "attempt": max_retries,
                    "max_attempts": max_retries,
                    "error": "All iteration attempts failed"
                })
            return None

        except Exception as e:
            print(f"CadAgent Error: {e}")
            import traceback
            traceback.print_exc()
            return None

