# ADA CAD — Reaching Its Actual Ceiling

**Written for Naz.** Revised after his correction: CAD modelling, physics testing, and
scene building are **three separate things** and must not be welded into one pipeline.

---

## Build status (2026-09-16)

**Phase 1 (mode separation) and Phase 2 (physics mode) are built and tested.** 70 tests
pass across the CAD and memory suites.

| New module | What it does |
|---|---|
| `backend/cad_v3/physics.py` | Exact B-rep interference, mass properties, clearance sweep, PASS/FAIL verdicts |
| `backend/cad_v3/modes.py` | `Mode` enum, intent detection, the three output builders |
| `backend/cad_v3/render/blender.py` | Blender scene backend — subprocess, 6 styles, 5 shots |
| `tests/test_cad_modes.py` | 41 tests, including that CAD output never mentions a scene |

Wired into `ada.py` as two new tools, `test_model` and `render_scene`, with a completion
report on every build.

### Measured on the real turbocharger (57 parts, 448,404 faces)

```
MODE 1 — CAD
  turbocharger — 57 parts, build pass.
    • Interference audit: PASS — no solid interpenetration.
    • Mass properties:  PASS — 822.3 cm³, 4.642 kg, CoM (0.3, -9.1, 20.7) mm.
        turbine_housing:  cast_iron — 2.095 kg
        compressor_cover: aluminium — 0.705 kg
        bearing_housing:  cast_iron — 0.611 kg
        vane_ring:        inconel   — 0.142 kg

MODE 2 — PHYSICS          (14.0s + 2.6s)
  Interference audit: PASS — 141 candidate pairs, 0 buried, 0 touching.
  Mass properties:    PASS — 4.642 kg.

MODE 3 — SCENE            (~87s, Cycles CPU)
  studio    → photoreal three-quarter, 57 parts got spec materials
  technical → orthographic front view, neutral lighting
```

### Two bugs found by building it

1. **`inst.solid` already carries its world placement.** Calling `.located(Location())`
   — which the rasteriser does, correctly, to get local geometry — silently moved every
   part to the origin. A block centred at x=5 measured a centre of mass at x=0, and every
   interference test was meaningless. Verified directly before fixing.
2. **The materials table is in g/cm³, not kg/m³.** `volume_mm3 * 1e-9 * rho` is only
   correct for kg/m³, so the first mass figure was 1000× out. Now uses
   `materials.mass_g`, which knows its own units.

Plus: `materials.get()` silently falls back to steel for an unknown name, so an unset
material would have weighed as steel. Unknown materials are now reported as a lower
bound instead.

### Blender tuning, measured
Exposure had to come down **~7×** from the first attempt — the initial render was blown
out to white plastic. Environment strength 0.55 → 0.06. Overexposure is the default
failure of a three-point rig: each light is reasonable alone and the sum is not.

---

## 0. The correction, and why it matters

My first version of this document had one CAD story with scene-building bolted on the end.
That was wrong. What's actually needed:

| Mode | When | What she does | What she must NOT do |
|---|---|---|---|
| **CAD** | You're building something | Geometry, dimensions, tolerances, validity, drawings | No scenery. No art direction. No pretty backdrop |
| **Physics** | You built it, now test it | Motion, clearance, kinematics, dynamics, stress handoff | No rendering-for-show |
| **Scene** | You explicitly asked for a scene | Compose a scene, light it, render it | Never trigger itself during CAD work |

The reason this matters beyond tidiness: **a scene is a claim about presentation, and a CAD
model is a claim about engineering.** Mixing them is how you get output that looks
authoritative and asserts nothing. If you're building a gearbox, a beautifully lit workbench
scene is actively harmful — it hides the geometry you're trying to inspect.

---

## 1. "Like art, not like Jarvis" — what I found when I checked

You said her models look like art rather than real machines. I tested that claim rather
than assuming it, and **I got it wrong twice before I got it right.** Both mistakes are
worth recording, because they are the same mistake.

### Mistake 1 — I called the turbocharger broken

I tested for interference by counting intersecting triangles. It reported **32 colliding
part pairs**, including `turbine_flange ↔ turbine_housing` with 3,893 shared triangles. I
was about to tell you the assembly was unmanufacturable.

That was wrong. **Two parts that merely touch also share triangles** — their surfaces are
coincident. Triangle count is a proxy for interference, not a measurement of it.

Re-measured by **sampling solid volume** — is a point inside *both* bodies? — the real
answer:

```
candidate pairs (bounds + triangle contact) : 7
  BURIED   (overlap >= 2% of the smaller part) : 0
  TOUCHING (overlap below that)                : 7
    largest overlap: 0.43% (285 mm³, turbine_housing <-> turbine_wheel)
    the rest: 0.00%
```

**Zero buried parts. The turbocharger is a valid assembly.** My alarm was a false positive
from a bad metric.

### Mistake 2 — I called the mechanism jammed

Then I swept the rotating group through 360° and reported **501 colliding pairs, 36,072
collision events**. Again wrong — for two reasons:

- **No baseline.** Parts already in contact at rest report overlap at every angle. A
  rotation collision is one whose overlap *grows* as the part turns.
- **Bad grouping.** I put `bearing_housing` in the rotating group. A bearing housing is the
  stationary carrier. I was classifying parts by string-matching their names.

Corrected — baseline at rest, then find pairs whose overlap grows:

```
pairs with contact at rest                      : 499
pairs whose overlap ever exceeded at-rest value : 0

VERDICT: the rotating group clears the static parts through a full revolution.
         The mechanism is kinematically valid.
```

### So what does the evidence actually say?

**ADA's geometry is better than you think. Its presentation and its validation story are
what's missing.**

The models are not art in the sense of being fake. They are art in the sense that
**nothing about them is ever demonstrated.** A valid assembly and a picture of a valid
assembly are indistinguishable when all you ever produce is a picture. Nothing in the
current pipeline says *"this clears at every angle"* or *"this weighs 4.2 kg"* or *"this
shaft has 0.05 mm running clearance"* — so the output reads as decoration even though the
geometry underneath is sound.

**That reframes the whole plan.** The fix is not "model better". It is **prove it** — and
that is precisely what Physics mode is for.

---

## 2. What ADA has today (audited)

**`backend/cad_v3/` — 14,501 lines across ~30 modules.**

| Module | Lines | Role |
|---|---|---|
| `repair.py` | 57K | Fixing broken geometry |
| `synth.py` | 40K | Feature synthesis |
| `layout.py` | 33K | Spatial arrangement |
| `features.py` | 32K | Geometry skills |
| `assembler.py` | 27K | Assembly from connector mates |
| `gate.py` / `validate.py` / `verify.py` | 27/30/24K | The referee loop |
| `fea_solver.py` | 13K | **Von Mises stress, 16 materials, safety factors** |
| `render/` | 987 | Software rasterizer + exporters |

### The architectural bet, which is correct

`brain.py`: *"It writes a SPEC. It never writes geometry."* The model emits a typed spec —
diameters, mates, fits — and the kernel builds it. A model asked for CadQuery code must
hold a coordinate frame in its head and will silently place a cutter in the wrong one.

**The 2026 benchmark data agrees.** Text2CAD-Bench measured output representations head to
head: switching one model from Python to DeepCAD-style command sequences took its
invalidity rate from **13.3% → 67.3%**. A constrained representation beats free code. ADA
already does the constrained-representation thing.

### Real output exists

`out/` and `out_ada/`: STEP (B-rep), GLB, STL, exploded and x-ray views, SVG drawings —
for a turbocharger (57 parts, 448,404 faces) and a planetary gearbox.

### The rendering gap

`render/shaded.py` is a hand-rolled numpy rasterizer — own z-buffer, edge detection, ink
pass. Impressive, and it cannot do: PBR materials, environment lighting, shadows, AO, GI,
post-processing, animation. That's Blender's list. But see §5 — **this is not the priority.**

---

## 3. Blender: what it can and cannot do

Verified on your machine, **Blender 5.2.1 LTS** at `/Applications/Blender.app`. `bpy` is
**not** importable into the venv — subprocess is the only route.

### Proven by spike

| Capability | Result |
|---|---|
| Headless Python | Works |
| Import ADA's GLB | 57 objects, 448,404 faces |
| Cycles render, CPU | **19s** basic / **32s** with sky + materials |
| Material response | Reads as brushed aluminium (specular, anisotropic, sky reflections) |
| Solid interference (BVH + volume sampling) | Works, ~1.3s for 57 parts |
| Rotation sweep + clearance check | Works, ~19s for 36 angles |

### The honest boundary

| Blender **can** | Blender **cannot** |
|---|---|
| Rigid-body dynamics, collisions, contact | FEA stress analysis → **`fea_solver.py` already does this** |
| Constraints/joints: hinge, slider, motor | CFD / fluid flow |
| Kinematic sweeps, range-of-motion, clearance | Real manufacturing tolerances |
| Soft body, cloth, particles | Replace a CAD kernel |
| Photoreal rendering, animation, scene composition | |

So **Physics mode is kinematics and dynamics**, and it hands stress off to the FEA solver
that already exists. That division is a strength, not a gap.

### Three Blender 5.x API gotchas

1. **Sky types renamed:** `NISHITA` → `MULTIPLE_SCATTERING`.
2. **`view_transform` is a dynamic enum.** Introspecting it before the OCIO config loads
   returns a placeholder `'NONE'`, so "read the choices, pick one" throws `TypeError`. Set
   it by attempting values in order.
3. **Gravity moved** to `scene.gravity`; `rigidbody_world.effective_gravity` no longer exists.

---

## 4. The plan — three modes, one shared kernel

### Mode 1 — CAD (default when building)

**The change: stop producing pictures, start producing engineering evidence.**

- **Part view, not art.** Orthographic, flat, measurable. Dimension lines, section views,
  detail callouts. `render/drawing.py` already does SVG technical drawings — promote this to
  the *default* CAD output.
- **Validity report on every build.** Run the volume-based interference check (spike 3b)
  automatically and state it: *"57 parts, 0 buried, largest contact 0.43%."* That single
  sentence is what converts "art" into "an engineering artifact", and it costs 1.3 seconds.
- **Mass properties.** Volume, mass from the spec's material density, centre of mass. Real
  numbers, not adjectives.
- **Key dimensions in the reply.** "Shaft Ø8 h6, running clearance 0.05 mm" — the things an
  engineer actually asks for.
- **No scene, no backdrop, no mood lighting.** White or transparent background. If you want
  a render, that's Mode 3.

**Acceptance:** building a gearbox returns geometry, a drawing, a validity verdict and mass
properties — and no scenery.

---

### Mode 2 — Physics (you built it, now test it) ⭐ *the one you asked for*

A distinct, explicitly-invoked capability. **It never runs during CAD.**

- **Clearance / range-of-motion sweep** — the spike-4b test, generalised. Drive each joint
  through its full travel and report whether anything collides. Baseline-relative, so parts
  in contact at rest are not false positives.
- **Solid interference audit** — the spike-3b test. Volume-measured, never triangle-counted.
- **Rigid-body dynamics** — gravity, contact, drop and settle. Real Bullet simulation.
- **Joint/motor simulation** — hinge + motor on the shaft, run for N seconds, report whether
  it turns freely or binds.
- **Stress handoff** — geometry + load case → `fea_solver.py` → Von Mises, safety factor.
  Blender does not do this; ADA already does.
- **A written verdict.** Not a number dump: *"The rotating group clears at all 36 angles,
  minimum running clearance 0.4 mm, no interference. Bearing housing sees 12 MPa at 40k rpm,
  safety factor 3.4 against 6061-T6 yield."*

**The design rule:** every physics test returns **PASS / FAIL / INCONCLUSIVE** plus the
evidence. A test that cannot decide says so — it does not round up to "looks fine".

**Acceptance:** "test the turbocharger" returns a verdict naming the test, the result, and
the numbers behind it.

---

### Mode 3 — Scene (only when asked) ⭐

Never implicit. Triggered by explicit language — "show me", "render a scene", "put it on a
workbench", "make it look good".

- Scene builder: camera framing, light rig scaled to the model, ground plane, environment.
- **Environment library** — studio, workshop, outdoor, dramatic, holographic.
- **Named camera shots** — hero, macro detail, three-quarter, top orthographic.
- **Motion** — exploded and assembly sequences from the connector graph (the mate graph
  already encodes what attaches to what, so explode directions are derivable), turntable,
  cutaway sweep. MP4/GIF out.
- **Holographic register** — emission + fresnel + bloom + build-on scan lines. Purely
  cosmetic; do it last.

**Acceptance:** "put the gearbox on a workbench with dramatic lighting" produces a composed
scene. "Build me a gearbox" produces no scene at all.

---

## 5. The hard problem, which is none of the above

Text2CAD-Bench (600 models, 10 LLMs) measured where text-to-CAD actually breaks:

| Complexity | Invalidity rate |
|---|---|
| **L1** basic | ~15% |
| **L2** intermediate | rising |
| **L3** advanced — **sweep, loft, shell** | **70–90%** |
| **L4** real-world | 17–81% |

Best models at L3: GPT-5.2 **68% invalid**, Claude-4.5 **70%**. And the sharpest finding:
**executability is not correctness.** One model hit 2% invalidity while scoring the *worst*
geometric similarity — runnable code that modelled the wrong part.

**This is where the real ceiling is.** Not rendering, not scenes. L3 is where ADA and
everyone else falls over. Moves:

1. **Decompose L3 into L2.** A sweep is profile + path + boolean. If the brain can't emit a
   sweep reliably, emit the decomposition and let the kernel recognise it. Trade model
   capability for kernel capability — the kernel is the part that works.
2. **Constrain the vocabulary until unsupported ops are unrepresentable**, not merely
   discouraged.
3. **Verify-then-repair as the default loop.** `verify.py` + `repair.py` are 81K lines of
   the right machinery. Measure how often they run today versus how often they could.
4. **Build a local 50-part benchmark.** You cannot improve L3 without measuring it. Grade
   each part on *did it build* **and** *is it right*.
5. **Learn from rejections.** Every spec the kernel refuses is training data.

**ADA's own L3 rate is unmeasured.** Given the field sits at 70–90% invalid, **50% valid
would be genuinely strong.** Take this number before investing in the visual phases.

---

## 6. What NOT to build

1. **Don't render scenes during CAD work.** This was your correction and it's right.
2. **Don't trust triangle overlap as interference.** I did, twice, and both times it was
   wrong. Volume-sample.
3. **Don't classify parts by string-matching names.** `bearing_housing` does not rotate.
   Read the mate graph.
4. **Don't replace the numpy rasterizer.** Instant, offline, dependency-free. Right tool
   for previews.
5. **Don't chase real-time interactivity.** Movie JARVIS is pre-rendered footage.
6. **Don't buy a GPU.** Cycles on CPU did 448k faces in 20s.
7. **Don't skip the L3 benchmark to get to pretty renders faster.** Backwards.

---

## 7. How to measure it

| What | How | Today | Target |
|---|---|---|---|
| **L3 validity** | % of 50 own parts that build **and** are correct | **unknown** | ≥50% |
| Interference | Buried pairs per assembly, volume-measured | not run | 0 |
| Clearance | Min running clearance over full travel | not run | measured |
| CAD output | Drawing + validity + mass on every build | picture only | always |
| Scene leakage | Scenes produced when not asked | n/a | **0** |
| Render quality | Human 1–5, fixed set of 10 | flat grey | 4+ |

The **scene leakage** row is a correctness property, not a nice-to-have. It is the thing you
just told me was broken.

---

## 8. Suggested order

**Phase 1 — Mode separation** (small, high value). Route output through three explicit
modes. CAD returns engineering evidence and never a scene. Mostly plumbing, and it directly
fixes what you flagged.

**Phase 2 — Physics mode** (the capability you asked for). Interference audit, clearance
sweep, dynamics, joint simulation, stress handoff, written verdicts. Reuses the spike code.

**Phase 3 — L3 validity** (the real ceiling). Local benchmark first, then decomposition and
vocabulary constraint.

**Phase 4 — Scene mode** (presentation). Blender render backend, materials, animation,
environments, holographic register.

My recommendation: **1 → 2 → 3 → 4.** Phase 1 is a few days and fixes the thing that annoyed
you. Phase 2 makes the models demonstrably real. Phase 3 is where the remaining engineering
value lives.

---

## Sources

**Measured this session** (all on your M1 Air, CPU only)
- Blender 5.2.1 LTS headless; `bpy` not importable into the venv
- Spike 1 (`_spike_blender.py`): 57 objects / 448,404 faces, 19.1s Cycles render
- Spike 2 (`_spike_blender2.py`): sky + scaled lights + 3 metals, 32.2s
- Spike 3 (`_spike_blender3.py`): 32 triangle-overlap pairs — **false positive**
- Spike 3b (`_spike_blender3b.py`): volume-measured → **0 buried, 7 touching, max 0.43%**
- Spike 4 (`_spike_blender4.py`): 501 colliding pairs — **false positive**
- Spike 4b (`_spike_blender4b.py`): baseline-relative → **0 pairs grow, mechanism clears**
- Reports: `out/_spike_interference_report.txt`, `out/_spike_mechanism_report.txt`

**Research**
- Text2CAD-Bench (arXiv 2605.18430, 2026) — 600 models, 10 LLMs; L3 invalidity 70–90%;
  best 68–70%; representation matters (13.3% → 67.3%); executability ≠ correctness
- Embodied CAD (2026), AssemCAD, ArtiCAD — basis for the spec-not-code architecture

**Codebase**
- `backend/cad_v3/` — `brain.py`, `schema.py`, `assembler.py`, `fea_solver.py`, `render/`
- `ADA_CAD_V3_ASSEMBLY_ENGINE.md` — the existing v3 build plan
