# ADA CAD v3 — The Assembly Engine

**Goal:** produce the four reference images for real — a 94-part turbocharger with a backswept
helical impeller, real ISO threads, deep-groove bearing races, gasket-grooved flanges — from a
sentence, with zero presets and zero AI-authored coordinates.

**Status:** plan grounded in an executed capability spike, not speculation.
Spike: [`backend/cad_v3_proto/spike_vocabulary.py`](backend/cad_v3_proto/spike_vocabulary.py)

---

## 1. Verdict

Your five-layer plan is the right architecture. It is also **not what ADA currently does**, in two
ways that matter more than the plan admits:

1. **ADA is preset-driven today, structurally.** [`spec_compiler.py`](backend/cad_v2/spec_compiler.py)
   is 2,160 lines of keyword regex → hardcoded part lists, including a literal
   `_arc_reactor_parts()` and `_complex_gripper_module_parts()`. Layer 2 does not sit on top of
   this — it *replaces* it. That deletion is the single biggest change in the whole plan.
2. **ADA cannot assemble anything today.** [`mate_solver.py:49`](backend/cad_v2/mate_solver.py:49)
   is a 184-line toy that mutates `transform["x"]/["y"]/["z"]/["rz"]` scalars. `coaxial` copies x,y
   and nothing else. `mate` mode `"auto"` appends a success record and does *literally nothing*.
   Parts are then spread along X by `_deoverlap_transforms()` so they don't visually collide.
   That is arranging, not assembling. Layer 3 is therefore not an upgrade — it is the first real
   assembler ADA will have.

The good news is bigger: **the kernel you already have can do all of it, and the frame algebra you
were going to write is already in build123d and completely unused.** Verified below.

---

## 2. Audit — what ADA actually is (evidence)

### Two generation paths, both wrong for a turbocharger

| Path | Entry | What it does | Why it fails on the reference images |
|---|---|---|---|
| Preset | `compile_spec()` → `_complex_assembly_parts()` | keyword match → fixed part list | Turbocharger has no branch. Falls to generic box+cylinder. |
| Agentic | `agentic_coder.generate_part_code()` | LLM writes raw build123d per part | Exactly the *Embodied CAD* failure: valid tokens, wrong coordinate frame. |
| Fallback | `kernel_build123d._fallback_body_for_role()` | 5 hardcoded bodies (bracket/enclosure/shaft/standoff/box) | Another preset layer, mislabelled "emergency". |

### The parameter space cannot express the parts

[`_default_params()`](backend/cad_v2/kernel_build123d.py:70) is a flat dict of ~40 scalars —
`width_mm`, `depth_mm`, `hole_count`, `heat_sink_fin_count`, `composite_ply_count`. Every fallback
body is a `Box` or `Cylinder` plus subtractive `Cylinder`s.

There is **no `revolve`, no `loft`, no `sweep`, no `Helix`, no thread, no profile curve** anywhere
in the vocabulary. One exception: [`primitives/involute_gear.py`](backend/cad_v2/primitives/involute_gear.py)
does real involute maths — proof the team can do this, applied to exactly one part type.

A compressor wheel is a swept 3D spline. A volute is a revolved+lofted scroll. Neither is
*expressible* in the current parameter space at any level of prompt cleverness.

### Everything downstream is real and worth keeping

- [`assembly_graph.py`](backend/cad_v2/assembly_graph.py) — undo/redo, snapshots, typed diffs, op log. 672 lines, solid.
- `session_store.py`, `operation_engine.py`, `feature_timeline.py` — persistence and history.
- `dfm_validator.py`, `intent_validator.py`, `adversarial_validator.py`, `quality_benchmark.py` — a real validation harness.
- [`CadWindow.jsx`](src/components/CadWindow.jsx) — already has explode, trails, hologram mode, per-part PBR materials.

**Keep all of it.** v3 replaces the geometry layer, not the app.

### Latent bugs found in passing

- `exporters.py` requires `trimesh` for GLB. **trimesh is not installed** → GLB export is dead code today.
- `standard_parts.load_catalog()` reads `assets/cad/standard_parts/catalog.json`. **That file does not exist** → the catalog is permanently `{}` and `standard_part_id` is a no-op.
- Missing deps: `bd_warehouse`, `trimesh`, `networkx`, `manifold3d`, `ocp_vscode`.

---

## 3. The capability spike — measured, not assumed

Ran on this machine: build123d **0.11.1**, OCCT **7.9.3.1**, Python 3.11, no LLM in the loop.

```
 ok  impeller           2.13s  9 backswept blades, swept aerofoil on 3D spline, vol=6883mm³, valid=True
 ok  iso_thread         3.86s  12.8 real helical turns, ISO 68-1 form (h=0.6134p), valid=True
 ok  bearing            0.06s  2 races w/ toroidal grooves + 8 balls, valid=True
 ok  flange             0.14s  12×M6 bolt circle + revolved gasket groove + fillet, valid=True
 ok  joint_mating       0.27s  RigidJoint.connect_to() → impeller origin at z=92 exactly
 ok  clash_detection    1.11s  21 B-rep pairs, 0 interferences
 ok  export             9.72s  STEP 1.2MB + binary glTF, 7 coloured parts
TOTAL 17.29s   parts_built=7
```

Artifacts: `backend/cad_v3_proto/turbo_core.step`, `turbo_core.glb`.

**Four findings that change the plan:**

1. **`RigidJoint` / `RevoluteJoint` / `CylindricalJoint` / `BallJoint` + `.connect_to()` already
   exist in build123d and cad_v2 never imports them.** This is full OCCT frame algebra —
   position *and* orientation, composed correctly. Layer 3's "connector-frame alignment, pure
   math" is a library call, not a solver to write.
2. **`IsoThread` and `TrapezoidalThread` are absent from build123d 0.11.1** — confirmed by module
   scan. They moved to `bd_warehouse.thread`. So the bd_warehouse install is load-bearing, not
   optional. (My spike hand-rolled a helix-swept ISO thread in 3.86s as proof it's possible, but
   3.9s × 40 fasteners = 156s. Use the library.)
3. **B-rep clash detection is cheap and exact** — 21 pairs in 1.11s via bbox reject then
   `a.intersect(b).volume`. No AI, no heuristic. This is Layer 3's auto-verification, done.
4. **Tessellation is the real scaling risk.** 7 parts → 13.4 MB glTF at `linear_deflection=0.001`.
   At 94 parts that's ~180 MB. Needs LOD + instancing (see §10, Phase 6).

---

## 4. Where your plan needs correcting

| Your plan | Correction |
|---|---|
| "bd_warehouse — already exists, just install!" | Right library, but it contains **no turbomachinery**. Confirmed modules: `thread`, `fastener`, `bearing`, `flange`, `pipe`, `gear`, `sprocket`, `o_rings`, `retaining_ring`, `shaft_key`, `open_builds`. Impeller, turbine wheel, volute, diffuser vanes, nozzle ring, thrust bearing, wastegate — **ADA must author these as skills.** That's ~14 skills of real work, not a pip install. |
| "Mates resolved as connector-frame alignment — pure math" | Don't write it. Use build123d `Joint.connect_to()`. Verified working. Rewriting it throws away OCCT's frame composition and reintroduces the exact orientation bugs you're trying to eliminate. |
| Spec as free-form JSON | Must be a **typed, validated contract** (pydantic v2). An unschema'd spec lets the brain invent `{skill:"helical_gear", teeth:21}` where the skill expects `tooth_count` — silent field drop, wrong geometry, no error. Unknown skill or unknown param → **hard reject before the kernel runs.** This is what converts invisible token-level errors into loud compile-time ones. |
| Layer 5 technical-drawing mode from the viewer | STL has no edge topology, so you cannot get image 4's line art from the current STL pipeline. But `build123d.drafting` exists (`DimensionLine`, `ExtensionLine`, `TechnicalDrawing`) plus `ExportSVG` and `section()` — **server-side SVG projection from B-rep**. That also gets you image 1's GD&T callouts, which no viewer toggle can produce. |

One thing your plan understates: **per-part subprocess + per-part LLM call does not scale.**
Today each part is one `subprocess.run()` plus one LLM round-trip. At 94 parts that is 94
subprocesses and 94 completions. The spike built 7 real parts **in-process in 17s with zero LLM
calls.** Skills must run in a persistent in-process worker pool. The LLM is called **once** for the
spec, plus once per repair.

---

## 5. Target architecture

```
backend/cad_v3/
  schema.py         # pydantic v2 SPEC contract — the whole API surface between brain and kernel
  brain.py          # LLM: prompt -> Spec (schema-constrained) ; failure -> SpecPatch
  registry.py       # skill name -> (build_fn, ParamModel, connector_names) ; unknown = hard reject
  vocabulary/
    profiles.py     #     revolved bodies from named profile curves (housings, hubs, scrolls)
    turbo.py        # ADA impeller, turbine_wheel, volute_scroll, diffuser_vanes, nozzle_ring
    power.py        # ADA shaft, spline, thrust_collar, oil_gallery, labyrinth_seal
    rotary.py       #  bd  spur/helical/bevel gear, ring_gear        (bd_warehouse.gear)
    bearings.py     #  bd  deep_groove / angular_contact / journal   (bd_warehouse.bearing)
    fasteners.py    #  bd  screws, nuts, washers, tapped holes       (bd_warehouse.fastener)
    threads.py      #  bd  IsoThread, AcmeThread, trapezoidal        (bd_warehouse.thread)
    retaining.py    #  bd  snap rings + grooves, keys + keyways      (retaining_ring, shaft_key)
    sealing.py      #  bd  o-ring glands, gasket grooves             (bd_warehouse.o_rings)
    structure.py    # ADA flange, bolt_circle, rib_web, boss, fin_array, brake_formed_bracket
  frames.py         # Connector convention — every skill publishes named joints
  assembler.py      # spec.mates -> topological connect_to() chain. No AI coordinates. Ever.
  verify.py         # B-rep clash, watertight, connectivity graph, mass props, DFM
  repair.py         # typed failure -> SpecPatch -> rebuild ONE node
  worker.py         # persistent in-process build pool + param-hash cache
  render/
    tessellate.py   # LOD glTF, instanced repeats
    drawing.py      # build123d.drafting: section, exploded, leader lines, GD&T -> SVG
```

`cad_v2` stays mounted and untouched. v3 is selected by a flag so nothing regresses mid-migration.

---

## 6. The SPEC contract

This is the whole interface between the brain and the kernel. If it isn't in this schema, the brain
cannot say it — which is precisely the property that kills coordinate hallucination.

```python
class ParamValue(RootModel[float | int | str | bool | list[float]]): ...

class PartSpec(BaseModel):
    id: str                       # unique, referenced by mates
    skill: str                    # MUST exist in registry — else hard reject
    params: dict[str, ParamValue] # MUST validate against skill's ParamModel
    material: MaterialEnum
    count: int = 1                # >1 => instanced, one mesh N transforms
    pattern: PatternSpec | None = None   # polar / linear / grid — kernel-generated

class MateSpec(BaseModel):
    a: ConnectorRef               # "shaft.compressor_end"  — part.connector
    b: ConnectorRef               # "impeller.bore"
    kind: Literal["rigid","revolute","cylindrical","ball","linear"] = "rigid"
    offset_mm: float = 0.0
    angle_deg: float = 0.0
    fit: Literal["press","slip","clearance","interference"] = "slip"

class Spec(BaseModel):
    name: str
    intent: str                   # free text, for the critics — never for geometry
    units: Literal["mm"] = "mm"
    datum: str                    # id of the root part; everything mates back to it
    parts: list[PartSpec]
    mates: list[MateSpec]

    @model_validator(mode="after")
    def _graph_is_sound(self):
        # every ConnectorRef resolves; mate graph is connected and acyclic from datum;
        # no part is unreachable (that's the "bolts in mid-air" check, done at spec time)
```

The critical validators — the ones that make this different from prompting harder:

1. **Unknown skill → reject.** The brain cannot invent `skill:"turbo_magic"`.
2. **Unknown or mistyped param → reject** with the skill's actual signature in the error, which is
   what the repair loop consumes.
3. **Unresolvable connector → reject.** `impeller.bore` must be a name the impeller skill publishes.
   This is where "bolt in mid-air" gets caught — at spec time, before any geometry exists.
4. **Disconnected mate graph → reject.** Every part must have a path to `datum`.
5. **No coordinates anywhere in the schema.** There is no `x`, no `y`, no `rz`. Position is
   *derived* from mates only. The brain is structurally incapable of placing a part wrongly in
   space, because it is incapable of placing a part in space at all.

---

## 7. The connector convention

Every skill returns `(solid, connectors)`. A connector is a named `Joint` with a documented meaning.

```python
def impeller(*, blade_count: int, hub_d: float, tip_d: float, ...) -> SkillResult:
    ...
    return SkillResult(
        solid=part,
        connectors={
            "bore":       RigidJoint("bore", part, Location((0,0,0))),          # shaft seat, z-up
            "nose":       RigidJoint("nose", part, Location((0,0,height))),     # nut face
            "inlet_eye":  RigidJoint("inlet_eye", part, Location((0,0,height), (180,0,0))),
            "tip_shroud": RigidJoint("tip_shroud", part, Location((tip_d/2,0,height*0.6))),
        },
        interfaces={"bore_d": 12.0, "tip_d": tip_d},   # for fit checks in verify.py
    )
```

Rules that make the assembler deterministic:

- **Z is the functional axis** for every rotational skill. No exceptions, no per-skill conventions.
- Connector `Location` carries orientation, not just translation. A mating face points *outward*,
  so `connect_to` produces face-to-face contact without a sign flip anywhere in the spec.
- `interfaces` publishes the numbers `verify.py` needs for fit checks: a `press` fit between a
  12.0 mm shaft and a 12.0 mm bore is flagged as zero-interference and auto-adjusted to the ISO
  H7/p6 band. The brain never computes a tolerance.

---

## 8. Vocabulary inventory

`bd` = wrap bd_warehouse. `ADA` = we author it. Effort is for the skill *plus* its connectors and tests.

| Skill | Source | Effort | Needed for |
|---|---|---|---|
| `iso_thread`, `acme_thread` | bd | 0.5d | every fastener, image 1's tapped bosses |
| `screw`, `nut`, `washer`, `tapped_hole` | bd | 1d | 40+ fasteners in images 2–4 |
| `deep_groove_bearing`, `angular_contact_bearing` | bd | 1d | turbo centre housing |
| `o_ring_gland`, `gasket_groove` | bd | 0.5d | image 1's sealing faces |
| `snap_ring` + `groove`, `key` + `keyway` | bd | 0.5d | shaft retention |
| `spur_gear`, `helical_gear`, `ring_gear` | bd | 1d | gearbox / clutch requests |
| `weld_neck_flange`, `pipe` | bd | 0.5d | inlet/outlet ducts |
| **`revolved_profile`** | ADA | 2d | **the keystone** — housings, hubs, volute sections, shrouds. Named profile curves (ogive, cone, torus-blend, stepped) → `revolve`. Replaces ~60% of what boxes do today. |
| `impeller` | ADA | 2d | image 3's gold compressor wheel |
| `turbine_wheel` | ADA | 1.5d | image 3's hot side |
| `volute_scroll` | ADA | 3d | the hardest part: swept section on a spiral with growing area |
| `diffuser_vanes`, `nozzle_ring` | ADA | 1.5d | image 3's pink/purple vane rings |
| `shaft` (stepped, w/ seats + galleries) | ADA | 1d | every assembly |
| `flange` + `bolt_circle` | ADA | 0.5d | done in spike, 0.14s |
| `rib_web`, `boss`, `fin_array` | ADA | 1d | image 1's cast ribs and bosses |
| `friction_plate`, `spring_pack`, `brake_formed_bracket` | ADA | 2d | image 4's clutch |
| `labyrinth_seal`, `thrust_collar`, `oil_gallery` | ADA | 1.5d | turbo bearing housing |

**~22 skills, ~24 engineer-days.** That is the honest number. It is also the *entire* library — a
turbocharger, a clutch pack, a pump, a drone gimbal and a gas turbine all compose from this set.

---

## 9. Worked example — the turbocharger, 94 parts

What the brain emits (abridged; ~180 lines of JSON in full). Note: no coordinates.

```jsonc
{ "name": "turbocharger_gt2860", "datum": "centre_housing", "units": "mm",
  "parts": [
    {"id":"centre_housing","skill":"revolved_profile","material":"cast_iron",
     "params":{"profile":"stepped_bore","od":78,"length":64,
               "steps":[[34,0],[34,12],[28,12],[28,52],[40,52],[40,64]],
               "oil_gallery":true,"water_jacket":true}},
    {"id":"compressor_housing","skill":"volute_scroll","material":"aluminum",
     "params":{"inlet_d":48,"outlet_d":38,"a_over_r":0.42,"wrap_deg":360,
               "tip_clearance":0.35,"shroud_from":"impeller"}},
    {"id":"turbine_housing","skill":"volute_scroll","material":"stainless_steel",
     "params":{"inlet_d":42,"outlet_d":52,"a_over_r":0.64,"wrap_deg":340}},
    {"id":"impeller","skill":"impeller","material":"aluminum",
     "params":{"blade_count":9,"splitter_count":9,"hub_d":14,"tip_d":56,
               "height":17,"backsweep_deg":42,"blade_t":1.5,"bore_d":9}},
    {"id":"turbine_wheel","skill":"turbine_wheel","material":"inconel",
     "params":{"blade_count":11,"tip_d":53,"hub_d":16,"exducer_angle":32}},
    {"id":"shaft","skill":"shaft","material":"steel",
     "params":{"length":92,"steps":[[9,0],[9,20],[12,20],[12,70],[9,70],[9,92]],
               "thread":"M8x1.25","thread_length":16,"oil_grooves":2}},
    {"id":"journal_bearing","skill":"journal_bearing","count":2,"material":"bronze",
     "params":{"bore":12,"od":20,"width":14,"oil_holes":4}},
    {"id":"thrust_collar","skill":"thrust_collar","material":"steel","params":{"bore":9,"od":24}},
    {"id":"backplate","skill":"flange","material":"aluminum",
     "params":{"od":76,"id":44,"thickness":9,"bolts":12,"bolt":"M6","pcd":62,
               "gasket_groove":"o_ring_2.5"}},
    {"id":"vband_clamp","skill":"retaining_ring","material":"stainless_steel","params":{"d":76}},
    {"id":"housing_bolt","skill":"screw","count":12,"material":"steel",
     "params":{"kind":"SocketHeadCapScrew","size":"M6-1","length":20},
     "pattern":{"type":"polar","count":12,"radius":31,"on":"backplate.bolt_circle"}},
    {"id":"snap_ring","skill":"snap_ring","count":2,"params":{"bore":20,"thickness":1.2}},
    {"id":"oil_seal","skill":"o_ring_gland","count":3,"params":{"id":12,"cs":2.5}}
    // ... nozzle_ring, diffuser_vanes, wastegate_actuator, heat_shield, studs, nuts, washers
  ],
  "mates": [
    {"a":"centre_housing.bore_front","b":"journal_bearing.od","fit":"press"},
    {"a":"journal_bearing.bore","b":"shaft.journal_seat_front","kind":"cylindrical","fit":"clearance"},
    {"a":"shaft.compressor_end","b":"impeller.bore","fit":"press"},
    {"a":"shaft.turbine_end","b":"turbine_wheel.bore","fit":"interference"},
    {"a":"centre_housing.front_face","b":"backplate.centre"},
    {"a":"backplate.bolt_circle","b":"housing_bolt.head_seat"},
    {"a":"backplate.outer_face","b":"compressor_housing.mount_flange"},
    {"a":"centre_housing.rear_face","b":"turbine_housing.mount_flange"},
    {"a":"impeller.nose","b":"shaft_nut.seat","offset_mm":0}
    // ...
  ]
}
```

Every geometric decision above is a **named engineering quantity** — `a_over_r`, `backsweep_deg`,
`exducer_angle`, `pcd`, `H7/p6` via `fit`. The brain is doing turbo design. It is not doing
coordinate geometry. `tip_clearance` and `shroud_from` mean the compressor housing's inner contour
is *derived from the impeller solid* by the kernel — the one place where two skills genuinely couple,
handled by an explicit dependency edge, not by the brain guessing matching numbers.

Estimated build cost, extrapolating spike timings: 94 parts, ~30 unique (rest instanced) →
**~45–70s of kernel time**, one LLM call for the spec, LOD glTF ~18 MB.

---

## 10. Phases and acceptance gates

Each gate is a command that either passes or fails. No phase is "done" on inspection.

### Phase 0 — Unblock (0.5 day)
```bash
./.venv/bin/pip install bd_warehouse trimesh networkx manifold3d pydantic
```
- Fix the two latent bugs: create `assets/cad/standard_parts/catalog.json` (or delete the dead
  `standard_parts` path), and make `exporters.py` fail loudly instead of silently when trimesh is absent.
- Put the preset generators behind `ADA_CAD_PRESETS=0` so v2 keeps working but v3 can't accidentally
  inherit them.
- **Gate:** `python -c "from bd_warehouse.thread import IsoThread; from bd_warehouse.bearing import Bearing; from bd_warehouse.fastener import SocketHeadCapScrew"` succeeds.

### Phase 1 — Schema + registry + first 8 skills (4 days)
`schema.py`, `registry.py`, `frames.py`, and: `revolved_profile`, `shaft`, `flange`, `bolt_circle`,
`impeller`, `deep_groove_bearing`, `screw`, `iso_thread`.
- **Gate:** a **hand-written** turbo spec (no LLM) builds 40+ parts, exports one STEP, reports
  `0 clashes / watertight: all`. Runtime < 60s. This proves the vocabulary before the brain exists.

### Phase 2 — Assembler on real joints (2 days)
`assembler.py`: topological sort from `datum`, `connect_to()` chain, fit resolution against
`interfaces`. Delete `mate_solver.py` and `_deoverlap_transforms()`.
- **Gate:** every part's pose comes from a mate; grep proves no literal coordinate reaches a part
  transform. Move `shaft.length` 92→110 and re-run: impeller, nut and bearings all follow, no spec edit.

### Phase 3 — Brain writes specs (3 days)
`brain.py` with schema-constrained output. Repair loop in `repair.py`: typed failure → `SpecPatch`
touching one part's params → rebuild that node only.
- **Gate:** 20 prompts (turbocharger, clutch pack, planetary gearbox, centrifugal pump, drone gimbal,
  ...) → ≥18 produce a valid, clash-free, connected assembly within 3 repair rounds. Two distinct
  prompts must produce provably different part lists — that is the no-presets test, automated.

### Phase 4 — Verification (2 days)
B-rep clash (spike-proven), watertight per part, connectivity graph via networkx, mass properties,
DFM per material, fit-band checking. Wire into the existing `ValidationReport`.
- **Gate:** deliberately inject 5 faults (bolt in mid-air, interference fit of 0, unreachable part,
  non-manifold blade, wall < 0.8 mm). All 5 caught, each with a `SpecPatch` suggestion.

### Phase 5 — Render modes (3 days)
- **X-ray** (image 3): glTF with per-part `transmission` + `KHR_materials_transmission`; housings
  alpha, internals opaque and emissive. Frontend already has the material plumbing.
- **Technical drawing** (images 1 & 4): server-side `build123d.drafting` + `ExportSVG` — projected
  edges, `section()` cutaway, exploded along the mate graph with leader lines, `DimensionLine` and
  GD&T frames from `interfaces`. Returns SVG, not a viewer toggle.
- **Gate:** side-by-side against the four reference images; edge-rendered SVG has real projected
  silhouette + hidden-line removal, not STL wireframe.

### Phase 6 — Scale to 150 parts (2 days)
Persistent worker pool (kills 94 subprocess spawns), param-hash cache (12 identical M6 bolts build
once), glTF instancing, LOD `linear_deflection` 0.01–0.05 by part size.
- **Gate:** 150-part assembly, cold < 120s, warm < 25s, glTF < 25 MB, viewer > 30 fps.

**Total ≈ 16.5 days** on top of the ~24 days of vocabulary from §8, which overlaps Phases 1–4.
Realistic end-to-end: **5–6 weeks**.

---

## 11. Risks and kill criteria

| Risk | Signal | Response |
|---|---|---|
| `volute_scroll` is genuinely hard (growing-area spiral sweep) | Phase 1 slips past 3 days on this one skill | Ship v1 as a segmented lofted approximation (12 sections), refine later. It looks right in X-ray mode; the A/R number stays honest. |
| Brain writes valid-but-nonsense specs (9-tooth gear on a 200 mm shaft) | Phase 3 gate passes structurally, output is absurd | This is what `design_critics.py` / `intent_validator.py` already exist for — reuse them on the spec, not the geometry. |
| bd_warehouse fasteners are slow (they build real threads) | Phase 6 cold build > 120s | Two-LOD fasteners: cosmetic cylinder for the viewer, real thread only on STEP export. |
| Apple Silicon: build123d ships fine but `manifold3d`/`trimesh` wheels can lag | Phase 0 gate fails | Drop manifold3d — the spike proved OCCT booleans are enough. trimesh is only needed by the legacy exporter, which v3 replaces. |
| Migration stalls with two live CAD stacks | Both `cad_v2` and `cad_v3` accumulating fixes | Hard rule: after Phase 3's gate, `cad_v2` geometry is frozen — bugfixes only, no features. |

**Kill criterion for the whole approach:** if the Phase 1 gate cannot be met — a *hand-written*
spec, no LLM involved, cannot produce a clash-free 40-part turbo — then the problem is the
vocabulary, not the brain, and no amount of Layer 2–4 work will help. The spike says this gate is
reachable; Phase 1 is where that gets confirmed cheaply, before any LLM work is committed.

---

## 12. Immediate next commands

```bash
./.venv/bin/pip install bd_warehouse trimesh networkx pydantic
```

```bash
./.venv/bin/python backend/cad_v3_proto/spike_vocabulary.py
```

Then Phase 1 starts with `backend/cad_v3/schema.py` — the contract first, because every other file
in v3 is downstream of it.

---

### Sources

- [build123d 0.11.1 — Joints](https://build123d.readthedocs.io/en/stable/joints.html)
- [build123d — External Tools and Libraries](https://build123d.readthedocs.io/en/stable/external.html)
- [bd_warehouse — documentation (module list: thread, fastener, bearing, flange, pipe, gear, sprocket, o_rings, retaining_ring, shaft_key, open_builds)](https://bd-warehouse.readthedocs.io/en/latest/)
- [gumyr/bd_warehouse — GitHub](https://github.com/gumyr/bd_warehouse)
- [bd-warehouse — PyPI](https://pypi.org/project/bd-warehouse/)
