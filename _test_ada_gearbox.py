"""End-to-end test: ask ADA for a Ferrari V8 gearbox and see what actually happens.

This runs the REAL pipeline — the model authors a spec, the kernel grows any vocabulary
it is missing, builds the geometry, verifies it and repairs what it can. Nothing here is
stubbed or hand-written, which is the point: the question is whether she performs, and a
test with the hard parts removed cannot answer that.

Then it does the two things that make the result checkable rather than decorative:
physics tests on the real B-rep, and a render from every side.

    .venv/bin/python _test_ada_gearbox.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

OUT = ROOT / "out" / "ferrari_gearbox"
PROMPT = "a Ferrari V8 gearbox"


def log(msg: str = "") -> None:
    print(msg, flush=True)


async def main() -> int:
    from cad_v3 import bootstrap, brain, physics, modes, verify
    from cad_v3.render import blender

    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log("=" * 78)
    log(f"ASKING ADA FOR: {PROMPT!r}")
    log("=" * 78)

    n = bootstrap()
    log(f"vocabulary loaded: {n} skills")

    # ── Layer 1-4: author the spec, grow missing vocabulary, build, verify, repair ──
    thoughts: list[str] = []

    def on_thought(msg: str) -> None:
        thoughts.append(str(msg))
        log(f"  · {msg}")

    t0 = time.time()
    draft, res = await brain.design(
        PROMPT, rounds=3, tries=2, grow=True, sweep=False, log=on_thought)
    design_s = time.time() - t0

    log()
    log(f"design pass: {design_s:.1f}s   model calls: {getattr(draft, 'calls', '?')}")

    if draft.spec is None:
        log("RESULT: she did not produce a spec at all.")
        errors = getattr(draft, "errors", None) or ["(no errors reported)"]
        for e in errors[:8]:
            log(f"   ! {str(e)[:200]}")
        return 1

    spec = draft.spec
    log(f"spec: {len(getattr(spec, 'parts', []) or [])} parts, "
        f"{len(getattr(spec, 'mates', []) or [])} mates")

    if res is None or getattr(res, "asm", None) is None:
        log()
        log("RESULT: a spec was authored but the kernel could not build it.")
        errors = getattr(draft, "errors", None) or []
        for e in errors[:10]:
            log(f"   ! {str(e)[:200]}")
        return 1

    asm = res.asm
    log(f"BUILT: {len(asm.instances)} instances")
    for line in (getattr(asm, "log", []) or [])[-6:]:
        log(f"   · {line}")

    # ── what she actually made ───────────────────────────────────────────────
    keys = [getattr(i, "key", "?") for i in asm.instances]
    log()
    log(f"parts ({len(keys)}): {', '.join(sorted(set(keys))[:24])}"
        f"{' ...' if len(set(keys)) > 24 else ''}")

    # ── MODE 1: CAD evidence ─────────────────────────────────────────────────
    log()
    log("=" * 78)
    log("MODE 1 — CAD (engineering evidence, no scene)")
    log("=" * 78)
    t0 = time.time()
    cad = modes.cad_output("ferrari_v8_gearbox", asm)
    log(cad.summary())
    log(f"  ({time.time() - t0:.1f}s)")

    # ── MODE 2: physics ──────────────────────────────────────────────────────
    log()
    log("=" * 78)
    log("MODE 2 — PHYSICS")
    log("=" * 78)
    t0 = time.time()
    verdicts = modes.physics_output(asm, "interference") + modes.physics_output(asm, "mass")
    log(modes.physics_report(verdicts, "ferrari_v8_gearbox"))
    log(f"  ({time.time() - t0:.1f}s)")

    # ── exports ──────────────────────────────────────────────────────────────
    try:
        from cad_v3.render import exporters

        exporters.export_step_file(asm, OUT / "ferrari_v8_gearbox.step")
        exporters.export_stl_file(asm, OUT / "ferrari_v8_gearbox.stl")
        exporters.export_gltf_file(asm, OUT / "ferrari_v8_gearbox.glb")
        log(f"exported STEP / STL / GLB to {OUT}")
    except Exception as e:
        log(f"export failed: {type(e).__name__}: {e}")

    # ── MODE 3: render from every side ───────────────────────────────────────
    log()
    log("=" * 78)
    log("MODE 3 — SCENE (every side)")
    log("=" * 78)
    if not blender.available():
        log("Blender unavailable — skipping renders.")
        return 0

    shots = ["front", "back", "left", "right", "top", "bottom", "iso", "three_quarter"]
    results = []
    for shot in shots:
        t0 = time.time()
        r = blender.render_assembly(
            asm, style="studio", shot=shot, samples=40, resolution=(1200, 800),
            out_path=str(OUT / f"view_{shot}.png"))
        results.append((shot, r))
        mark = "ok " if r.ok else "FAIL"
        log(f"  [{mark}] {shot:14s} {time.time() - t0:5.0f}s  "
            f"{r.path.split('/')[-1] if r.ok else r.message}")

    ok = sum(1 for _, r in results if r.ok)
    log()
    log(f"rendered {ok}/{len(shots)} views")

    # ── verdict ──────────────────────────────────────────────────────────────
    log()
    log("=" * 78)
    log("HOW SHE PERFORMED")
    log("=" * 78)
    overall = physics.overall(verdicts)
    log(f"  spec authored      : yes ({len(spec.parts)} parts, {len(spec.mates)} mates)")
    log(f"  model calls        : {getattr(draft, 'calls', '?')}")
    log(f"  built              : {len(asm.instances)} instances in {design_s:.0f}s")
    log(f"  physics            : {overall}")
    log(f"  views rendered     : {ok}/{len(shots)}")
    log(f"  total wall time    : {time.time() - t_start:.0f}s")

    summary = {
        "prompt": PROMPT,
        "built": True,
        "parts": len(spec.parts),
        "mates": len(spec.mates),
        "instances": len(asm.instances),
        "model_calls": getattr(draft, "calls", None),
        "design_seconds": round(design_s, 1),
        "physics": overall,
        "verdicts": [{"test": v.test, "status": v.status, "summary": v.summary,
                      "metrics": v.metrics} for v in verdicts],
        "views": [s for s, r in results if r.ok],
        "thoughts": thoughts[-40:],
    }
    (OUT / "run_report.json").write_text(json.dumps(summary, indent=2))
    log(f"report -> {OUT / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
