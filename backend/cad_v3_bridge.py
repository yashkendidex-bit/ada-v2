"""The seam between ADA's live CAD entrypoint and the cad_v3 assembly engine.

`cad_agent.generate_prototype` used to be the whole of ADA's CAD ability: ask a model
for a build123d script, run it, keep whatever came out. That path can only ever produce
what a language model can hold in its head in one go — which in practice is one solid.

This module puts Layers 1-5 in front of it instead. The model is asked for a *spec*
(parts, materials, mates), never for code; any part ADA does not have a word for is
synthesised and refereed by the geometry gate; the kernel places every part by connector
frame; `verify` measures the result; and only then is anything exported.

The exports deliberately keep the old filenames — `output_0.stl`, `output_1.stl`, …
and `output.step` in the caller's work dir — so `cad_agent._collect_outputs` and
`_build_cad_payload` are reused verbatim and the viewer, the exploded view and the BOM
all keep working. One STL per *spec part*, not per instance: six identical nuts are one
mesh, which is what the viewer means by a part and keeps the payload small.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable

# One STL per part. Past the cap the tail is merged into the last mesh so a 60-part
# machine cannot blow up a websocket payload — but the cap is deliberately generous
# now and overridable, because the old hard 28 silently folded every part past the
# 28th into a blob labelled "N smaller parts". That is a complexity limit on what the
# machine may be, and it showed up as a viewer missing parts rather than as an error.
def _mesh_cap() -> int:
    try:
        from cad_v3.team import mesh_cap
        return mesh_cap()
    except Exception:
        return 96


STL_TOL = 0.12          # mm of chord error; coarser than the CLI's 0.06 to keep files small


def _log(on_thought: Callable[[str], None] | None):
    def emit(msg: str) -> None:
        print(f"[CAD_V3] {msg}")
        if on_thought:
            try:
                on_thought(f"[engine] {msg}")
            except Exception:
                pass
    return emit


def _clear(work_dir: str) -> None:
    for old in glob.glob(os.path.join(work_dir, "output*.stl")) + \
            glob.glob(os.path.join(work_dir, "output*.step")):
        try:
            os.remove(old)
        except OSError:
            pass


def _part_meshes(asm, work_dir: str, emit) -> list[str]:
    """One STL per spec part, in spec order. Returns the paths written."""
    from build123d import Compound, export_stl

    groups: list[tuple[str, list]] = []
    for part in asm.spec.parts:
        solids = [i.solid for i in asm.of_part(part.id) if i.solid is not None]
        if solids:
            groups.append((part.label or part.id, solids))
    cap = _mesh_cap()
    if len(groups) > cap:
        head, tail = groups[:cap - 1], groups[cap - 1:]
        merged = [s for _, solids in tail for s in solids]
        groups = head + [(f"{len(tail)} smaller parts", merged)]

    written: list[str] = []
    for i, (label, solids) in enumerate(groups):
        path = os.path.join(work_dir, f"output_{i}.stl")
        try:
            shape = solids[0] if len(solids) == 1 else Compound(children=list(solids))
            export_stl(shape, path, tolerance=STL_TOL, angular_tolerance=0.3)
            if os.path.exists(path):
                written.append(path)
        except Exception as exc:
            emit(f"part '{label}' would not tessellate: {type(exc).__name__}: {exc}")
    return written


async def design_assembly(prompt: str, work_dir: str, *, on_thought=None,
                          rounds: int = 3, tries: int = 2, categories=None,
                          ask=None, chat=None, clarify: bool = True,
                          plan: bool = True, refine_rounds: int = 1) -> dict | None:
    """Clarify, plan, author, grow, build, verify, inspect, refine, export.

    Returns a report dict on success, or None so the caller can fall back to the old
    script-writing path. The heavy lifting is `cad_v3.team.run` — the clarifier and
    planner in front, the placeholder refiner behind — which itself delegates the
    design loop to `brain.design` unchanged. Nothing here writes a spec or a skill by
    hand: the spec comes from the model, the missing vocabulary from the model through
    the geometry gate, and the placement from the kernel.

    `ask` is the clarifier's channel to the operator (async, returns the answer text).
    With no `ask` the questions are still detected and logged, just not asked.
    """
    from cad_v3 import team, verify

    emit = _log(on_thought)
    os.makedirs(work_dir, exist_ok=True)
    _clear(work_dir)

    # `emit` already routes every line to `on_thought` with the engine prefix, so the
    # team is given the log callback only — passing both would duplicate every line.
    tr = await team.run(prompt, ask=ask, chat=chat, categories=categories,
                        rounds=rounds, tries=tries, clarify=clarify, plan=plan,
                        refine_rounds=refine_rounds, log=emit)
    draft, res = tr.draft, tr.result

    if tr.spec is None or tr.asm is None:
        errors = getattr(draft, "errors", None) or []
        why = "; ".join(str(e)[:160] for e in errors) or "no spec returned"
        emit(f"design did not produce a buildable spec ({why})")
        return None

    asm, spec = tr.asm, tr.spec
    rep = tr.report if tr.report is not None else verify.verify(asm)
    stls = _part_meshes(asm, work_dir, emit)
    if not stls:
        emit("assembly built but nothing tessellated; falling back")
        return None

    step = os.path.join(work_dir, "output.step")
    try:
        from cad_v3.render import export_step_file

        export_step_file(asm, step)
    except Exception as exc:
        emit(f"STEP export skipped: {type(exc).__name__}: {exc}")

    # Layer 5 was built and never called from here, so every live run shipped meshes
    # with no picture of them and whoever looked at the result had to point some other
    # renderer at the STLs. `export_shaded` owns its own rasteriser — z-buffer, vertex
    # normals, two lights, outlines from depth discontinuity — which is the difference
    # between a CAD viewport and a triangle plot. It runs on the assembly already in
    # memory, so the only cost is tessellation.
    shots: list[str] = []
    try:
        from cad_v3.render import export_shaded

        pics = export_shaded(asm, work_dir, views=("iso", "front"),
                             modes=("solid",))
        shots = list(pics.files)
        emit(f"shaded: {len(shots)} view(s), {pics.triangles} triangles, "
             f"{pics.seconds:.1f}s")
    except Exception as exc:
        emit(f"shaded render skipped: {type(exc).__name__}: {exc}")

    grown = list(tr.grown)
    stubs = list(tr.stubbed)
    # Engineering evidence on EVERY build (~1s on the in-memory assembly, no
    # rebuild): interference audit + mass properties via the Mode-1 builder.
    # This is what converts "a picture of a model" into an artifact with a
    # verdict. Evidence is a bonus — it must never fail the build. `team.run`
    # already measured it; this only falls back if that measurement was unavailable.
    evidence: dict = tr.evidence or {"status": "UNKNOWN", "parts": len(spec.parts),
                                     "verdicts": [], "note": ""}
    if evidence.get("verdicts"):
        emit(f"evidence: {evidence.get('status')} — "
             + "; ".join(f"{v['test']}: {v['status']}" for v in evidence["verdicts"]))
    try:
        import json as _json

        (Path(work_dir) / "evidence.json").write_text(
            _json.dumps(evidence, indent=1), encoding="utf-8")
    except Exception:
        pass
    report = {
        "engine": "cad_v3",
        "spec_name": spec.name,
        "parts": len(spec.parts),
        "instances": spec.total_instances,
        "mates": len(spec.mates),
        "model_calls": tr.calls,
        "skills_written": grown,
        "placeholders": stubs,
        "repaired": bool(getattr(res, "fixed", False)),
        "rounds": int(getattr(res, "rounds", 0) or 0),
        "verify": rep.text() if rep is not None else "",
        "passed": bool(rep.passed) if rep is not None else False,
        "evidence": evidence,
        "spec": spec.to_dict(),
        "stls": stls,
        "steps": [step] if os.path.exists(step) else [],
        "shots": shots,
        # Level 3, for the HUD and the completion line: what was asked, what was
        # planned, what the inspector found, and what the refiner did about it.
        "clarifications": [{"key": q.key, "question": q.text, "why": q.why,
                            "answer": tr.brief.answers.get(q.key), "choices": list(q.choices),
                            "default": q.default} for q in tr.brief.questions],
        "plan": ({"function": tr.plan.function, "envelope_mm": list(tr.plan.envelope_mm),
                  "parts": list(tr.plan.parts), "materials": dict(tr.plan.materials),
                  "interfaces": list(tr.plan.interfaces), "checks": list(tr.plan.checks),
                  "source": tr.plan.source} if tr.plan is not None else None),
        "findings": [{"stage": f.stage, "subject": f.subject, "ok": f.ok,
                      "detail": f.detail, "action": f.action} for f in tr.findings],
        "refinements": [{"round": r.round, "action": r.action, "before": r.before,
                         "after": r.after, "improved": r.improved, "detail": r.detail}
                        for r in tr.refinements],
        "team_log": list(tr.loglines),
    }
    emit(f"{spec.name}: {len(spec.parts)} parts -> {spec.total_instances} instances, "
         f"{len(spec.mates)} mates, {tr.calls} model call(s), "
         f"{'PASS' if report['passed'] else 'OPEN'}")
    prove = [f for f in tr.findings if not f.ok]
    if prove:
        emit(f"inspector: {len(prove)} open finding(s) — "
             + "; ".join(f.subject for f in prove[:6]))
    if grown:
        emit(f"new vocabulary written by ADA: {', '.join(grown)}")
    if stubs:
        emit(f"PLACEHOLDERS (declared, not silent): {', '.join(stubs)}")

    spec_path = Path(work_dir) / "design_spec.json"
    try:
        import json

        spec_path.write_text(json.dumps(spec.to_dict(), indent=1), encoding="utf-8")
        report["spec_path"] = str(spec_path)
    except Exception:
        pass
    try:
        # The measured truth about these meshes, next to the meshes. `_build_cad_payload`
        # copies both sidecars into the timestamped export dir.
        (Path(work_dir) / "verify.txt").write_text(
            f"{spec.name}\n{'PASS' if report['passed'] else 'OPEN'} after "
            f"{report['rounds']} repair round(s)\n\n" + (report["verify"] or ""),
            encoding="utf-8")
    except Exception:
        pass
    return report
