"""Command line for the assembly engine.

    python -m cad_v3 vocab
    python -m cad_v3 sig impeller
    python -m cad_v3 check backend/cad_v3/specs/turbocharger.json
    python -m cad_v3 synth backend/cad_v3/specs/gearbox_5speed.json --grow
    python -m cad_v3 build backend/cad_v3/specs/turbocharger.json --out out --draw front,iso
    python -m cad_v3 build --out out --shaded iso --shaded-modes solid,xray,exploded
    python -m cad_v3 build a_new_machine.json --stubs --out out
    python -m cad_v3 repair backend/cad_v3/specs/turbocharger.json --spec-out fixed.json
    python -m cad_v3 design "a variable-geometry turbocharger for a 2.0 L diesel" --out out

Only `design` and `--grow`/`synth` involve an LLM, and even they only ever produce a
spec file and vocabulary the rest of these commands consume; `repair` closes the
geometry loop with no model at all, and `--stubs` grows a spec's missing parts into
dimensioned placeholders without a single request.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import assembler, bootstrap, registry, validate, verify
from .schema import Spec

SPEC_DIR = Path(__file__).parent / "specs"


def _load(path: str) -> Spec:
    text = Path(path).read_text(encoding="utf-8")
    return Spec.from_json(text)


def cmd_vocab(args) -> int:
    bootstrap()
    if args.json:
        print(json.dumps(registry.vocabulary_digest(args.category), indent=2))
        return 0
    by_cat: dict[str, list[str]] = {}
    for name in registry.names():
        sk = registry.get(name)
        if args.category and sk.category not in args.category:
            continue
        by_cat.setdefault(sk.category, []).append(name)
    grown = 0
    for cat in sorted(by_cat):
        print(f"\n{cat}")
        for name in by_cat[cat]:
            sk = registry.get(name)
            doc = (sk.doc or "").strip().splitlines()[0] if sk.doc else ""
            tag = "" if sk.source == "ada" else f"[{sk.source}] "
            grown += sk.source != "ada"
            print(f"  {name:18s} {tag}{doc[:64]}")
            print(f"  {'':18s} connectors: {', '.join(sk.connectors)}")
    print(f"\n{len(registry.names())} skills"
          + (f", {grown} of them written by ADA herself" if grown else ""))
    return 0


def cmd_sig(args) -> int:
    bootstrap()
    if registry.get(args.skill) is None:
        print(f"unknown skill '{args.skill}'; known: {', '.join(registry.names())}")
        return 2
    print(registry.signature_text(args.skill))
    return 0


def _work_orders(spec: Spec) -> list:
    """What this spec needs that the vocabulary does not have yet."""
    from . import synth

    orders = synth.work_orders(spec)
    if orders:
        print(f"\nvocabulary gap: {len(orders)} skill(s) this spec names and ADA "
              "does not have yet")
        for o in orders:
            print("  " + o.text().replace("\n", "\n  "))
    return orders


def _grow(spec: Spec, args, orders: list | None = None) -> None:
    """Fulfil the spec's work orders before the kernel ever sees it."""
    from . import synth

    orders = _work_orders(spec) if orders is None else orders
    if not orders:
        return
    if getattr(args, "stubs", False):
        res = synth.stub_all(orders)
    elif getattr(args, "grow", False):
        res = synth.fulfil_sync(orders, sweep=getattr(args, "sweep", False),
                                log=lambda m: print(f"  {m}"))
    else:
        print("  --grow writes them with the brain (1 request for all of them); "
              "--stubs holds their space with placeholder envelopes, no request")
        return
    print(res.text())


def cmd_check(args) -> int:
    bootstrap()
    spec = _load(args.spec)
    gaps = validate.unknown_skills(spec)
    # With a gap in the vocabulary, the strict pass reports knock-on errors — a mate
    # onto a seat of a part nobody has written yet cannot be judged. Report only the
    # errors that survive the gap, then the gap itself, as work orders.
    errs = validate.validate(spec, allow_unknown=bool(gaps))
    print(validate.summary(errs)
          + (" apart from the vocabulary gap below" if gaps and not errs else ""))
    print(f"\n{len(spec.parts)} parts -> {spec.total_instances} instances, "
          f"{len(spec.mates)} mates, datum '{spec.datum}'")
    if gaps:
        _work_orders(spec)
        print("\nthese are work orders, not mistakes: `build --grow` writes them with "
              "the brain, `build --stubs` holds their space with no request at all.")
    return 0 if not errs else 1


def cmd_gate(args) -> int:
    """Layer 1's referee: put skills through the geometry gate, no model involved."""
    from . import gate

    bootstrap()
    names = [args.skill] if args.skill else None
    reps = gate.run_all(names, sweep=not args.fast)
    print(gate.table(reps))
    return 0 if all(r.passed for r in reps) else 1


def cmd_build(args) -> int:
    t0 = time.time()
    n = bootstrap()
    spec = _load(args.spec)
    print(f"vocabulary: {n} skills")
    print(f"spec:       {spec.name} — {len(spec.parts)} parts, "
          f"{spec.total_instances} instances, {len(spec.mates)} mates")
    _grow(spec, args)

    asm = assembler.build(spec, strict=not args.lenient)
    for line in asm.log:
        print(f"build:      {line}")
    if asm.errors:
        print(f"\n{len(asm.errors)} error(s):")
        for e in asm.errors[:20]:
            print(f"  {e}")
        if not args.lenient:
            return 1

    if args.dump:
        print("\nplacement:")
        for inst in asm.instances:
            try:
                bb = inst.solid.bounding_box()
                box = (f"x[{bb.min.X:7.1f},{bb.max.X:7.1f}] "
                       f"y[{bb.min.Y:7.1f},{bb.max.Y:7.1f}] z[{bb.min.Z:7.1f},{bb.max.Z:7.1f}]")
            except Exception:
                box = "no bbox"
            print(f"  {inst.key:22s} {box}  via {inst.via}")

    rep = verify.verify(asm, clash=not args.no_clash)
    print("\nverify:")
    print(rep.text())

    if args.out:
        _export(asm, spec, args)

    print(f"\ntotal:      {time.time() - t0:.2f}s   "
          f"{'PASS' if rep.passed and not asm.errors else 'FAIL'}")
    return 0 if rep.passed and not asm.errors else 1


def _export(asm, spec: Spec, args) -> None:
    """Layer 5 for whichever subcommand produced the assembly.

    `repair` and `design` end holding a built, verified assembly, so they export from
    that object rather than re-running the build on the written spec — the solids are
    already in memory and rebuilding them would double the cost of the command.
    """
    from .render import export_all, export_lod, export_shaded, technical_svg

    exp = export_all(asm, args.out, step=not args.no_step, gltf=not args.no_gltf,
                     xray=not args.no_xray, exploded=args.explode, stl=args.stl,
                     linear=args.deflection)
    print("\nexport:")
    print(exp.text())
    levels = [v.strip() for v in (args.lod or "").split(",") if v.strip()]
    if levels:
        lod = export_lod(asm, args.out, levels=levels)
        print("\nlod:")
        print(lod.text())
    shots = [v.strip() for v in (args.shaded or "").split(",") if v.strip()]
    if shots:
        modes = [m.strip() for m in (args.shaded_modes or "solid").split(",") if m.strip()]
        pics = export_shaded(asm, args.out, views=shots, modes=modes,
                             explode=args.shaded_explode)
        print("\nshaded:")
        print(pics.text())
    for view in [v for v in (args.draw or "").split(",") if v.strip()]:
        stem = spec.name.replace(" ", "_").lower()
        out = Path(args.out) / f"{stem}_{view.strip()}.svg"
        try:
            technical_svg(asm, out, view=view.strip(), labels=args.labels,
                          cutaway=args.cutaway, keep_whole=set(args.keep_whole.split(","))
                          if args.keep_whole else None)
            print(f"  {out.name:34s} {out.stat().st_size / 1024.0:9.1f} KB")
        except Exception as exc:
            print(f"  {view}: drawing failed: {type(exc).__name__}: {exc}")


def _render_args(p, *, draw: str = "") -> None:
    """The Layer 5 options, shared by `build`, `repair` and `design`."""
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--draw", default=draw, help="comma separated views, e.g. front,iso")
    p.add_argument("--labels", action="store_true", help="exploded-style callouts")
    p.add_argument("--cutaway", action="store_true", help="section the drawing")
    p.add_argument("--keep-whole", default="", help="skills left uncut in a cutaway")
    p.add_argument("--explode", type=float, default=0.0, help="exploded glTF factor")
    p.add_argument("--lod", default="", help="instanced LOD glTFs, e.g. high,mid,low")
    p.add_argument("--shaded", default="", help="shaded PNG views, e.g. iso,front")
    p.add_argument("--shaded-modes", default="solid",
                   help="comma separated: solid, xray, exploded")
    p.add_argument("--shaded-explode", type=float, default=0.6,
                   help="separation factor for the exploded PNG")
    p.add_argument("--deflection", type=float, default=0.06)
    p.add_argument("--no-clash", action="store_true")
    p.add_argument("--no-step", action="store_true")
    p.add_argument("--no-gltf", action="store_true")
    p.add_argument("--no-xray", action="store_true")
    p.add_argument("--stl", action="store_true")


def _write_spec(spec: Spec, path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec.to_dict(), indent=1), encoding="utf-8")
    print(f"spec out:   {p}  ({len(spec.parts)} parts, {len(spec.mates)} mates)")


def cmd_repair(args) -> int:
    """Layer 4 on its own: no model, just the kernel and the deterministic rules."""
    from . import repair as repair_mod

    t0 = time.time()
    bootstrap()
    spec = _load(args.spec)
    print(f"spec:       {spec.name} — {len(spec.parts)} parts, {len(spec.mates)} mates")
    _grow(spec, args)
    res = repair_mod.repair(spec, rounds=args.rounds, clash=not args.no_clash,
                            log=(lambda s: print(f"  {s}")) if args.verbose else None)
    print("\n" + res.text())
    if res.report is not None:
        print("\nverify:\n" + res.report.text())
    _write_spec(res.spec, args.spec_out)
    if args.out and res.asm is not None:
        _export(res.asm, res.spec, args)
    print(f"\ntotal:      {time.time() - t0:.2f}s   "
          f"{'REPAIRED' if res.fixed else 'OPEN'} after {res.rounds} round(s)")
    return 0 if res.fixed else 1


def cmd_design(args) -> int:
    """Layers 1-4: the brain writes the spec, ADA writes any word she is missing,
    the kernel referees both.

    This is the only subcommand that talks to a model, and it is deliberately the
    thinnest one — everything after `design_sync` returns is the same deterministic
    path `build` takes, so a designed spec and a hand-written one are
    indistinguishable downstream.
    """
    from . import brain

    t0 = time.time()
    n = bootstrap()
    print(f"vocabulary: {n} skills")
    intent = sys.stdin.read() if args.intent == "-" else args.intent
    draft, res = brain.design_sync(intent, rounds=args.rounds, clash=not args.no_clash,
                                   tries=args.tries, categories=args.category,
                                   grow=not args.no_grow, sweep=args.sweep,
                                   log=lambda s: print(f"  {s}"))
    print(f"\nauthored in {draft.calls} model call(s)")
    if draft.synth is not None:
        print(draft.synth.text())
    if draft.spec is None or draft.errors:
        # `errors` holds validator objects when the spec parsed and plain strings when
        # the transport or the JSON did not, so both have to print.
        for err in draft.errors:
            print(f"  {err}" if isinstance(err, str) else validate.summary([err]))
        if not draft.errors:
            print("  no spec returned")
        if args.raw and draft.raw:
            print("\n--- raw reply ---\n" + draft.raw[:4000])
        return 2
    print(f"spec:       {res.spec.name} — {len(res.spec.parts)} parts, "
          f"{len(res.spec.mates)} mates, datum '{res.spec.datum}'")
    print("\n" + res.text())
    if res.report is not None:
        print("\nverify:\n" + res.report.text())
    _write_spec(res.spec, args.spec_out)
    if args.out and res.asm is not None:
        _export(res.asm, res.spec, args)
    print(f"\ntotal:      {time.time() - t0:.2f}s   "
          f"{'REPAIRED' if res.fixed else 'OPEN'} after {res.rounds} round(s)")
    return 0 if res.fixed else 1


def _add_growth_flags(sp) -> None:
    """Every command that feeds the kernel can grow the vocabulary on the way in."""
    sp.add_argument("--grow", action="store_true",
                    help="write any missing skill with the brain — ONE request for all of them")
    sp.add_argument("--stubs", action="store_true",
                    help="hold each missing skill's space with a placeholder envelope, no request")
    sp.add_argument("--sweep", action="store_true",
                    help="also sweep a synthesised skill's parameter range in the gate (slower)")


def cmd_synth(args) -> int:
    """Layer 1 growing itself, on its own: no assembly, just the missing words."""
    bootstrap()
    spec = _load(args.spec)
    orders = _work_orders(spec)
    if not orders:
        print("nothing to synthesise: every skill this spec names already exists")
        return 0
    if not (args.grow or args.stubs):
        print("\n  --grow writes them with the brain (1 request for all of them); "
              "--stubs holds their space with placeholder envelopes, no request")
        return 1
    _grow(spec, args, orders)
    from . import registry as reg

    print("\nregistry now:")
    real = 0
    for o in orders:
        sk = reg.get(o.skill)
        real += bool(sk) and sk.source == "synth"
        print(f"  {o.skill:22s} {(sk.source if sk else 'MISSING'):10s} "
              f"connectors: {', '.join(sk.connectors) if sk else '-'}")
    print(f"\n{real} of {len(orders)} are real skills; the rest are placeholders.")
    return 0 if real == len(orders) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cad_v3", description="ADA assembly engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("vocab", help="list the closed vocabulary")
    v.add_argument("--category", nargs="*", default=None)
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_vocab)

    s = sub.add_parser("sig", help="show one skill's parameters")
    s.add_argument("skill")
    s.set_defaults(fn=cmd_sig)

    c = sub.add_parser("check", help="validate a spec without building")
    c.add_argument("spec")
    c.set_defaults(fn=cmd_check)

    g = sub.add_parser("gate", help="run the vocabulary through the geometry gate")
    g.add_argument("skill", nargs="?", default=None, help="one skill, or all of them")
    g.add_argument("--fast", action="store_true", help="skip the parameter-range sweep")
    g.set_defaults(fn=cmd_gate)

    b = sub.add_parser("build", help="build, verify and export a spec")
    b.add_argument("spec", nargs="?", default=str(SPEC_DIR / "turbocharger.json"))
    _render_args(b)
    _add_growth_flags(b)
    b.add_argument("--lenient", action="store_true", help="build past spec errors")
    b.add_argument("--dump", action="store_true", help="print every placed bounding box")
    b.set_defaults(fn=cmd_build)

    y = sub.add_parser("synth", help="write the skills a spec needs and ADA does not have")
    y.add_argument("spec")
    _add_growth_flags(y)
    y.set_defaults(fn=cmd_synth)

    r = sub.add_parser("repair", help="close the loop on a spec: build, verify, patch, rebuild")
    r.add_argument("spec", nargs="?", default=str(SPEC_DIR / "turbocharger.json"))
    _render_args(r)
    _add_growth_flags(r)
    r.add_argument("--rounds", type=int, default=4, help="maximum repair rounds")
    r.add_argument("--spec-out", default=None, help="write the repaired spec here")
    r.add_argument("--verbose", action="store_true", help="log every round")
    r.set_defaults(fn=cmd_repair)

    d = sub.add_parser("design", help="write a spec from an intent, then repair it")
    d.add_argument("intent", help="what to design, or '-' to read stdin")
    _render_args(d)
    d.add_argument("--rounds", type=int, default=4, help="maximum repair rounds")
    d.add_argument("--tries", type=int, default=2, help="authoring attempts before giving up")
    d.add_argument("--category", nargs="*", default=None,
                   help="restrict the catalogue shown to the model")
    d.add_argument("--spec-out", default=None, help="write the designed spec here")
    d.add_argument("--raw", action="store_true", help="print the reply when authoring fails")
    d.add_argument("--no-grow", action="store_true",
                   help="refuse parts outside the vocabulary instead of writing them")
    d.add_argument("--sweep", action="store_true",
                   help="also sweep a synthesised skill's parameter range in the gate")
    d.set_defaults(fn=cmd_design)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
