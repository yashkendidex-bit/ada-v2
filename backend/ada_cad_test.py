"""Hands-off capability test: ask ADA for whole machines and report what came back.

Nothing in here describes a machine. Each prompt is one line of English; every part,
dimension, material and mate in the result is ADA's own — authored by her CAD chain as a
spec, grown through the geometry gate when she lacks a word for a part, placed by the
kernel, measured by `verify`.

    python ada_cad_test.py                    # the two default prompts
    python ada_cad_test.py "a bench vice" "a quadcopter arm"

Run it from `backend/`. It goes through CadAgent.generate_prototype — the exact call
ADA's voice loop and the Electron app make — so whatever it prints is what the product
does, not a bench rig.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROMPTS = [
    "an Iron Man repulsor",
    "a helicopter main rotor head with a swashplate",
]


def thought(msg):
    print(f"    {msg}", flush=True)


def status(info):
    if info.get("error"):
        print(f"    [{info.get('status')}] {str(info['error'])[:110]}", flush=True)


def show(prompt, payload, seconds):
    print()
    print("-" * 78)
    if not payload:
        print(f"RESULT for {prompt!r}: nothing produced in {seconds:.1f}s")
        return False
    d = payload.get("design") or {}
    engine = payload.get("engine", "legacy script path")
    print(f"RESULT for {prompt!r}   [{engine}]   {seconds:.1f}s")
    if d:
        print(f"  ADA named it     : {d.get('spec_name')}")
        print(f"  parts            : {d.get('parts')} distinct -> "
              f"{d.get('instances')} instances, {d.get('mates')} mates")
        print(f"  model calls      : {d.get('model_calls')}  "
              f"(one spec + one batch for any new vocabulary)")
        print(f"  skills she wrote : {', '.join(d.get('skills_written') or []) or 'none needed'}")
        print(f"  placeholders     : {', '.join(d.get('placeholders') or []) or 'none — all real geometry'}")
        print(f"  repair rounds    : {d.get('rounds')}  -> {'CLEAN' if d.get('repaired') else 'OPEN'}")
        print(f"  verify           : {'PASS' if d.get('passed') else 'FAIL'}")
        for line in (d.get("verify") or "").splitlines():
            print(f"      {line}")
    else:
        print("  (legacy path: a single script-written solid, no assembly structure)")
    meshes = payload.get("parts") or []
    print(f"  meshes for viewer: {len(meshes)}")
    bom = payload.get("bom") or []
    if bom:
        print(f"  BOM rows         : {len(bom)}")
    exports = payload.get("exports") or {}
    if exports.get("dir"):
        print(f"  exported to      : {exports['dir']}")
    return bool(d and d.get("passed"))


async def main(prompts):
    from cad_agent import CadAgent

    agent = CadAgent(on_thought=thought, on_status=status)
    print(f"engine: {os.getenv('ADA_CAD_ENGINE', 'cad_v3')}")
    from providers import cad_chain

    print("brain : " + " -> ".join(f"{p}/{m}" for p, m in cad_chain()))
    verdicts = []
    for prompt in prompts:
        print("\n" + "=" * 78)
        print(f"ASKING ADA: {prompt}")
        print("=" * 78, flush=True)
        t0 = time.time()
        try:
            payload = await agent.generate_prototype(prompt)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            payload = None
            print(f"  raised {type(exc).__name__}: {exc}")
        verdicts.append(show(prompt, payload, time.time() - t0))
    print()
    print("=" * 78)
    for prompt, ok in zip(prompts, verdicts):
        print(f"  {'PASS' if ok else 'no verified assembly':22s} {prompt}")
    print("=" * 78)
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    args = sys.argv[1:] or DEFAULT_PROMPTS
    sys.exit(asyncio.run(main(args)))
