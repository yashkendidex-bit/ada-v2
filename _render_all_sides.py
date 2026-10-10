"""Render the current model from every side, using the new scene backend.

This is Mode 3 doing its job: an explicitly requested presentation pass, after a model
already exists. It touches no geometry and validates nothing — that is CAD mode's and
physics mode's work, and mixing them is what the owner objected to.

    cd backend && ../.venv/bin/python ../_render_all_sides.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "sides"

SHOTS = [
    ("three_quarter", 64),
    ("front", 56),
    ("top", 56),
    ("hero", 56),
    ("macro", 64),
]


def main() -> int:
    import cad_v3
    cad_v3.bootstrap()

    from cad_v3 import assembler
    from cad_v3.cli import _load
    from cad_v3.render import blender

    if not blender.available():
        print("Blender not available")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    spec = _load(str(ROOT / "backend" / "cad_v3" / "specs" / "turbocharger.json"))
    print(f"building {spec.name} ...")
    t0 = time.time()
    asm = assembler.build(spec)
    print(f"  {len(asm.instances)} parts in {time.time() - t0:.0f}s")
    print()

    for shot, samples in SHOTS:
        t = time.time()
        path = OUT / f"{shot}.png"
        r = blender.render_assembly(asm, style="studio", shot=shot, samples=samples,
                                    resolution=(1400, 900), out_path=str(path))
        status = "ok " if r.ok else "FAIL"
        print(f"  {status} {shot:14s} {time.time() - t:5.0f}s  "
              f"{r.path if r.ok else r.message}")
        if not r.ok and r.log_tail:
            print("      " + r.log_tail[-300:].replace("\n", "\n      "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
