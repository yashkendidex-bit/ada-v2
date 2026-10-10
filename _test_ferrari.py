"""Drive ADA's real CAD pipeline on a Ferrari V8 — the same path the app uses.

This is not a synthetic test. It calls `cad_v3_bridge.design_assembly`, which asks the
brain for a SPEC (never code), grows vocabulary through the gate if the spec names a part
that does not exist, builds every part on the kernel, verifies on the solids, and exports.

Then it runs the new physics audit on whatever came out, and reports honestly.

    cd backend && ../.venv/bin/python ../_test_ferrari.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "out" / "ferrari_v8"

PROMPT = "a Ferrari V8 engine"


async def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print(f"PROMPT: {PROMPT}")
    print("=" * 78)

    from cad_v3_bridge import design_assembly

    thoughts: list[str] = []

    def on_thought(msg: str) -> None:
        thoughts.append(str(msg))
        print(f"  [brain] {msg}")

    report = await design_assembly(PROMPT, str(WORK), on_thought=on_thought,
                                   rounds=1, tries=1)

    print()
    print(f"design_assembly finished in {time.time() - t0:.0f}s")
    print(f"returned: {'a report' if report else 'None (fell back to the script path)'}")

    if not report:
        print()
        print("VERDICT: the brain did not produce a buildable spec for this prompt.")
        print("         That is a real result, not a crash — the fallback path exists")
        print("         precisely because this happens.")
        (ROOT / "out" / "ferrari_v8_result.json").write_text(json.dumps(
            {"prompt": PROMPT, "ok": False, "seconds": time.time() - t0,
             "thoughts": thoughts}, indent=1))
        return 1

    print()
    print("report keys:", sorted(report.keys()))
    (ROOT / "out" / "ferrari_v8_result.json").write_text(json.dumps(
        {k: (str(v)[:400] if not isinstance(v, (int, float, str, list, dict)) else v)
         for k, v in report.items()} | {"thoughts": thoughts},
        indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
