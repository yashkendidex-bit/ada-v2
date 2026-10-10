"""Scorecard — KAREN's performance assessment, weekly and honest.

"5 prints, 4 clean, 1 warp — all warps on large flat bases. Brim fixes it."

Reads fabricator job outcomes + world deviations, groups failures by
keyword (warp, adhesion, stringing, layer, support, tolerance), and
reports counts + the one pattern that repeats + one fix to try. Patterns
come from YOUR history, never from generic advice. No jobs yet → says
so instead of inventing a record.
"""
from __future__ import annotations

import json
from pathlib import Path


FAILURE_KEYS = {
    "warp": ("warp", "curl", "lift", "corner"),
    "adhesion": ("adhesion", "bed", "first layer", "brim", "raft"),
    "stringing": ("string", "ooze", "retract"),
    "layer": ("layer shift", "layer line", "delamin", "split"),
    "support": ("support", "overhang", "droop"),
    "tolerance": ("tolerance", "fit", "press", "loose", "tight"),
    "under-extrusion": ("under-extru", "clog", "gap", "starv"),
}


def build(root_dir: str | Path, state_dir: str | Path = "") -> str:
    jobs_dir = Path(root_dir) / "projects" / "fabricator"
    try:
        jobs = [json.loads(p.read_text()) for p in sorted(jobs_dir.glob("*.json"))]
    except Exception:
        jobs = []
    outcomes = [j.get("outcome") or {} for j in jobs]
    outcomes = [o for o in outcomes if o.get("result")]
    if not outcomes:
        return ("No finished prints on record yet, Sir — run the loop once "
                "and I'll start keeping score.")
    clean = sum(1 for o in outcomes if o.get("result") == "clean")
    total = len(outcomes)
    # group failures by keyword family
    groups: dict[str, list[str]] = {}
    for o in outcomes:
        if o.get("result") == "clean":
            continue
        note = ((o.get("result") or "") + " " + (o.get("note") or "")).lower()
        fam = "other"
        for name, keys in FAILURE_KEYS.items():
            if any(k in note for k in keys):
                fam = name
                break
        groups.setdefault(fam, []).append(o.get("note", "")[:100])
    head = (f"{total} prints on record, {clean} clean"
            + (f", {total - clean} with lessons." if total - clean else " — flawless record, Sir."))
    if not groups:
        return head
    top = max(groups, key=lambda k: len(groups[k]))
    fixes = {
        "warp": "Brim + hotter bed on large flat bases fixes most warps.",
        "adhesion": "Slow the first layer and re-level — adhesion is almost always mechanical.",
        "stringing": "Retraction +2mm and 5 degrees cooler usually dries it up.",
        "layer": "Check belt tension and print speed before blaming the model.",
        "support": "Rotate the part or add supports where the pre-flight flagged.",
        "tolerance": "Print a 20mm calibration cube and scale holes +0.2mm.",
        "under-extrusion": "Nozzle check + esteps before touching slicer settings.",
        "other": "Log one more line about each failure and the pattern will sharpen.",
    }
    return (head + f" Repeat offender: {top} ({len(groups[top])}x"
            + (f" — e.g. '{groups[top][0]}'" if groups[top][0] else "") + "). "
            + fixes.get(top, ""))
