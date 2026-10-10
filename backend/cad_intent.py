"""CAD intent — the ReliCAD explicit-spec stage (deterministic, no LLM).

Before the assembly engine runs, intent is made EXPLICIT:
  object, materials, dimensions found, what's MISSING, and at most 3
  targeted questions for the missing slots. Saved beside the build so
  engineering intent is auditable, not vibes.

Missing-slot scan (rules, never guesses):
  no dimensions  → ask overall size
  no material    → ask material
  shaft/gear/bearing words + no fit → ask shaft/fit
  mount/housing + no interface → ask mounting interface
  print words + no printer/material → ask printer + material

Repair discipline lives in cad_v3_bridge.design_assembly
(rounds=3, tries=2, refine_rounds=1) — this module never widens it.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


_DIM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m\b|in\b|ft\b|°|deg)\b", re.I)
_MAT = re.compile(r"\b(titanium|aluminium|aluminum|steel|brass|petg|pla|abs|"
                  r"carbon|nylon|tpu|inconel|cast_iron)\b", re.I)
_FIT_WORDS = ("shaft", "gear", "bearing", "bore", "axle", "pin", "hole")
_MOUNT_WORDS = ("mount", "housing", "bracket", "flange", "enclosure", "case")
_PRINT_WORDS = ("print", "slice", "stl", "3d print", "filament")


def draft_intent(prompt: str, photo_caption: str = "") -> dict:
    prompt = (prompt or "").strip()
    t = prompt.lower()
    dims = _DIM.findall(prompt)
    mats = list({m.lower() for m in _MAT.findall(prompt)})
    questions = []
    if not dims:
        questions.append("What overall size — bounding box or key dimensions in mm?")
    if not mats:
        questions.append("Which material — PETG, PLA, aluminium, steel, brass?")
    if any(w in t for w in _FIT_WORDS) and not re.search(
            r"(?:Ø|diameter|dia|fit|h7|h6|clearance|tolerance)", t):
        questions.append("Any shaft/bore diameters or fits — running clearance in mm?")
    if any(w in t for w in _MOUNT_WORDS) and not re.search(
            r"(?:bolt|m\d|screw|hole|mount.*mm)", t):
        questions.append("How does it mount — bolt pattern, hole sizes?")
    if any(w in t for w in _PRINT_WORDS) and not mats:
        questions.append("Which printer and filament — so I slice it right first time?")
    questions = questions[:3]
    enriched = prompt
    if photo_caption:
        enriched = f"{prompt} [Reference photo shows: {photo_caption}]"
    return {
        "object": prompt[:200],
        "photo_caption": (photo_caption or "")[:500],
        "dimensions_found": dims[:8],
        "materials_found": mats,
        "missing": (([] if dims else ["dimensions"])
                    + ([] if mats else ["material"])),
        "questions": questions,
        "enriched_prompt": enriched,
        "ambiguous": bool(questions),
    }


def save_intent(intent: dict, root_dir: str | Path) -> str:
    try:
        d = Path(root_dir) / "projects" / "cad_intents"
        d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = d / f"{stamp}.json"
        path.write_text(json.dumps(intent, indent=1))
        return str(path)
    except Exception as e:
        return f"intent save failed: {type(e).__name__}"


def spoken(intent: dict) -> str:
    """What she says when the intent needs answers (max 3 questions)."""
    qs = intent.get("questions", [])
    if not qs:
        return ""
    if len(qs) == 1:
        return f"One thing before I build, Sir — {qs[0]}"
    flow = " First, " + " Then, ".join(qs)
    return f"Three quick things before I build, Sir. {flow}."
