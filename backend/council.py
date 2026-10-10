"""Council — three minds are harder to fool than one (contemplating mode,
honest edition).

One brain reviews its own work in the same context and calls it verified.
A council reviews in FRESH contexts and reports dissent. Three parallel
perspectives, then an independent grader that never saw the debate:

  skeptic   — feasibility: what breaks, what's hand-waved, edge cases
  engineer  — correctness: numbers, constraints, manufacturability
  simplifier — is there a smaller move that gets 80% of the result?

The grader reads the question + all three takes + any evidence, and writes
the verdict: what survives, what dies, what to do next. Dissent is kept,
not averaged away — a 2-1 split is reported as a split.

Free-tier discipline: 4 deep-model calls, run concurrently, ONLY on
explicit ask (`second_opinion`) or before risky CAD builds. Never inline
in chit-chat. Each call ~240s timeout like the dossier chain; total wall
time is one call's worth when providers aren't queued.
"""
from __future__ import annotations

import asyncio
import time


PERSPECTIVES = (
    ("skeptic",
     "You are the skeptic on Ada's engineering council. Attack feasibility: "
     "what breaks, what's hand-waved, which edge cases kill it. Be blunt, "
     "be specific, never flatter. End with your top 2 objections."),
    ("engineer",
     "You are the engineer on Ada's council. Check correctness: numbers, "
     "constraints, tolerances, manufacturability, printability. Redo any "
     "math that matters. End with PASS/FAIL per claim plus what would fix it."),
    ("simplifier",
     "You are the simplifier on Ada's council. Find the smaller move: what "
     "gets 80% of the result with 20% of the complexity? Kill beloved "
     "complications. End with your one recommended simplification."),
)

GRADER_SYS = (
    "You are the grader on Ada's council. You never saw the debate form — "
    "only this evidence. Read the question, the three takes, and the evidence. "
    "Write: VERDICT (one line), SURVIVES (bullets that withstood attack), "
    "DIES (bullets that didn't, with why), NEXT (one concrete step). "
    "If the council split 2-1 or drew, say so plainly instead of averaging. "
    "If evidence is thin, mark claims UNVERIFIED rather than blessing them.")


async def review(question: str, evidence: str = "", chat_fn=None,
                 timeout_s: float = 240.0) -> dict:
    """Run the full council. Returns {verdict, takes, dissent, ms}."""
    t0 = time.time()
    question = (question or "").strip()
    if not question:
        return {"verdict": "No question asked.", "takes": {}, "dissent": False,
                "ms": 0}
    if chat_fn is None:
        return {"verdict": "Council has no voice (chat_fn missing).",
                "takes": {}, "dissent": False, "ms": 0}

    async def one(name: str, sys: str) -> tuple[str, str]:
        prompt = (f"{sys}\n\nQUESTION:\n{question}\n\n"
                  f"EVIDENCE:\n{(evidence or '(none provided)')[:3000]}")
        try:
            out = await asyncio.wait_for(chat_fn(prompt), timeout=timeout_s)
            return name, (out or "").strip()[:2500] or "(silent)"
        except Exception as e:
            return name, f"(abstained: {type(e).__name__})"

    takes_list = await asyncio.gather(
        *(one(n, s) for n, s in PERSPECTIVES))
    takes = dict(takes_list)
    abstained = sum(1 for v in takes.values() if v.startswith("(abstained"))
    grade_prompt = (
        f"{GRADER_SYS}\n\nQUESTION:\n{question}\n\n"
        + "\n\n".join(f"{k.upper()}:\n{v}" for k, v in takes.items())
        + f"\n\nEVIDENCE:\n{(evidence or '(none)')[:3000]}")
    try:
        verdict = await asyncio.wait_for(chat_fn(grade_prompt), timeout=timeout_s)
        verdict = (verdict or "").strip()[:3000] or "Grader silent — takes stand as-is."
    except Exception as e:
        verdict = f"Grader unavailable ({type(e).__name__}) — takes stand as-is."
    dissent = abstained > 0 or any(
        w in verdict.lower() for w in ("split", "disagree", "dissent", "2-1", "unverified"))
    return {"verdict": verdict, "takes": takes, "dissent": dissent,
            "ms": int((time.time() - t0) * 1000)}
