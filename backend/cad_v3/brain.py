"""Layer 2 — the brain. It writes a SPEC. It never writes geometry.

The whole point of the engine is visible in this module's return type: `Spec`, not
`str` of Python. A model asked for CadQuery code has to hold a coordinate frame in
its head and will silently place a cutter in the wrong one — the failure Embodied
CAD (2026) describes as invisible at the token level and fatal at the kernel level.
A model asked for a spec cannot make that mistake, because the schema has nowhere
to put a coordinate. It can still be *wrong* — a bad diameter, a mate onto the
wrong face — but wrong in a way `verify` measures and `repair` patches.

Following AssemCAD and ArtiCAD (2026), this is training-free: the leverage comes
from handing the model the exact vocabulary, the exact connector names, and a
worked example, then letting the kernel referee.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import registry, repair, validate
from .schema import FITS, MATE_KINDS, ORIENTS, Spec

Chat = Callable[[list[dict]], Awaitable[str]]

SPEC_DIR = Path(__file__).resolve().parent / "specs"

CONTRACT = f"""You design mechanical assemblies by writing a SPEC in JSON. You never write code.

{{
  "name": "<snake_case>", "units": "mm", "datum": "<the part id everything hangs off>",
  "intent": "<one paragraph: what this machine is and how power flows through it>",
  "parts": [{{"id": "<unique snake_case>", "skill": "<from the catalogue>",
             "material": "<steel|stainless|aluminium|bronze|cast_iron|inconel|titanium|brass>",
             "count": 1, "label": "<human name>", "params": {{...}}}}],
  "mates": [{{"a": "<part_id>.<connector>", "b": "<part_id>.<connector>",
             "kind": "{'|'.join(MATE_KINDS)}", "offset_mm": 0.0, "angle_deg": 0.0,
             "fit": "{'|'.join(FITS)}", "orient": "{'|'.join(ORIENTS)}"}}]
}}

HARD RULES — a spec that breaks one of these is rejected before it reaches the kernel:
1. NO COORDINATES, EVER. There is no x, y, z, rx, ry, rz, position, transform or
   origin field, and params may not smuggle one in. Every part's pose is DERIVED by
   the assembler from the mates you write. This is not a restriction to work around;
   it is the reason the output is buildable.
2. Prefer the catalogue below — those skills are proven and cost nothing to use, and
   for a catalogue skill you may pass ONLY the params its signature declares and
   reference ONLY the connectors it publishes. But the catalogue is not a cage: if
   the machine genuinely needs a part that is not in it, invent a snake_case skill
   name for that part (`shift_fork`, `selector_drum`) and pass the params it
   obviously needs. ADA will write that skill herself, referee its geometry, and
   keep it. Two rules for an invented part: never invent a name for something the
   catalogue already covers, and name its connectors the way the catalogue does —
   `front_face`, `rear_face`, `bore`, `od_mid`, `bolt_circle`, `seats` — because you
   are also specifying the seats that part will publish.
3. `datum` names one part. Every other part must reach it through a chain of mates,
   or the build fails with FLOATING_PARTS. Mate each part to the part it physically
   touches, not to a convenient hub.
4. `offset_mm` is measured ALONG THE HOST SEAT'S OUTWARD NORMAL — positive moves the
   child away from the face it sits on. It is relative, so it is legal. Use it for
   running clearance and axial spacing (a 0.5 mm gap, a 1.0 mm bearing float).
5. A part with `count` > N must be mated to a patterned connector (a bolt_circle, a
   vane_ring, a seats ring) whose station count is the same N. One mate seats the
   whole set: instance i takes seat i.
6. A fastener mated to a bolt_circle grows along that seat's +Z. To make a stud reach
   back through the flange it is threaded into, give it a NEGATIVE offset_mm equal to
   its own length.
7. `fit` is a real engineering statement, and the verifier reads it: "interference"
   and "press" are allowed to overlap solid, "slip"/"clearance"/"free" are not.
   Naming the wrong fit turns a correct assembly into a reported clash.
8. Dimension the assembly so it actually fits: a bore must exceed the shaft it runs
   on, a housing's rotor cavity must exceed the wheel's tip diameter.
9. SEAT FACING — the mistake that ruins otherwise correct designs, and the one thing
   you can now look up instead of guessing. Every connector in the catalogue is printed
   with the direction it faces, measured off its own frame: `bolt_circle(ring of 6,
   axial+)` faces ALONG the host's axis, `pitch_ring(ring of 24, radial)` faces OUT of
   it. A child mated to a seat is built along that seat's +Z. So a part whose job is to
   reach outward from a hub belongs on a seat printed `radial`; put it on an `axial+`
   seat and it stands up parallel to the shaft instead of lying in the disc.
   Two legal ways to get the facing you need, in this order:
     a. Mate to a seat whose printed facing is already `radial`.
     b. If the only seat with the right station count faces axially, keep that mate and
        set `"orient": "radial"` on it. That re-aims the child on the seat it already
        sits on — no pose, no coordinate — and `offset_mm` then measures outward along
        the radius instead of up the axis.
   And one fact the catalogue cannot fix for you: every body of revolution in it
   (`revolved_profile`, `shaft`, `disc`, `flange`, `ring`, `ball_bearing`) is
   rotationally symmetric by construction, so no combination of params will make one
   reach sideways, however close a profile name sounds. If the part you need is not
   symmetric about its own axis, it is not in the catalogue: invent it (rule 2),
   publish a root seat on it, and mate that root to a station. The giveaway that you
   have tried to force it is a catalogue part whose `bore_d` is nearly its `od`.
10. `count` follows the seat in both directions. Rule 5 stops you seating 4 parts on a
   6-station ring; this stops the opposite: a bolt_circle with 6 stations wants a
   fastener with `count: 6`, and a 4-blade rotor wants ONE blade part with `count: 4`
   on a 4-station seat — never four separately-named blade parts, and never one blade
   standing in for the set.
"""


def _worked_example(path: Path | None = None, parts: int = 3, mates: int = 4) -> str:
    """A few real lines from the shipped turbocharger spec.

    A hand-written example in the prompt would drift from the schema the moment the
    schema changed; quoting the spec that the test suite actually builds cannot.
    """
    src = path or (SPEC_DIR / "turbocharger.json")
    try:
        raw = json.loads(src.read_text())
    except (OSError, ValueError):
        return ""
    trimmed = {
        "name": raw.get("name", ""), "units": "mm", "datum": raw.get("datum", ""),
        "parts": raw.get("parts", [])[:parts],
        "mates": raw.get("mates", [])[:mates],
    }
    return ("WORKED EXAMPLE (the first few lines of a real, verified assembly — note that\n"
            "the shaft is the datum and every other part is placed by touching something):\n"
            + json.dumps(trimmed, indent=1))


def system_prompt(categories=None, *, example: bool = True) -> str:
    """The contract, the whole catalogue, and one real example."""
    blocks = [CONTRACT, "CATALOGUE — every skill you may name, with its exact params:",
              registry.vocabulary_digest(categories)]
    if example:
        blocks.append(_worked_example())
    blocks.append("Reply with the JSON spec and nothing else. No prose, no code fences "
                  "around anything but the JSON, no comments inside the JSON.")
    return "\n\n".join(b for b in blocks if b)


def extract_json(text: str) -> dict:
    """Pull one JSON object out of a model reply. Tolerant of fences and preamble."""
    body = text.strip()
    if "```" in body:
        chunks = [c for c in body.split("```") if "{" in c]
        if chunks:
            body = max(chunks, key=len)
            if body.lstrip().lower().startswith("json"):
                body = body.split("\n", 1)[1] if "\n" in body else body
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    blob = body[start:end + 1]
    try:
        return json.loads(blob)
    except ValueError:
        # Trailing commas are the single most common malformation; worth one retry
        # locally rather than another round trip to the model.
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", blob))


async def _default_chat(messages: list[dict]) -> str:
    """ADA's own provider chain. Imported lazily so cad_v3 stays standalone.

    The budget is deliberately large. One reply here is a whole machine — a 20-part
    spec runs past 6k tokens of JSON on its own, and a reasoning model pays for its
    thinking out of the same allowance, so a tight cap comes back as an *empty* reply
    with finish_reason='length' rather than as a short one. Synthesis shares this
    transport and is even hungrier: it writes every missing skill in one batch.
    """
    from providers import cad_chain, chat_with_fallback

    return await chat_with_fallback(cad_chain(), messages, temperature=0.2,
                                    max_tokens=int(os.getenv("ADA_CAD_MAX_TOKENS", "32000")),
                                    timeout=420.0)


@dataclass
class Draft:
    spec: Spec | None
    raw: str = ""
    errors: list = field(default_factory=list)
    calls: int = 0
    synth: Any = None          # SynthResult, when the spec needed new vocabulary

    @property
    def ok(self) -> bool:
        return self.spec is not None and not self.errors


async def author(intent: str, *, chat: Chat | None = None, categories=None,
                 tries: int = 2, allow_unknown: bool = False,
                 log: Callable[[str], None] | None = None) -> Draft:
    """Ask for a spec, repair its near-misses for free, and send the validator's own
    words back once for the errors that are actually design mistakes.

    The retry is deliberately *not* a geometry loop: schema violations are free to
    detect and cost one short message to fix, so they get resolved before a single
    solid is built. Anything that survives validation is a geometry question, which
    belongs to `repair`.

    Two rules keep a retry from being a net loss, both learned from live runs:
    `validate.autofix` first substitutes legal tokens for illegal ones so a single
    misspelled connector cannot cost an assembly; and every attempt is *scored*, so
    the draft that comes back is the best one seen rather than the last one seen. A
    reasoning model that spends its second reply thinking and returns no JSON used to
    destroy a perfectly good first draft.
    """
    transport = chat or _default_chat
    messages = [
        {"role": "system", "content": system_prompt(categories)},
        {"role": "user", "content":
            f"Design this as an assembly spec:\n\n{intent}\n\n"
            "Use as many parts as the machine really needs. Every part must touch "
            "something. Return only JSON."},
    ]
    draft = Draft(None)
    best: tuple[int, Spec, list] | None = None
    for attempt in range(1, max(1, tries) + 1):
        try:
            raw = await transport(messages)
        except Exception as exc:
            # A dead provider is not a design failure, and it must not surface as a
            # traceback from three layers down: the caller gets a Draft that says so.
            draft.calls = attempt
            draft.errors = [f"transport failed: {type(exc).__name__}: {exc}"]
            if log:
                log(f"author attempt {attempt}: {draft.errors[0]}")
            return _settle(draft, best, log)
        draft.calls = attempt
        draft.raw = raw
        try:
            spec = Spec.from_dict(extract_json(raw))
        except ValueError as exc:
            draft.errors = [f"reply was not JSON: {exc}"]
            if log:
                log(f"author attempt {attempt}: {draft.errors[0]}")
            messages += [{"role": "assistant", "content": raw[:2000]},
                         {"role": "user", "content":
                          "That was not parseable JSON. Return the spec as a single "
                          "JSON object and nothing else."}]
            continue
        healed = validate.autofix(spec, allow_unknown=allow_unknown)
        if healed and log:
            log(f"author attempt {attempt}: healed {len(healed)} near-miss(es) "
                "without a model call: " + "; ".join(healed[:4])
                + (f" (+{len(healed) - 4} more)" if len(healed) > 4 else ""))
        errs = validate.validate(spec, allow_unknown=allow_unknown)
        draft.spec, draft.errors = spec, errs
        if best is None or len(errs) < best[0]:
            best = (len(errs), spec, errs)
        if log:
            gaps = validate.unknown_skills(spec) if allow_unknown else []
            log(f"author attempt {attempt}: {len(spec.parts)} parts, "
                f"{len(spec.mates)} mates, {len(errs)} spec errors"
                + (f", {len(gaps)} new skill(s) needed: {', '.join(gaps)}" if gaps else ""))
        if not errs:
            return draft
        if attempt == tries:
            break
        messages += [{"role": "assistant", "content": raw[:4000]},
                     {"role": "user", "content":
                      "The spec failed validation. Fix exactly these problems and "
                      "return the whole corrected spec as JSON:\n\n"
                      + validate.summary(errs)}]
    return _settle(draft, best, log)


def _settle(draft: Draft, best: tuple[int, Spec, list] | None,
            log: Callable[[str], None] | None) -> Draft:
    """Return the best draft seen, not the last one.

    Without this, a retry can only ever lose: the second reply overwrites the first
    even when it is unparseable or the provider died, so a 15-part spec with two
    fixable errors gets thrown away in favour of nothing at all. `best` is by
    construction the fewest-error *validated* attempt, so if the final reply is not
    that attempt, the final reply is not the one to keep.
    """
    if best is None:
        return draft
    score, spec, errs = best
    if draft.spec is spec and draft.errors is errs:
        return draft
    if log:
        log(f"author: keeping the best attempt ({len(spec.parts)} parts, "
            f"{score} spec error(s)) over the last reply")
    draft.spec, draft.errors = spec, errs
    return draft


# Errors the kernel genuinely cannot survive. Everything else — an unresolvable mate,
# a floating part, a pattern with no seat — the assembler records and `verify` measures,
# which is the entire point of having a closed loop: a spec with two open questions
# still builds 22 correct parts, and that beats handing the prompt to a code writer.
FATAL_CODES = {"EMPTY", "UNKNOWN_SKILL", "MISSING_ID", "DUPLICATE_ID", "BAD_COUNT"}


def blocking(errors: list) -> list:
    """The subset of `errors` that must stop a build. Strings are always blocking:
    they mean no spec came back at all."""
    out = []
    for e in errors:
        code = getattr(e, "code", None)
        if code is None or code in FATAL_CODES:
            out.append(e)
    return out




async def patch(spec: Spec, diagnoses: list, *, chat: Chat | None = None) -> list:
    """Hand the failures the deterministic rules could not fix to the model.

    It answers in `SpecPatch` objects, so even here the model is editing one spec
    cell — it is never shown code and never asked for a coordinate.
    """
    transport = chat or _default_chat
    prompt = repair.repair_prompt(spec, diagnoses)
    raw = await transport([
        {"role": "system", "content":
            "You repair mechanical assembly specs. You answer only with a JSON array "
            "of patches. You never write code and never write coordinates."},
        {"role": "user", "content": prompt},
    ])
    return repair.parse_patches(raw)


async def design(intent: str, *, chat: Chat | None = None, categories=None,
                 rounds: int = repair.MAX_ROUNDS, clash: bool = True,
                 tries: int = 2, grow: bool = True, sweep: bool = False,
                 log: Callable[[str], None] | None = None):
    """Layers 1-4 end to end: author a spec, grow the vocabulary it needs, build,
    verify, repair.

    Returns `(draft, result)`. `result` is None when the spec never became valid,
    which is the one failure the kernel cannot referee.

    Two model requests is the normal cost of a machine ADA has never seen: one to
    author the spec, one to write every missing skill in it. Synthesis sits between
    authoring and repair for a reason — `repair` edits spec cells against measured
    geometry, and there is no geometry to measure until every named part exists.
    """
    draft = await author(intent, chat=chat, categories=categories, tries=tries,
                         allow_unknown=grow, log=log)
    if draft.spec is None or blocking(draft.errors):
        return draft, None
    if draft.errors and log:
        log(f"building anyway with {len(draft.errors)} open spec question(s) — the "
            "kernel places what it can and verify reports the rest: "
            + validate.summary(draft.errors[:3]).replace("\n", " | "))

    if grow:
        from . import synth

        orders = synth.work_orders(draft.spec)
        if orders:
            if log:
                log(f"synth: {len(orders)} missing skill(s) in one request — "
                    + ", ".join(o.skill for o in orders))
            draft.synth = await synth.fulfil(orders, chat=chat, sweep=sweep, log=log)
            draft.calls += draft.synth.calls
            # Strict now: every named skill exists, as real geometry or as an
            # envelope, so anything still wrong is an ordinary spec error. Run it
            # through autofix once more: the seats synth just published are new
            # words, and a connector the model guessed may now be one edit away
            # from a real one.
            validate.autofix(draft.spec)
            draft.errors = validate.validate(draft.spec)
            if blocking(draft.errors):
                return draft, None


    loop = asyncio.get_running_loop()
    transport = chat or _default_chat

    def ask(spec_now: Spec, diagnoses: list) -> list:
        # `repair.repair` is synchronous CAD work, so it runs in a worker thread;
        # its escape hatch has to hop back onto this loop to reach the model.
        future = asyncio.run_coroutine_threadsafe(
            patch(spec_now, diagnoses, chat=transport), loop)
        try:
            return future.result(timeout=300)
        except Exception as exc:              # a dead provider must not kill the loop
            if log:
                log(f"brain repair unavailable: {type(exc).__name__}: {exc}")
            return []

    result = await asyncio.to_thread(repair.repair, draft.spec, rounds=rounds,
                                     clash=clash, ask=ask, log=log)
    return draft, result


def design_sync(intent: str, **kw):
    """Blocking entry point for the CLI."""
    return asyncio.run(design(intent, **kw))
