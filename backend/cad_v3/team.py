"""Layer 3 — the team that plans like an engineer, not like a prompt-follower.

Layers 1-4 take a sentence and turn it into a measured assembly. That is a
translator. This module is the part that behaves like a small engineering office:
somebody works out what you actually meant, somebody writes a dimensioned plan,
somebody designs to it, somebody else inspects the result, and the inspector's
findings get acted on.

The five roles, none of which reimplements anything below it:

  CLARIFIER   decides whether the request is under-specified, asks Sir at most two
              questions, and folds the answers into a brief. Silent when the prompt
              is already a drawing — a clarifier that interrogates a complete
              request is worse than no clarifier, because it is a gate.
  PLANNER     turns the brief into a dimensioned engineering plan: overall
              envelope, the parts the machine is made of, materials, the interfaces
              that have to match, and the measurements the finished assembly must
              satisfy. That plan is what gets authored — so the acceptance criteria
              are decided before the design, not checked against whatever came back.
  DESIGNER    Layers 1-4, unchanged: author a spec, grow the vocabulary it needs,
              place every part by connector frame, measure it, repair it.
  INSPECTOR   reads the result back the way a second engineer would — `verify`'s
              report, the evidence verdicts, the geometry gate's own words on every
              rejected skill, and the placeholder census — and states findings with
              the action that would clear each one.
  REFINER     acts on the one class of finding the loop below cannot. `repair` edits
              cells of a spec that is already fully synthesised; it has no move when
              a part is a *placeholder*, because the fix is new geometry, not a new
              number. So the refiner re-synthesises exactly the stubbed skills at the
              spec's repaired parameters, rebuilds, and re-verifies.

Two honest boundaries. This adds model calls — a plan, and possibly one synthesis
retry — on top of Layers 1-4, so it is slower than the bare loop; it is on by
default because a plan is what makes a design reviewable, and off in the CLI when
you ask for speed. And the refiner only re-attempts *placeholders*: everything else
`repair` has already had four rounds at, and a fifth guess at a clash is not a
refinement.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

Chat = Callable[[list[dict]], Awaitable[str]]
Ask = Callable[..., Awaitable[str]]

MAX_QUESTIONS = 2


# ══════════════════════════════════════════════════════════════════════════
# CLARIFIER
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Question:
    key: str
    text: str
    choices: tuple[str, ...] = ()
    default: str = ""
    why: str = ""

    def line(self) -> str:
        out = self.text
        if self.choices:
            out += "  Options: " + " / ".join(self.choices)
        return out


# A prompt that already answers "how big" — a real dimension, a ratio, or a load
# figure (which pins scale indirectly: nobody asks for a 5 Nm coupling the size of a
# house).
_HAS_SIZE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|millimetres?|millimeters?|metres?|meters?|"
    r"inches|inch|in|ft|feet|m)\b"
    r"|\b\d+(?:\.\d+)?\s*[x×]\s*\d+"
    r"|\b\d+(?:\.\d+)?\s*(?:g|kg|n|kn|nm|rpm|kw|w)\b", re.I)

# The operator has already said "I don't want to be asked".
_SKIP = re.compile(
    r"\bjust build it\b|\bjust make it\b|\byour call\b|\bup to you\b|\bsurprise me\b"
    r"|\bwhatever\b|\buse your judgement\b|\bdon'?t ask\b|\bno questions\b"
    r"|\bjust do it\b|\banything you (?:like|want)\b", re.I)

# It fits something, but the counterpart is not named or sized.
_FITS = re.compile(
    r"\b(?:fits?|mates?|threads?|slides?|press(?:es)?|splines?|keys?|bolts?|screws?|"
    r"mounts?|attaches?|connects?|couples?|interfaces?)\b", re.I)
_HAS_BORE = re.compile(
    r"\b\d+(?:\.\d+)?\s*mm\s*(?:shaft|bore|hole|pin|rod|bolt|thread)"
    r"|\b(?:m\d+(?:\.\d+)?)\b|\bbore\b\s*(?:of\s*)?\d", re.I)

# Nouns whose whole design turns on how much load they carry. Deliberately NOT the
# full list of mechanical nouns: a bracket with a stated size and material is already
# a complete request, and asking what a bracket is for is the clarifier grading its
# own homework. A gearbox with no torque rating is genuinely undecidable — the housing
# wall and the module follow from the load, not from the envelope.
_POWERED = re.compile(
    r"\b(?:gear|gears|gearbox|gear ?box|transmission|motor|engine|turbine|pump|"
    r"actuator|coupling|winch|hoist|press|crane|conveyor|ratchet|damper|spindle|"
    r"crank|driveshaft|drive ?shaft|reducer|differential)\b", re.I)
_HAS_LOAD = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:n|kn|nm|kg|g|tonne|tons?|lb|lbf)\b"
    r"|\b(?:light|heavy|load|torque|stress|pressure|force|duty|rated)\b", re.I)


def _q_scale() -> Question:
    return Question(
        key="scale",
        text="How big should it be, Sir?",
        choices=("hand-sized, about 150 mm",
                 "desktop, about 400 mm",
                 "bench-scale, about 1 metre"),
        default="desktop, about 400 mm",
        why="no dimension anywhere in the request",
    )


def _q_duty() -> Question:
    return Question(
        key="duty",
        text="What does it have to carry?",
        choices=("light — display or prototype",
                 "moderate — hand tools and fixtures",
                 "heavy — machinery under real load"),
        default="moderate — hand tools and fixtures",
        why="a load-carrying part with no load stated",
    )


def _q_interface() -> Question:
    return Question(
        key="interface",
        text="What does it have to mate with?",
        choices=("nothing — it stands alone",
                 "a standard shaft or bolt",
                 "a specific part I will size for you"),
        default="nothing — it stands alone",
        why="the request says it fits something but not what",
    )


def ambiguity(prompt: str, *, mode: Any = None, limit: int = MAX_QUESTIONS) -> list[Question]:
    """What this request does not say that the design genuinely needs.

    Deliberately conservative, because every question is a cost paid by somebody who
    asked for a part. Three tests have to fail before one is asked: the prompt must
    have no dimension, the object must be one the schema cannot size on its own, and
    something about the request must make the missing value load-bearing.

    A SCENE or PHYSICS request gets nothing: there is no new part to size, so the
    only honest question would be about the picture, and that is not this module's.
    """
    text = (prompt or "").strip()
    if len(text) < 3 or _SKIP.search(text):
        return []
    if mode is None:
        try:
            from . import modes

            mode = modes.detect(text).mode
        except Exception:
            mode = None
    if mode is not None and getattr(mode, "value", mode) != "cad":
        return []

    out: list[Question] = []
    if not _HAS_SIZE.search(text):
        out.append(_q_scale())
    if _FITS.search(text) and not _HAS_BORE.search(text):
        out.append(_q_interface())
    if _POWERED.search(text) and not _HAS_LOAD.search(text):
        out.append(_q_duty())
    return out[:max(0, limit)]


@dataclass
class Brief:
    """The request as the planner and designer should see it — with the answers."""

    prompt: str
    questions: list[Question] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def clarified(self) -> bool:
        return bool(self.answers)

    def text(self) -> str:
        if not self.answers:
            return self.prompt
        lines = [self.prompt, "",
                 "THE OPERATOR ANSWERED THESE BEFORE YOU DESIGN. They are requirements, "
                 "not suggestions — size the machine to them:"]
        for q in self.questions:
            ans = self.answers.get(q.key)
            if ans:
                lines.append(f"- {q.key}: {ans}   (asked because {q.why})")
        return "\n".join(lines)


async def clarify_brief(prompt: str, ask: Ask, *, limit: int = MAX_QUESTIONS,
                        log: Callable[[str], None] | None = None) -> Brief:
    """Ask at most `limit` questions and fold the answers in.

    `ask` is injected — in the live system it is ADA speaking and listening; in the
    CLI it can be `input`; in a test it is whatever the test wants. When it raises or
    the operator says "just build it", the remaining questions are dropped and the
    defaults carry, so a broken ask channel degrades to the old behaviour rather than
    hanging a build behind a question nobody can answer.
    """
    brief = Brief(prompt=prompt)
    for q in ambiguity(prompt, limit=limit):
        try:
            raw = await ask(q.line(), q.choices, q.default)
        except Exception as exc:
            if log:
                log(f"clarifier: could not ask about {q.key} ({type(exc).__name__}) "
                    "— using the default")
            raw = q.default
        answer = (raw or "").strip() or q.default
        if _SKIP.search(answer):
            if log:
                log(f"clarifier: operator said to proceed — dropping {q.key} and the rest")
            break
        brief.questions.append(q)
        brief.answers[q.key] = answer
        if log:
            log(f"clarifier: {q.key} = {answer}")
    return brief


# ══════════════════════════════════════════════════════════════════════════
# PLANNER
# ══════════════════════════════════════════════════════════════════════════

PLAN_CONTRACT = """You are ADA's design planner. You are given a request and, sometimes, the
operator's answers to questions about it. You produce a short, dimensioned
engineering plan — never code, never a spec, never prose beyond the JSON.

Reply with ONE JSON object and nothing else:

{
  "function": "<one sentence: what the machine does and how power or load moves through it>",
  "envelope_mm": [<x>, <y>, <z>],          // overall bounding box of the FINISHED machine
  "parts": ["<snake_case part>", ...],     // every distinct part, in the order it matters
  "materials": {"<part>": "steel|stainless|aluminium|bronze|cast_iron|inconel|titanium|brass"},
  "interfaces": ["<the numbers that must match between two parts, e.g. shaft od 12 mm runs in a 12.05 mm bore>"],
  "checks": ["<a measurement the finished assembly must satisfy, e.g. overall width 120 mm>"]
}

RULES
1. Every number is in millimetres and is a real engineering figure for a real object
   of that size — a hand-sized bracket is ~80 mm, a gearbox for a bench is ~300 mm.
2. `envelope_mm` is the assembly's overall box, not any one part's.
3. `parts` names functions, not copies: one "planet_gear" with a count of 3, never
   three separately named planet gears.
4. Include the parts that make the thing work — the fasteners that hold it, the
   bearing that carries the shaft — so the design has something to hang together
   with. Keep it to the parts that earn their place; a 40-part plan for a bracket is
   as wrong as a 2-part plan for a gearbox.
5. `checks` must be measurable on the finished solids. "looks good" is not a check;
   "overall height 150 mm" and "rotor tip clears the housing by 0.5 mm" are.
6. If the operator answered a question, the plan MUST honour that answer."""


@dataclass
class Plan:
    function: str = ""
    envelope_mm: tuple[float, float, float] = ()
    parts: list[str] = field(default_factory=list)
    materials: dict[str, str] = field(default_factory=dict)
    interfaces: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    source: str = "heuristic"
    raw: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.function and self.envelope_mm)

    def envelope_text(self) -> str:
        if not self.envelope_mm:
            return "unspecified"
        return " x ".join(f"{v:g}" for v in self.envelope_mm) + " mm"

    def as_intent(self, brief: Brief) -> str:
        """The design intent handed to Layer 2 — the brief plus the plan plus the test."""
        lines = [brief.text(), "",
                 "ENGINEERING PLAN — realise this plan. It is the acceptance test, not a "
                 "mood board, so where it states a number, the spec must carry it:"]
        if self.function:
            lines.append(f"- function: {self.function}")
        if self.envelope_mm:
            lines.append(f"- overall envelope: {self.envelope_text()}  "
                         "-> put this in `expect.bbox_mm`")
        if self.parts:
            lines.append(f"- parts: {', '.join(self.parts)}  (one part per function; use "
                         "`count` for repeats rather than duplicating the name)")
        if self.materials:
            lines.append("- materials: " + ", ".join(f"{k}={v}"
                                                     for k, v in sorted(self.materials.items())))
        if self.interfaces:
            lines.append("- interfaces that must match: " + "; ".join(self.interfaces))
        if self.checks:
            lines.append("- the finished assembly must measure: " + "; ".join(self.checks)
                         + "  -> put these in `expect.dims` and `checks`")
        lines.append(
            "Use a catalogue skill wherever it already has a word for a part. Where it "
            "does not, invent a snake_case skill for it and publish its seats — the "
            "catalogue is a starting point, not a cage. A machine nobody has built "
            "before is a valid answer.")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict, *, raw: str = "") -> "Plan":
        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        env = data.get("envelope_mm") or []
        if isinstance(env, dict):
            env = [env.get(k) for k in ("x", "y", "z")]
        envelope = tuple(_f(v) for v in list(env)[:3])
        if len(envelope) != 3 or any(v <= 0 for v in envelope):
            envelope = ()

        def _strs(v: Any) -> list[str]:
            if isinstance(v, str):
                return [v]
            return [str(x) for x in v if str(x).strip()] if isinstance(v, (list, tuple)) else []

        mats = data.get("materials") or {}
        return cls(
            function=str(data.get("function") or "").strip(),
            envelope_mm=envelope,
            parts=_strs(data.get("parts")),
            materials={str(k): str(v) for k, v in mats.items()} if isinstance(mats, dict) else {},
            interfaces=_strs(data.get("interfaces")),
            checks=_strs(data.get("checks")),
            raw=raw,
        )

    @classmethod
    def heuristic(cls, prompt: str) -> "Plan":
        """The no-model fallback: what the sentence already pins down, and nothing more.

        It does not invent parts, because inventing parts without a reason is how you
        get a plan that reads well and means nothing. It extracts a size if the prompt
        has one and leaves the rest for the catalogue. A plan that says less than the
        prompt is still better than blocking the build on a provider.
        """
        env: tuple[float, float, float] = ()
        m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)"
                      r"(?:\s*[x×]\s*(\d+(?:\.\d+)?))?", prompt or "")
        if m:
            vals = [float(g) for g in m.groups() if g]
            env = tuple(vals) if len(vals) == 3 else (vals[0], vals[1], min(vals))
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m\b)", prompt or "")
            if m:
                v = float(m.group(1))
                env = (v, v, v)
        return cls(function=(prompt or "").strip()[:200], envelope_mm=env, source="heuristic")


_AGENT = "cad_team"


async def plan_brief(brief: Brief, *, chat: Chat | None = None,
                     log: Callable[[str], None] | None = None) -> Plan:
    """One model call for a dimensioned plan, with the sentence as the fallback."""
    from . import brain

    transport = chat or brain._default_chat
    try:
        raw = await transport([
            {"role": "system", "content": PLAN_CONTRACT},
            {"role": "user", "content": brief.text()},
        ])
    except Exception as exc:
        if log:
            log(f"planner: model unavailable ({type(exc).__name__}) — planning from the "
                "sentence alone")
        return Plan.heuristic(brief.prompt)

    try:
        plan = Plan.from_dict(brain.extract_json(raw or ""), raw=raw)
    except Exception as exc:
        if log:
            log(f"planner: reply was not a plan ({type(exc).__name__}) — falling back")
        return Plan.heuristic(brief.prompt)
    if not plan.ok:
        if log:
            log("planner: plan was incomplete — filling the gaps from the sentence")
        fallback = Plan.heuristic(brief.prompt)
        return Plan(function=plan.function or fallback.function,
                    envelope_mm=plan.envelope_mm or fallback.envelope_mm,
                    parts=plan.parts, materials=plan.materials,
                    interfaces=plan.interfaces, checks=plan.checks,
                    source="model+heuristic", raw=raw)
    plan.source = "model"
    return plan


# ══════════════════════════════════════════════════════════════════════════
# INSPECTOR
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Finding:
    """One thing a second engineer would say about the result, and what clears it."""

    stage: str          # verify | evidence | gate | placeholder | spec
    subject: str
    ok: bool
    detail: str
    action: str = ""

    def __str__(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        tail = f"  -> {self.action}" if self.action else ""
        return f"{mark} [{self.stage}] {self.subject}: {self.detail}{tail}"


def _evidence(name: str, asm, log: Callable[[str], None] | None = None) -> dict:
    """Interference audit + mass properties on the in-memory assembly.

    Never fatal: evidence is a bonus and a missing bonus must not read as a failed
    build.
    """
    from . import modes as cad_modes

    try:
        cad = cad_modes.cad_output(name or "model", asm)
    except Exception as exc:
        if log:
            log(f"evidence unavailable: {type(exc).__name__}: {exc}")
        return {"status": "UNKNOWN", "parts": 0, "verdicts": [],
                "note": f"evidence unavailable: {type(exc).__name__}"}
    if cad is None:
        return {"status": "UNKNOWN", "parts": 0, "verdicts": [],
                "note": "evidence returned nothing"}
    return {
        "status": cad.status,
        "parts": cad.parts,
        "verdicts": [{"test": v.test, "status": v.status, "summary": v.summary}
                     for v in cad.verdicts],
        "note": "",
    }


def inspect(*, report=None, evidence: dict | None = None, synth=None,
            spec=None) -> list[Finding]:
    """Every objective statement the run produced about the design, as findings.

    Nothing here measures anything new — it reads the measurements Layers 4 and 5
    already made and states them as findings with an action. The one thing it adds is
    that they all appear in one place, in severity order, so a build that "passed"
    with three placeholders in it cannot be mistaken for a finished part.
    """
    out: list[Finding] = []

    if synth is not None:
        for name, verdict in sorted(getattr(synth, "verdicts", {}).items()):
            if not verdict.admitted:
                out.append(Finding("gate", name, False,
                                   (verdict.why or "rejected")[:300],
                                   "the refiner will re-attempt this skill at the "
                                   "spec's repaired parameters"))

    if report is not None:
        if report.unplaced:
            out.append(Finding("verify", "connectivity", False,
                               f"{len(report.unplaced)} instance(s) never received a "
                               f"pose: {', '.join(report.unplaced[:6])}",
                               "the matings do not reach these parts"))
        if report.invalid:
            out.append(Finding("verify", "watertight", False,
                               f"{len(report.invalid)} solid(s) are not valid B-reps",
                               "the skill that makes that part has to be fixed"))
        if report.severe_clashes:
            worst = max(report.severe_clashes, key=lambda c: c.volume)
            out.append(Finding("verify", "clash", False,
                               f"{len(report.severe_clashes)} severe overlap(s); worst "
                               f"{worst.a} <-> {worst.b} at {worst.volume:.1f} mm3 "
                               f"(fit '{worst.fit}' allows {worst.allowed:.1f})",
                               "repair re-spaces or re-sizes the parts"))
        if report.detached:
            out.append(Finding("verify", "attachment", False,
                               f"{len(report.detached)} part(s) touch nothing: "
                               f"{', '.join(report.detached[:6])}",
                               "their mates place them in free space"))
        if report.misaligned:
            out.append(Finding("verify", "mates", False,
                               f"{len(report.misaligned)} mate(s) do not hold",
                               "the solver could not satisfy them together"))
        if report.collapsed:
            out.append(Finding("verify", "pattern", False,
                               f"{len(report.collapsed)} patterned part(s) share stations",
                               "the seat has fewer stations than the part has count"))
        if getattr(report, "bad_dims", None):
            out.append(Finding("verify", "dimensions", False,
                               f"{len(report.bad_dims)} of {len(report.dims)} stated "
                               "dimension(s) are wrong",
                               "the design does not match its own stated sizes"))
        if getattr(report, "unmeasurable_dims", None):
            out.append(Finding("verify", "dimensions", False,
                               f"{len(report.unmeasurable_dims)} stated dimension(s) "
                               "could not be measured at all",
                               "the spec asserts something about a part with no geometry"))
        if getattr(report, "bad_conditions", None):
            out.append(Finding("verify", "conditions", False,
                               f"{len(report.bad_conditions)} engineering condition(s) "
                               "are violated",
                               "tooth counts, modules or fit clearances are wrong"))
        if getattr(report, "envelope", None):
            out.append(Finding("verify", "envelope", False,
                               "; ".join(report.envelope[:4]),
                               "the design falls outside the plan's own bounds"))
        if getattr(report, "unstated", False):
            out.append(Finding("verify", "unstated", False,
                               "the spec states no dimensions, conditions or envelope, "
                               "so nothing checked whether it is the right shape",
                               "the plan's checks belong in the spec's `expect` block"))
        if report.placeholders:
            out.append(Finding("placeholder", "geometry", False,
                               f"{len(report.placeholders)} of {report.parts} instance(s) "
                               "are dimensioned envelopes, not real parts: "
                               + ", ".join(sorted(set(report.placeholder_skills))[:8]),
                               "the refiner will re-attempt these skills"))
        if getattr(report, "open_shells", None):
            out.append(Finding("verify", "watertight", False,
                               f"{len(report.open_shells)} open shell(s)",
                               "the part is not a solid"))

    for v in (evidence or {}).get("verdicts", []):
        ok = str(v.get("status", "")).upper() in ("PASS", "OK", "SKIP")
        out.append(Finding("evidence", str(v.get("test", "?")), ok,
                           str(v.get("summary", ""))[:300],
                           "" if ok else "the physics audit disagreed with the design"))

    if spec is not None and not out:
        out.append(Finding("verify", "assembly", True,
                           "no finding: every check Layers 4 and 5 could make passed",
                           ""))
    return out


# ══════════════════════════════════════════════════════════════════════════
# REFINER
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Refinement:
    round: int
    action: str
    before: str
    after: str
    improved: bool
    detail: str = ""
    admitted: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        mark = "improved" if self.improved else "no gain"
        return (f"refine {self.round} [{self.action}] {mark}: {self.before} -> {self.after}"
                + (f"  ({self.detail})" if self.detail else ""))


def _stub_skills(asm) -> list[str]:
    names: set[str] = set()
    for inst in getattr(asm, "instances", []) or []:
        meta = getattr(getattr(inst, "result", None), "meta", None) or {}
        if meta.get("source") == "stub":
            names.add(inst.skill)
    return sorted(names)


async def refine(*, spec, asm, report, chat: Chat | None = None, rounds: int = 1,
                 ask_patch: Callable[..., list] | None = None, log=None) -> tuple:
    """Re-synthesise the parts that came back as envelopes, then rebuild and re-verify.

    Why this is a separate pass and not another `repair` round: by the time we get
    here `repair` has already run its four rounds on a spec whose every part was
    either real geometry or an envelope. A clash caused by a placeholder sized wrong
    cannot be repaired by moving a number in the spec, because the number belongs to a
    part that does not exist yet. So the fix is to ask for the geometry again — now at
    the parameters repair settled on, which is a materially different brief from the
    one the first synthesis attempt answered.

    Returns `(asm, report, refinements)`.
    """
    from . import registry, repair as repair_mod, synth

    out: list[Refinement] = []

    def say(msg: str) -> None:
        if log:
            log(msg)

    stubs = _stub_skills(asm)
    if not stubs or rounds < 1:
        return asm, report, out

    before_n = len(getattr(report, "placeholders", []) or [])
    say(f"refiner: {len(stubs)} placeholder skill(s) to re-attempt — {', '.join(stubs)}")

    for n in range(1, max(1, rounds) + 1):
        # `fulfil` refuses to touch a name that is already registered (which is what
        # protects the curated vocabulary), so the stub has to be retired first. That
        # is safe precisely because stubs are never persisted.
        retired = []
        for name in stubs:
            sk = registry.get(name)
            if sk is not None and sk.source == "stub":
                synth.uninstall(name)
                retired.append(name)
        if not retired:
            break

        orders = [o for o in synth.work_orders(spec) if o.skill in retired]
        if not orders:
            break
        try:
            res = await synth.fulfil(orders, chat=chat, log=say)
        except Exception as exc:
            say(f"refiner: synthesis failed ({type(exc).__name__}: {exc}) — keeping the "
                "placeholders")
            break

        new_asm, new_rep = await asyncio.to_thread(repair_mod.build_and_verify, spec)
        after_n = len(getattr(new_rep, "placeholders", []) or [])
        # Real parts replaced envelopes iff the placeholder census shrank or a new
        # skill was admitted. Both are checked because a skill can be admitted and
        # still leave a placeholder behind if another part of the assembly stubbed.
        improved = after_n < before_n or bool(res.admitted)
        detail = (f"admitted {', '.join(res.admitted)}" if res.admitted
                  else "no candidate survived the gate: "
                       + "; ".join(f"{k}: {v.why[:80]}"
                                   for k, v in sorted(res.verdicts.items()))[:200])
        out.append(Refinement(n, "resynthesise",
                              f"{before_n} placeholder(s)",
                              f"{after_n} placeholder(s)", improved, detail,
                              admitted=list(res.admitted)))
        say(str(out[-1]))

        asm, report, before_n = new_asm, new_rep, after_n
        stubs = [s for s in stubs if s not in set(res.admitted)]
        if not stubs:
            break
        if not improved:
            break

    # The geometry changed, so the numbers repair chose may no longer be right: a part
    # that was a 40 mm envelope and is now a real 60 mm rotor can clash with what the
    # envelope sat beside. One more repair round on the new solids.
    if any(r.improved for r in out):
        say("refiner: geometry changed — re-running repair on the new solids")
        try:
            rr = await asyncio.to_thread(
                repair_mod.repair, spec, rounds=2, clash=True,
                ask=ask_patch, log=say)
            asm, report = rr.asm, rr.report
            out.append(Refinement(len(out) + 1, "repair",
                                  "geometry updated", "re-repaired",
                                  bool(getattr(rr, "fixed", False)),
                                  rr.text().replace("\n", " | ")[:200]))
            say(str(out[-1]))
        except Exception as exc:
            say(f"refiner: re-repair failed ({type(exc).__name__}: {exc}) — keeping "
                "the verified geometry")
    return asm, report, out


# ══════════════════════════════════════════════════════════════════════════
# THE TEAM
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TeamResult:
    prompt: str
    brief: Brief
    plan: Plan | None = None
    draft: Any = None
    result: Any = None
    report: Any = None
    evidence: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    refinements: list[Refinement] = field(default_factory=list)
    loglines: list[str] = field(default_factory=list)
    grown: list[str] = field(default_factory=list)
    stubbed: list[str] = field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0

    @property
    def spec(self):
        return getattr(self.result, "spec", None) or getattr(self.draft, "spec", None)

    @property
    def asm(self):
        return getattr(self.result, "asm", None)

    @property
    def built(self) -> bool:
        return self.asm is not None and bool(getattr(self.asm, "instances", None))

    def summary(self) -> str:
        lines = []
        if self.brief.clarified:
            lines.append("clarified: " + "; ".join(f"{k}={v}" for k, v in self.brief.answers.items()))
        if self.plan is not None:
            lines.append(f"plan ({self.plan.source}): {self.plan.function[:100]}")
            lines.append(f"  envelope {self.plan.envelope_text()}, {len(self.plan.parts)} part(s)")
        bad = [f for f in self.findings if not f.ok]
        lines.append(f"findings: {len(self.findings) - len(bad)} ok, {len(bad)} open")
        lines += [f"  {f}" for f in bad[:10]]
        lines += [f"  {r}" for r in self.refinements]
        lines.append(f"model calls: {self.calls}, {self.seconds:.1f}s")
        return "\n".join(lines)


def _patch_bridge(chat: Chat | None, loop, log) -> Callable[..., list]:
    """`repair`'s escape hatch, hopping from its worker thread back onto our loop."""
    from . import brain

    def ask(spec_now, diagnoses):
        if chat is None:
            return []
        future = asyncio.run_coroutine_threadsafe(
            brain.patch(spec_now, diagnoses, chat=chat), loop)
        try:
            return future.result(timeout=300)
        except Exception as exc:
            if log:
                log(f"repair: brain unavailable ({type(exc).__name__}: {exc})")
            return []
    return ask


async def run(prompt: str, *, ask: Ask | None = None, chat: Chat | None = None,
              categories=None, rounds: int = 3, tries: int = 2,
              clarify: bool = True, plan: bool = True, refine_rounds: int = 1,
              on_thought: Callable[[str], None] | None = None,
              log: Callable[[str], None] | None = None) -> TeamResult:
    """Clarify, plan, design, inspect, refine. Returns a `TeamResult`.

    Every step degrades: no `ask` means no questions but a noted ambiguity; a dead
    planner means Layer 2 gets the sentence; a failed refinement keeps the verified
    geometry it already had. The one thing that is never degraded is the design itself
    — if Layers 1-4 cannot produce a spec, this returns a TeamResult with no assembly
    and the caller falls back exactly as it did before.
    """
    from . import bootstrap, brain

    t0 = time.time()
    tr = TeamResult(prompt=prompt, brief=Brief(prompt=prompt))

    def say(msg: str) -> None:
        tr.loglines.append(msg)
        if log:
            try:
                log(msg)
            except Exception:
                pass
        if on_thought:
            try:
                on_thought(f"[team] {msg}")
            except Exception:
                pass

    n = bootstrap()
    say(f"vocabulary: {n} skills")

    if clarify:
        if ask is not None:
            tr.brief = await clarify_brief(prompt, ask, log=say)
        else:
            found = ambiguity(prompt)
            if found:
                say(f"clarifier: {len(found)} ambiguity(ies) found "
                    f"({', '.join(q.key for q in found)}) but no channel to ask — "
                    "proceeding with defaults")

    intent = prompt
    if plan:
        tr.plan = await plan_brief(tr.brief, chat=chat, log=say)
        say(f"planner({tr.plan.source}): envelope {tr.plan.envelope_text()}, "
            f"{len(tr.plan.parts)} part(s), {len(tr.plan.checks)} check(s)")
        intent = tr.plan.as_intent(tr.brief)

    draft, result = await brain.design(intent, chat=chat, categories=categories,
                                       rounds=rounds, tries=tries, grow=True, log=say)
    tr.draft, tr.result = draft, result
    tr.calls = int(getattr(draft, "calls", 0) or 0)
    if draft.synth is not None:
        tr.grown = sorted(set(getattr(draft.synth, "admitted", []) or []))
        tr.stubbed = sorted(set(getattr(draft.synth, "stubbed", []) or []))
        say(draft.synth.text())

    if result is None or getattr(result, "asm", None) is None:
        say("designer: no buildable spec — the caller falls back")
        tr.seconds = time.time() - t0
        return tr

    spec = result.spec
    tr.report = result.report
    tr.evidence = _evidence(getattr(spec, "name", "") or "model", result.asm, say)
    tr.findings = inspect(report=result.report, evidence=tr.evidence,
                          synth=draft.synth, spec=spec)
    n_bad = sum(1 for f in tr.findings if not f.ok)
    say(f"inspector: {len(tr.findings)} finding(s), {n_bad} open")
    for f in tr.findings:
        if not f.ok:
            say(f"  {f}")

    if refine_rounds > 0:
        loop = asyncio.get_running_loop()
        asm, report, refs = await refine(spec=spec, asm=result.asm, report=result.report,
                                         chat=chat, rounds=refine_rounds,
                                         ask_patch=_patch_bridge(chat, loop, say),
                                         log=say)
        tr.refinements = refs
        for r in refs:
            for name in r.admitted:
                if name not in tr.grown:
                    tr.grown.append(name)
        tr.grown = sorted(tr.grown)
        # The final census is the truth about what is still an envelope: a skill the
        # refiner admitted stopped being one, and anything it stubbed is now named here.
        tr.stubbed = sorted(set(getattr(report, "placeholder_skills", []) or []))
        if asm is not result.asm:
            result.asm, result.report = asm, report
            tr.report = report
            tr.evidence = _evidence(getattr(spec, "name", "") or "model", asm, say)
            tr.findings = inspect(report=report, evidence=tr.evidence,
                                  synth=draft.synth, spec=spec)

    tr.seconds = time.time() - t0
    return tr


def run_sync(prompt: str, **kw) -> TeamResult:
    return asyncio.run(run(prompt, **kw))


def mesh_cap() -> int:
    """How many STLs the payload may carry before the tail is merged.

    This used to be a hard 28, which silently folded every part past the 28th into one
    blob named "3 smaller parts" — a real complexity cap on the vocabulary, and the
    kind of limit that shows up as a viewer that is missing parts rather than as an
    error. It is now generous by default and overridable for a machine with a lot of
    parts, and `cad_v3_bridge` reads it instead of a constant.
    """
    try:
        return max(1, int(os.getenv("ADA_CAD_MAX_MESHES", "96")))
    except (TypeError, ValueError):
        return 96
