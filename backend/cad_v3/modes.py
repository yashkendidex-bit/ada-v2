"""Three modes: CAD, PHYSICS, SCENE. They must not bleed into each other.

The owner's correction, and the reason this module exists:

    "i dont want her to build scene everytime like the scene building and cad modelling
     both should be different, for example i am working on a gearbox or any other cad
     model i dont want scene there because i am building"

That is a correctness requirement, not a style preference. A scene is a claim about
*presentation*; a CAD model is a claim about *engineering*. Mixing them produces output
that looks authoritative and asserts nothing — and while you are building, a beautifully
lit workbench hides exactly the geometry you are trying to inspect.

| Mode | Trigger | Produces | Never produces |
|---|---|---|---|
| `CAD` | default when building | geometry, validity verdict, mass, dimensions | **any scenery** |
| `PHYSICS` | "test it", "does it work", "clearance" | PASS/FAIL verdicts with evidence | render-for-show |
| `SCENE` | "render", "show me", "make it look" | a lit, materialled image | nothing — it is the endpoint |

`CAD` is the default and the safe one. A request that mentions no mode word gets `CAD`,
because the failure of accidentally rendering scenery while someone is trying to inspect
a bore is worse than the failure of not rendering when they wanted a picture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

from . import physics


class Mode(str, Enum):
    CAD = "cad"
    PHYSICS = "physics"
    SCENE = "scene"


# ── intent detection ─────────────────────────────────────────────────────────
# Ordered by specificity. SCENE and PHYSICS both override the CAD default, so their
# patterns must be words a person would not use while simply describing a part.
_SCENE_PATTERNS = [
    r"\brender\b", r"\bre-?render\b", r"\bscene\b", r"\bscenery\b",
    r"\bshow me\b", r"\bshow it\b", r"\bshowcase\b", r"\bvisuali[sz]e\b",
    r"\bmake it look\b", r"\blook (?:good|nice|real|great)\b",
    r"\bphoto(?:realistic|graph)?\b", r"\bcinematic\b", r"\bbeautiful\b",
    r"\bpresentation\b", r"\bhero shot\b", r"\bturntable\b", r"\banimat(?:e|ion)\b",
    r"\bworkbench\b", r"\bstudio\b", r"\benvironment\b", r"\bwallpaper\b",
    r"\bhologram\b", r"\bholographic\b", r"\bjarvis look\b",
]

_PHYSICS_PATTERNS = [
    r"\btest\b", r"\bsimulat(?:e|ion)\b", r"\bdoes it work\b", r"\bwill it work\b",
    r"\bwill it (?:turn|spin|move|fit|hold|break|fail)\b",
    r"\bclearance\b", r"\binterfer\w*\b", r"\bcollid\w*\b", r"\bclash\b",
    r"\bdoes it (?:move|fit|clear|collide|spin)\b",
    r"\bphysics\b", r"\bstress\b", r"\bvon ?mises\b", r"\bsafety factor\b",
    r"\bfea\b", r"\bdrop test\b", r"\brigid ?body\b", r"\bkinematic\w*\b",
    r"\bcheck (?:it|the model|the assembly|for)\b", r"\bvalidate\b", r"\bverify\b",
    r"\bcan it\b", r"\banything (?:collide|clash|jam)\b", r"\bwill it jam\b",
    r"\bmechanism\b", r"\brange of motion\b", r"\btravel\b",
]

# A CAD verb always wins over an ambiguous noun, so "build me a gearbox that I can test"
# builds first and tests on request rather than silently doing both.
_CAD_PATTERNS = [
    r"\bbuild\b", r"\bdesign\b", r"\bmodel\b", r"\bgenerate\b", r"\bcreate\b",
    r"\bmake me\b", r"\bmake a\b", r"\bmake an\b", r"\bdraw\b", r"\bsketch\b",
    r"\bmodify\b", r"\bchange the\b", r"\bresize\b", r"\bedit\b", r"\biterate\b",
]


@dataclass
class Intent:
    mode: Mode
    reason: str
    matched: list[str] = field(default_factory=list)


def detect(text: str) -> Intent:
    """Which mode does this request want?

    Defaults to CAD. A build verb takes precedence, so "build me a gearbox I can test"
    is a build — testing is then an explicit second request, which is the behaviour the
    owner asked for.
    """
    t = (text or "").lower()

    cad_hits = [p for p in _CAD_PATTERNS if re.search(p, t)]
    phys_hits = [p for p in _PHYSICS_PATTERNS if re.search(p, t)]
    scene_hits = [p for p in _SCENE_PATTERNS if re.search(p, t)]

    if cad_hits and not (phys_hits or scene_hits):
        return Intent(Mode.CAD, "build/design language", cad_hits)
    if scene_hits and not cad_hits:
        return Intent(Mode.SCENE, "presentation language", scene_hits)
    if phys_hits and not cad_hits:
        return Intent(Mode.PHYSICS, "testing language", phys_hits)
    if cad_hits and scene_hits:
        # "build me X and show me a render" — build first; the scene is a follow-up.
        return Intent(Mode.CAD, "build verb outranks presentation", cad_hits + scene_hits)
    if cad_hits and phys_hits:
        return Intent(Mode.CAD, "build verb outranks testing", cad_hits + phys_hits)
    if phys_hits and scene_hits:
        return Intent(Mode.PHYSICS, "testing outranks presentation", phys_hits + scene_hits)

    return Intent(Mode.CAD, "default — building is the safe assumption", [])


# ── Mode 1: CAD ──────────────────────────────────────────────────────────────
@dataclass
class CadOutput:
    """What a build returns. Note what is absent: there is no image and no scene."""
    name: str
    parts: int
    verdicts: list[physics.Verdict] = field(default_factory=list)
    exports: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return physics.overall(self.verdicts)

    def summary(self) -> str:
        """The one-paragraph engineering statement.

        This is the sentence that converts "art" into "an artifact": it says what was
        built, whether it can exist, and what it weighs.
        """
        lines = [f"{self.name} — {self.parts} parts, build {self.status.lower()}."]
        for v in self.verdicts:
            lines.append(f"  • {v.test}: {v.status} — {v.summary}")
        return "\n".join(lines)

    def full(self) -> str:
        blocks = [self.summary()]
        for v in self.verdicts:
            blocks.append("")
            blocks.append(v.render())
        return "\n".join(blocks)


def cad_output(name: str, asm: Any, *, exports: Optional[dict] = None,
               audit: bool = True, sweep: bool = False) -> CadOutput:
    """Mode 1. Geometry plus the evidence that it is real. **No scenery.**

    The validity check is on by default because it is the difference between a model and
    a drawing of one, and it costs about a second on a 57-part assembly.
    """
    insts = physics.instances_of(asm)
    verdicts: list[physics.Verdict] = []
    if audit and insts:
        # Interference needs two bodies to compare; mass properties do not. Gating both
        # on `len >= 2` meant a single-part model reported no evidence at all, which is
        # exactly the "it's ready" output this method exists to replace.
        if len(insts) >= 2:
            verdicts.append(physics.interference_audit(asm))
        verdicts.append(physics.mass_properties(asm))
    if sweep and len(insts) >= 2:
        verdicts.append(physics.clearance_sweep(asm))
    return CadOutput(name=name, parts=len(insts), verdicts=verdicts,
                     exports=exports or {})


# ── Mode 2: PHYSICS ──────────────────────────────────────────────────────────
# Named tests, so "test it" is a real request rather than a vague one.
TESTS = ("interference", "mass", "clearance", "all")


def physics_output(asm: Any, test: str = "all", *, axis: str = "Z",
                   steps: int = 36) -> list[physics.Verdict]:
    """Mode 2. Explicitly invoked, never automatic during a build."""
    test = (test or "all").strip().lower()
    if test in ("interference", "clash", "fit"):
        return [physics.interference_audit(asm)]
    if test in ("mass", "weight", "properties"):
        return [physics.mass_properties(asm)]
    if test in ("clearance", "motion", "sweep", "kinematic", "range"):
        return [physics.clearance_sweep(asm, axis=axis, steps=steps)]
    if test in ("all", "full", ""):
        return [physics.interference_audit(asm),
                physics.mass_properties(asm),
                physics.clearance_sweep(asm, axis=axis, steps=steps)]
    return [physics.Verdict("Physics", "INCONCLUSIVE",
                            f"Unknown test {test!r}. Known tests: {', '.join(TESTS)}.")]


def physics_report(verdicts: Sequence[physics.Verdict], subject: str = "model") -> str:
    overall = physics.overall(verdicts)
    head = f"{subject} — physics: {overall}"
    body = [v.render() for v in verdicts]
    return "\n".join([head, "=" * 78, ""] + body)


# ── Mode 3: SCENE ────────────────────────────────────────────────────────────
# Named looks, so a scene request can be specific without the caller knowing Blender.
STYLES = ("studio", "technical", "dramatic", "workshop", "holographic", "outdoor")


@dataclass
class SceneRequest:
    style: str = "studio"
    shot: str = "three_quarter"     # three_quarter | front | top | macro | hero
    resolution: tuple[int, int] = (1600, 1000)
    samples: int = 96
    transparent: bool = False

    def normalised(self) -> "SceneRequest":
        style = (self.style or "studio").strip().lower()
        if style not in STYLES:
            style = "studio"
        return SceneRequest(style=style, shot=self.shot,
                            resolution=self.resolution, samples=self.samples,
                            transparent=self.transparent)


def scene_style_from_text(text: str) -> str:
    """Pick a look from the request. Cheap, and it makes 'make it look dramatic' work."""
    t = (text or "").lower()
    for style, words in (
        ("holographic", ("hologram", "holographic", "jarvis", "glow", "scan")),
        ("workshop", ("workbench", "workshop", "garage", "bench", "shop")),
        ("dramatic", ("dramatic", "cinematic", "moody", "dark", "hero")),
        ("technical", ("technical", "drawing", "blueprint", "orthographic", "cad view")),
        ("outdoor", ("outdoor", "outside", "sun", "field", "sky")),
        ("studio", ("studio", "clean", "white", "product", "presentation")),
    ):
        if any(w in t for w in words):
            return style
    return "studio"


def scene_output(asm: Any, request: Optional[SceneRequest] = None,
                 *, out_path: Optional[str] = None) -> str:
    """Mode 3. Renders. Imported lazily so CAD mode never pays for Blender.

    Returns a human-readable status either way — a missing renderer is reported, never
    raised, because a scene is the one thing a build can safely go without.
    """
    from .render import blender

    req = (request or SceneRequest()).normalised()
    if not blender.available():
        return ("Scene mode unavailable: Blender was not found at "
                f"{blender.BLENDER_PATHS[0]}. The model itself is unaffected — "
                "CAD output does not depend on a renderer.")

    result = blender.render_assembly(asm, style=req.style, shot=req.shot,
                                     resolution=req.resolution, samples=req.samples,
                                     transparent=req.transparent, out_path=out_path)
    return result.summary()


# ── routing ──────────────────────────────────────────────────────────────────
def describe(mode: Mode) -> str:
    return {
        Mode.CAD: "CAD — geometry, validity and mass. No scene.",
        Mode.PHYSICS: "Physics — test the model and report a verdict.",
        Mode.SCENE: "Scene — compose and render. Only when asked.",
    }[mode]


# ── loading a previously built model ─────────────────────────────────────────
def find_latest_spec(search_dir: Any) -> Optional[Any]:
    """Newest `design_spec.json` under a project's CAD output, or None.

    The CAD agent writes the spec beside the meshes rather than keeping the assembly in
    memory, so a later physics request has to find it again. The spec is the durable
    artefact — the geometry is reproducible from it, which is the whole point of the
    spec-not-code architecture.
    """
    from pathlib import Path

    root = Path(search_dir)
    if not root.exists():
        return None
    candidates = list(root.rglob("design_spec.json"))
    if not candidates:
        # Some export paths keep the spec at the top level under a different name.
        candidates = [p for p in root.rglob("*.json")
                      if "spec" in p.name.lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_assembly(spec_path: Any):
    """Rebuild an Assembly from a saved spec.

    Costs about 17 seconds on a 57-part model. That is why physics mode is an explicit
    request rather than something that runs on every build — a validity check is worth a
    second, a full rebuild is worth asking for.
    """
    import cad_v3
    from . import assembler
    from .cli import _load

    cad_v3.bootstrap()
    spec = _load(str(spec_path))
    return spec, assembler.build(spec)


def assembly_for(project_dir: Any):
    """The assembly to test or render.

    Prefers the one already built in this process — the caller has just watched it be
    built, and rebuilding a 57-part model costs ~17 seconds. Falls back to the newest
    saved spec so a request after a restart still works.
    """
    from . import assembler

    spec, asm = assembler.last_built()
    if asm is not None and getattr(asm, "instances", None):
        return spec, asm

    spec_path = find_latest_spec(project_dir)
    if spec_path is None:
        return None, None
    try:
        return load_assembly(spec_path)
    except Exception:
        return None, None
