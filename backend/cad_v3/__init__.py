"""ADA CAD v3 — the Assembly Engine.

Architecture in one paragraph: the brain emits a typed SPEC that contains no
coordinates. A closed vocabulary of deterministic skills turns each spec line
into real B-rep geometry with named connector frames. The assembler mates
connectors using build123d's Joint algebra, so every pose is *derived*, never
guessed. Verification runs on the solids (boolean clash, watertight,
connectivity), and failures come back as typed spec patches — never as raw code.

Nothing in this package asks an LLM for geometry. It does ask an LLM for *new
vocabulary* when a spec names a part that does not exist yet (synth.py), but a
synthesised skill only becomes vocabulary after the geometry gate admits it, and
what it produces afterwards is deterministic like everything else.
"""

from __future__ import annotations

__version__ = "3.1.0"

VOCABULARY_READY = False
GENERATED_SKILLS: list[str] = []


def bootstrap() -> int:
    """Import every vocabulary module so the registry is populated.

    Curated skills first, then anything ADA has synthesised for herself. The
    order matters: `synth.install` refuses to overwrite a name that is already
    registered, so the hand-written vocabulary always wins a collision.
    """
    global VOCABULARY_READY, GENERATED_SKILLS
    from . import vocabulary  # noqa: F401  (import side-effect registers skills)
    from .registry import REGISTRY

    from . import synth

    GENERATED_SKILLS = synth.load_generated()
    VOCABULARY_READY = True
    return len(REGISTRY)
