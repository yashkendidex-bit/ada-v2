"""The two ways a live run lost a good spec, pinned so they cannot come back.

Both cases are taken from a real run of `ada_cad_test.py` on 2026-08-25:

  * the Iron Man repulsor died on ONE error — `revolved_profile` publishes no
    connector `bore`, when it publishes `bore_front` — and fell back to the code
    writer, discarding 24 valid parts;
  * the rotor head authored 15 parts and declared three work orders on attempt 1,
    then spent its retry thinking and returned no JSON, which overwrote the good
    draft with a parse error.

Neither is a design failure, so neither may cost an assembly. No spec is authored
here: the fixtures are the shipped turbocharger spec with one field corrupted.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

from .. import bootstrap, brain, validate
from ..schema import Spec

SPEC = Path(__file__).resolve().parents[1] / "specs" / "turbocharger.json"


def _clean() -> dict:
    bootstrap()
    return json.loads(SPEC.read_text())


def _replay(*replies: str):
    """A transport that hands back canned replies in order."""
    queue = list(replies)

    async def chat(_messages):
        return queue.pop(0) if queue else ""

    return chat


def test_autofix_heals_a_near_miss_connector():
    raw = _clean()
    victim = next(m for m in raw["mates"] if m["b"].endswith(".rear_face"))
    victim["b"] = victim["b"][: -len("rear_face")] + "rear"    # nobody publishes 'rear'
    spec = Spec.from_dict(raw)
    before = validate.validate(spec)
    assert any(e.code == "UNKNOWN_CONNECTOR" for e in before), before

    notes = validate.autofix(spec)
    after = validate.validate(spec)
    assert notes, "autofix said nothing"
    assert not any(e.code == "UNKNOWN_CONNECTOR" for e in after), after


def test_the_repulsor_killer_bore_resolves_to_bore_front():
    """The live 2026-08-25 failure, verbatim: `revolved_profile` has no `bore`."""
    from .. import registry

    bootstrap()
    conns = registry.get("revolved_profile").connectors
    assert "bore" not in conns
    assert validate._closest("bore", conns) == "bore_front"



def test_autofix_is_idempotent_on_a_clean_spec():
    spec = Spec.from_dict(_clean())
    assert validate.validate(spec) == []
    assert validate.autofix(spec) == []
    assert validate.validate(spec) == []


def test_autofix_deletes_smuggled_coordinates_and_clamps_range():
    raw = _clean()
    raw["parts"][0]["params"]["x"] = 12.5
    spec = Spec.from_dict(raw)
    notes = validate.autofix(spec)
    assert "x" not in spec.parts[0].params
    assert any("coordinate" in n for n in notes), notes


def test_autofix_repoints_a_missing_datum_at_the_busiest_part():
    raw = _clean()
    raw["datum"] = "no_such_part"
    spec = Spec.from_dict(raw)
    validate.autofix(spec)
    assert spec.datum in set(spec.ids)
    assert validate.validate(spec) == []


def test_a_junk_retry_cannot_destroy_a_good_first_draft():
    """The rotor-head failure: attempt 2 returns 23k chars of thinking, no JSON."""
    raw = _clean()
    raw["mates"][0]["fit"] = "sliding"        # one healable near-miss
    good = json.dumps(raw)
    draft = asyncio.run(brain.author("a turbocharger", chat=_replay(good, "I think..." * 500),
                                     tries=2))
    assert draft.spec is not None
    assert len(draft.spec.parts) == len(raw["parts"])
    assert draft.ok, draft.errors


def test_a_dead_provider_on_the_retry_keeps_the_first_draft():
    raw = _clean()
    raw["parts"][1]["material"] = "steal"     # healable
    good = json.dumps(raw)

    async def chat(messages):
        if len(messages) > 2:
            raise RuntimeError("provider exploded")
        return good

    draft = asyncio.run(brain.author("a turbocharger", chat=chat, tries=2))
    assert draft.spec is not None and draft.ok, draft.errors
    assert draft.spec.parts[1].material == "steel"


def test_only_fatal_codes_block_a_build():
    raw = _clean()
    spec = Spec.from_dict(raw)
    spec.parts[0].skill = "no_such_skill_at_all"
    errs = validate.validate(spec)
    assert brain.blocking(errs), "an unwritable part must block"

    other = Spec.from_dict(copy.deepcopy(raw))
    other.mates = other.mates[:1]             # strands most parts -> FLOATING_PARTS
    errs = validate.validate(other)
    assert errs and not brain.blocking(errs), \
        "a floating part is for verify to report, not a reason to refuse to build"
