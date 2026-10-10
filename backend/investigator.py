"""Investigator — Ada's second night loop (the self-model drive).

SkillLoop learns SKILLS (what she can do).
DreamRunner consolidates FACTS (what happened).
Investigator learns SELF (who she is becoming).

Weekly (or on demand), it:
  1. READS — soul files + recent memory + diary + corrections
  2. NOTICES — cross-session patterns ("Sir corrects me when X",
     "I am bad at Y, good at Z", "3 of 5 sessions asked about D")
  3. PROPOSES — exactly ONE persona/human edit, staged via
     soul.propose_identity_edit(). NEVER applies directly.
     Sir approves by voice. No silent self-rewrites, ever.

This is the Letta #1676 finding: reflection consolidates user facts
but never updates persona. Without this loop she learns about YOU
but never about HERSELF — no character forms.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Investigator:
    MIN_INTERVAL_S = 7 * 24 * 3600

    def __init__(self, kernel, chat_fn, state_dir, soul=None):
        self.kernel = kernel
        self.chat_fn = chat_fn  # async: prompt -> deep-model text
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "soul" / "investigator_state.json"
        self.soul = soul
        self._state = self._load()

    def _load(self):
        try:
            if self.state_path.exists():
                return json.loads(self.state_path.read_text())
        except Exception:
            pass
        return {"last_rep_ts": 0, "reps": 0}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=1))
        except Exception as e:
            print(f"[INVESTIGATOR] state save failed: {e}")

    def due(self) -> bool:
        return (time.time() - float(self._state.get("last_rep_ts", 0))) > self.MIN_INTERVAL_S

    def status(self) -> str:
        last = float(self._state.get("last_rep_ts", 0))
        when = time.strftime("%d %b", time.localtime(last)) if last else "never"
        return (f"Investigator reps: {int(self._state.get('reps', 0))}, "
                f"last: {when}.")

    async def rep_now(self, force: bool = False) -> str:
        """Read across sessions, notice ONE pattern, propose ONE edit."""
        if not force and not self.due():
            return "Self-image is fresh — no investigation due."
        soul = self.soul
        if soul is None:
            return "Soul not wired yet."
        persona = soul.persona()[:1200]
        human = soul.human()[:1500]
        now = soul.now()[:1000]
        diary = soul.latest_diary()[:800]
        # Recent corrections + identity facts from DB as evidence
        evidence = ""
        try:
            mem = self.kernel.memory
            changed = ""
            try:
                changed = mem.memory.changed_block(limit=5) if hasattr(mem, "memory") else ""
            except Exception:
                changed = ""
            evidence = (changed or "")[:1200]
        except Exception:
            pass
        prompt = (
            "You are Ada's investigator — her forward-looking self-model, not a summariser.\n"
            f"WHO SHE IS:\n{persona}\n\nWHO SIR IS:\n{human}\n\n"
            f"WHAT'S NOW:\n{now}\n\nLAST DIARY:\n{diary}\n\n"
            f"RECENT CHANGES:\n{evidence}\n\n"
            "Across ALL of the above, notice ONE cross-session pattern about "
            "Ada herself or Sir that is durable (a strength, a repeated mistake, "
            "a preference Sir keeps correcting, a working-style truth).\n"
            "Skip one-offs. Skip flattery. If nothing durable, reply exactly: NOTHING DURABLE.\n"
            "Else reply EXACTLY:\n"
            "TARGET: self|sip\n".replace("sip", "sir") +
            "KEY: <snake_case>\nVALUE: <one sentence, plain>\nWHY: <one sentence evidence>")
        try:
            raw = await self.chat_fn(prompt)
        except Exception as e:
            return f"Investigation failed thinking: {type(e).__name__}"
        raw = (raw or "").strip()
        if not raw or "NOTHING DURABLE" in raw.upper():
            self._mark_done()
            return "Investigation complete — nothing durable. Self unchanged."
        target, key, val = self._parse(raw)
        if not key or not val:
            return "Investigation found no usable proposal."
        if target not in ("self", "sir"):
            target = "self"
        msg = soul.propose_identity_edit(target, key, val)
        self._mark_done()
        return (f"Investigator noticed one pattern — staged for Sir's voice approval:\n"
                f"{target}/{key}: {val}\n{msg}")

    def _parse(self, raw: str):
        target, key, val = "", "", ""
        for line in raw.splitlines():
            s = line.strip()
            if s.upper().startswith("TARGET:"):
                target = s[7:].strip().lower()
            elif s.upper().startswith("KEY:"):
                key = s[4:].strip().strip("`")[:60]
            elif s.upper().startswith("VALUE:"):
                val = s[6:].strip()[:300]
        key = "_".join(key.lower().split())
        return target, key, val

    def _mark_done(self):
        self._state["last_rep_ts"] = time.time()
        self._state["reps"] = int(self._state.get("reps", 0)) + 1
        self._save()
