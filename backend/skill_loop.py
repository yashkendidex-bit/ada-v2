"""Skill loop — Ada's Voyager drive (automatic curriculum + skill library).

Each night (or on demand) she does ONE rep:
  1. PROPOSE — one small skill just beyond what she can already do, informed
     by her tool list and her recent failures (no repeats of dead ends).
  2. ATTEMPT — write the code through the ToolForge (AST-gated sandbox).
  3. VERIFY — run it against a self-proposed test call; non-error wins.
  4. COMMIT — verified skills enter skills.json with purpose + test; failures
     go to the notebook so tomorrow's proposal routes around them.

Skills compound: every verified skill is callable forever via run_custom_tool,
which is what kills catastrophic forgetting — she never re-learns, she reuses.
"""
import json
import time
from pathlib import Path


class SkillLoop:
    MIN_INTERVAL_S = 24 * 3600

    def __init__(self, kernel, chat_fn, state_dir):
        self.kernel = kernel
        self.chat_fn = chat_fn      # async: prompt -> deep-model text
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "skill_loop_state.json"
        self.registry_path = self.state_dir / "skills.json"
        self._state = self._load()

    # ── state ─────────────────────────────────────────────────────────────────
    def _load(self):
        try:
            if self.state_path.exists():
                return json.loads(self.state_path.read_text())
        except Exception:
            pass
        return {"last_rep_ts": 0, "reps": 0, "verified": []}

    def _save(self):
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=1))
        except Exception as e:
            print(f"[SKILL] state save failed: {e}")

    def due(self) -> bool:
        return (time.time() - float(self._state.get("last_rep_ts", 0))) > self.MIN_INTERVAL_S

    def status(self) -> str:
        return (f"Skill reps: {int(self._state.get('reps', 0))}, verified: "
                f"{len(self._state.get('verified', []))} "
                f"({', '.join(self._state.get('verified', [])[:6])}).")

    def _registry(self) -> dict:
        try:
            if self.registry_path.exists():
                return json.loads(self.registry_path.read_text())
        except Exception:
            pass
        return {}

    def _save_registry(self, reg: dict):
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.registry_path.write_text(json.dumps(reg, indent=1))
        except Exception as e:
            print(f"[SKILL] registry save failed: {e}")

    # ── one rep ───────────────────────────────────────────────────────────────
    async def rep_now(self, force: bool = False) -> str:
        """Propose -> attempt -> verify -> commit. One rep, bounded."""
        if not force and not self.due():
            return "Practiced recently — skills are fresh."
        forge = getattr(self.kernel, "forge", None)
        if forge is None:
            return "Skill forge not available yet."
        have = forge.list_tools()
        reg = self._registry()
        avoid = self._state.get("avoid", [])[-8:]
        proposal_prompt = (
            "You are Ada's training coach. She already has these custom skills: "
            f"{have or '(none yet)'}. Past dead ends to avoid: {avoid or '(none)'}.\n"
            "Propose ONE small, pure-function skill just beyond her reach, useful "
            "to Sir (unit/format conversion, text shaping, math, date logic, "
            "small estimators — never network, files, or system access).\n"
            "Reply EXACTLY:\nNAME: <snake_case>\nPURPOSE: <one line>\n"
            "TEST_CALL: <python kwargs dict, e.g. {\"km\": 5}>\n"
            "CODE:\n```python\n<only safe imports: math json re datetime time random statistics>\n"
            "def <NAME>(...): ...\n```")
        try:
            raw = await self.chat_fn(proposal_prompt)
        except Exception as e:
            return f"Practice failed proposing: {type(e).__name__}"
        name, purpose, test_call, code = self._parse(raw or "")
        if not name or not code:
            return "Practice failed: coach gave no usable skill."
        if name in have:
            return f"Practice skipped: '{name}' already known."
        made = forge.synthesise(name, code)
        if "created and loaded" not in made:
            self._note_avoid(name, f"rejected: {made[:80]}")
            return f"Practice rep failed at synthesis: {made}"
        try:
            kwargs = dict(eval(test_call, {"__builtins__": {}})) if test_call else {}
        except Exception:
            kwargs = {}
        try:
            import inspect as _inspect
            sig = _inspect.signature(forge.registry[name])
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            pass
        result = await forge.run(name, kwargs, timeout=20)
        if isinstance(result, str) and result.startswith(f"Tool '{name}'"):
            self._note_avoid(name, f"verify failed: {result[:80]}")
            return f"Practice rep failed verification: {result}"
        reg[name] = {"purpose": purpose or "coach purpose",
                     "verified_ts": time.time(), "test": str(kwargs)[:200],
                     "result": str(result)[:200]}
        self._save_registry(reg)
        self._state["last_rep_ts"] = time.time()
        self._state["reps"] = int(self._state.get("reps", 0)) + 1
        verified = self._state.get("verified", [])
        if name not in verified:
            verified.append(name)
        self._state["verified"] = verified
        self._save()
        return f"Practice rep complete — skill '{name}' verified and committed ({result})."

    def _parse(self, raw: str):
        name, purpose, test_call, code = "", "", "", ""
        for line in raw.splitlines():
            s = line.strip()
            if s.upper().startswith("NAME:"):
                name = s[5:].strip().strip("`")
            elif s.upper().startswith("PURPOSE:"):
                purpose = s[8:].strip()[:160]
            elif s.upper().startswith("TEST_CALL:"):
                test_call = s[10:].strip()
        if "```python" in raw:
            code = raw.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            code = raw.split("```", 1)[1].split("```", 1)[0].strip()
        return name, purpose, test_call, code

    def _note_avoid(self, name: str, why: str):
        avoid = self._state.get("avoid", [])
        avoid.append(f"{name}: {why}")
        self._state["avoid"] = avoid[-12:]
        self._state["last_rep_ts"] = time.time()
        self._save()
