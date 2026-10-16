"""Skill loop — Ada's Voyager drive (automatic curriculum + skill library).

Each night (or on demand) she does ONE rep:
  1. PROPOSE — one small skill just beyond what she can already do, informed
     by her tool list and her recent failures (no repeats of dead ends).
  2. ATTEMPT — write the code through the ToolForge (AST-gated sandbox).
  3. VERIFY — run it against a self-proposed test call; non-error wins.
  4. COMMIT (eval-gated, Feature 6) — the skill climbs the ladder before it
     touches live traffic:
       L0 deterministic (schema, AST gate, timeout, idempotency — 100%),
       L1 sandboxed behavior trials (TEST_CALL + generated edge inputs,
            no-network subprocess),
       L2 the `@pytest.mark.skill_eval` slice of tests/eval_ada.py,
     then registers as `candidate` and auto-promotes to `shadow`.
     Shadow runs alongside on live calls (ADA_SKILL_SHADOW=1, logged,
     never shown); canary takes a fraction (ADA_SKILL_CANARY=<name>);
     high-blast-radius skills need a SPOKEN yes before `active` — the
     approval queue lives in proposals_pending.json and is surfaced BY
     VOICE ("I forged X, tested it, want it?"), never a popup.

Skills compound: every promoted skill is callable forever via run_custom_tool,
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
        self._eval_gate = None

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

    # ── eval gate (Feature 6) ─────────────────────────────────────────────────
    @property
    def eval_gate(self):
        """The SkillEvalGate for this loop (registry + ladder + serving).

        Lazily built so SkillLoop stays cheap to construct in tests; the
        forge is attached from the kernel when available.
        """
        if self._eval_gate is None:
            from skill_eval import SkillEvalGate
            tools_root = self._tools_root()
            gate = SkillEvalGate(self.state_dir, tools_root)
            forge = getattr(self.kernel, "forge", None)
            if forge is not None:
                gate.attach_forge(forge)
            try:
                from observe import ObserveTiers
                gate.attach_observe(ObserveTiers(self.state_dir))
            except Exception:
                pass
            self._eval_gate = gate
        return self._eval_gate

    def _tools_root(self) -> Path:
        forge = getattr(self.kernel, "forge", None)
        mcp_dir = getattr(forge, "mcp_dir", None)
        if mcp_dir:
            return Path(mcp_dir)
        tools_dir = getattr(forge, "tools_dir", None)
        if tools_dir:
            return Path(tools_dir)
        return self.state_dir / "mcp_tools"

    # ── voice approval queue (surfaced BY VOICE, no popups) ───────────────────
    def propose_for_voice(self, name: str, version: str, summary: str) -> str:
        """Enqueue an approval; returns the line the brain should SPEAK."""
        return self.eval_gate.voice.propose_for_voice(name, version, summary)

    def next_voice_proposal(self) -> dict | None:
        """The pending item for the brain to ask about out loud."""
        return self.eval_gate.voice.next_voice_proposal()

    def voice_proposal_pending(self) -> str:
        """One spoken line describing the pending proposal, or ''."""
        item = self.next_voice_proposal()
        if not item:
            return ""
        return (f"I forged {item['name']}, version {item['version']} — "
                f"{item.get('summary', 'a new skill')}. "
                f"Say 'approve {item['name']}' to make it live, or "
                f"'reject {item['name']}' to retire it.")

    def resolve_voice_proposal(self, name: str, approved: bool) -> str:
        """Apply Sir's spoken verdict: activate or retire the version."""
        item = self.next_voice_proposal()
        target = None
        if item and item.get("name") == name:
            target = item
        else:
            # find any pending for this name (not just the head)
            for i in self.eval_gate.voice._load():
                if i.get("name") == name and i.get("status") == "pending":
                    target = i
                    break
        if not target:
            return f"No pending proposal for '{name}'."
        spoken = self.eval_gate.voice.resolve_voice_proposal(name, approved)
        gate = self.eval_gate
        if approved:
            r = gate._go_active(name, target["version"],
                                reason="voice approval (spoken yes)",
                                via_voice=True)
            if r.get("ok"):
                return spoken + " It's live now."
            return f"Approval noted, but activation failed: {r.get('error')}"
        gate.registry.set_status(name, target["version"], "retired",
                                 reason="voice rejection (spoken no)")
        return spoken

    # ── one rep ───────────────────────────────────────────────────────────────
    async def rep_now(self, force: bool = False) -> str:
        """Propose -> attempt -> verify -> eval-gated commit.

        Each finished step is checkpointed (runtime/checkpoints.py, SQLite) so
        a crash mid-rep resumes where it stopped instead of redoing work —
        e.g. a dead process after synthesis does not re-propose or
        re-synthesise on the next run.
        """
        if not force and not self.due():
            return "Practiced recently — skills are fresh."
        # Durable checkpoints for this long task (F1 runtime). Function-level
        # import keeps skill_loop importable even if runtime/ is mid-edit.
        from datetime import datetime as _dt
        from runtime.checkpoints import checkpoint as _ckpt, resume as _resume, clear as _clear
        task_id = f"skill-rep-{_dt.now().strftime('%Y-%m-%d')}"
        ckpt_db = str(self.state_dir / "skill_checkpoints.db")
        done = _resume(task_id, ckpt_db)
        forge = getattr(self.kernel, "forge", None)
        if forge is None:
            _clear(task_id, ckpt_db)
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
        # ── step 1: PROPOSE ──────────────────────────────────────────────
        prop = done.get("proposal") or {}
        if {"name", "purpose", "test_call", "code"} <= set(prop):
            name, purpose, test_call, code = (prop["name"], prop["purpose"],
                                              prop["test_call"], prop["code"])
        else:
            try:
                raw = await self.chat_fn(proposal_prompt)
            except Exception as e:
                _clear(task_id, ckpt_db)
                return f"Practice failed proposing: {type(e).__name__}"
            name, purpose, test_call, code = self._parse(raw or "")
            if not name or not code:
                _clear(task_id, ckpt_db)
                return "Practice failed: coach gave no usable skill."
            _ckpt(task_id, "proposal",
                  {"name": name, "purpose": purpose,
                   "test_call": test_call, "code": code}, ckpt_db)
        if name in have:
            _clear(task_id, ckpt_db)
            return f"Practice skipped: '{name}' already known."
        # ── step 2: ATTEMPT (synthesise) ─────────────────────────────────
        synth = done.get("synthesised") or {}
        if synth.get("ok") and synth.get("name") == name:
            made = synth.get("made", "")
        else:
            made = forge.synthesise(name, code)
            if "created" not in made:
                self._note_avoid(name, f"rejected: {made[:80]}")
                _clear(task_id, ckpt_db)
                return f"Practice rep failed at synthesis: {made}"
            _ckpt(task_id, "synthesised",
                  {"ok": True, "name": name, "made": made[:200]}, ckpt_db)
        # ── step 3: VERIFY ───────────────────────────────────────────────
        ver = done.get("verified") or {}
        if ver.get("name") == name and "result" in ver:
            result, kwargs = ver["result"], ver.get("kwargs", {})
        else:
            try:
                kwargs = dict(eval(test_call, {"__builtins__": {}})) if test_call else {}
            except Exception:
                kwargs = {}
            try:
                import inspect as _inspect
                fn = None
                try:
                    fn = forge.registry[name]          # legacy path
                except Exception:
                    fn = (forge.mcp_registry.get(name) or {}).get("fn")
                if fn is not None:
                    sig = _inspect.signature(fn)
                    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            except Exception:
                pass
            result = await forge.run(name, kwargs, timeout=20)
            if isinstance(result, str) and result.startswith(f"Tool '{name}'"):
                self._note_avoid(name, f"verify failed: {result[:80]}")
                _clear(task_id, ckpt_db)
                return f"Practice rep failed verification: {result}"
            _ckpt(task_id, "verified",
                  {"name": name, "kwargs": kwargs,
                   "result": str(result)[:500]}, ckpt_db)
        # ── step 4: COMMIT (eval-gated) ──────────────────────────────────
        module_path = self._module_path(forge, name)
        gate = self.eval_gate
        verdict = gate.evaluate_candidate(
            name, purpose, code, kwargs, module_path,
            repo_root=self._repo_root())
        _ckpt(task_id, "eval_verdict",
              {"name": name, "verdict": str(verdict)[:500]}, ckpt_db)
        if verdict.get("status") == "blocked":
            stage = verdict.get("stage")
            self._note_avoid(name, f"eval blocked at {stage}: "
                             f"{verdict.get('failed')}")
            _clear(task_id, ckpt_db)
            return (f"Practice rep blocked at eval {stage}: "
                    f"{verdict.get('failed')}. '{name}' stays out of the "
                    f"live set.")
        version = verdict["version"]
        reg[name] = {"purpose": purpose or "coach purpose",
                     "verified_ts": time.time(), "test": str(kwargs)[:200],
                     "result": str(result)[:200],
                     "eval_status": verdict["status"],
                     "version": str(version),
                     "blast_radius": verdict.get("blast_radius", "low")}
        self._save_registry(reg)
        self._state["last_rep_ts"] = time.time()
        self._state["reps"] = int(self._state.get("reps", 0)) + 1
        verified = self._state.get("verified", [])
        if name not in verified:
            verified.append(name)
        self._state["verified"] = verified
        self._save()
        _clear(task_id, ckpt_db)
        msg = (f"Practice rep complete — skill '{name}' v{version} verified "
               f"and committed (eval-gated, now in shadow) ({result}).")
        if verdict.get("voice_proposal"):
            msg += f" Voice approval needed: {verdict['voice_proposal']}"
        return msg

    def _repo_root(self) -> Path:
        """Repo root for the eval subset: walk up from state_dir until
        tests/eval_ada.py is found (production: state_dir is
        <root>/ada_kernel, so this lands on <root>)."""
        p = self.state_dir
        for _ in range(6):
            if (p / "tests" / "eval_ada.py").exists():
                return p
            p = p.parent
        return self.state_dir.parent

    def _module_path(self, forge, name: str) -> Path:
        mcp_dir = getattr(forge, "mcp_dir", None)
        if mcp_dir:
            p = Path(mcp_dir) / f"{name}.py"
            if p.exists():
                return p
        tools_dir = getattr(forge, "tools_dir", None)
        if tools_dir:
            return Path(tools_dir) / f"{name}.py"
        return self.state_dir / f"{name}.py"

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
