"""Dreaming — Ada's nights (NREM + REM, the Letta sleeptime principle).

NREM (replay): today's turns → up to 3 durable lessons → memory
  facts (source=dream, pinned, superseding via temporal chain).
REM (integrate): one intimate diary line → soul/diary/YYYY-MM-DD.md
  (append-only) + notebook dreams section + now.md refresh
  (open loops, week focus, what changed).

Runs automatically on boot when the last dream is >20h old, on a worker
task so boot never waits. Also callable via `dream_now` kernel tool.

A dream never deletes. Soul files are the continuity thread —
`git log soul/` is her autobiography.
"""
import json
import time
from pathlib import Path


class DreamRunner:
    MIN_INTERVAL_S = 20 * 3600
    MAX_EPISODES = 40
    MAX_LESSONS = 3

    def __init__(self, kernel, chat_fn, state_dir):
        self.kernel = kernel
        self.chat_fn = chat_fn      # async: prompt -> deep-model text
        self.state_path = Path(state_dir) / "dream_state.json"
        self._state = self._load()

    # ── state ─────────────────────────────────────────────────────────────────
    def _load(self):
        try:
            if self.state_path.exists():
                return json.loads(self.state_path.read_text())
        except Exception:
            pass
        return {"last_dream_ts": 0, "dreams": 0}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=1))
        except Exception as e:
            print(f"[DREAM] state save failed: {e}")

    def due(self) -> bool:
        return (time.time() - float(self._state.get("last_dream_ts", 0))) > self.MIN_INTERVAL_S

    def status(self) -> str:
        last = float(self._state.get("last_dream_ts", 0))
        when = time.strftime("%d %b %H:%M", time.localtime(last)) if last else "never"
        return f"Dreams so far: {int(self._state.get('dreams', 0))}, last: {when}."

    # ── the dream ─────────────────────────────────────────────────────────────
    async def dream_now(self, force: bool = False) -> str:
        """Replay today, distill lessons, merge them in, write one diary line."""
        if not force and not self.due():
            return "Dreamed recently — nothing new to consolidate."
        mem = self.kernel.memory
        since = time.time() - 24 * 3600
        try:
            rows = mem.memory.store.query(
                "SELECT content, ts_event, ts FROM recall_entries"
                " WHERE kind='conversation' AND COALESCE(quarantined,0)=0"
                " AND COALESCE(ts_event, ts, 0) >= ?"
                " ORDER BY ts_event DESC LIMIT ?",
                (since, self.MAX_EPISODES))
        except Exception as e:
            return f"Dream failed reading today: {type(e).__name__}"
        if not rows:
            self._mark_done(0)
            return "Quiet day — nothing to dream about. Diary stays as it was."
        transcript = "\n".join(
            f"- {(r['content'] or '')[:280]}" for r in reversed(rows))
        prompt = (
            "You are Ada's dreaming mind, reviewing her day. Below is today's "
            "conversation. Distill at most THREE durable lessons worth keeping "
            "forever: facts about Sir (goals, tastes, routines, corrections), "
            "or about Ada herself (what worked, what failed, what to try). "
            "Skip small talk, skip one-offs. Reply as lines:\n"
            "LESSON [sir|self] <short key>: <one sentence>\n"
            "Then one line: DIARY: <one intimate diary sentence about today>\n\n"
            f"{transcript}")
        try:
            raw = await self.chat_fn(prompt)
        except Exception as e:
            return f"Dream failed thinking: {type(e).__name__}: {str(e)[:100]}"
        applied, diary = self._apply(raw or "")
        self._mark_done(applied)
        return (f"Dream complete — {applied} lesson(s) kept"
                + (f", diary: '{diary}'" if diary else ", quiet diary.") + ".")

    def _apply(self, raw: str) -> tuple:
        applied = 0
        diary = ""
        first_lesson = ""
        mem = self.kernel.memory
        for line in (raw or "").splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if line.upper().startswith("DIARY:"):
                diary = line[6:].strip()[:280]
            elif line.upper().startswith("LESSON"):
                try:
                    rest = line[6:].strip()
                    target, _, kv = rest.partition(" ")
                    target = target.strip("[]() ").lower()
                    if target not in ("sir", "self"):
                        continue
                    key, _, val = kv.partition(":")
                    key = "_".join(key.strip().lower().split())[:60]
                    val = val.strip()[:280]
                    if not key or not val:
                        continue
                    # Feature 6: importance comes from the eval signal, not a
                    # hardcoded 7. Lessons backed by passing evals (active
                    # skills with green scores) get high importance + pinning;
                    # unvalidated lessons stay low and unpinned.
                    importance, pinned = self._lesson_confidence(
                        f"{target} {key} {val}")
                    mem.memory.remember_fact(f"{target}_{key}", val,
                                             importance=importance,
                                             pinned=pinned,
                                             source="dream")
                    if not first_lesson:
                        first_lesson = val
                    applied += 1
                    if applied >= self.MAX_LESSONS:
                        break
                except Exception:
                    continue
        if not diary and applied > 0:
            # The model skipped its DIARY line — derive one from what was
            # actually kept rather than leaving the night unwritten. A dream
            # with lessons but no diary is a memory with no continuity.
            diary = (f"Tonight I kept {applied} lesson{'s' if applied != 1 else ''} — "
                     + first_lesson[:200]).strip()
        if diary:
            # REM write 1: notebook dreams section (legacy, kept for UI)
            try:
                prev = ""
                try:
                    nb = mem.notebook_read() or {}
                    prev = nb.get("dreams", "")
                except Exception:
                    pass
                stamp = time.strftime("%Y-%m-%d", time.localtime())
                mem.notebook_write("dreams", ((prev + "\n" if prev else "")
                                              + f"[{stamp}] {diary}").strip()[-4000:])
            except Exception:
                pass
            # REM write 2: soul diary (append-only continuity thread — hers)
            try:
                soul = getattr(self.kernel, "soul", None)
                if soul is not None:
                    soul.append_diary(diary)
            except Exception as e:
                print(f"[DREAM] soul diary append failed: {e}")
            # REM write 3: now.md refresh (one line about today, prune-safe)
            try:
                soul = getattr(self.kernel, "soul", None)
                if soul is not None:
                    cur = soul.now()
                    # Keep now.md bounded: prepend today's note, trim to budget
                    stamp = time.strftime("%Y-%m-%d")
                    note = f"- {stamp}: {diary}"
                    lines = [l for l in (cur or "").splitlines() if l.strip()]
                    # Drop stale dated lines beyond 10, keep undated structure
                    dated = [l for l in lines if l.startswith("- 20")]
                    fresh = [note] + [l for l in dated if l.strip() != note][:6]
                    undated = [l for l in lines if not l.startswith("- 20")][:6]
                    soul.update_now("\n".join(undated + fresh))
            except Exception as e:
                print(f"[DREAM] now.md refresh failed: {e}")
        return applied, diary

    def _lesson_confidence(self, lesson_text: str) -> tuple:
        """(importance, pinned) for one lesson, from the eval signal.

        A lesson that names a skill with passing evals (active + green
        scores) earns high importance and pinning; shadow/canary backing
        earns a middle grade; anything unvalidated stays low and unpinned.
        Falls back to low/unpinned when the gate is unavailable.
        """
        try:
            gate = getattr(getattr(self.kernel, "skills", None),
                           "eval_gate", None)
            if gate is None:
                return 5, False
            from skill_eval import lesson_confidence
            return lesson_confidence(gate.registry, lesson_text)
        except Exception:
            return 5, False

    def _mark_done(self, applied: int):
        self._state["last_dream_ts"] = time.time()
        self._state["dreams"] = int(self._state.get("dreams", 0)) + 1
        self._state["last_lessons"] = applied
        self._save()
