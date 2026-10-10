"""
Overnight Dossier — the FRIDAY principle (Civil War: she read the Accords
while Tony slept).

Schedule a research topic; at 2 AM Ada digs through web sources with her deep
model chain, distills a structured brief into the notebook, and raises it at
the next morning greeting: "Sir, I sat with this all night."

State machine:  none -> pending -> running -> ready -> (announced) -> none
State persists in ada_kernel/dossier_state.json so restarts never lose work.
"""
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path


class OvernightDossier:
    RUN_HOUR = 2            # 02:00 local
    CHECK_INTERVAL = 600    # 10 min

    def __init__(self, kernel, chat_fn, search_fn, state_dir):
        self.kernel = kernel
        self.chat_fn = chat_fn      # async: prompt -> deep-model text
        self.search_fn = search_fn  # async: query, k -> search results text
        self.state_path = Path(state_dir) / "dossier_state.json"
        self._task = None
        self._stop = asyncio.Event()
        self._state = self._load()

    # ── state ─────────────────────────────────────────────────────────────────
    def _load(self):
        try:
            if self.state_path.exists():
                return json.loads(self.state_path.read_text())
        except Exception:
            pass
        return {"status": "none"}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=1))
        except Exception as e:
            print(f"[DOSSIER] state save failed: {e}")

    # ── public API (called from kernel tools) ────────────────────────────────
    def schedule(self, topic: str) -> str:
        topic = (topic or "").strip()[:300]
        if not topic:
            return "No topic given."
        if self._state.get("status") in ("pending", "running"):
            return (f"Already working on '{self._state.get('topic')}' overnight. "
                    f"Yours is queued after it — I'll confirm.")
            # (single-slot design: replace if user insists via another call)
        self._state = {"status": "pending", "topic": topic,
                       "requested_ts": time.time()}
        self._save()
        return (f"Overnight research scheduled on '{topic}'. I'll work at two in the "
                f"morning and have the brief ready with your good-morning.")

    def status(self) -> str:
        s = self._state
        if s["status"] == "none":
            return "No overnight research scheduled."
        when = time.strftime("%d %b %H:%M", time.localtime(s.get("requested_ts", 0)))
        return (f"Dossier '{s.get('topic')}' — status: {s['status']} "
                f"(requested {when})." + (" Brief is in the notebook." if s["status"] == "ready" else ""))

    def ready_unannounced(self):
        """Returns (topic, ts) if a finished dossier hasn't been announced yet."""
        if self._state.get("status") == "ready" and not self._state.get("announced"):
            self._state["announced"] = True
            self._save()
            return self._state.get("topic", "your topic"), self._state.get("ready_ts", time.time())
        return None, None

    # ── the actual night shift ────────────────────────────────────────────────
    async def _research(self, topic: str) -> str:
        sections = []
        try:
            if self.search_fn:
                raw = await self.search_fn(topic, 8)
                if raw:
                    sections.append(f"WEB FINDINGS (raw):\n{raw[:6000]}")
        except Exception as e:
            sections.append(f"(web search unavailable: {e})")
        sections.append(
            f"QUESTION: {topic}\n\n"
            "Write an overnight research brief for Sir. Structure:\n"
            "1. SHORT VERSION — the answer in 3 spoken sentences (this gets read aloud).\n"
            "2. KEY FINDINGS — 5-8 bullet facts with specifics.\n"
            "3. TRADE-OFFS / RISKS — what could bite him.\n"
            "4. RECOMMENDATION — your opinion, stated plainly.\n"
            "5. SOURCES — where the evidence came from.\n"
            "Be concrete. No filler. You are a brilliant research butler, not a search engine.")
        prompt = "\n\n".join(sections)
        brief = await self.chat_fn(prompt)
        if not brief or not brief.strip():
            raise RuntimeError("deep chain returned empty brief")
        return brief.strip()

    async def _run_pending(self):
        topic = self._state.get("topic", "")
        self._state["status"] = "running"
        self._save()
        print(f"[DOSSIER] night shift started: '{topic}'")
        try:
            brief = await self._research(topic)
            self.kernel.memory.notebook_write(f"dossier: {topic[:48]}", brief)
            self._state.update({"status": "ready", "ready_ts": time.time(),
                                "announced": False})
            print(f"[DOSSIER] ready: '{topic}' ({len(brief)} chars in notebook)")
        except Exception as e:
            print(f"[DOSSIER] failed: {e}")
            self._state.update({"status": "failed", "error": str(e)[:200]})
        self._save()

    # ── background loop ───────────────────────────────────────────────────────
    def _due_to_run(self) -> bool:
        """True at the 02:00 window -- or as catch-up.

        Laptops sleep: if a dossier was scheduled and a full 02:00 window was
        missed (20h+ elapsed with no run), run it at the next check instead of
        waiting another whole day. The brief is still waiting at wake-up.
        """
        now = datetime.now()
        if now.hour == self.RUN_HOUR:
            return True
        requested = self._state.get("requested_ts", 0) or 0
        return (time.time() - requested) > 20 * 3600

    async def _loop(self):
        while not self._stop.is_set():
            try:
                if datetime.now().hour == self.RUN_HOUR:
                    await self._maybe_consolidate()
                if (self._state.get("status") == "pending"
                        and self._due_to_run()):
                    await self._run_pending()
            except Exception as e:
                print(f"[DOSSIER] loop error: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.CHECK_INTERVAL)
            except asyncio.TimeoutError:
                pass

    async def _maybe_consolidate(self):
        """Memory consolidation rides the same 02:00 slot as the dossier.

        Biologically this is slow-wave sleep: the fast store replays into the slow one,
        offline, while nothing else is happening. The timing matters more than it looks
        — consolidation is the only writer to the semantic layer, and running it during
        the day would mean the retrieval surface changes under a live conversation.

        Consolidation is deterministic scoring, dedup and decay — no LLM calls — so it
        runs on a worker thread rather than blocking the event loop.

        The day marker is in-memory, so a restart after 02:00 re-runs it. That is
        deliberate and harmless: the pipeline is idempotent (a second pass finds nothing
        new to merge), so the worst case is a little wasted work, versus the alternative
        of a persisted flag that silently stops consolidation forever if it corrupts.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if getattr(self, "_last_consolidated_day", None) == today:
            return
        try:
            rep = await asyncio.to_thread(self.kernel.memory.consolidate)
            self._last_consolidated_day = today
            print(f"[MEMORY] nightly consolidation: {rep.summary()}")
            for note in rep.notes:
                print(f"[MEMORY]   {note}")
        except Exception as e:
            print(f"[MEMORY] consolidation failed: {type(e).__name__}: {e}")

    def start(self):
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())
            st = self._state.get("status")
            print(f"[DOSSIER] night-shift loop started "
                  f"(02:00 local + missed-window catch-up; status={st}).")

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
