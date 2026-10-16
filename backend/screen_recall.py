"""Screen recall — "what was I just doing?"

On demand: screenshot -> vision caption -> answer. Silent watch mode: a
snapshot every few minutes distilled into the notebook, so "what was I doing
all afternoon" is answerable without ever having watched anything live.

Privacy rule, stated plainly: captures happen only on Sir's explicit ask, or
while screen_watch is ON — and he can switch it off any time with the same
tool. Captions are one-line task descriptions, never keystrokes or passwords.
"""
import asyncio
import base64
import time
from pathlib import Path


class ScreenRecall:
    SECTION = "screen_log"
    MAX_ENTRIES = 20

    def __init__(self, kernel, shot_fn, caption_fn):
        self.kernel = kernel
        self.shot_fn = shot_fn        # sync () -> png path
        self.caption_fn = caption_fn  # async (b64_png) -> one-line caption
        self._watch_task = None
        self._watch_stop = asyncio.Event()

    # ── on demand ─────────────────────────────────────────────────────────────
    async def snapshot(self, silent: bool = False) -> str:
        """Capture + caption now. Returns the caption (or an error line)."""
        try:
            path = await asyncio.to_thread(self.shot_fn)
            raw = Path(str(path)).read_bytes()
        except Exception as e:
            return f"Could not capture the screen: {type(e).__name__}"
        try:
            caption = await self.caption_fn(base64.b64encode(raw).decode())
        except Exception as e:
            return f"Screen captured but I couldn't read it: {type(e).__name__}"
        caption = (caption or "").strip()[:300] or "unreadable screen"
        self._log(caption)
        # Timeline (Feature 4): screen event — caption text only, never pixels.
        try:
            from perception import emit as _tline_emit
            _tline_emit("screen", "screen.note", {"caption": caption})
        except Exception:
            pass
        if silent:
            return caption
        return f"Right now: {caption}"

    def _log(self, caption: str):
        try:
            mem = self.kernel.memory
            nb = mem.notebook_read() or {}
            entries = [l for l in (nb.get(self.SECTION, "") or "").splitlines() if l.strip()]
            stamp = time.strftime("%H:%M", time.localtime())
            entries.append(f"[{stamp}] {caption}")
            mem.notebook_write(self.SECTION, "\n".join(entries[-self.MAX_ENTRIES:]))
        except Exception as e:
            print(f"[SCREEN] log failed: {e}")

    def recent(self, n: int = 5) -> str:
        try:
            nb = self.kernel.memory.notebook_read() or {}
            entries = [l for l in (nb.get(self.SECTION, "") or "").splitlines() if l.strip()]
            if not entries:
                return "No screen snapshots yet — ask me to take one, or switch screen_watch on."
            return "Recent screen snapshots:\n" + "\n".join(entries[-n:])
        except Exception as e:
            return f"Could not read the screen log: {e}"

    # ── silent watch ──────────────────────────────────────────────────────────
    @property
    def watching(self) -> bool:
        return self._watch_task is not None and not self._watch_task.done()

    async def set_watch(self, on: bool, interval_s: int = 600) -> str:
        if on and not self.watching:
            self._watch_stop.clear()
            self._watch_task = asyncio.create_task(self._loop(interval_s))
            return (f"Screen watch ON — silent snapshot every {interval_s // 60} minutes, "
                    f"logged only. Say 'screen watch off' any time.")
        if not on and self.watching:
            self._watch_stop.set()
            self._watch_task.cancel()
            self._watch_task = None
            return "Screen watch OFF — no more snapshots."
        return "Screen watch is already ON." if on else "Screen watch is already OFF."

    async def _loop(self, interval_s: int):
        while not self._watch_stop.is_set():
            try:
                await asyncio.wait_for(self._watch_stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass
            if self._watch_stop.is_set():
                return
            try:
                await self.snapshot(silent=True)
            except Exception as e:
                print(f"[SCREEN] watch snapshot failed: {e}")
