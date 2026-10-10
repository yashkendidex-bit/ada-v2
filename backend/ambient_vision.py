"""
Ambient Vision — JARVIS sees the lab (Avengers).

A quiet background loop: every ~30s it looks at the latest camera frame and
distills ONE line of scene awareness — who is present, what they're holding,
what changed. The rolling scene state is injected into Ada's context every
turn, so she simply *knows the room* — and when something notable appears
(a second person, you holding an object, a big change), she can mention it
once, naturally, only while you're idle.

Privacy doctrine: frames never leave the vision provider un-asked-for beyond
this awareness loop; nothing is stored but the one-line description.
"""
import time
import asyncio
import os

# Every look is a vision-model call (~30-90s of thinking on Glimmer) plus a
# camera frame through the loop. 30s cadence = ~3k calls/day and constant CPU
# churn for a background sense — 90s keeps the room-awareness while cutting
# quota burn and heat by 3x. Override: ADA_AMBIENT_INTERVAL_S.
LOOK_INTERVAL_S = float(os.getenv("ADA_AMBIENT_INTERVAL_S", "90"))
MAX_SCENE_FACTS = 6

AWARENESS_PROMPT = """You are Ada's ambient eyes. One line, max 25 words.
Describe ONLY what is visually notable in this frame: who/what is present,
what they are doing or holding, significant changes. If the frame shows
nothing notable or is dark/empty, reply exactly: CLEAR.
No preamble, no markdown — just the line or CLEAR."""


class AmbientVision:
    def __init__(self, get_frame_fn, vision_fn, speak_fn,
                 interval_s=LOOK_INTERVAL_S):
        self.get_frame_fn = get_frame_fn      # -> {"mime_type","data"(b64)} | None
        self.vision_fn = vision_fn            # async(messages) -> str
        self.speak_fn = speak_fn              # async(line) -> bool
        self.interval_s = interval_s
        self.scene_desc = ""
        self.scene_facts = []                 # rolling notable facts
        self._last_desc = ""
        self._task = None
        self._stop = asyncio.Event()
        self._paused = False   # heavy work (CAD/web/autopilot) pauses looks

    def pause(self):
        """Suspends looks during heavy work — vision calls + camera churn stop."""
        self._paused = True

    def resume(self):
        """Resumes looks after heavy work."""
        self._paused = False

    # ── one look ──────────────────────────────────────────────────────────────
    async def look(self) -> str:
        frame = self.get_frame_fn()
        if not frame or not frame.get("data"):
            return ""
        content = [{"type": "text", "text": AWARENESS_PROMPT},
                   {"type": "image_url",
                    "image_url": {"url": f"data:{frame.get('mime_type', 'image/jpeg')};base64,{frame['data']}"}}]
        messages = [{"role": "user", "content": content}]
        try:
            desc = (await self.vision_fn(messages) or "").strip()
        except Exception as e:
            print(f"[VISION] look failed: {str(e)[:80]}")
            return ""
        if not desc or desc.upper().startswith("CLEAR"):
            return ""
        return desc[:200]

    def _absorb(self, desc: str):
        """Returns True if this is NEW information worth a spoken mention."""
        if not desc:
            return False
        low = desc.lower()
        # near-duplicate of last observation -> just refresh, don't speak
        if low[:60] == self._last_desc.lower()[:60]:
            self.scene_desc = desc
            return False
        is_first = not self.scene_desc
        self._last_desc = desc
        self.scene_desc = desc
        stamp = time.strftime("%H:%M")
        fact = f"[{stamp}] {desc}"
        if fact not in self.scene_facts:
            self.scene_facts.append(fact)
            if len(self.scene_facts) > MAX_SCENE_FACTS:
                self.scene_facts.pop(0)
        return not is_first   # don't narrate the very first look — she just learns the room

    def context_block(self) -> str:
        """Injected into Ada's system prompt every turn."""
        if not self.scene_desc and not self.scene_facts:
            return ""
        lines = [f"CURRENT SCENE: {self.scene_desc}."]
        if self.scene_facts[:-1]:
            lines.append("RECENTLY SEEN: " + " | ".join(self.scene_facts[-4:-1]))
        lines.append("(You see this naturally — reference it only when relevant.)")
        return "AMBIENT VISION:\n" + "\n".join(lines)

    # ── loop ──────────────────────────────────────────────────────────────────
    async def _loop(self):
        await asyncio.sleep(8)   # let the camera warm up
        while not self._stop.is_set():
            try:
                if not self._paused:
                    desc = await self.look()
                    if desc and self._absorb(desc):
                        await self.speak_fn(
                            f"[emotion:curious] {desc} — I don't think I've seen that before, Sir.")
            except Exception as e:
                print(f"[VISION] loop error: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass

    def start(self):
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
