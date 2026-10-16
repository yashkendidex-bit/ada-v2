"""SensoryRing — Ada's 60-second present (the missing tier).

Biology: iconic store ~250-500ms, echoic ~2-4s. Everything not
attended in that window was never encoded — never THERE.

Ada analogue: rolling RAM-only buffers, mostly discarded:
  audio  — last 30s of transcribed utterances (text, never raw mic)
  screen — last 60s of screen OCR / captions (text only)
  frames — last 5 camera frame descriptors (one-line each, no pixels)

Nothing persists unless promoted by an explicit cue:
  "what did I just say/do/show?" → answer from ring, instantly.
  "remember this" → promote ring head into memory via remember.

Privacy: RAM only, no DB, no disk. Cleared on demand
("forget the last minute"). Captions are task descriptions,
never keystrokes or passwords — enforced at the caption site,
not here.
"""
from __future__ import annotations

import time
from collections import deque


_RECALL_PATTERNS = (
    "what did i just", "what just happened", "what was i just",
    "say again", "repeat that",
)


def _is_recall_question(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(p in t for p in _RECALL_PATTERNS)


class SensoryRing:
    def __init__(self, audio_s: int = 30, screen_s: int = 60, max_frames: int = 5,
                 timeline=None):
        self.audio_s = audio_s
        self.screen_s = screen_s
        self.timeline = timeline  # optional bridge: ring events also flow into
                                  # the perception timeline (honors tier; tier 0 = ring only)
        self._audio: deque = deque(maxlen=20)   # (ts, text)
        self._screen: deque = deque(maxlen=20)  # (ts, text)
        self._frames: deque = deque(maxlen=max_frames)  # (ts, desc)

    def _forward(self, source: str, type: str, text: str):
        """One-line bridge: timeline.emit is a safe no-op when unset or tier 0."""
        tl = self.timeline
        if tl is not None:
            try:
                tl.emit(source, type, {"text": text})
            except Exception:
                pass

    # ── pushes (called by input paths, never blocking) ──────────────────
    def push_audio(self, text: str):
        text = (text or "").strip()[:300]
        if text:
            self._audio.append((time.time(), text))
            self._forward("sensor", "ring.audio", text)

    def push_screen(self, text: str):
        text = (text or "").strip()[:300]
        if text:
            self._screen.append((time.time(), text))
            self._forward("sensor", "ring.screen", text)

    def push_frame(self, desc: str):
        desc = (desc or "").strip()[:200]
        if desc:
            self._frames.append((time.time(), desc))
            self._forward("sensor", "ring.frame", desc)

    # ── reads ───────────────────────────────────────────────────────────
    def _fresh(self, buf: deque, window_s: int) -> list[str]:
        now = time.time()
        return [t for ts, t in list(buf) if now - ts <= window_s]

    def snapshot(self) -> str:
        """One-line present for the prompt when relevant. Cheap, no LLM."""
        a = [t for t in self._fresh(self._audio, self.audio_s)
             if not _is_recall_question(t)]
        s = self._fresh(self._screen, self.screen_s)
        f = list(self._frames)[-1:]  # only freshest frame desc
        parts = []
        if a:
            parts.append(f"just heard: '{a[-1]}'")
        if s:
            parts.append(f"just on screen: '{s[-1]}'")
        if f:
            parts.append(f"just seen: '{f[0][1]}'")
        if not parts:
            return ""
        return "SENSORY RING (seconds-old present, RAM only): " + " | ".join(parts)

    def answer_what_just(self, kind: str = "all") -> str:
        """Explicit recall from ring — 'what did I just say/do?'

        The recall question itself is always the freshest push (the input
        path records every utterance before the brain runs), so the last
        entry matching a recall pattern is skipped — otherwise she'd just
        quote your question back at you.
        """
        a = self._fresh(self._audio, self.audio_s)
        s = self._fresh(self._screen, self.screen_s)
        f = [d for _, d in list(self._frames)]
        a = [t for t in a if not _is_recall_question(t)]
        if kind == "say":
            return ("You just said: '" + a[-1] + "'") if a else "Nothing in the last 30 seconds."
        if kind == "screen":
            return ("On screen a moment ago: '" + s[-1] + "'") if s else "No screen snapshot in the last minute."
        bits = []
        if a:
            bits.append(f"heard: '{a[-1]}'")
        if s:
            bits.append(f"screen: '{s[-1]}'")
        if f:
            bits.append(f"seen: '{f[-1]}'")
        return "Just now — " + (" | ".join(bits) if bits else "quiet, nothing fresh.")

    def clear(self) -> str:
        self._audio.clear()
        self._screen.clear()
        self._frames.clear()
        return "Sensory ring cleared — last minute forgotten."
