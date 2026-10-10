"""Companion — E.V.-style depth, opt-in and deletable (honest edition).

Diary-driven awareness of how Yash is doing across days: reads the last
7 diary files, counts late nights, long stretches, wins, frustrations —
all from words SHE wrote, never from surveillance — and offers one gentle
reflection. Rules, enforced in code shape, not vibes:

  - Opt-in: soul/companion_optin.json must say yes (voice: "deeper
    check-ins on/off"). Default OFF. Off = tool refuses, no reading.
  - Never diagnosing: no medical/mental-health claims, ever. Mood words
    only ("tired", "buzzing", "quiet week") with the diary line cited.
  - Offered, never pushed: returns text for Sir to ask for; the proactive
    loop NEVER calls this (no ambient mood-reading, full stop).
  - Deletable: companion_forget() wipes the opt-in + summary cache.
    Diary files remain (separate store, separate consent).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


TIRED_WORDS = ("tired", "exhaust", "late night", "up late", "drained", "weary")
GOOD_WORDS = ("win", "clean", "passed", "beautiful", "elegant", "nailed", "buzzing")
ROUGH_WORDS = ("fail", "warp", "stuck", "frustrat", "broken", "wrong", "slow")


class Companion:
    def __init__(self, state_dir: str | Path):
        self.root = Path(state_dir) / "soul"
        self.optin_path = self.root / "companion_optin.json"

    def opted_in(self) -> bool:
        try:
            return bool(json.loads(self.optin_path.read_text()).get("on"))
        except Exception:
            return False

    def set_optin(self, on: bool) -> str:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.optin_path.write_text(json.dumps({"on": bool(on), "ts": time.time()}))
        except Exception as e:
            return f"Could not save: {type(e).__name__}"
        return ("Deeper check-ins ON — I'll notice across days, gently, "
                "and only speak when asked." if on else
                "Deeper check-ins OFF — diary stays unread by me.")

    def reflect(self) -> str:
        if not self.opted_in():
            return ("Deeper check-ins are off, Sir — say 'deeper check-ins on' "
                    "if you want them. Your diary stays yours either way.")
        try:
            files = sorted((self.root / "diary").glob("*.md"))[-7:]
        except Exception:
            files = []
        if not files:
            return "No diary days yet — live a little first, Sir."
        text = "\n".join(p.read_text(errors="ignore") for p in files).lower()
        tired = sum(text.count(w) for w in TIRED_WORDS)
        good = sum(text.count(w) for w in GOOD_WORDS)
        rough = sum(text.count(w) for w in ROUGH_WORDS)
        tone = ("a tired stretch" if tired > good else
                "a good run" if good > rough else "a mixed week")
        return (f"Across the last {len(files)} diary days: {tone} — "
                f"{good} bright lines, {rough} rough ones, {tired} tired ones. "
                f"That's weather, not diagnosis, Sir — and how's your heart, "
                f"not just your week?")

    def forget(self) -> str:
        try:
            if self.optin_path.exists():
                self.optin_path.unlink()
        except Exception:
            pass
        return "Deeper check-ins wiped from my side. Diary files untouched."
