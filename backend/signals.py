"""Signals — streaming demand detection (PASK IntentFlow, honest edition).

The proactive loop was clock-driven: reminders, lunch, wind-down. This is
EVENT-driven: the world speaks, she listens, she decides whether the
moment earns one spoken line. Precision gating keeps her from ever
nagging — a signal without proof stays silent in the log.

Three detectors (all local, all cheap, no model calls):
  error-repeat  — same tool/turn failing 2x in 30 min → offer help once
  file-landed   — STL/STEP/3MF appears in a watched project dir → verdict offer
  long-focus    — one task-board entry active 40+ min → gentle check once

Each returns (key, line, proof). Proof is shown, never hidden: "saw the
same CAD error twice in 20 minutes" is why she spoke. No proof, no speech.

Cooldowns ride the ProactiveEngine's own gate, so the hourly ceiling and
quiet hours apply unchanged. Max one interjection per cycle, as before.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path


ERROR_WINDOW_S = 30 * 60
ERROR_REPEATS = 2
FOCUS_MIN_S = 40 * 60
CAD_EXTS = (".stl", ".step", ".stp", "3mf")
# keep .3mf without dot bug-free: normalize below


class SignalBus:
    def __init__(self, max_events: int = 60):
        self.errors: deque = deque(maxlen=max_events)   # (ts, signature)
        self.files: deque = deque(maxlen=max_events)    # (ts, path)
        self._file_announced: set = set()
        # Feature 8: push subscriptions. The proactivity policy wakes the
        # instant a detection fires instead of waiting for the poll.
        self._subscribers: list = []

    # ── push subscription (Feature 8) ────────────────────────────────────
    def subscribe(self, fn):
        """fn(detection) where detection is (key, line, proof) or None.
        Subscriber exceptions are swallowed — a wake listener must never
        break the tool path that called note_*."""
        if callable(fn) and fn not in self._subscribers:
            self._subscribers.append(fn)
        return fn

    def unsubscribe(self, fn):
        try:
            self._subscribers.remove(fn)
        except ValueError:
            pass

    def _notify(self, detection):
        for fn in list(self._subscribers):
            try:
                fn(detection)
            except Exception:
                pass

    # ── inputs (called from tool paths, never blocking) ─────────────
    def note_error(self, signature: str):
        sig = (signature or "").strip()[:160]
        if sig:
            self.errors.append((time.time(), sig))
            # Push: a repeat detection wakes the policy immediately.
            try:
                self._notify(self.error_repeat())
            except Exception:
                pass

    def note_file(self, path: str):
        p = (path or "").strip()
        if p:
            self.files.append((time.time(), p))
            # Push: a landed-file detection wakes the policy immediately.
            # (Non-consuming: the announcement is only spent when the
            # policy cycle actually evaluates it.)
            try:
                self._notify(self._detect_file_landed())
            except Exception:
                pass

    # ── detectors (called from the proactive cycle) ─────────────────
    def error_repeat(self) -> tuple | None:
        now = time.time()
        recent = [(ts, s) for ts, s in self.errors if now - ts <= ERROR_WINDOW_S]
        counts: dict[str, list] = {}
        for ts, s in recent:
            counts.setdefault(s, []).append(ts)
        for sig, tss in counts.items():
            if len(tss) >= ERROR_REPEATS:
                mins = int((now - tss[0]) / 60) or 1
                return (f"errrepeat_{abs(hash(sig)) % 9999}",
                        "[emotion:thoughtful] Sir — that's the same failure "
                        f"{len(tss)} times in about {mins} minutes. Want me to "
                        "take it apart a different way?",
                        f"saw '{sig[:60]}' {len(tss)}x in {mins} min")
        return None

    def _detect_file_landed(self) -> tuple | None:
        """Non-consuming detection: does a fresh, unannounced CAD file sit in
        the window? Used by the push path so noting a file doesn't consume
        the announcement before the policy decides."""
        exts = tuple(e if e.startswith(".") else "." + e for e in CAD_EXTS)
        for ts, p in list(self.files):
            if time.time() - ts > ERROR_WINDOW_S:
                continue
            if not p.lower().endswith(exts):
                continue
            if p in self._file_announced:
                continue
            name = Path(p).name
            return (f"filelanded_{abs(hash(p)) % 9999}",
                    "[emotion:confident] Sir — a fresh model landed "
                    f"({name}). Shall I run the verdict — interference, "
                    "mass, clearance?",
                    f"saw new file {name}")
        return None

    def file_landed(self, watch_dirs: list | None = None) -> tuple | None:
        det = self._detect_file_landed()
        if det is None:
            return None
        # find the path behind the detection to mark it announced
        for ts, p in list(self.files):
            if f"filelanded_{abs(hash(p)) % 9999}" == det[0]:
                self._file_announced.add(p)
                break
        # cap the announced set so it can't grow forever
        if len(self._file_announced) > 200:
            self._file_announced = set(list(self._file_announced)[-100:])
        return det

    def long_focus(self, task_board: dict) -> tuple | None:
        now = time.time()
        for key, item in (task_board or {}).items():
            if not isinstance(item, dict):
                continue
            if now - item.get("t0", now) >= FOCUS_MIN_S and not item.get("_focus_nudged"):
                item["_focus_nudged"] = True
                label = str(item.get("label", key))[:60]
                return (f"focus_{key}",
                        "[emotion:warm] Sir — forty minutes deep on "
                        f"'{label}'. Still sharp, or want a second pair of eyes?",
                        f"'{label}' active 40+ min")
        return None
