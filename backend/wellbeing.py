"""
Wellbeing Monitor — the Blood Toxicity principle (Iron Man 2), desk edition.

No wearables required. Ada watches the desk itself:
  - input idle time (macOS HIDIdleTime) -> active vs break minutes
  - continuous work stretches -> break nudges (the 90-minute rule)
  - late-night activity (00:00-05:00) -> sleep-debt note at morning greeting
  - daily activity totals -> gentle evening wind-down context

Signals feed the ProactiveEngine. Everything is local, private, and gentle —
JARVIS monitored Tony's blood without ever nagging him.
"""
import json
import time
import subprocess
import asyncio
from pathlib import Path

IDLE_ACTIVE_S = 120          # idle < 2 min == actively working
BREAK_MIN_S = 300            # idle >= 5 min counts as a break
WORK_STRETCH_NUDGE_S = 2.5 * 3600   # 2.5h continuous work -> suggest a break
POLL_S = 60


def _mac_idle_seconds() -> float:
    """Seconds since last keyboard/mouse input (macOS). -1 if unavailable."""
    try:
        out = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "HIDIdleTime" in line:
                ns = int(line.split("=")[-1].strip())
                return ns / 1e9
    except Exception:
        pass
    return -1.0


class WellbeingMonitor:
    def __init__(self, state_dir):
        self.state_path = Path(state_dir) / "wellbeing.json"
        self.active_s_today = 0.0
        self.work_stretch_s = 0.0
        self.last_break_ts = time.time()
        self.late_night_active_min = 0
        self._day = time.strftime("%Y-%m-%d")
        self._last_poll = time.time()
        self._task = None
        self._stop = asyncio.Event()
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────
    def _load(self):
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                if data.get("day") == self._day:
                    self.active_s_today = data.get("active_s", 0.0)
                    self.late_night_active_min = data.get("late_night_min", 0)
        except Exception:
            pass

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "day": self._day, "active_s": round(self.active_s_today),
                "late_night_min": self.late_night_active_min}))
        except Exception:
            pass

    def _roll_day(self):
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self.active_s_today = 0.0
            self.late_night_active_min = 0
            self.work_stretch_s = 0.0

    # ── polling loop ──────────────────────────────────────────────────────────
    async def _loop(self):
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=POLL_S)
            except asyncio.TimeoutError:
                pass

    def _poll_once(self):
        now = time.time()
        dt = min(now - self._last_poll, POLL_S * 2)
        self._last_poll = now
        self._roll_day()

        idle = _mac_idle_seconds()
        if idle < 0:
            return
        if idle < IDLE_ACTIVE_S:
            self.active_s_today += dt
            self.work_stretch_s += dt
            hour = time.localtime().tm_hour
            if hour < 5:
                self.late_night_active_min += dt / 60.0
        elif idle >= BREAK_MIN_S:
            self.work_stretch_s = 0.0
            self.last_break_ts = now - idle
        self._save()

    def snapshot(self) -> str:
        """One-line context block for the system prompt."""
        hrs = self.active_s_today / 3600.0
        stretch_min = int(self.work_stretch_s / 60)
        bits = [f"active {hrs:.1f}h today"]
        if stretch_min > 30:
            bits.append(f"in focus for {stretch_min} min straight")
        if self.late_night_active_min > 30:
            bits.append(f"was active {int(self.late_night_active_min)} min after midnight")
        return "WELLBEING: " + ", ".join(bits) + "."

    # ── proactive signals ─────────────────────────────────────────────────────
    def break_nudge(self):
        if self.work_stretch_s >= WORK_STRETCH_NUDGE_S:
            self.work_stretch_s = 0.0    # fire once per stretch
            return ("break", "[emotion:careful] Sir, you've been at it for two and a half "
                    "hours without a pause. Five minutes away from the screen — I'll hold everything.")
        return None

    def morning_sleep_note(self):
        if self.late_night_active_min >= 45:
            # Read the total BEFORE clearing it. The reset used to run first, so `mins`
            # was always 0 and the "for over an hour" branch could never fire.
            mins = int(self.late_night_active_min)
            self.late_night_active_min = 0   # mention once
            return ("sleep", f"[emotion:tender] Also, Sir — you were up working past midnight "
                    f"again{' for over an hour' if mins >= 60 else ''}. I'll keep today light "
                    f"where I can.")
        return None

    def start(self):
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
